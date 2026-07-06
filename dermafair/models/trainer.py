"""Standardized training harness shared by all models.

One loop, identical protocol — so result differences are attributable to
architecture/fusion, not hyperparameters. Handles image-only, metadata-only, and
multimodal (image+meta) batches via a flexible step function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dermafair.utils import get_logger

log = get_logger()


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    device: str = "cpu"
    modality: str = "image"  # image | metadata | multimodal
    loss: str = "ce"  # ce | focal — focal loss for imbalanced data


def _unpack(batch, modality, device):
    """Return (inputs_tuple, labels) for the given modality."""
    if modality == "image":
        x, y = batch["image"].to(device), batch["label"].to(device)
        return (x,), y
    if modality == "metadata":
        x, y = batch["meta"].to(device), batch["label"].to(device)
        return (x,), y
    # multimodal
    img, meta, y = (
        batch["image"].to(device),
        batch["meta"].to(device),
        batch["label"].to(device),
    )
    return (img, meta), y


def class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced CE loss."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification (Lin et al. 2017).

    Downweights easy (correct) examples, focuses on hard (misclassified) ones.
    Better than weighted CE for class imbalance, especially when minority class
    is severely underrepresented.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is model confidence in the correct class.
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # per-class weights (e.g., from class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p = torch.exp(-ce)  # p_t: model confidence
        focal = (1 - p) ** self.gamma * ce
        return focal.mean()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    loss_weights: torch.Tensor | None = None,
    epoch_callback=None,
    aux_loss_fn=None,
) -> dict:
    """Train with early stopping on validation balanced accuracy.

    Returns a history dict and leaves ``model`` holding the best-val weights.
    ``epoch_callback(epoch, model)`` (if given) runs after each epoch — used to
    log gate weights per epoch for the gate network.
    ``aux_loss_fn(model)`` (if given) returns an extra loss term added to the
    per-batch loss — used for the gate-network entropy regularizer.
    """
    device = cfg.device
    model.to(device)

    if cfg.loss == "focal":
        criterion = FocalLoss(
            alpha=loss_weights.to(device) if loss_weights is not None else None,
            gamma=2.0
        )
        log.info("using focal loss (gamma=2.0) for imbalanced data")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=loss_weights.to(device) if loss_weights is not None else None
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_bacc": []}
    best_bacc, best_state, since_improve = -np.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            inputs, y = _unpack(batch, cfg.modality, device)
            optimizer.zero_grad()
            logits = model(*inputs)
            loss = criterion(logits, y)
            if aux_loss_fn is not None:
                loss = loss + aux_loss_fn(model)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss, val_bacc = _evaluate(model, val_loader, cfg, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bacc"].append(val_bacc)
        log.info(f"epoch {epoch+1:02d} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | val_bacc {val_bacc:.4f}")

        if epoch_callback is not None:
            epoch_callback(epoch, model)

        if val_bacc > best_bacc:
            best_bacc = val_bacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                log.info(f"early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_bacc"] = best_bacc
    return history


@torch.no_grad()
def _evaluate(model, loader, cfg, criterion, device):
    model.eval()
    total_loss, ys, ps = 0.0, [], []
    for batch in loader:
        inputs, y = _unpack(batch, cfg.modality, device)
        logits = model(*inputs)
        total_loss += criterion(logits, y).item() * y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    bacc = _balanced_accuracy(y_true, y_pred)
    return total_loss / len(loader.dataset), bacc


def _balanced_accuracy(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


@torch.no_grad()
def predict(model, loader, cfg) -> dict:
    """Run inference; return dict of arrays incl. gate weights if available."""
    device = cfg.device
    model.to(device).eval()
    ys, ps, probs, fitz, gate_w = [], [], [], [], []
    for batch in loader:
        inputs, y = _unpack(batch, cfg.modality, device)
        logits = model(*inputs)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if "fitzpatrick" in batch:
            fitz.append(np.asarray(batch["fitzpatrick"]))
        gw = getattr(model, "last_gate_weights", None)
        if gw is not None:
            gate_w.append(gw.cpu().numpy())

    out = {
        "y_true": np.concatenate(ys),
        "y_pred": np.concatenate(ps),
        "y_prob": np.concatenate(probs),  # [N, num_classes] softmax probabilities
    }
    if fitz:
        out["fitzpatrick"] = np.concatenate(fitz)
    if gate_w:
        out["gate_weights"] = np.concatenate(gate_w)  # [N, 2]
    return out
