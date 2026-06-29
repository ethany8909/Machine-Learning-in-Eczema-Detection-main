"""Utility helpers: deterministic seeding, config loading, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # allow fairness module to run without torch
    _HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    """Fix all RNGs for reproducibility (NumPy, Python, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str = "dermafair") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_device(requested: str = "cuda") -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if requested == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"
