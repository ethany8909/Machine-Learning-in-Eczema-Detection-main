"""Model architectures: image backbones, metadata branch, fusion strategies."""
from dermafair.models.fusion_strategies import GateNetwork, LateFusion, build_fusion
from dermafair.models.image_models import build_image_model
from dermafair.models.metadata_model import build_metadata_model

__all__ = [
    "build_image_model",
    "build_metadata_model",
    "build_fusion",
    "GateNetwork",
    "LateFusion",
]
