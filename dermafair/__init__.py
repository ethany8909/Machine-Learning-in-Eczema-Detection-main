"""DermaFair: Fitzpatrick-stratified fairness evaluation for multimodal dermatology AI."""

__version__ = "0.1.0"

from dermafair.fairness import FairnessEvaluator, FairnessReport

__all__ = ["FairnessEvaluator", "FairnessReport", "__version__"]
