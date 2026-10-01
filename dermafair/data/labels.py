"""Label and skin-tone definitions shared by every analysis."""
from __future__ import annotations

import numpy as np


def harmonized_label(disease: str) -> int | None:
    """Relabel a fine-grained diagnosis with the keyword rule used for the external SCIN
    cohort: anything containing "psoriasis" -> 1, else "eczema" or "dermatitis" -> 0, else
    excluded (None).

    Applied to DermaCon-IN this removes lichen planus variants, pityriasis alba and rosea,
    lichen simplex chronicus and post-inflammatory hyperpigmentation (all in the original
    "psoriasis and lichenoid" group) and topical-steroid-damaged face and purpura (in the
    "eczema and dermatitis" group), so the model is trained on the question it is tested on.
    """
    d = str(disease).lower()
    if "psoriasis" in d:
        return 1
    if "eczema" in d or "dermatitis" in d:
        return 0
    return None


def tone_group(fitzpatrick) -> np.ndarray:
    """Binary skin-tone contrast used throughout (Groh et al.): 0 = Fitzpatrick I-IV,
    1 = Fitzpatrick V-VI, -1 = unknown."""
    f = np.asarray(fitzpatrick)
    return np.where(f >= 5, 1, np.where(f >= 1, 0, -1))
