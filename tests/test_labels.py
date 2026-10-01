"""The keyword rule that aligns internal labels with the external SCIN cohort."""

import numpy as np
import pytest

from dermafair.data.labels import harmonized_label, tone_group


@pytest.mark.parametrize(
    ("diagnosis", "expected"),
    [
        ("Plaque psoriasis", 1),
        ("Scalp Psoriasis", 1),
        ("pustular psoriasis", 1),
        ("Atopic dermatitis", 0),
        ("Nummular eczema", 0),
        ("Allergic contact dermatitis", 0),
        ("Seborrheic dermatitis", 0),
        ("Lichen planus", None),
        ("Lichen simplex chronicus", None),
        ("Pityriasis rosea", None),
        ("Pityriasis alba", None),
        ("", None),
    ],
)
def test_harmonized_label(diagnosis, expected):
    assert harmonized_label(diagnosis) == expected


def test_psoriasis_keyword_takes_precedence():
    assert harmonized_label("eczema with psoriasis overlap") == 1


def test_non_string_input_is_excluded():
    assert harmonized_label(None) is None
    assert harmonized_label(float("nan")) is None


def test_tone_group():
    groups = tone_group([1, 2, 3, 4, 5, 6, 0, -1])
    np.testing.assert_array_equal(groups, [0, 0, 0, 0, 1, 1, -1, -1])
