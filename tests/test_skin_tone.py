"""Colour conversion and skin-tone transformation used for the synthetic-image experiment."""

import numpy as np
import pytest

from dermafair.data.skin_tone import darken, ita, lab2rgb, rgb2lab, skin_mask

LIGHT_SKIN = (224, 172, 140)
LESION = (200, 110, 100)


def _patch(size=96):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = LIGHT_SKIN
    img[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = LESION
    return img


def test_lab_round_trip_is_lossless():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    np.testing.assert_array_equal(lab2rgb(rgb2lab(img)), img)


def test_reference_white():
    lab = rgb2lab(np.full((1, 1, 3), 255, np.uint8))[0, 0]
    np.testing.assert_allclose(lab, [100.0, 0.0, 0.0], atol=0.05)


@pytest.mark.parametrize("target", [10.0, -10.0, -30.0])
def test_darken_reaches_target_ita(target):
    out = darken(_patch(), target)
    assert abs(ita(out) - target) < 1.0
    assert ita(out) < ita(_patch())


def test_darken_keeps_lesion_contrast_ordering():
    out = darken(_patch(), -30.0).astype(int)
    skin, lesion = out[5, 5], out[48, 48]
    assert lesion[0] - lesion[1] > skin[0] - skin[1]  # the lesion stays redder than the skin around it


def test_darken_is_a_no_op_for_already_darker_targets():
    img = _patch()
    np.testing.assert_array_equal(darken(img, 80.0), img)


def test_skin_mask_excludes_non_skin_background():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = LIGHT_SKIN
    img[:, 32:] = (40, 90, 200)  # blue background
    mask = skin_mask(img, blur=0.1)
    assert mask[:, :28].mean() > 0.95
    assert mask[:, 36:].mean() < 0.05
