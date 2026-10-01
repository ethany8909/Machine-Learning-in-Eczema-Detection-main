"""Skin-tone measurement and transformation in CIELAB colour space.

ita()     Individual Typology Angle (Chardon et al., 1991), the standard colorimetric skin-tone
          measure: ITA = atan((L* - 50) / b*) in degrees. Lower is darker; Fitzpatrick V-VI
          photographs typically fall below about 10 degrees.
darken()  re-renders the skin pixels of a photograph at a target ITA, used to create
          darker-skin versions of lighter-skin training images (experiments/improvements).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# skin-tone measurement: Individual Typology Angle (Chardon et al.), lower = darker.
# Fitzpatrick V-VI photos typically sit below ~10 degrees.
# --------------------------------------------------------------------------- #
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_D = 6 / 29


def rgb2lab(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > _D**3, np.cbrt(xyz), xyz / (3 * _D**2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > _D, f**3, 3 * _D**2 * (f - 4 / 29)) * _WHITE
    lin = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def skin_mask(img: np.ndarray, blur: float = 6.0) -> np.ndarray:
    """Soft skin mask from the classic YCrCb skin-colour box (widened to keep red and brown
    lesions), feathered so the darkening has no hard edges."""
    from PIL import ImageFilter

    r, g, b = [img[..., i].astype(np.float64) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr, cb = (r - y) * 0.713 + 128, (b - y) * 0.564 + 128
    m = ((cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 130)).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(blur)), dtype=np.float64) / 255.0


def ita(img: np.ndarray, skin_only: bool = True) -> float:
    lab = rgb2lab(img).reshape(-1, 3)
    if skin_only:
        m = skin_mask(img).reshape(-1) > 0.5
        if m.sum() > 500:
            lab = lab[m]
    return float(np.median(np.degrees(np.arctan2(lab[:, 0] - 50, lab[:, 2]))))


def darken(img: np.ndarray, target_ita: float) -> np.ndarray:
    """Scale the lightness of the skin pixels by k (and their chroma by 0.7 + 0.3k, since
    darker skin photographs slightly less saturated) so the median skin ITA reaches
    target_ita. Lesion-to-skin contrast shrinks with the skin, which is also why redness
    is harder to see on darker skin."""
    lab = rgb2lab(img)
    w = skin_mask(img)
    m = w > 0.5
    if m.sum() < 500:
        m = np.ones_like(m)
    L0, b0 = np.median(lab[..., 0][m]), np.median(lab[..., 2][m])
    ita_at = lambda k: np.degrees(np.arctan2(k * L0 - 50, b0 * (0.7 + 0.3 * k)))  # noqa: E731
    if ita_at(1.0) <= target_ita:
        return img.copy()
    lo, hi = 0.3, 1.0  # ITA increases with k, so bisect
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ita_at(mid) < target_ita else (lo, mid)
    k = (lo + hi) / 2
    out = lab.copy()
    out[..., 0] = lab[..., 0] * (1 - w + w * k)
    out[..., 1:] = lab[..., 1:] * (1 - w + w * (0.7 + 0.3 * k))[..., None]
    return lab2rgb(out)
