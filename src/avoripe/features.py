"""Turn one avocado photo into the colour features the model expects.

Shared by:
- scripts/extract_features.py: one-off extraction of the whole dataset into the CSV;
- app/streamlit_app.py:        live extraction from a photo picked or uploaded in the UI.

Using the same code in both places guarantees that a photo sent from the UI is described
exactly like the photos the model was trained on.

Pillow is imported inside image_features() only, so the API and CI (which never decode
images) do not need it installed.
"""

from __future__ import annotations

import io

import numpy as np

# Photos are 800x800; 200x200 keeps the colour statistics and is 16x less work.
THUMB_SIZE = (200, 200)

# Background is a white sheet: bright and almost unsaturated (HSV value / saturation,
# both in [0, 1]). Thresholds checked visually on green (unripe) and black (overripe) fruit.
BG_MIN_VALUE = 0.60
BG_MAX_SATURATION = 0.18

# Lab lightness under which a fruit pixel counts as "dark skin" (Hass skin turns black
# as it ripens, so the share of dark pixels grows with the ripening stage).
DARK_L_THRESHOLD = 30.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) sRGB array in [0, 1] to CIE Lab (D65)."""
    # Standard sRGB -> linear RGB -> XYZ -> Lab conversion, written with numpy.
    # Lab is used because its axes are meaningful for ripeness:
    #   L* = lightness, a* = green (-) to red (+), b* = blue (-) to yellow (+).

    # 1. Undo the sRGB gamma curve.
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # 2. Linear RGB -> XYZ, normalised by the D65 white point.
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883])

    # 3. XYZ -> Lab (cube root with a linear segment near black).
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lightness = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([lightness, a, b], axis=1)


def fruit_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask of fruit pixels (everything that is not white background)."""
    # HSV value = brightest channel; saturation = how far the pixel is from grey.
    value = rgb.max(axis=-1)
    saturation = np.where(value > 0, (value - rgb.min(axis=-1)) / np.maximum(value, 1e-6), 0)

    # Background = bright AND grey/white. Everything else (green, brown, black skin) is fruit.
    background = (value >= BG_MIN_VALUE) & (saturation <= BG_MAX_SATURATION)
    return ~background


def image_features(image_bytes: bytes) -> dict[str, float]:
    """Compute colour and size statistics of the fruit in one photograph."""
    from PIL import Image  # local import: only needed where photos are decoded

    image = Image.open(io.BytesIO(image_bytes))
    image.draft("RGB", THUMB_SIZE)  # fast JPEG downscale while decoding
    image = image.convert("RGB").resize(THUMB_SIZE)

    # Pixel values scaled to [0, 1]; keep only the fruit pixels, then convert to Lab.
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    mask = fruit_mask(rgb)
    lab = srgb_to_lab(rgb[mask])

    # Mean = overall skin colour, std = how uneven it is (spots, patches).
    # Rounded to 4 decimals, exactly like the committed CSV.
    features = {
        "l_mean": lab[:, 0].mean(),
        "l_std": lab[:, 0].std(),
        "a_mean": lab[:, 1].mean(),
        "a_std": lab[:, 1].std(),
        "b_mean": lab[:, 2].mean(),
        "b_std": lab[:, 2].std(),
        "dark_fraction": (lab[:, 0] < DARK_L_THRESHOLD).mean(),
        # Share of the image covered by fruit: a rough size proxy (camera distance is fixed).
        "fruit_fraction": mask.mean(),
    }
    # np.round (not Python's round): same half-way rule as the pandas rounding that
    # produced the CSV, so values like 0.24455 round identically in both places.
    return {name: float(np.round(value, 4)) for name, value in features.items()}
