"""One-off feature extraction: real avocado photos -> small tabular CSV committed in data/raw/.

Not part of CI. Run once after downloading the Mendeley archive (see data/README.md):

    python scripts/extract_features.py path/to/3xd9n945v8-1.zip

Every row of the output is computed from one real photograph; nothing is generated.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OUTPUT_CSV = Path("data/raw/avocado_features.csv")
METADATA_XLSX = "Avocado Ripening Dataset.xlsx"
THUMB_SIZE = (200, 200)
# Background is a white sheet: bright and almost unsaturated.
BG_MIN_VALUE = 0.60
BG_MAX_SATURATION = 0.18
# Lab lightness under which a fruit pixel counts as "dark skin".
DARK_L_THRESHOLD = 30.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) sRGB array in [0, 1] to CIE Lab (D65)."""
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lightness = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([lightness, a, b], axis=1)


def fruit_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask of fruit pixels (everything that is not white background)."""
    value = rgb.max(axis=-1)
    saturation = np.where(value > 0, (value - rgb.min(axis=-1)) / np.maximum(value, 1e-6), 0)
    background = (value >= BG_MIN_VALUE) & (saturation <= BG_MAX_SATURATION)
    return ~background


def image_features(image_bytes: bytes) -> dict[str, float]:
    """Compute colour and size statistics of the fruit in one photograph."""
    image = Image.open(io.BytesIO(image_bytes))
    image.draft("RGB", THUMB_SIZE)  # fast JPEG downscale while decoding
    image = image.convert("RGB").resize(THUMB_SIZE)
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    mask = fruit_mask(rgb)
    lab = srgb_to_lab(rgb[mask])
    return {
        "l_mean": float(lab[:, 0].mean()),
        "l_std": float(lab[:, 0].std()),
        "a_mean": float(lab[:, 1].mean()),
        "a_std": float(lab[:, 1].std()),
        "b_mean": float(lab[:, 2].mean()),
        "b_std": float(lab[:, 2].std()),
        "dark_fraction": float((lab[:, 0] < DARK_L_THRESHOLD).mean()),
        "fruit_fraction": float(mask.mean()),
    }


def load_metadata(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Read the official metadata spreadsheet shipped inside the archive."""
    name = next(n for n in archive.namelist() if n.endswith(METADATA_XLSX))
    meta = pd.read_excel(io.BytesIO(archive.read(name)))
    return meta.rename(
        columns={
            "File Name": "file_name",
            "Storage Group": "storage_group",
            "Sample": "sample",
            "Day of Experiment": "day",
            "Ripening Index Classification": "ripening_index",
        }
    )[["file_name", "sample", "storage_group", "day", "ripening_index"]]


def main(zip_path: str) -> None:
    """Extract features for every photo listed in the metadata and write the CSV."""
    with zipfile.ZipFile(zip_path) as archive:
        meta = load_metadata(archive)
        images = {Path(n).stem: n for n in archive.namelist() if n.lower().endswith(".jpg")}
        rows = []
        for i, record in enumerate(meta.itertuples(index=False)):
            if record.file_name not in images:
                continue
            features = image_features(archive.read(images[record.file_name]))
            rows.append({**record._asdict(), **features})
            if i % 1000 == 0:
                print(f"{i}/{len(meta)} images processed")
    table = pd.DataFrame(rows).round(4)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(table)} rows ({len(meta) - len(table)} metadata rows without image)")


if __name__ == "__main__":
    main(sys.argv[1])
