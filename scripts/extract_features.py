"""One-off feature extraction: real avocado photos -> small tabular CSV committed in data/raw/.

Not part of CI. Run once after downloading the Mendeley archive (see data/README.md):

    python scripts/extract_features.py path/to/3xd9n945v8-1.zip

Every row of the output is computed from one real photograph; nothing is generated.

Why this step exists: the original dataset is ~420 MB of JPEGs. Decoding it on every CI
run would blow the "train + evaluate in under 2 minutes" budget. Instead we summarise each
photo once into a handful of colour statistics (~1.3 MB CSV) and train on that table.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

# The script is run directly (not installed): make the src/ package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same feature code as the Streamlit UI, so both describe a photo identically.
from avoripe.features import image_features  # noqa: E402

OUTPUT_CSV = Path("data/raw/avocado_features.csv")
METADATA_XLSX = "Avocado Ripening Dataset.xlsx"  # official labels, shipped in the archive


def load_metadata(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Read the official metadata spreadsheet shipped inside the archive."""
    name = next(n for n in archive.namelist() if n.endswith(METADATA_XLSX))
    meta = pd.read_excel(io.BytesIO(archive.read(name)))

    # Rename the spreadsheet headers to snake_case column names used everywhere else,
    # and keep only what the pipeline needs (the timestamp and photo side are dropped).
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

        # Map "T20_d05_001_a_3" -> path of that JPEG inside the zip.
        images = {Path(n).stem: n for n in archive.namelist() if n.lower().endswith(".jpg")}

        rows = []
        for i, record in enumerate(meta.itertuples(index=False)):
            # 12 spreadsheet rows have no matching photo in the archive: skip them
            # (listed in data/README.md) rather than inventing values.
            if record.file_name not in images:
                continue
            features = image_features(archive.read(images[record.file_name]))
            # One output row = official metadata + label + features of that photo.
            rows.append({**record._asdict(), **features})
            if i % 1000 == 0:
                print(f"{i}/{len(meta)} images processed")

    # Rounding to 4 decimals keeps the CSV small and byte-identical across machines,
    # so its SHA-256 in data/README.md can be checked.
    table = pd.DataFrame(rows).round(4)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(table)} rows ({len(meta) - len(table)} metadata rows without image)")


if __name__ == "__main__":
    main(sys.argv[1])
