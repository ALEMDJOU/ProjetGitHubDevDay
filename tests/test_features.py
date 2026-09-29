"""Tests for the photo -> features code shared by the extraction script and the UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avoripe import config
from avoripe.features import fruit_mask, image_features, srgb_to_lab

FIXTURES = Path(__file__).parent / "fixtures"
# One real photo of the dataset (CC BY 4.0), the same one as known_sample.json.
PHOTO = FIXTURES / "T20_d05_001_a_3.jpg"


def test_features_match_committed_dataset() -> None:
    """The real photo gives exactly the values stored in the CSV for that photo."""
    # Exact equality on purpose: any drift (even a rounding rule) would make the UI
    # describe photos differently from the training data.
    features = image_features(PHOTO.read_bytes())
    row = pd.read_csv(config.DATA_PATH).set_index("file_name").loc[PHOTO.stem]
    assert features == {name: float(row[name]) for name in features}


def test_features_match_known_sample() -> None:
    """The Monitor workflow's known sample is the features of this same photo."""
    known = json.loads((FIXTURES / "known_sample.json").read_text(encoding="utf-8"))
    features = image_features(PHOTO.read_bytes())
    assert features == {name: known[name] for name in features}


def test_srgb_to_lab_reference_colours() -> None:
    """Pure white is L*=100 and pure black is L*=0, both neutral (a*, b* ~ 0)."""
    lab = srgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.1)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=0.1)


def test_fruit_mask_ignores_white_background() -> None:
    """A white pixel is background; green and black pixels are fruit."""
    # Test fixture: three reference colours (white sheet, green skin, black skin).
    pixels = np.array([[[1.0, 1.0, 1.0], [0.2, 0.5, 0.1], [0.05, 0.05, 0.05]]])
    assert fruit_mask(pixels).tolist() == [[False, True, True]]
