"""Streamlit UI: pick or upload an avocado photo and get its ripeness from the live API.

Run with `make ui` (or `streamlit run app/streamlit_app.py`), then open http://localhost:8501.

Flow of one prediction:
1. the photo is turned into colour features with avoripe.features (same code as the dataset);
2. the features + storage group + day are sent to POST /predict of the API;
3. the predicted stage and the probability of each stage are displayed.

The UI never loads the model itself: it always goes through the deployed API, so what you
see is exactly what production answers. Set AVORIPE_API_URL to use another API
(e.g. http://localhost:7860 when running `make run`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Make the src/ package importable when Streamlit runs this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from avoripe import config  # noqa: E402
from avoripe.features import image_features  # noqa: E402

DEFAULT_API_URL = "https://avocado-ripeness.onrender.com"
IMAGES_DIR = ROOT_DIR / "data" / "images"  # local copy of the photos (git-ignored)
REQUEST_TIMEOUT_S = 90  # a sleeping free instance can take ~1 minute to wake up

# Readable labels and colours for each ripening stage, in ripening order.
STAGE_LABELS = {
    "underripe": "1 · Underripe",
    "breaking": "2 · Breaking",
    "ripe_first_stage": "3 · Ripe (first stage)",
    "ripe_second_stage": "4 · Ripe (second stage)",
    "overripe": "5 · Overripe",
}
STORAGE_LABELS = {"T10": "T10 · 10 °C", "T20": "T20 · 20 °C", "Tam": "Tam · ambient"}


# ---------------------------------------------------------------------------
# Data and API helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Load the real dataset table (one row per photo, with its official label)."""
    df = pd.read_csv(config.DATA_PATH)
    df["label"] = df[config.TARGET_COLUMN].map(config.CLASS_NAMES)
    return df


def check_health(api_url: str) -> dict | None:
    """Return the /health payload of the API, or None if it cannot be reached."""
    try:
        response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict(api_url: str, payload: dict) -> dict:
    """Send the features to POST /predict and return the JSON answer."""
    response = requests.post(f"{api_url}/predict", json=payload, timeout=REQUEST_TIMEOUT_S)
    # A 422 means the payload was rejected by the API's validation: show its message.
    if response.status_code == 422:
        raise ValueError(response.json().get("detail"))
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def sidebar() -> str:
    """Render the sidebar (API URL + status) and return the API URL to use."""
    st.sidebar.header("⚙️ API")
    api_url = st.sidebar.text_input(
        "API URL", os.getenv("AVORIPE_API_URL", DEFAULT_API_URL)
    ).rstrip("/")

    # Show whether the API is up, and which commit is deployed.
    with st.sidebar, st.spinner("Checking the API..."):
        health = check_health(api_url)
    if health and health.get("model_loaded"):
        st.sidebar.success(f"Online · version `{str(health.get('version', '?'))[:7]}`")
    else:
        st.sidebar.error("API unreachable or model not loaded")

    st.sidebar.markdown(
        "Data: *'Hass' Avocado Ripening Photographic Dataset* "
        "(Xavier, Rodrigues & Silva, 2024, CC BY 4.0)."
    )
    return api_url


def pick_dataset_photo(catalog: pd.DataFrame) -> tuple[bytes, str, int, str] | None:
    """Let the user pick a real photo of the dataset; return (bytes, group, day, true label)."""
    if not IMAGES_DIR.exists():
        st.info(f"No local photos found in `{IMAGES_DIR}`. See data/README.md to download them.")
        return None

    # Filter by true stage so it is easy to try each ripening stage.
    stage = st.selectbox("True stage", list(STAGE_LABELS), format_func=STAGE_LABELS.get)
    names = catalog.loc[catalog["label"] == stage, "file_name"].tolist()
    file_name = st.selectbox(f"Photo ({len(names)} available)", names)

    row = catalog.loc[catalog["file_name"] == file_name].iloc[0]
    path = IMAGES_DIR / f"{file_name}.jpg"
    if not path.exists():
        st.warning(f"Photo `{path.name}` is missing from {IMAGES_DIR}.")
        return None
    return path.read_bytes(), str(row["storage_group"]), int(row["day"]), str(row["label"])


def upload_photo() -> tuple[bytes, str, int, None] | None:
    """Let the user upload a photo and enter its storage conditions."""
    st.caption("Best results with photos like the dataset's: one avocado on a white background.")
    uploaded = st.file_uploader("Avocado photo", type=["jpg", "jpeg", "png"])

    # The model also uses storage conditions: the photo alone does not contain them.
    col1, col2 = st.columns(2)
    group = col1.selectbox("Storage", list(STORAGE_LABELS), format_func=STORAGE_LABELS.get)
    day = col2.number_input("Days in storage", min_value=1, max_value=60, value=5)

    if uploaded is None:
        return None
    return uploaded.getvalue(), group, int(day), None


def show_result(result: dict, true_label: str | None) -> None:
    """Display the predicted stage, the true stage if known, and all probabilities."""
    predicted = result["ripeness"]
    st.metric("Predicted stage", STAGE_LABELS[predicted])

    # For dataset photos we know the answer: say whether the model got it right.
    if true_label is not None:
        if predicted == true_label:
            st.success(f"Correct: the dataset label is {STAGE_LABELS[true_label]}.")
        else:
            st.warning(f"The dataset label is {STAGE_LABELS[true_label]}.")

    # Probabilities in ripening order (1 -> 5), as percentages.
    percents = [100 * result["probabilities"].get(name, 0.0) for name in STAGE_LABELS]
    probabilities = pd.DataFrame(
        {"stage": list(STAGE_LABELS.values()), "probability (%)": percents}
    ).set_index("stage")
    st.bar_chart(probabilities, horizontal=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the page: choose a photo, extract features, call the API, show the result."""
    st.set_page_config(page_title="Avocado Ripeness", page_icon="🥑", layout="wide")
    st.title("🥑 Avocado Ripeness")
    st.caption("Photo → colour features → live API → ripening stage (1 to 5)")

    api_url = sidebar()
    catalog = load_catalog()

    left, right = st.columns([1, 1])
    with left:
        source = st.radio("Photo source", ["Dataset photo", "Upload"], horizontal=True)
        choice = pick_dataset_photo(catalog) if source == "Dataset photo" else upload_photo()

    if choice is None:
        return
    image_bytes, group, day, true_label = choice

    with left:
        st.image(image_bytes, width="stretch")

    with right:
        # Same feature code as the dataset extraction, then the API does the rest.
        payload = {"storage_group": group, "day": day, **image_features(image_bytes)}
        try:
            with st.spinner("Asking the model..."):
                result = predict(api_url, payload)
        except (requests.RequestException, ValueError) as error:
            st.error(f"Prediction failed: {error}")
            return
        show_result(result, true_label)

        with st.expander("Features sent to the API"):
            st.json(payload)


main()
