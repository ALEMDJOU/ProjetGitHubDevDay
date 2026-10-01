"""RipeVision: Streamlit UI that predicts the ripeness of a Hass avocado from a photo.

Run with `make ui` (or `streamlit run app/streamlit_app.py` from the repository root, so
that .streamlit/config.toml is picked up), then open http://localhost:8501.

How one prediction works:
1. the photo is turned into colour features with avoripe.features (same code as the dataset);
2. the features + storage group + day are sent to POST /predict of the live API;
3. the API answers with the 5 real ripening stages of the dataset and their probabilities;
4. for readability, the UI groups the 5 stages into 3 families (Unripe / Ripe / Overripe)
   and shows the family confidence = sum of the probabilities of its stages.

No result is ever simulated: if the API cannot be reached, the UI says so.
Design rules: no emoji, no image file, Material Symbols icons only, palette and fonts below.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import UnidentifiedImageError

# Make the src/ package importable when Streamlit runs this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from avoripe import config  # noqa: E402
from avoripe.features import image_features  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_API_URL = "https://avocado-ripeness.onrender.com"
# Backend URL: not shown in the UI. Override with AVORIPE_API_URL (e.g. http://localhost:7860).
API_URL = os.getenv("AVORIPE_API_URL", DEFAULT_API_URL).rstrip("/")
IMAGES_DIR = ROOT_DIR / "data" / "images"  # local copy of the dataset photos (git-ignored)
REQUEST_TIMEOUT_S = 90  # a sleeping free Render instance can take ~1 minute to wake up

# The 5 real stages of the dataset, grouped into 3 display families.
STAGE_FAMILY = {
    "underripe": "unripe",
    "breaking": "unripe",
    "ripe_first_stage": "ripe",
    "ripe_second_stage": "ripe",
    "overripe": "overripe",
}
STAGE_LABELS = {
    "underripe": "1 - Pas mûr",
    "breaking": "2 - En cours de maturation",
    "ripe_first_stage": "3 - Mûr (début)",
    "ripe_second_stage": "4 - Mûr (fin)",
    "overripe": "5 - Trop mûr",
}

# Everything the UI shows for each family: label, colour, icon, verdict, recommendation.
FAMILIES = {
    "unripe": {
        "label": "Unripe",
        "color": "#74C69D",
        "icon": ":material/eco:",
        "message": "Encore un peu de patience.",
        "advice": "Attendre",
    },
    "ripe": {
        "label": "Ripe",
        "color": "#F4A261",
        "icon": ":material/check_circle:",
        "message": "Parfait, à consommer aujourd'hui.",
        "advice": "Consommer",
    },
    "overripe": {
        "label": "Overripe",
        "color": "#6A4C3B",
        "icon": ":material/warning:",
        "message": "Trop mûr, idéal pour un guacamole.",
        "advice": "Guacamole",
    },
}
STORAGE_LABELS = {"T10": "T10 - 10 °C", "T20": "T20 - 20 °C", "Tam": "Tam - ambiant"}

EMPTY_STATE = "Aucune analyse pour l'instant. Ajoutez une photo pour commencer."
UNREADABLE_IMAGE = "Image illisible. Essayez un autre format (JPG, PNG)."


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject fonts and the RipeVision visual style (palette, cards, sidebar, header)."""
    # Note: font-family is never set on `span`, so Material Symbols icons keep their font.
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&family=Poppins:wght@700&display=swap');

/* Typography: Inter for text, Poppins for titles, JetBrains Mono for numbers */
html, body, .stApp, p, label, input, textarea, button, li { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important;
                 color: #1B4332; }

/* Main container: centred, max 1200px, 32px padding */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1200px; margin: 0 auto; padding: 2rem;
}

/* Header banner */
.rv-header {
  background: linear-gradient(90deg, #1B4332 0%, #52B788 100%);
  color: #FFFFFF; border-radius: 12px; padding: 1.5rem; margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(27, 67, 50, 0.15);
}
.rv-header .rv-title { font-family: 'Poppins', sans-serif; font-weight: 700;
                       font-size: 2.25rem; line-height: 1.2; margin: 0; }
.rv-header .rv-subtitle { font-family: 'Inter', sans-serif; font-size: 1.05rem;
                          opacity: 0.92; margin-top: 8px; }

/* Buttons */
.stButton > button, [data-testid^="stBaseButton"] {
  background-color: #52B788; color: #FFFFFF; border: none; border-radius: 10px;
  padding: 8px 24px; font-weight: 500;
}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {
  background-color: #1B4332; color: #FFFFFF;
  box-shadow: 0 2px 8px rgba(27, 67, 50, 0.15);
}

/* KPI cards (st.metric) */
[data-testid="stMetric"] {
  background: #FFFFFF; border-left: 4px solid #95D5B2; border-radius: 12px;
  padding: 1rem; box-shadow: 0 2px 8px rgba(27, 67, 50, 0.08);
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #1B4332; }
.st-key-kpi-unripe  [data-testid="stMetric"] { border-left-color: #74C69D; }
.st-key-kpi-ripe    [data-testid="stMetric"] { border-left-color: #F4A261; }
.st-key-kpi-overripe [data-testid="stMetric"] { border-left-color: #6A4C3B; }

/* Verdict banner, coloured by family */
[class*="st-key-verdict-"] { border-radius: 12px; padding: 16px 24px; }
[class*="st-key-verdict-"] p { font-size: 1.1rem; font-weight: 500; margin: 0; }
.st-key-verdict-unripe   { background: #74C69D; color: #1B4332; }
.st-key-verdict-ripe     { background: #F4A261; color: #212529; }
.st-key-verdict-overripe { background: #6A4C3B; color: #FFFFFF; }
.st-key-verdict-overripe p { color: #FFFFFF; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #95D5B2; border-radius: 12px; background: #FFFFFF;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1B4332; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] li, [data-testid="stSidebar"] summary,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #F1FAEE;
}
[data-testid="stSidebar"] hr { border-color: #52B788; }
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border: 1px solid #52B788; border-radius: 12px; background: rgba(255, 255, 255, 0.04);
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { color: #95D5B2; }

/* Tabs */
[data-baseweb="tab-list"] { gap: 8px; }
[data-baseweb="tab"] { font-weight: 500; }
[data-baseweb="tab-highlight"] { background-color: #52B788; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the gradient header with the app name and subtitle."""
    st.markdown(
        '<div class="rv-header"><div class="rv-title">RipeVision</div>'
        '<div class="rv-subtitle">Prédiction de maturité d\'avocat</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data, API and result helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Load the real dataset table (one row per photo, with its official stage)."""
    df = pd.read_csv(config.DATA_PATH)
    df["label"] = df[config.TARGET_COLUMN].map(config.CLASS_NAMES)
    return df


@st.cache_data
def load_model_metrics() -> dict | None:
    """Read the real metrics of the last local evaluation, if `make evaluate` was run."""
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def check_health(api_url: str) -> dict | None:
    """Return the /health payload of the API (cached 60 s), or None if unreachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_predict(api_url: str, payload: dict) -> dict:
    """Send the features to POST /predict and return the JSON answer."""
    response = requests.post(f"{api_url}/predict", json=payload, timeout=REQUEST_TIMEOUT_S)
    if response.status_code == 422:  # rejected by the API's input validation
        raise ValueError(response.json().get("detail"))
    response.raise_for_status()
    return response.json()


def summarize(prediction: dict) -> dict:
    """Group the 5 stage probabilities into 3 families and pick the winning family."""
    family_scores = dict.fromkeys(FAMILIES, 0.0)
    for stage, probability in prediction["probabilities"].items():
        family_scores[STAGE_FAMILY[stage]] += probability
    family = max(family_scores, key=family_scores.get)
    return {
        "stage": prediction["ripeness"],
        "family": family,
        "confidence": family_scores[family],
        "family_scores": family_scores,
        "stage_scores": prediction["probabilities"],
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(metrics: dict | None) -> float:
    """Render settings, model info and service status; return the confidence threshold."""
    with st.sidebar:
        st.markdown("### RipeVision")
        with st.expander("Paramètres", icon=":material/tune:", expanded=True):
            threshold = st.slider("Seuil de confiance", 0.5, 1.0, 0.75, 0.05)
        with st.expander("Modèle", icon=":material/info:"):
            st.markdown("**Architecture** : Random Forest (100 arbres), scikit-learn")
            st.markdown("**Données** : 14 710 photos réelles de 478 avocats Hass")
            if metrics:
                st.markdown(f"**Accuracy** : {metrics['accuracy']:.1%}")
        st.divider()
        render_api_status(API_URL)
    return threshold


def render_api_status(api_url: str) -> None:
    """Show whether the prediction service answers and has its model loaded."""
    health = check_health(api_url)
    if health and health.get("model_loaded"):
        st.success("Service de prédiction en ligne", icon=":material/task_alt:")
    else:
        st.error("Service de prédiction indisponible", icon=":material/error:")


# ---------------------------------------------------------------------------
# Analyse tab
# ---------------------------------------------------------------------------
def choose_photo(catalog: pd.DataFrame) -> dict | None:
    """Let the user upload a photo or pick a real one; return photo + storage conditions."""
    source = st.radio(
        "Source",
        ["Importer une photo", "Photo du dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Photo du dataset":
        return pick_dataset_photo(catalog)
    uploaded = st.file_uploader("Photo de l'avocat", type=["jpg", "jpeg", "png"])
    # The model also uses the storage conditions, which a photo alone does not contain.
    left, right = st.columns(2)
    group = left.selectbox("Stockage", list(STORAGE_LABELS), format_func=STORAGE_LABELS.get)
    day = right.number_input("Jours de stockage", min_value=1, max_value=60, value=5)
    if uploaded is None:
        return None
    return {
        "bytes": uploaded.getvalue(),
        "name": uploaded.name,
        "group": group,
        "day": int(day),
        "true_stage": None,
    }


def pick_dataset_photo(catalog: pd.DataFrame) -> dict | None:
    """Pick a real photo of the dataset; its storage conditions and stage are known."""
    if not IMAGES_DIR.exists():
        st.info(
            "Photos du dataset absentes de data/images (voir data/README.md).",
            icon=":material/info:",
        )
        return None
    stage = st.selectbox("Stade réel", list(STAGE_LABELS), format_func=STAGE_LABELS.get)
    names = catalog.loc[catalog["label"] == stage, "file_name"].tolist()
    name = st.selectbox(f"Photo ({len(names)} disponibles)", names)
    row = catalog.loc[catalog["file_name"] == name].iloc[0]
    path = IMAGES_DIR / f"{name}.jpg"
    if not path.exists():
        st.info(f"Photo {path.name} absente de data/images.", icon=":material/info:")
        return None
    return {
        "bytes": path.read_bytes(),
        "name": name,
        "group": str(row["storage_group"]),
        "day": int(row["day"]),
        "true_stage": str(row["label"]),
    }


def run_analysis(photo: dict, api_url: str) -> dict | None:
    """Extract features, call the API inside a status box; return the summary or None."""
    with st.status("Analyse en cours…", expanded=True) as status:
        st.write("Extraction des couleurs de la peau")
        try:
            features = image_features(photo["bytes"])
        except (UnidentifiedImageError, OSError, ValueError):
            status.update(label="Analyse interrompue", state="error")
            st.error(UNREADABLE_IMAGE, icon=":material/error:")
            return None
        st.write("Interrogation du modèle en ligne")
        payload = {"storage_group": photo["group"], "day": photo["day"], **features}
        try:
            result = summarize(call_predict(api_url, payload))
        except (requests.RequestException, ValueError) as error:
            status.update(label="Analyse interrompue", state="error")
            st.error(f"L'API n'a pas pu répondre : {error}", icon=":material/error:")
            return None
        status.update(label="Analyse terminée", state="complete", expanded=False)
    return {**result, "payload": payload, "photo": photo["name"], "true_stage": photo["true_stage"]}


def render_result(result: dict, threshold: float) -> None:
    """Show the 3 KPI cards, the confidence bar and the coloured verdict banner."""
    family = FAMILIES[result["family"]]
    with st.container(key=f"kpi-{result['family']}"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Classe prédite", family["label"])
        col2.metric("Confiance", f"{result['confidence']:.0%}")
        col3.metric("Recommandation", family["advice"])
    st.progress(
        min(result["confidence"], 1.0), text=f"Stade exact : {STAGE_LABELS[result['stage']]}"
    )
    with st.container(key=f"verdict-{result['family']}"):
        st.markdown(f"{family['icon']} {family['message']}")
    # Below the threshold, the answer is shown but flagged as uncertain.
    if result["confidence"] < threshold:
        st.warning(
            f"Confiance inférieure au seuil de {threshold:.0%} : résultat à confirmer.",
            icon=":material/warning:",
        )
    if result["true_stage"]:
        render_ground_truth(result)


def render_ground_truth(result: dict) -> None:
    """For dataset photos, compare the prediction with the official label."""
    truth = result["true_stage"]
    if truth == result["stage"]:
        st.success(
            f"Correct : le dataset indique {STAGE_LABELS[truth]}.", icon=":material/verified:"
        )
    else:
        st.info(f"Le dataset indique {STAGE_LABELS[truth]}.", icon=":material/info:")


def render_analysis_tab(catalog: pd.DataFrame, api_url: str, threshold: float) -> None:
    """Analyse tab: choose a photo, preview it, predict and show the result."""
    left, right = st.columns([1, 1], gap="large")
    with left:
        photo = choose_photo(catalog)
        if photo:
            st.image(photo["bytes"], width="stretch")
    with right:
        if photo and st.button("Prédire", icon=":material/neurology:", type="primary"):
            result = run_analysis(photo, api_url)
            if result:
                record_history(result)
                st.session_state["last_result"] = result
                st.toast("Analyse terminée", icon=":material/task_alt:")
        if st.session_state.get("last_result"):
            render_result(st.session_state["last_result"], threshold)
        else:
            st.info(EMPTY_STATE, icon=":material/info:")


# ---------------------------------------------------------------------------
# Results and Model tabs
# ---------------------------------------------------------------------------
def record_history(result: dict) -> None:
    """Append one prediction to the session history."""
    st.session_state.setdefault("history", []).append(
        {
            "Heure": datetime.now().strftime("%H:%M:%S"),
            "Photo": result["photo"],
            "Classe": FAMILIES[result["family"]]["label"],
            "Stade exact": STAGE_LABELS[result["stage"]],
            "Confiance": round(result["confidence"], 3),
            "Stade réel": STAGE_LABELS.get(result["true_stage"], "inconnu"),
        }
    )


def render_results_tab() -> None:
    """Results tab: table of every prediction made during this session."""
    history = st.session_state.get("history", [])
    if not history:
        st.info(EMPTY_STATE, icon=":material/info:")
        return
    st.markdown(f"#### Historique de la session ({len(history)} analyses)")
    st.dataframe(
        pd.DataFrame(history[::-1]),
        hide_index=True,
        width="stretch",
        column_config={"Confiance": st.column_config.ProgressColumn(min_value=0, max_value=1)},
    )


def render_model_tab(metrics: dict | None) -> None:
    """Model tab: how the model works and its real evaluation metrics."""
    st.markdown("#### Comment fonctionne le modèle")
    st.markdown(
        "Chaque photo est résumée en statistiques de couleur de la peau (espace Lab), "
        "auxquelles s'ajoutent le mode de stockage et le nombre de jours. Une forêt aléatoire "
        "(scikit-learn) prédit l'un des **5 stades** officiels du dataset ; l'interface les "
        "regroupe en 3 classes : Unripe (stades 1-2), Ripe (3-4), Overripe (5)."
    )
    if not metrics:
        st.info("Métriques indisponibles : lancez `make train evaluate`.", icon=":material/info:")
        return
    st.markdown(f"#### Évaluation sur {metrics['n_test']} photos d'avocats jamais vus")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Précision (macro)", f"{metrics['precision_macro']:.1%}")
    col3.metric("Rappel (macro)", f"{metrics['recall_macro']:.1%}")
    st.caption(
        "Données : 'Hass' Avocado Ripening Photographic Dataset, Xavier, Rodrigues & Silva "
        "(2024), CC BY 4.0."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the page: style, header, sidebar and the three tabs."""
    st.set_page_config(
        page_title="RipeVision", page_icon=":material/energy_savings_leaf:", layout="wide"
    )
    inject_css()
    render_header()
    metrics = load_model_metrics()
    threshold = render_sidebar(metrics)
    analyse, results, model = st.tabs(
        [
            ":material/photo_camera: Analyse",
            ":material/analytics: Résultats",
            ":material/info: Modèle",
        ]
    )
    with analyse:
        render_analysis_tab(load_catalog(), API_URL, threshold)
    with results:
        render_results_tab()
    with model:
        render_model_tab(metrics)


main()
