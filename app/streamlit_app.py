from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import DEFAULT_DATA_DIR, SENSOR_INFO, load_fd001
from src.evaluate import health_status

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="AeroSense — Turbofan RUL",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")
    return model, metadata


@st.cache_data
def load_data():
    train, test = load_fd001(DEFAULT_DATA_DIR)
    return train, test


def make_gauge(rul: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(rul),
            number={"suffix": " cycles", "font": {"size": 34}},
            title={"text": "Predicted Remaining Useful Life"},
            gauge={
                "axis": {"range": [0, max(125, float(rul) * 1.25)]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 20], "color": "#ffd6d6"},
                    {"range": [20, 60], "color": "#fff0b3"},
                    {"range": [60, 125], "color": "#d8f3dc"},
                ],
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=15, r=15, t=60, b=10),
    )
    return fig


def make_feature_importance_chart(importance: pd.DataFrame):
    chart_df = importance.head(10).sort_values("importance")

    fig = go.Figure(
        go.Bar(
            x=chart_df["importance"],
            y=chart_df["feature"],
            orientation="h",
            text=chart_df["importance"].round(3),
            textposition="outside",
            hovertemplate=(
                "%{y}<br>"
                "Importance: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Top 10 features used by the Random Forest",
        xaxis_title="Feature importance",
        yaxis_title="",
        height=420,
        margin=dict(l=20, r=70, t=65, b=45),
    )

    return fig


def make_telemetry_chart(selected: pd.Series, sensor_features: list[str]):
    rows = []

    for feature in sensor_features:
        info = SENSOR_INFO[feature]

        rows.append(
            {
                "Sensor": feature,
                "Symbol": info["symbol"],
                "Value": float(selected[feature]),
                "Unit": info["unit"],
            }
        )

    chart_df = pd.DataFrame(rows)

    fig = go.Figure(
        go.Bar(
            x=chart_df["Sensor"],
            y=chart_df["Value"],
            customdata=chart_df[["Symbol", "Unit"]],
            hovertemplate=(
                "<b>%{x}</b> (%{customdata[0]})<br>"
                "Value: %{y:.4f} %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Current values of the five adjustable sensors",
        xaxis_title="Sensor",
        yaxis_title="Sensor value",
        height=380,
        margin=dict(l=20, r=20, t=65, b=70),
    )

    return fig


# -----------------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------------

st.title("✈️ AeroSense")

st.caption(
    "NASA C-MAPSS FD001 turbofan Remaining Useful Life prediction using Random Forest"
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎛️ AeroSense controls")

    st.caption(
        "Choose a test engine, then adjust the five most important sensor values "
        "to see how the model responds."
    )


# -----------------------------------------------------------------------------
# Load model and metadata
# -----------------------------------------------------------------------------

model_path = ARTIFACT_DIR / "model.joblib"
meta_path = ARTIFACT_DIR / "metadata.joblib"

if not model_path.exists() or not meta_path.exists():
    st.error(
        "Model artifacts are missing. Run "
        "`python -m src.model_train` first."
    )
    st.stop()


model, metadata = load_artifacts()


# -----------------------------------------------------------------------------
# Load test data
# -----------------------------------------------------------------------------

try:
    _, test_df = load_data()

except Exception as exc:
    st.error(f"Could not load the C-MAPSS data: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Engine selection
# -----------------------------------------------------------------------------

engine_ids = sorted(
    test_df["unit_number"].unique().tolist()
)

engine_id = st.sidebar.selectbox(
    "Engine ID",
    engine_ids,
    index=0,
)


selected = (
    test_df[test_df["unit_number"] == engine_id]
    .sort_values("cycle")
    .iloc[-1]
    .copy()
)


# -----------------------------------------------------------------------------
# Sensor controls
# -----------------------------------------------------------------------------

st.sidebar.markdown("### Telemetry controls")

st.sidebar.caption(
    "The demo starts from the engine's latest observed telemetry. "
    "Change the five most important sensor inputs to see how the prediction reacts."
)


input_row = selected[metadata["features"]].copy()

# Keep the original latest telemetry so we can compare it with slider changes.
original_input_row = input_row.copy()


for feature in metadata["top_sensor_features"]:

    info = metadata["sensor_info"][feature]

    stats = metadata["feature_ranges"][feature]

    min_value = float(stats["min"])
    max_value = float(stats["max"])

    default = float(input_row[feature])

    margin = max(
        (max_value - min_value) * 0.05,
        1e-6,
    )

    slider_min = min_value - margin
    slider_max = max_value + margin

    step = max(
        (slider_max - slider_min) / 500,
        0.0001,
    )

    label = (
        f"{feature.replace('_', ' ').title()} "
        f"({info['symbol']})"
    )

    input_row[feature] = st.sidebar.slider(
        label,
        min_value=float(slider_min),
        max_value=float(slider_max),
        value=default,
        step=float(step),
        help=f"{info['description']} — {info['unit']}",
    )


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

# Original baseline prediction
baseline_prediction = float(
    max(
        model.predict(
            pd.DataFrame([original_input_row])
        )[0],
        0.0,
    )
)


# Prediction after slider adjustments
prediction = float(
    max(
        model.predict(
            pd.DataFrame([input_row])
        )[0],
        0.0,
    )
)


status, explanation = health_status(prediction)

prediction_delta = prediction - baseline_prediction


# -----------------------------------------------------------------------------
# Main prediction area
# -----------------------------------------------------------------------------

st.markdown("## 🔎 Engine health prediction")


col1, col2 = st.columns([1.3, 1])


with col1:

    st.plotly_chart(
        make_gauge(prediction),
        width="stretch",
    )


with col2:

    st.metric(
        "Current cycle",
        int(selected["cycle"]),
    )

    st.metric(
        "Predicted RUL",
        f"{prediction:.1f} cycles",
    )

    st.metric(
        "Change from baseline",
        f"{prediction_delta:+.1f} cycles",
    )

    if status == "SAFE":

        st.success(
            f"✅ {status}\n\n{explanation}"
        )

    elif status == "MAINTENANCE WARNING":

        st.warning(
            f"⚠️ {status}\n\n{explanation}"
        )

    else:

        st.error(
            f"🚨 {status}\n\n{explanation}"
        )


st.caption(
    "Baseline prediction = the model's prediction from the engine's original "
    "latest telemetry. The change value shows the effect of your slider adjustments."
)


# -----------------------------------------------------------------------------
# Model insight
# -----------------------------------------------------------------------------

st.markdown("## 🧠 Model insight")


importance_path = REPORT_DIR / "feature_importance.csv"


if importance_path.exists():

    importance = pd.read_csv(
        importance_path
    )

    st.plotly_chart(
        make_feature_importance_chart(importance),
        width="stretch",
    )

else:

    st.warning(
        "Feature-importance report is missing. "
        "Run `python -m src.model_train` first."
    )


with st.expander("📖 What does feature importance mean?"):

    st.markdown(
        "Feature importance shows how much each input feature contributed "
        "to the Random Forest's decisions during training. A larger value "
        "means the model relied more heavily on that feature across its trees; "
        "it does not by itself prove that the feature causes engine degradation."
    )


# -----------------------------------------------------------------------------
# Sensor visualization
# -----------------------------------------------------------------------------

st.markdown("## 📡 Sensor telemetry")


st.plotly_chart(
    make_telemetry_chart(
        selected,
        metadata["top_sensor_features"],
    ),
    width="stretch",
)


sensor_table = pd.DataFrame(
    [
        {
            "Sensor": feature,
            "Symbol": metadata["sensor_info"][feature]["symbol"],
            "Current value": float(selected[feature]),
            "Unit": metadata["sensor_info"][feature]["unit"],
            "Meaning": metadata["sensor_info"][feature]["description"],
        }
        for feature in metadata["top_sensor_features"]
    ]
)


st.dataframe(
    sensor_table,
    hide_index=True,
    width="stretch",
)


# -----------------------------------------------------------------------------
# Raw engine snapshot
# -----------------------------------------------------------------------------

with st.expander("📋 Engine telemetry snapshot"):

    display_cols = [
        "unit_number",
        "cycle",
        *metadata["top_sensor_features"],
    ]

    st.dataframe(
        pd.DataFrame(
            [
                selected[display_cols].to_dict()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


# -----------------------------------------------------------------------------
# Educational explanation
# -----------------------------------------------------------------------------

with st.expander("🎓 How AeroSense works"):

    st.markdown(
        """
        **1. Data** — Each row is one operating cycle from a simulated turbofan engine.

        **2. Target** — Training RUL is computed as
        `maximum engine cycle - current cycle`,
        with the training target capped at 125 cycles.

        **3. Model** — A Random Forest Regressor learns non-linear relationships
        between telemetry and RUL. Tree-based models do not require feature scaling.

        **4. Evaluation** — We report RMSE, MAE, R², and the asymmetric NASA/PHM score.

        **5. Demo** — Select a test engine, start from its latest telemetry,
        and vary the five most important sensor values to observe model sensitivity.
        """
    )


# -----------------------------------------------------------------------------
# Final disclaimer
# -----------------------------------------------------------------------------

st.info(
    "⚠️ Educational project only. The Safe / Warning / Critical thresholds are "
    "demo UI categories, not certified aviation maintenance limits."
)
