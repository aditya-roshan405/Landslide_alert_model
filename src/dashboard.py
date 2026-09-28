"""
Landslide Early Warning Dashboard
Streamlit UI: gradual live sensor simulation, per-sensor status,
separate charts per channel, risk probability, and a history table.

Run with: streamlit run src/dashboard.py
"""

import os
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# Make `src` importable regardless of the directory streamlit is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import LABELS, MODEL_PATH  # noqa: E402
from src.simulation import VALID_SCENARIOS, SensorSimulator  # noqa: E402
from src.validation import SensorValidationError, validate_reading  # noqa: E402

STATUS_COLOR = {"Safe": "#2ecc71", "Warning": "#f1c40f", "Danger": "#e74c3c"}
MAX_POINTS = 40  # rolling window for charts
HISTORY_ROWS = 15  # rows shown in the history table

# Per-sensor "yellow" and "red" thresholds, used only for the small
# individual sensor-status badges (independent of the model's own
# combined risk prediction).
SENSOR_THRESHOLDS = {
    "soil_moisture": (40, 70),   # (warn >=, danger >=)
    "tilt_angle": (15, 25),
    "rainfall_rate": (50, 100),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def sensor_status(channel: str, value: float) -> str:
    warn, danger = SENSOR_THRESHOLDS[channel]
    if value >= danger:
        return "Danger"
    if value >= warn:
        return "Warning"
    return "Safe"


def predict_risk(model, reading: dict) -> dict:
    from src.model import FEATURES

    X = pd.DataFrame([reading], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    return {
        "risk_label": LABELS[pred_idx],
        "risk_probability": float(proba[pred_idx]),
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(proba)},
    }


def render_status_box(placeholder, risk_label: str, risk_probability: float) -> None:
    color = STATUS_COLOR[risk_label]
    placeholder.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:24px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:28px;
            font-weight:700;
            letter-spacing:1px;
        ">
            STATUS: {risk_label.upper()} &nbsp;&middot;&nbsp; {risk_probability:.0%} confidence
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sensor_badges(placeholder, reading: dict) -> None:
    cols = placeholder.columns(3)
    labels = {
        "soil_moisture": ("Soil Moisture", "%"),
        "tilt_angle": ("Tilt Angle", "\u00b0"),
        "rainfall_rate": ("Rainfall Rate", " mm/hr"),
    }
    for col, (channel, (name, unit)) in zip(cols, labels.items()):
        value = reading[channel]
        status = sensor_status(channel, value)
        color = STATUS_COLOR[status]
        col.markdown(
            f"""
            <div style="
                border:2px solid {color};
                border-radius:8px;
                padding:10px;
                text-align:center;
            ">
                <div style="font-size:13px; color:#888;">{name}</div>
                <div style="font-size:22px; font-weight:700;">{value}{unit}</div>
                <div style="color:{color}; font-weight:600;">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def init_state() -> None:
    if "chart_history" not in st.session_state:
        st.session_state.chart_history = pd.DataFrame(
            columns=["soil_moisture", "tilt_angle", "rainfall_rate", "Safe", "Warning", "Danger"]
        )
    if "table_history" not in st.session_state:
        st.session_state.table_history = pd.DataFrame(
            columns=[
                "time", "soil_moisture", "tilt_angle", "rainfall_rate",
                "risk_label", "confidence",
            ]
        )
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator(scenario="random")
    if "rejected_count" not in st.session_state:
        st.session_state.rejected_count = 0


def main() -> None:
    st.set_page_config(page_title="Landslide Early Warning", layout="wide")
    st.title("\U0001f6f0\ufe0f Landslide Early Warning System")
    st.caption("Live IoT sensor monitoring dashboard \u2014 gradual, physically-bounded simulation")

    model = load_model()
    init_state()

    run = st.sidebar.toggle("Start live simulation", value=True)
    delay = st.sidebar.slider("Update interval (sec)", 0.2, 2.0, 0.5)
    scenario = st.sidebar.selectbox(
        "Simulation scenario", VALID_SCENARIOS, index=VALID_SCENARIOS.index("random")
    )
    if scenario != st.session_state.simulator.scenario:
        st.session_state.simulator = SensorSimulator(scenario=scenario)

    if st.sidebar.button("Reset simulation"):
        st.session_state.simulator = SensorSimulator(scenario=scenario)
        st.session_state.chart_history = st.session_state.chart_history.iloc[0:0]
        st.session_state.table_history = st.session_state.table_history.iloc[0:0]
        st.session_state.rejected_count = 0

    if st.session_state.rejected_count:
        st.sidebar.warning(f"Rejected readings so far: {st.session_state.rejected_count}")

    status_placeholder = st.empty()
    badges_placeholder = st.empty()

    st.subheader("Sensor Trends")
    c1, c2, c3 = st.columns(3)
    moisture_chart = c1.empty()
    tilt_chart = c2.empty()
    rain_chart = c3.empty()

    st.subheader("Risk Probability Over Time")
    proba_chart = st.empty()

    st.subheader("Risk History")
    table_placeholder = st.empty()

    while run:
        raw_reading = st.session_state.simulator.step()

        try:
            reading = validate_reading(raw_reading)
        except SensorValidationError as exc:
            st.session_state.rejected_count += 1
            st.sidebar.error(f"Invalid reading skipped: {exc}")
            time.sleep(delay)
            st.rerun()

        result = predict_risk(model, reading)

        # -- charts (separate per channel + probability chart) --
        chart_row = pd.DataFrame([{**reading, **result["probabilities"]}])
        st.session_state.chart_history = pd.concat(
            [st.session_state.chart_history, chart_row], ignore_index=True
        ).tail(MAX_POINTS)

        moisture_chart.line_chart(st.session_state.chart_history[["soil_moisture"]])
        tilt_chart.line_chart(st.session_state.chart_history[["tilt_angle"]])
        rain_chart.line_chart(st.session_state.chart_history[["rainfall_rate"]])
        proba_chart.line_chart(st.session_state.chart_history[["Safe", "Warning", "Danger"]])

        # -- history table --
        table_row = pd.DataFrame([{
            "time": datetime.now().strftime("%H:%M:%S"),
            "soil_moisture": reading["soil_moisture"],
            "tilt_angle": reading["tilt_angle"],
            "rainfall_rate": reading["rainfall_rate"],
            "risk_label": result["risk_label"],
            "confidence": f"{result['risk_probability']:.0%}",
        }])
        st.session_state.table_history = pd.concat(
            [table_row, st.session_state.table_history], ignore_index=True
        ).head(HISTORY_ROWS)
        table_placeholder.dataframe(st.session_state.table_history, use_container_width=True, hide_index=True)

        # -- status + badges --
        render_status_box(status_placeholder, result["risk_label"], result["risk_probability"])
        render_sensor_badges(badges_placeholder, reading)

        time.sleep(delay)
        st.rerun()


if __name__ == "__main__":
    main()
