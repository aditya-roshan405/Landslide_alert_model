"""
Landslide Early Warning Dashboard
Minimalist Streamlit UI: live sensor charts + status indicator.
Run with: streamlit run landslide_dashboard.py
"""

import time
import random
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "landslide_model.pkl"
LABELS = {0: "Safe", 1: "Warning", 2: "Danger"}
STATUS_COLOR = {"Safe": "#2ecc71", "Warning": "#f1c40f", "Danger": "#e74c3c"}
MAX_POINTS = 30  # rolling window for chart


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def generate_reading():
    """Simulate one sensor reading."""
    return {
        "soil_moisture": round(random.uniform(0, 100), 1),
        "tilt_angle": round(random.uniform(0, 45), 1),
        "rainfall_rate": round(random.uniform(0, 200), 1),
    }


def predict_risk(model, reading):
    X = pd.DataFrame([reading], columns=["soil_moisture", "tilt_angle", "rainfall_rate"])
    return LABELS[model.predict(X)[0]]


def render_status_box(placeholder, risk_label):
    color = STATUS_COLOR[risk_label]
    placeholder.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:28px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:32px;
            font-weight:700;
            letter-spacing:1px;
        ">
            STATUS: {risk_label.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Landslide Early Warning", layout="wide")
    st.title("🛰️ Landslide Early Warning System")
    st.caption("Live IoT sensor monitoring dashboard")

    model = load_model()

    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(
            columns=["soil_moisture", "tilt_angle"]
        )

    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()

    run = st.sidebar.toggle("Start live simulation", value=True)
    delay = st.sidebar.slider("Update interval (sec)", 0.2, 2.0, 0.5)

    while run:
        reading = generate_reading()
        risk_label = predict_risk(model, reading)

        # update rolling history
        new_row = pd.DataFrame([{
            "soil_moisture": reading["soil_moisture"],
            "tilt_angle": reading["tilt_angle"],
        }])
        st.session_state.history = pd.concat(
            [st.session_state.history, new_row], ignore_index=True
        ).tail(MAX_POINTS)

        # render status
        render_status_box(status_placeholder, risk_label)

        # render chart
        chart_placeholder.line_chart(st.session_state.history)

        # render current metrics
        c1, c2, c3 = metrics_placeholder.columns(3)
        c1.metric("Soil Moisture (%)", reading["soil_moisture"])
        c2.metric("Tilt Angle (°)", reading["tilt_angle"])
        c3.metric("Rainfall Rate (mm/hr)", reading["rainfall_rate"])

        time.sleep(delay)
        st.rerun()


if __name__ == "__main__":
    main()
