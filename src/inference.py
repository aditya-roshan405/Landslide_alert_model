"""
Landslide real-time inference.

Loads the trained model, runs validated sensor readings through it,
and returns both the predicted risk label and the full class
probability distribution (not just a single hard label).
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from src.model import FEATURES, LABELS, MODEL_PATH
from src.simulation import SensorSimulator
from src.validation import SensorValidationError, validate_reading


def load_model(path: str = MODEL_PATH):
    """Load the trained model from disk."""
    return joblib.load(path)


def predict(model, reading: dict) -> dict:
    """Validate a reading, run inference, and return label + probabilities.

    Raises SensorValidationError if the reading is invalid.
    """
    clean = validate_reading(reading)
    X = pd.DataFrame([clean], columns=FEATURES)

    proba = model.predict_proba(X)[0]
    pred_idx = int(proba.argmax())

    return {
        "reading": clean,
        "risk_label": LABELS[pred_idx],
        "risk_probability": float(proba[pred_idx]),
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(proba)},
    }


def alert_if_danger(result: dict) -> None:
    """Print an emergency alert if the predicted risk is Danger."""
    if result["risk_label"] == "Danger":
        r = result["reading"]
        print("\U0001f6a8" * 10)
        print("EMERGENCY ALERT: LANDSLIDE DANGER DETECTED!")
        print(f"  Soil Moisture: {r['soil_moisture']}%")
        print(f"  Tilt Angle:    {r['tilt_angle']}\u00b0")
        print(f"  Rainfall Rate: {r['rainfall_rate']} mm/hr")
        print(f"  Confidence:    {result['risk_probability']:.1%}")
        print("  ACTION: EVACUATE AREA IMMEDIATELY")
        print("\U0001f6a8" * 10)


def stream_simulation(model, n_readings: int = 20, delay: float = 0.3, scenario: str = "random") -> None:
    """Simulate a real-time stream of sensor data and run inference on each."""
    simulator = SensorSimulator(scenario=scenario)

    for i in range(1, n_readings + 1):
        reading = simulator.step()
        try:
            result = predict(model, reading)
        except SensorValidationError as exc:
            print(f"[{i}] Rejected reading {json.dumps(reading)}: {exc}")
            time.sleep(delay)
            continue

        print(
            f"[{i}] Reading: {json.dumps(result['reading'])} -> "
            f"Risk: {result['risk_label']} ({result['risk_probability']:.1%})"
        )
        alert_if_danger(result)
        time.sleep(delay)


if __name__ == "__main__":
    model = load_model()
    stream_simulation(model, n_readings=30, delay=0.2, scenario="escalating")
