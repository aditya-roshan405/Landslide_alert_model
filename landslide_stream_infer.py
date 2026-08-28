"""
Landslide Real-Time Inference Simulator
Loads trained model, simulates incoming sensor JSON stream, runs inference.
"""

import json
import random
import time
import joblib
import pandas as pd

MODEL_PATH = "landslide_model.pkl"
LABELS = {0: "Safe", 1: "Warning", 2: "Danger"}


def load_model(path=MODEL_PATH):
    """Load trained model from disk."""
    return joblib.load(path)


def generate_sensor_reading():
    """Simulate one incoming sensor JSON payload."""
    reading = {
        "soil_moisture": round(random.uniform(0, 100), 1),
        "tilt_angle": round(random.uniform(0, 45), 1),
        "rainfall_rate": round(random.uniform(0, 200), 1),
    }
    return json.dumps(reading)


def predict(model, sensor_json):
    """Run inference on a single JSON sensor reading."""
    data = json.loads(sensor_json)
    X = pd.DataFrame([data], columns=["soil_moisture", "tilt_angle", "rainfall_rate"])
    pred = model.predict(X)[0]
    return data, LABELS[pred]


def alert_if_danger(data, risk_label):
    """Print emergency alert if risk is Danger."""
    if risk_label == "Danger":
        print("🚨" * 10)
        print("EMERGENCY ALERT: LANDSLIDE DANGER DETECTED!")
        print(f"  Soil Moisture: {data['soil_moisture']}%")
        print(f"  Tilt Angle:    {data['tilt_angle']}°")
        print(f"  Rainfall Rate: {data['rainfall_rate']} mm/hr")
        print("  ACTION: EVACUATE AREA IMMEDIATELY")
        print("🚨" * 10)


def stream_simulation(model, n_readings=20, delay=0.5):
    """Simulate a real-time stream of sensor data and run inference on each."""
    for i in range(1, n_readings + 1):
        sensor_json = generate_sensor_reading()
        data, risk_label = predict(model, sensor_json)

        print(f"[{i}] Reading: {sensor_json} -> Risk: {risk_label}")
        alert_if_danger(data, risk_label)

        time.sleep(delay)


if __name__ == "__main__":
    model = load_model()
    stream_simulation(model, n_readings=20, delay=0.3)
