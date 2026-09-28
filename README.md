# Landslide Early Warning System

IoT-style landslide risk monitoring: a Random Forest classifier over
soil moisture, tilt angle, and rainfall rate, served through a live
Streamlit dashboard and a standalone stream-inference script.

## Project structure

```
landslide_alert_model/
├── src/
│   ├── simulation.py    # gradual, bounded sensor simulator (replaces random.uniform per tick)
│   ├── validation.py    # input validation for sensor readings
│   ├── model.py          # synthetic data, training, evaluation (confusion matrix, P/R/F1)
│   ├── inference.py      # stream inference: label + full risk-probability distribution
│   └── dashboard.py      # Streamlit UI: per-sensor charts, status badges, history table
├── tests/
│   ├── test_simulation.py
│   ├── test_validation.py
│   ├── test_model.py
│   └── test_inference.py
├── models/               # generated: landslide_model.pkl, confusion_matrix.png, metrics_report.txt
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.model
```

Writes `models/landslide_model.pkl`, `models/confusion_matrix.png`, and
`models/metrics_report.txt` (accuracy plus per-class precision/recall/F1).

## Run the dashboard

```bash
streamlit run src/dashboard.py
```

Sidebar controls: start/stop the simulation, update interval, and a
simulation **scenario** (`random`, `stable`, `escalating`,
`recovering`) so you can demo a slide toward Danger instead of only
watching noise.

## Run the stream-inference script

```bash
python -m src.inference
```

## Run the tests

```bash
pytest
```

## What changed from the original prototype

- **Simulation**: `SensorSimulator` does a bounded random walk per
  channel (with a rainfall→soil-moisture coupling) instead of
  independently re-rolling `random.uniform()` every tick, so
  consecutive readings move gradually like real telemetry.
- **Risk probability**: predictions now return the full class
  probability distribution (`predict_proba`), not just a hard label.
- **Separate charts**: soil moisture, tilt angle, and rainfall rate
  each get their own chart, plus a dedicated risk-probability chart.
- **Risk history table**: the last 15 readings with timestamp,
  sensor values, predicted label, and confidence.
- **Input validation**: `validate_reading()` checks types, range,
  and required fields before a reading ever reaches the model.
- **Sensor-status indicators**: independent per-sensor Safe/Warning
  /Danger badges based on fixed thresholds, alongside the model's
  combined risk status.
- **Confusion matrix + precision/recall/F1**: `evaluate_model()`
  computes and `model.py`'s `main()` saves a confusion matrix image
  and a metrics report.
- **Tests**: pytest coverage for the simulator's bounds/gradualness,
  validation edge cases, model training/metrics, and inference
  output shape.
