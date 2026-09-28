import numpy as np

from src.model import (
    FEATURES,
    LABELS,
    classify_risk,
    evaluate_model,
    generate_synthetic_data,
    train_model,
)


def test_generate_synthetic_data_shape_and_labels():
    df = generate_synthetic_data(n=200, seed=1)
    assert len(df) == 200
    assert set(FEATURES).issubset(df.columns)
    assert set(df["risk_level"].unique()).issubset(set(LABELS.keys()))


def test_classify_risk_rules():
    import pandas as pd

    danger = pd.Series({"soil_moisture": 80, "tilt_angle": 30, "rainfall_rate": 120})
    warning = pd.Series({"soil_moisture": 50, "tilt_angle": 18, "rainfall_rate": 60})
    safe = pd.Series({"soil_moisture": 5, "tilt_angle": 2, "rainfall_rate": 5})

    assert classify_risk(danger) == 2
    assert classify_risk(warning) == 1
    assert classify_risk(safe) == 0


def test_train_and_evaluate_produces_reasonable_metrics():
    df = generate_synthetic_data(n=600, seed=7)
    model, X_test, y_test = train_model(df, seed=7)
    results = evaluate_model(model, X_test, y_test)

    assert 0.0 <= results["accuracy"] <= 1.0
    # The rule-based labels are cleanly separable, so a forest should do well.
    assert results["accuracy"] > 0.85

    cm = results["confusion_matrix"]
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_test)

    for label_name, metrics in results["per_class"].items():
        assert label_name in LABELS.values()
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0


def test_model_predicts_expected_labels():
    df = generate_synthetic_data(n=400, seed=3)
    model, X_test, _ = train_model(df, seed=3)
    preds = model.predict(X_test)
    assert set(np.unique(preds)).issubset(set(LABELS.keys()))
