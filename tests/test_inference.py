import pytest

from src.inference import predict
from src.model import LABELS, generate_synthetic_data, train_model
from src.validation import SensorValidationError


@pytest.fixture(scope="module")
def trained_model():
    df = generate_synthetic_data(n=500, seed=11)
    model, _, _ = train_model(df, seed=11)
    return model


def test_predict_returns_label_and_probabilities(trained_model):
    reading = {"soil_moisture": 80.0, "tilt_angle": 30.0, "rainfall_rate": 120.0}
    result = predict(trained_model, reading)

    assert result["risk_label"] in LABELS.values()
    assert set(result["probabilities"].keys()) == set(LABELS.values())
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-6) == 1.0
    assert 0.0 <= result["risk_probability"] <= 1.0
    # the reported confidence should match the max probability
    assert result["risk_probability"] == max(result["probabilities"].values())


def test_predict_rejects_invalid_reading(trained_model):
    with pytest.raises(SensorValidationError):
        predict(trained_model, {"soil_moisture": 500.0, "tilt_angle": 10.0, "rainfall_rate": 5.0})


def test_predict_rejects_missing_field(trained_model):
    with pytest.raises(SensorValidationError):
        predict(trained_model, {"soil_moisture": 10.0, "rainfall_rate": 5.0})
