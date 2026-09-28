import pytest

from src.validation import SensorValidationError, validate_reading


def valid_reading(**overrides):
    base = {"soil_moisture": 50.0, "tilt_angle": 10.0, "rainfall_rate": 20.0}
    base.update(overrides)
    return base


def test_valid_reading_passes_through():
    cleaned = validate_reading(valid_reading())
    assert cleaned == {"soil_moisture": 50.0, "tilt_angle": 10.0, "rainfall_rate": 20.0}


def test_accepts_ints_and_returns_floats():
    cleaned = validate_reading({"soil_moisture": 50, "tilt_angle": 10, "rainfall_rate": 20})
    assert all(isinstance(v, float) for v in cleaned.values())


def test_missing_field_raises():
    reading = valid_reading()
    del reading["tilt_angle"]
    with pytest.raises(SensorValidationError, match="Missing required field"):
        validate_reading(reading)


def test_unexpected_field_raises():
    reading = valid_reading(extra_sensor=1.0)
    with pytest.raises(SensorValidationError, match="Unexpected field"):
        validate_reading(reading)


@pytest.mark.parametrize("field,value", [
    ("soil_moisture", -5),
    ("soil_moisture", 150),
    ("tilt_angle", -1),
    ("tilt_angle", 91),
    ("rainfall_rate", -1),
    ("rainfall_rate", 501),
])
def test_out_of_range_raises(field, value):
    with pytest.raises(SensorValidationError, match="out of valid range"):
        validate_reading(valid_reading(**{field: value}))


def test_non_numeric_type_raises():
    with pytest.raises(SensorValidationError, match="must be numeric"):
        validate_reading(valid_reading(soil_moisture="wet"))


def test_boolean_rejected_as_numeric():
    with pytest.raises(SensorValidationError, match="must be numeric"):
        validate_reading(valid_reading(tilt_angle=True))


def test_nan_and_inf_rejected():
    with pytest.raises(SensorValidationError, match="finite"):
        validate_reading(valid_reading(rainfall_rate=float("nan")))
    with pytest.raises(SensorValidationError, match="finite"):
        validate_reading(valid_reading(rainfall_rate=float("inf")))


def test_non_dict_input_raises():
    with pytest.raises(SensorValidationError, match="must be a dict"):
        validate_reading(["not", "a", "dict"])
