"""
Input validation for incoming sensor readings.

Used by both the live dashboard loop and the batch/stream inference
script so that a malformed, missing, out-of-range, or wrong-typed
reading is rejected with a clear error instead of silently reaching
the model or crashing deep inside pandas/sklearn.
"""

from __future__ import annotations

import math

# field -> (min, max) accepted range. Kept slightly wider than the
# simulator's own bounds so real sensors with a bit of overshoot
# aren't rejected outright.
VALID_RANGES = {
    "soil_moisture": (0.0, 100.0),
    "tilt_angle": (0.0, 90.0),
    "rainfall_rate": (0.0, 500.0),
}

REQUIRED_FIELDS = tuple(VALID_RANGES.keys())


class SensorValidationError(ValueError):
    """Raised when a sensor reading fails validation."""


def validate_reading(reading: dict) -> dict:
    """Validate and normalize a raw sensor reading.

    Returns a new dict with plain ``float`` values on success.
    Raises :class:`SensorValidationError` on any problem, with a
    message describing exactly what was wrong.
    """
    if not isinstance(reading, dict):
        raise SensorValidationError(
            f"Reading must be a dict, got {type(reading).__name__}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in reading]
    if missing:
        raise SensorValidationError(f"Missing required field(s): {', '.join(missing)}")

    extra = [f for f in reading if f not in VALID_RANGES]
    if extra:
        raise SensorValidationError(f"Unexpected field(s): {', '.join(extra)}")

    cleaned = {}
    for field, (low, high) in VALID_RANGES.items():
        value = reading[field]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SensorValidationError(
                f"'{field}' must be numeric, got {type(value).__name__}: {value!r}"
            )

        value = float(value)
        if math.isnan(value) or math.isinf(value):
            raise SensorValidationError(f"'{field}' must be a finite number, got {value}")

        if not (low <= value <= high):
            raise SensorValidationError(
                f"'{field}' = {value} is out of valid range [{low}, {high}]"
            )

        cleaned[field] = value

    return cleaned
