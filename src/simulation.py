"""
Sensor simulation for the landslide early-warning system.

Replaces naive `random.uniform()` sampling (which produces an
uncorrelated, teleporting reading every tick) with a stateful
mean-reverting simulator: each channel has a slowly-moving
"baseline" (which scenarios can push up or down over time) and an
observed value that tracks its baseline with small, bounded,
per-step noise — the way real sensor telemetry behaves, rather than
jumping to a fresh independent random number every tick.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

# Physical bounds for each sensor channel.
BOUNDS = {
    "soil_moisture": (0.0, 100.0),   # percent
    "tilt_angle": (0.0, 45.0),       # degrees
    "rainfall_rate": (0.0, 200.0),   # mm/hr
}

# Maximum change allowed in the *observed* reading per step.
MAX_STEP = {
    "soil_moisture": 2.0,
    "tilt_angle": 0.6,
    "rainfall_rate": 6.0,
}

# How strongly the observed value is pulled back toward its baseline
# each step (0-1). Higher = tighter tracking, less lag.
REVERSION_RATE = 0.25

# How much the baseline itself is allowed to shift per step under
# each named scenario. "random" re-rolls a small mild trend
# periodically instead of using a fixed value (see _drift_for).
SCENARIO_DRIFT = {
    "stable": {"soil_moisture": 0.0, "tilt_angle": 0.0, "rainfall_rate": 0.0},
    "random": {"soil_moisture": 0.0, "tilt_angle": 0.0, "rainfall_rate": 0.0},
    "escalating": {"soil_moisture": 0.35, "tilt_angle": 0.12, "rainfall_rate": 1.2},
    "recovering": {"soil_moisture": -0.35, "tilt_angle": -0.08, "rainfall_rate": -1.2},
}

# Coefficient coupling rainfall into the soil-moisture baseline:
# sustained rain gradually pushes moisture up, independent of scenario.
RAIN_TO_MOISTURE_COUPLING = 0.02

VALID_SCENARIOS = tuple(SCENARIO_DRIFT.keys())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class SensorSimulator:
    """Stateful generator of gradually-evolving sensor readings.

    Parameters
    ----------
    scenario:
        One of "random", "stable", "escalating", "recovering".
        "random" occasionally re-rolls a small mild trend so a long
        run still wanders realistically instead of flat-lining.
    seed:
        Optional RNG seed for reproducible sequences (used by tests).
    start:
        Optional starting reading; defaults to calm/typical values.
    """

    scenario: str = "random"
    seed: int | None = None
    start: dict | None = None

    def __post_init__(self) -> None:
        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                f"Expected one of {VALID_SCENARIOS}."
            )
        self._rng = random.Random(self.seed)
        start = self.start or {
            "soil_moisture": self._rng.uniform(15, 35),
            "tilt_angle": self._rng.uniform(1, 6),
            "rainfall_rate": self._rng.uniform(0, 8),
        }
        # `baseline` is the slow-moving target each channel tracks;
        # `state` is the actual noisy observed reading.
        self.baseline = dict(start)
        self.state = dict(start)
        self._steps = 0
        self._random_trend = {"soil_moisture": 0.0, "tilt_angle": 0.0, "rainfall_rate": 0.0}

    def _drift_for(self, channel: str) -> float:
        if self.scenario == "random":
            # Re-roll a small mild trend every ~40 steps so the
            # baseline meanders instead of sitting perfectly flat.
            if self._steps % 40 == 0:
                self._random_trend[channel] = self._rng.uniform(-0.12, 0.12)
            return self._random_trend[channel]
        return SCENARIO_DRIFT[self.scenario][channel]

    def step(self) -> dict:
        """Advance the simulation by one tick and return the new reading."""
        self._steps += 1

        for channel, (low, high) in BOUNDS.items():
            max_step = MAX_STEP[channel]

            drift = self._drift_for(channel)
            if channel == "soil_moisture":
                # Sustained rain gradually raises the moisture baseline.
                drift += self.state["rainfall_rate"] * RAIN_TO_MOISTURE_COUPLING

            self.baseline[channel] = _clamp(self.baseline[channel] + drift, low, high)

            # Observed value reverts toward the baseline with capped
            # speed, plus small bounded noise -> gradual, not jumpy.
            reversion = _clamp(
                (self.baseline[channel] - self.state[channel]) * REVERSION_RATE,
                -max_step / 2, max_step / 2,
            )
            noise = self._rng.uniform(-max_step / 2, max_step / 2)
            self.state[channel] = _clamp(self.state[channel] + reversion + noise, low, high)

        return {k: round(v, 1) for k, v in self.state.items()}

    def reset(self, start: dict | None = None) -> None:
        """Reset the simulator to a fresh starting state."""
        if start is not None:
            self.start = start
        self.__post_init__()
