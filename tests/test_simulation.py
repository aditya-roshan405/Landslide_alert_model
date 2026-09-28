import pytest

from src.simulation import BOUNDS, MAX_STEP, SensorSimulator


def test_readings_stay_within_physical_bounds():
    sim = SensorSimulator(scenario="escalating", seed=1)
    for _ in range(500):
        reading = sim.step()
        for channel, (low, high) in BOUNDS.items():
            assert low <= reading[channel] <= high


def test_steps_are_gradual_not_teleporting():
    sim = SensorSimulator(scenario="random", seed=2)
    prev = sim.step()
    for _ in range(200):
        curr = sim.step()
        for channel, max_step in MAX_STEP.items():
            assert abs(curr[channel] - prev[channel]) <= max_step + 0.01
        prev = curr


def test_escalating_scenario_trends_upward():
    sim = SensorSimulator(scenario="escalating", seed=3, start={
        "soil_moisture": 10.0, "tilt_angle": 2.0, "rainfall_rate": 5.0,
    })
    first = sim.step()
    for _ in range(100):
        last = sim.step()
    assert last["soil_moisture"] > first["soil_moisture"]
    assert last["tilt_angle"] > first["tilt_angle"]


def test_stable_scenario_stays_near_start():
    start = {"soil_moisture": 30.0, "tilt_angle": 5.0, "rainfall_rate": 10.0}
    sim = SensorSimulator(scenario="stable", seed=4, start=start)
    readings = [sim.step() for _ in range(150)]
    final = readings[-1]
    # No directional drift, so it shouldn't wander to the extremes.
    assert 0 < final["soil_moisture"] < 80
    assert 0 <= final["tilt_angle"] < 30


def test_invalid_scenario_raises():
    with pytest.raises(ValueError):
        SensorSimulator(scenario="apocalyptic")


def test_reproducible_with_seed():
    sim1 = SensorSimulator(scenario="random", seed=42)
    sim2 = SensorSimulator(scenario="random", seed=42)
    for _ in range(20):
        assert sim1.step() == sim2.step()
