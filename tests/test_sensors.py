"""Sensor models, noise and fault injection."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.pool import make_pool
from zimablue.rng import RngTree
from zimablue.sensors import (
    IMU,
    ContactSensor,
    Encoder,
    PressureSensor,
    SensorConfig,
    SensorContext,
    SensorSuite,
    Sonar,
)


def _drive(suite, steps=400, dt=0.01, **truth):
    """Poll a suite over time and collect the fresh readings."""
    ctx = SensorContext(**truth)
    out = []
    for i in range(steps):
        ctx.time = i * dt
        readings = suite.update(ctx)
        out.append({k: v for k, v in readings.items() if v.fresh})
    return out


def test_sensor_must_be_attached_before_use():
    sensor = PressureSensor()
    with pytest.raises(RuntimeError, match="attached"):
        sensor.update(SensorContext())


def test_sampling_rate_is_respected():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=10.0, latency=0.0))])
    suite.attach(RngTree(0))
    fresh = sum(1 for frame in _drive(suite, steps=100, dt=0.01, depth=1.0) if "p" in frame)
    assert fresh == pytest.approx(10, abs=1)


def test_readings_are_held_between_samples():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=5.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    ctx = SensorContext(depth=1.5)
    values = []
    for i in range(40):
        ctx.time = i * 0.01
        reading = suite.update(ctx).get("p")
        if reading is not None:
            values.append((reading.value, reading.fresh))
    held = [v for v, fresh in values if not fresh]
    assert held, "expected some held samples between the 5 Hz updates"


def test_noise_is_zero_mean_and_scaled_correctly():
    sigma = 0.05
    suite = SensorSuite(
        [PressureSensor("p", SensorConfig(rate_hz=100.0, noise_std=sigma, latency=0.0))]
    )
    suite.attach(RngTree(3))
    samples = [f["p"].value for f in _drive(suite, steps=4000, dt=0.01, depth=2.0) if "p" in f]
    assert np.mean(samples) == pytest.approx(2.0, abs=0.01)
    assert np.std(samples) == pytest.approx(sigma, rel=0.15)


def test_latency_delays_the_first_reading():
    suite = SensorSuite(
        [PressureSensor("p", SensorConfig(rate_hz=50.0, latency=0.1, noise_std=0.0))]
    )
    suite.attach(RngTree(0))
    ctx = SensorContext(depth=1.0)
    first = None
    for i in range(50):
        ctx.time = i * 0.01
        if suite.update(ctx).get("p") is not None:
            first = ctx.time
            break
    assert first == pytest.approx(0.1, abs=0.011)


def test_saturation_clips_to_range():
    suite = SensorSuite(
        [PressureSensor("p", SensorConfig(rate_hz=50.0, max_value=2.0, noise_std=0.0))]
    )
    suite.attach(RngTree(0))
    values = [f["p"].value for f in _drive(suite, steps=50, depth=99.0) if "p" in f]
    assert values and max(values) <= 2.0


def test_encoder_reports_wheel_speed_not_ground_speed():
    """Slip is what makes odometry drift, so the encoder must see the wheels."""
    suite = SensorSuite([Encoder("encoder", SensorConfig(rate_hz=50.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    frames = _drive(suite, steps=20, wheel_speed_left=0.3, wheel_speed_right=0.1, speed=0.05)
    reading = next(f["encoder"] for f in frames if "encoder" in f)
    assert reading[0] == pytest.approx(0.3, abs=0.01)
    assert reading[1] == pytest.approx(0.1, abs=0.01)


def test_imu_gyro_carries_a_turn_on_bias():
    suite = SensorSuite([IMU("imu", SensorConfig(rate_hz=100.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    frames = _drive(suite, steps=10, yaw_rate=0.0)
    gz = next(f["imu"][2] for f in frames if "imu" in f)
    assert abs(gz) > 1e-4, "a real gyro is not perfectly zeroed at rest"


def test_contact_channels_are_binary():
    suite = SensorSuite([ContactSensor("contact")])
    suite.attach(RngTree(0))
    frames = _drive(suite, steps=10, contacts=(True, False, True, False))
    values = next(f["contact"].values for f in frames if "contact" in f)
    assert list(values) == [1.0, 0.0, 1.0, 0.0]


def test_sonar_measures_pool_geometry():
    pool = make_pool("rectangular")
    suite = SensorSuite(
        [
            Sonar(
                "sonar",
                SensorConfig(rate_hz=50.0, noise_std=0.0, latency=0.0, dropout_probability=0.0),
                beam_angles=(0.0,),
                max_range=8.0,
            )
        ]
    )
    suite.attach(RngTree(0))
    frames = _drive(suite, steps=10, pool=pool, water=pool.water, x=2.0, y=2.5, heading=0.0)
    value = next(f["sonar"].value for f in frames if "sonar" in f)
    assert value == pytest.approx(8.0, abs=0.05)


def test_dropout_marks_readings_invalid():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=100.0, dropout_probability=0.5))])
    suite.attach(RngTree(5))
    frames = [f["p"] for f in _drive(suite, steps=2000, dt=0.01, depth=1.0) if "p" in f]
    invalid = sum(1 for r in frames if not r.valid)
    assert 0.3 < invalid / len(frames) < 0.7


def test_injected_bias_shifts_readings():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=50.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    suite.p.inject_fault(bias=0.4)
    value = next(f["p"].value for f in _drive(suite, steps=10, depth=1.0) if "p" in f)
    assert value == pytest.approx(1.4, abs=0.01)


def test_fault_can_start_partway_through_a_run():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=100.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    suite.p.inject_fault(bias=1.0, start_time=1.0)
    frames = _drive(suite, steps=200, dt=0.01, depth=2.0)
    before = [f["p"].value for i, f in enumerate(frames) if "p" in f and i * 0.01 < 0.9]
    after = [f["p"].value for i, f in enumerate(frames) if "p" in f and i * 0.01 > 1.1]
    assert np.mean(before) == pytest.approx(2.0, abs=0.02)
    assert np.mean(after) == pytest.approx(3.0, abs=0.02)


def test_stuck_fault_freezes_the_output():
    suite = SensorSuite([PressureSensor("p", SensorConfig(rate_hz=100.0, noise_std=0.0))])
    suite.attach(RngTree(0))
    suite.p.inject_fault(stuck=True, start_time=0.5)
    ctx = SensorContext()
    values = []
    for i in range(200):
        ctx.time = i * 0.01
        ctx.depth = 1.0 + i * 0.01
        reading = suite.update(ctx).get("p")
        if reading is not None and reading.fresh and ctx.time > 0.6:
            values.append(reading.value)
    assert len(set(np.round(values, 6))) == 1, "a stuck sensor must not change"


def test_same_seed_gives_identical_noise():
    def run(seed):
        suite = SensorSuite([IMU("imu")])
        suite.attach(RngTree(seed))
        return [
            f["imu"].values.copy() for f in _drive(suite, steps=200, yaw_rate=0.3) if "imu" in f
        ]

    assert np.array_equal(run(11), run(11))
    assert not np.array_equal(run(11), run(12))


def test_adding_a_sensor_does_not_perturb_another():
    """Named RNG streams mean sensors are independent of the suite's contents."""

    def imu_values(sensors):
        suite = SensorSuite(sensors)
        suite.attach(RngTree(99))
        return [
            f["imu"].values.copy() for f in _drive(suite, steps=100, yaw_rate=0.1) if "imu" in f
        ]

    alone = imu_values([IMU("imu")])
    with_company = imu_values([IMU("imu"), Encoder("encoder"), PressureSensor("pressure")])
    assert np.array_equal(alone, with_company)


def test_suite_reports_helpful_error_for_unknown_sensor():
    suite = SensorSuite([IMU("imu")])
    with pytest.raises(AttributeError, match="sonar"):
        _ = suite.sonar


def test_sensor_specs_round_trip():
    suite = SensorSuite([Sonar("sonar", beam_angles=(0.0, 0.5)), IMU("imu")])
    suite.sonar.inject_fault(bias=0.2, dropout_probability=0.1)
    restored = SensorSuite.from_specs(suite.specs())
    assert list(restored) == list(suite)
    assert restored.sonar.beam_angles == (0.0, 0.5)
    assert restored.sonar.faults[0].bias == pytest.approx(0.2)
