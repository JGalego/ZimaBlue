"""The EKF pose estimator."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.estimation import EstimatorConfig, PoseEstimator
from zimablue.geometry import wrap_angle


def test_straight_line_dead_reckoning():
    estimator = PoseEstimator()
    for _ in range(1000):
        estimator.predict(0.25, 0.0, 0.01)
    estimate = estimator.estimate
    assert estimate.x == pytest.approx(2.5, rel=1e-6)
    assert estimate.y == pytest.approx(0.0, abs=1e-9)


def test_turning_in_place_moves_only_the_heading():
    estimator = PoseEstimator()
    for _ in range(100):
        estimator.predict(0.0, 1.0, 0.01)
    estimate = estimator.estimate
    assert estimate.heading == pytest.approx(1.0, rel=1e-6)
    assert (estimate.x, estimate.y) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_a_closed_circle_returns_to_the_start():
    """The midpoint update must not spiral, or filter error would masquerade
    as sensor error."""
    estimator = PoseEstimator()
    dt, omega, v = 0.001, 2 * np.pi, 0.5
    for _ in range(1000):
        estimator.predict(v, omega, dt)
    estimate = estimator.estimate
    assert (estimate.x, estimate.y) == pytest.approx((0.0, 0.0), abs=2e-4)
    assert wrap_angle(estimate.heading) == pytest.approx(0.0, abs=1e-9)


def test_uncertainty_grows_while_driving():
    estimator = PoseEstimator()
    first = []
    for i in range(500):
        estimator.predict(0.25, 0.0, 0.01)
        if i in (10, 499):
            first.append(estimator.estimate.position_sigma)
    assert first[1] > first[0] > 0.0


def test_zupt_recovers_the_gyro_bias():
    """The whole reason the bias is a state.

    A stationary gyro reading 0.02 rad/s *is* biased by 0.02 rad/s, and only a
    zero-velocity update can see that.
    """
    bias = 0.02
    estimator = PoseEstimator()
    assert estimator.estimate.gyro_bias == pytest.approx(0.0)

    for _ in range(400):
        estimator.predict(0.0, bias, 0.01)
        estimator.zero_velocity_update(bias, 0.01, moving=False)

    estimate = estimator.estimate
    assert estimate.gyro_bias == pytest.approx(bias, abs=0.004)
    assert estimate.bias_sigma < 0.02


def test_zupt_waits_for_the_robot_to_settle():
    """Firing immediately would fold the deceleration transient into the bias."""
    estimator = PoseEstimator(EstimatorConfig(zupt_settle=0.5))
    applied = [estimator.zero_velocity_update(0.01, 0.01, moving=False) for _ in range(100)]
    assert not any(applied[:49]), "fired before the 0.5 s settle time"
    assert any(applied[50:]), "never fired after settling"


def test_zupt_does_not_fire_while_moving():
    estimator = PoseEstimator()
    for _ in range(200):
        assert not estimator.zero_velocity_update(0.02, 0.01, moving=True)
    assert estimator.estimate.zupt_count == 0


def test_bias_estimate_reduces_heading_drift():
    """End to end: with ZUPTs the heading error should be far smaller."""
    bias = 0.02
    dt = 0.01

    def run(with_zupt: bool) -> float:
        estimator = PoseEstimator()
        truth = 0.0
        # The robot drives dead straight, so true heading never changes and
        # every radian of estimated heading is pure bias error.
        for step in range(6000):
            paused = with_zupt and (step % 500) >= 460
            speed = 0.0 if paused else 0.25
            estimator.predict(speed, bias, dt)
            estimator.zero_velocity_update(bias, dt, moving=not paused)
        return abs(wrap_angle(estimator.estimate.heading - truth))

    assert run(True) < 0.25 * run(False)


def test_position_update_pulls_the_estimate():
    estimator = PoseEstimator()
    for _ in range(500):
        estimator.predict(0.25, 0.0, 0.01)
    before = estimator.estimate.x
    estimator.position_update(0.5, 0.0, sigma=0.01)
    after = estimator.estimate
    assert abs(after.x - 0.5) < abs(before - 0.5)
    assert after.position_sigma < 1.0


def test_heading_update_wraps_correctly():
    estimator = PoseEstimator()
    estimator.predict(0.0, 0.0, 0.01)
    estimator.state[2] = np.pi - 0.05
    estimator.heading_update(-np.pi + 0.05, sigma=0.01)
    # Should move the short way across the wrap, not 2*pi the long way.
    assert abs(wrap_angle(estimator.estimate.heading)) > 3.0


def test_covariance_stays_symmetric_and_positive_definite():
    """Joseph-form updates over a long run; the naive form drifts indefinite."""
    estimator = PoseEstimator()
    rng = np.random.default_rng(0)
    for step in range(5000):
        estimator.predict(0.25, float(rng.normal(0, 0.1)), 0.01)
        if step % 100 == 0:
            estimator.zero_velocity_update(0.01, 0.01, moving=False)
    covariance = estimator.estimate.covariance
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > -1e-12)


def test_reset_clears_state():
    estimator = PoseEstimator()
    for _ in range(100):
        estimator.predict(0.3, 0.1, 0.01)
    estimator.reset()
    estimate = estimator.estimate
    assert (estimate.x, estimate.y, estimate.heading) == pytest.approx((0.0, 0.0, 0.0))
    assert estimate.zupt_count == 0


def test_zupt_refuses_an_implausible_stationary_reading():
    """A guard against the caller's stillness detection being wrong.

    A robot spinning on the spot has equal and opposite wheel speeds, so a
    naive "average speed is zero" test calls it stationary. If the filter
    accepts a ZUPT then, it charges the entire rotation rate to the bias --
    which is exactly the failure this guard exists to prevent.
    """
    estimator = PoseEstimator()
    spin_rate = 0.9  # rad/s, well above any real bias
    for _ in range(2000):
        estimator.predict(0.0, spin_rate, 0.01)
        estimator.zero_velocity_update(spin_rate, 0.01, moving=False)
    estimate = estimator.estimate
    assert estimate.zupt_count == 0
    assert abs(estimate.gyro_bias) < 0.01, (
        f"rotation leaked into the bias: {estimate.gyro_bias:.3f} rad/s"
    )


def test_a_plausible_reading_is_still_accepted():
    estimator = PoseEstimator()
    for _ in range(400):
        estimator.predict(0.0, 0.03, 0.01)
        estimator.zero_velocity_update(0.03, 0.01, moving=False)
    assert estimator.estimate.zupt_count > 0
    assert estimator.estimate.gyro_bias == pytest.approx(0.03, abs=0.006)
