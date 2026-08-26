"""Analytical derivatives of smooth differential-drive motion."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.physics import exact_arc_step


def central_jacobian(function, value, epsilon=1e-6):
    value = np.asarray(value, dtype=float)
    columns = []
    for index in range(value.size):
        delta = np.zeros_like(value)
        delta[index] = epsilon
        columns.append((function(value + delta) - function(value - delta)) / (2.0 * epsilon))
    return np.column_stack(columns)


@pytest.mark.parametrize("command", [[0.2, 0.2], [0.1, 0.3], [-0.2, 0.25]])
def test_step_matches_exact_arc_and_analytical_jacobians(command):
    state = np.array([1.2, -0.4, 0.7])
    control = np.asarray(command, dtype=float)
    dt, width = 0.08, 0.42

    actual, jacobian = zb.differentiable_drive_step(state, control, dt, width)
    velocity = float(control.mean())
    omega = float((control[1] - control[0]) / width)
    expected = exact_arc_step(*state, velocity, omega, dt)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    numerical_state = central_jacobian(
        lambda candidate: zb.differentiable_drive_step(candidate, control, dt, width)[0],
        state,
    )
    numerical_control = central_jacobian(
        lambda candidate: zb.differentiable_drive_step(state, candidate, dt, width)[0],
        control,
    )
    np.testing.assert_allclose(jacobian.state, numerical_state, rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(jacobian.control, numerical_control, rtol=2e-7, atol=2e-9)

    epsilon = 1e-6
    wider = zb.differentiable_drive_step(state, control, dt, width + epsilon)[0]
    narrower = zb.differentiable_drive_step(state, control, dt, width - epsilon)[0]
    numerical_width = (wider - narrower) / (2.0 * epsilon)
    np.testing.assert_allclose(jacobian.track_width, numerical_width, rtol=2e-7, atol=2e-9)


def test_rollout_propagates_terminal_command_sensitivity():
    initial = np.array([0.3, 0.5, -0.2])
    commands = np.array([[0.2, 0.25], [0.3, 0.1], [0.15, 0.22]])
    trajectory = zb.differentiable_rollout(initial, commands, dt=0.1, track_width=0.4)

    numerical = np.empty((3, len(commands), 2))
    epsilon = 1e-6
    for step in range(len(commands)):
        for side in range(2):
            delta = np.zeros_like(commands)
            delta[step, side] = epsilon
            plus = zb.differentiable_rollout(initial, commands + delta, 0.1, 0.4).states[-1]
            minus = zb.differentiable_rollout(initial, commands - delta, 0.1, 0.4).states[-1]
            numerical[:, step, side] = (plus - minus) / (2.0 * epsilon)

    assert trajectory.states.shape == (4, 3)
    np.testing.assert_allclose(
        trajectory.terminal_control_jacobian, numerical, rtol=3e-7, atol=2e-9
    )


def test_differentiable_step_validates_its_smooth_domain_inputs():
    with pytest.raises(ValueError, match="shape"):
        zb.differentiable_drive_step(np.zeros(2), np.zeros(2), 0.1, 0.4)
    with pytest.raises(ValueError, match="track_width"):
        zb.differentiable_drive_step(np.zeros(3), np.zeros(2), 0.1, 0.0)
    with pytest.raises(ValueError, match="commands"):
        zb.differentiable_rollout(np.zeros(3), np.zeros(3), 0.1, 0.4)
