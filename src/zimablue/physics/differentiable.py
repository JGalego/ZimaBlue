"""Analytical derivatives of the smooth differential-drive motion model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DifferentiableTrajectory",
    "DriveJacobians",
    "differentiable_drive_step",
    "differentiable_rollout",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DriveJacobians:
    """Local derivatives of the next pose."""

    state: FloatArray
    """Shape ``(3, 3)``: next pose with respect to current pose."""

    control: FloatArray
    """Shape ``(3, 2)``: next pose with respect to left/right wheel speed."""

    track_width: FloatArray
    """Shape ``(3,)``: next pose with respect to axle track width."""


@dataclass(frozen=True)
class DifferentiableTrajectory:
    """A smooth rollout and its terminal command sensitivity."""

    states: FloatArray
    """Shape ``(steps + 1, 3)`` including the initial pose."""

    jacobians: tuple[DriveJacobians, ...]
    terminal_control_jacobian: FloatArray
    """Shape ``(3, steps, 2)`` for terminal pose versus every command."""


def differentiable_drive_step(
    state: FloatArray,
    wheel_speeds: FloatArray,
    dt: float,
    track_width: float,
) -> tuple[FloatArray, DriveJacobians]:
    """Integrate constant wheel speeds exactly and return analytical Jacobians.

    State is ``[x, y, heading]`` and control is ground speed ``[left, right]``
    in metres per second. The sinc form is smooth at zero yaw rate and avoids a
    straight/turn branch in both the value and its derivatives.
    """
    pose = np.asarray(state, dtype=float)
    control = np.asarray(wheel_speeds, dtype=float)
    if pose.shape != (3,) or control.shape != (2,):
        raise ValueError("state must have shape (3,) and wheel_speeds shape (2,)")
    if not np.isfinite(pose).all() or not np.isfinite(control).all():
        raise ValueError("state and wheel speeds must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(track_width) or track_width <= 0.0:
        raise ValueError("track_width must be finite and positive")

    left, right = control
    velocity = 0.5 * (left + right)
    omega = (right - left) / track_width
    z = 0.5 * omega * dt
    sinc, sinc_derivative = _sinc_and_derivative(z)
    arc = dt * sinc
    arc_omega = 0.5 * dt**2 * sinc_derivative
    middle = pose[2] + z
    cosine, sine = np.cos(middle), np.sin(middle)

    dx = velocity * arc * cosine
    dy = velocity * arc * sine
    next_state = np.array([pose[0] + dx, pose[1] + dy, pose[2] + omega * dt])

    state_jacobian = np.array(
        [
            [1.0, 0.0, -dy],
            [0.0, 1.0, dx],
            [0.0, 0.0, 1.0],
        ]
    )
    derivative_velocity = np.array([arc * cosine, arc * sine, 0.0])
    derivative_omega = np.array(
        [
            velocity * (arc_omega * cosine - 0.5 * dt * arc * sine),
            velocity * (arc_omega * sine + 0.5 * dt * arc * cosine),
            dt,
        ]
    )
    twist_control = np.array(
        [
            [0.5, 0.5],
            [-1.0 / track_width, 1.0 / track_width],
        ]
    )
    motion_twist = np.column_stack([derivative_velocity, derivative_omega])
    control_jacobian = motion_twist @ twist_control
    omega_width = -omega / track_width
    width_jacobian = derivative_omega * omega_width

    return next_state, DriveJacobians(
        state=state_jacobian,
        control=control_jacobian,
        track_width=width_jacobian,
    )


def differentiable_rollout(
    initial_state: FloatArray,
    commands: FloatArray,
    dt: float,
    track_width: float,
) -> DifferentiableTrajectory:
    """Roll out commands and propagate terminal sensitivity by the chain rule."""
    controls = np.asarray(commands, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("commands must have shape (steps, 2)")
    state = np.asarray(initial_state, dtype=float)
    states = [state.copy()]
    local = []
    sensitivity = np.zeros((3, len(controls), 2), dtype=float)

    for index, command in enumerate(controls):
        state, jacobian = differentiable_drive_step(state, command, dt, track_width)
        sensitivity = np.einsum("ij,jkl->ikl", jacobian.state, sensitivity)
        sensitivity[:, index, :] += jacobian.control
        states.append(state)
        local.append(jacobian)

    return DifferentiableTrajectory(
        states=np.asarray(states),
        jacobians=tuple(local),
        terminal_control_jacobian=sensitivity,
    )


def _sinc_and_derivative(value: float) -> tuple[float, float]:
    magnitude = abs(value)
    if magnitude < 1e-4:
        square = value * value
        sinc = 1.0 - square / 6.0 + square * square / 120.0
        derivative = -value / 3.0 + value * square / 30.0
        return sinc, derivative
    return np.sin(value) / value, (value * np.cos(value) - np.sin(value)) / value**2
