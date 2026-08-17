"""Run a trained policy as an ordinary controller.

Training in a Gymnasium env and then only ever evaluating in that same env is
how a policy comes to look better than it is. Wrapping it back into the
:class:`~zimablue.controllers.base.Controller` interface puts it on the same
footing as every other controller here: the same metrics, the same recordings,
the same ``zimablue batch`` across seeds, the same replay to watch it fail::

    from stable_baselines3 import PPO
    from zimablue.rl import PolicyController

    model = PPO.load("cleaner")
    controller = PolicyController(lambda obs: model.predict(obs, deterministic=True)[0])

    result = zb.Simulation(pool="kidney", controller=controller, seed=7).run(minutes=30)
    print(result.metrics.summary())

Any callable taking an observation and returning two numbers in ``[-1, 1]``
works -- there is no dependency on a training framework, and a hand-written
function is a perfectly good way to test the wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.controllers.base import ControlInput
from zimablue.rl.env import observe
from zimablue.robot import Cleaner, DriveCommand

__all__ = ["Policy", "PolicyController"]

Policy = Callable[[NDArray[np.float32]], "NDArray[np.float32] | tuple[float, float]"]


class PolicyController:
    """A :class:`~zimablue.controllers.base.Controller` that asks a policy.

    ``control_hz`` must match what the policy was trained at, and
    ``extra_observations`` must be the same kind of object the env was given.
    Both are quiet when wrong: the policy still runs, it is just being fed
    something other than what it learned on. The simulation steps at 50 Hz
    whatever happens, and between decisions the last command is held, exactly
    as the env's frame skipping does.
    """

    name = "rl_policy"

    def __init__(
        self, policy: Policy, *, control_hz: float = 5.0, extra_observations: Any = None
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.policy = policy
        self.control_hz = float(control_hz)
        self.extra_observations = extra_observations
        self.run_duration = 1800.0
        """Set by :meth:`~zimablue.simulation.Simulation.run` so the elapsed
        channel means the same thing it did in training."""

        self._command = DriveCommand.stop()
        self._next_decision = 0.0
        self._limit = 0.32

    def reset(self, robot: Cleaner) -> None:
        self._command = DriveCommand.stop()
        self._next_decision = 0.0
        self._limit = robot.locomotion.max_speed
        if self.extra_observations is not None:
            self.extra_observations.reset(robot)

    def step(self, control_input: ControlInput) -> DriveCommand:
        observation = observe(
            control_input, elapsed=control_input.time / max(self.run_duration, 1e-6)
        )
        if self.extra_observations is not None:
            # Every tick, not every decision: an EKF integrating at 5 Hz when
            # it was trained at 50 is a different filter.
            extra = np.asarray(self.extra_observations(control_input), dtype=np.float32)
            observation = np.concatenate([observation, extra])

        if control_input.time >= self._next_decision:
            action = np.asarray(self.policy(observation), dtype=float).ravel()
            left, right = np.clip(action[:2], -1.0, 1.0)
            self._command = DriveCommand(left=left * self._limit, right=right * self._limit)
            self._next_decision = control_input.time + 1.0 / self.control_hz
        return self._command
