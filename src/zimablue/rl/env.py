"""A Gymnasium environment over :class:`~zimablue.simulation.Simulation`.

Inverting the loop
------------------

A :class:`~zimablue.controllers.base.Controller` is pulled: the simulation
calls it once per tick and it answers. An RL agent wants the opposite, so the
env installs a controller that answers with whatever the agent last chose, and
drives ``Simulation.step()`` itself. Nothing about the simulation changes, and
a policy trained here can be wrapped back into a Controller and run through
the CLI, scenarios and batches like any other.

Deciding slower than the physics
--------------------------------

The simulation integrates at 50 Hz. Asking a policy for a fresh command 50
times a second means 90,000 decisions in half an hour, almost all of them
identical to the one before, and a credit assignment problem stretched over a
horizon no algorithm will thank you for. Real cleaner firmware does not do
that either. ``control_hz`` decimates: one action is held for however many
physics ticks it covers, which at the default 5 Hz makes the episode 10x
shorter without changing the dynamics at all.

What to reward
--------------

This is the interesting parameter, not a detail to be defaulted past.
``reward="dirt"`` pays grams collected; ``reward="coverage"`` pays newly
visited floor. They do not agree -- the whole project exists because they do
not agree -- and a policy trained on coverage will learn the oracle's failure
mode of driving beautifully over dirt it never picks up. Dirt is the default
because it is the one that means anything.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from zimablue.controllers.base import ControlInput
from zimablue.robot import Cleaner, DriveCommand
from zimablue.simulation import Simulation

__all__ = ["REWARDS", "PoolCleaningEnv", "channel_names", "observe"]

REWARDS = ("dirt", "coverage")

FloatArray = NDArray[np.float32]


def channel_names(robot: Cleaner) -> list[str]:
    """Every observed quantity, in a fixed order.

    Sorted by sensor name rather than by whatever order the suite iterates in,
    so adding a sensor cannot silently permute a trained policy's input.
    """
    names = ["battery", "filter_load", "elapsed"]
    for sensor in sorted(robot.sensors):
        names.extend(f"{sensor}.{channel}" for channel in robot.sensors[sensor].channels)
        names.append(f"{sensor}.fresh")
    return names


def observe(control_input: ControlInput, *, elapsed: float) -> FloatArray:
    """Build the observation vector from one tick's :class:`ControlInput`.

    Shared by the env and by :class:`~zimablue.rl.policy.PolicyController`, so
    a policy sees the same numbers in the same order whether it is being
    trained or being run through the CLI. Two copies of this drifting apart is
    the sort of bug that shows up as "it worked in training".
    """
    robot = control_input.robot
    capacity = max(robot.cleaning.filter.capacity, 1e-6)
    values = [
        control_input.battery,
        min(control_input.filter_load / capacity, 1.0),
        min(max(elapsed, 0.0), 1.0),
    ]
    for sensor in sorted(robot.sensors):
        channels = robot.sensors[sensor].channels
        reading = control_input.readings.get(sensor)
        if reading is None:
            values.extend([0.0] * (len(channels) + 1))
            continue
        values.extend(float(v) for v in reading.values[: len(channels)])
        values.append(1.0 if reading.fresh else 0.0)
    return np.asarray(values, dtype=np.float32)


class _AgentController:
    """The controller that does as it is told.

    Also the env's window into the tick: the simulation hands a
    :class:`ControlInput` here and nowhere else, so this is where the
    observation is captured from.
    """

    name = "rl_agent"

    def __init__(self) -> None:
        self.command = DriveCommand.stop()
        self.last: ControlInput | None = None

    def reset(self, robot: Cleaner) -> None:
        self.command = DriveCommand.stop()
        self.last = None

    def step(self, control_input: ControlInput) -> DriveCommand:
        self.last = control_input
        return self.command


class PoolCleaningEnv(gym.Env[FloatArray, FloatArray]):
    """Clean a pool, one wheel command at a time.

    ::

        env = PoolCleaningEnv(pool="kidney", dirt="autumn", minutes=10)
        obs, info = env.reset(seed=0)
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        print(info["dirt_removed"], info["coverage"])

    Actions are the two track speeds as fractions of the motor limit, in
    ``[-1, 1]``. The brush and pump stay on: switching the brush off is a way
    to score zero that an agent would find in ten minutes and learn nothing
    from.

    Observations are what the controller sees -- every sensor channel, its
    freshness flag, the battery, the filter load and how much of the episode
    is gone. No pose, no map, no dirt field. A policy that needs to know where
    it is has to work it out, which is the same problem the shipped
    ``systematic`` controller solves with an EKF; hand that estimate in
    through ``extra_observations`` if you would rather learn the planner
    alone.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pool: Any = "kidney",
        robot: Any = "tracked",
        *,
        dirt: Any = "light_sediment",
        minutes: float = 10.0,
        control_hz: float = 5.0,
        reward: str = "dirt",
        seed: int = 0,
        timestep: float = 0.02,
        record: bool = False,
        **simulation_kwargs: Any,
    ) -> None:
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        if control_hz <= 0 or control_hz > 1.0 / timestep:
            raise ValueError(
                f"control_hz must be positive and no faster than the {1 / timestep:g} Hz "
                f"physics, got {control_hz}"
            )

        self.pool = pool
        self.robot = robot
        self.dirt = dirt
        self.minutes = float(minutes)
        self.reward_kind = reward
        self.timestep = float(timestep)
        self.record = record
        self.simulation_kwargs = simulation_kwargs
        self.base_seed = int(seed)

        self.repeat = max(round(1.0 / (control_hz * timestep)), 1)
        """Physics ticks per agent decision."""

        self.control_hz = 1.0 / (self.repeat * timestep)
        """What the decimation actually came out at, after rounding."""

        self.max_steps = max(round(self.minutes * 60.0 / (self.repeat * timestep)), 1)

        self.controller = _AgentController()
        self.sim: Simulation | None = None
        self.elapsed = 0
        self._collected = 0.0
        self._visited = 0
        self._saved = False

        # Build one throwaway simulation to learn the observation's shape. The
        # sensor suite is fixed by the robot, so this is a one-off cost and it
        # beats making the caller declare the layout twice.
        probe = self._build(self.base_seed)
        self.channels = channel_names(probe.robot)
        probe.backend.close()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        low, high = self._bounds()
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    def _build(self, seed: int) -> Simulation:
        return Simulation(
            pool=self.pool,
            robot=self.robot,
            dirt=self.dirt,
            controller=self.controller,
            seed=seed,
            timestep=self.timestep,
            record=self.record,
            scenario_name="rl",
            **self.simulation_kwargs,
        )

    def _bounds(self) -> tuple[FloatArray, FloatArray]:
        """Range per observation channel, taken from each sensor's saturation.

        Real limits beat a normalisation invented here: a sonar that saturates
        at 6 m says so, and an algorithm that wants everything in [-1, 1] can
        wrap the env in ``gymnasium.wrappers.RescaleObservation`` knowing the
        numbers are honest. Channels with no declared limit stay unbounded
        rather than being given a made-up one.
        """
        probe = self._build(self.base_seed)
        low = [0.0, 0.0, 0.0]
        high = [1.0, 1.0, 1.0]
        for sensor in sorted(probe.robot.sensors):
            config = probe.robot.sensors[sensor].config
            width = len(probe.robot.sensors[sensor].channels)
            low.extend([config.min_value] * width + [0.0])
            high.extend([config.max_value] * width + [1.0])
        probe.backend.close()
        return (
            np.asarray(low, dtype=np.float32),
            np.asarray(high, dtype=np.float32),
        )

    def _observe(self) -> FloatArray:
        control = self.controller.last
        if control is None:  # pragma: no cover - reset() always ticks first
            return np.zeros(len(self.channels), dtype=np.float32)
        return observe(control, elapsed=self.elapsed / self.max_steps)

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.sim is not None:
            self.sim.backend.close()
        self.sim = self._build(self.base_seed if seed is None else int(seed))
        self.elapsed = 0
        self._collected = 0.0
        self._saved = False
        self._visited = self._visited_cells()
        # One tick so the sensors have produced something to look at. The
        # command is a stop, so nothing has happened yet.
        self.sim.step()
        return self._observe(), self._info()

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("call reset() before step()")

        limit = self.sim.robot.locomotion.max_speed
        left, right = np.clip(np.asarray(action, dtype=float).ravel()[:2], -1.0, 1.0)
        self.controller.command = DriveCommand(left=left * limit, right=right * limit)

        for _ in range(self.repeat):
            self.sim.step()
        self.elapsed += 1

        reward = self._reward()
        terminated = self.sim.termination_reason() is not None
        truncated = self.elapsed >= self.max_steps
        return self._observe(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    def _visited_cells(self) -> int:
        assert self.sim is not None
        navigable = self.sim.pool.navigable_mask(self.sim.world.cell)
        return int((navigable & (self.sim.backend.visit_grid > 0)).sum())

    def _reward(self) -> float:
        """Paid per decision, on what changed during it."""
        assert self.sim is not None
        if self.reward_kind == "dirt":
            collected = float(self.sim.state.dirt_collected)
            gained = collected - self._collected
            self._collected = collected
            return gained
        # Baselined at reset, so the swath the robot happens to be dropped
        # onto is not paid for. It did not clean that; it was put there.
        visited = self._visited_cells()
        gained = visited - self._visited
        self._visited = visited
        # In cells of floor, scaled to m2 so the number does not depend on the
        # grid resolution the pool happens to use.
        return float(gained) * self.sim.world.cell**2

    def _info(self) -> dict[str, Any]:
        assert self.sim is not None
        navigable = self.sim.pool.navigable_mask(self.sim.world.cell)
        total = max(int(navigable.sum()), 1)
        return {
            "time": self.sim.state.time,
            "coverage": self._visited_cells() / total,
            "dirt_removed": self.sim.world.dirt.removed_fraction,
            "dirt_collected": self.sim.state.dirt_collected,
            "distance": self.sim.state.distance,
            "battery": self.sim.state.battery_fraction,
        }

    def save(self, path: str) -> Any:
        """Write the episode to a ``.zbr``, to watch it in the replay viewer.

        Needs ``record=True``. Worth the memory: a policy's coverage number
        tells you it is bad, and thirty seconds of the replay tells you why.
        """
        if self.sim is None:
            raise RuntimeError("nothing to save -- call reset() first")
        if not self.record:
            raise RuntimeError("construct the env with record=True to save episodes")
        if self._saved:
            raise RuntimeError("this episode has already been saved; reset() for another")
        self._saved = True
        return self.sim.finish().save(path)

    def close(self) -> None:
        # finish() already closed the backend, and closing twice is not
        # something a backend has to tolerate.
        if self.sim is not None and not self._saved:
            self.sim.backend.close()
        self.sim = None
