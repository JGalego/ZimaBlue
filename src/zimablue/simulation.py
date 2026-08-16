"""The simulation loop -- ZimaBlue's front door.

::

    sim = zb.Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=42)
    result = sim.run(minutes=30)
    print(result.metrics.summary())
    result.save("runs/example.zbr")

One tick, in order: sense, decide, actuate and integrate, clean, record, score.
No wall-clock reads, no unseeded randomness, no order-dependent iteration --
which is what makes the recording an exact replay rather than an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zimablue._version import __version__
from zimablue.backends.base import Event, SimState
from zimablue.backends.fast2d import Fast2DBackend
from zimablue.controllers.base import ControlInput, Controller
from zimablue.controllers.baseline import BaselineCoverage
from zimablue.dirt import DirtSpec, make_dirt
from zimablue.metrics import Metrics, SpatialMetrics, compute_metrics
from zimablue.pool import DEFAULT_CELL, Pool, make_pool
from zimablue.recording import Recorder, Recording
from zimablue.rng import RngTree
from zimablue.robot import Cleaner, DriveCommand, make_robot
from zimablue.world import World

__all__ = ["RunResult", "Simulation"]


@dataclass
class RunResult:
    """Everything a finished run produced."""

    metrics: Metrics
    spatial: SpatialMetrics
    recording: Recording | None
    world: World
    state: SimState
    events: list[Event] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        """Write the recording to a ``.zbr`` file."""
        if self.recording is None:
            raise RuntimeError(
                "this run was not recorded; construct it with "
                "Simulation(..., record=True) to be able to save it"
            )
        return self.recording.save(path)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RunResult(coverage={self.metrics.coverage:.1%}, "
            f"dirt_removed={self.metrics.dirt_removed_fraction:.1%}, "
            f"runtime={self.metrics.runtime / 60:.1f} min)"
        )


class Simulation:
    """Runs one robot, in one pool, under one controller."""

    def __init__(
        self,
        pool: Pool | str = "rectangular",
        robot: Cleaner | str = "tracked",
        *,
        dirt: DirtSpec | str = "light_sediment",
        controller: Controller | str = "baseline_coverage",
        seed: int = 0,
        timestep: float = 0.02,
        backend: Any = "fast2d",
        cell: float = DEFAULT_CELL,
        record: bool = True,
        dirt_keyframe_interval: float = 10.0,
        expose_truth: bool = False,
        coverage_target: float | None = None,
        dirt_target: float | None = None,
        stop_on_empty_battery: bool = True,
        scenario_name: str = "adhoc",
    ) -> None:
        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        self.timestep = timestep
        self.seed = seed
        self.rng = RngTree(seed)
        self.expose_truth = expose_truth
        self.coverage_target = coverage_target
        self.dirt_target = dirt_target
        self.stop_on_empty_battery = stop_on_empty_battery
        self.scenario_name = scenario_name

        self.pool = make_pool(pool) if isinstance(pool, str) else pool
        self.robot = make_robot(robot) if isinstance(robot, str) else robot
        self.dirt_spec = make_dirt(dirt) if isinstance(dirt, str) else dirt
        self.controller: Controller = (
            _make_controller(controller) if isinstance(controller, str) else controller
        )

        self.world = World.build(self.pool, self.dirt_spec, self.rng.stream("dirt"), cell)
        self.backend = _make_backend(backend)

        self._initial_dirt = self.world.dirt.field.total_grid().copy()
        self.state = self.backend.reset(self.world, self.robot, self.rng)
        self.controller.reset(self.robot)
        self.events: list[Event] = []
        self._last_command = DriveCommand.stop()
        self._finished = False
        self._termination = "running"

        self.recorder = Recorder(
            self._manifest(),
            dirt_keyframe_interval=dirt_keyframe_interval,
            enabled=record,
        )
        self.recorder.maybe_keyframe(0.0, self.world.dirt, force=True)

    # ------------------------------------------------------------------
    def _manifest(self) -> dict[str, Any]:
        return {
            "zimablue_version": __version__,
            "seed": self.seed,
            "timestep": self.timestep,
            "cell": self.world.cell,
            "backend": getattr(self.backend, "name", "unknown"),
            "scenario": {
                "name": self.scenario_name,
                "pool": self.pool.name,
                "robot": self.robot.name,
                "dirt": self.dirt_spec.name,
                "controller": getattr(self.controller, "name", "custom"),
            },
            # Embedded in full so the recording stays replayable even if the
            # presets it came from later change.
            "pool_config": self.pool.to_dict(),
            "robot_config": self.robot.to_dict(),
            "dirt_config": self.dirt_spec.to_dict(),
        }

    # ------------------------------------------------------------------
    def step(self) -> SimState:
        """Advance one timestep."""
        dt = self.timestep

        # 1. Sense.
        observations = self.backend.sense(self.state)

        # 2. Decide.
        control_input = ControlInput(
            time=self.state.time,
            dt=dt,
            readings=observations,
            battery=self.state.battery_fraction,
            filter_load=self.state.filter_load,
            robot=self.robot,
            truth=self._truth_view() if self.expose_truth else None,
            extras={
                "stuck": 1.0 if self.state.stuck else 0.0,
                "collided": 1.0 if self.state.collided else 0.0,
            },
        )
        command = self.controller.step(control_input)

        # 3-5. Actuate, integrate, clean.
        result = self.backend.step(self.state, command, dt)
        self.state = result.state
        self.events.extend(result.events)

        # 6. Record.
        self._record(self.state, command, observations)
        for event in result.events:
            self.recorder.add_event(event)
        self.recorder.maybe_keyframe(self.state.time, self.world.dirt)

        self._last_command = command
        return self.state

    def _truth_view(self) -> SimState:
        """Ground truth, with the pool attached, for oracle controllers only."""
        state = self.state
        state.pool = self.pool
        return state

    def _record(self, state: SimState, command: DriveCommand, observations: dict[str, Any]) -> None:
        if not self.recorder.enabled:
            return
        contacts = sum(1 << i for i, flag in enumerate(state.contacts) if flag)
        frame: dict[str, float] = {
            "time": state.time,
            "step": state.step,
            "x": state.x,
            "y": state.y,
            "heading": state.heading,
            "v": state.v,
            "omega": state.omega,
            "wheel_left": state.wheel_left,
            "wheel_right": state.wheel_right,
            "slip_left": state.slip_left,
            "slip_right": state.slip_right,
            "depth": state.depth,
            "battery": state.battery_fraction,
            "power": state.power_w,
            "filter_load": state.filter_load,
            "distance": state.distance,
            "dirt_collected": state.dirt_collected,
            "contacts": contacts,
            "collided": 1 if state.collided else 0,
            "stuck": 1 if state.stuck else 0,
            "cmd_left": command.left,
            "cmd_right": command.right,
            "cmd_brush": 1 if command.brush else 0,
            "cmd_pump": command.pump,
        }
        # A controller may publish its own channels -- an estimated pose, a
        # planner phase. Recording them next to ground truth is what lets
        # replay show estimation error rather than merely assert it.
        telemetry = getattr(self.controller, "telemetry", None)
        if telemetry is not None:
            for key, value in telemetry().items():
                frame[f"ctl.{key}"] = float(value)

        for name, reading in observations.items():
            for channel, value in zip(
                self.robot.sensors[name].channels, reading.values, strict=False
            ):
                frame[f"{name}.{channel}"] = float(value)
            frame[f"{name}.valid"] = 1.0 if reading.valid else 0.0
        self.recorder.add_frame(frame)

    # ------------------------------------------------------------------
    def run(
        self,
        minutes: float | None = None,
        *,
        seconds: float | None = None,
        max_steps: int | None = None,
        progress: Any = None,
    ) -> RunResult:
        """Run to a termination condition and return the result.

        Stops on whichever comes first: the duration, the battery cutoff, a
        coverage or dirt target, or ``max_steps``.
        """
        if minutes is None and seconds is None:
            seconds = 1800.0
        duration = float(seconds if seconds is not None else (minutes or 0) * 60.0)
        if duration <= 0:
            raise ValueError("run duration must be positive")

        # Let a duration-aware controller pace itself.
        if hasattr(self.controller, "run_duration"):
            self.controller.run_duration = duration

        steps = int(np.ceil(duration / self.timestep))
        if max_steps is not None:
            steps = min(steps, max_steps)

        self._termination = "duration"
        for i in range(steps):
            self.step()
            if progress is not None and i % 250 == 0:
                progress(self.state.time, duration)
            reason = self._check_termination()
            if reason is not None:
                self._termination = reason
                break

        return self.finish()

    def _check_termination(self) -> str | None:
        state = self.state
        if self.stop_on_empty_battery and state.battery_fraction <= self.robot.power.battery.cutoff:
            return "battery_empty"
        if self.dirt_target is not None and self.world.dirt.removed_fraction >= self.dirt_target:
            return "target_reached"
        if self.coverage_target is not None:
            visits = self.backend.visit_grid
            navigable = self.pool.navigable_mask(self.world.cell)
            total = int(navigable.sum())
            if total and float((navigable & (visits > 0)).sum()) / total >= self.coverage_target:
                return "target_reached"
        return None

    def finish(self) -> RunResult:
        """Score the run and close the recording."""
        if self._finished:
            raise RuntimeError("this simulation has already been finished")
        self._finished = True
        self.recorder.maybe_keyframe(self.state.time, self.world.dirt, force=True)

        metrics, spatial = compute_metrics(
            self.world,
            self.state,
            self.events,
            self.backend.visit_grid,
            self.backend.wall_visits,
            self._initial_dirt,
            termination=self._termination,
        )
        recording = None
        if self.recorder.enabled:
            recording = self.recorder.finish(
                metrics=metrics.to_dict(),
                spatial={
                    "visits": spatial.visits,
                    "remaining_dirt": spatial.remaining_dirt.astype(np.float32),
                    "initial_dirt": spatial.initial_dirt.astype(np.float32),
                    "wall_visits": spatial.wall_visits,
                    "navigable": spatial.navigable,
                },
            )
        self.backend.close()
        return RunResult(
            metrics=metrics,
            spatial=spatial,
            recording=recording,
            world=self.world,
            state=self.state,
            events=list(self.events),
        )


def _make_backend(backend: Any) -> Any:
    if isinstance(backend, str):
        if backend == "fast2d":
            return Fast2DBackend()
        from zimablue.backends.base import BACKENDS

        return BACKENDS.create(backend)
    return backend


def _make_controller(name: str) -> Controller:
    from zimablue.controllers.base import CONTROLLERS

    if name == "baseline_coverage":
        return BaselineCoverage()
    return CONTROLLERS.create(name)
