"""The control loop, with the simulator taken out from under it.

``Simulation.step`` runs sense, decide, actuate, record, in that order.
Sensing and actuating are the backend's job, and on a robot they are the
driver's.  Deciding and recording are the same work either way, and this
module is what is left when the backend's half is pulled out and replaced
with a :class:`~zimablue.hardware.sources.ReadingSource` and a callable that
writes motors.

The consequence is the point of the whole exercise: a controller written and
tuned against the simulator runs here unmodified, because
:class:`~zimablue.controllers.base.ControlInput` was always the entire contract
and it never contained anything a real robot does not have.

::

    runtime = HardwareRuntime(
        controller=SystematicCoverage(),
        robot=zb.make_robot("tracked"),
        source=DeviceSource(channels=..., poll=read_my_bus),
        actuate=my_motor_driver,
        pool=my_pool,
    )
    run = runtime.run(minutes=20)
    run.save("runs/pool_tuesday.zbr")

What comes back replays in the ordinary viewer.  What it does *not* contain is
coverage or dirt removed: both are computed from the true pose against the true
dirt field, and a robot has neither.  See :meth:`HardwareRun.metrics`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zimablue._version import __version__
from zimablue.backends.base import Event, SimState
from zimablue.controllers.base import ControlInput, Controller
from zimablue.hardware.motors import MotorEffort, WheelSpeedLoop
from zimablue.hardware.safety import Watchdog
from zimablue.hardware.sources import ReadingSource
from zimablue.recording import Recorder, Recording, build_frame
from zimablue.robot import Cleaner, DriveCommand
from zimablue.sensors import Reading

__all__ = ["HardwareRun", "HardwareRuntime", "Survey", "Tick"]

Actuator = Callable[[Any], None]
Platform = Callable[[], Mapping[str, float]]


@dataclass(frozen=True)
class Survey:
    """What was measured before the run: the pool's shape, and the start pose.

    This is the deployable half of what a simulation reveals through
    ``truth``. Surveying a pool once and loading the result is how an offline
    plan gets onto a real machine, and it is a different thing from reading
    the dirt field -- a survey holds nothing a tape measure and a photograph
    cannot produce. Passing one to :class:`HardwareRuntime` lets
    :class:`~zimablue.planners.base.PathFollower` (on ``odometry``) run on
    hardware; the true pose stays as unavailable as it really is.
    """

    pool: Any
    """A :class:`~zimablue.pool.Pool` -- traced from a photo, sketched, or
    measured by hand."""

    start: tuple[float, float, float]
    """Where the robot was placed in the survey's frame: x, y, heading."""


@dataclass(frozen=True)
class Tick:
    """One pass of the loop, for a caller driving :meth:`HardwareRuntime.tick`."""

    time: float
    command: DriveCommand
    effort: MotorEffort | None
    readings: dict[str, Reading]
    tripped: list[str]
    decide_time: float
    loop_period: float


@dataclass
class HardwareRun:
    """What a real run produced."""

    recording: Recording | None
    ticks: int
    duration: float
    distance: float
    """Metres, integrated from the encoders. Odometry, so it runs long under
    slip -- the same bias a real cleaner reports about itself."""

    events: list[Event] = field(default_factory=list)
    watchdog_reasons: list[str] = field(default_factory=list)
    overruns: int = 0

    def save(self, path: str | Path) -> Path:
        if self.recording is None:
            raise RuntimeError("this run was not recorded; construct the runtime with record=True")
        return self.recording.save(path)

    def metrics(self) -> dict[str, float]:
        """The metrics a robot can actually produce about itself.

        Deliberately short, and deliberately missing the two the rest of this
        library is about. Coverage needs the true pose; dirt removed needs the
        true dirt field. A robot has neither, and computing either one from the
        pose estimate would produce a number that looks like the simulator's
        and means something else -- a controller with a drifting estimate would
        report near-perfect coverage of a pool it never crossed.

        To score cleaning on real hardware you need an external measurement:
        an overhead camera for the pose, and a before-and-after of the floor
        for the dirt.
        """
        return {
            "runtime": self.duration,
            "distance": self.distance,
            "ticks": float(self.ticks),
            "collisions": float(sum(1 for e in self.events if e.kind == "collision")),
            "overruns": float(self.overruns),
            "watchdog_trips": float(len(self.watchdog_reasons)),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"HardwareRun(duration={self.duration:.1f}s, ticks={self.ticks}, "
            f"distance={self.distance:.1f}m, trips={len(self.watchdog_reasons)})"
        )


class HardwareRuntime:
    """Runs one controller against one robot, with no simulator underneath.

    ``actuate`` receives a :class:`~zimablue.robot.DriveCommand` -- or, when a
    ``speed_loop`` is given, a :class:`~zimablue.hardware.motors.MotorEffort`
    with normalised per-side duty. Which one you want depends on whether your
    driver closes the speed loop itself.

    ``platform`` is the robot's own bus: state of charge, filter load, power
    draw, depth. It returns whatever it knows, keyed by the names in
    :attr:`PLATFORM_CHANNELS`; anything absent is recorded as NaN rather than
    guessed, so a replay of a robot without a filter sensor shows a gap
    instead of a flat line at zero.
    """

    PLATFORM_CHANNELS = ("battery", "filter_load", "power_w", "depth")

    def __init__(
        self,
        controller: Controller,
        robot: Cleaner,
        source: ReadingSource,
        actuate: Actuator,
        *,
        pool: Any = None,
        survey: Survey | None = None,
        platform: Platform | None = None,
        speed_loop: WheelSpeedLoop | None = None,
        watchdog: Watchdog | None = None,
        rate_hz: float = 50.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        record: bool = True,
        name: str = "hardware",
    ) -> None:
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self.controller = controller
        self.robot = robot
        self.source = source
        self.actuate = actuate
        self.survey = survey
        # A survey *is* a believed pool, so it doubles as the replay geometry
        # unless a different one is given explicitly.
        self.pool = pool if pool is not None else (survey.pool if survey else None)
        self.platform = platform
        self.speed_loop = speed_loop
        self.watchdog = watchdog if watchdog is not None else Watchdog()
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / self.rate_hz
        self.clock = clock
        self.sleep = sleep
        self.name = name

        self.controller.reset(robot)
        if self.speed_loop is not None:
            self.speed_loop.reset()

        self._origin = self.clock()
        self._last_tick: float | None = None
        self._ticks = 0
        self._distance = 0.0
        self._contacts = (False, False, False, False)
        self._was_touching = False
        self.events: list[Event] = []
        self.overruns = 0
        self._finished = False

        self.recorder = Recorder(self._manifest(), enabled=record)

    # ------------------------------------------------------------------
    def _manifest(self) -> dict[str, Any]:
        return {
            "zimablue_version": __version__,
            "seed": 0,
            "timestep": self.period,
            "backend": "hardware",
            # The single most important key in a recording written off a robot.
            # Everything downstream that says "pose" means the estimate here,
            # and a reader that does not check this will compare a real run's
            # coverage against a simulated one as though they were the same
            # measurement.
            "pose_source": "estimate",
            "ground_truth": False,
            "scenario": {
                "name": self.name,
                "pool": getattr(self.pool, "name", "unknown"),
                "robot": self.robot.name,
                "dirt": "unknown",
                "controller": getattr(self.controller, "name", "custom"),
            },
            "pool_config": self.pool.to_dict() if self.pool is not None else None,
            "robot_config": self.robot.to_dict(),
            "dirt_config": None,
        }

    # ------------------------------------------------------------------
    def tick(self) -> Tick:
        """Sense, decide, actuate, record. One pass, no dynamics."""
        now = self.clock() - self._origin
        loop_period = self.period if self._last_tick is None else now - self._last_tick
        dt = loop_period
        self._last_tick = now
        if loop_period > 1.5 * self.period:
            self.overruns += 1

        readings = self.source.read(now)
        platform = dict(self.platform() or {}) if self.platform is not None else {}

        started = self.clock()
        error: BaseException | None = None
        command = self.watchdog.safe_command()
        try:
            proposed = self.controller.step(self._control_input(now, dt, readings, platform))
            if not isinstance(proposed, DriveCommand):
                raise TypeError(
                    f"controller returned {type(proposed).__name__}, expected DriveCommand"
                )
            if not np.isfinite((proposed.left, proposed.right, proposed.pump)).all():
                raise ValueError("controller returned a non-finite drive command")
            command = proposed
        except Exception as exc:
            error = exc
        decide_time = self.clock() - started

        tripped = self.watchdog.check(
            now,
            readings,
            loop_period=loop_period if self._ticks else None,
            decide_time=decide_time,
            error=error,
        )
        if self.watchdog.tripped:
            command = self.watchdog.safe_command()
            for reason in tripped:
                self._emit(Event(time=now, kind="fault", detail={"reason": reason}))

        effort: MotorEffort | None = None
        if self.speed_loop is not None:
            effort = self.speed_loop(command, self._wheel_speeds(readings), dt)
            self.actuate(effort)
        else:
            self.actuate(command)

        state = self._state(now, dt, readings, platform)
        self._record(state, command, readings)
        self._ticks += 1

        if error is not None and not isinstance(error, Exception):  # pragma: no cover
            raise error
        return Tick(
            time=now,
            command=command,
            effort=effort,
            readings=readings,
            tripped=tripped,
            decide_time=decide_time,
            loop_period=loop_period,
        )

    def _control_input(
        self,
        now: float,
        dt: float,
        readings: dict[str, Reading],
        platform: Mapping[str, float],
    ) -> ControlInput:
        contacts = self._read_contacts(readings)
        return ControlInput(
            time=now,
            dt=dt,
            readings=readings,
            battery=float(platform.get("battery", 1.0)),
            filter_load=float(platform.get("filter_load", 0.0)),
            robot=self.robot,
            # `truth` stays None, always, and there is no option to change it.
            # An oracle cannot run on a robot, and offering the argument would
            # invite somebody to wire the pose estimate into it. A survey is
            # different: it holds what was measured, not what only a simulator
            # could reveal.
            truth=None,
            survey=self.survey,
            extras={
                "stuck": 0.0,
                "collided": 1.0 if any(contacts) else 0.0,
            },
        )

    def _wheel_speeds(self, readings: Mapping[str, Reading]) -> tuple[float, float]:
        encoder = readings.get("encoder")
        if encoder is None or encoder.values.size < 2:
            return (0.0, 0.0)
        return (float(encoder.values[0]), float(encoder.values[1]))

    def _read_contacts(self, readings: Mapping[str, Reading]) -> tuple[bool, ...]:
        contact = readings.get("contact")
        if contact is None or not contact.valid:
            return (False, False, False, False)
        flags = tuple(bool(v > 0.5) for v in contact.values[:4])
        return flags + (False,) * (4 - len(flags))

    def _state(
        self,
        now: float,
        dt: float,
        readings: Mapping[str, Reading],
        platform: Mapping[str, float],
    ) -> SimState:
        """A :class:`SimState` filled in from what the robot can observe.

        The pose fields carry the controller's estimate when it publishes one
        and NaN when it does not. NaN rather than zero: a replay drawing a
        robot parked at the origin for twenty minutes is a lie, and a gap is
        not.
        """
        left, right = self._wheel_speeds(readings)
        v, omega = self.robot.locomotion.to_body_velocity(left, right)
        gyro = readings.get("imu")
        if gyro is not None and gyro.valid and gyro.values.size >= 3:
            omega = float(gyro.values[2])
        self._distance += abs(v) * dt

        telemetry = self._telemetry()
        state = SimState(
            time=now,
            step=self._ticks,
            x=float(telemetry.get("est_x", np.nan)),
            y=float(telemetry.get("est_y", np.nan)),
            heading=float(telemetry.get("est_heading", np.nan)),
            v=v,
            omega=omega,
            wheel_left=left,
            wheel_right=right,
            slip_left=np.nan,
            slip_right=np.nan,
            depth=float(platform.get("depth", np.nan)),
            contacts=self._read_contacts(readings),  # type: ignore[arg-type]
            collided=any(self._read_contacts(readings)),
            power_w=float(platform.get("power_w", np.nan)),
            filter_load=float(platform.get("filter_load", np.nan)),
            distance=self._distance,
            dirt_collected=np.nan,
            dirt_removed=np.nan,
        )
        state._battery_fraction = float(platform.get("battery", np.nan))

        touching = bool(state.collided)
        if touching and not self._was_touching:
            self._emit(Event(time=now, kind="collision", detail={}))
        self._was_touching = touching
        return state

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        self.recorder.add_event(event)

    def _telemetry(self) -> dict[str, float]:
        publish = getattr(self.controller, "telemetry", None)
        if publish is None:
            return {}
        try:
            return {k: float(v) for k, v in publish().items()}
        except Exception:
            return {}

    def _record(
        self, state: SimState, command: DriveCommand, readings: Mapping[str, Reading]
    ) -> None:
        if not self.recorder.enabled:
            return
        self.recorder.add_frame(
            build_frame(state, command, readings, self.source.channels, self._telemetry())
        )

    # ------------------------------------------------------------------
    def run(
        self,
        minutes: float | None = None,
        *,
        seconds: float | None = None,
        max_ticks: int | None = None,
        until: Callable[[Tick], bool] | None = None,
        stop_on_trip: bool = True,
    ) -> HardwareRun:
        """Tick at ``rate_hz`` until something says to stop.

        Paces itself against the clock rather than sleeping a fixed period, so
        a slow tick is absorbed rather than compounded. A tick that overruns
        the period does not sleep at all, and is counted in
        :attr:`HardwareRun.overruns` -- silently falling behind is how a 50 Hz
        loop becomes a 30 Hz loop that nobody notices until the estimator
        starts drifting.
        """
        if minutes is None and seconds is None and max_ticks is None and until is None:
            raise ValueError(
                "run() needs a stopping condition: minutes, seconds, max_ticks or until"
            )
        duration = (
            float(seconds)
            if seconds is not None
            else (float(minutes) * 60.0 if minutes is not None else float("inf"))
        )
        if duration <= 0:
            raise ValueError("run duration must be positive")

        if hasattr(self.controller, "run_duration") and np.isfinite(duration):
            self.controller.run_duration = duration

        deadline = self.clock() + duration if np.isfinite(duration) else float("inf")
        while True:
            target = self.clock() + self.period
            tick = self.tick()
            if stop_on_trip and self.watchdog.tripped:
                break
            if max_ticks is not None and self._ticks >= max_ticks:
                break
            if until is not None and until(tick):
                break
            if self.clock() >= deadline:
                break
            remaining = target - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        return self.finish()

    def finish(self) -> HardwareRun:
        """Stop the motors and close the recording.

        The stop goes out first and unconditionally. Everything after it is
        bookkeeping, and bookkeeping that runs before the motors are off is
        a design that has never met a robot.
        """
        if self._finished:
            raise RuntimeError("this runtime has already been finished")
        self._finished = True
        stop = DriveCommand.stop()
        try:
            self.actuate(MotorEffort(0.0, 0.0, brush=False, pump=0.0) if self.speed_loop else stop)
        finally:
            self.source.close()

        run = HardwareRun(
            recording=None,
            ticks=self._ticks,
            duration=self._last_tick or 0.0,
            distance=self._distance,
            events=list(self.events),
            watchdog_reasons=list(self.watchdog.reasons),
            overruns=self.overruns,
        )
        recording = self.recorder.finish(metrics=run.metrics()) if self.recorder.enabled else None
        run.recording = recording
        return run
