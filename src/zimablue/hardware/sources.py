"""Where a control loop gets its readings.

In simulation a controller is handed a coherent snapshot: every sensor sampled
at the same instant, from the same integrator, with no possibility of a driver
timing out.  On a robot none of that holds.  Readings arrive on their own
schedules, some late, some not at all, and the loop has to run anyway.

:class:`Reading` already has the right shape for that world -- it carries the
time the sample became available, ``valid`` for "this one was dropped, the
values are the last good ones" and ``fresh`` for "this is a held sample, not a
new measurement".  What was missing is anything that *produces* readings from
something other than a simulated sensor.  That is what this module is.

Two sources ship:

:class:`DeviceSource`
    Wraps a callable that reads real hardware.  It does the bookkeeping the
    simulator's sensor pipeline was doing for free -- timestamping, holding the
    last good value, marking staleness -- so the driver you write can be a
    function that returns raw numbers or ``None``.

:class:`RecordedSource`
    Replays the sensor columns of a ``.zbr`` back out as readings, optionally
    with timing jitter and dropouts on top.  A recorded run becomes a test
    fixture for the runtime, which is the cheapest way to find out whether a
    controller survives conditions the simulator never produces.

:class:`TrajectorySource`
    Drives the sensor models from a trajectory a real robot really drove, so
    the estimator can be scored against motion this package did not generate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from zimablue.recording import Recording
from zimablue.sensors import Reading

__all__ = ["DeviceSource", "ReadingSource", "RecordedSource", "TrajectorySource"]

RawSample = "Sequence[float] | float | None"


@runtime_checkable
class ReadingSource(Protocol):
    """Anything a :class:`~zimablue.hardware.runtime.HardwareRuntime` can read."""

    channels: Mapping[str, tuple[str, ...]]
    """Sensor name to channel names, in the order the values arrive.

    The runtime needs it to name the recorded columns, and it is the one place
    a wiring mistake shows up as a shape error rather than as a controller that
    quietly steers on the wrong number.
    """

    def read(self, now: float) -> dict[str, Reading]:
        """Latest reading per sensor. Never raises for a missing sample."""
        ...

    def close(self) -> None: ...


class DeviceSource:
    """Readings from a real device, via one polling function.

    ``poll`` is called once per control tick and returns a mapping of sensor
    name to values -- a sequence, a scalar for a single-channel sensor, or
    ``None`` for "nothing new".  Anything it leaves out is treated as ``None``,
    so a driver that only has fresh encoder data every other tick can just not
    mention the others::

        def poll():
            return {"encoder": (left_mps, right_mps), "imu": (ax, ay, gz)}

        source = DeviceSource(
            channels={"encoder": ("left", "right"), "imu": ("ax", "ay", "gz")},
            poll=poll,
        )

    What this adds on top of ``poll``:

    * **Timestamps.** Readings are stamped with the loop's clock, so the
      controller's ``dt`` bookkeeping works the same as it does in simulation.
    * **Hold-last-good.** A sensor with no new sample keeps its previous values
      and reports ``fresh=False``. A sensor that has never reported reads as
      zeros with ``valid=False``, which is the honest answer and one a
      controller can test for.
    * **Staleness.** :attr:`ages` gives seconds since each sensor last produced
      a fresh sample, which is what the watchdog acts on.

    It deliberately does *not* filter, scale or unit-convert. Those belong in
    the driver, where the datasheet is.
    """

    def __init__(
        self,
        channels: Mapping[str, Sequence[str]],
        poll: Callable[[], Mapping[str, Any]],
        *,
        stale_after: float = 1.0,
    ) -> None:
        if not channels:
            raise ValueError("a source with no channels cannot drive a controller")
        self.channels = {name: tuple(chans) for name, chans in channels.items()}
        self.poll = poll
        self.stale_after = float(stale_after)

        self._values = {
            name: np.zeros(len(chans), dtype=float) for name, chans in self.channels.items()
        }
        self._last_fresh: dict[str, float] = dict.fromkeys(self.channels, -np.inf)
        self._seen: dict[str, bool] = dict.fromkeys(self.channels, False)
        self._now = 0.0

    def read(self, now: float) -> dict[str, Reading]:
        self._now = now
        raw = self.poll() or {}
        unknown = set(raw) - set(self.channels)
        if unknown:
            raise KeyError(
                f"poll() returned sensors that are not declared in channels: {sorted(unknown)}; "
                f"declared: {sorted(self.channels)}"
            )

        readings: dict[str, Reading] = {}
        for name, chans in self.channels.items():
            sample = raw.get(name)
            fresh = sample is not None
            if fresh:
                values = np.atleast_1d(np.asarray(sample, dtype=float)).ravel()
                if values.size != len(chans):
                    raise ValueError(
                        f"sensor {name!r} declares {len(chans)} channels {chans} "
                        f"but poll() returned {values.size} values"
                    )
                self._values[name] = values
                self._last_fresh[name] = now
                self._seen[name] = True

            # `valid` is about whether the values mean anything, not about
            # whether they are new: a held sample from 20 ms ago is perfectly
            # usable, one from four seconds ago is not, and a sensor that has
            # never spoken has nothing to hold.
            age = now - self._last_fresh[name]
            readings[name] = Reading(
                name=name,
                time=now,
                values=self._values[name].copy(),
                valid=self._seen[name] and age <= self.stale_after,
                fresh=fresh,
            )
        return readings

    @property
    def ages(self) -> dict[str, float]:
        """Seconds since each sensor last produced a fresh sample.

        ``inf`` for a sensor that has never reported -- distinguishable from a
        merely late one, and a different fault.
        """
        return {name: self._now - t for name, t in self._last_fresh.items()}

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


class RecordedSource:
    """Readings replayed out of a ``.zbr``.

    The inverse of what the recorder does: a run's ``encoder.left``,
    ``imu.gz``, ``sonar.beam_0`` columns go back out as :class:`Reading`
    objects, so a controller can be driven from a recorded run without a
    simulator underneath it.

    ``jitter`` and ``dropout`` inject the two things a simulator never produces
    and hardware always does.  They are the point: a controller that only ever
    sees a metronome is a controller whose timing assumptions have never been
    tested.  Both draw from a seeded generator, so a failure reproduces.

    Note what this is *not*. The readings are the ones the simulator wrote, so
    the noise is still the simulator's. This tests the plumbing and the
    controller's timing, not the sensor model.
    """

    def __init__(
        self,
        recording: Recording,
        *,
        jitter: float = 0.0,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.recording = recording
        self.jitter = float(jitter)
        self.dropout = float(dropout)
        self._rng = np.random.default_rng(seed)

        self.channels = _channels_of(recording)
        if not self.channels:
            raise ValueError(
                "this recording has no sensor columns to replay; it was written "
                "with recording disabled, or by a much older version"
            )
        self._times = np.asarray(recording.frames["time"], dtype=float)
        self._values = {
            name: np.column_stack(
                [np.asarray(recording.frames[f"{name}.{ch}"], dtype=float) for ch in chans]
            )
            for name, chans in self.channels.items()
        }
        self._valid = {
            name: np.asarray(recording.frames.get(f"{name}.valid", np.ones_like(self._times)))
            for name in self.channels
        }
        self._held: dict[str, Reading] = {}

    @property
    def duration(self) -> float:
        return float(self._times[-1]) if self._times.size else 0.0

    def read(self, now: float) -> dict[str, Reading]:
        sample_time = now + (self._rng.normal(0.0, self.jitter) if self.jitter else 0.0)
        index = int(
            np.clip(np.searchsorted(self._times, sample_time, "right") - 1, 0, len(self._times) - 1)
        )

        readings: dict[str, Reading] = {}
        for name in self.channels:
            dropped = self.dropout > 0 and bool(self._rng.random() < self.dropout)
            previous = self._held.get(name)
            if dropped and previous is not None:
                readings[name] = Reading(
                    name=name, time=now, values=previous.values, valid=False, fresh=False
                )
                continue
            values = self._values[name][index].copy()
            # NaN in a recording means "this channel had not reported yet" --
            # the recorder back-fills a late-arriving sensor with it. A sensor
            # that samples at 5 Hz therefore has NaN in its first few frames,
            # and handing those to a controller as though they were
            # measurements is how a replayed run crashes on tick one.
            reading = Reading(
                name=name,
                time=now,
                values=values,
                valid=bool(self._valid[name][index]) and bool(np.isfinite(values).all()),
                fresh=True,
            )
            self._held[name] = reading
            readings[name] = reading
        return readings

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def _channels_of(recording: Recording) -> dict[str, tuple[str, ...]]:
    """Recover the sensor layout from a recording's manifest, or its columns.

    The manifest carries the robot configuration, so prefer that -- it is the
    authoritative order. Fall back to parsing column names for a recording
    written without it, taking care not to mistake the ``ctl.`` telemetry
    namespace or a ``.valid`` flag for a sensor channel.
    """
    robot = recording.manifest.get("robot_config") or {}
    specs = robot.get("sensors") or []
    layout = {
        spec["name"]: tuple(spec["channels"])
        for spec in specs
        if spec.get("name") and spec.get("channels")
    }
    if layout:
        return {
            name: chans for name, chans in layout.items() if f"{name}.valid" in recording.frames
        }

    found: dict[str, list[str]] = {}
    for column in recording.frames:
        name, _, channel = column.partition(".")
        if not channel or name == "ctl" or channel == "valid":
            continue
        found.setdefault(name, []).append(channel)
    return {name: tuple(chans) for name, chans in found.items()}


class TrajectorySource:
    """Readings synthesised from a trajectory somebody else's robot drove.

    This is the one source that puts a number in front of a claim.  Every other
    figure in this library -- the estimator's drift, the coverage a controller
    achieves, how well the lane plan holds up -- is measured against motion the
    package generated itself, using a noise model the package invented.  Feed
    the same pipeline a real robot's real trajectory and at least half of that
    becomes checkable: the accelerations, the stop-start, the turning
    behaviour, the gaps where the tracker lost the robot, are all things that
    happened.

    The other half does not become checkable, and it matters which half.  The
    readings still come out of :class:`~zimablue.sensors.base.Sensor`, so the
    *noise* is still ours -- a guessed bias walk on a guessed rate.  Closing
    that gap needs a log from a real IMU, not a real trajectory.

    Only proprioceptive sensors are driven by default.  Encoders and the IMU
    are functions of the motion, which a trajectory has.  Sonar and pressure
    are functions of the geometry the robot was moving through, which it does
    not -- so driving them from a trajectory recorded in somebody's office
    would be inventing a pool and then measuring it.  Pass ``pool=`` and name
    them explicitly if you have a reason.

    Encoder readings are derived from the true body velocity by inverse
    kinematics, which means **no slip**: the encoders agree with the ground
    truth up to the noise model. A real drivetrain's encoders run long, and
    that bias is the largest single term in dead-reckoning drift, so an
    estimator scored this way is being flattered. It is still the harder half
    of the problem -- integrating a real, jerky, badly-conditioned trajectory
    -- and the flattery is at least in a known direction.
    """

    DEFAULT_SENSORS = ("encoder", "imu")

    def __init__(
        self,
        trajectory: Any,
        robot: Any,
        *,
        sensors: Sequence[str] | None = None,
        pool: Any = None,
        depth: float = 1.5,
        seed: int = 0,
        smooth: float = 0.1,
    ) -> None:
        from zimablue.rng import RngTree
        from zimablue.sensors import SensorContext

        self.trajectory = trajectory
        self.robot = robot
        self.pool = pool
        self.depth = float(depth)
        self._context_cls = SensorContext

        names = tuple(sensors) if sensors is not None else self.DEFAULT_SENSORS
        missing = [n for n in names if n not in robot.sensors]
        if missing:
            raise KeyError(
                f"robot {robot.name!r} has no sensors named {missing}; "
                f"it carries {sorted(robot.sensors)}"
            )
        self._sensors = {name: robot.sensors[name] for name in names}
        self.channels = {name: tuple(s.channels) for name, s in self._sensors.items()}

        rng = RngTree(seed)
        for sensor in self._sensors.values():
            sensor.reset()
            sensor.attach(rng)

        self._t = np.asarray(trajectory.time, dtype=float)
        self._x = np.asarray(trajectory.x, dtype=float)
        self._y = np.asarray(trajectory.y, dtype=float)
        self._heading = np.asarray(trajectory.heading, dtype=float)
        self._v, self._omega = trajectory.body_velocities(smooth=smooth)
        grad_t = np.gradient(self._t)
        self._accel = np.gradient(self._v) / grad_t
        self._held: dict[str, Reading] = {}

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def truth_at(self, now: float) -> tuple[float, float, float]:
        """The real pose at ``now`` -- what an estimator should have said."""
        return (
            float(np.interp(now, self._t, self._x)),
            float(np.interp(now, self._t, self._y)),
            float(np.interp(now, self._t, self._heading)),
        )

    def read(self, now: float) -> dict[str, Reading]:
        v = float(np.interp(now, self._t, self._v))
        omega = float(np.interp(now, self._t, self._omega))
        x, y, heading = self.truth_at(now)
        left, right = self.robot.locomotion.to_wheel_speeds(v, omega)

        ctx = self._context_cls(
            time=now,
            x=x,
            y=y,
            heading=heading,
            vx=v * np.cos(heading),
            vy=v * np.sin(heading),
            speed=v,
            yaw_rate=omega,
            accel_forward=float(np.interp(now, self._t, self._accel)),
            accel_lateral=v * omega,
            wheel_speed_left=left,
            wheel_speed_right=right,
            depth=self.depth,
            pool=self.pool,
        )

        readings: dict[str, Reading] = {}
        for name, sensor in self._sensors.items():
            reading = sensor.update(ctx)
            if reading is None:
                # Before the sensor's first sample. Nothing to hold, and
                # nothing worth inventing.
                previous = self._held.get(name)
                reading = (
                    previous
                    if previous is not None
                    else Reading(
                        name=name,
                        time=now,
                        values=np.zeros(len(sensor.channels)),
                        valid=False,
                        fresh=False,
                    )
                )
            self._held[name] = reading
            readings[name] = reading
        return readings

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
