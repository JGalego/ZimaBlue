"""The failures a simulator is not obliged to have.

Every controller in this library was written against a loop that cannot miss a
tick, cannot lose a sensor and cannot raise.  None of those hold on a robot,
and the difference is not a detail: a coverage planner that keeps steering on a
four-second-old sonar reading will drive confidently into a wall, and it will
do it without any of its own code being wrong.

:class:`Watchdog` is the piece that decides when the controller has stopped
being entitled to drive.  It knows nothing about the mission -- it checks
liveness, timing and the loop's own health, trips on the first breach, and
latches until something clears it.  Latching is deliberate: a sensor that
flickers back for one tick has not recovered, and a watchdog that un-trips on
its own turns an intermittent fault into an intermittent robot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from zimablue.robot import DriveCommand
from zimablue.sensors import Reading

__all__ = ["SafetyLimits", "Watchdog"]


@dataclass(frozen=True)
class SafetyLimits:
    """When to stop trusting the controller.

    Defaults are deliberately loose enough not to trip on a healthy 50 Hz loop
    with the sensor rates the shipped robots declare, and tight enough that a
    genuinely wedged loop stops the motors inside a second.
    """

    max_reading_age: float = 0.5
    """How long a sensor may go without a usable reading before it counts as lost.

    Measured as a duration, not as a single sample. One dropped packet is
    normal on any real bus -- the shipped sensor models have a dropout
    probability for exactly that reason -- and a watchdog that stops the robot
    on the first one stops it constantly.
    """

    max_loop_period: float = 0.25
    """Longest tolerable gap between ticks, seconds.

    Catches the loop being starved -- a controller that blocked on I/O, a
    garbage collection pause, a scheduler that lost the thread.
    """

    max_decide_time: float = 0.1
    """Longest the controller may take to return a command, seconds."""

    required: tuple[str, ...] = ()
    """Sensors without which the robot must not drive.

    Empty by default because it depends on the controller: a dead-reckoning
    planner cannot survive losing the encoders, a bump-and-turn one can.
    """

    startup_grace: float = 1.0
    """Seconds after start during which staleness is not checked.

    Everything is stale before its first sample arrives, and tripping on that
    would mean no run ever begins.
    """


@dataclass
class Watchdog:
    """Trips on liveness, timing or controller failure, and latches.

    Call :meth:`check` once per tick with the readings and how long the tick
    took.  It returns the reasons that tripped this tick -- empty when all is
    well.  Once tripped, :attr:`tripped` stays true and :attr:`reasons`
    accumulates, until :meth:`clear` is called by something that has decided
    the robot is fit to drive again.  That decision is not the watchdog's.
    """

    limits: SafetyLimits = field(default_factory=SafetyLimits)
    tripped: bool = False
    reasons: list[str] = field(default_factory=list)
    _started: float | None = field(default=None, repr=False)
    _last_good: dict[str, float] = field(default_factory=dict, repr=False)

    def check(
        self,
        now: float,
        readings: Mapping[str, Reading],
        *,
        loop_period: float | None = None,
        decide_time: float | None = None,
        error: BaseException | None = None,
    ) -> list[str]:
        if self._started is None:
            self._started = now
        tripped_now: list[str] = []

        if error is not None:
            # A controller that raised has no opinion about the motors, and the
            # loop must not carry on holding its last command as though it did.
            tripped_now.append(f"controller raised {type(error).__name__}: {error}")

        if loop_period is not None and loop_period > self.limits.max_loop_period:
            tripped_now.append(
                f"loop period {loop_period * 1e3:.0f} ms over "
                f"{self.limits.max_loop_period * 1e3:.0f} ms"
            )
        if decide_time is not None and decide_time > self.limits.max_decide_time:
            tripped_now.append(
                f"controller took {decide_time * 1e3:.0f} ms, over "
                f"{self.limits.max_decide_time * 1e3:.0f} ms"
            )

        for name in self.limits.required:
            reading = readings.get(name)
            if reading is not None and reading.valid:
                self._last_good[name] = min(now, reading.time)

        if now - self._started >= self.limits.startup_grace:
            for name in self.limits.required:
                if name not in readings:
                    tripped_now.append(f"required sensor {name!r} is not present")
                    continue
                # A sensor that has never been valid is timed from the start of
                # the run, so it trips as soon as the grace period is over
                # rather than never -- the failure mode of a sensor that is
                # unplugged is that it says nothing at all.
                age = now - self._last_good.get(name, self._started)
                if age > self.limits.max_reading_age:
                    tripped_now.append(
                        f"sensor {name!r} has had no usable reading for {age * 1e3:.0f} ms"
                    )

        if tripped_now:
            self.tripped = True
            self.reasons.extend(tripped_now)
        return tripped_now

    def clear(self) -> None:
        """Declare the robot fit to drive again."""
        self.tripped = False
        self.reasons.clear()

    def ages(self, now: float) -> dict[str, float]:
        """Seconds since each required sensor last had a usable reading."""
        start = self._started if self._started is not None else now
        return {name: now - self._last_good.get(name, start) for name in self.limits.required}

    @staticmethod
    def safe_command() -> DriveCommand:
        """What to send when the watchdog has tripped.

        Everything off, including the pump. A tripped watchdog means the loop
        does not know what is happening, and a pump running in an unknown
        state is a pump running dry.
        """
        return DriveCommand.stop()
