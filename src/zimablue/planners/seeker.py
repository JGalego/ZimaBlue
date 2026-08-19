"""Cleaning as evidence-chasing.

Every planner in this package optimises where the robot *goes*. This one
optimises what it *collects*, and does it without reading anything a real
machine could not: the turbidity probe reports the dirt under the hull, the
EKF reports where the hull probably is, and the controller keeps a memory of
where the readings were high.

The behaviour is three habits layered together, each with a commercial
precedent:

* **scrub what you are on** -- a high reading triggers a tightening spiral
  over the spot, because adhered dirt needs passes, not a drive-by;
* **remember where it was** -- readings are binned into a coarse grid in the
  *estimated* frame, decayed when scrubbed, and the best remembered hotspot
  is worth driving back to;
* **wander when the trail goes cold** -- straight lines and random bounces,
  which is also how the evidence gets gathered in the first place.

What it cannot do is also the point. The memory lives in the estimate's
frame, so odometry drift smears it exactly as it smears the systematic
controller's map; the probe reads the water's haze as dirt everywhere, so the
controller tracks its own running floor and chases the excess. It is the
deployable sibling of :class:`~zimablue.controllers.simple.DirtOracle`, and
the gap between the two is the price of sensing instead of knowing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.geometry import wrap_angle
from zimablue.rng import RngTree
from zimablue.robot import Cleaner, DriveCommand

__all__ = ["DirtSeeker"]

WANDER = "wander"
GOTO = "goto"
SCRUB = "scrub"


class DirtSeeker:
    """Chase sensed dirt: scrub it, remember it, come back for it."""

    name = "dirt_seeker"

    def __init__(
        self,
        seed: int = 0,
        *,
        cell: float = 0.4,
        trigger_ratio: float = 1.7,
        threshold: float = 2.0,
        scrub_time: float = 5.0,
        cooldown: float = 8.0,
        explore_after: float = 12.0,
        goto_patience: float = 25.0,
        replan_interval: float = 3.0,
        travel_cost: float = 0.5,
        turn_range: tuple[float, float] = (0.6, 2.4),
    ) -> None:
        self.seed = seed
        self.cell = float(cell)
        self.trigger_ratio = float(trigger_ratio)
        """How far above the ambient level a reading must rise to be a find.

        Relative, not absolute: dirt is everywhere, and a probe over a lightly
        silted floor reads a steady several g/m2 all day. The signal worth
        acting on is *dirtier than usual*, so the trigger compares against a
        running average of what the probe has been seeing."""

        self.threshold = float(threshold)
        """Anomaly size, g/m2, below which a remembered spot is not worth a trip."""

        self.scrub_time = float(scrub_time)
        self.cooldown = float(cooldown)
        """Seconds after a scrub before another reading can trigger one, so a
        dirty region reads as one find rather than an endless spiral."""

        self.explore_after = float(explore_after)
        """Seconds of forced wandering after working a spot. Without it the
        memory keeps winning: the controller ping-pongs between the hotspots
        it already knows and never finds the pile in the other corner."""

        self.goto_patience = float(goto_patience)
        self.replan_interval = float(replan_interval)
        self.travel_cost = float(travel_cost)
        self.turn_range = turn_range
        self.reset(None)

    def reset(self, robot: Cleaner | None) -> None:
        self._rng = RngTree(self.seed).stream("controller:dirt_seeker")
        self._estimator: Any = None
        self._last_time = 0.0
        self._floor: float | None = None
        self._ambient: float | None = None
        self._cooldown_until = 0.0
        self._goto_block_until = 0.0
        self.hotspots: dict[tuple[int, int], float] = {}
        self.mode = WANDER
        self._target: tuple[float, float] | None = None
        self._mode_since = 0.0
        self._scrub_until = 0.0
        self._turning_until = 0.0
        self._turn_sign = 1.0
        self._last_plan = -1e9
        self._scrubs = 0

    # -- perception ---------------------------------------------------------
    def _pose(self, ctl: ControlInput) -> tuple[float, float, float]:
        """Dead reckoning in the controller's own frame, anchored at reset."""
        from zimablue.estimation import EstimatorConfig, PoseEstimator

        if self._estimator is None:
            self._estimator = PoseEstimator(EstimatorConfig(), origin=(0.0, 0.0, 0.0))
            self._last_time = ctl.time
        dt = max(ctl.time - self._last_time, 0.0)
        self._last_time = ctl.time
        encoder = ctl.reading("encoder")
        imu = ctl.reading("imu")
        speed = 0.5 * (encoder[0] + encoder[1]) if encoder is not None and encoder.valid else 0.0
        gyro = imu[2] if imu is not None and imu.valid else 0.0
        if dt > 0:
            self._estimator.predict(float(speed), float(gyro), dt)
            self._estimator.zero_velocity_update(float(gyro), dt, moving=abs(speed) > 0.02)
        pose = self._estimator.estimate
        return (pose.x, pose.y, pose.heading)

    def _evidence(self, ctl: ControlInput, pose: tuple[float, float, float]) -> float:
        """The anomaly in the latest reading -- how far above usual -- if any.

        Logged into memory when positive, so a find is remembered even when
        the controller is too busy to act on it now.
        """
        reading = ctl.reading("turbidity")
        if reading is None or not reading.valid or not reading.fresh:
            return 0.0
        value = float(reading[0])
        # The probe cannot subtract the water's haze; the controller learns
        # the ambient level instead. EMA over roughly the last few metres of
        # driving, plus a floor that only falls.
        self._floor = value if self._floor is None else min(self._floor, value)
        self._ambient = value if self._ambient is None else 0.97 * self._ambient + 0.03 * value
        anomaly = max(value - self._ambient * self.trigger_ratio, 0.0)
        if anomaly > 0.0:
            key = self._key(pose[0], pose[1])
            self.hotspots[key] = max(anomaly, self.hotspots.get(key, 0.0))
        return anomaly

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.cell)), int(np.floor(y / self.cell)))

    def _centre(self, key: tuple[int, int]) -> tuple[float, float]:
        return ((key[0] + 0.5) * self.cell, (key[1] + 0.5) * self.cell)

    def _best_hotspot(self, pose: tuple[float, float, float]) -> tuple[float, float] | None:
        """The remembered spot most worth the trip, discounted by distance."""
        best, best_score = None, 0.0
        here = self._key(pose[0], pose[1])
        for key, value in self.hotspots.items():
            if value < self.threshold or key == here:
                continue
            cx, cy = self._centre(key)
            distance = float(np.hypot(cx - pose[0], cy - pose[1]))
            if distance < 2.0 * self.cell:
                continue
            score = value / (distance + self.cell) ** self.travel_cost
            if score > best_score:
                best, best_score = (cx, cy), score
        return best

    # -- behaviour ----------------------------------------------------------
    def step(self, ctl: ControlInput) -> DriveCommand:
        top = ctl.robot.locomotion.max_speed
        if ctl.battery <= ctl.robot.power.battery.cutoff:
            return DriveCommand.stop()

        pose = self._pose(ctl)
        excess = self._evidence(ctl, pose)

        contact = ctl.reading("contact")
        bumped = bool(contact is not None and contact.valid and np.any(contact.values > 0.5))
        stuck = ctl.extras.get("stuck", 0.0) > 0.5
        if (bumped or stuck) and ctl.time >= self._turning_until:
            self._turning_until = ctl.time + float(self._rng.uniform(*self.turn_range))
            self._turn_sign = 1.0 if self._rng.random() < 0.5 else -1.0
            if self.mode == SCRUB:
                # A wall ended the spiral early. The spot keeps its score, but
                # the trigger still cools down -- corner dirt would otherwise
                # re-fire on every bounce and pin the robot to the wall.
                self._cooldown_until = ctl.time + 0.5 * self.cooldown
                self._enter(WANDER, ctl.time)
            if self.mode == GOTO and stuck:
                self.hotspots.pop(self._key(*self._target), None)  # type: ignore[misc]
                self._goto_block_until = ctl.time + self.explore_after
                self._enter(WANDER, ctl.time)

        if ctl.time < self._turning_until:
            turn = top * 0.6 * self._turn_sign
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)

        # A hot reading beats every other plan: the dirt is *here*.
        if self.mode != SCRUB and excess > 0.0 and ctl.time >= self._cooldown_until:
            self._enter(SCRUB, ctl.time)
            self._scrub_until = ctl.time + self.scrub_time
            self._scrubs += 1

        if self.mode == SCRUB:
            if excess > 0.0:
                # Still reading dirtier than usual: stay on it, within reason.
                self._scrub_until = min(
                    self._scrub_until + ctl.dt, self._mode_since + 3.0 * self.scrub_time
                )
            if ctl.time >= self._scrub_until:
                key = self._key(pose[0], pose[1])
                self.hotspots[key] = self.hotspots.get(key, 0.0) * 0.25
                self._cooldown_until = ctl.time + self.cooldown
                self._goto_block_until = ctl.time + self.explore_after
                self._enter(WANDER, ctl.time)
            else:
                return self._spiral(ctl, top)

        if self.mode == GOTO:
            assert self._target is not None
            distance = float(np.hypot(self._target[0] - pose[0], self._target[1] - pose[1]))
            if distance < 0.3:
                # Arrived where the memory says the dirt was; work the spot
                # even if the first reading has not confirmed it yet.
                self._enter(SCRUB, ctl.time)
                self._scrub_until = ctl.time + self.scrub_time
                self._scrubs += 1
                return self._spiral(ctl, top)

            if ctl.time - self._mode_since > self.goto_patience:
                # Cannot seem to get there -- drift, walls, or a bad memory.
                self.hotspots.pop(self._key(*self._target), None)
                self._goto_block_until = ctl.time + self.explore_after
                self._enter(WANDER, ctl.time)
            else:
                return self._pursue(pose, self._target, top)

        if ctl.time - self._last_plan >= self.replan_interval:
            self._last_plan = ctl.time
            # Memories age: the estimate they were binned in has drifted, and
            # the dirt itself may have. Fade them, and forget the faint ones.
            self.hotspots = {
                key: value * 0.98
                for key, value in self.hotspots.items()
                if value * 0.98 >= 0.5 * self.threshold
            }
            target = self._best_hotspot(pose) if ctl.time >= self._goto_block_until else None
            if target is not None:
                self._target = target
                self._enter(GOTO, ctl.time)
                return self._pursue(pose, target, top)

        return DriveCommand(left=top * 0.85, right=top * 0.85, brush=True, pump=1.0)

    def _enter(self, mode: str, now: float) -> None:
        self.mode = mode
        self._mode_since = now
        if mode != GOTO:
            self._target = None

    def _spiral(self, ctl: ControlInput, top: float) -> DriveCommand:
        """A tightening-then-opening pass over the spot the reading fired on."""
        elapsed = ctl.time - self._mode_since
        # Start tight and open up: the ratio between tracks rises from 0.25
        # towards 0.8, so the spiral grows a swath per turn or so.
        ratio = 0.25 + 0.55 * min(elapsed / self.scrub_time, 1.0)
        speed = top * 0.55
        if self._turn_sign > 0:
            return DriveCommand(left=speed * ratio, right=speed, brush=True, pump=1.0)
        return DriveCommand(left=speed, right=speed * ratio, brush=True, pump=1.0)

    def _pursue(
        self, pose: tuple[float, float, float], target: tuple[float, float], top: float
    ) -> DriveCommand:
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        error = float(wrap_angle(np.arctan2(dy, dx) - pose[2]))
        if abs(error) > np.deg2rad(35.0):
            turn = top * 0.5 * float(np.sign(error))
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)
        correction = float(np.clip(error * 1.1, -0.5, 0.5))
        speed = top * 0.9
        return DriveCommand(
            left=speed * (1.0 - correction),
            right=speed * (1.0 + correction),
            brush=True,
            pump=1.0,
        )

    # -- introspection -------------------------------------------------------
    def telemetry(self) -> dict[str, float]:
        est = self._estimator.estimate if self._estimator is not None else None
        return {
            "est_x": est.x if est else 0.0,
            "est_y": est.y if est else 0.0,
            "est_heading": est.heading if est else 0.0,
            "mode": {WANDER: 0.0, GOTO: 1.0, SCRUB: 2.0}[self.mode],
            "hotspots": float(len(self.hotspots)),
            "scrubs": float(self._scrubs),
            "ambient": self._ambient if self._ambient is not None else 0.0,
        }


@CONTROLLERS.register("dirt_seeker")
def _make_dirt_seeker(**kwargs: object) -> DirtSeeker:
    return DirtSeeker(**kwargs)  # type: ignore[arg-type]
