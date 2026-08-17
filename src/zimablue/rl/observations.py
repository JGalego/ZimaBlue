"""Extra observation channels, for splitting the problem up.

The env's default observation is what a controller sees: sensor channels and
nothing derived. A policy learning from that has to solve estimation and
planning at once, from a pool with no absolute reference, which means a
recurrent policy and a long training run.

Splitting the two is usually the better experiment. Hand the agent a pose
estimate and an occupancy map summary, computed by the same classical code the
``systematic`` controller uses, and what is left to learn is the planner ::

    from zimablue.rl import PoolCleaningEnv
    from zimablue.rl.observations import EstimatedPose

    env = PoolCleaningEnv(extra_observations=EstimatedPose())

That is also the shape of a fix for the result in the README where a *better*
position estimate halves coverage: the estimator is fine and the lane planner
is brittle, so replacing the planner is the interesting move.

Nothing here reads ground truth. An extra observation that did would be an
oracle wearing a policy's clothes, and every number it produced would be a
lie -- the same rule the controller interface has always had.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from zimablue.controllers.base import ControlInput
from zimablue.controllers.systematic import OccupancyMap
from zimablue.estimation import EstimatorConfig, PoseEstimator
from zimablue.robot import Cleaner

__all__ = ["EstimatedPose", "ExtraObservations"]


class ExtraObservations(Protocol):
    """Derived channels appended to the env's observation.

    ``channels`` names them, ``bounds`` gives the low and high for each, and
    the object is called once per physics tick with that tick's
    :class:`ControlInput`. Per tick rather than per decision, because anything
    integrating -- an EKF, a map -- needs every sample, not one in ten.
    """

    channels: tuple[str, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]

    def reset(self, robot: Cleaner) -> None: ...

    def __call__(self, control_input: ControlInput) -> NDArray[np.float32]: ...


class EstimatedPose:
    """Where the robot thinks it is, and how much it has seen.

    Runs the same EKF and occupancy map as
    :class:`~zimablue.controllers.systematic.SystematicCoverage`, over position,
    heading and gyro bias, with zero-velocity updates. Adds seven channels:

    ``est.x``, ``est.y``
        Position in the estimator's own frame, which starts at the origin and
        drifts. Not metres from any wall -- there is no absolute reference in
        a pool.
    ``est.heading_sin``, ``est.heading_cos``
        Heading, split so it is continuous across the wrap. A policy fed a
        raw angle has to learn that 179 degrees and -179 are neighbours.
    ``est.sigma``
        The filter's own position uncertainty. Worth having: it tells a policy
        when to stop trusting the rest of this.
    ``est.explored``, ``est.covered``
        Fraction of the map's cells known and swept. The only progress signal
        in the observation that is not the clock.
    """

    channels = (
        "est.x",
        "est.y",
        "est.heading_sin",
        "est.heading_cos",
        "est.sigma",
        "est.explored",
        "est.covered",
    )

    def __init__(
        self,
        *,
        config: EstimatorConfig | None = None,
        map_cell: float = 0.25,
        extent: float = 30.0,
    ) -> None:
        self.config = config or EstimatorConfig()
        self.map_cell = map_cell
        self.extent = extent
        self.estimator = PoseEstimator(self.config)
        self.map = OccupancyMap(cell=map_cell, extent=extent)
        # Taken from the map rather than recomputed from extent and cell: the
        # first version of that arithmetic was out by a factor of four, and
        # nothing downstream of a fraction between 0 and 1 would have noticed.
        self._cells = max(self.map.grid.size, 1)
        self._last_time = 0.0
        self._swath = 0.3
        self._radius = 0.2

        # The position bound is deliberately loose. The map clips at half the
        # extent, but the estimate behind it is dead reckoning with no absolute
        # reference, so it is free to wander off the map and saying otherwise
        # would put a number outside its own observation space.
        self.bounds = (
            (-extent, -extent, -1.0, -1.0, 0.0, 0.0, 0.0),
            (extent, extent, 1.0, 1.0, np.inf, 1.0, 1.0),
        )

    def reset(self, robot: Cleaner) -> None:
        self.estimator = PoseEstimator(self.config)
        self.map = OccupancyMap(cell=self.map_cell, extent=self.extent)
        self._cells = max(self.map.grid.size, 1)
        self._last_time = 0.0
        self._swath = robot.swath_width
        self._radius = robot.radius

    def __call__(self, control_input: ControlInput) -> NDArray[np.float32]:
        dt = max(control_input.time - self._last_time, 0.0)
        self._last_time = control_input.time

        encoder = control_input.reading("encoder")
        if encoder is not None and encoder.valid:
            left, right = float(encoder[0]), float(encoder[1])
        else:
            left = right = 0.0
        imu = control_input.reading("imu")
        gyro = float(imu[2]) if imu is not None and imu.valid else 0.0

        self.estimator.predict(0.5 * (left + right), gyro, dt)
        moving = max(abs(left), abs(right)) > self.config.zupt_speed
        self.estimator.zero_velocity_update(gyro, dt, moving=moving)

        pose = self.estimator.estimate
        self.map.mark_free(pose.x, pose.y, self._radius)
        self.map.mark_covered(pose.x, pose.y, 0.5 * self._swath)
        contact = control_input.reading("contact")
        if contact is not None and contact.valid and any(v > 0.5 for v in contact.values):
            ahead = self._radius + 0.05
            self.map.mark_wall(
                pose.x + ahead * np.cos(pose.heading), pose.y + ahead * np.sin(pose.heading)
            )

        return np.asarray(
            [
                pose.x,
                pose.y,
                np.sin(pose.heading),
                np.cos(pose.heading),
                pose.position_sigma,
                self.map.explored_cells / self._cells,
                self.map.covered_cells / self._cells,
            ],
            dtype=np.float32,
        )
