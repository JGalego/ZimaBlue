"""State estimation -- turning drifting sensors into a usable pose.

Until now nothing in ZimaBlue consumed the sensor data. The baseline controller
integrated the raw gyro and steered on bump switches, which is why injecting
sensor faults barely changed its behaviour: there was almost nothing for a
fault to corrupt.

This module closes that gap. :class:`PoseEstimator` is a four-state extended
Kalman filter -- position, heading, and **gyro bias** -- fed by wheel encoders
and the IMU, with zero-velocity updates to observe the bias.

Why the bias is the interesting state:

A MEMS gyro has a turn-on bias of a few tenths of a degree per second. Integrate
that for thirty minutes and heading is wrong by tens of degrees, which turns a
boustrophedon lane pattern into a fan. The bias is *not* observable while the
robot is turning -- rotation and bias enter the measurement identically. It
becomes observable the moment the robot stops: a stationary gyro should read
zero, so whatever it does read is the bias. That is a zero-velocity update
(ZUPT), and it is the standard trick in pedestrian and vehicle dead reckoning.

What this filter does *not* do: bound position drift. With no absolute position
reference -- no GNSS, no beacons, no map matching -- position error grows
without bound, exactly as the underwater-localization literature describes
(``docs/research.md`` section 3). The filter slows the growth and reports how
uncertain it is; it does not eliminate it. A consumer that needs bounded
position error has to close the loop against the world, which is what
:class:`~zimablue.controllers.systematic.OccupancyMap` starts to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from zimablue.geometry import wrap_angle

__all__ = ["EstimatorConfig", "PoseEstimate", "PoseEstimator"]

FloatArray = NDArray[np.float64]

# State vector layout.
X, Y, THETA, BIAS = 0, 1, 2, 3
N_STATES = 4


@dataclass(frozen=True)
class EstimatorConfig:
    """Noise parameters. Defaults are matched to the shipped sensor suite."""

    encoder_scale: float = 1.0
    """Multiplier applied to the encoder speed before integrating.

    Track slip is a *bias*, not noise: the wheels always turn further than the
    robot travels, never less. Measured on the reference cleaner, encoder-
    derived distance runs about 6% long on straight sections. An EKF whose
    process noise is zero-mean cannot absorb that -- it integrates the error
    steadily and stays confident while doing it, which is the worst
    combination. Odometry calibration is the standard answer and is what every
    real robot does at commissioning; the *residual* slip variation is what
    ``speed_noise`` then models.

    Left at 1.0 by default because the right value belongs to a specific robot
    on a specific surface. :class:`~zimablue.controllers.systematic.SystematicCoverage`
    sets its own."""

    speed_noise: float = 0.06
    """Std-dev of encoder-derived forward speed, m/s. Dominated by track slip
    rather than by encoder resolution, so it is far larger than the sensor's
    own noise figure."""

    gyro_noise: float = 0.02
    """Std-dev of the gyro rate measurement, rad/s."""

    bias_walk: float = 2e-4
    """Random-walk rate of the gyro bias, rad/s per sqrt(s)."""

    lateral_noise: float = 0.01
    """Std-dev of unmodelled sideways motion, m/s. A tracked robot scrubs
    sideways when it turns."""

    zupt_speed: float = 0.01
    """Wheel speed below which the robot counts as stationary, m/s.

    Compared against the *fastest* wheel, never the average. A robot spinning
    on the spot has equal and opposite wheel speeds, so its average is zero --
    and a filter that trusts the average will happily absorb the entire
    rotation rate into the bias state."""

    zupt_max_rate: float = 0.35
    """Reject a zero-velocity update whose gyro reading exceeds this, rad/s.

    A stationary robot cannot really be yawing at 20 deg/s, so a reading that
    large means the stillness detection is wrong. Refusing the update is far
    cheaper than corrupting the bias with it."""

    zupt_settle: float = 0.25
    """Seconds of continuous stillness before a ZUPT is applied. Waiting a
    moment avoids folding the deceleration transient into the bias."""

    initial_bias_sigma: float = 0.02
    """Prior uncertainty on the gyro bias at power-on, rad/s."""

    contact_noise: float = 0.12
    """Std-dev of a wall-contact position fix, m."""


@dataclass
class PoseEstimate:
    """The filter's belief about where the robot is."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    gyro_bias: float = 0.0
    covariance: FloatArray = field(default_factory=lambda: np.zeros((N_STATES, N_STATES)))
    zupt_count: int = 0
    """How many zero-velocity updates have been applied."""

    @property
    def position_sigma(self) -> float:
        """One-sigma position uncertainty in metres (mean of the two axes)."""
        return float(np.sqrt(0.5 * (self.covariance[X, X] + self.covariance[Y, Y])))

    @property
    def heading_sigma(self) -> float:
        return float(np.sqrt(self.covariance[THETA, THETA]))

    @property
    def bias_sigma(self) -> float:
        return float(np.sqrt(self.covariance[BIAS, BIAS]))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.heading)


class PoseEstimator:
    """An EKF over ``[x, y, heading, gyro_bias]``.

    Usage is one call per tick::

        estimator.predict(v_encoder, gyro_rate, dt)
        estimator.zero_velocity_update(gyro_rate)   # when standing still
        estimate = estimator.estimate

    The filter runs in its **own frame**, anchored wherever the robot started.
    It has no idea where that is in world coordinates, which is the honest
    situation for a cleaner dropped into a pool: it can track relative motion,
    not absolute position.
    """

    def __init__(
        self,
        config: EstimatorConfig | None = None,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.config = config or EstimatorConfig()
        self._origin = origin
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.state = np.array([self._origin[0], self._origin[1], self._origin[2], 0.0], dtype=float)
        self.covariance = np.diag([1e-6, 1e-6, 1e-6, cfg.initial_bias_sigma**2]).astype(float)
        self._still_for = 0.0
        self._zupt_count = 0

    # ------------------------------------------------------------------
    @property
    def estimate(self) -> PoseEstimate:
        return PoseEstimate(
            x=float(self.state[X]),
            y=float(self.state[Y]),
            heading=float(self.state[THETA]),
            gyro_bias=float(self.state[BIAS]),
            covariance=self.covariance.copy(),
            zupt_count=self._zupt_count,
        )

    # ------------------------------------------------------------------
    def predict(self, speed: float, gyro_rate: float, dt: float) -> None:
        """Propagate the state with an encoder speed and a gyro rate.

        Uses the midpoint heading over the interval rather than the heading at
        the start, which is the same second-order correction the simulator's
        own integrator makes -- an estimator that integrates more crudely than
        the plant produces error that says more about the filter than about the
        sensors.
        """
        if dt <= 0:
            return
        cfg = self.config
        speed = speed * cfg.encoder_scale
        theta = self.state[THETA]
        omega = gyro_rate - self.state[BIAS]
        mid = theta + 0.5 * omega * dt

        self.state[X] += speed * np.cos(mid) * dt
        self.state[Y] += speed * np.sin(mid) * dt
        self.state[THETA] = wrap_angle(theta + omega * dt)

        # Jacobian of the motion model with respect to the state.
        f = np.eye(N_STATES)
        f[X, THETA] = -speed * np.sin(mid) * dt
        f[Y, THETA] = speed * np.cos(mid) * dt
        # A bias error rotates the robot, which curves the path: the position
        # sensitivity to bias is second order in dt and easy to forget.
        f[X, BIAS] = 0.5 * speed * np.sin(mid) * dt * dt
        f[Y, BIAS] = -0.5 * speed * np.cos(mid) * dt * dt
        f[THETA, BIAS] = -dt

        # Process noise: forward noise along the heading, lateral noise across
        # it, gyro noise on heading, random walk on the bias.
        cos_t, sin_t = np.cos(mid), np.sin(mid)
        along = (cfg.speed_noise * dt) ** 2
        across = (cfg.lateral_noise * dt) ** 2
        q = np.zeros((N_STATES, N_STATES))
        q[X, X] = along * cos_t**2 + across * sin_t**2
        q[Y, Y] = along * sin_t**2 + across * cos_t**2
        q[X, Y] = q[Y, X] = (along - across) * cos_t * sin_t
        # The same rate error that changes the heading also rotates this
        # interval's displacement through ``mid``. Mapping it through the
        # motion model keeps those errors correlated; adding only a heading
        # variance makes the filter overconfident about curved motion.
        gyro_effect = np.array(
            [
                -0.5 * speed * sin_t * dt * dt,
                0.5 * speed * cos_t * dt * dt,
                dt,
                0.0,
            ]
        )
        q += cfg.gyro_noise**2 * np.outer(gyro_effect, gyro_effect)
        q[BIAS, BIAS] = cfg.bias_walk**2 * dt

        self.covariance = f @ self.covariance @ f.T + q

    # ------------------------------------------------------------------
    def zero_velocity_update(self, gyro_rate: float, dt: float, *, moving: bool) -> bool:
        """Observe the gyro bias while the robot is stationary.

        Returns whether an update was applied. A stationary gyro should read
        zero, so its reading *is* the bias -- the one moment the bias becomes
        observable. Without this the filter can only ever propagate its
        power-on prior.
        """
        if moving:
            self._still_for = 0.0
            return False
        self._still_for += dt
        if self._still_for < self.config.zupt_settle:
            return False
        if abs(gyro_rate) > self.config.zupt_max_rate:
            # We were told the robot is still, but the gyro disagrees loudly.
            # Believe the gyro and skip: a bad ZUPT is worse than none.
            return False

        # z = bias + noise, so H picks out the bias state directly.
        h = np.zeros((1, N_STATES))
        h[0, BIAS] = 1.0
        self._update(np.array([gyro_rate]), h, np.array([[self.config.gyro_noise**2]]))
        self._zupt_count += 1
        return True

    def position_update(self, x: float, y: float, sigma: float | None = None) -> None:
        """Fold in an absolute position fix.

        Nothing in the shipped stack produces one -- there is no beacon and no
        map matching. It exists so that a user who adds a localisation source
        has somewhere to put it, and so the mapping controller can correct
        against a recognised wall.
        """
        noise = (sigma if sigma is not None else self.config.contact_noise) ** 2
        h = np.zeros((2, N_STATES))
        h[0, X] = 1.0
        h[1, Y] = 1.0
        self._update(np.array([x, y]), h, np.diag([noise, noise]))

    def wall_update(
        self, point: tuple[float, float], normal: tuple[float, float], sigma: float = 0.1
    ) -> None:
        """Fold in a wall touch: the robot's centre lies on a known line.

        Touching a wall pins exactly one dimension -- how far from the wall
        the robot is -- and says nothing about where *along* it the robot
        sits. So this is a 1D update along the wall's ``normal``, through
        ``point`` (the wall's surface pushed in by the hull radius), rather
        than a 2D position fix that would also drag the estimate sideways to
        wherever the touch was guessed to be.
        """
        nx, ny = normal
        h = np.zeros((1, N_STATES))
        h[0, X] = nx
        h[0, Y] = ny
        z = np.array([nx * point[0] + ny * point[1]])
        self._update(z, h, np.array([[sigma**2]]))

    def heading_update(self, heading: float, sigma: float = 0.05) -> None:
        """Fold in an absolute heading observation."""
        h = np.zeros((1, N_STATES))
        h[0, THETA] = 1.0
        innovation = np.array([wrap_angle(heading - self.state[THETA])])
        self._update_with_innovation(innovation, h, np.array([[sigma**2]]))

    # ------------------------------------------------------------------
    def _update(self, measurement: FloatArray, h: FloatArray, r: FloatArray) -> None:
        innovation = measurement - h @ self.state
        self._update_with_innovation(innovation, h, r)

    def _update_with_innovation(self, innovation: FloatArray, h: FloatArray, r: FloatArray) -> None:
        s = h @ self.covariance @ h.T + r
        gain = self.covariance @ h.T @ np.linalg.inv(s)
        self.state = self.state + gain @ innovation
        self.state[THETA] = wrap_angle(self.state[THETA])

        # Joseph form: it costs a couple of extra matrix products and keeps the
        # covariance symmetric and positive-definite over a 90 000-step run,
        # where the simple (I - KH)P form drifts and can go indefinite.
        identity = np.eye(N_STATES)
        factor = identity - gain @ h
        self.covariance = factor @ self.covariance @ factor.T + gain @ r @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
