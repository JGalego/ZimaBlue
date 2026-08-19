"""The five sensor models a pool cleaner actually carries.

Chosen from the underwater-localization literature (``docs/research.md``
section 3): proprioception (encoders, IMU) drifts without bound, pressure depth
is the one cheap drift-free channel, and contact plus short-range sonar are the
exteroceptive bounds a cleaner can afford.  Nothing here needs a camera.

Default noise figures are consumer-MEMS-class order of magnitude, not
datasheet values for a specific part.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.sensors.base import Sensor, SensorConfig, SensorContext

__all__ = [
    "GRAVITY",
    "IMU",
    "ContactSensor",
    "Encoder",
    "PressureSensor",
    "Sonar",
    "TurbidityProbe",
]

FloatArray = NDArray[np.float64]

GRAVITY = 9.80665


class Encoder(Sensor):
    """Wheel/track encoders, one channel per side.

    Reports the speed of the *wheel surface*, not the robot's ground speed.
    When the tracks slip, the encoder over-reports and odometry integrated from
    it runs long -- which is exactly the dead-reckoning drift described in the
    research notes, arising from the physics rather than from an injected error.
    """

    channels = ("left", "right")
    kind = "encoder"

    def __init__(
        self,
        name: str = "encoder",
        config: SensorConfig | None = None,
        *,
        ticks_per_metre: float = 4000.0,
    ) -> None:
        # Encoder resolution is a quantisation of speed: one tick per sampling
        # period is (1/ticks_per_metre) * rate_hz m/s.  Derive it rather than
        # making the user state the same physical fact twice.
        cfg = config if config is not None else SensorConfig(rate_hz=50.0, noise_std=0.002)
        if cfg.quantization == 0.0 and ticks_per_metre > 0:
            cfg = replace(cfg, quantization=cfg.rate_hz / ticks_per_metre)
        super().__init__(name, cfg)
        self.ticks_per_metre = ticks_per_metre

    def _measure(self, ctx: SensorContext) -> FloatArray:
        return np.array([ctx.wheel_speed_left, ctx.wheel_speed_right], dtype=float)

    def _params(self) -> dict[str, Any]:
        return {"ticks_per_metre": self.ticks_per_metre}


class IMU(Sensor):
    """A 6-axis MEMS IMU: planar accelerations plus yaw rate.

    No absolute heading is reported, because a gyro cannot provide one.  Heading
    must be integrated from ``gz``, and it will drift -- see
    :class:`~zimablue.controllers.baseline.HeadingEstimator`.
    """

    channels = ("ax", "ay", "gz")
    kind = "imu"

    def __init__(
        self,
        name: str = "imu",
        config: SensorConfig | None = None,
        *,
        gyro_bias_deg_s: float = 0.4,
    ) -> None:
        cfg = (
            config
            if config is not None
            else SensorConfig(
                rate_hz=100.0,
                noise_std=0.02,
                bias_walk=0.001,
                min_value=-40.0,
                max_value=40.0,
            )
        )
        super().__init__(name, cfg)
        self.gyro_bias = np.deg2rad(gyro_bias_deg_s)

    def reset(self) -> None:
        super().reset()
        # Only the gyro channel carries the turn-on bias; accelerometers get the
        # config bias like every other channel.
        self._bias_state = self._bias_state.astype(float)
        self._bias_state[2] += self.gyro_bias

    def _measure(self, ctx: SensorContext) -> FloatArray:
        return np.array([ctx.accel_forward, ctx.accel_lateral, ctx.yaw_rate], dtype=float)

    def _params(self) -> dict[str, Any]:
        return {"gyro_bias_deg_s": float(np.rad2deg(self.gyro_bias))}


class PressureSensor(Sensor):
    """Depth from hydrostatic pressure: ``p = rho * g * h``.

    The drift-free channel.  In a pool of known bathymetry it is also a weak
    position prior, since depth varies with location on a sloped floor.
    """

    channels = ("depth",)
    kind = "pressure"

    def __init__(
        self,
        name: str = "pressure",
        config: SensorConfig | None = None,
        *,
        max_depth: float = 5.0,
    ) -> None:
        cfg = (
            config
            if config is not None
            else SensorConfig(
                rate_hz=10.0,
                noise_std=0.01,
                bias=0.0,
                latency=0.02,
                min_value=0.0,
                max_value=max_depth,
                quantization=0.005,
            )
        )
        super().__init__(name, cfg)
        self.max_depth = max_depth

    def _measure(self, ctx: SensorContext) -> FloatArray:
        return np.array([ctx.depth], dtype=float)

    def pressure_pa(self, depth: float, density: float = 997.0) -> float:
        """Gauge pressure for a depth, for callers that want raw units."""
        return density * GRAVITY * depth

    def _params(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth}


class TurbidityProbe(Sensor):
    """Optical turbidity at the intake, calibrated to dirt under the hull.

    Commercial cleaners ship a version of this as "dirt detect": an LED and a
    photodiode in the intake throat, reading the light scattered by whatever
    the brush just lifted. The clean value is the mean dirt mass per square
    metre under the hull, plus the water's own haze as a baseline the probe
    cannot subtract -- a hazy pool reads dirty everywhere, which is a property
    of the instrument and not a bug in it.

    This is the one sensor whose channel *is* the project's thesis: a
    controller that reads it can chase grams instead of area, and remains
    deployable, because the probe measures scattered light rather than
    anything only a simulator knows.
    """

    channels = ("density",)
    kind = "turbidity"

    def __init__(
        self,
        name: str = "turbidity",
        config: SensorConfig | None = None,
        *,
        haze_gain: float = 6.0,
    ) -> None:
        cfg = (
            config
            if config is not None
            else SensorConfig(
                rate_hz=5.0,
                noise_std=0.4,
                latency=0.05,
                min_value=0.0,
                max_value=400.0,
                quantization=0.1,
            )
        )
        super().__init__(name, cfg)
        self.haze_gain = haze_gain
        """How many g/m2 a fully hazy pool adds to every reading."""

    def _measure(self, ctx: SensorContext) -> FloatArray:
        haze = self.haze_gain * float(getattr(ctx.water, "turbidity", 0.0) or 0.0)
        return np.array([ctx.dirt_density + haze], dtype=float)

    def _params(self) -> dict[str, Any]:
        return {"haze_gain": self.haze_gain}


class ContactSensor(Sensor):
    """Bump switches: front, left, right, rear.

    Binary channels carried as floats so the whole pipeline stays one code path;
    quantisation to 1.0 keeps them binary after noise is added, and a dropout
    or a stuck fault behaves exactly as it would on the analogue sensors.
    """

    channels = ("front", "left", "right", "rear")
    kind = "contact"

    def __init__(self, name: str = "contact", config: SensorConfig | None = None) -> None:
        cfg = (
            config
            if config is not None
            else SensorConfig(
                rate_hz=100.0,
                noise_std=0.0,
                min_value=0.0,
                max_value=1.0,
                quantization=1.0,
            )
        )
        super().__init__(name, cfg)

    def _measure(self, ctx: SensorContext) -> FloatArray:
        return np.array([1.0 if c else 0.0 for c in ctx.contacts], dtype=float)

    @staticmethod
    def any_contact(values: FloatArray) -> bool:
        return bool(np.any(np.asarray(values) > 0.5))


class Sonar(Sensor):
    """A short-range acoustic rangefinder with one or more fixed beams.

    Beams are cast against the pool's collision geometry.  Turbid water
    attenuates returns, so beyond an effective range set by
    ``water.turbidity`` the sensor reports its max range -- the same "no
    return" behaviour a real transducer shows.
    """

    kind = "sonar"

    def __init__(
        self,
        name: str = "sonar",
        config: SensorConfig | None = None,
        *,
        beam_angles: tuple[float, ...] = (0.0,),
        max_range: float = 3.0,
        min_range: float = 0.04,
    ) -> None:
        self.beam_angles = tuple(float(a) for a in beam_angles)
        self.channels = tuple(f"beam_{i}" for i in range(len(self.beam_angles)))
        self.max_range = max_range
        self.min_range = min_range
        cfg = (
            config
            if config is not None
            else SensorConfig(
                rate_hz=20.0,
                noise_std=0.015,
                latency=0.03,
                dropout_probability=0.01,
                min_value=0.0,
                max_value=max_range,
                quantization=0.005,
            )
        )
        super().__init__(name, cfg)

    def _measure(self, ctx: SensorContext) -> FloatArray:
        pool = ctx.pool
        if pool is None:
            return np.full(len(self.beam_angles), self.max_range, dtype=float)
        angles = np.array(self.beam_angles, dtype=float) + ctx.heading
        ranges = pool.raycast((ctx.x, ctx.y), angles, self.max_range)
        if ctx.neighbours:
            ranges = np.minimum(ranges, _range_to_discs(ctx.x, ctx.y, angles, ctx.neighbours))

        turbidity = getattr(ctx.water, "turbidity", 0.0) if ctx.water is not None else 0.0
        if turbidity > 0:
            # Effective range falls off with turbidity; anything beyond it reads
            # as "no return", i.e. max range.
            effective = self.max_range * float(np.exp(-2.0 * turbidity))
            ranges = np.where(ranges > effective, self.max_range, ranges)
        return np.maximum(ranges, self.min_range)

    def _params(self) -> dict[str, Any]:
        return {
            "beam_angles": list(self.beam_angles),
            "max_range": self.max_range,
            "min_range": self.min_range,
        }


def _range_to_discs(
    x: float,
    y: float,
    angles: FloatArray,
    discs: tuple[tuple[float, float, float], ...],
) -> FloatArray:
    """Distance along each ray to the nearest disc, or infinity.

    Closed-form ray-circle intersection rather than marching: a beam is a
    half-line, a robot is a disc, and the quadratic has an answer. Discs the
    ray starts inside are ignored -- that is two robots already overlapping,
    which is the collision resolver's problem and not the sonar's.
    """
    dx, dy = np.cos(angles), np.sin(angles)
    best = np.full(angles.shape, np.inf)
    for cx, cy, radius in discs:
        ox, oy = cx - x, cy - y
        along = dx * ox + dy * oy
        gap = ox * ox + oy * oy - radius * radius
        discriminant = along * along - gap
        hit = (discriminant >= 0.0) & (along > 0.0) & (gap > 0.0)
        if not hit.any():
            continue
        distance = np.where(hit, along - np.sqrt(np.maximum(discriminant, 0.0)), np.inf)
        best = np.minimum(best, distance)
    return best
