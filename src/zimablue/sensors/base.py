"""Sensor base class and the imperfection pipeline every sensor shares.

The design follows the argument in ``docs/research.md`` section 4: a policy or
estimator developed against perfect sensors learns to use information the real
robot does not have.  So there is exactly one place where a clean measurement
becomes a realistic one, and every sensor goes through it.

The pipeline, in order:

1. ``_measure()`` -- the sensor's own ground-truth model
2. fault scale and offset
3. bias: a constant plus a random walk (the tractable core of the Allan-variance
   model -- white noise plus a Gauss-Markov bias)
4. Gaussian white noise
5. quantisation
6. saturation
7. stuck-value hold
8. dropout

Sampling rate is handled *outside* the pipeline: a sensor only measures when its
own period has elapsed, and latency delays when the sample becomes readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.rng import RngTree

__all__ = [
    "Reading",
    "Sensor",
    "SensorConfig",
    "SensorContext",
    "SensorFault",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SensorConfig:
    """Nominal (fault-free) imperfections of a sensor.

    These describe a *healthy* unit: even a working sensor is noisy, biased,
    band-limited and slightly late.  Faults are a separate, additive concept.
    """

    rate_hz: float = 50.0
    """Sampling frequency. Between samples the last reading is held."""

    noise_std: float = 0.0
    """Standard deviation of additive white noise, in the sensor's units."""

    bias: float = 0.0
    """Constant offset present from power-on."""

    bias_walk: float = 0.0
    """Random-walk rate of the bias, units per sqrt(second)."""

    latency: float = 0.0
    """Delay between measurement and availability, in seconds."""

    dropout_probability: float = 0.0
    """Per-sample probability of returning no new data."""

    min_value: float = -np.inf
    max_value: float = np.inf
    """Saturation limits; readings clip here."""

    quantization: float = 0.0
    """Resolution of the ADC/encoder. 0 disables quantisation."""

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")
        if self.noise_std < 0 or self.bias_walk < 0:
            raise ValueError("noise_std and bias_walk must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError(
                f"dropout_probability must be in [0, 1], got {self.dropout_probability}"
            )
        if self.latency < 0:
            raise ValueError(f"latency must be non-negative, got {self.latency}")
        if self.min_value > self.max_value:
            raise ValueError("min_value must not exceed max_value")

    @property
    def period(self) -> float:
        return 1.0 / self.rate_hz

    def to_dict(self) -> dict[str, float]:
        def clean(v: float) -> float:
            # JSON has no infinity; recordings stay valid JSON.
            return float(v) if np.isfinite(v) else (1e308 if v > 0 else -1e308)

        return {
            "rate_hz": self.rate_hz,
            "noise_std": self.noise_std,
            "bias": self.bias,
            "bias_walk": self.bias_walk,
            "latency": self.latency,
            "dropout_probability": self.dropout_probability,
            "min_value": clean(self.min_value),
            "max_value": clean(self.max_value),
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensorConfig:
        out = dict(data)
        for key in ("min_value", "max_value"):
            if key in out and abs(float(out[key])) >= 1e308:
                out[key] = -np.inf if float(out[key]) < 0 else np.inf
        return cls(**{k: float(v) for k, v in out.items()})


@dataclass
class SensorFault:
    """A deliberate defect layered on top of a sensor's nominal behaviour.

    Mutable and additive: ``inject_fault`` updates fields in place so a fault
    can begin part-way through a run, which is the case fault-injection testing
    actually cares about.
    """

    bias: float = 0.0
    """Extra offset added to every channel."""

    scale: float = 1.0
    """Gain error: the true value is multiplied by this."""

    noise_multiplier: float = 1.0
    """Scales the nominal noise standard deviation."""

    dropout_probability: float = 0.0
    """Extra per-sample dropout on top of the nominal rate."""

    stuck: bool = False
    """When set, the sensor keeps returning its last value forever."""

    start_time: float = 0.0
    """Simulation time at which the fault becomes active."""

    label: str = "fault"

    @property
    def is_identity(self) -> bool:
        return (
            self.bias == 0.0
            and self.scale == 1.0
            and self.noise_multiplier == 1.0
            and self.dropout_probability == 0.0
            and not self.stuck
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": self.bias,
            "scale": self.scale,
            "noise_multiplier": self.noise_multiplier,
            "dropout_probability": self.dropout_probability,
            "stuck": self.stuck,
            "start_time": self.start_time,
            "label": self.label,
        }


@dataclass(frozen=True)
class Reading:
    """One sensor output."""

    name: str
    time: float
    """Simulation time the reading became available (measurement time + latency)."""

    values: FloatArray
    valid: bool = True
    """False when the sample was dropped; ``values`` then holds the last good data."""

    fresh: bool = True
    """False when this is a held sample rather than a new measurement."""

    def __getitem__(self, index: int) -> float:
        return float(self.values[index])

    @property
    def value(self) -> float:
        """First channel, for single-channel sensors."""
        return float(self.values[0])


@dataclass
class SensorContext:
    """Ground truth handed to sensors each tick.

    Sensors read from this and never from the simulation state directly, which
    keeps them usable from any backend: a 3D backend fills the same struct from
    PhysX instead of from the 2D integrator.
    """

    time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    """Yaw in radians."""

    vx: float = 0.0
    vy: float = 0.0
    """World-frame linear velocity, m/s."""

    speed: float = 0.0
    """Body-frame forward ground speed, m/s."""

    yaw_rate: float = 0.0
    accel_forward: float = 0.0
    accel_lateral: float = 0.0
    wheel_speed_left: float = 0.0
    wheel_speed_right: float = 0.0
    """*Wheel surface* speeds -- these exceed ground speed when slipping."""

    depth: float = 0.0
    """Water depth above the robot, m."""

    contacts: tuple[bool, bool, bool, bool] = (False, False, False, False)
    """Front, left, right, rear bump state."""

    pool: Any = None
    """The :class:`~zimablue.pool.Pool`, for sensors that must query geometry."""

    water: Any = None
    """The :class:`~zimablue.pool.Water` properties."""

    neighbours: tuple[tuple[float, float, float], ...] = ()
    """Other robots, as ``(x, y, radius)`` discs.

    Empty for a single-robot run. A fleet fills it, and a rangefinder that
    ignored it would let a robot drive confidently into a team-mate it was
    pointing straight at."""

    extras: dict[str, float] = field(default_factory=dict)


class Sensor:
    """Base class: sampling, imperfection pipeline, latency and faults.

    Subclasses implement :meth:`_measure` and set ``channels``.
    """

    channels: tuple[str, ...] = ("value",)
    kind: str = "sensor"

    def __init__(self, name: str, config: SensorConfig | None = None) -> None:
        self.name = name
        self.config = config if config is not None else SensorConfig()
        self.faults: list[SensorFault] = []
        self._rng: np.random.Generator | None = None
        self._next_sample_time = 0.0
        self._bias_state: FloatArray = np.zeros(len(self.channels))
        self._last_values: FloatArray = np.zeros(len(self.channels))
        self._held: FloatArray | None = None
        self._pending: list[tuple[float, FloatArray, bool]] = []
        self._latest: Reading | None = None

    # -- lifecycle ------------------------------------------------------
    def attach(self, rng: RngTree) -> None:
        """Bind this sensor to its own named RNG stream and clear state."""
        self._rng = rng.stream(f"sensor:{self.name}")
        self.reset()

    def reset(self) -> None:
        n = len(self.channels)
        self._next_sample_time = 0.0
        self._bias_state = np.full(n, self.config.bias, dtype=float)
        self._last_values = np.zeros(n, dtype=float)
        self._held = None
        self._pending = []
        self._latest = None

    # -- fault injection ------------------------------------------------
    def inject_fault(
        self,
        *,
        bias: float = 0.0,
        scale: float = 1.0,
        noise_multiplier: float = 1.0,
        dropout_probability: float = 0.0,
        stuck: bool = False,
        start_time: float = 0.0,
        label: str = "fault",
    ) -> SensorFault:
        """Schedule a defect on this sensor.

        >>> sonar.inject_fault(bias=0.15, dropout_probability=0.02)

        Faults accumulate: biases and dropout probabilities add, scales and
        noise multipliers compose.  ``start_time`` lets a fault appear part-way
        through a run.
        """
        fault = SensorFault(
            bias=bias,
            scale=scale,
            noise_multiplier=noise_multiplier,
            dropout_probability=dropout_probability,
            stuck=stuck,
            start_time=start_time,
            label=label,
        )
        self.faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        self.faults.clear()

    def active_fault(self, t: float) -> SensorFault:
        """Composition of every fault active at time ``t``."""
        combined = SensorFault()
        for fault in self.faults:
            if t + 1e-12 < fault.start_time:
                continue
            combined.bias += fault.bias
            combined.scale *= fault.scale
            combined.noise_multiplier *= fault.noise_multiplier
            # Independent dropout sources compose as 1 - prod(1 - p).
            combined.dropout_probability = 1.0 - (1.0 - combined.dropout_probability) * (
                1.0 - fault.dropout_probability
            )
            combined.stuck = combined.stuck or fault.stuck
        return combined

    # -- measurement ----------------------------------------------------
    def _measure(self, ctx: SensorContext) -> FloatArray:
        """Return the perfect measurement. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _measure(); "
            "it returns the ideal value the noise pipeline then corrupts"
        )

    def update(self, ctx: SensorContext) -> Reading | None:
        """Advance the sensor to ``ctx.time`` and return the readable output.

        Returns ``None`` only before the very first sample is available.  After
        that it always returns a :class:`Reading`, with ``fresh=False`` when the
        value is being held between samples.
        """
        if self._rng is None:
            raise RuntimeError(
                f"sensor {self.name!r} was never attached to an RngTree; "
                "call sensor.attach(rng) or add it to a Cleaner before stepping"
            )
        t = ctx.time
        if t + 1e-12 >= self._next_sample_time:
            values, valid = self._sample(ctx, t)
            self._pending.append((t + self.config.latency, values, valid))
            # Snap to the sampling lattice rather than accumulating from t, so
            # the sample times stay exactly periodic.
            period = self.config.period
            missed = int(np.floor((t - self._next_sample_time) / period)) + 1
            self._next_sample_time += missed * period

        fresh = False
        while self._pending and self._pending[0][0] <= t + 1e-12:
            release, values, valid = self._pending.pop(0)
            self._latest = Reading(self.name, release, values, valid=valid, fresh=True)
            fresh = True
        if self._latest is None:
            return None
        if not fresh:
            self._latest = Reading(
                self._latest.name,
                self._latest.time,
                self._latest.values,
                valid=self._latest.valid,
                fresh=False,
            )
        return self._latest

    def _sample(self, ctx: SensorContext, t: float) -> tuple[FloatArray, bool]:
        """Run one measurement through the imperfection pipeline."""
        assert self._rng is not None
        rng = self._rng
        cfg = self.config
        fault = self.active_fault(t)

        raw = np.asarray(self._measure(ctx), dtype=float).reshape(len(self.channels))

        # 2. fault gain and offset
        values = raw * fault.scale + fault.bias

        # 3. bias: constant (seeded into _bias_state) plus random walk
        if cfg.bias_walk > 0:
            self._bias_state = self._bias_state + rng.normal(
                0.0, cfg.bias_walk * np.sqrt(cfg.period), size=values.shape
            )
        values = values + self._bias_state

        # 4. white noise
        sigma = cfg.noise_std * fault.noise_multiplier
        if sigma > 0:
            values = values + rng.normal(0.0, sigma, size=values.shape)

        # 5. quantisation
        if cfg.quantization > 0:
            values = np.round(values / cfg.quantization) * cfg.quantization

        # 6. saturation
        values = np.clip(values, cfg.min_value, cfg.max_value)

        # 7. stuck: freeze on the value held when the fault began
        if fault.stuck:
            if self._held is None:
                self._held = values.copy()
            values = self._held.copy()
        else:
            self._held = None

        # 8. dropout: report the previous value but flag it invalid
        p_drop = 1.0 - (1.0 - cfg.dropout_probability) * (1.0 - fault.dropout_probability)
        if p_drop > 0 and rng.random() < p_drop:
            return self._last_values.copy(), False

        self._last_values = values.copy()
        return values, True

    # -- description ----------------------------------------------------
    def spec(self) -> dict[str, Any]:
        """JSON-safe description, embedded in recordings."""
        return {
            "name": self.name,
            "kind": self.kind,
            "class": type(self).__name__,
            "channels": list(self.channels),
            "config": self.config.to_dict(),
            "params": self._params(),
            "faults": [f.to_dict() for f in self.faults],
        }

    def _params(self) -> dict[str, Any]:
        """Subclass-specific parameters for :meth:`spec`."""
        return {}

    def with_config(self, **changes: Any) -> Sensor:
        """A copy of this sensor with configuration overrides applied."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.config = replace(self.config, **changes)
        clone.faults = list(self.faults)
        clone.reset()
        return clone

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(name={self.name!r}, rate={self.config.rate_hz} Hz)"
