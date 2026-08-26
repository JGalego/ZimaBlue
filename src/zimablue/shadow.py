"""A read-only digital twin that follows live commands and sensor readings."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from zimablue.controllers.base import ControlInput
from zimablue.dirt import DirtSpec
from zimablue.pool import Pool
from zimablue.robot import Cleaner, DriveCommand
from zimablue.sensors import Reading
from zimablue.simulation import Simulation

__all__ = ["ResidualStats", "ShadowHealth", "ShadowTwin"]


@dataclass(frozen=True)
class ResidualStats:
    """Rolling statistics for one scalar sensor channel."""

    samples: int
    mean: float
    rms: float
    maximum: float


@dataclass(frozen=True)
class ShadowHealth:
    """Current agreement between the live machine and its shadow."""

    time: float
    ticks: int
    residuals: dict[str, ResidualStats]
    anomalies: tuple[str, ...]
    score: float

    @property
    def healthy(self) -> bool:
        return not self.anomalies

    def summary(self) -> str:
        state = "healthy" if self.healthy else f"anomalies: {', '.join(self.anomalies)}"
        return f"shadow t={self.time:.2f}s score={self.score:.2f} — {state}"


class _CommandReplay:
    name = "shadow_replay"

    def __init__(self) -> None:
        self.command = DriveCommand.stop()
        self.readings: dict[str, Reading] = {}

    def reset(self, robot: Cleaner) -> None:
        self.command = DriveCommand.stop()
        self.readings = {}

    def step(self, control_input: ControlInput) -> DriveCommand:
        self.readings = dict(control_input.readings)
        return self.command


class ShadowTwin:
    """Advance a model under real commands and monitor sensor residuals.

    The shadow has no actuator callback and cannot command hardware. One call to
    :meth:`observe` mirrors one hardware tick: the model senses its current
    state, applies the command that the real controller already chose, advances
    physics, and compares its predicted readings with the live readings.
    """

    def __init__(
        self,
        *,
        pool: Pool | str = "rectangular",
        robot: str | Cleaner = "tracked",
        dirt: DirtSpec | str = "clean",
        timestep: float = 0.02,
        seed: int = 0,
        thresholds: Mapping[str, float] | None = None,
        window: int = 250,
        minimum_samples: int = 10,
    ) -> None:
        if window <= 0:
            raise ValueError("shadow residual window must be positive")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        self._controller = _CommandReplay()
        self.simulation = Simulation(
            pool=pool,
            robot=robot,
            dirt=dirt,
            controller=self._controller,
            timestep=timestep,
            seed=seed,
            record=False,
        )
        self.timestep = timestep
        self.minimum_samples = minimum_samples
        self.thresholds = dict(thresholds or {})
        if any(not np.isfinite(value) or value <= 0.0 for value in self.thresholds.values()):
            raise ValueError("shadow thresholds must be finite and positive")
        self._window = window
        self._residuals: dict[str, deque[float]] = {}
        self._ticks = 0
        self._channels = {
            name: tuple(sensor.channels) for name, sensor in self.simulation.robot.sensors.items()
        }

    def observe(
        self,
        command: DriveCommand,
        readings: Mapping[str, Reading],
        *,
        dt: float | None = None,
    ) -> ShadowHealth:
        """Mirror one completed control decision without issuing a command."""
        interval = self.timestep if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("shadow observation interval must be finite and positive")
        self.simulation.timestep = interval
        self._controller.command = command
        self.simulation.step()
        predicted = self._controller.readings
        self._accumulate(predicted, readings)
        self._ticks += 1
        return self.health()

    def health(self) -> ShadowHealth:
        """Return rolling residual statistics and threshold violations."""
        statistics = {}
        anomalies = []
        ratios = []
        for channel, values in sorted(self._residuals.items()):
            data = np.asarray(values, dtype=float)
            stats = ResidualStats(
                samples=len(values),
                mean=float(data.mean()),
                rms=float(np.sqrt(np.mean(data**2))),
                maximum=float(np.max(np.abs(data))),
            )
            statistics[channel] = stats
            threshold = self.thresholds.get(channel)
            if threshold is not None and stats.samples >= self.minimum_samples:
                ratio = stats.rms / threshold
                ratios.append(ratio)
                if ratio > 1.0:
                    anomalies.append(channel)
        return ShadowHealth(
            time=float(self.simulation.state.time),
            ticks=self._ticks,
            residuals=statistics,
            anomalies=tuple(anomalies),
            score=max(ratios, default=0.0),
        )

    def close(self) -> None:
        self.simulation.backend.close()

    def _accumulate(
        self,
        predicted: Mapping[str, Reading],
        observed: Mapping[str, Reading],
    ) -> None:
        for sensor_name, actual in observed.items():
            model = predicted.get(sensor_name)
            if model is None or not actual.valid or not model.valid:
                continue
            names = self._channels.get(sensor_name, ())
            count = min(len(names), len(actual.values), len(model.values))
            for index in range(count):
                residual = float(model.values[index] - actual.values[index])
                if not np.isfinite(residual):
                    continue
                channel = f"{sensor_name}.{names[index]}"
                self._residuals.setdefault(channel, deque(maxlen=self._window)).append(residual)
