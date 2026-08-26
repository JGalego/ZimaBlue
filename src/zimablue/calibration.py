"""Fit a digital twin to an observed trajectory.

Calibration is split at the simulation boundary. ZimaBlue owns the optimizer,
trajectory comparison and reproducibility record; the caller owns how a
parameter vector builds a candidate twin. Built-in robots, custom backends and
hardware logs can therefore use the same calibration machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from zimablue.recording import Recording

__all__ = [
    "CalibrationResult",
    "CalibrationStep",
    "Parameter",
    "TwinCalibrator",
    "trajectory_loss",
]


@dataclass(frozen=True)
class Parameter:
    """One bounded scalar to identify."""

    name: str
    lower: float
    upper: float
    initial: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name cannot be empty")
        if not np.isfinite([self.lower, self.upper]).all() or self.lower >= self.upper:
            raise ValueError(f"invalid bounds for {self.name!r}: ({self.lower}, {self.upper})")
        if self.initial is not None and not self.lower <= self.initial <= self.upper:
            raise ValueError(
                f"initial value for {self.name!r} must be inside [{self.lower}, {self.upper}]"
            )


@dataclass(frozen=True)
class CalibrationStep:
    """Best candidate after one optimizer generation."""

    generation: int
    evaluations: int
    loss: float
    parameters: dict[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    """The identified twin and enough provenance to reproduce it."""

    parameters: dict[str, float]
    loss: float
    history: tuple[CalibrationStep, ...]
    evaluations: int
    seed: int
    method: str = "differential_evolution"
    recording: Recording | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "method": self.method,
            "seed": self.seed,
            "evaluations": self.evaluations,
            "loss": self.loss,
            "parameters": dict(self.parameters),
            "history": [
                {
                    "generation": step.generation,
                    "evaluations": step.evaluations,
                    "loss": step.loss,
                    "parameters": dict(step.parameters),
                }
                for step in self.history
            ],
        }

    def annotate(self, recording: Recording | None = None) -> Recording:
        """Return a recording carrying this calibration in its manifest."""
        source = recording if recording is not None else self.recording
        if source is None:
            raise ValueError("no fitted recording is available to annotate")
        manifest = dict(source.manifest)
        manifest["calibration"] = self.to_dict()
        return replace(source, manifest=manifest)


LossFunction = Callable[[Recording, Recording], float]
Simulator = Callable[[Mapping[str, float]], Recording]


def trajectory_loss(
    reference: Recording,
    candidate: Recording,
    *,
    channels: Sequence[str] = ("x", "y", "heading"),
    weights: Mapping[str, float] | None = None,
) -> float:
    """Mean squared trajectory error on the reference clock.

    Candidate channels are linearly interpolated onto reference timestamps.
    Heading residuals take the short way around the circle. Missing channels,
    empty recordings and non-overlapping clocks are rejected rather than
    silently producing a persuasive but meaningless score.
    """
    if not channels:
        raise ValueError("at least one loss channel is required")
    reference_time = _time(reference)
    candidate_time = _time(candidate)
    start = max(float(reference_time[0]), float(candidate_time[0]))
    stop = min(float(reference_time[-1]), float(candidate_time[-1]))
    selected = (reference_time >= start) & (reference_time <= stop)
    if not selected.any():
        raise ValueError("recordings have no overlapping timestamps")

    clock = reference_time[selected]
    total = 0.0
    total_weight = 0.0
    channel_weights = weights or {}
    for channel in channels:
        if channel not in reference.frames or channel not in candidate.frames:
            raise KeyError(f"channel {channel!r} is not present in both recordings")
        weight = float(channel_weights.get(channel, 1.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"weight for {channel!r} must be finite and non-negative")
        if weight == 0.0:
            continue
        observed = np.asarray(reference.frames[channel], dtype=float)[selected]
        simulated = np.interp(
            clock, candidate_time, np.asarray(candidate.frames[channel], dtype=float)
        )
        residual = simulated - observed
        if channel in {"heading", "yaw"}:
            residual = np.arctan2(np.sin(residual), np.cos(residual))
        finite = np.isfinite(residual)
        if not finite.any():
            raise ValueError(f"channel {channel!r} has no finite overlapping samples")
        total += weight * float(np.mean(residual[finite] ** 2))
        total_weight += weight
    if total_weight == 0.0:
        raise ValueError("loss channel weights cannot all be zero")
    return total / total_weight


def _time(recording: Recording) -> np.ndarray:
    if "time" not in recording.frames:
        raise KeyError("recording has no 'time' channel")
    time = np.asarray(recording.frames["time"], dtype=float)
    if time.size == 0:
        raise ValueError("recording has no frames")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("recording timestamps must be finite and strictly increasing")
    return time


class TwinCalibrator:
    """Identify bounded twin parameters against one observed recording.

    ``simulate`` receives a name-to-value mapping and returns the candidate
    recording. It can construct a regular :class:`~zimablue.Simulation`, call a
    custom backend, or replay a hardware model. A fixed optimizer seed makes
    the search itself reproducible; deterministic simulation remains the
    caller's responsibility.
    """

    def __init__(
        self,
        reference: Recording,
        simulate: Simulator,
        parameters: Sequence[Parameter],
        *,
        loss: LossFunction = trajectory_loss,
    ) -> None:
        _time(reference)
        if not parameters:
            raise ValueError("at least one calibration parameter is required")
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("calibration parameter names must be unique")
        self.reference = reference
        self.simulate = simulate
        self.parameters = tuple(parameters)
        self.loss = loss

    def fit(
        self,
        *,
        seed: int = 0,
        generations: int = 40,
        population: int | None = None,
        mutation: float = 0.8,
        crossover: float = 0.7,
    ) -> CalibrationResult:
        """Fit with bounded differential evolution."""
        if generations < 0:
            raise ValueError("generations must be non-negative")
        dimensions = len(self.parameters)
        size = population if population is not None else max(8, 6 * dimensions)
        if size < 4:
            raise ValueError("population must contain at least four candidates")
        if not 0.0 < mutation <= 2.0:
            raise ValueError("mutation must be in (0, 2]")
        if not 0.0 <= crossover <= 1.0:
            raise ValueError("crossover must be in [0, 1]")

        rng = np.random.default_rng(seed)
        lower = np.array([parameter.lower for parameter in self.parameters], dtype=float)
        upper = np.array([parameter.upper for parameter in self.parameters], dtype=float)
        vectors = rng.uniform(lower, upper, size=(size, dimensions))
        initial = [parameter.initial for parameter in self.parameters]
        if all(value is not None for value in initial):
            vectors[0] = np.asarray(initial, dtype=float)

        evaluations = 0
        recordings: list[Recording] = []
        scores = np.empty(size, dtype=float)
        for index, vector in enumerate(vectors):
            scores[index], recording = self._evaluate(vector)
            recordings.append(recording)
            evaluations += 1

        history = [self._step(0, evaluations, vectors, scores)]
        indices = np.arange(size)
        for generation in range(1, generations + 1):
            for index in range(size):
                choices = rng.choice(indices[indices != index], size=3, replace=False)
                a, b, c = vectors[choices]
                mutant = np.clip(a + mutation * (b - c), lower, upper)
                crossed = rng.random(dimensions) < crossover
                crossed[rng.integers(dimensions)] = True
                trial = np.where(crossed, mutant, vectors[index])
                score, recording = self._evaluate(trial)
                evaluations += 1
                if score <= scores[index]:
                    vectors[index] = trial
                    scores[index] = score
                    recordings[index] = recording
            history.append(self._step(generation, evaluations, vectors, scores))

        best = int(np.argmin(scores))
        return CalibrationResult(
            parameters=self._mapping(vectors[best]),
            loss=float(scores[best]),
            history=tuple(history),
            evaluations=evaluations,
            seed=seed,
            recording=recordings[best],
        )

    def score(self, parameters: Mapping[str, float]) -> float:
        """Evaluate one complete parameter mapping without optimization."""
        expected = {parameter.name for parameter in self.parameters}
        if set(parameters) != expected:
            missing = sorted(expected - set(parameters))
            extra = sorted(set(parameters) - expected)
            raise ValueError(f"parameter mismatch; missing={missing}, extra={extra}")
        vector = np.array([parameters[parameter.name] for parameter in self.parameters])
        for value, parameter in zip(vector, self.parameters, strict=True):
            if not parameter.lower <= value <= parameter.upper:
                raise ValueError(f"{parameter.name!r} is outside its bounds")
        score, _ = self._evaluate(vector)
        return score

    def _evaluate(self, vector: np.ndarray) -> tuple[float, Recording]:
        recording = self.simulate(self._mapping(vector))
        if not isinstance(recording, Recording):
            raise TypeError("simulate must return a Recording")
        score = float(self.loss(self.reference, recording))
        if not np.isfinite(score) or score < 0.0:
            raise ValueError(f"calibration loss must be finite and non-negative, got {score}")
        return score, recording

    def _mapping(self, vector: np.ndarray) -> dict[str, float]:
        return {
            parameter.name: float(value)
            for parameter, value in zip(self.parameters, vector, strict=True)
        }

    def _step(
        self,
        generation: int,
        evaluations: int,
        vectors: np.ndarray,
        scores: np.ndarray,
    ) -> CalibrationStep:
        best = int(np.argmin(scores))
        return CalibrationStep(
            generation=generation,
            evaluations=evaluations,
            loss=float(scores[best]),
            parameters=self._mapping(vectors[best]),
        )
