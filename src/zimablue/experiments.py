"""Deterministic sequential design for simulation experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from zimablue.calibration import Parameter

__all__ = [
    "AutonomousExperiment",
    "CandidateResult",
    "ExperimentGeneration",
    "ExperimentObjective",
    "ExperimentResult",
]


@dataclass(frozen=True)
class ExperimentObjective:
    """A scalar outcome and whether larger or smaller values are preferable."""

    name: str
    maximize: bool = True
    unit: str = ""


@dataclass(frozen=True)
class CandidateResult:
    """One parameter proposal evaluated under common random seeds."""

    parameters: dict[str, float]
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def standard_deviation(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    @property
    def standard_error(self) -> float:
        return self.standard_deviation / np.sqrt(len(self.values))

    @property
    def confidence_interval(self) -> tuple[float, float]:
        radius = 1.96 * self.standard_error
        return self.mean - radius, self.mean + radius


@dataclass(frozen=True)
class ExperimentGeneration:
    """Candidates tested together and the incumbent after that generation."""

    index: int
    candidates: tuple[CandidateResult, ...]
    best: CandidateResult
    evaluations: int


@dataclass(frozen=True)
class ExperimentResult:
    """Complete reproducible history of an autonomous search."""

    objective: ExperimentObjective
    parameters: dict[str, float]
    value: float
    confidence_interval: tuple[float, float]
    history: tuple[ExperimentGeneration, ...]
    evaluations: int
    seed: int
    replicate_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": {
                "name": self.objective.name,
                "maximize": self.objective.maximize,
                "unit": self.objective.unit,
            },
            "parameters": dict(self.parameters),
            "value": self.value,
            "confidence_interval": list(self.confidence_interval),
            "evaluations": self.evaluations,
            "seed": self.seed,
            "replicate_seeds": list(self.replicate_seeds),
            "history": [
                {
                    "generation": generation.index,
                    "best": dict(generation.best.parameters),
                    "best_value": generation.best.mean,
                    "evaluations": generation.evaluations,
                }
                for generation in self.history
            ],
        }


class AutonomousExperiment:
    """Adaptively search bounded parameters using replicated simulation trials.

    ``evaluate(parameters, seed)`` supplies the experiment-specific simulation
    and returns one finite scalar objective. Every candidate is tested under
    the same replicate seeds, making differences paired rather than confounded
    by different random worlds.
    """

    def __init__(
        self,
        evaluate: Callable[[Mapping[str, float], int], float],
        parameters: Sequence[Parameter],
        objective: ExperimentObjective,
        *,
        seed: int = 0,
    ) -> None:
        if not parameters:
            raise ValueError("at least one experiment parameter is required")
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("experiment parameter names must be unique")
        self.evaluate = evaluate
        self.parameters = tuple(parameters)
        self.objective = objective
        self.seed = int(seed)

    def run(
        self,
        *,
        generations: int = 8,
        population: int = 10,
        replicates: int = 3,
        replicate_seed: int | None = None,
        elite_fraction: float = 0.3,
    ) -> ExperimentResult:
        if generations <= 0 or population < 2 or replicates <= 0:
            raise ValueError(
                "generations and replicates must be positive; population must be at least 2"
            )
        if not 0.0 < elite_fraction <= 0.5:
            raise ValueError("elite_fraction must be in (0, 0.5]")

        rng = np.random.default_rng(self.seed)
        lower = np.asarray([parameter.lower for parameter in self.parameters], dtype=float)
        upper = np.asarray([parameter.upper for parameter in self.parameters], dtype=float)
        spans = upper - lower
        initial = np.asarray(
            [
                (parameter.lower + parameter.upper) / 2.0
                if parameter.initial is None
                else parameter.initial
                for parameter in self.parameters
            ],
            dtype=float,
        )
        seed_start = self.seed if replicate_seed is None else replicate_seed
        seeds = tuple(seed_start + i for i in range(replicates))
        elite_count = max(1, int(np.ceil(population * elite_fraction)))
        centre = initial
        scale = spans / 3.0
        incumbent: CandidateResult | None = None
        history = []
        evaluations = 0

        for generation in range(generations):
            vectors = np.empty((population, len(self.parameters)), dtype=float)
            vectors[0] = (
                centre if incumbent is None else _vector(incumbent.parameters, self.parameters)
            )
            if generation == 0:
                vectors[1:] = rng.uniform(lower, upper, size=(population - 1, len(lower)))
            else:
                vectors[1:] = np.clip(
                    rng.normal(centre, scale, size=(population - 1, len(lower))), lower, upper
                )
            candidates = tuple(self._evaluate(vector, seeds) for vector in vectors)
            evaluations += population * replicates
            ranked = sorted(candidates, key=self._rank, reverse=True)
            if incumbent is None or self._rank(ranked[0]) > self._rank(incumbent):
                incumbent = ranked[0]
            elites = ranked[:elite_count]
            elite_vectors = np.asarray(
                [_vector(candidate.parameters, self.parameters) for candidate in elites]
            )
            centre = elite_vectors.mean(axis=0)
            empirical = elite_vectors.std(axis=0)
            scale = np.maximum(np.minimum(scale * 0.7, empirical + scale * 0.25), spans * 0.01)
            history.append(
                ExperimentGeneration(
                    index=generation,
                    candidates=candidates,
                    best=incumbent,
                    evaluations=evaluations,
                )
            )

        assert incumbent is not None
        return ExperimentResult(
            objective=self.objective,
            parameters=dict(incumbent.parameters),
            value=incumbent.mean,
            confidence_interval=incumbent.confidence_interval,
            history=tuple(history),
            evaluations=evaluations,
            seed=self.seed,
            replicate_seeds=seeds,
        )

    def _evaluate(self, vector: np.ndarray, seeds: tuple[int, ...]) -> CandidateResult:
        parameters = {
            parameter.name: float(value)
            for parameter, value in zip(self.parameters, vector, strict=True)
        }
        values = tuple(float(self.evaluate(parameters, seed)) for seed in seeds)
        if not np.isfinite(values).all():
            raise ValueError(f"experiment returned a non-finite {self.objective.name}")
        return CandidateResult(parameters=parameters, values=values)

    def _rank(self, candidate: CandidateResult) -> float:
        return candidate.mean if self.objective.maximize else -candidate.mean


def _vector(parameters: Mapping[str, float], definitions: Sequence[Parameter]) -> np.ndarray:
    return np.asarray([parameters[item.name] for item in definitions], dtype=float)
