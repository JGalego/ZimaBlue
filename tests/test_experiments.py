"""Autonomous sequential simulation experiments."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb


def objective(parameters, seed):
    noise = np.random.default_rng(seed).normal(0.0, 0.002)
    return -((parameters["x"] - 0.7) ** 2 + (parameters["y"] - 0.2) ** 2) + noise


def test_autonomous_experiment_converges_and_reports_uncertainty():
    experiment = zb.AutonomousExperiment(
        objective,
        [zb.Parameter("x", 0.0, 1.0), zb.Parameter("y", 0.0, 1.0)],
        zb.ExperimentObjective("quality", maximize=True),
        seed=12,
    )

    result = experiment.run(generations=8, population=12, replicates=4)

    assert result.parameters["x"] == pytest.approx(0.7, abs=0.12)
    assert result.parameters["y"] == pytest.approx(0.2, abs=0.12)
    assert result.evaluations == 8 * 12 * 4
    assert result.replicate_seeds == (12, 13, 14, 15)
    assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
    assert len(result.history) == 8
    assert result.to_dict()["parameters"] == result.parameters


def test_autonomous_experiment_is_deterministic():
    def run():
        return zb.AutonomousExperiment(
            objective,
            [zb.Parameter("x", 0.0, 1.0), zb.Parameter("y", 0.0, 1.0)],
            zb.ExperimentObjective("quality"),
            seed=44,
        ).run(generations=3, population=5, replicates=2)

    assert run().to_dict() == run().to_dict()


def test_every_candidate_uses_common_random_numbers():
    seen = []

    def evaluate(parameters, seed):
        seen.append(seed)
        return parameters["x"] + seed * 0.0

    result = zb.AutonomousExperiment(
        evaluate,
        [zb.Parameter("x", 0.0, 1.0)],
        zb.ExperimentObjective("score"),
    ).run(generations=2, population=3, replicates=2, replicate_seed=100)

    assert seen == [100, 101] * 6
    assert result.replicate_seeds == (100, 101)


def test_minimization_and_input_validation():
    experiment = zb.AutonomousExperiment(
        lambda parameters, seed: abs(parameters["x"] - 0.25),
        [zb.Parameter("x", 0.0, 1.0, initial=0.25)],
        zb.ExperimentObjective("error", maximize=False),
    )
    result = experiment.run(generations=2, population=4, replicates=1)
    assert result.parameters["x"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="population"):
        experiment.run(population=1)


def test_non_finite_experiment_outcome_is_rejected():
    experiment = zb.AutonomousExperiment(
        lambda parameters, seed: np.nan,
        [zb.Parameter("x", 0.0, 1.0)],
        zb.ExperimentObjective("score"),
    )
    with pytest.raises(ValueError, match="non-finite"):
        experiment.run(generations=1, population=2, replicates=1)
