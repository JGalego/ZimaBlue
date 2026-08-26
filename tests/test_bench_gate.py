"""Benchmark regression gates for continuous integration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from zimablue.bench import (
    BenchDefinition,
    BenchResult,
    BenchTolerance,
    compare_benchmarks,
)
from zimablue.planners.compare import Comparison, Trial

DEFINITION = BenchDefinition(
    name="gate-test",
    entries=("planner",),
    pools=("rectangular",),
    seeds=(1, 2),
    minutes=0.1,
)


def result(coverage=(0.8, 0.82), energy=(2.0, 2.2)):
    trials = [
        Trial(
            planner="planner",
            pool="rectangular",
            seed=seed,
            scores={"coverage": coverage[index], "energy": energy[index]},
            path=np.zeros((0, 2)),
            curve=(np.zeros(0), np.zeros(0)),
        )
        for index, seed in enumerate((1, 2))
    ]
    return BenchResult(DEFINITION, Comparison(trials=trials))


def test_benchmark_gate_is_direction_aware():
    baseline = result()
    current = result(coverage=(0.75, 0.77), energy=(2.4, 2.6))

    gate = compare_benchmarks(
        current,
        baseline.to_dict(),
        {
            "coverage": BenchTolerance(absolute=0.02),
            "energy": BenchTolerance(relative=0.05),
        },
    )

    assert not gate.passed
    assert {(check.metric, check.regressed) for check in gate.checks} == {
        ("coverage", True),
        ("energy", True),
    }
    with pytest.raises(AssertionError, match="coverage"):
        gate.assert_passed()
    assert "FAIL" in gate.to_markdown()


def test_benchmark_gate_accepts_improvements_and_small_noise(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(result().to_dict()))
    current = result(coverage=(0.805, 0.825), energy=(2.02, 2.22))

    gate = compare_benchmarks(
        current,
        baseline_path,
        {
            "coverage": BenchTolerance(absolute=0.01),
            "energy": BenchTolerance(relative=0.02),
        },
    )

    assert gate.passed
    gate.assert_passed()
    assert "PASS" in gate.to_markdown()


def test_benchmark_gate_rejects_incomparable_or_incomplete_results():
    baseline = result().to_dict()
    baseline["trials"].pop()
    gate = compare_benchmarks(result(), baseline, {"coverage": BenchTolerance(absolute=0.01)})
    assert not gate.passed
    assert gate.missing

    baseline = result().to_dict()
    baseline["definition"]["minutes"] = 99
    with pytest.raises(ValueError, match="definitions differ"):
        compare_benchmarks(result(), baseline, {"coverage": BenchTolerance(absolute=0.01)})


def test_benchmark_tolerances_are_explicit_and_valid():
    with pytest.raises(ValueError, match="at least one"):
        compare_benchmarks(result(), result().to_dict(), {})
    with pytest.raises(ValueError, match="non-negative"):
        BenchTolerance(absolute=-0.1)
