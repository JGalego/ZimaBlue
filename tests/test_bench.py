"""The frozen benchmark suite."""

from __future__ import annotations

import dataclasses
import json

import pytest

from zimablue._version import __version__
from zimablue.bench import BENCH_QUICK, BENCH_V1, BenchDefinition, run_bench

TINY = BenchDefinition(
    name="zb-bench-tiny",
    entries=("random_bounce", "baseline_coverage"),
    pools=("rectangular",),
    seeds=(1,),
    minutes=0.5,
)


def test_the_definition_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        BENCH_V1.minutes = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"entries": ()}, "entries"),
        ({"pools": ()}, "pools"),
        ({"seeds": ()}, "seeds"),
        ({"minutes": 0.0}, "minutes"),
        ({"minutes": float("nan")}, "minutes"),
    ],
)
def test_a_definition_rejects_invalid_work(changes, message):
    values = {
        "name": "invalid",
        "entries": ("random_bounce",),
        "pools": ("rectangular",),
        "seeds": (1,),
        "minutes": 1.0,
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        BenchDefinition(**values)


def test_v1_names_its_entries_rather_than_asking_the_package():
    """The suite must not grow when the package does."""
    assert isinstance(BENCH_V1.entries, tuple)
    assert len(BENCH_V1.entries) == 21
    assert BENCH_V1.runs == 21 * 3 * 3


def test_a_bench_run_writes_all_of_its_outputs(tmp_path):
    result = run_bench(TINY)
    paths = result.save(tmp_path)

    header = json.loads(paths["json"].read_text())
    assert header["bench"] == "zb-bench-tiny"
    assert header["zimablue_version"] == __version__
    assert {t["planner"] for t in header["trials"]} == {"random_bounce", "baseline_coverage"}
    # JSON has no inf: a pool never half-covered serialises as null, not 1e999.
    halves = [t["scores"]["half"] for t in header["trials"]]
    assert all(h is None or isinstance(h, float) for h in halves)

    table = paths["markdown"].read_text()
    assert "zimablue bench" in table
    assert "`random_bounce`" in table
    assert "| planner |" in table

    csv_lines = paths["csv"].read_text().splitlines()
    assert len(csv_lines) == 1 + TINY.runs


def test_the_quick_tier_is_actually_quick():
    assert BENCH_QUICK.runs <= 3
    assert BENCH_QUICK.minutes <= 2.0
