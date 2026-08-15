"""Batch experiments."""

from __future__ import annotations

import json

import pytest

from zimablue.batch import run_batch
from zimablue.scenarios import load_scenario


@pytest.fixture
def scenario():
    s = load_scenario("scenarios/rectangular.yaml")
    s.duration = 30.0
    return s


def test_batch_runs_every_episode(scenario):
    result = run_batch(scenario, episodes=4)
    assert len(result.episodes) == 4
    assert [e.seed for e in result.episodes] == [42, 43, 44, 45]


def test_batch_is_reproducible(scenario):
    first = run_batch(scenario, episodes=3)
    second = run_batch(scenario, episodes=3)
    assert [e.metrics.coverage for e in first.episodes] == [
        e.metrics.coverage for e in second.episodes
    ]


def test_episodes_differ_from_each_other(scenario):
    result = run_batch(scenario, episodes=4)
    coverages = {e.metrics.coverage for e in result.episodes}
    assert len(coverages) > 1, "different seeds should give different outcomes"


def test_aggregate_statistics(scenario):
    result = run_batch(scenario, episodes=4)
    stats = result.stats("coverage")
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std"] >= 0.0
    assert 0.0 <= result.success_rate <= 1.0
    assert "mean_coverage" in result.summary()


def test_worst_episodes_are_ordered(scenario):
    result = run_batch(scenario, episodes=4)
    worst = result.worst("coverage", 3)
    assert [e.metrics.coverage for e in worst] == sorted(e.metrics.coverage for e in worst)


def test_batch_can_keep_recordings(scenario, tmp_path):
    result = run_batch(scenario, episodes=2, record_dir=tmp_path)
    assert all(e.recording_path is not None and e.recording_path.exists() for e in result.episodes)


def test_batch_json_is_reproducible_metadata(scenario, tmp_path):
    result = run_batch(scenario, episodes=2)
    data = json.loads(result.save(tmp_path / "batch.json").read_text())
    assert data["aggregate"]["episodes"] == 2
    # Enough to re-run any individual episode exactly.
    assert data["scenario"]["pool"]["preset"] == "rectangular"
    assert [e["seed"] for e in data["episodes"]] == [42, 43]


def test_batch_csv_has_a_row_per_episode(scenario, tmp_path):
    result = run_batch(scenario, episodes=3)
    lines = result.to_csv(tmp_path / "b.csv").read_text().strip().splitlines()
    assert len(lines) == 4  # header plus three rows


def test_explicit_seeds_are_honoured(scenario):
    result = run_batch(scenario, seeds=[100, 200])
    assert [e.seed for e in result.episodes] == [100, 200]
