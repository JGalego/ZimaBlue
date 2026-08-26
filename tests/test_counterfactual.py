"""Deterministic counterfactual replay."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb


@pytest.fixture
def baseline():
    return (
        zb.Simulation(
            pool="rectangular",
            robot="tracked",
            dirt="clean",
            controller="baseline_coverage",
            seed=31,
            record=True,
        )
        .run(seconds=2.0)
        .require_recording()
    )


def test_same_policy_reproduces_the_recorded_trajectory(baseline):
    result = zb.run_counterfactual(baseline, "baseline_coverage")

    assert not result.diverged
    assert result.divergence_time is None
    assert result.trajectory_rms == pytest.approx(0.0)
    for channel in ("time", "x", "y", "heading", "cmd_left", "cmd_right"):
        assert np.array_equal(baseline.column(channel), result.alternative.column(channel))


def test_alternative_policy_has_reproducible_divergence_and_deltas(baseline):
    first = zb.run_counterfactual(baseline, "random_bounce")
    second = zb.run_counterfactual(baseline, "random_bounce")

    assert first.diverged
    assert first.divergence_time is not None
    assert first.trajectory_rms > 0.0
    assert first.metric_deltas == second.metric_deltas
    assert np.array_equal(first.alternative.column("x"), second.alternative.column("x"))
    assert first.alternative.manifest["counterfactual"]["baseline_seed"] == 31
    assert "trajectory RMS" in first.summary()


def test_counterfactual_preserves_an_explicit_initial_pose():
    pool = zb.make_pool("rectangular")
    point = pool.navigable.representative_point()
    pose = (point.x, point.y, 0.4)
    baseline = (
        zb.Simulation(
            pool=pool,
            dirt="clean",
            seed=4,
            start_pose=pose,
        )
        .run(seconds=0.2)
        .require_recording()
    )

    result = zb.run_counterfactual(baseline, "baseline_coverage")

    assert not result.diverged
    assert baseline.manifest["start_pose"] == pytest.approx(pose)


def test_counterfactual_rejects_hardware_pose_as_ground_truth(baseline):
    baseline.manifest["ground_truth"] = False
    with pytest.raises(ValueError, match="ground-truth"):
        zb.run_counterfactual(baseline, "random_bounce")
