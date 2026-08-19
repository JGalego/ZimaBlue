"""A bump is a measurement: wall touches correct the follower's estimate."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.estimation import EstimatorConfig, PoseEstimator
from zimablue.planners import PathFollower


def test_a_wall_update_pins_only_the_normal_direction():
    estimator = PoseEstimator(EstimatorConfig(), origin=(0.0, 0.0, 0.0))
    # Let the filter accumulate some position uncertainty to correct.
    for _ in range(200):
        estimator.predict(0.5, 0.0, 0.02)
    before = estimator.estimate
    # The touched wall says the centre sits on the line x = before.x + 0.4.
    estimator.wall_update((before.x + 0.4, before.y), (1.0, 0.0), sigma=0.05)
    after = estimator.estimate
    moved = after.x - before.x
    assert moved > 0, "the estimate should move toward the wall line"
    # How far is the filter's business -- a confident filter treats a large
    # innovation with suspicion. What the update must never do is drag the
    # estimate sideways along the wall.
    assert after.y == pytest.approx(before.y, abs=1e-9)
    assert after.heading == pytest.approx(before.heading, abs=1e-6)


def run_follower(relocalise, minutes=8.0):
    sim = zb.Simulation(
        pool="rectangular",
        dirt="autumn",
        controller=PathFollower("boustrophedon", relocalise=relocalise),
        expose_truth=True,
        seed=5,
    )
    result = sim.run(minutes=minutes)
    recording = result.require_recording()
    est_x = np.asarray(recording.column("ctl.est_x"), float)
    est_y = np.asarray(recording.column("ctl.est_y"), float)
    x = np.asarray(recording.column("x"), float)
    y = np.asarray(recording.column("y"), float)
    error = np.hypot(est_x - x, est_y - y)
    return float(np.nanmean(error)), recording


def test_wall_touches_keep_the_estimate_bounded():
    blind, _ = run_follower(relocalise=False)
    fixed, recording = run_follower(relocalise=True)
    assert fixed < blind * 0.5, (blind, fixed)
    assert float(recording.column("ctl.fixes")[-1]) > 0


def test_fixes_are_edge_triggered_not_per_tick():
    _, recording = run_follower(relocalise=True, minutes=5.0)
    fixes = float(recording.column("ctl.fixes")[-1])
    scraping_ticks = int(
        (np.asarray(recording.column("contact.front"), float) > 0.5).sum()
        + (np.asarray(recording.column("contact.left"), float) > 0.5).sum()
    )
    assert 0 < fixes < max(scraping_ticks, 1), "one fix per touch, not one per tick"


def test_the_estimator_sigma_is_reported():
    _, recording = run_follower(relocalise=True, minutes=2.0)
    sigma = np.asarray(recording.column("ctl.est_sigma"), float)
    assert np.nanmax(sigma) > 0
