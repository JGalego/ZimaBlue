"""Dynamical-systems analysis.

Two kinds of test here, and the distinction matters. Some feed a *synthetic*
signal with a known answer -- a perfectly periodic section, a uniform
trajectory -- because that is the only way to know a detector detects. The rest
run real simulations and check the analysis says something structurally true
about them: that the mushroom's stem is found, that a parked robot scores
worse, that identical runs do not diverge.

No test here pins a measured number. Those move when the estimator or the
controller is touched, and a suite full of them becomes a suite everybody
disables.
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.dynamics import (
    ErgodicScore,
    ReturnMap,
    divergence,
    ergodic_score,
    forecast_cleaning,
    occupancy,
    return_map,
    target_measure,
    transfer_operator,
)
from zimablue.dynamics.returnmap import _counterclockwise


@pytest.fixture(scope="module")
def kidney():
    return (
        zb.Simulation(pool="kidney", dirt="autumn", controller="baseline_coverage", seed=7)
        .run(minutes=12)
        .recording
    )


@pytest.fixture(scope="module")
def bouncing():
    return (
        zb.Simulation(pool="rectangular", controller="random_bounce", seed=3)
        .run(minutes=12)
        .recording
    )


# ======================================================================
# Return map
# ======================================================================
def periodic_section(period: int = 3, cycles: int = 8, decay: float = 1.0) -> ReturnMap:
    """A section that repeats exactly every ``period`` contacts.

    The only way to know the detector detects. ``decay`` scales the wobble
    around the orbit each cycle: below 1 the orbit attracts, above 1 it
    repels, and 1 is neutral.
    """
    base_s = np.array([2.0, 7.0, 13.0])[:period]
    base_theta = np.array([0.3, -0.6, 0.1])[:period]
    wobble = 0.4
    s, theta = [], []
    for cycle in range(cycles):
        offset = wobble * decay**cycle
        s.extend(base_s + offset)
        theta.extend(base_theta + 0.02 * offset)
    return ReturnMap(
        s=np.asarray(s),
        theta=np.asarray(theta),
        time=np.arange(len(s), dtype=float) * 12.0,
        perimeter=30.0,
        source="synthetic",
    )


def test_a_perfectly_periodic_section_is_detected():
    orbits = periodic_section(period=3, cycles=8).periodic_orbits()
    assert orbits, "a section that repeats every three contacts should be found"
    assert orbits[0].period == 3
    assert orbits[0].repeats >= 5


def test_a_shrinking_orbit_reads_as_attracting_and_a_growing_one_as_repelling():
    """The sign of the multiplier is the whole diagnostic. Backwards, it would
    report every trap as safe."""
    attracting = periodic_section(period=3, cycles=10, decay=0.6).periodic_orbits()
    repelling = periodic_section(period=3, cycles=10, decay=1.5).periodic_orbits()
    assert attracting and attracting[0].attracting
    assert repelling and not repelling[0].attracting
    assert attracting[0].multiplier < 1.0 < repelling[0].multiplier


def test_a_scattered_section_has_no_periodic_orbits():
    rng = np.random.default_rng(0)
    scattered = ReturnMap(
        s=rng.uniform(0, 30, 80),
        theta=rng.uniform(-np.pi, np.pi, 80),
        time=np.arange(80, dtype=float) * 10.0,
        perimeter=30.0,
    )
    assert scattered.periodic_orbits() == []
    assert scattered.trapped_fraction() == 0.0


def test_the_shortest_period_wins():
    """A period-3 orbit also satisfies the period-6 and period-9 tests. Without
    de-duplication every real loop is reported once per multiple of itself."""
    periods = {o.period for o in periodic_section(period=3, cycles=12).periodic_orbits()}
    assert periods == {3}


def test_arc_length_wraps_around_the_perimeter():
    """Contacts either side of the seam are neighbours, not opposites."""
    section = ReturnMap(
        s=np.array([0.1, 29.9]),
        theta=np.array([0.0, 0.0]),
        time=np.array([0.0, 10.0]),
        perimeter=30.0,
    )
    assert section.separation(0, 1) < 0.02


def test_contacts_are_debounced(bouncing):
    """A bumping robot re-triggers its switches within a fraction of a second.

    69% of raw rising edges in a real run land less than half a second after
    the previous one. Counting those as separate arrivals filled the section
    with zero-second "periodic orbits" -- chatter dressed as dynamics.
    """
    raw = return_map(bouncing, debounce=0.0, min_travel=0.0)
    debounced = return_map(bouncing)
    assert len(debounced) < len(raw)
    gaps = np.diff(debounced.time)
    assert gaps.min() > 0.1, "a debounced section should have no near-simultaneous contacts"


def test_a_contact_far_along_the_wall_survives_the_debounce():
    """In a narrow channel the robot really does touch opposite walls a moment
    apart, and both are real arrivals."""
    quick = return_map(
        zb.Simulation(pool="mushroom", controller="random_bounce", seed=2).run(minutes=6).recording,
        debounce=5.0,
        min_travel=0.6,
    )
    slow = return_map(
        zb.Simulation(pool="mushroom", controller="random_bounce", seed=2).run(minutes=6).recording,
        debounce=5.0,
        min_travel=1e6,
    )
    assert len(quick) > len(slow)


def test_incidence_is_measured_from_the_outward_normal(kidney):
    """0 is driving into the wall and +/-90 is sliding along it.

    Two earlier conventions put every contact in the same half of the range:
    against the tangent, and against the *inward* normal -- a robot arriving at
    a wall points out of the pool, so those all read near 180 degrees.
    """
    section = return_map(kidney)
    degrees = np.degrees(section.theta)
    assert np.median(np.abs(degrees)) < 75, "contacts should cluster toward head-on, not sideways"
    assert (np.abs(degrees) < 30).mean() > 0.15, "some contacts should be near head-on"


def test_boundary_winding_is_detected():
    from shapely.geometry import Polygon

    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _counterclockwise(Polygon(square).exterior)
    assert not _counterclockwise(Polygon(square[::-1]).exterior)


def test_a_run_that_never_touches_a_wall_gives_an_empty_section():
    section = return_map(
        zb.Simulation(pool="rectangular", controller="baseline_coverage", seed=1)
        .run(seconds=2)
        .recording
    )
    assert len(section) >= 0
    assert section.periodic_orbits() == []


def test_a_recording_without_contacts_is_a_clear_error(kidney):
    stripped = zb.Recording(manifest=kidney.manifest, frames=dict(kidney.frames))
    del stripped.frames["contacts"]
    with pytest.raises(KeyError, match="no contact channel"):
        return_map(stripped)


def test_the_recurrence_matrix_is_symmetric_with_a_full_diagonal(bouncing):
    matrix = return_map(bouncing).recurrence_matrix()
    assert np.array_equal(matrix, matrix.T)
    assert matrix.diagonal().all()


# ======================================================================
# Transfer operator
# ======================================================================
@pytest.fixture(scope="module")
def mushroom_operator():
    runs = [
        zb.Simulation(pool="mushroom", controller="baseline_coverage", seed=s)
        .run(minutes=12)
        .recording
        for s in (1, 2, 3)
    ]
    return transfer_operator(runs, cell=0.5, lag=10.0)


def test_the_matrix_is_row_stochastic(mushroom_operator):
    assert np.allclose(mushroom_operator.matrix.sum(axis=1), 1.0)
    assert (mushroom_operator.matrix >= 0).all()


def test_the_invariant_measure_is_a_probability_distribution(mushroom_operator):
    measure = mushroom_operator.invariant_measure()
    assert measure.sum() == pytest.approx(1.0)
    assert (measure >= 0).all()


def test_the_leading_eigenvalue_is_one(mushroom_operator):
    """A row-stochastic matrix has one by construction. If it does not, the
    normalisation is broken and every other number here is wrong."""
    assert abs(mushroom_operator.eigenvalues[0]) == pytest.approx(1.0, abs=1e-9)


def test_unvisited_cells_are_dropped_rather_than_made_absorbing():
    """Each self-looping cell contributes an eigenvalue of exactly 1.

    Twenty-one unreachable cells in an L-shaped pool put twenty-one spurious
    ones at the top of the spectrum, and the second eigenvalue -- the entire
    point of the exercise -- vanished underneath them.
    """
    runs = [
        zb.Simulation(pool="l_shaped", controller="baseline_coverage", seed=s)
        .run(minutes=12)
        .recording
        for s in (1, 2)
    ]
    operator = transfer_operator(runs, cell=0.75, lag=10.0)
    near_one = int((np.abs(operator.eigenvalues) > 1.0 - 1e-9).sum())
    assert near_one == 1, f"{near_one} eigenvalues at 1: unvisited cells are still in the matrix"
    assert operator.unvisited > 0, "this pool at this cell size should have unreached cells"


def test_the_mushroom_split_isolates_the_bottom_of_the_stem(mushroom_operator):
    """The headline claim: the operator finds the trap without being told.

    It does not find the *geometric* neck, and that is the interesting part.
    The pool's neck is at y = 3.25, and the partition falls at about y = 2.7 --
    lower down, inside the stem. Which is right: the robot moves in and out of
    the top of the stem perfectly well, and it is the bottom it struggles to
    leave. Almost-invariant sets look for where the traffic is thin, not for
    where the walls are, and the two are not the same place.

    An earlier version of this test demanded the partition match the geometry
    and failed at 72% agreement. The 28% was the answer.
    """
    labels = mushroom_operator.almost_invariant_sets(2)
    assert len(np.unique(labels)) == 2

    counts = np.bincount(labels)
    trapped = int(np.argmin(counts))
    ys = mushroom_operator.centres[labels == trapped][:, 1]

    assert counts[trapped] < 0.4 * len(labels), "the trap should be the smaller region"
    assert ys.max() < 3.0, "every cell of it should be inside the stem, below the cap"
    assert mushroom_operator.leak_rate(labels)[trapped] < 0.15, (
        "a region the robot leaves freely is not a trap"
    )


def test_the_leak_rate_is_a_fraction(mushroom_operator):
    labels = mushroom_operator.almost_invariant_sets(2)
    for _, rate in mushroom_operator.leak_rate(labels).items():
        assert 0.0 <= rate <= 1.0


def test_a_slow_mixer_is_flagged_as_unreliable():
    """A twenty-minute run reporting a two-hour mixing time is saying it never
    got there. That is worth knowing and is not a measurement of two hours."""
    short = [
        zb.Simulation(pool="l_shaped", controller="baseline_coverage", seed=1)
        .run(minutes=3)
        .recording
    ]
    operator = transfer_operator(short, cell=1.0, lag=10.0)
    assert not operator.reliable
    assert "longer than the run" in operator.summary()


def test_a_quantity_can_be_put_back_on_the_floor(mushroom_operator):
    grid = mushroom_operator.to_grid(mushroom_operator.invariant_measure())
    assert grid.shape == mushroom_operator.shape
    assert np.isnan(grid).any(), "cells outside the pool should be blank, not zero"
    assert np.nansum(grid) == pytest.approx(1.0)


def test_too_few_cells_is_a_clear_error(kidney):
    with pytest.raises(ValueError, match="use a finer cell"):
        transfer_operator([kidney], cell=40.0)


def test_no_recordings_is_a_clear_error():
    with pytest.raises(ValueError, match="no recordings"):
        transfer_operator([])


def test_one_recording_does_not_have_to_be_a_list(kidney):
    assert len(transfer_operator(kidney, cell=1.0)) > 4


# ======================================================================
# Ergodic metric
# ======================================================================
def test_a_uniform_target_sums_to_one(kidney):
    density, _ = target_measure(kidney, "uniform")
    assert density.sum() == pytest.approx(1.0)
    assert (density >= 0).all()


def test_the_dirt_target_is_not_the_uniform_one(kidney):
    dirt, _ = target_measure(kidney, "dirt")
    uniform, _ = target_measure(kidney, "uniform")
    assert not np.allclose(dirt, uniform)


def test_an_unknown_target_names_the_real_ones(kidney):
    with pytest.raises(ValueError, match="'dirt', 'remaining' or 'uniform'"):
        target_measure(kidney, "sparkliness")


def test_covering_the_pool_scores_better_than_sitting_still(kidney):
    """The most basic property the metric has to have."""
    moving = ergodic_score(kidney, target="uniform")

    parked = zb.Recording(manifest=kidney.manifest, frames=dict(kidney.frames))
    parked.frames["x"] = np.full_like(kidney.frames["x"], kidney.frames["x"][0])
    parked.frames["y"] = np.full_like(kidney.frames["y"], kidney.frames["y"][0])
    assert ergodic_score(parked, target="uniform").value > moving.value * 5


def test_the_score_is_not_monotone_and_that_is_the_point(kidney):
    """Coverage can only go up and dirt removed can only go up, so neither can
    see a robot finishing early and parking. This can."""
    frames = dict(kidney.frames)
    half = len(frames["x"]) // 2
    frames["x"] = np.concatenate([frames["x"][:half], np.full(half, frames["x"][half])])
    frames["y"] = np.concatenate([frames["y"][:half], np.full(half, frames["y"][half])])
    parks = zb.Recording(manifest=kidney.manifest, frames=frames)

    score = ergodic_score(parks, target="uniform")
    assert score.value > score.best * 1.5, "parking should undo progress"
    assert score.wasted > 0.3


def test_the_score_has_a_history_and_a_best(kidney):
    score = ergodic_score(kidney, target="dirt", samples=50)
    assert isinstance(score, ErgodicScore)
    assert score.history.size == score.times.size
    assert score.best <= score.value
    assert 0.0 <= score.wasted <= 1.0
    assert "dirt" in score.describe()


def test_more_modes_do_not_change_the_answer_much(kidney):
    """The Sobolev weighting suppresses fine modes, so the score should be
    close to converged by eight -- otherwise the weighting is not working and
    the metric is judging centimetres."""
    coarse = ergodic_score(kidney, target="uniform", modes=6).value
    fine = ergodic_score(kidney, target="uniform", modes=14).value
    assert fine == pytest.approx(coarse, rel=0.35)


# ======================================================================
# Divergence
# ======================================================================
def test_a_twin_started_at_the_reference_pose_is_the_reference():
    """The simulator is bit-reproducible, so a twin with no displacement at all
    must stay on top of the reference for the whole run. If it does not, every
    exponent this module reports is measuring arithmetic noise.

    This caught a real bug. The start pose was being read off frame zero of the
    recording -- which is written *after* the first step, and in float32 -- so
    every twin began a third of a millimetre downstream of where it should
    have, and the divergence being measured partly predated the perturbation.
    """
    baseline = zb.Simulation(pool="rectangular", controller="baseline_coverage", seed=0)
    reference = baseline.run(minutes=3).recording
    twin = (
        zb.Simulation(
            pool="rectangular",
            controller="baseline_coverage",
            seed=0,
            start_pose=baseline.start_pose,
        )
        .run(minutes=3)
        .recording
    )
    assert np.array_equal(reference.frames["x"], twin.frames["x"])
    assert np.array_equal(reference.frames["y"], twin.frames["y"])


def test_a_perturbed_run_eventually_differs():
    run = divergence(
        controller="baseline_coverage", pool="kidney", minutes=8, runs=4, epsilon=1e-3, seed=3
    )
    assert run.separation.max() > 0.05
    assert 0.0 <= run.diverged <= 1.0
    assert run.exponent() >= 0.0


def test_the_typical_separation_is_a_median_not_a_mean():
    """The twins come out bimodal -- some glued to the reference, some at the
    far wall -- and a geometric mean of those reports a value no twin ever
    had."""
    run = divergence(
        controller="baseline_coverage", pool="kidney", minutes=6, runs=4, epsilon=1e-3, seed=3
    )
    assert run.typical.shape == run.time.shape
    assert (run.typical >= run.separation.min(axis=0) - 1e-12).all()
    assert (run.typical <= run.separation.max(axis=0) + 1e-12).all()


def test_a_bad_epsilon_is_a_clear_error():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        divergence(epsilon=0.0, minutes=1)
    with pytest.raises(ValueError, match="at least one twin"):
        divergence(runs=0, minutes=1)


# ======================================================================
# Averaging
# ======================================================================
def test_occupancy_accounts_for_the_whole_run(kidney):
    """Every tick puts the head over some cells, so the total seconds recorded
    is the run length times the number of cells under the head."""
    seconds = occupancy(kidney)
    assert seconds.sum() > kidney.duration, "the head is wider than one cell"
    assert (seconds >= 0).all()


def test_occupancy_grows_with_the_window(kidney):
    early = occupancy(kidney, until=kidney.duration / 4).sum()
    late = occupancy(kidney, until=kidney.duration).sum()
    assert late > early * 3


def test_a_forecast_beats_assuming_nothing_changes(kidney):
    """The weakest possible bar, and the one that has to be cleared: predicting
    the rest of the cycle should be better than predicting no cleaning at all."""
    forecast = forecast_cleaning(kidney, fit_fraction=0.3)
    flat = float(np.mean(np.abs(forecast.actual[0] - forecast.actual)) / forecast.actual[0])
    assert forecast.forecast_error < flat


def test_the_forecast_is_only_scored_on_what_it_did_not_see(kidney):
    forecast = forecast_cleaning(kidney, fit_fraction=0.3)
    assert 0.0 < forecast.fitted_from < kidney.duration
    assert forecast.rate > 0
    assert "forecast error" in forecast.describe()


def test_a_run_with_no_dirt_keyframes_is_a_clear_error(kidney):
    bare = zb.Recording(manifest=kidney.manifest, frames=dict(kidney.frames))
    with pytest.raises(ValueError, match="no dirt keyframes"):
        forecast_cleaning(bare)


# ======================================================================
# The billiard pools
# ======================================================================
@pytest.mark.parametrize("name", ["stadium", "mushroom"])
def test_the_billiard_pools_are_valid_and_navigable(name):
    pool = zb.make_pool(name)
    assert pool.boundary.is_valid
    assert pool.boundary.area > 5.0
    assert pool.navigable.area > 0
    from shapely.geometry import Point

    x, y, _ = pool.start_pose()
    assert pool.navigable.buffer(0.01).contains(Point(x, y))


def test_a_stadium_is_wider_than_it_is_tall():
    pool = zb.make_pool("stadium")
    minx, miny, maxx, maxy = pool.boundary.bounds
    assert maxx - minx > maxy - miny


def test_the_mushroom_really_has_a_stem():
    """A mushroom whose stem is as wide as its cap is a circle, and the whole
    point of the shape is the narrow neck."""
    from shapely.geometry import box

    pool = zb.make_pool("mushroom")
    minx, _, maxx, maxy = pool.boundary.bounds
    stem = pool.boundary.intersection(box(minx, 0.0, maxx, 2.0))
    cap = pool.boundary.intersection(box(minx, maxy - 2.0, maxx, maxy))
    assert stem.bounds[2] - stem.bounds[0] < 0.5 * (cap.bounds[2] - cap.bounds[0])
    assert 0.05 < stem.area / pool.boundary.area < 0.4


def test_a_cleaner_spends_longer_in_the_stem_than_its_share_of_the_floor():
    """The trap, measured. Nothing about the controller causes this -- it is
    the room."""
    from shapely.geometry import box

    pool = zb.make_pool("mushroom")
    share = pool.boundary.intersection(box(-1, -1, 10, 3.0)).area / pool.boundary.area

    fractions = []
    for seed in (1, 2, 3):
        frames = (
            zb.Simulation(pool="mushroom", controller="baseline_coverage", seed=seed)
            .run(minutes=10)
            .recording.frames
        )
        fractions.append(float((np.asarray(frames["y"]) < 3.0).mean()))
    assert np.mean(fractions) > 1.5 * share


# ======================================================================
# Plots
# ======================================================================
def test_every_plot_draws(kidney, bouncing, mushroom_operator, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from zimablue.dynamics.plots import (
        plot_divergence,
        plot_ergodic,
        plot_forecast,
        plot_return_map,
        plot_transfer,
    )

    figures = [
        plot_return_map(return_map(bouncing)),
        plot_transfer(mushroom_operator),
        plot_ergodic({"baseline": ergodic_score(kidney, samples=40)}),
        plot_forecast({"baseline": forecast_cleaning(kidney, samples=20)}),
        plot_divergence(
            {"baseline": divergence(pool="rectangular", minutes=2, runs=2, epsilon=1e-3)}
        ),
    ]
    for index, figure in enumerate(figures):
        path = tmp_path / f"{index}.png"
        figure.savefig(path)
        assert path.stat().st_size > 0
        plt.close(figure)


def test_a_plot_survives_an_empty_section(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from zimablue.dynamics.plots import plot_return_map

    empty = ReturnMap(
        s=np.zeros(0), theta=np.zeros(0), time=np.zeros(0), perimeter=20.0, source="none"
    )
    figure = plot_return_map(empty)
    figure.savefig(tmp_path / "empty.png")
    plt.close(figure)
