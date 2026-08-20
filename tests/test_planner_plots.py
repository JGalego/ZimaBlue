"""Pictures of a comparison.

Every figure here is built from a hand-made :class:`Comparison` rather than
from a real sweep. The plotting code does not care where the trials came
from, and a synthetic one lets these tests assert on the *content* of the
figure -- the tick labels, the number of panels, the Pareto front -- in a
second rather than in the twenty minutes a real comparison takes.

What is asserted is what a reader of the picture would notice: that every
planner got a row, that the cells carry the real measurement and not the
rescaled one, that a dominated planner is left off the front. Pixel output
is not compared; that would test matplotlib.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

from zimablue.planners.compare import Comparison, Trial  # noqa: E402
from zimablue.planners.plots import (  # noqa: E402
    _pareto,
    plot_comparison,
    plot_curves,
    plot_matrix,
    plot_paths,
    plot_plans,
    plot_tradeoff,
)

POOL = "rectangular"


@pytest.fixture(autouse=True)
def close_figures():
    """Every test here makes figures; leaking them warns and then eats memory."""
    yield
    plt.close("all")


def _trial(planner: str, *, seed: int = 1, pool: str = POOL, **scores: float) -> Trial:
    base = {
        "coverage": 0.8,
        "dirt": 0.7,
        "possible": 0.95,
        "evenness": 0.6,
        "gap": 1.2,
        "edges": 0.5,
        "efficiency": 0.4,
        "turning": 3.0,
        "half": 300.0,
        "ergodic": 0.02,
        "wasted": 0.3,
        "energy": 12.0,
        "thrift": 1.1,
        "trouble": 0.4,
    }
    base.update(scores)
    steps = np.linspace(0.0, 600.0, 40)
    return Trial(
        planner=planner,
        pool=pool,
        seed=seed,
        scores=base,
        # A lazy spiral: enough points that the scatter has something to
        # colour along, and it stays inside the rectangular preset.
        path=np.column_stack(
            [
                4.0 + 2.5 * np.cos(np.linspace(0, 8 * np.pi, 200)),
                3.0 + 1.8 * np.sin(np.linspace(0, 8 * np.pi, 200)),
            ]
        ),
        curve=(steps, np.clip(steps / 600.0, 0.0, 1.0) * base["coverage"]),
    )


@pytest.fixture
def comparison() -> Comparison:
    return Comparison(
        trials=[
            _trial("sweep_optimal", coverage=0.92, efficiency=0.55, half=240.0),
            _trial("boustrophedon", coverage=0.85, efficiency=0.40, half=320.0),
            _trial("random_bounce", coverage=0.60, efficiency=0.20, half=520.0),
        ],
        minutes=10.0,
    )


# ----------------------------------------------------------------------
def test_the_matrix_gives_every_planner_a_row_and_every_dimension_a_column(comparison):
    fig = plot_matrix(comparison)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == comparison.planners
    assert len(ax.get_xticklabels()) == len(comparison.dimensions)


def test_each_cell_prints_the_measurement_not_the_rescaled_rank(comparison):
    """The colour is normalised per column; the number has to stay real.

    Quoting a rescaled 1.0 as "100% coverage" is the mistake this guards.
    """
    ax = plot_matrix(comparison).axes[0]
    printed = {t.get_text() for t in ax.texts}
    assert "92.0%" in printed, printed
    assert "60.0%" in printed


def test_the_matrix_column_headers_say_which_way_is_good(comparison):
    ax = plot_matrix(comparison).axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "coverage\nhigher" in labels
    assert "worst gap\nlower" in labels


def test_the_matrix_titles_itself_with_the_comparison_label(comparison):
    comparison.label = "fleet comparison -- 3 robots"
    title = plot_matrix(comparison).axes[0].get_title()
    assert "fleet comparison -- 3 robots" in title
    assert "10 min" in title


def test_naming_a_pool_puts_it_in_the_title(comparison):
    assert POOL in plot_matrix(comparison, pool=POOL).axes[0].get_title()


def test_the_matrix_can_be_drawn_onto_a_supplied_axes(comparison):
    """plot_comparison passes its own axes in; the figure must not be replaced."""
    fig, ax = plt.subplots()
    assert plot_matrix(comparison, ax=ax) is fig


# ----------------------------------------------------------------------
def test_paths_get_one_panel_per_planner(comparison):
    fig = plot_paths(comparison)
    assert len(fig.axes) == len(comparison.planners)


def test_each_path_panel_is_titled_with_its_coverage(comparison):
    titles = [ax.get_title() for ax in plot_paths(comparison).axes]
    assert any("sweep_optimal" in t and "92%" in t for t in titles), titles


def test_paths_can_be_narrowed_to_one_seed(comparison):
    comparison.trials.append(_trial("sweep_optimal", seed=2, coverage=0.5))
    fig = plot_paths(comparison, seed=2)
    assert len(fig.axes) == 1


def test_paths_wrap_onto_more_than_one_row(comparison):
    """columns=2 with three planners is two rows -- the ceil, not the floor."""
    fig = plot_paths(comparison, columns=2)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_subplotspec().get_gridspec().nrows == 2


# ----------------------------------------------------------------------
def test_curves_draw_one_line_per_planner_and_label_it(comparison):
    ax = plot_curves(comparison).axes[0]
    lines = [line for line in ax.get_lines() if line.get_label() in comparison.planners]
    assert len(lines) == len(comparison.planners)
    assert ax.get_xlabel() == "minutes"


def test_curves_plot_minutes_and_percent_not_seconds_and_fractions(comparison):
    """The curve is stored in seconds and fractions; the axes are not."""
    ax = plot_curves(comparison).axes[0]
    line = next(line for line in ax.get_lines() if line.get_label() == "sweep_optimal")
    xs, ys = line.get_xydata().T
    assert xs.max() == pytest.approx(10.0)
    assert ys.max() == pytest.approx(92.0)


def test_a_planner_with_no_trial_in_this_pool_is_skipped_not_crashed(comparison):
    """select() comes back empty for a planner that never ran here."""
    comparison.trials.append(_trial("contour", pool="kidney"))
    ax = plot_curves(comparison, pool=POOL).axes[0]
    labels = {line.get_label() for line in ax.get_lines()}
    assert "contour" not in labels
    assert "sweep_optimal" in labels


# ----------------------------------------------------------------------
def test_the_tradeoff_annotates_every_planner(comparison):
    ax = plot_tradeoff(comparison).axes[0]
    assert {t.get_text() for t in ax.texts} == set(comparison.planners)


def test_the_tradeoff_axis_labels_say_which_way_is_better(comparison):
    ax = plot_tradeoff(comparison, x="energy", y="coverage").axes[0]
    assert "lower is better" in ax.get_xlabel()
    assert "higher is better" in ax.get_ylabel()


def test_a_planner_scoring_infinity_is_left_out_of_the_tradeoff(comparison):
    """``half`` is inf for a pool never half covered; it cannot be plotted."""
    comparison.trials.append(_trial("stalled", half=float("inf")))
    ax = plot_tradeoff(comparison, x="half", y="coverage").axes[0]
    assert "stalled" not in {t.get_text() for t in ax.texts}


def test_the_pareto_front_joins_the_planners_nobody_beats(comparison):
    """Three planners ordered on both axes leave a front of one.

    sweep_optimal wins coverage and efficiency outright, so the other two are
    dominated and there is no line to draw.
    """
    ax = plot_tradeoff(comparison).axes[0]
    assert not ax.get_lines(), "a single-point front should not be joined up"


def test_a_genuine_trade_off_draws_a_front(comparison):
    comparison.trials.append(_trial("thorough", coverage=0.99, efficiency=0.05))
    ax = plot_tradeoff(comparison).axes[0]
    assert len(ax.get_lines()) == 1


def test_pareto_keeps_the_undominated_and_sorts_them_by_x():
    points = [(1.0, 5.0, "a"), (2.0, 4.0, "b"), (1.5, 1.0, "c")]
    front = _pareto(points, +1, +1)
    assert [p[2] for p in front] == ["a", "b"]


def test_pareto_follows_the_sign_of_each_dimension():
    """With less-is-better on x, the winner is the leftmost, not the rightmost."""
    points = [(1.0, 5.0, "cheap"), (9.0, 5.0, "dear")]
    assert [p[2] for p in _pareto(points, -1, +1)] == ["cheap"]


# ----------------------------------------------------------------------
def test_plot_comparison_stacks_the_matrix_over_the_curves_and_the_tradeoff(comparison):
    fig = plot_comparison(comparison)
    assert len(fig.axes) == 3
    titles = [ax.get_title() for ax in fig.axes]
    assert any("planner comparison" in t for t in titles)
    assert any("coverage over time" in t for t in titles)
    assert any("trade-off" in t for t in titles)


# ----------------------------------------------------------------------
def test_plot_plans_draws_the_route_before_anyone_drives_it():
    fig = plot_plans(POOL, ["boustrophedon", "sweep_optimal"], columns=2)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_title(), "each panel names its planner and the plan's cost"
        assert ax.get_lines(), "each panel draws the pool outline and the waypoints"


def test_plot_plans_reports_the_length_and_the_turning_it_costs():
    title = plot_plans(POOL, ["boustrophedon"], columns=1).axes[0].get_title()
    assert "m," in title and "deg" in title


def test_plot_plans_takes_a_pool_object_as_readily_as_a_name():
    import zimablue as zb

    fig = plot_plans(zb.make_pool(POOL), ["boustrophedon"], columns=1)
    assert len(fig.axes) == 1
