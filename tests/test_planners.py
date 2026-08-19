"""Coverage path planning: the plans, the followers, and the decision rules.

The split here is deliberate.

The **decision rules** are tested without a simulator. An online planner is a
function from a grid to a next cell, and walking it over a synthetic grid tests
that function directly -- no physics, no noise, no minutes of wall clock, and a
failure points at the rule rather than at the robot. That is where the
algorithms' actual claims get checked: Spiral-STC covering a clean rectangle
with no cell twice is a theorem, so it belongs in a test.

The **whole thing** is then tested end to end, briefly, to catch the class of
bug the synthetic grid cannot see -- a controller that decides well and drives
into a wall.
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.controllers.systematic import MapCell
from zimablue.planners import PLANNERS, CoveragePath, PathFollower, make_planner
from zimablue.planners.compare import DIMENSIONS, Comparison, Trial, coverage_curve
from zimablue.planners.online import (
    CCW,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    EvidenceMap,
    _order_from,
    _turn,
)

ONLINE = (
    "spiral_stc",
    "full_stc",
    "bsa",
    "ba_star",
    "brick_and_mortar",
    "binn",
    "epsilon_star",
    "ppcpp",
    "frontier",
    "smc",
)


# ======================================================================
# the plan object
# ======================================================================
def test_a_path_measures_its_own_length_and_turning():
    square = CoveragePath(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]]))
    assert square.length == pytest.approx(10.0)
    assert square.turns == pytest.approx(np.pi)
    assert square.sharp_turns == 2


def test_turning_is_the_half_that_coverage_metrics_ignore():
    """Two paths of the same length, one of which a robot would hate."""
    straight = CoveragePath(np.array([[0.0, 0.0], [4.0, 0.0]]))
    zigzag = CoveragePath(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]))
    assert zigzag.length == pytest.approx(straight.length)
    assert straight.turns == 0.0
    assert zigzag.sharp_turns == 3


# ======================================================================
# offline planners
# ======================================================================
@pytest.mark.parametrize("name", PLANNERS.names())
@pytest.mark.parametrize("pool_name", ["rectangular", "kidney", "l_shaped"])
def test_every_offline_planner_covers_the_pool_it_was_given(name, pool_name):
    """The floor of what a planner has to do: a non-empty path, inside the
    pool, that passes within a swath of most of it."""
    pool = zb.make_pool(pool_name)
    robot = zb.make_robot("tracked")
    plan = make_planner(name).plan(pool, robot)

    assert len(plan) > 2, f"{name} planned {len(plan)} waypoints"
    inside = pool.navigable.buffer(0.05)
    stray = [p for p in plan.waypoints if not inside.contains(zb.Pool and _point(p))]
    assert not stray, f"{name} put {len(stray)} waypoints outside the pool"

    covered = plan.length * robot.swath_width
    assert covered > 0.8 * pool.navigable.area, (
        f"{name} sweeps {covered:.0f} m2 of a {pool.navigable.area:.0f} m2 pool"
    )


def _point(xy):
    from shapely.geometry import Point

    return Point(float(xy[0]), float(xy[1]))


def test_lane_spacing_follows_the_swath_and_not_a_constant():
    """Double the cleaning width and the plan should be about half as long.

    A planner that spaced its lanes by a hard-coded number would pass every
    other test in this file and quietly leave stripes on any robot but the
    default one.
    """
    from dataclasses import replace

    pool = zb.make_pool("rectangular")
    narrow = zb.make_robot("tracked")
    wide = zb.Cleaner(
        chassis=narrow.chassis,
        locomotion=narrow.locomotion,
        power=narrow.power,
        cleaning=replace(
            narrow.cleaning,
            brush=replace(narrow.cleaning.brush, width=2 * narrow.swath_width),
        ),
    )
    assert wide.swath_width == pytest.approx(2 * narrow.swath_width)

    long_plan = make_planner("boustrophedon").plan(pool, narrow)
    short_plan = make_planner("boustrophedon").plan(pool, wide)
    assert short_plan.length == pytest.approx(long_plan.length / 2, rel=0.25)


def test_the_optimal_sweep_is_never_worse_than_the_fixed_one():
    """The whole reason to search over the sweep angle.

    Zero degrees is one of the angles searched, so the optimiser can always
    fall back on it -- on a pool aligned with the axes it does, and the two
    plans come out identical. On the kidney it finds 35 metres for free.
    """
    robot = zb.make_robot("tracked")
    for name in ("rectangular", "kidney", "l_shaped"):
        pool = zb.make_pool(name)
        fixed = make_planner("boustrophedon").plan(pool, robot)
        searched = make_planner("sweep_optimal").plan(pool, robot)
        cost = lambda plan: plan.length + 0.35 * plan.turns  # noqa: E731
        assert cost(searched) <= cost(fixed) + 1e-6, name
    assert (
        make_planner("sweep_optimal").plan(zb.make_pool("kidney"), robot).length
        < make_planner("boustrophedon").plan(zb.make_pool("kidney"), robot).length
    )


def test_boustrophedon_decomposition_merges_what_trapezoidal_splits():
    """The difference between the two decompositions, on a shape that shows it.

    Trapezoidal decomposition cuts at every vertex; boustrophedon cuts only
    where the slice's *connectivity* changes. On an L-shaped pool nothing
    splits -- every vertical slice through an L is a single interval -- so BCD
    correctly returns one cell where trapezoidal returns eighteen, and
    eighteen cells means eighteen separate sweeps with travel between them.
    """
    pool = zb.make_pool("l_shaped")
    robot = zb.make_robot("tracked")
    merged = make_planner("boustrophedon_cells").plan(pool, robot)
    split = make_planner("trapezoidal").plan(pool, robot)
    assert len(split.cells) > len(merged.cells) >= 1
    assert merged.length < split.length


# ======================================================================
# the follower
# ======================================================================
def test_the_follower_needs_a_localisation_it_understands():
    with pytest.raises(ValueError, match="truth' or 'odometry'"):
        PathFollower("boustrophedon", localisation="gps")


def test_the_follower_says_why_it_needs_the_pool():
    follower = PathFollower("boustrophedon")
    follower.reset(zb.make_robot("tracked"))
    with pytest.raises(RuntimeError, match="expose_truth"):
        follower.step(_blank_input())


def _blank_input():
    from zimablue.controllers.base import ControlInput

    return ControlInput(
        time=0.0,
        dt=0.05,
        readings={},
        battery=1.0,
        filter_load=0.0,
        robot=zb.make_robot("tracked"),
        truth=None,
    )


def test_a_waypoint_the_hull_cannot_reach_does_not_deadlock_the_follower():
    """The regression that cost a rewrite of the pursuit rule.

    A plan's first waypoint sits in a corner of the workspace. The robot
    closes to within a lookahead of it, and the earlier follower -- which
    advanced its index only on *arrival* -- started aiming at the far end of
    the lane while still counting the corner as its target. It reversed out,
    found itself more than a lookahead away again, turned back, and paced a
    15 cm stretch of tile for the whole run at 4% coverage.
    """
    follower = PathFollower("sweep_optimal", localisation="truth")
    result = zb.Simulation(pool="rectangular", controller=follower, seed=1, expose_truth=True).run(
        minutes=6
    )
    assert follower.target > 4, "the follower is stuck on an early waypoint"
    assert result.metrics.coverage > 0.3


def test_following_a_plan_on_odometry_costs_something_and_the_gap_is_the_point():
    """Both halves of the comparison the package exists to make."""
    scores = {}
    for mode in ("truth", "odometry"):
        follower = PathFollower("sweep_optimal", localisation=mode)
        scores[mode] = (
            zb.Simulation(pool="rectangular", controller=follower, seed=1, expose_truth=True)
            .run(minutes=8)
            .metrics.coverage
        )
    assert scores["truth"] > scores["odometry"], (
        f"perfect pose should not be worse than dead reckoning: {scores}"
    )


# ======================================================================
# the online decision rules, on a synthetic grid
# ======================================================================
def _rig(name: str, *, span: int = 8, origin: int | None = None):
    """An online controller over a clean rectangle of free floor.

    No simulator: the grid is written directly and the robot is teleported
    from cell to cell. What is under test is the decision rule.
    """
    controller = zb.CONTROLLERS.create(name)
    controller.reset(zb.make_robot("tracked"))
    base = origin if origin is not None else 2 * (controller.map.origin // 2)
    controller.map.grid[:, :] = MapCell.WALL
    controller.map.grid[base : base + span, base : base + span] = MapCell.FREE
    controller.here = (base, base)
    controller.done = {(base, base)}
    controller.begin()
    return controller, base, span


def _walk(controller, limit: int = 400):
    """Follow the controller's own decisions, perfectly."""
    visited = [controller.here]
    for _ in range(limit):
        route = controller.choose(controller.here)
        if not route:
            break
        for cell in route:
            step = (cell[0] - controller.here[0], cell[1] - controller.here[1])
            if step in CCW:
                controller.facing = step
            controller.here = cell
            controller.done.add(cell)
            visited.append(cell)
    return visited


@pytest.mark.parametrize("name", [n for n in ONLINE if n != "smc"])
def test_every_online_rule_finishes_a_clean_rectangle(name):
    """Given a room with no surprises in it, all of them should finish it."""
    controller, base, span = _rig(name)
    visited = _walk(controller)
    floor = {(r, c) for r in range(base, base + span) for c in range(base, base + span)}
    missed = floor - set(visited)
    assert not missed, f"{name} left {len(missed)} of {len(floor)} cells"


def test_spiral_stc_visits_every_cell_exactly_once():
    """The theorem, as a test.

    Spiral-STC's claim is not that it covers -- anything covers eventually --
    but that circumnavigating a spanning tree of 2x2 cells passes through each
    sub-cell exactly once. On an obstacle-free rectangle there is nothing to
    stop it, so any repetition here is a bug in the walk rather than a
    concession to the room.
    """
    controller, _, span = _rig("spiral_stc")
    visited = _walk(controller)
    assert len(visited) == span * span
    assert len(set(visited)) == len(visited), "a cell was visited twice"
    assert controller._backtracks == 0, "a clean rectangle should need no backtracking"


def test_a_wall_that_clips_one_cell_breaks_spiral_stc_s_spiral():
    """This measures the published limitation, and the published fix.

    Spiral-STC will not enter a 2x2 cell unless all four sub-cells are clear,
    so one chipped corner puts the other three off the tree and the walk has
    to leave the spiral and come back for them. Full-STC enters on whichever
    sub-cell it is stepping into and never breaks stride.

    Both still finish the room -- the backtracking sees to that -- so what
    separates them is not coverage but how much of the run was spiral.
    """
    repeats = {}
    for name in ("spiral_stc", "full_stc"):
        controller, base, span = _rig(name)
        controller.map.grid[base + 4, base + 4] = MapCell.WALL
        visited = _walk(controller)
        assert len(set(visited)) == span * span - 1, f"{name} missed something"
        repeats[name] = len(visited) - len(set(visited))
    assert repeats["spiral_stc"] > repeats["full_stc"]


def test_brick_and_mortar_never_seals_a_corridor():
    """The connectivity test is the whole algorithm; without it the robot
    bricks itself in."""
    controller, base, span = _rig("brick_and_mortar")
    # A one-cell-wide neck between two rooms.
    neck = (base + 4, base + 4)
    for row in range(base, base + span):
        if row != neck[0]:
            controller.map.grid[row, base + 4] = MapCell.WALL
    controller.here = neck
    controller.done.add(neck)
    controller._seal(neck)
    assert neck not in controller.sealed, "sealing the neck cuts the floor in two"


def test_the_neural_field_is_brightest_where_the_work_is():
    """Pins the fix to BINN's integration.

    Integrating the shunting equation forward with a step large enough to
    propagate activity overshoots its own bound and rings between the ceiling
    and the floor, so on an even iteration count the *uncovered* cells report
    the lowest activity in the grid. The robot then paced between two cells
    for an entire run and covered 0.4% of the pool.
    """
    controller, base, _ = _rig("binn")
    controller._relax()
    uncovered = controller.activity[base + 4, base + 4]
    covered = controller.activity[base, base]
    wall = controller.activity[0, 0]
    assert uncovered > covered > wall
    assert uncovered > 0.5


def test_epsilon_star_leaves_the_crumb_for_the_loaf():
    """The point of the coarse levels: a big patch of work outranks a nearer
    single cell, which is what separates this from nearest-frontier."""
    controller = zb.CONTROLLERS.create("epsilon_star")
    controller.reset(zb.make_robot("tracked"))
    origin = 2 * (controller.map.origin // 2)
    controller.map.grid[:, :] = MapCell.WALL
    controller.map.grid[origin - 12 : origin + 12, origin - 12 : origin + 12] = MapCell.FREE
    controller.here = (origin, origin)
    # Everything done except one stray cell nearby and a block further off.
    controller.done = {
        (r, c) for r in range(origin - 12, origin + 12) for c in range(origin - 12, origin + 12)
    }
    stray = (origin, origin + 3)
    controller.done.discard(stray)
    block = {(r, c) for r in range(origin + 6, origin + 10) for c in range(origin - 10, origin - 6)}
    controller.done -= block
    route = controller.choose(controller.here)
    assert route, "there is work left"
    assert route[-1] in block, f"went for the crumb at {route[-1]}, not the loaf"


def test_direction_helpers_turn_the_way_they_say():
    assert _turn(EAST, 1) == NORTH
    assert _turn(EAST, -1) == SOUTH
    assert _turn(EAST, 2) == WEST
    assert _order_from(WEST) == (WEST, SOUTH, EAST, NORTH)


# ======================================================================
# the map these planners drive on
# ======================================================================
def test_one_stray_echo_is_not_a_wall_but_three_are():
    grid = EvidenceMap(cell=0.5, extent=20.0, votes=3)
    grid.mark_wall(1.0, 1.0)
    assert grid.state_at(1.0, 1.0) != MapCell.WALL
    grid.mark_wall(1.0, 1.0)
    grid.mark_wall(1.0, 1.0)
    assert grid.state_at(1.0, 1.0) == MapCell.WALL


def test_a_bump_counts_for_more_than_a_ping():
    grid = EvidenceMap(cell=0.5, extent=20.0, votes=3)
    grid.mark_wall(1.0, 1.0, weight=3)
    assert grid.state_at(1.0, 1.0) == MapCell.WALL


def test_driving_over_a_wall_proves_it_was_not_one():
    """The strongest evidence the robot has about a cell is having been in it.

    Without this the map only ever gains walls, and three minutes of echoes
    scattered by a drifting pose walled an eight-metre pool down to an eighth
    of itself.
    """
    grid = EvidenceMap(cell=0.5, extent=20.0, votes=1)
    grid.mark_wall(1.0, 1.0)
    assert grid.state_at(1.0, 1.0) == MapCell.WALL
    grid.mark_free(1.0, 1.0, 0.4)
    assert grid.state_at(1.0, 1.0) == MapCell.FREE


def test_a_beam_passing_through_a_cell_argues_against_the_wall():
    grid = EvidenceMap(cell=0.5, extent=20.0, votes=2)
    grid.mark_wall(1.0, 0.0)
    grid.mark_wall(1.0, 0.0)
    assert grid.state_at(1.0, 0.0) == MapCell.WALL
    grid.observe_ray(0.0, 0.0, 0.0, 2.0, hit=False)
    assert grid.state_at(1.0, 0.0) == MapCell.FREE


# ======================================================================
# end to end
# ======================================================================
@pytest.mark.parametrize("name", ONLINE)
def test_every_online_planner_drives_a_real_pool(name):
    """Short, but enough to catch a controller that decides well and crashes."""
    result = zb.Simulation(pool="rectangular", controller=name, seed=1).run(minutes=1.0)
    assert result.metrics.distance_traveled > 3.0, f"{name} barely moved"
    assert result.metrics.coverage > 0.03


def test_smc_drives_down_the_metric_it_is_named_after():
    """SMC's objective is the ergodic score, so the score should fall.

    Not a tautology: the controller minimises its own running estimate over
    the map it built, and this measures the finished trajectory against the
    real pool. The two agreeing is the thing worth testing.
    """
    from zimablue.dynamics import ergodic_score

    controller = zb.CONTROLLERS.create("smc")
    result = zb.Simulation(pool="rectangular", controller=controller, seed=1).run(minutes=6)
    score = ergodic_score(result.recording, target="uniform")
    early = score.history[len(score.history) // 6]
    assert score.value < early, f"ergodic score rose from {early:.3f} to {score.value:.3f}"


# ======================================================================
# the comparison
# ======================================================================
def test_the_coverage_curve_only_goes_up_and_ends_where_the_metrics_do():
    result = zb.Simulation(pool="rectangular", controller="bsa", seed=1).run(minutes=2)
    pool = zb.make_pool("rectangular")
    times, values = coverage_curve(result.recording, pool, swath=0.34)
    assert np.all(np.diff(values) >= -1e-12), "coverage cannot go down"
    assert values[-1] == pytest.approx(result.metrics.coverage, abs=0.06)
    assert times[-1] == pytest.approx(result.recording.column("time")[-1], rel=0.02)


def test_the_matrix_is_normalised_per_column_with_one_at_the_best():
    trials = [
        Trial(
            "a", "p", 1, {"coverage": 0.9, "turning": 40.0}, np.zeros((2, 2)), (np.zeros(2),) * 2
        ),
        Trial(
            "b", "p", 1, {"coverage": 0.5, "turning": 10.0}, np.zeros((2, 2)), (np.zeros(2),) * 2
        ),
    ]
    grid = Comparison(trials).matrix()
    coverage = [d.key for d in DIMENSIONS].index("coverage")
    turning = [d.key for d in DIMENSIONS].index("turning")
    assert grid[0, coverage] == 1.0 and grid[1, coverage] == 0.0
    # Turning is a cost, so the smaller number is the bright one.
    assert grid[1, turning] == 1.0 and grid[0, turning] == 0.0


def test_a_dimension_nobody_reached_takes_the_worst_value_rather_than_vanishing():
    trials = [
        Trial("a", "p", 1, {"half": 200.0}, np.zeros((2, 2)), (np.zeros(2),) * 2),
        Trial("b", "p", 1, {"half": float("inf")}, np.zeros((2, 2)), (np.zeros(2),) * 2),
    ]
    grid = Comparison(trials).matrix()
    half = [d.key for d in DIMENSIONS].index("half")
    assert grid[0, half] == 1.0 and grid[1, half] == 0.0


def test_the_table_marks_the_winner_in_every_column():
    trials = [
        Trial("a", "p", 1, {"coverage": 0.9}, np.zeros((2, 2)), (np.zeros(2),) * 2),
        Trial("b", "p", 1, {"coverage": 0.5}, np.zeros((2, 2)), (np.zeros(2),) * 2),
    ]
    table = Comparison(trials).table()
    assert "*90.0%" in table
    assert "*50.0%" not in table


def test_the_mosaic_is_a_gif_of_everyone_on_a_shared_clock(tmp_path):
    """A quick one with a pair of short runs; the asset script does the rest."""
    from PIL import Image

    from zimablue.planners.plots import export_mosaic

    recordings = {
        name: zb.Simulation(pool="rectangular", controller=name, seed=1).run(minutes=1).recording
        for name in ("bsa", "random_bounce")
    }
    out = export_mosaic(recordings, tmp_path / "mosaic.gif", columns=2, fps=5, frames=6, dpi=40)
    gif = Image.open(out)
    assert gif.n_frames == 6
    assert gif.size[0] > gif.size[1], "a pair of pool panels side by side lies wider than tall"
