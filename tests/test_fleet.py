"""Several robots in one pool.

The tests that matter here are the ones a single-robot suite cannot have: that
the robots are really in the same water rather than in parallel copies of it,
that what they tell each other is limited to what they could actually know, and
that dividing a pool five different ways produces five measurably different
divisions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import zimablue as zb
from zimablue.fleet import Blackboard, Fleet, spread_poses
from zimablue.planners import PARTITIONS, make_partition, mstc, partitioned
from zimablue.planners.cooperative import _arc

POOL = "rectangular"


# ======================================================================
# placing a fleet
# ======================================================================
def test_start_poses_are_inside_the_pool_and_apart():
    pool = zb.make_pool("kidney")
    robot = zb.make_robot("tracked")
    poses = spread_poses(pool, robot, 4)
    assert len(poses) == 4
    for x, y, _ in poses:
        assert pool.navigable.contains(zb.Pool and _point(x, y))
    gaps = [
        math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(poses) for b in poses[i + 1 :]
    ]
    assert min(gaps) > 4 * robot.radius, "robots placed on top of each other"


def _point(x, y):
    from shapely.geometry import Point

    return Point(x, y)


def test_a_pool_too_small_for_the_fleet_says_so():
    from shapely.geometry import box

    tiny = zb.Pool(boundary=box(0, 0, 1.2, 1.0), depth=1.2, name="puddle")
    with pytest.raises(ValueError, match="too small"):
        spread_poses(tiny, zb.make_robot("tracked"), 8)


def test_spread_poses_is_deterministic():
    pool, robot = zb.make_pool("kidney"), zb.make_robot("tracked")
    assert spread_poses(pool, robot, 3) == spread_poses(pool, robot, 3)


# ======================================================================
# they are in the same pool
# ======================================================================
def test_robots_cannot_drive_through_each_other():
    """The property that makes a fleet a fleet rather than N replays."""
    fleet = Fleet(pool=POOL, robots=3, controllers="random_bounce", seed=2)
    result = fleet.run(seconds=45)
    frames = result.recording.frames
    radius = fleet.robots[0].radius
    for a in range(3):
        for b in range(a + 1, 3):
            gap = np.hypot(
                frames[f"r{a}.x"] - frames[f"r{b}.x"], frames[f"r{a}.y"] - frames[f"r{b}.y"]
            )
            # A penetration-based resolver allows a sliver of overlap for one
            # tick before pushing out; a whole radius of it would be a robot
            # driving through another.
            assert gap.min() > 1.2 * radius, (
                f"r{a} and r{b} overlapped by {2 * radius - gap.min():.2f} m"
            )


def test_the_dirt_is_shared_not_copied():
    """Two robots over the same patch must not each collect a full load.

    If every robot had its own dirt field the team would remove twice what is
    there, and the fleet's dirt number would be a straight sum of independent
    runs. It is the single easiest thing to get wrong when composing backends.
    """
    solo = zb.Simulation(pool=POOL, dirt="autumn", controller="random_bounce", seed=4).run(
        minutes=1
    )
    pair = Fleet(pool=POOL, robots=2, dirt="autumn", controllers="random_bounce", seed=4).run(
        minutes=1
    )
    assert pair.metrics.team.dirt_removed_fraction <= 1.0
    assert pair.metrics.team.dirt_removed_fraction > solo.metrics.dirt_removed_fraction
    initial = pair.world.dirt.initial_mass
    assert pair.world.dirt.total_mass >= 0.0
    assert pair.metrics.team.dirt_removed <= initial + 1e-6


def test_a_team_mate_shows_up_on_the_sonar():
    from zimablue.sensors.models import _range_to_discs

    angles = np.array([0.0, np.pi / 2])
    ranges = _range_to_discs(0.0, 0.0, angles, ((2.0, 0.0, 0.3),))
    assert ranges[0] == pytest.approx(1.7)
    assert not np.isfinite(ranges[1]), "a beam pointing away should see nothing"


def test_team_coverage_is_the_union_and_never_less_than_a_member():
    result = Fleet(pool=POOL, robots=3, controllers="frontier", seed=1).run(minutes=1)
    metrics = result.metrics
    assert metrics.team.coverage >= max(m.coverage for m in metrics.robots)
    assert 0.0 <= metrics.overlap <= 1.0
    assert 0.0 < metrics.balance <= 1.0
    assert metrics.team.distance_traveled == pytest.approx(sum(s.distance for s in result.states))


def test_the_run_is_reproducible():
    a = Fleet(pool=POOL, robots=2, controllers="bsa", seed=7).run(seconds=40)
    b = Fleet(pool=POOL, robots=2, controllers="bsa", seed=7).run(seconds=40)
    assert a.metrics.team.coverage == b.metrics.team.coverage
    assert np.array_equal(a.visits_by_robot[1], b.visits_by_robot[1])


# ======================================================================
# the radio
# ======================================================================
def test_the_blackboard_carries_estimates_and_range_limits_them():
    board = Blackboard(3, comms_range=2.0)
    board.publish(0, 0.0, 0.0, 0.0)
    board.publish(1, 1.0, 0.0, 0.0)
    board.publish(2, 9.0, 0.0, 0.0)
    heard = {peer.index for peer in board.peers(0)}
    assert heard == {1}, "robot 2 is seven metres out of range"
    assert board.dropped == 1


def test_an_unlimited_radio_hears_everyone():
    board = Blackboard(3)
    for index in range(3):
        board.publish(index, index * 20.0, 0.0, 0.0)
    assert len(board.peers(0)) == 2
    assert board.dropped == 0


def test_sharing_coverage_removes_it_from_everyone_elses_list():
    """The one-line change that makes every online planner cooperative."""
    fleet = Fleet(pool=POOL, robots=2, controllers="frontier", seed=1, share=True)
    fleet.run(seconds=40)
    first, second = fleet.controllers
    assert first.done, "robot 0 covered nothing"
    assert first.done <= second._peer_done, "robot 1 never heard about robot 0's work"
    assert second.done <= first._peer_done, "and the other way round"
    # What sharing is for: a cell a team-mate has done is not a candidate.
    theirs = next(iter(first.done - second.done), None)
    if theirs is not None:
        assert not second.candidate(theirs), "robot 1 would still drive to robot 0's cell"


def test_a_fleet_of_strangers_hears_nothing():
    """share=False still collides and still shares the dirt -- it just does not
    tell anyone anything. The baseline every cooperative method has to beat."""
    fleet = Fleet(pool=POOL, robots=2, controllers="bsa", seed=1, share=False)
    result = fleet.run(seconds=40)
    assert result.metrics.team.coverage > 0.02
    for controller in fleet.controllers:
        assert controller.done, "a robot covered nothing"
        assert not controller._peer_done, "a stranger heard about someone else's work"


# ======================================================================
# building the team
# ======================================================================
def test_every_robot_gets_its_own_controller():
    fleet = Fleet(pool=POOL, robots=3, controllers="bsa", seed=1)
    assert len({id(c) for c in fleet.controllers}) == 3


def test_one_controller_object_cannot_drive_a_fleet():
    """It holds one map, one estimator and one plan."""
    shared = zb.CONTROLLERS.create("bsa")
    with pytest.raises(ValueError, match="cannot drive a fleet"):
        Fleet(pool=POOL, robots=2, controllers=shared, seed=1)
    with pytest.raises(ValueError, match="own controller instance"):
        Fleet(pool=POOL, robots=2, controllers=[shared, shared], seed=1)


def test_a_mixed_fleet_is_allowed():
    fleet = Fleet(pool=POOL, robots=2, controllers=["bsa", "frontier"], seed=1)
    assert [c.name for c in fleet.controllers] == ["bsa", "frontier"]


def test_the_wrong_number_of_controllers_is_an_error():
    with pytest.raises(ValueError, match="3 controllers for 2 robots"):
        Fleet(pool=POOL, robots=2, controllers=["bsa", "bsa", "bsa"], seed=1)


def test_a_controller_that_needs_truth_gets_it():
    """Otherwise a partitioned fleet raises on its first tick."""
    fleet = Fleet(pool=POOL, robots=2, controllers=partitioned("strips", "boustrophedon"), seed=1)
    assert fleet.expose_truth


# ======================================================================
# the recording
# ======================================================================
def test_the_flat_channels_mirror_robot_zero():
    """Which is what lets every single-robot tool open a fleet recording."""
    result = Fleet(pool=POOL, robots=2, controllers="bsa", seed=1).run(seconds=30)
    frames = result.recording.frames
    assert np.array_equal(frames["x"], frames["r0.x"])
    assert np.array_equal(frames["heading"], frames["r0.heading"])
    assert not np.array_equal(frames["r0.x"], frames["r1.x"])
    assert result.recording.manifest["fleet"]["count"] == 2


def test_a_fleet_recording_replays(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from zimablue.replay.renderer import ReplayRenderer

    result = Fleet(pool=POOL, robots=3, controllers="bsa", seed=1).run(seconds=30)
    path = result.save(tmp_path / "fleet.zbr")
    renderer = ReplayRenderer(zb.Recording.load(path))
    assert renderer.fleet_size == 3
    assert len(renderer._crew) == 2, "the other two robots should be drawn"
    renderer.draw(10)


# ======================================================================
# partitioning
# ======================================================================
@pytest.mark.parametrize("method", PARTITIONS.names())
def test_every_partitioner_covers_the_pool_in_connected_pieces(method):
    pool = zb.make_pool("kidney")
    robots = [zb.make_robot("tracked") for _ in range(3)]
    poses = spread_poses(pool, robots[0], 3)
    division = make_partition(method).divide(pool, robots, poses, cell=0.5)

    assert len(division) == 3
    assert all(t.cells > 0 for t in division.territories), f"{method} left a robot with nothing"
    # Every reachable cell belongs to exactly one robot.
    owned = sum(t.mask.astype(int) for t in division.territories)
    assert owned.max() <= 1
    assert owned.sum() == int((division.labels >= 0).sum())
    assert division.connected, f"{method} split a territory across the pool"


def test_darp_divides_more_evenly_than_voronoi():
    """The claim the paper is about, on the pool that shows it.

    A Voronoi cut of a kidney gives the robot in the waist a little over half
    what the robot in a lobe gets. DARP iterates until they are within a few
    percent, which is the entire difference between the two methods.
    """
    pool = zb.make_pool("kidney")
    robots = [zb.make_robot("tracked") for _ in range(3)]
    poses = spread_poses(pool, robots[0], 3)
    rough = make_partition("voronoi").divide(pool, robots, poses, cell=0.5)
    even = make_partition("darp").divide(pool, robots, poses, cell=0.5)
    assert rough.fairness < 0.6
    assert even.fairness > 0.85
    assert even.fairness > rough.fairness + 0.25


def test_geodesic_and_voronoi_disagree_on_a_concave_pool():
    """Straight-line distance crosses the concavity; the robot cannot."""
    pool = zb.make_pool("kidney")
    robots = [zb.make_robot("tracked") for _ in range(3)]
    poses = spread_poses(pool, robots[0], 3)
    straight = make_partition("voronoi").divide(pool, robots, poses, cell=0.5)
    around = make_partition("geodesic").divide(pool, robots, poses, cell=0.5)
    differing = int((straight.labels != around.labels).sum())
    assert differing > 0, "the two measures should not agree on a kidney"


def test_a_territory_is_a_pool_a_planner_can_sweep():
    pool = zb.make_pool("rectangular")
    robots = [zb.make_robot("tracked") for _ in range(2)]
    poses = spread_poses(pool, robots[0], 2)
    division = make_partition("strips").divide(pool, robots, poses, cell=0.5)
    from zimablue.planners import make_planner

    for territory in division.territories:
        share = territory.as_pool(pool)
        plan = make_planner("boustrophedon").plan(share, robots[0])
        assert len(plan) > 2
        assert plan.length * robots[0].swath_width > 0.6 * share.navigable.area


def test_a_partitioned_fleet_overlaps_less_than_an_unpartitioned_one():
    """The whole argument for partitioning, measured.

    Three robots all running the same greedy rule cover the pool in plaid.
    Give each one a region and the overlap collapses to the seams.
    """
    shared = Fleet(pool="kidney", robots=3, controllers="frontier", seed=1).run(minutes=3)
    divided = Fleet(
        pool="kidney", robots=3, controllers=partitioned("darp", "boustrophedon"), seed=1
    ).run(minutes=3)
    assert divided.metrics.overlap < shared.metrics.overlap


# ======================================================================
# cooperative methods
# ======================================================================
def test_mstc_arcs_tile_the_circuit_exactly_once():
    """Cutting a closed tour into k arcs must not lose or duplicate a cell."""
    pool = zb.make_pool("rectangular")
    robots = [zb.make_robot("tracked") for _ in range(3)]
    poses = spread_poses(pool, robots[0], 3)
    team = mstc(backtracking=False)(pool, robots, poses)
    total = sum(len(follower.planner.waypoints) for follower in team)
    circuit = team[0].circuit
    assert total == len(circuit), f"{total} waypoints across the arcs, {len(circuit)} in the tour"


def test_an_arc_wraps_around_the_end_of_the_circuit():
    circuit = np.arange(10).reshape(10, 1).astype(float)
    assert _arc(circuit, 8, 12).ravel().tolist() == [8.0, 9.0, 0.0, 1.0]


def test_backtracking_mstc_shares_out_the_work_that_plain_mstc_leaves():
    """The paper's own point, measured.

    Where the robots happen to sit on the circuit decides how long an arc each
    gets, and nothing balances that. Plain MSTC therefore has a robot that
    finishes in a minute and sits there; the backtracking variant sends it to
    help, and the fleet's balance goes from terrible to reasonable.
    """
    plain = Fleet(pool="kidney", robots=3, controllers=mstc(backtracking=False), seed=1).run(
        minutes=4
    )
    helping = Fleet(pool="kidney", robots=3, controllers=mstc(backtracking=True), seed=1).run(
        minutes=4
    )
    assert helping.metrics.balance > plain.metrics.balance
    assert helping.metrics.team.coverage > plain.metrics.team.coverage


def test_the_auction_stops_two_robots_chasing_one_cell():
    board = Blackboard(2)
    board.publish(0, 0.0, 0.0, 0.0, extras={"bid_row": 5.0, "bid_col": 5.0, "bid_cost": 3.0})
    board.publish(1, 1.0, 0.0, 0.0)

    bidder = zb.CONTROLLERS.create("auction")
    bidder.reset(zb.make_robot("tracked"))
    bidder.attach_fleet(index=1, blackboard=board, origin=(0.0, 0.0, 0.0), fleet_size=2)
    assert bidder._contested((5, 5), 9.0), "a cheaper bid should win the cell"
    assert not bidder._contested((5, 5), 1.0), "a cheaper bid of ours should take it back"
    assert not bidder._contested((4, 4), 9.0), "a different cell is not contested"


def test_the_swarm_field_pushes_back_where_a_team_mate_is():
    controller = zb.CONTROLLERS.create("binn_swarm")
    controller.reset(zb.make_robot("tracked"))
    middle = (controller.map.origin, controller.map.origin)
    from zimablue.controllers.systematic import MapCell

    controller.map.grid[middle[0] - 6 : middle[0] + 6, middle[1] - 6 : middle[1] + 6] = MapCell.FREE
    quiet = controller._input()[middle[0] + 3, middle[1] + 3]
    controller._peer_cells = {(middle[0] + 3, middle[1] + 3)}
    loud = controller._input()[middle[0] + 3, middle[1] + 3]
    assert quiet > 0 and loud < 0, "a team-mate should inhibit, not attract"


def test_the_swarm_spectrum_adds_up_what_the_fleet_has_done():
    """Multi-agent SMC shares one time-average; that is the whole extension."""
    fleet = Fleet(pool=POOL, robots=2, controllers="smc_swarm", seed=1)
    fleet.run(seconds=60)
    first = fleet.controllers[0]
    peers = fleet.blackboard.peers(0)
    assert peers and "spectrum" in peers[0].extras
    assert np.shape(peers[0].extras["spectrum"]) == np.shape(first._sum)


# ======================================================================
# pictures
# ======================================================================
def test_the_fleet_plots_draw():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from zimablue.fleetplots import plot_fleet

    result = Fleet(pool=POOL, robots=3, controllers="bsa", seed=1).run(seconds=40)
    figure = plot_fleet(result)
    assert len(figure.axes) >= 4


def test_the_territory_map_only_names_robots_that_were_there():
    result = Fleet(pool=POOL, robots=2, controllers="bsa", seed=1).run(seconds=30)
    territory = result.territory
    assert territory.min() >= -1
    assert territory.max() <= 1
    assert (territory >= 0).sum() > 0
