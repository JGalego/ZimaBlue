"""The mapping controller and its occupancy grid."""

from __future__ import annotations

import numpy as np

from zimablue.controllers import SystematicCoverage
from zimablue.controllers.systematic import MapCell, OccupancyMap, Phase
from zimablue.simulation import Simulation


# -- the map ---------------------------------------------------------------
def test_map_starts_unknown():
    grid = OccupancyMap()
    assert grid.explored_cells == 0
    assert grid.state_at(0.0, 0.0) == MapCell.UNKNOWN


def test_driving_marks_free_and_covered():
    grid = OccupancyMap(cell=0.25)
    grid.mark_free(1.0, 1.0, 0.3)
    grid.mark_covered(1.0, 1.0, 0.2)
    assert grid.state_at(1.0, 1.0) == MapCell.FREE
    assert grid.covered_cells > 0


def test_walls_are_not_overwritten_by_free_space():
    """A wall seen once must not be erased by a later free-space observation.

    Otherwise the robot forgets obstacles as soon as its estimate drifts a
    little, and plans straight through them.
    """
    grid = OccupancyMap(cell=0.25)
    grid.mark_wall(1.0, 0.0)
    grid.mark_free(1.0, 0.0, 0.3)
    assert grid.state_at(1.0, 0.0) == MapCell.WALL


def test_sonar_ray_carves_free_space_and_marks_the_hit():
    grid = OccupancyMap(cell=0.25)
    grid.observe_ray(0.0, 0.0, 0.0, 2.0, hit=True)
    assert grid.state_at(1.0, 0.0) == MapCell.FREE
    assert grid.state_at(2.0, 0.0) == MapCell.WALL


def test_a_max_range_ray_marks_no_wall():
    grid = OccupancyMap(cell=0.25)
    grid.observe_ray(0.0, 0.0, 0.0, 3.0, hit=False)
    assert grid.state_at(3.0, 0.0) != MapCell.WALL


def test_frontier_finds_uncovered_free_space():
    grid = OccupancyMap(cell=0.25)
    grid.observe_ray(0.0, 0.0, 0.0, 3.0, hit=False)
    grid.mark_covered(0.0, 0.0, 0.3)
    target = grid.nearest_frontier(0.0, 0.0, min_distance=0.5)
    assert target is not None
    assert target[0] > 0.4


def test_frontier_includes_the_edge_of_the_unknown():
    """Covered floor still counts as a target if it borders unexplored space.

    Without this the robot stops as soon as it has covered the little patch it
    happens to know about, and declares a mostly-unexplored pool finished.
    """
    grid = OccupancyMap(cell=0.25)
    grid.observe_ray(0.0, 0.0, 0.0, 1.0, hit=True)
    for x in np.arange(0.0, 1.01, 0.1):
        grid.mark_covered(float(x), 0.0, 0.2)
    # Everything free is covered, but the corridor is one cell wide, so its
    # cells still touch unknown space on either side.
    assert grid.nearest_frontier(0.0, 0.0) is not None


def test_frontier_is_none_when_a_closed_region_is_covered():
    """Fully explored and fully covered really does mean finished."""
    grid = OccupancyMap(cell=0.25)
    # A 1 m square of floor, walled all round, and swept.
    for x in np.arange(0.0, 1.01, 0.25):
        for y in np.arange(0.0, 1.01, 0.25):
            grid.mark_free(float(x), float(y), 0.1)
            grid.mark_covered(float(x), float(y), 0.2)
    for t in np.arange(-0.5, 1.51, 0.25):
        for wall in ((float(t), -0.25), (float(t), 1.25), (-0.25, float(t)), (1.25, float(t))):
            grid.mark_wall(*wall)
    assert grid.nearest_frontier(0.5, 0.5) is None


def test_frontier_does_not_route_through_walls():
    """BFS over free cells, so the target must be reachable."""
    grid = OccupancyMap(cell=0.25)
    # A free cell beyond a solid wall of the map's full height.
    for y in np.arange(-3.0, 3.0, 0.25):
        grid.mark_wall(1.0, float(y))
    grid.mark_free(2.0, 0.0, 0.2)
    grid.mark_free(0.0, 0.0, 0.2)
    grid.mark_covered(0.0, 0.0, 0.3)
    target = grid.nearest_frontier(0.0, 0.0)
    # It may find nothing, but it must never propose the walled-off cell.
    assert target is None or target[0] < 1.0


def test_blocked_ahead_sees_a_wall():
    grid = OccupancyMap(cell=0.25)
    grid.mark_wall(1.0, 0.0)
    assert grid.blocked_ahead(0.0, 0.0, 0.0, 1.5)
    assert not grid.blocked_ahead(0.0, 0.0, np.pi, 1.5)


# -- the controller ---------------------------------------------------------
def test_controller_drives_and_builds_a_map():
    controller = SystematicCoverage()
    result = Simulation(pool="rectangular", controller=controller, seed=3, record=False).run(
        minutes=6
    )
    assert result.metrics.distance_traveled > 30.0
    assert controller.map.explored_cells > 200
    assert result.metrics.coverage > 0.15


def test_controller_does_not_finish_in_the_first_minute():
    """It once declared the pool done after 75 seconds: the frontier search
    counted ticks rather than re-plans and exhausted its give-up budget."""
    controller = SystematicCoverage()
    sim = Simulation(pool="rectangular", controller=controller, seed=3, record=False)
    for _ in range(3000):  # one minute at 50 Hz
        sim.step()
    assert controller.phase is not Phase.DONE


def test_zupts_actually_happen_and_stay_plausible():
    controller = SystematicCoverage()
    Simulation(pool="rectangular", controller=controller, seed=3, record=False).run(minutes=6)
    telemetry = controller.telemetry()
    assert telemetry["zupts"] > 0, "the settle pauses should produce ZUPTs"
    # A real gyro bias is under a degree or two per second. Anything larger
    # means rotation is leaking into the bias state.
    assert abs(np.degrees(telemetry["est_bias"])) < 5.0


def test_estimate_tracks_ground_truth_within_a_metre_or_so():
    """Dead reckoning drifts -- but slowly enough to plan with.

    This is a regression guard, not a claim of accuracy: an estimator that
    diverges wildly makes the map useless and the whole controller pointless.
    """
    from zimablue.geometry import wrap_angle

    controller = SystematicCoverage()
    sim = Simulation(pool="rectangular", controller=controller, seed=3, record=False)
    x0, y0, h0 = sim.state.x, sim.state.y, sim.state.heading
    for _ in range(9000):  # three minutes
        sim.step()
    estimate = controller.estimator.estimate
    # The filter works in its own frame, anchored at the start pose.
    wx = x0 + estimate.x * np.cos(h0) - estimate.y * np.sin(h0)
    wy = y0 + estimate.x * np.sin(h0) + estimate.y * np.cos(h0)
    error = float(np.hypot(wx - sim.state.x, wy - sim.state.y))
    assert error < 2.5, f"estimate drifted {error:.2f} m in three minutes"
    assert abs(wrap_angle(estimate.heading + h0 - sim.state.heading)) < np.deg2rad(60)


def test_telemetry_is_recorded_as_channels():
    result = Simulation(pool="rectangular", controller=SystematicCoverage(), seed=3).run(seconds=60)
    frames = result.recording.frames
    for channel in ("ctl.est_x", "ctl.est_y", "ctl.est_heading", "ctl.zupts", "ctl.phase"):
        assert channel in frames, f"missing {channel}"
    assert frames["ctl.est_x"].size == result.recording.n_frames


def test_controller_is_deterministic():
    def run():
        return (
            Simulation(pool="kidney", controller=SystematicCoverage(), seed=11, record=False)
            .run(minutes=3)
            .metrics.coverage
        )

    assert run() == run()


def test_registered_by_name():
    from zimablue.controllers.base import CONTROLLERS

    assert "systematic" in CONTROLLERS.names()
    assert isinstance(CONTROLLERS.create("systematic"), SystematicCoverage)
