"""Small units that the end-to-end tests exercise only incidentally.

A simulation run touches most of this code, so it looks covered while its edge
cases -- saturation, empty inputs, error paths -- never run. These are the
cases a full run happens not to hit.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

import zimablue as zb
from zimablue.geometry import Grid, closest_point_on_segments, polygon_segments, raycast, wrap_angle
from zimablue.pool.materials import get_material
from zimablue.registry import Registry
from zimablue.robot.command import DriveCommand


# -- DriveCommand ----------------------------------------------------------
def test_stop_is_completely_off():
    stop = DriveCommand.stop()
    assert (stop.left, stop.right, stop.brush, stop.pump) == (0.0, 0.0, False, 0.0)


def test_from_body_preserves_the_turn_radius_when_saturating():
    """Clipping the wheels independently would change the turn radius -- a
    classic and hard-to-spot differential-drive bug."""
    loco = zb.make_robot("tracked").locomotion
    v, omega = 10.0, 4.0  # far beyond the motors
    command = DriveCommand.from_body(v, omega, loco)

    assert max(abs(command.left), abs(command.right)) <= loco.max_speed + 1e-9
    got_v, got_omega = command.body_velocity(loco)
    assert got_v / got_omega == pytest.approx(v / omega, rel=1e-6)


def test_from_body_leaves_a_feasible_twist_alone():
    loco = zb.make_robot("tracked").locomotion
    command = DriveCommand.from_body(0.1, 0.2, loco)
    assert command.body_velocity(loco) == pytest.approx((0.1, 0.2))


def test_clamped_bounds_speed_and_pump():
    loco = zb.make_robot("tracked").locomotion
    clamped = DriveCommand(left=99.0, right=-99.0, pump=5.0).clamped(loco)
    assert abs(clamped.left) <= loco.max_speed and abs(clamped.right) <= loco.max_speed
    assert clamped.pump == 1.0
    assert DriveCommand(pump=-3.0).clamped(loco).pump == 0.0


def test_as_array_is_the_recorded_layout():
    assert DriveCommand(0.2, 0.3, brush=True, pump=0.5).as_array() == pytest.approx(
        [0.2, 0.3, 1.0, 0.5]
    )
    assert DriveCommand(brush=False).as_array()[2] == 0.0


# -- dirt types ------------------------------------------------------------
def test_settling_velocity_orders_the_dirt_types():
    """Sand outruns silt, and floating debris rises rather than sinks."""
    sand = zb.DIRT_TYPES["sand"].settling_velocity
    sediment = zb.DIRT_TYPES["sediment"].settling_velocity
    floating = zb.DIRT_TYPES["floating"].settling_velocity
    assert sand > sediment > 0.0
    assert floating < 0.0


def test_buoyant_and_adhered_flags():
    assert zb.DIRT_TYPES["leaves"].buoyant
    assert not zb.DIRT_TYPES["sand"].buoyant
    assert zb.DIRT_TYPES["biofilm"].adhered
    assert not zb.DIRT_TYPES["sand"].adhered


def test_dirt_type_round_trips():
    original = zb.DIRT_TYPES["algae"]
    clone = zb.DirtType.from_dict(original.to_dict())
    assert clone.name == original.name
    assert clone.settling_velocity == pytest.approx(original.settling_velocity)


def test_an_explicit_settling_velocity_overrides_the_derived_one():
    custom = zb.DirtType(
        name="ballast", density=8000.0, particle_size=1e-3, adhesion=0.0, _settling_velocity=0.123
    )
    assert custom.settling_velocity == 0.123


def test_unknown_dirt_type_lists_the_known_ones():
    from zimablue.dirt.types import get_dirt_type

    with pytest.raises(KeyError, match="sand"):
        get_dirt_type("unobtainium")


# -- registry --------------------------------------------------------------
def test_registry_rejects_a_duplicate_name():
    registry: Registry[int] = Registry("thing")
    registry.add("a", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(lambda: 2)


def test_registry_error_lists_the_alternatives():
    registry: Registry[int] = Registry("thing")
    registry.add("alpha", lambda: 1)
    with pytest.raises(KeyError, match="alpha"):
        registry.create("beta")


def test_registry_supports_len_and_contains():
    registry: Registry[int] = Registry("thing")
    registry.add("a", lambda: 1)
    assert len(registry) == 1 and "a" in registry and "b" not in registry


# -- materials -------------------------------------------------------------
def test_get_material_passes_an_instance_through():
    plaster = get_material("plaster")
    assert get_material(plaster) is plaster


def test_unknown_material_lists_the_alternatives():
    with pytest.raises(KeyError, match="plaster"):
        get_material("marzipan")


def test_material_rejects_nonsense_values():
    with pytest.raises(ValueError, match="friction"):
        zb.SurfaceMaterial("bad", friction=0.0, brush_gain=1.0, adhesion_factor=1.0, roughness=1.0)


# -- geometry --------------------------------------------------------------
def test_wrap_angle_folds_into_the_half_open_turn():
    """The interval is [-pi, pi): exactly +pi comes back as -pi."""
    assert wrap_angle(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert -np.pi <= wrap_angle(123.456) < np.pi


def test_grid_rejects_a_non_positive_cell():
    with pytest.raises(ValueError, match="positive"):
        Grid.covering((0, 0, 1, 1), 0.0)


def test_grid_indexing_clips_to_the_raster():
    grid = Grid.covering((0, 0, 1, 1), 0.5)
    assert grid.index_of(-99.0, -99.0) == (0, 0)
    assert grid.index_of(99.0, 99.0) == (grid.nrows - 1, grid.ncols - 1)
    assert not grid.contains(-1.0, 0.5)
    assert grid.contains(0.5, 0.5)


def test_disk_mask_selects_a_disk():
    grid = Grid.covering((0, 0, 2, 2), 0.1)
    mask = grid.disk_mask(1.0, 1.0, 0.5)
    assert mask.any() and not mask.all()
    xs, ys = grid.cell_centers()
    assert np.hypot(xs[mask] - 1.0, ys[mask] - 1.0).max() <= 0.5 + 1e-9


def test_raycast_with_no_geometry_returns_max_range():
    empty = np.zeros((0, 4))
    assert raycast(empty, (0.0, 0.0), np.array([0.0, 1.0]), 3.0) == pytest.approx([3.0, 3.0])


def test_raycast_ignores_geometry_behind_the_ray():
    segments = polygon_segments(Polygon([(-2, -1), (-1, -1), (-1, 1), (-2, 1)]))
    assert raycast(segments, (0.0, 0.0), np.array([0.0]), 5.0) == pytest.approx([5.0])


def test_closest_point_on_an_empty_segment_set_is_infinitely_far():
    distance, x, y, index = closest_point_on_segments(np.zeros((0, 4)), 1.0, 2.0)
    assert distance == float("inf") and (x, y, index) == (1.0, 2.0, -1)


def test_polygon_segments_includes_holes():
    outer = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (2, 1), (2, 2), (1, 2)]])
    assert len(polygon_segments(outer)) == 8


# -- cleaner ---------------------------------------------------------------
def test_footprint_rotates_with_heading():
    robot = zb.make_robot("tracked")
    flat = robot.footprint(0.0, 0.0, 0.0).bounds
    turned = robot.footprint(0.0, 0.0, np.pi / 2).bounds
    assert flat[2] - flat[0] == pytest.approx(turned[3] - turned[1])


def test_power_draw_grows_with_load_and_accessories():
    robot = zb.make_robot("tracked")
    idle = robot.power_draw(0.0, 0.0, 0.0, 0.0, brush_on=False, pump_duty=0.0)
    working = robot.power_draw(0.3, 0.3, 20.0, 20.0, brush_on=True, pump_duty=1.0)
    assert working > idle > 0.0


def test_cleaning_footprint_is_the_swath():
    robot = zb.make_robot("tracked")
    _, _, radius = robot.cleaning_footprint(1.0, 2.0, 0.0)
    assert radius == pytest.approx(robot.swath_width / 2)


# -- sensor suite ----------------------------------------------------------
def test_duplicate_sensor_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        zb.SensorSuite([zb.IMU(name="a"), zb.Encoder(name="a")])


def test_missing_sensor_attribute_names_what_is_available():
    suite = zb.SensorSuite([zb.IMU()])
    with pytest.raises(AttributeError, match="imu"):
        _ = suite.lidar


def test_clear_faults_clears_every_sensor():
    suite = zb.SensorSuite([zb.IMU(), zb.Sonar()])
    suite.imu.inject_fault(bias=1.0)
    suite.sonar.inject_fault(stuck=True)
    suite.clear_faults()
    assert not suite.imu.faults and not suite.sonar.faults


def test_unknown_sensor_class_in_a_spec_is_reported():
    from zimablue.sensors.suite import sensor_from_spec

    with pytest.raises(KeyError, match="Sonar"):
        sensor_from_spec({"class": "Lidar", "name": "x", "config": {}, "params": {}})


def test_sensor_without_attach_explains_itself():
    from zimablue.sensors.base import SensorContext

    with pytest.raises(RuntimeError, match="attach"):
        zb.IMU().update(SensorContext())
