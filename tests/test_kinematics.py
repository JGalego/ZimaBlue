"""Differential-drive motion and contact."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

import zimablue as zb
from zimablue.geometry import wrap_angle
from zimablue.physics.collision import resolve
from zimablue.physics.kinematics import exact_arc_step, slip_factors
from zimablue.pool import Pool
from zimablue.robot import make_robot


def test_straight_line_motion():
    x, y, heading = exact_arc_step(0.0, 0.0, 0.0, 1.0, 0.0, 2.0)
    assert (x, y) == pytest.approx((2.0, 0.0))
    assert heading == pytest.approx(0.0)


def test_pure_rotation_does_not_translate():
    x, y, heading = exact_arc_step(3.0, 4.0, 0.0, 0.0, 1.0, 1.0)
    assert (x, y) == pytest.approx((3.0, 4.0))
    assert heading == pytest.approx(1.0)


def test_arc_integration_closes_a_circle():
    """A full 2*pi turn must return to the start.

    Euler integration does not: it spirals outward, and that error would be
    indistinguishable from the sensor drift the simulator is meant to model.
    """
    # Chosen so the steps divide the circle exactly: 1000 steps of 1 ms at
    # 2*pi rad/s is precisely one revolution, leaving only float error.
    steps, dt, omega, v = 1000, 0.001, 2 * np.pi, 0.5
    x, y, heading = 0.0, 0.0, 0.0
    for _ in range(steps):
        x, y, heading = exact_arc_step(x, y, heading, v, omega, dt)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert wrap_angle(heading) == pytest.approx(0.0, abs=1e-12)


def test_differential_drive_kinematics_round_trip():
    locomotion = make_robot("tracked").locomotion
    v, omega = 0.21, 0.65
    left, right = locomotion.to_wheel_speeds(v, omega)
    assert locomotion.to_body_velocity(left, right) == pytest.approx((v, omega))


def test_slip_rises_on_slippery_surfaces():
    locomotion = make_robot("tracked").locomotion
    grippy = slip_factors(locomotion, 0.95, 0.2, 0.2)
    slippery = slip_factors(locomotion, 0.4, 0.2, 0.2)
    assert slippery[0] > grippy[0]
    assert all(0.0 <= s < 1.0 for s in (*grippy, *slippery))


def test_turning_costs_more_slip_than_driving_straight():
    locomotion = make_robot("tracked").locomotion
    straight = slip_factors(locomotion, 0.85, 0.25, 0.25)
    turning = slip_factors(locomotion, 0.85, -0.25, 0.25)
    assert turning[0] > straight[0]


def test_collision_pushes_the_robot_back_inside():
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 0.05, 2.5, 0.0, radius=0.25)
    assert contact.touching
    assert contact.x >= 0.25 - 1e-9
    assert bool(pool.contains(contact.x, contact.y))


def test_robot_pushed_through_a_wall_is_recovered():
    """A large step must not teleport the robot outside the pool."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, -1.5, 2.5, 0.0, radius=0.25)
    assert contact.touching
    assert bool(pool.contains(contact.x, contact.y))


def test_no_contact_in_open_water():
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 5.0, 2.5, 0.0, radius=0.25)
    assert not contact.touching
    assert (contact.x, contact.y) == (5.0, 2.5)


def test_bump_flags_point_at_the_wall():
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    # Facing +x into the right-hand wall: the front switch should close.
    front = resolve(pool, 9.9, 2.5, 0.0, radius=0.25)
    assert front.flags[0]
    # Same wall, but facing -x: it is now behind us.
    rear = resolve(pool, 9.9, 2.5, np.pi, radius=0.25)
    assert rear.flags[3]


def test_two_robots_dropped_on_the_same_spot_still_separate():
    """Coincident discs have no line between them to push along.

    Only reachable by placing two robots at one point, which a fleet
    configuration can do -- and a zero-length normal would be a divide by
    zero followed by a NaN pose that poisons the whole recording.
    """
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(5.0, 2.5, 0.25)])
    assert contact.touching
    assert np.isfinite([contact.x, contact.y]).all()
    assert (contact.x, contact.y) != (5.0, 2.5), "they have to end up somewhere else"


def test_a_team_mate_further_away_than_the_wall_does_not_win():
    """A robot squeezed between a team-mate and the tiles ends up against the
    wall, not inside it."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 0.05, 2.5, 0.0, radius=0.25, neighbours=[(0.54, 2.5, 0.25)])
    assert bool(pool.contains(contact.x, contact.y))


def test_a_neighbour_out_of_reach_is_not_a_contact():
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(9.0, 2.5, 0.25)])
    assert not contact.touching
    assert (contact.x, contact.y) == (5.0, 2.5)


def test_the_deepest_overlap_is_the_one_resolved():
    """With two team-mates touching, the one further in decides the push."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    shallow = resolve(pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(5.45, 2.5, 0.25)])
    both = resolve(
        pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(5.45, 2.5, 0.25), (5.1, 2.5, 0.25)]
    )
    assert both.penetration > shallow.penetration


# ----------------------------------------------------------------------
# A robot exactly on a surface.
#
# The wall-to-robot vector is zero there, so the normal has to come from the
# surface itself. This used to fall back to the direction of the pool's
# centroid and then flip it -- shapely's contains() is strict, so a point on
# the boundary is not "inside" -- which pushed the robot out through the wall
# it was resting on.


def test_a_robot_exactly_on_a_wall_is_pushed_into_the_pool():
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    for x, y in ((0.0, 2.5), (10.0, 2.5), (5.0, 0.0), (5.0, 5.0)):
        contact = resolve(pool, x, y, 0.0, radius=0.25)
        assert contact.touching
        assert np.isfinite([contact.x, contact.y]).all()
        assert bool(pool.contains(contact.x, contact.y)), f"({x}, {y}) ended up outside"


def test_a_robot_exactly_on_a_corner_is_pushed_along_the_bisector():
    """On a vertex the robot touches two edges, and neither normal alone
    points into the corner -- a step perpendicular to one leaves through the
    other. Their sum is the bisector, which does."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    contact = resolve(pool, 0.0, 0.0, 0.0, radius=0.25)
    assert bool(pool.contains(contact.x, contact.y))
    assert contact.x == pytest.approx(contact.y), "the bisector of a right angle is diagonal"


def test_the_push_is_local_so_a_concave_wall_does_not_send_it_across_the_pool():
    """The kidney's waist is the case the centroid shortcut got wrong: the
    straight line to the middle leaves through the opposite wall."""
    pool = zb.make_pool("kidney")
    ring = pool.boundary.exterior
    outside = []
    for station in np.linspace(0.0, ring.length, 200, endpoint=False):
        point = ring.interpolate(float(station))
        contact = resolve(pool, point.x, point.y, 0.0, radius=0.17)
        if not pool.contains(contact.x, contact.y):
            outside.append((round(point.x, 3), round(point.y, 3)))
    assert outside == [], f"pushed out of the pool at {outside[:5]}"


def test_every_preset_survives_a_robot_placed_on_its_wall():
    for name in zb.POOL_PRESETS.names():
        pool = zb.make_pool(name)
        ring = pool.boundary.exterior
        for station in np.linspace(0.0, ring.length, 60, endpoint=False):
            point = ring.interpolate(float(station))
            contact = resolve(pool, point.x, point.y, 0.0, radius=0.17)
            assert np.isfinite([contact.x, contact.y]).all(), name
            # Never further from the water than it started.
            assert (
                pool.navigable.distance(Point(contact.x, contact.y))
                <= pool.navigable.distance(point) + 1e-9
            ), name


def test_a_contact_is_truthy_about_whether_it_touched_anything():
    """`any` is what the stepping path reads to decide there was a collision."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    assert not resolve(pool, 5.0, 2.5, 0.0, radius=0.25).any
    assert resolve(pool, 0.05, 2.5, 0.0, radius=0.25).any


def test_a_shallower_team_mate_does_not_displace_a_deeper_one():
    """Whichever overlaps most decides, whatever order they arrive in."""
    pool = Pool(shapely_box(0, 0, 10, 5), 1.5)
    deep_first = resolve(
        pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(5.1, 2.5, 0.25), (5.45, 2.5, 0.25)]
    )
    deep_last = resolve(
        pool, 5.0, 2.5, 0.0, radius=0.25, neighbours=[(5.45, 2.5, 0.25), (5.1, 2.5, 0.25)]
    )
    assert deep_first.penetration == pytest.approx(deep_last.penetration)
    assert (deep_first.x, deep_first.y) == pytest.approx((deep_last.x, deep_last.y))
