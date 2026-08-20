"""Differential-drive motion and contact."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box as shapely_box

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
