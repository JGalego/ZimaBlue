"""Mixed robot designs in one physical fleet."""

from __future__ import annotations

import math

import pytest

import zimablue as zb
from zimablue.replay.renderer import load_scene


def test_fleet_members_keep_independent_robots_and_controllers():
    fleet = zb.Fleet(
        pool="rectangular",
        dirt="clean",
        members=[
            zb.FleetMember("compact", "bsa"),
            zb.FleetMember("heavy_duty", "random_bounce"),
            zb.FleetMember("tracked", "frontier"),
        ],
        seed=8,
    )

    assert [robot.name for robot in fleet.robots] == ["compact", "heavy_duty", "tracked"]
    assert [controller.name for controller in fleet.controllers] == [
        "bsa",
        "random_bounce",
        "frontier",
    ]
    for left in range(fleet.count):
        for right in range(left + 1, fleet.count):
            distance = math.dist(fleet.start_poses[left][:2], fleet.start_poses[right][:2])
            assert distance > fleet.robots[left].radius + fleet.robots[right].radius

    recording = fleet.run(seconds=0.2).require_recording()
    assert [item["name"] for item in recording.manifest["robot_configs"]] == [
        "compact",
        "heavy_duty",
        "tracked",
    ]
    assert recording.manifest["fleet"]["robots"] == ["compact", "heavy_duty", "tracked"]


def test_heterogeneous_recording_rebuilds_each_replay_geometry():
    fleet = zb.Fleet(
        members=[
            zb.FleetMember("compact", "bsa"),
            zb.FleetMember("heavy_duty", "bsa"),
        ],
        dirt="clean",
    )
    recording = fleet.run(seconds=0.1).require_recording()

    scene = load_scene(recording)

    assert len(scene.fleet_geometry) == 2
    assert scene.fleet_geometry[0][:2] != scene.fleet_geometry[1][:2]


def test_member_start_poses_are_all_explicit_or_all_automatic():
    with pytest.raises(ValueError, match="every fleet member"):
        zb.Fleet(
            members=[
                zb.FleetMember("compact", start_pose=(1.0, 1.0, 0.0)),
                zb.FleetMember("tracked"),
            ]
        )


def test_each_explicit_pose_must_clear_its_own_hull():
    pool = zb.make_pool("rectangular")
    boundary = pool.navigable.bounds
    near_wall = (boundary[0] + 0.01, (boundary[1] + boundary[3]) / 2.0, 0.0)
    with pytest.raises(ValueError, match="lacks clearance for robot 1"):
        zb.Fleet(
            pool=pool,
            members=[
                zb.FleetMember("compact", start_pose=pool.start_pose(clearance=0.5)),
                zb.FleetMember("heavy_duty", start_pose=near_wall),
            ],
        )
