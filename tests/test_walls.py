"""The wall stops being a line: climbs, the strip, and the waterline."""

from __future__ import annotations

import numpy as np

import zimablue as zb
from zimablue.robot import DriveCommand
from zimablue.robot.components import Locomotion


class PressOn:
    """Drive straight at whatever is ahead, forever. The wall's best friend."""

    name = "press_on"

    def reset(self, robot):
        pass

    def step(self, ctl):
        return DriveCommand(left=0.25, right=0.25, brush=True, pump=1.0)


def climb_run(robot="heavy_duty", minutes=3.0):
    return zb.Simulation(
        pool="rectangular",
        robot=robot,
        dirt="light_sediment",
        controller=PressOn(),
        seed=2,
    ).run(minutes=minutes)


def test_a_grip_robot_pressing_the_wall_climbs_it():
    result = climb_run()
    kinds = [e["kind"] for e in result.recording.events]
    assert "climb_started" in kinds
    assert "climb_topped" in kinds
    assert "climb_ended" in kinds

    # The 3D story comes free: depth runs to the surface and back.
    depth = np.asarray(result.require_recording().column("depth"), float)
    assert np.nanmin(depth) < 0.1, "the robot should reach the waterline"
    assert result.metrics.waterline_coverage > 0.0


def test_a_floor_robot_just_bumps():
    result = climb_run(robot="tracked")
    kinds = {e["kind"] for e in result.recording.events}
    assert "climb_started" not in kinds
    assert result.metrics.waterline_coverage == 0.0
    # It still gets cove credit for pressing against the wall.
    assert result.metrics.wall_coverage > 0.0


def test_wall_coverage_is_now_an_area():
    """A floor robot can no longer reach 100% of the wall by circling it:
    the mid wall and the waterline are out of its reach by construction."""
    result = zb.Simulation(
        pool="rectangular", dirt="light_sediment", controller="baseline_coverage", seed=2
    ).run(minutes=10)
    bands = result.spatial.wall_bands
    assert bands is not None
    assert bands[:, 0].sum() > 0, "cove visits"
    assert bands[:, 1].sum() == 0 and bands[:, 2].sum() == 0
    # Rectangular is 1.5 m deep: the cove is 0.3/1.5 of the area, so even a
    # perfect perimeter lap caps at a fifth of the wall.
    assert result.metrics.wall_coverage <= 0.30


def test_wall_grip_round_trips():
    locomotion = Locomotion(wall_grip=True)
    rebuilt = Locomotion.from_dict(locomotion.to_dict())
    assert rebuilt.wall_grip is True
    old = {k: v for k, v in Locomotion().to_dict().items() if k != "wall_grip"}
    assert Locomotion.from_dict(old).wall_grip is False


def test_the_climb_pays_for_itself_in_energy_and_time():
    result = climb_run(minutes=2.0)
    climbs = [e for e in result.recording.events if e["kind"] == "climb_started"]
    assert climbs
    assert result.metrics.energy_consumed > 0
    # Distance keeps accumulating during the climb -- the tracks are moving.
    assert result.metrics.distance_traveled > 0
