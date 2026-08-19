"""A planner on the hardware runtime, rehearsed against the simulator."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.controllers.base import ControlInput
from zimablue.hardware import HardwareRuntime, SimulatedPlant, Survey
from zimablue.hardware.motors import MotorEffort
from zimablue.planners import PathFollower


def drive(runtime, plant, seconds):
    """Tick the loop on a fake clock, so the rehearsal is deterministic."""
    for _ in range(int(seconds / runtime.period)):
        runtime.tick()


@pytest.fixture
def loop():
    plant = SimulatedPlant(pool="rectangular", seed=5)
    clock = {"t": 0.0}

    def fake_clock():
        clock["t"] += 0.0  # read-only; ticks advance it below
        return clock["t"]

    runtime = HardwareRuntime(
        controller=PathFollower("boustrophedon"),
        robot=plant.robot,
        source=plant,
        actuate=plant.actuate,
        platform=plant.platform,
        survey=plant.survey(),
        clock=fake_clock,
        sleep=lambda s: None,
    )
    # Advance the clock by one period per tick, from inside the clock itself.
    original_tick = runtime.tick

    def paced_tick():
        clock["t"] += runtime.period
        return original_tick()

    runtime.tick = paced_tick  # type: ignore[method-assign]
    yield runtime, plant
    plant.close()


def test_a_planner_drives_on_the_hardware_runtime(loop):
    runtime, plant = loop
    drive(runtime, plant, seconds=30.0)
    run = runtime.finish()

    follower = runtime.controller
    assert follower.target > 0, "the plan never advanced past its first waypoint"
    assert run.distance > 1.0
    assert not run.watchdog_reasons, run.watchdog_reasons
    # The recording replays: it carries the surveyed pool, not none at all.
    assert run.recording is not None
    assert run.recording.manifest["pool_config"] is not None
    assert run.recording.manifest["ground_truth"] is False


def test_the_rehearsal_can_score_its_own_drift(loop):
    runtime, plant = loop
    drive(runtime, plant, seconds=20.0)
    believed = runtime.controller._pose
    actual = plant.truth_pose()
    drift = float(np.hypot(believed[0] - actual[0], believed[1] - actual[1]))
    # Dead reckoning drifts; the point is that the number exists and is sane.
    assert drift < 2.0


def test_a_follower_without_truth_or_survey_says_what_to_do():
    follower = PathFollower("boustrophedon")
    follower.reset(zb.make_robot("tracked"))
    empty = ControlInput(
        time=0.0,
        dt=0.02,
        readings={},
        battery=1.0,
        filter_load=0.0,
        robot=zb.make_robot("tracked"),
    )
    with pytest.raises(RuntimeError, match="Survey"):
        follower.step(empty)


def test_a_truth_follower_refuses_to_run_from_a_survey_alone():
    pool = zb.make_pool("rectangular")
    follower = PathFollower("boustrophedon", localisation="truth")
    follower.reset(zb.make_robot("tracked"))
    surveyed = ControlInput(
        time=0.0,
        dt=0.02,
        readings={},
        battery=1.0,
        filter_load=0.0,
        robot=zb.make_robot("tracked"),
        survey=Survey(pool=pool, start=(1.0, 1.0, 0.0)),
    )
    with pytest.raises(RuntimeError, match="simulator"):
        follower.step(surveyed)


def test_the_plant_translates_motor_duty_into_track_speed():
    plant = SimulatedPlant(pool="rectangular", seed=1)
    plant.actuate(MotorEffort(left=0.5, right=0.5))
    command = plant._puppet.command
    limit = plant.robot.locomotion.max_speed
    assert command.left == pytest.approx(0.5 * limit)
    assert command.right == pytest.approx(0.5 * limit)
    plant.close()


def test_simulation_followers_still_run_unchanged():
    result = zb.Simulation(
        pool="rectangular",
        controller=PathFollower("boustrophedon"),
        expose_truth=True,
        seed=2,
    ).run(minutes=0.5)
    assert result.metrics.distance_traveled > 1.0
