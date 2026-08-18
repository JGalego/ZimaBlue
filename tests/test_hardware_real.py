"""Checks against motion this package did not generate.

Every other test in this suite is a closed loop.  The simulator produces the
motion, the sensor models produce readings from it under assumptions the
estimator shares, and the estimator is scored against the same ground truth
that generated all of it.  That catches bugs and cannot catch a wrong model:
nothing in it could tell you the filter would fall over on a real robot,
because nothing in it has ever seen one.

These run the whole hardware stack over a Pioneer 3-DX driving a real building,
tracked by a real motion capture rig.  They skip unless the log is present::

    python tools/fetch_trajectory.py --all

They are deliberately loose.  The point is not to pin down a number that will
change whenever the estimator is touched -- it is to fail if a real trajectory
breaks something, which is a thing that has already happened twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import zimablue as zb
from zimablue.controllers.systematic import SystematicCoverage
from zimablue.hardware import (
    HardwareRuntime,
    SafetyLimits,
    TrajectorySource,
    Watchdog,
    read_tum_trajectory,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "trajectories"
SEQUENCE = DATA / "pioneer_slam.txt"

pytestmark = pytest.mark.skipif(
    not SEQUENCE.exists(),
    reason=f"no real trajectory at {SEQUENCE}; run tools/fetch_trajectory.py",
)


@pytest.fixture(scope="module")
def trajectory():
    return read_tum_trajectory(SEQUENCE, forward="z").recentre()


@pytest.fixture(scope="module")
def replayed(trajectory):
    """The full stack -- sensors, loop, watchdog, recorder -- over real motion."""
    robot = zb.make_robot("tracked")
    now = [0.0]
    runtime = HardwareRuntime(
        controller=SystematicCoverage(),
        robot=robot,
        source=TrajectorySource(trajectory, robot, seed=1),
        actuate=lambda command: None,
        watchdog=Watchdog(SafetyLimits(required=("encoder", "imu"))),
        rate_hz=50.0,
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + max(seconds, 0.0)),
    )
    return runtime.run(seconds=trajectory.duration - 1.0)


def position_error(run, trajectory):
    frames = run.recording.frames
    t = frames["time"]
    return np.hypot(
        frames["ctl.est_x"] - np.interp(t, trajectory.time, trajectory.x),
        frames["ctl.est_y"] - np.interp(t, trajectory.time, trajectory.y),
    )


# -- the log is what we think it is ---------------------------------------


def test_the_trajectory_is_a_ground_robot_not_a_drone(trajectory):
    """Guards against the projection being taken off the wrong plane.

    Get ``up`` wrong and everything downstream still runs, on a trajectory
    that is a shadow of the real one cast in the wrong direction.
    """
    speed, yaw_rate = trajectory.body_velocities()
    assert np.abs(speed).max() < 1.5, "a Pioneer 3-DX does not do 1.5 m/s"
    assert np.abs(yaw_rate).max() < 3.0
    assert 0.1 < np.abs(speed).mean() < 0.6


def test_the_forward_axis_is_the_one_that_agrees_with_the_direction_of_travel():
    """A ground robot mostly drives forwards, and that identifies the axis.

    TUM's poses are the camera's, so the body-forward axis is its optical +z.
    Reading yaw straight off the quaternion instead gives a heading rotated
    away from the direction of travel, which shows up as a robot that spends
    two fifths of the run apparently reversing.
    """
    reversing = {}
    for axis in ("x", "y", "z"):
        speed, _ = read_tum_trajectory(SEQUENCE, forward=axis).body_velocities()
        reversing[axis] = float((speed < -0.02).mean())
    assert min(reversing, key=lambda a: reversing[a]) == "z"
    assert reversing["z"] < 0.1, "even the right axis should not read as mostly reverse"


def test_a_real_log_has_dropouts(trajectory):
    """If it did not, it would not be testing the thing it is here to test."""
    gaps = trajectory.gaps
    assert gaps.max() > 0.2, "expected the tracker to lose the robot at least once"
    assert np.median(gaps) < 0.02


def test_reading_a_missing_log_says_how_to_get_one():
    with pytest.raises(FileNotFoundError, match="fetch_trajectory"):
        read_tum_trajectory(DATA / "does-not-exist.txt")


# -- the stack survives it -------------------------------------------------


def test_the_whole_stack_runs_a_real_trajectory_without_tripping(replayed, trajectory):
    assert replayed.watchdog_reasons == []
    assert replayed.ticks > 0.9 * 50 * trajectory.duration
    assert replayed.overruns == 0


def test_odometry_distance_is_close_to_the_distance_really_driven(replayed, trajectory):
    """Slightly short, because differentiating a tracker needs smoothing and
    smoothing rounds off the corners. Wildly wrong would mean a unit error."""
    assert replayed.distance == pytest.approx(trajectory.path_length, rel=0.15)


def test_the_estimate_tracks_a_real_robot(replayed, trajectory):
    """The first estimator figure in this project that is not self-referential.

    Loose on purpose. The claim being defended is "does not fall over on real
    motion", not a particular number -- 42 m of real driving currently ends
    about 0.15 m out, and pinning that would make every estimator change a
    test failure.

    Read it with the caveat attached: encoder readings are derived from the
    true body velocity by inverse kinematics, so there is no slip. A real
    drivetrain's encoders run long, and that is the largest single term in
    dead-reckoning drift. This is the easy half of the problem, done on hard
    data.
    """
    error = position_error(replayed, trajectory)
    assert np.isfinite(error).all()
    assert error[-1] < 0.1 * trajectory.path_length
    assert error.mean() < 2.0


def test_the_filter_is_not_wildly_overconfident_about_real_motion(replayed, trajectory):
    """In simulation the covariance is measurably optimistic. On this log it is
    not, which is worth knowing and worth noticing if it changes."""
    error = position_error(replayed, trajectory)
    sigma = replayed.recording.frames["ctl.est_sigma"]
    assert sigma[-1] > 0.2 * error[-1]


def test_a_real_run_replays_in_the_ordinary_viewer(replayed, tmp_path):
    """The same acceptance test the roadmap sets for the 3D backend."""
    path = replayed.save(tmp_path / "pioneer.zbr")
    reloaded = zb.Recording.load(path)
    assert reloaded.n_frames == replayed.ticks
    assert reloaded.manifest["pose_source"] == "estimate"
    assert {"x", "y", "heading", "encoder.left", "imu.gz"} <= set(reloaded.channels)


@pytest.mark.parametrize("name", ["pioneer_slam", "pioneer_slam2", "pioneer_360"])
def test_every_available_log_parses_into_something_physical(name):
    path = DATA / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"{name} not fetched")
    trajectory = read_tum_trajectory(path, forward="z").recentre()
    speed, yaw_rate = trajectory.body_velocities()
    assert trajectory.duration > 10.0
    assert 1.0 < trajectory.path_length < 500.0
    assert np.abs(speed).max() < 1.5
    assert np.isfinite(speed).all() and np.isfinite(yaw_rate).all()


def test_the_derived_yaw_rate_integrates_back_to_the_real_heading(trajectory):
    """Guards the harness rather than the estimator.

    If the smoothing that makes differentiation possible also ate the turns,
    every heading result measured this way would be wrong in a direction
    nothing else would reveal.
    """
    _, yaw_rate = trajectory.body_velocities()
    integrated = np.trapezoid(yaw_rate, trajectory.time)
    real = trajectory.heading[-1] - trajectory.heading[0]
    assert integrated == pytest.approx(real, abs=np.radians(10.0))
