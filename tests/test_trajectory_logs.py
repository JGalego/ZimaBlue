"""Reading a motion-capture log, checked against a trajectory we wrote.

``tests/test_hardware_real.py`` runs this parser over a real Pioneer log and
skips when that file has not been fetched -- which is every CI run. What is
here instead is a synthetic TUM file with a known answer, so the parts that
are easy to get quietly wrong (which plane is the floor, which body axis is
forward, sorting, near-duplicate samples) are checked on every run.

Synthetic input cannot tell you the model is right; that is what the real log
is for. It can tell you the file was read the way it says it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.hardware.logs import Trajectory, read_tum_trajectory


def _quaternion_about_z(yaw: np.ndarray) -> np.ndarray:
    """``(x, y, z, w)`` for a rotation of ``yaw`` about +z."""
    zeros = np.zeros_like(yaw)
    return np.column_stack([zeros, zeros, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def write_tum(path, *, t, xyz, quaternion):
    rows = np.column_stack([t, xyz, quaternion])
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(" ".join(f"{v:.9f}" for v in row) for row in rows)
        + "\n"
    )
    return path


@pytest.fixture
def straight_line(tmp_path):
    """Two metres along +x at 10 Hz, facing the way it is going."""
    t = np.arange(0.0, 2.0, 0.1) + 1000.0  # a tracker's clock starts wherever
    xyz = np.column_stack([np.linspace(0.0, 2.0, t.size), np.zeros(t.size), np.zeros(t.size)])
    return write_tum(
        tmp_path / "line.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(np.zeros(t.size))
    )


# ----------------------------------------------------------------------
def test_a_log_reads_back_as_the_motion_that_was_written(straight_line):
    trajectory = read_tum_trajectory(straight_line, forward="x")
    assert trajectory.time[0] == 0.0, "the tracker's epoch should be subtracted off"
    assert trajectory.duration == pytest.approx(1.9)
    assert trajectory.path_length == pytest.approx(2.0)
    assert trajectory.source == "line.txt"


def test_the_up_axis_chooses_which_plane_is_the_floor(tmp_path):
    """With up='y' the floor is xz, so what was a z offset becomes the y one."""
    t = np.arange(0.0, 1.0, 0.1)
    xyz = np.column_stack([np.zeros(t.size), np.zeros(t.size), np.linspace(0.0, 1.0, t.size)])
    path = write_tum(
        tmp_path / "climb.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(np.zeros(t.size))
    )

    flat = read_tum_trajectory(path, up="z", forward="x")
    assert flat.path_length == pytest.approx(0.0), "motion along +z is vertical when up is z"

    tilted = read_tum_trajectory(path, up="y", forward="x")
    assert tilted.path_length == pytest.approx(1.0)


def test_the_forward_axis_chooses_where_the_robot_is_looking(straight_line):
    """A chassis points along its +x; a sensor bolted on sideways does not.

    The log is a robot driving along +x with an identity orientation. Read
    with the wrong body axis, the same file reports a heading ninety degrees
    off the direction of travel -- which downstream looks like a robot
    driving sideways for the whole run.
    """
    chassis = read_tum_trajectory(straight_line, forward="x")
    sideways = read_tum_trajectory(straight_line, forward="y")
    assert chassis.heading[0] == pytest.approx(0.0, abs=1e-9)
    assert sideways.heading[0] == pytest.approx(np.pi / 2, abs=1e-9)


def test_a_body_axis_parallel_to_up_projects_to_nothing(straight_line):
    """forward='z' with up='z' asks for the heading of a vector pointing up.

    There is no direction on the floor to report, so it degenerates to zero
    rather than raising -- worth knowing, because it is the combination
    somebody reaches for after reading that TUM poses are the camera's.
    """
    assert np.all(read_tum_trajectory(straight_line, forward="z").heading == 0.0)


def test_a_turning_robot_has_a_heading_that_turns_with_it(tmp_path):
    t = np.arange(0.0, 2.0, 0.05)
    yaw = np.linspace(0.0, np.pi / 2, t.size)
    xyz = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(t.size)])
    path = write_tum(tmp_path / "turn.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(yaw))

    heading = read_tum_trajectory(path, forward="x").heading
    assert heading[-1] - heading[0] == pytest.approx(np.pi / 2, abs=1e-6)
    assert np.all(np.diff(heading) > 0), "unwrapped, a steady turn never goes backwards"


def test_samples_out_of_order_are_sorted_rather_than_believed(tmp_path):
    t = np.array([0.0, 0.3, 0.1, 0.2])
    xyz = np.column_stack([t * 10.0, np.zeros(4), np.zeros(4)])
    path = write_tum(tmp_path / "jumbled.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0))

    trajectory = read_tum_trajectory(path, forward="x")
    assert np.all(np.diff(trajectory.time) > 0)
    assert trajectory.x.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_samples_too_close_together_are_dropped(tmp_path):
    """A tracker publishing twice for one frame makes a 5 m/s derivative.

    Two of these five samples are 200 microseconds from their neighbour --
    the interval that turns a millimetre of tracker noise into a speed no
    ground robot can reach.
    """
    t = np.array([0.0, 0.0002, 0.1, 0.1002, 0.2])
    xyz = np.column_stack([np.zeros(5), np.zeros(5), np.zeros(5)])
    path = write_tum(tmp_path / "dupes.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0))

    assert read_tum_trajectory(path, forward="x").time.size == 3
    assert read_tum_trajectory(path, forward="x", min_interval=0.0).time.size == 5


def test_a_missing_log_says_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_trajectory"):
        read_tum_trajectory(tmp_path / "nothing.txt")


def test_a_file_that_is_not_tum_format_says_so(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("0.0 1.0 2.0\n0.1 1.1 2.1\n")
    with pytest.raises(ValueError, match="TUM format needs 8"):
        read_tum_trajectory(path)


def test_an_unknown_up_axis_is_refused(straight_line):
    with pytest.raises(ValueError, match="up must be one of"):
        read_tum_trajectory(straight_line, up="w")


# ----------------------------------------------------------------------
def test_resampling_lands_on_a_uniform_grid(straight_line):
    original = read_tum_trajectory(straight_line, forward="x")
    resampled = original.resample(50.0)

    assert np.allclose(np.diff(resampled.time), 0.02)
    assert resampled.time[0] == 0.0
    # The grid is half-open, so it stops just short of the final sample --
    # within one step of the original, never past it.
    assert original.duration - 0.02 <= resampled.duration <= original.duration
    assert resampled.path_length == pytest.approx(original.path_length, abs=0.05)
    assert "50 Hz" in resampled.source


def test_resampling_destroys_the_gaps_it_was_asked_to_destroy(tmp_path):
    """The docstring's warning, asserted: a resampled log is not an arrival pattern."""
    t = np.array([0.0, 0.1, 0.9, 1.0])  # a 0.8 s dropout in the middle
    xyz = np.column_stack([t, np.zeros(4), np.zeros(4)])
    path = write_tum(tmp_path / "gappy.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0))

    original = read_tum_trajectory(path, forward="x")
    assert original.gaps.max() == pytest.approx(0.8)
    assert original.resample(20.0).gaps.max() == pytest.approx(0.05)


def test_a_nonsense_rate_is_refused(straight_line):
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        read_tum_trajectory(straight_line, forward="x").resample(0.0)


def test_recentring_puts_the_start_at_the_origin_facing_along_x(tmp_path):
    """The tracker's frame is wherever the rig was; that offset is not an error."""
    t = np.arange(0.0, 1.0, 0.1)
    yaw = np.full(t.size, np.pi / 2)  # driving along +y, from a corner
    xyz = np.column_stack([np.full(t.size, 5.0), 3.0 + t, np.zeros(t.size)])
    path = write_tum(tmp_path / "offset.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(yaw))

    moved = read_tum_trajectory(path, forward="x").recentre()
    assert (moved.x[0], moved.y[0], moved.heading[0]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # Rotated onto +x, the whole run is now a straight line along x.
    assert moved.x[-1] == pytest.approx(0.9, abs=1e-6)
    assert moved.y[-1] == pytest.approx(0.0, abs=1e-6)


def test_recentring_does_not_change_the_shape_of_the_run(tmp_path):
    t = np.arange(0.0, 2.0, 0.05)
    yaw = np.linspace(0.4, 0.4 + np.pi, t.size)
    xyz = np.column_stack([2.0 + np.cos(yaw), 7.0 + np.sin(yaw), np.zeros(t.size)])
    path = write_tum(tmp_path / "arc.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(yaw))

    original = read_tum_trajectory(path, forward="x")
    assert original.recentre().path_length == pytest.approx(original.path_length)


# ----------------------------------------------------------------------
def test_the_extent_is_the_bounding_box(straight_line):
    assert read_tum_trajectory(straight_line, forward="x").extent() == pytest.approx(
        (0.0, 0.0, 2.0, 0.0)
    )


def test_ragged_columns_are_refused_at_construction():
    with pytest.raises(ValueError, match="different lengths"):
        Trajectory(time=np.arange(3.0), x=np.arange(3.0), y=np.arange(2.0), heading=np.arange(3.0))


def test_one_sample_is_not_a_trajectory():
    with pytest.raises(ValueError, match="at least two samples"):
        Trajectory(time=np.zeros(1), x=np.zeros(1), y=np.zeros(1), heading=np.zeros(1))


# ----------------------------------------------------------------------
# Differentiating the poses. The real log is what proves the smoothing is
# worth having; these check the arithmetic against motion whose speed and
# yaw rate were chosen rather than measured.


def _driving(tmp_path, *, speed: float = 0.5, seconds: float = 4.0, rate: float = 50.0):
    t = np.arange(0.0, seconds, 1.0 / rate)
    xyz = np.column_stack([speed * t, np.zeros(t.size), np.zeros(t.size)])
    return read_tum_trajectory(
        write_tum(tmp_path / "drive.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0)),
        forward="x",
    )


def test_a_constant_speed_reads_back_as_that_speed(tmp_path):
    speed, _ = _driving(tmp_path, speed=0.5).body_velocities()
    # The ends are edge-padded by the smoothing window, so judge the middle.
    assert np.median(speed) == pytest.approx(0.5, rel=0.02)


def test_a_straight_line_has_no_yaw_rate(tmp_path):
    _, omega = _driving(tmp_path).body_velocities()
    assert np.abs(omega).max() < 1e-6


def test_reversing_reads_as_a_negative_speed_not_a_fast_run_the_wrong_way(tmp_path):
    """Speed is signed against the heading, which is the whole point.

    Driving backwards down a lane and driving forwards up it have the same
    ``hypot``; only the sign tells them apart.
    """
    t = np.arange(0.0, 4.0, 0.02)
    xyz = np.column_stack([-0.5 * t, np.zeros(t.size), np.zeros(t.size)])  # backwards along +x
    trajectory = read_tum_trajectory(
        write_tum(tmp_path / "back.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0)),
        forward="x",
    )
    speed, _ = trajectory.body_velocities()
    assert np.median(speed) == pytest.approx(-0.5, rel=0.02)


def test_a_steady_turn_reads_back_as_its_yaw_rate(tmp_path):
    t = np.arange(0.0, 4.0, 0.02)
    rate = 0.3  # rad/s
    yaw = rate * t
    xyz = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(t.size)])
    trajectory = read_tum_trajectory(
        write_tum(tmp_path / "circle.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(yaw)),
        forward="x",
    )
    _, omega = trajectory.body_velocities()
    assert np.median(omega) == pytest.approx(rate, rel=0.05)


def test_smoothing_is_what_stops_quantisation_becoming_speed(tmp_path):
    """Position arrives in discrete steps; the derivative of a step is huge.

    Unsmoothed, a millimetre-quantised log of a robot crawling at 0.05 m/s
    reports speeds several times that: at 300 Hz the robot covers a sixth of
    a millimetre per sample, so the position sits still for six samples and
    then jumps a whole one. This is why ``smooth`` is a parameter and not a
    constant somebody removed.
    """
    t = np.arange(0.0, 4.0, 1.0 / 300.0)
    quantised = np.round(0.05 * t, 3)  # a tracker reporting millimetres
    xyz = np.column_stack([quantised, np.zeros(t.size), np.zeros(t.size)])
    trajectory = read_tum_trajectory(
        write_tum(tmp_path / "steps.txt", t=t, xyz=xyz, quaternion=_quaternion_about_z(t * 0)),
        forward="x",
    )

    raw, _ = trajectory.body_velocities(smooth=0.0)
    smoothed, _ = trajectory.body_velocities(smooth=0.15)
    assert raw.max() > 2 * 0.05, "unsmoothed, the quantisation should dominate"
    assert np.median(smoothed) == pytest.approx(0.05, abs=0.01)
