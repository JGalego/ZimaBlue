"""Reading somebody else's robot.

Everything else in this library produces its own data, which is a comfortable
position and an unfalsifiable one.  The estimator's error figures, the timing
assumptions, the noise parameters -- all of them are measured against motion
this package generated, using a noise model this package invented.  They cannot
be wrong in any way the test suite can see.

This module reads trajectories logged off real robots so that at least some of
those numbers can be checked against something that happened.  A trajectory is
a poor substitute for a full sensor log, and :class:`Trajectory` is honest about
which half of the problem it can speak to: it gives you real *motion* -- real
accelerations, real stop-start, real turning behaviour, real gaps where the
tracker lost sight of the robot -- and nothing at all about real *sensors*.

The format read here is TUM's, from the RGB-D benchmark: whitespace-separated
``timestamp tx ty tz qx qy qz qw``, comments on ``#``.  It is the closest thing
the field has to a lingua franca for ground-truth trajectories, and half a dozen
other datasets publish in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["FORWARD_AXIS", "Trajectory", "read_tum_trajectory"]

FORWARD_AXIS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
"""Which body axis points the way the robot is going, by name."""

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    """A real robot's pose over time, projected onto the floor.

    ``time`` is seconds from the first sample. ``x``, ``y`` are metres and
    ``heading`` is radians, unwrapped so differentiation does not see the
    wrap as a spike.

    A pool cleaner moves in a plane, so the vertical component and the roll and
    pitch of the source trajectory are dropped. For a ground robot that costs
    nothing; for a flying one it would be nonsense, which is why the reader
    says what it is reading.
    """

    time: FloatArray
    x: FloatArray
    y: FloatArray
    heading: FloatArray
    source: str = "unknown"

    def __post_init__(self) -> None:
        sizes = {self.time.size, self.x.size, self.y.size, self.heading.size}
        if len(sizes) != 1:
            raise ValueError(f"trajectory columns have different lengths: {sizes}")
        if self.time.size < 2:
            raise ValueError("a trajectory needs at least two samples")

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def path_length(self) -> float:
        return float(np.hypot(np.diff(self.x), np.diff(self.y)).sum())

    @property
    def gaps(self) -> FloatArray:
        """Sample intervals. Real logs have long ones; that is the point."""
        return np.diff(self.time)

    def extent(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` -- how big an area this covers."""
        return (float(self.x.min()), float(self.y.min()), float(self.x.max()), float(self.y.max()))

    def body_velocities(self, smooth: float = 0.15) -> tuple[FloatArray, FloatArray]:
        """Forward speed and yaw rate, differentiated from the poses.

        Differentiating a real tracker's output is not the two-line job it
        looks like. Three things bite, and all three were found by doing it
        wrong on a real file first:

        *Uneven sampling.* Mocap timestamps are not a grid. A pair of samples
        200 microseconds apart, which this log has, turns a millimetre of
        tracker noise into fifty metres per second. So the poses are put on a
        uniform grid before anything is differenced, and the answer is
        interpolated back.

        *Dropouts.* When the tracker loses the robot the gap is filled by
        interpolation, which reports a smooth constant velocity across it.
        That is a fiction, but it is the least-wrong one: the alternative is a
        single enormous spike at the moment tracking resumes. Check
        :attr:`gaps` to see how much of a run is fiction.

        *Quantisation.* Position comes out in discrete steps, and at 300 Hz the
        derivative of that step is far larger than the signal. ``smooth`` is
        the width in seconds of a moving average applied before differencing.
        Long against the sample interval, short against how fast a ground robot
        can change speed.

        Speed is signed against the heading, so reversing reads as negative
        rather than as a fast run in the wrong direction.
        """
        rate = 1.0 / float(np.median(self.gaps))
        grid = np.arange(self.time[0], self.time[-1], 1.0 / rate)
        x = np.interp(grid, self.time, self.x)
        y = np.interp(grid, self.time, self.y)
        heading = np.interp(grid, self.time, self.heading)

        width = max(round(smooth * rate), 1)
        if width > 1:
            x, y, heading = (_moving_average(a, width) for a in (x, y, heading))

        step = 1.0 / rate
        vx = np.gradient(x, step)
        vy = np.gradient(y, step)
        forward = vx * np.cos(heading) + vy * np.sin(heading)
        v = np.copysign(np.hypot(vx, vy), forward)
        omega = np.gradient(heading, step)
        return (
            np.interp(self.time, grid, v),
            np.interp(self.time, grid, omega),
        )

    def resample(self, rate_hz: float) -> Trajectory:
        """Interpolate onto a uniform grid.

        Note what this destroys: the gaps. A resampled trajectory is a
        convenience for driving a fixed-rate loop, not a realistic arrival
        pattern -- use :attr:`gaps` on the original if what you are testing is
        how the loop handles missing data.
        """
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        grid = np.arange(self.time[0], self.time[-1], 1.0 / rate_hz)
        return Trajectory(
            time=grid - grid[0],
            x=np.interp(grid, self.time, self.x),
            y=np.interp(grid, self.time, self.y),
            heading=np.interp(grid, self.time, self.heading),
            source=f"{self.source} resampled to {rate_hz:g} Hz",
        )

    def recentre(self) -> Trajectory:
        """Shift so the run starts at the origin facing along +x.

        Trackers report in their own frame, which is wherever the calibration
        rig happened to be. For comparing an estimator against the truth that
        offset is a constant error nobody cares about.
        """
        c, s = np.cos(-self.heading[0]), np.sin(-self.heading[0])
        dx, dy = self.x - self.x[0], self.y - self.y[0]
        return Trajectory(
            time=self.time - self.time[0],
            x=c * dx - s * dy,
            y=s * dx + c * dy,
            heading=self.heading - self.heading[0],
            source=self.source,
        )


def read_tum_trajectory(
    path: str | Path,
    *,
    up: str = "z",
    forward: str = "z",
    min_interval: float = 1e-3,
) -> Trajectory:
    """Read a TUM-format ground-truth file.

    ``up`` names the world axis perpendicular to the floor and ``forward`` the
    *body* axis that points where the robot is going. Both have to be stated
    rather than guessed, because datasets disagree and the wrong choice
    produces a trajectory that looks entirely plausible and is a projection of
    the wrong plane, or a heading rotated ninety degrees from the direction of
    travel. TUM's poses are the camera's, and a camera looks along its own +z.

    ``min_interval`` drops samples closer together than this, in seconds. See
    :meth:`Trajectory.body_velocities` for why that is not optional.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no trajectory at {path}. Real-data checks need a log to check "
            f"against; tools/fetch_trajectory.py downloads one."
        )
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; TUM format needs 8 "
            "(timestamp tx ty tz qx qy qz qw)"
        )

    t = data[:, 0]
    translation = data[:, 1:4]
    quaternion = data[:, 4:8]

    axes = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}
    if up not in axes:
        raise ValueError(f"up must be one of {sorted(axes)}, got {up!r}")
    i, j, _ = axes[up]

    # Heading is the body's forward axis rotated into the world and projected
    # onto the floor -- *not* the yaw Euler angle of the quaternion. They are
    # not the same thing when the sensor is mounted at an angle to the chassis,
    # and TUM's poses are the camera's, whose optical axis is its +z. Reading
    # the yaw straight off the quaternion gives a heading that disagrees with
    # the direction of travel, which shows up downstream as a robot that spends
    # forty per cent of the run apparently driving backwards.
    heading_vector = _rotate(quaternion, np.array(FORWARD_AXIS[forward], dtype=float))
    heading = np.arctan2(heading_vector[:, j], heading_vector[:, i])

    order = np.argsort(t, kind="stable")
    t, translation, heading = t[order], translation[order], heading[order]

    # Samples too close together break every downstream derivative: this log
    # has pairs 200 microseconds apart, and a millimetre of tracker noise over
    # that interval is five metres per second. Real logs have exact duplicates
    # too, from a tracker publishing twice for one frame.
    keep = np.concatenate([[True], np.diff(t) >= min_interval])
    t, translation, heading = t[keep], translation[keep], heading[keep]

    return Trajectory(
        time=t - t[0],
        x=translation[:, i],
        y=translation[:, j],
        heading=np.unwrap(heading),
        source=str(path.name),
    )


def _rotate(q: FloatArray, vector: FloatArray) -> FloatArray:
    """Rotate ``vector`` by each unit quaternion ``(x, y, z, w)``.

    The standard ``v + 2w(u x v) + 2(u x (u x v))`` form, which avoids building
    a rotation matrix per sample.
    """
    u = q[:, :3]
    w = q[:, 3:4]
    cross1 = np.cross(u, np.broadcast_to(vector, u.shape))
    cross2 = np.cross(u, cross1)
    return vector + 2.0 * (w * cross1 + cross2)


def _moving_average(values: FloatArray, width: int) -> FloatArray:
    """Centred moving average, edge-padded so the length is preserved."""
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
