#!/usr/bin/env python3
"""Score the pose estimator against a real robot's real trajectory.

Every other number this project quotes about its estimator was measured
against motion the project generated itself.  That is a comfortable position
and an unfalsifiable one: the filter and the thing it is tracking come out of
the same assumptions, so the filter cannot be wrong in a way the test suite can
see.

This runs the whole hardware stack -- sensor models, control loop, watchdog,
recorder -- over a trajectory driven by a Pioneer 3-DX through a real building,
tracked at 300 Hz by a real motion capture rig, and asks how far the estimate
has drifted from where the robot actually was.

    python tools/fetch_trajectory.py --all
    python examples/replay_real_trajectory.py

What it can and cannot tell you, stated up front because the difference is the
whole value of the exercise:

* The *motion* is real. Real accelerations, real stop-start, real turning,
  real tracker dropouts including one of a full second. Integrating that is
  meaningfully harder than integrating a trajectory produced by the same
  kinematics the filter assumes.
* The *sensors* are not. Encoder and gyro readings are synthesised from the
  true motion through the shipped noise models, so the noise is still ours --
  a guessed bias walk at a guessed rate.
* There is **no slip**. Encoders are derived from the true body velocity by
  inverse kinematics, so they agree with the ground truth up to that noise. A
  real drivetrain's encoders run long, and that bias is the largest single
  term in dead-reckoning drift. The estimator is being flattered here, in a
  known direction.

Closing the last two needs a log from a real IMU on a real drivetrain. That is
an afternoon with a phone taped to something that moves, and it is the highest
value single thing anybody could contribute to this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import zimablue as zb
from zimablue.controllers.systematic import SystematicCoverage
from zimablue.hardware import (
    HardwareRuntime,
    SafetyLimits,
    TrajectorySource,
    Watchdog,
    read_tum_trajectory,
)

DEFAULT_DIR = Path("data/trajectories")


def score(path: Path, *, robot_name: str = "tracked", rate_hz: float = 50.0, seed: int = 1):
    trajectory = read_tum_trajectory(path, forward="z").recentre()
    robot = zb.make_robot(robot_name)
    source = TrajectorySource(trajectory, robot, seed=seed)

    # A fake clock, so the run is not gated on wall time and is reproducible.
    now = [0.0]
    runtime = HardwareRuntime(
        controller=SystematicCoverage(),
        robot=robot,
        source=source,
        actuate=lambda command: None,  # the trajectory is the plant; it does not listen
        watchdog=Watchdog(SafetyLimits(required=("encoder", "imu"))),
        rate_hz=rate_hz,
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + max(seconds, 0.0)),
        name=path.stem,
    )
    run = runtime.run(seconds=trajectory.duration - 1.0)

    frames = run.recording.frames
    t = frames["time"]
    error = np.hypot(
        frames["ctl.est_x"] - np.interp(t, trajectory.time, trajectory.x),
        frames["ctl.est_y"] - np.interp(t, trajectory.time, trajectory.y),
    )
    heading_error = np.abs(
        np.arctan2(
            np.sin(frames["ctl.est_heading"] - np.interp(t, trajectory.time, trajectory.heading)),
            np.cos(frames["ctl.est_heading"] - np.interp(t, trajectory.time, trajectory.heading)),
        )
    )
    return trajectory, run, error, heading_error


def viewing_frame(trajectory, margin: float = 1.0):
    """A rectangle around where the robot went, so the run can be drawn.

    Not a model of the building. The trajectory says where the robot was and
    nothing about the walls, and inventing them would put geometry into a
    recording that carries ``ground_truth: false`` for good reason.
    """
    from shapely.geometry import box

    from zimablue.pool import Pool

    min_x, min_y, max_x, max_y = trajectory.extent()
    return Pool(
        box(min_x - margin, min_y - margin, max_x + margin, max_y + margin),
        depth=1.5,
        name="viewing_frame",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--save", type=Path, default=None, help="write each run as a .zbr here")
    args = parser.parse_args(argv)

    logs = sorted(args.dir.glob("*.txt"))
    if not logs:
        raise SystemExit(f"no trajectories in {args.dir}. Run tools/fetch_trajectory.py first.")

    print(f"{'log':<15}{'drove':>8}{'time':>8}{'final':>8}{'mean':>8}{'worst':>8}{'heading':>9}")
    print(f"{'':<15}{'m':>8}{'s':>8}{'m':>8}{'m':>8}{'m':>8}{'deg':>9}")
    print("-" * 64)
    for log in logs:
        trajectory, run, error, heading_error = score(log)
        print(
            f"{log.stem:<15}{trajectory.path_length:>8.1f}{trajectory.duration:>8.0f}"
            f"{error[-1]:>8.2f}{error.mean():>8.2f}{error.max():>8.2f}"
            f"{np.degrees(heading_error[-1]):>9.1f}"
        )
        if run.watchdog_reasons:
            print(f"{'':<15}watchdog: {run.watchdog_reasons[0]}")
        if args.save is not None:
            args.save.mkdir(parents=True, exist_ok=True)
            # Replay has to draw the run against something, and a robot does
            # not know the shape of what it is in -- so this is a rectangle
            # around where it went, which is a viewing frame and not a claim
            # about the room. The recording still says pose_source: estimate.
            run.recording.manifest["pool_config"] = viewing_frame(trajectory).to_dict()
            run.save(args.save / f"{log.stem}.zbr")

    print(
        "\nFor scale: the pose estimate over a 25-minute simulated kidney-pool run\n"
        "drifts 13.7 m, and 3.8 m once the odometry is calibrated. That is not the\n"
        "same measurement -- the simulator models track slip and this does not --\n"
        "but it does say the estimator handles real motion without falling over,\n"
        "and that the drift it shows in simulation comes from the slip model rather\n"
        "than from real trajectories being hard to integrate."
    )
    if args.save is not None:
        print(f"\nRecordings in {args.save}. They replay: zimablue replay {args.save}/*.zbr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
