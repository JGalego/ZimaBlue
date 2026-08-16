#!/usr/bin/env python3
"""Run the map-building controller and render the replay as a GIF.

    python examples/estimation_replay.py
    python examples/estimation_replay.py --minutes 30 --pool l_shaped

This is the end-to-end path in one file: build a dirty pool, run a cleaner that
estimates its own pose and maps as it goes, record everything, score it, and
turn the recording into something you can watch.

The thing to watch for in the output is the **amber ghost**: where the robot
believes it is. It starts on top of the robot and drifts away over the run,
because dead reckoning has no absolute reference. The dashed line between the
two is the estimation error, and the HUD prints it in metres.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import zimablue as zb
from zimablue.controllers import SystematicCoverage
from zimablue.geometry import wrap_angle
from zimablue.replay import export_movie, export_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--dirt", default="autumn")
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--speed", type=float, default=260.0, help="GIF playback rate")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    controller = SystematicCoverage()
    sim = zb.Simulation(
        pool=args.pool,
        robot="tracked",
        dirt=args.dirt,
        controller=controller,
        seed=args.seed,
        scenario_name=f"estimation_{args.pool}",
    )

    print(f"pool  {sim.pool}")
    print(f"robot {sim.robot.describe()}")
    print(f"dirt  {sim.world.dirt.initial_mass:.0f} g to remove")
    print(f"\nrunning {args.minutes:g} simulated minutes...")

    start_pose = (sim.state.x, sim.state.y, sim.state.heading)
    began = time.perf_counter()
    result = sim.run(minutes=args.minutes)
    elapsed = time.perf_counter() - began
    print(f"done in {elapsed:.1f} s wall clock ({args.minutes * 60 / elapsed:.0f}x real time)\n")

    print(result.metrics.summary())

    # How well did the robot know where it was?
    estimate = controller.estimator.estimate
    x0, y0, h0 = start_pose
    wx = x0 + estimate.x * np.cos(h0) - estimate.y * np.sin(h0)
    wy = y0 + estimate.x * np.sin(h0) + estimate.y * np.cos(h0)
    error = float(np.hypot(wx - sim.state.x, wy - sim.state.y))
    heading_error = float(wrap_angle(estimate.heading + h0 - sim.state.heading))

    print("\nstate estimation")
    print(f"  final position error   {error:6.2f} m")
    print(f"  final heading error    {np.degrees(heading_error):6.1f} deg")
    print(f"  filter's own estimate  {estimate.position_sigma:6.2f} m (1 sigma)")
    print(f"  gyro bias recovered    {np.degrees(estimate.gyro_bias):6.2f} deg/s")
    print(f"  zero-velocity updates  {estimate.zupt_count:6d}")
    print(f"  map cells explored     {controller.map.explored_cells:6d}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"estimation_{args.pool}_{args.seed}"
    recording_path = result.save(args.out / f"{stem}.zbr")
    print(f"\nrecorded {recording_path} ({recording_path.stat().st_size / 1e6:.1f} MB)")

    summary = export_summary(result.recording, args.out / f"{stem}_summary.png")
    print(f"summary  {summary}")

    if not args.no_gif:
        gif = args.out / f"{stem}.gif"
        print("rendering GIF...")
        export_movie(result.recording, gif, speed=args.speed, fps=14, dpi=52)
        _shrink(gif)
        print(f"gif      {gif} ({gif.stat().st_size / 1e6:.1f} MB)")

    print(f"\nwatch it live:  zimablue replay {recording_path}")


def _shrink(path: Path, colours: int = 96) -> None:
    """Re-quantise the GIF.

    matplotlib's writer keeps a full palette per frame, which triples the file
    size for no visible gain at this resolution.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with matplotlib
        return
    image = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(image.convert("RGB").quantize(colors=colours))
            image.seek(image.tell() + 1)
    except EOFError:
        pass
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=image.info.get("duration", 70),
            loop=0,
            optimize=True,
        )


if __name__ == "__main__":
    main()
