#!/usr/bin/env python3
"""Record a run, then replay it from every camera the library has.

    python examples/replay.py            # summary image
    python examples/replay.py --gif      # animated GIF too
    python examples/replay.py --3d       # the pool as a basin with real depth
    python examples/replay.py --chase    # from a metre behind the cleaner
    python examples/replay.py --dirtcam  # the view from the cleaner itself
    python examples/replay.py --watch    # open the interactive player

A recording carries its own pool geometry and robot configuration, so every
one of these renders from the file alone -- no access to the code that
produced it, and no dependence on presets that may have changed since.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import zimablue as zb
from zimablue.recording import Recording
from zimablue.replay import (
    export_3d_frames,
    export_3d_movie,
    export_chasecam,
    export_chasecam_frames,
    export_dirtcam,
    export_dirtcam_frames,
    export_movie,
    export_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="open the interactive player")
    parser.add_argument("--gif", action="store_true", help="also render an animated GIF")
    parser.add_argument(
        "--3d",
        dest="three_d",
        action="store_true",
        help="render the pool as a 3D basin built from its depth model",
    )
    parser.add_argument(
        "--chase",
        action="store_true",
        help="render from a metre behind the cleaner, with the cleaner in shot",
    )
    parser.add_argument(
        "--dirtcam",
        action="store_true",
        help="render the view from the cleaner's own bumper",
    )
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    path = args.out / "replay_example.zbr"
    if path.exists():
        print(f"reusing {path}")
        recording = Recording.load(path)
    else:
        print(f"running a {args.minutes:.0f}-minute kidney-pool clean...")
        result = zb.Simulation(pool="kidney", dirt="autumn", seed=42).run(minutes=args.minutes)
        print(result.metrics.summary())
        result.save(path)
        recording = result.recording
        print(f"\nsaved {path}")

    # A recording is self-describing: everything needed to replay it is inside.
    print()
    print(recording.describe())

    summary = args.out / "replay_example_summary.png"
    export_summary(recording, summary)
    print(f"\nsummary  {summary}")

    if args.gif:
        gif = args.out / "replay_example.gif"
        print("rendering GIF (this takes a moment)...")
        export_movie(recording, gif, speed=90.0)
        print(f"gif      {gif}")

    if args.three_d:
        # 3D rendering, not 3D simulation: the motion came from the 2D backend.
        # What is genuinely three-dimensional is the geometry -- the floor is a
        # surface sampled from the pool's depth model, so the robot sits deeper
        # at the deep end because the pool really is deeper there.
        sheet = args.out / "replay_example_3d.png"
        export_3d_frames(recording, sheet, count=4)
        print(f"3d       {sheet}")
        if args.gif:
            gif3d = args.out / "replay_example_3d.gif"
            print("rendering the 3D animation (slower -- it rebuilds the floor each frame)...")
            export_3d_movie(recording, gif3d, speed=200.0, fps=12, dpi=54)
            print(f"3d gif   {gif3d}")

    if args.chase:
        # The camera unbolted from the robot and floated behind it, so the
        # machine itself is in shot -- drawn from its own design.
        sheet = args.out / "replay_example_chase.png"
        export_chasecam_frames(recording, sheet, count=3)
        print(f"chase    {sheet}")
        if args.gif:
            chase_gif = args.out / "replay_example_chase.gif"
            print("rendering the chase cam...")
            export_chasecam(recording, chase_gif, speed=120.0)
            print(f"chase gif {chase_gif}")

    if args.dirtcam:
        # Same recording, same dirt field, a camera 18 cm off the floor. The
        # top-down panel stays alongside because the disagreement between the
        # two is the interesting part.
        sheet = args.out / "replay_example_dirtcam.png"
        export_dirtcam_frames(recording, sheet)
        print(f"dirtcam  {sheet}")
        if args.gif:
            cam_gif = args.out / "replay_example_dirtcam.gif"
            print("rendering the dirt cam...")
            export_dirtcam(recording, cam_gif, speed=140.0)
            print(f"cam gif  {cam_gif}")

    if args.watch:
        from zimablue.replay import ReplayPlayer

        print("\nspace pause · ←/→ step · ↑/↓ speed · r restart · q quit")
        ReplayPlayer(recording, speed=8.0).show()
    else:
        print(f"\nwatch it:  zimablue replay {path}")


if __name__ == "__main__":
    main()
