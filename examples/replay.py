#!/usr/bin/env python3
"""Record a run, then replay it.

python examples/replay.py            # render to files
python examples/replay.py --watch    # open the interactive player
"""

from __future__ import annotations

import argparse
from pathlib import Path

import zimablue as zb
from zimablue.recording import Recording
from zimablue.replay import export_movie, export_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="open the interactive player")
    parser.add_argument("--gif", action="store_true", help="also render an animated GIF")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    path = args.out / "replay_example.zbr"
    if path.exists():
        print(f"reusing {path}")
        recording = Recording.load(path)
    else:
        print("running a 15-minute kidney-pool clean...")
        result = zb.Simulation(pool="kidney", dirt="autumn", seed=42).run(minutes=15)
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

    if args.watch:
        from zimablue.replay import ReplayPlayer

        print("\nspace pause · ←/→ step · ↑/↓ speed · r restart · q quit")
        ReplayPlayer(recording, speed=8.0).show()
    else:
        print(f"\nwatch it:  zimablue replay {path}")


if __name__ == "__main__":
    main()
