#!/usr/bin/env python3
"""Download a real robot's trajectory to check this library's numbers against.

Nothing in the test suite needs a network, and nothing here is bundled: the
files belong to their authors, they are megabytes, and a package that ships
somebody else's dataset is a licensing problem waiting to happen.  Run this
once and the real-data checks in ``tests/test_hardware_real.py`` start
running instead of skipping.

    python tools/fetch_trajectory.py

Default is a Pioneer 3-DX driving a real building, tracked by a real motion
capture rig at about 300 Hz, from the TUM RGB-D benchmark.  A differential
drive of roughly a pool cleaner's mass, moving at roughly its speed, over
roughly a pool's worth of floor -- which is as close an analogue as a public
dataset gets to a machine nobody has instrumented.

If you cite anything measured with it, cite them:

    J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers,
    "A Benchmark for the Evaluation of RGB-D SLAM Systems",
    IROS 2012.  https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://cvg.cit.tum.de/rgbd/dataset"

SEQUENCES = {
    "pioneer_slam": (
        f"{BASE}/freiburg2/rgbd_dataset_freiburg2_pioneer_slam-groundtruth.txt",
        "Pioneer 3-DX, large loop through an office floor, 156 s",
    ),
    "pioneer_slam2": (
        f"{BASE}/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2-groundtruth.txt",
        "Pioneer 3-DX, second run, tighter turns",
    ),
    "pioneer_360": (
        f"{BASE}/freiburg2/rgbd_dataset_freiburg2_pioneer_360-groundtruth.txt",
        "Pioneer 3-DX, mostly rotation in place",
    ),
}

DEFAULT_DIR = Path("data/trajectories")


def fetch(name: str, directory: Path) -> Path:
    url, description = SEQUENCES[name]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.txt"
    if target.exists():
        print(f"{target} already here ({target.stat().st_size // 1024} KB)")
        return target

    print(f"{name}: {description}")
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if not payload.lstrip().startswith(b"#"):
        raise SystemExit(f"{url} did not return a TUM trajectory; got {payload[:80]!r}")
    target.write_bytes(payload)
    print(f"  -> {target} ({len(payload) // 1024} KB)")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", nargs="?", default="pioneer_slam", choices=sorted(SEQUENCES))
    parser.add_argument("--all", action="store_true", help="fetch every sequence")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)

    names = sorted(SEQUENCES) if args.all else [args.sequence]
    for name in names:
        fetch(name, args.dir)

    print("\nNow run:  pytest tests/test_hardware_real.py -v")
    print("      or:  python examples/replay_real_trajectory.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
