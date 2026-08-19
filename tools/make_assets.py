#!/usr/bin/env python3
"""Regenerate the images the README points at.

    python tools/make_assets.py            # everything
    python tools/make_assets.py replay     # just one

Every asset here is produced from a real simulation rather than drawn by hand,
which is the point: if a change makes the renderer lie, the README stops
matching the code and this script is what catches it. The runs are seeded, so
rerunning this reproduces the same pictures.

The logo is generated separately by ``tools/make_logo.py`` -- it is the only
asset built from geometry rather than from a recording.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import zimablue as zb
from zimablue.recording import Recording

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
CACHE = ROOT / "runs" / "assets"

# The headline run. These numbers appear verbatim in the README caption, so
# changing one here means changing it there.
KIDNEY = {"pool": "kidney", "dirt": "autumn", "seed": 42, "minutes": 25.0}
REPLAY_SPEED = 260.0

# Close-up cameras are limited by `speed / fps` -- the simulated seconds
# between one displayed frame and the next -- rather than by playback speed on
# its own. The robot covers its own length in about a second, so past two or
# three seconds a frame the floor arrives already swept and dirt reads as
# popping into existence. The top-down view has no such limit: it shows the
# whole pool and nothing in it moves far between frames, which is why that one
# runs the entire clean at 260x and these two show a window.
CLOSE_UP_SPEED = 24.0
CLOSE_UP_FPS = 12

SLOPED = {"pool": "sloped", "dirt": "autumn", "seed": 7, "minutes": 12.0}


def recording_for(name: str, **spec: object) -> Recording:
    """Run a scenario once and keep the ``.zbr`` so reruns are cheap."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.zbr"
    if path.exists():
        print(f"  reusing {path.relative_to(ROOT)}")
        return Recording.load(path)

    minutes = float(spec.pop("minutes"))  # type: ignore[arg-type]
    print(f"  simulating {minutes:.0f} minutes...")
    result = zb.Simulation(**spec).run(minutes=minutes)  # type: ignore[arg-type]
    print("   ", result.metrics.summary().strip().splitlines()[0].strip())
    result.save(path)
    return result.recording


# ----------------------------------------------------------------------
def make_replay() -> None:
    """The animation at the top of the README, plus the summary sheet."""
    from zimablue.replay import export_movie, export_summary

    rec = recording_for("kidney", **KIDNEY)
    export_movie(rec, ASSETS / "replay.gif", speed=REPLAY_SPEED, fps=12, dpi=52)
    export_summary(rec, ASSETS / "summary.png")


def make_dirtcam() -> None:
    """The bumper view, with the top-down panel beside it.

    The first two minutes, at two simulated seconds a displayed frame. Both
    numbers matter: a camera this close needs frames near enough in time to
    show a patch *being* swept rather than already swept, and at that rate the
    whole twenty-five minutes is not a GIF anyone would download. The opening
    is the stretch worth having anyway -- the floor starts as a silt plain and
    you watch the first lanes cut through it.
    """
    from zimablue.replay import export_dirtcam

    rec = recording_for("kidney", **KIDNEY)
    export_dirtcam(
        rec, ASSETS / "dirtcam.gif", speed=CLOSE_UP_SPEED, fps=CLOSE_UP_FPS, seconds=120.0, dpi=52
    )


def make_chase() -> None:
    """The chase cam, from a metre behind the robot. Same window, same rate."""
    from zimablue.replay import export_chasecam

    rec = recording_for("kidney", **KIDNEY)
    export_chasecam(
        rec, ASSETS / "chase.gif", speed=CLOSE_UP_SPEED, fps=CLOSE_UP_FPS, seconds=120.0, dpi=54
    )


def make_3d() -> None:
    """The basin renders: an orbit of the sloped pool, and a kidney sheet."""
    from zimablue.replay import export_3d_frames, export_3d_movie

    sloped = recording_for("sloped", **SLOPED)
    export_3d_frames(sloped, ASSETS / "3d-sloped.png", count=2)
    export_3d_movie(sloped, ASSETS / "3d-sloped.gif", speed=200.0, fps=12, dpi=54)

    export_3d_frames(recording_for("kidney", **KIDNEY), ASSETS / "3d-kidney.png", count=4)


TARGETS = {
    "replay": make_replay,
    "dirtcam": make_dirtcam,
    "chase": make_chase,
    "3d": make_3d,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets", nargs="*", metavar="TARGET", help=f"any of: {', '.join(TARGETS)}"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="re-simulate instead of reusing cached recordings"
    )
    args = parser.parse_args()

    unknown = [t for t in args.targets if t not in TARGETS]
    if unknown:
        parser.error(f"unknown target(s) {', '.join(unknown)} -- pick from {', '.join(TARGETS)}")

    if args.fresh:
        for stale in CACHE.glob("*.zbr"):
            stale.unlink()

    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in args.targets or list(TARGETS):
        print(f"\n{name}")
        started = time.perf_counter()
        TARGETS[name]()
        print(f"  took {time.perf_counter() - started:.0f}s")

    print()
    for asset in sorted(ASSETS.iterdir()):
        print(f"  {asset.stat().st_size / 1e6:6.2f} MB  {asset.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
