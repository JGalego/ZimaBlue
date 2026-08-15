"""Replay: watch a recorded run.

matplotlib is an optional extra (``pip install zimablue[viz]``); importing this
package is cheap and only the functions that draw actually import it.
"""

from __future__ import annotations

from zimablue.replay.player import (
    SPEEDS,
    ReplayPlayer,
    export_frames,
    export_movie,
    export_summary,
)
from zimablue.replay.renderer import PALETTE, ReplayRenderer, load_scene

__all__ = [
    "PALETTE",
    "SPEEDS",
    "ReplayPlayer",
    "ReplayRenderer",
    "export_frames",
    "export_movie",
    "export_summary",
    "load_scene",
]
