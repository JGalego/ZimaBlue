"""Replay: watch a recorded run.

matplotlib is an optional extra (``pip install zimablue[viz]``); importing this
package is cheap and only the functions that draw actually import it.
"""

from __future__ import annotations

from zimablue.replay._deps import VIZ_HINT, require_matplotlib
from zimablue.replay.dirtcam import (
    DirtCam,
    DirtCamConfig,
    export_dirtcam,
    export_dirtcam_frames,
    render_dirtcam,
)
from zimablue.replay.player import (
    SPEEDS,
    ReplayPlayer,
    export_frames,
    export_movie,
    export_summary,
)
from zimablue.replay.renderer import PALETTE, ReplayRenderer, load_scene
from zimablue.replay.scene3d import (
    Scene3D,
    export_3d_frames,
    export_3d_movie,
    render_3d,
)

__all__ = [
    "PALETTE",
    "SPEEDS",
    "VIZ_HINT",
    "DirtCam",
    "DirtCamConfig",
    "ReplayPlayer",
    "ReplayRenderer",
    "Scene3D",
    "export_3d_frames",
    "export_3d_movie",
    "export_dirtcam",
    "export_dirtcam_frames",
    "export_frames",
    "export_movie",
    "export_summary",
    "load_scene",
    "render_3d",
    "render_dirtcam",
    "require_matplotlib",
]
