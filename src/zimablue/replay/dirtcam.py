"""Dirt cam -- the view from the cleaner's own front bumper.

Top-down replay is calming. You watch a small machine trace patient lanes
across clean blue water and it is easy to believe the pool is fine. Dirt cam is
the corrective: it puts the camera 12 cm off the floor behind the brush, and
from down there the pool is not calm at all. It is a silt plain with leaves in
it.

(The working name for this was "un-zen view", which is funnier but needs
explaining. "Dirt cam" tells you what you get.)

Both views answer the same question and disagree, which is the point the whole
project keeps making: from above you see *where the robot went*, and from the
bumper you see *what it left behind*.

How it works
------------

Inverse perspective mapping, the technique behind a driving game's road view.
For every output pixel, cast a ray from the camera through it, intersect that
ray with the floor plane, and sample the dirt raster at the point where it
lands. No 3D engine, no mesh, no z-buffer -- one vectorised NumPy expression
per frame over a grid of rays.

The floor is treated as flat for the *ray geometry* while the colour comes from
the real depth and dirt fields. In a pool whose floor slopes by two metres over
twelve, that approximation costs a little foreshortening accuracy at the far
edge of frame and saves solving a ray-surface intersection per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zimablue.replay._deps import require_matplotlib
from zimablue.replay.floorcam import FloorCamConfig, FloorCamera, frame_window
from zimablue.replay.floorcam import rgb as _rgb
from zimablue.replay.renderer import PALETTE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = [
    "DirtCam",
    "DirtCamConfig",
    "export_dirtcam",
    "export_dirtcam_frames",
    "render_dirtcam",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DirtCamConfig(FloorCamConfig):
    """Where the bumper camera sits and what it can see.

    The defaults are the whole character of the view. Low, because that is
    where a cleaner's intake is and because a low camera makes the near field
    enormous, which is the honest impression of driving through silt. Wide,
    like an action camera, so the swath edges stay in frame. Short-sighted,
    because underwater visibility is short and turbidity makes it shorter.
    """

    camera_height: float = 0.18
    """Metres above the floor -- roughly the top of the brush housing."""

    pitch: float = 0.38
    """Downward tilt in radians. Enough that the floor fills most of the frame
    and the horizon sits high."""

    fov: float = 1.65
    """Horizontal field of view in radians (~95 degrees)."""


class DirtCam(FloorCamera):
    """The forward view from the robot's own front bumper.

    A :class:`~zimablue.replay.floorcam.FloorCamera` bolted to the robot, plus
    the one thing that makes it first-person: a bit of the machine intruding at
    the bottom of frame.
    """

    def __init__(self, recording: Recording, config: DirtCamConfig | None = None) -> None:
        super().__init__(recording, config or DirtCamConfig())

    def draw_overlays(self, image: FloatArray, index: int) -> None:
        self._draw_hull(image)

    def _draw_hull(self, image: FloatArray) -> None:
        """The robot's own brush housing, intruding at the bottom of frame.

        Every action camera sees a bit of the thing it is bolted to, and it
        anchors the viewer: without it the image reads as a floating drone
        rather than a machine dragging a brush through silt.
        """
        cfg = self.config
        design = self.scene.design
        hull = np.array(_rgb(design.hull))

        # A curved top edge, low in the middle and rising at the corners, the
        # way a rounded housing looks from a camera sitting on it. A straight
        # bar reads as the image having been cropped.
        cols = np.arange(cfg.width)
        nx = (cols / (cfg.width - 1) - 0.5) * 2.0
        top = cfg.height - (cfg.height * (0.11 + 0.055 * nx**2))

        rows = np.arange(cfg.height)[:, None]
        body = rows >= top[None, :]
        image[body] = hull

        # The brush roller: a lighter band along the leading edge, with bristle
        # ticks so it turns rather than sits there. Coloured from the design,
        # so a machine with a cyan roller has a cyan roller here too.
        roller = cfg.height * 0.045
        band = body & (rows < (top[None, :] + roller))
        image[band] = np.array(_rgb(design.trim))
        bristles = band & (np.broadcast_to(cols, band.shape) % 7 == 0)
        image[bristles] = np.array(_rgb(design.accent)) * 0.8


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------
def render_dirtcam(
    recording: Recording,
    index: int,
    *,
    ax: Any = None,
    camera: DirtCam | None = None,
    config: DirtCamConfig | None = None,
) -> Any:
    """Draw the bumper view for one frame onto ``ax``."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    cam = camera or DirtCam(recording, config)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=PALETTE["panel"])
    ax.clear()
    ax.imshow(cam.frame(index), interpolation="bilinear", aspect="auto")
    ax.set_axis_off()
    return ax


def export_dirtcam_frames(
    recording: Recording,
    path: str | Path,
    *,
    count: int = 4,
    config: DirtCamConfig | None = None,
    dpi: int = 130,
) -> Path:
    """A contact sheet of ``count`` bumper views spread across the run.

    The still version of the argument: the first panel is a silt plain and the
    last is tile, and nothing in between required a 3D engine.
    """
    require_matplotlib()
    # Deliberately not switching the global backend to Agg. Writing a file
    # never needed it -- savefig renders through Agg whichever backend is
    # selected -- and matplotlib.use() is process-wide, so exporting one image
    # from a notebook silently unplugged the inline backend and every figure
    # after it came out blank. Nothing here calls plt.show(), so no window
    # opens on a machine that has a display.
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = DirtCam(recording, config)
    count = max(1, count)
    indices = np.linspace(0, recording.n_frames - 1, count).astype(int)

    rows = 1 if count <= 2 else 2
    cols = int(np.ceil(count / rows))
    figure, axes = plt.subplots(
        rows, cols, figsize=(5.4 * cols, 3.1 * rows), facecolor=PALETTE["panel"], squeeze=False
    )
    for ax, index in zip(axes.ravel(), indices, strict=False):
        t = float(recording.frames["time"][index])
        ax.imshow(cam.frame(int(index)), interpolation="bilinear", aspect="auto")
        ax.set_title(
            f"{int(t // 60):02d}:{int(t % 60):02d}",
            color=PALETTE["ink"],
            fontsize=10,
            family="monospace",
        )
        ax.set_axis_off()
    for ax in axes.ravel()[len(indices) :]:
        ax.set_axis_off()

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, facecolor=PALETTE["panel"])
    plt.close(figure)
    return path


def export_dirtcam(
    recording: Recording,
    path: str | Path,
    *,
    speed: float = 24.0,
    fps: int = 12,
    dpi: int = 72,
    with_map: bool = True,
    config: DirtCamConfig | None = None,
    start: float = 0.0,
    seconds: float | None = None,
) -> Path:
    """Render part of the run as a dirt-cam animation.

    ``with_map`` keeps a small top-down panel alongside, which is the version
    worth watching: the two disagree constantly, and the disagreement is the
    interesting part.

    ``start`` and ``seconds`` cut a window out of the run, because a close-up
    camera and a whole twenty-five-minute run do not go together. What decides
    that is ``speed / fps`` -- the simulated seconds between one *displayed*
    frame and the next. The robot clears its own length in about a second, so
    at more than two or three seconds a frame you never see a patch being
    swept, only that it has been: the floor arrives already clean and the dirt
    reads as popping rather than lifting. The default here is two seconds a
    frame; the whole run at that rate is not a GIF anyone wants, which is what
    ``seconds`` is for.

    The top-down view has no such limit and is happy at 260x, because it shows
    the whole pool at once and nothing in it moves a body length between
    frames.
    """
    require_matplotlib()
    # Deliberately not switching the global backend to Agg. Writing a file
    # never needed it -- savefig renders through Agg whichever backend is
    # selected -- and matplotlib.use() is process-wide, so exporting one image
    # from a notebook silently unplugged the inline backend and every figure
    # after it came out blank. Nothing here calls plt.show(), so no window
    # opens on a machine that has a display.
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from zimablue.replay.renderer import ReplayRenderer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = DirtCam(recording, config)
    step = max(1, round(speed / (fps * max(recording.frame_dt, 1e-6))))
    indices = frame_window(recording, start, seconds, step)

    if with_map:
        figure = plt.figure(figsize=(11.0, 4.0), facecolor=PALETTE["panel"])
        cam_ax = figure.add_axes((0.006, 0.02, 0.55, 0.96))
        map_ax = figure.add_axes((0.565, 0.02, 0.43, 0.96))
        # Passing ax= drops the HUD strip: this figure already has a caption
        # and half its width is the camera.
        renderer: ReplayRenderer | None = ReplayRenderer(recording, ax=map_ax, show_sensors=True)
    else:
        figure = plt.figure(figsize=(7.2, 4.05), facecolor=PALETTE["panel"])
        cam_ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        renderer = None

    image = cam_ax.imshow(cam.frame(0), interpolation="bilinear", aspect="auto")
    cam_ax.set_axis_off()
    caption = cam_ax.text(
        0.015,
        0.955,
        "",
        transform=cam_ax.transAxes,
        va="top",
        color=PALETTE["ink"],
        fontsize=9.5,
        family="monospace",
        linespacing=1.5,
    )

    scenario = recording.manifest.get("scenario", {})
    frames = recording.frames
    collected = np.asarray(frames["dirt_collected"], dtype=float)
    total = float(collected[-1]) if collected.size else 0.0

    def draw(n: int) -> tuple:
        i = indices[n]
        image.set_data(cam.frame(i))
        t = float(frames["time"][i])
        share = float(collected[i]) / total if total > 0 else 0.0
        caption.set_text(
            f"dirt cam · {scenario.get('pool', '?')} · {scenario.get('dirt', '?')}\n"
            f"{int(t // 60):02d}:{int(t % 60):02d}   "
            f"{float(frames['distance'][i]):5.1f} m   "
            f"{float(collected[i]):5.0f} g lifted ({share:.0%} of the run)"
        )
        if renderer is not None:
            renderer.draw(i)
        return ()

    animation = FuncAnimation(figure, draw, frames=len(indices), blit=False)
    animation.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return path
