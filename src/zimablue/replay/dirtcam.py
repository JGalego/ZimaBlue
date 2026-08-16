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
from zimablue.replay.renderer import PALETTE, Scene, load_scene

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
class DirtCamConfig:
    """Where the camera sits and what it can see."""

    width: int = 320
    height: int = 180

    camera_height: float = 0.18
    """Metres above the floor -- roughly the top of the brush housing. Low,
    because that is where a cleaner's intake is, and because a low camera makes
    the near field enormous, which is the honest impression of driving through
    silt."""

    pitch: float = 0.38
    """Downward tilt in radians. Enough that the floor fills most of the frame
    and the horizon sits high."""

    fov: float = 1.65
    """Horizontal field of view in radians (~95 degrees). Wide, like an action
    camera, so the swath edges stay in frame."""

    far: float = 4.0
    """Metres beyond which everything fades into the water. Underwater
    visibility is short and turbidity makes it shorter. Pushing this out does
    not buy detail -- past a few metres a floor-height camera compresses
    everything into a band a handful of pixels tall."""

    tile: float = 0.25
    """Grout pitch in metres. The floor pattern is what carries the sense of
    speed: without something world-locked sliding past, a colour field just
    changes shade and the view reads as a gradient rather than as motion."""

    def aspect(self) -> float:
        return self.height / self.width


class DirtCam:
    """Renders the forward view from a recorded run.

    The ray grid depends only on the configuration, so it is built once and
    reused for every frame; per frame the work is a rotation, a translation and
    two raster lookups.
    """

    def __init__(self, recording: Recording, config: DirtCamConfig | None = None) -> None:
        self.recording = recording
        self.config = config or DirtCamConfig()
        self.scene: Scene = load_scene(recording)
        self._debris_outlines: dict[int, FloatArray] | None = None
        self._debris_colours: dict[int, FloatArray] = {}
        self._build_rays()

        dirt0 = recording.dirt_at(0.0)
        positive = dirt0[dirt0 > 0] if dirt0.size else np.zeros(0)
        self._dirt_max = float(np.percentile(positive, 92)) if positive.size else 1.0

    # ------------------------------------------------------------------
    def _build_rays(self) -> None:
        """Ground coordinates, in the robot's frame, for every output pixel.

        Computed once: the camera is bolted to the robot, so the pattern of
        where each pixel lands on the floor never changes -- only where that
        pattern is placed in the world.
        """
        cfg = self.config
        half_w = np.tan(cfg.fov / 2.0)
        half_h = half_w * cfg.aspect()

        cols = np.linspace(-half_w, half_w, cfg.width)
        rows = np.linspace(half_h, -half_h, cfg.height)
        screen_x, screen_y = np.meshgrid(cols, rows)

        # Ray direction in the camera frame: forward 1, right screen_x, up
        # screen_y, then tilted down by the pitch.
        up = screen_y * np.cos(cfg.pitch) - np.sin(cfg.pitch)
        forward = np.cos(cfg.pitch) + screen_y * np.sin(cfg.pitch)

        # Rays angled upward never meet the floor: that is the horizon. They
        # get a large finite range rather than infinity -- inf would give the
        # centre column ``inf * 0`` and poison the whole frame with NaN, and
        # these pixels are painted as open water anyway.
        beyond = 1e3 * max(cfg.far, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(up < -1e-6, cfg.camera_height / -up, beyond)
        t = np.clip(np.nan_to_num(t, nan=beyond, posinf=beyond), 0.0, beyond)

        self._ahead = t * forward
        self._lateral = t * screen_x
        self._distance = np.hypot(self._ahead, self._lateral)
        self.sky = self._distance > cfg.far

        # How much floor each pixel covers. Near the horizon one pixel spans
        # metres, and a grout line drawn at a fixed width there turns into
        # moire; widening the line to the footprint is a poor man's mipmap.
        span = np.abs(np.gradient(self._ahead, axis=0)) + np.abs(np.gradient(self._lateral, axis=1))
        self._footprint = np.clip(span, 1e-3, cfg.far)

        # A fixed noise tile, sampled in world coordinates, gives the floor a
        # grain that stays put while the robot moves over it.
        self._grain = np.random.default_rng(0xD127CA).random((256, 256))

        # Brightness of the open water above the horizon, normalised over the
        # sky band so the gradient is visible however high the horizon sits.
        horizon = int(self.sky.sum(axis=0).max())
        depth_up = np.arange(cfg.height) / max(horizon - 1, 1)
        self._murk = np.repeat(
            np.clip(1.25 - 0.55 * (1.0 - depth_up), 0.45, 1.3)[:, None], cfg.width, axis=1
        )

    # ------------------------------------------------------------------
    def frame(self, index: int) -> NDArray[np.float64]:
        """Render one frame as an RGB array in ``[0, 1]``."""
        rec = self.recording
        frames = rec.frames
        index = int(np.clip(index, 0, rec.n_frames - 1))
        t = float(frames["time"][index])
        x = float(frames["x"][index])
        y = float(frames["y"][index])
        heading = float(frames["heading"][index])

        # Place the ray pattern in the world.
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        world_x = x + self._ahead * cos_h - self._lateral * sin_h
        world_y = y + self._ahead * sin_h + self._lateral * cos_h

        grid = self.scene.grid
        rows = np.clip(((world_y - grid.miny) / grid.cell).astype(int), 0, grid.nrows - 1)
        cols = np.clip(((world_x - grid.minx) / grid.cell).astype(int), 0, grid.ncols - 1)

        inside = self.scene.navigable[rows, cols] & ~self.sky
        dirt = np.asarray(rec.dirt_at(t))[rows, cols]

        image = self._paint(dirt, inside, world_x, world_y)
        self._draw_debris(image, rec, t, x, y, heading)
        self._draw_hull(image)
        return image

    # ------------------------------------------------------------------
    def _paint(
        self,
        dirt: FloatArray,
        inside: NDArray[np.bool_],
        world_x: FloatArray,
        world_y: FloatArray,
    ) -> FloatArray:
        """Floor, walls and water, shaded by distance."""
        cfg = self.config
        floor = np.array(_rgb(PALETTE["shallow"]))
        filth = np.array(_rgb(PALETTE["dirt"]))
        water = np.array(_rgb(PALETTE["deep"]))

        intensity = np.clip(dirt / max(self._dirt_max, 1e-9), 0.0, 1.0)[..., None]
        image = floor * (1.0 - intensity) + filth * intensity

        # Walls are tiled in the same material as the floor, just turned
        # vertical and in their own shadow. Shading rather than replacing keeps
        # the grout running up them, which is what stops the pool edge from
        # reading as a painted band across the frame.
        shade = np.where(inside[..., None], 1.0, 0.42)
        image = image * shade + water * (1.0 - shade) * 0.55
        image = image * self._floor_texture(world_x, world_y, intensity[..., 0])[..., None]

        # Turbidity: contrast falls off with distance, and the far field
        # dissolves into water rather than ending at a hard line.
        turbidity = float(getattr(self.scene.pool.water, "turbidity", 0.05))
        depth_fade = np.clip(self._distance / cfg.far, 0.0, 1.0)[..., None]
        fog = np.clip(depth_fade ** (1.0 - 0.6 * turbidity), 0.0, 1.0)
        image = image * (1.0 - fog) + water * fog

        # A cheap vignette sells the porthole.
        vignette = self._vignette()[..., None]
        image = image * vignette + water * (1.0 - vignette) * 0.35

        # Above the horizon: open water, brightest where it meets the floor and
        # darkening upward. A flat fill there gives a hard seam that looks like
        # a cropping error rather than like distance.
        image = np.where(self.sky[..., None], water[None, None, :] * self._murk[..., None], image)
        return np.clip(image, 0.0, 1.0)

    def _floor_texture(
        self, world_x: FloatArray, world_y: FloatArray, intensity: FloatArray
    ) -> FloatArray:
        """A brightness multiplier locked to the pool floor, not to the screen.

        Two layers: the grout grid of the tiling, and a silt grain. Both are
        sampled in world coordinates, so they slide past as the robot drives --
        which is the only thing in the frame that conveys speed. Both fade with
        the per-pixel footprint, because detail finer than a pixel can only
        alias.
        """
        cfg = self.config
        footprint = self._footprint

        # --- grout ---------------------------------------------------------
        # Distance to the nearest line of the grid, in metres.
        half = cfg.tile / 2.0
        gap_x = np.abs(np.mod(world_x + half, cfg.tile) - half)
        gap_y = np.abs(np.mod(world_y + half, cfg.tile) - half)
        gap = np.minimum(gap_x, gap_y)
        line_width = np.maximum(0.005, footprint)
        grout = np.clip(1.0 - gap / line_width, 0.0, 1.0)
        # Where a pixel spans several tiles the "line" covers everything, which
        # would darken the far field uniformly; scale it away instead.
        grout *= np.clip(1.0 - footprint / (cfg.tile * 0.7), 0.0, 1.0)

        # --- silt grain ------------------------------------------------------
        n = self._grain.shape[0]
        gx = np.mod((world_x * 42.0).astype(np.int64), n)
        gy = np.mod((world_y * 42.0).astype(np.int64), n)
        grain = self._grain[gy, gx]
        grain_fade = np.clip(1.0 - footprint / 0.05, 0.0, 1.0)

        shade = 1.0 - 0.18 * grout
        # Grain only shows in the dirt: clean tile is smooth.
        shade *= 1.0 + (grain - 0.5) * 0.35 * grain_fade * np.clip(intensity + 0.25, 0.0, 1.0)
        return np.clip(shade, 0.0, 1.4)

    def _vignette(self) -> FloatArray:
        cfg = self.config
        ys, xs = np.mgrid[0 : cfg.height, 0 : cfg.width]
        nx = (xs / (cfg.width - 1) - 0.5) * 2
        ny = (ys / (cfg.height - 1) - 0.5) * 2
        return np.clip(1.15 - 0.45 * (nx**2 + ny**2), 0.0, 1.0)

    def _project(self, ahead: FloatArray, lateral: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Ground points in the robot frame to pixel coordinates.

        The exact inverse of :meth:`_build_rays`. Solving its two equations for
        the ray parameter gives ``t = ahead*cos(pitch) + camera_height*sin
        (pitch)``, and the screen coordinates fall out of that.
        """
        cfg = self.config
        half_w = np.tan(cfg.fov / 2.0)
        half_h = half_w * cfg.aspect()
        cos_p, sin_p = np.cos(cfg.pitch), np.sin(cfg.pitch)

        t = np.maximum(ahead * cos_p + cfg.camera_height * sin_p, 1e-6)
        screen_x = lateral / t
        screen_y = (sin_p - cfg.camera_height / t) / cos_p
        cols = (screen_x / half_w * 0.5 + 0.5) * (cfg.width - 1)
        rows = (0.5 - screen_y / half_h * 0.5) * (cfg.height - 1)
        return cols, rows

    def _draw_debris(
        self, image: FloatArray, rec: Recording, t: float, x: float, y: float, heading: float
    ) -> None:
        """Leaves and twigs, as their own outlines lying on the floor.

        Each item's silhouette is built in world coordinates and every vertex
        put through the same projection as the floor, so a leaf a hand's width
        from the bumper is a foreshortened shape filling the frame and the same
        leaf three metres out is a fleck. Drawing them as flat discs threw that
        away and made a soaked oak leaf indistinguishable from a twig -- which
        matters, because one of those is about to jam the intake.
        """
        debris = rec.debris_at(t)
        if not debris.size:
            return
        indices = np.nonzero(debris[:, 4] < 0.5)[0]
        if not indices.size:
            return

        cfg = self.config
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)
        dx, dy = debris[indices, 0] - x, debris[indices, 1] - y
        ahead = dx * cos_h - dy * sin_h

        visible = (ahead > 0.05) & (ahead < cfg.far)
        if not visible.any():
            return
        # Painter's algorithm: without it a leaf four metres out can be drawn
        # over one under the bumper.
        order = np.argsort(-ahead[visible])
        selected = indices[visible][order]

        outlines = self._outlines()
        for item in selected:
            polygon = outlines.get(int(item))
            if polygon is None:
                continue
            # World outline into the robot frame, then through the projection.
            px, py = polygon[:, 0] - x, polygon[:, 1] - y
            cols, rows = self._project(px * cos_h - py * sin_h, px * sin_h + py * cos_h)
            distance = float(np.hypot(*(polygon.mean(axis=0) - (x, y))))
            self._fill_polygon(image, cols, rows, self._debris_colours[int(item)], distance)

    def _outlines(self) -> dict[int, FloatArray]:
        """World-space outlines for every debris item, built once."""
        if self._debris_outlines is not None:
            return self._debris_outlines

        from zimablue.replay.debris_shapes import debris_colour, debris_polygons

        first = self.recording.debris_at(0.0)
        self._debris_outlines = {}
        self._debris_colours = {}
        if first.size:
            names = self.recording.debris_type_names()
            types = np.clip(first[:, 5].astype(int), 0, max(len(names) - 1, 0))
            kinds = [names[k] for k in types]
            indices = np.arange(len(first))
            polygons = debris_polygons(first[:, 0], first[:, 1], first[:, 3], kinds, indices)
            for i, (polygon, kind) in enumerate(zip(polygons, kinds, strict=True)):
                self._debris_outlines[i] = polygon
                self._debris_colours[i] = np.array(_rgb(debris_colour(kind, i)))
        return self._debris_outlines

    def _fill_polygon(
        self,
        image: FloatArray,
        cols: FloatArray,
        rows: FloatArray,
        colour: FloatArray,
        distance: float,
    ) -> None:
        """Scanline-fill a screen polygon, faded by how far away it is.

        An even-odd crossing test over the polygon's bounding box. Small enough
        to keep the module free of a drawing library, and the boxes are a few
        dozen pixels except for the item directly under the bumper.
        """
        cfg = self.config
        c0 = int(np.floor(max(cols.min(), 0)))
        c1 = int(np.ceil(min(cols.max(), cfg.width - 1)))
        r0 = int(np.floor(max(rows.min(), 0)))
        r1 = int(np.ceil(min(rows.max(), cfg.height - 1)))
        if c1 < c0 or r1 < r0:
            return

        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = np.zeros(rr.shape, dtype=bool)
        n = len(cols)
        for i in range(n):
            j = (i + 1) % n
            yi, yj = rows[i], rows[j]
            straddles = (yi > rr) != (yj > rr)
            with np.errstate(divide="ignore", invalid="ignore"):
                crossing = cols[i] + (rr - yi) / (yj - yi + 1e-12) * (cols[j] - cols[i])
            inside ^= straddles & (cc < crossing)

        if not inside.any():
            return
        fade = 1.0 - min(distance / cfg.far, 1.0) * 0.7
        patch = image[r0 : r1 + 1, c0 : c1 + 1]
        patch[inside] = colour * fade + patch[inside] * (1.0 - fade)

    def _draw_hull(self, image: FloatArray) -> None:
        """The robot's own brush housing, intruding at the bottom of frame.

        Every action camera sees a bit of the thing it is bolted to, and it
        anchors the viewer: without it the image reads as a floating drone
        rather than a machine dragging a brush through silt.
        """
        cfg = self.config
        hull = np.array(_rgb(PALETTE["hull"]))

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
        # ticks so it turns rather than sits there.
        roller = cfg.height * 0.045
        band = body & (rows < (top[None, :] + roller))
        image[band] = hull * 1.9
        bristles = band & (np.broadcast_to(cols, band.shape) % 7 == 0)
        image[bristles] = np.array(_rgb(PALETTE["accent"])) * 0.7


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


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
    import matplotlib

    matplotlib.use("Agg")
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
    speed: float = 240.0,
    fps: int = 14,
    dpi: int = 72,
    with_map: bool = True,
    config: DirtCamConfig | None = None,
) -> Path:
    """Render the run as a dirt-cam animation.

    ``with_map`` keeps a small top-down panel alongside, which is the version
    worth watching: the two disagree constantly, and the disagreement is the
    interesting part.
    """
    require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from zimablue.replay.renderer import ReplayRenderer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = DirtCam(recording, config)
    step = max(1, round(speed / (fps * max(recording.frame_dt, 1e-6))))
    indices = list(range(0, recording.n_frames, step))

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
