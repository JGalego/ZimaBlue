"""A camera looking at the pool floor, without a 3D engine.

Inverse perspective mapping, the technique behind a driving game's road view.
For every output pixel, cast a ray from the camera through it, intersect that
ray with the floor plane, and sample the dirt raster where it lands.  No mesh,
no z-buffer, no renderer -- one vectorised NumPy expression per frame over a
grid of rays.

Two cameras are built on this and they differ in exactly one thing: where the
camera sits.  Bolt it to the bumper 18 cm off the floor and you get
:class:`~zimablue.replay.dirtcam.DirtCam`, the view the robot has.  Float it a
metre behind and half a metre up and you get
:class:`~zimablue.replay.chasecam.ChaseCam`, the view a diver following the
robot would have.  Everything else -- the ray grid, the projection, the floor
texture, the turbidity, the debris -- is shared, and has to be: two cameras
that disagreed about how far a leaf was would be two different simulators.

The floor is treated as flat for the *ray geometry* while the colour comes from
the real depth and dirt fields. In a pool whose floor slopes two metres over
twelve, that approximation costs a little foreshortening accuracy at the far
edge of frame and saves solving a ray-surface intersection per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from zimablue.replay.renderer import PALETTE, Scene, load_scene

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["FloorCamConfig", "FloorCamera", "rgb"]

FloatArray = NDArray[np.float64]


def rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass(frozen=True)
class FloorCamConfig:
    """Where the camera sits and what it can see."""

    width: int = 320
    height: int = 180

    camera_height: float = 0.18
    """Metres above the floor."""

    pitch: float = 0.38
    """Downward tilt in radians."""

    fov: float = 1.65
    """Horizontal field of view in radians (~95 degrees)."""

    far: float = 4.0
    """Metres beyond which everything fades into the water.

    Underwater visibility is short and turbidity makes it shorter. Pushing this
    out does not buy detail -- past a few metres a low camera compresses
    everything into a band a handful of pixels tall.
    """

    tile: float = 0.25
    """Grout pitch in metres. The floor pattern is what carries the sense of
    speed: without something world-locked sliding past, a colour field just
    changes shade and the view reads as a gradient rather than as motion."""

    def aspect(self) -> float:
        return self.height / self.width


class FloorCamera:
    """Renders the pool floor from a moving camera.

    The ray grid depends only on the configuration, so it is built once and
    reused for every frame; per frame the work is a rotation, a translation and
    two raster lookups.

    Subclasses override :meth:`camera_pose` to say where the camera is, and
    :meth:`draw_overlays` to put things in front of the floor.
    """

    def __init__(self, recording: Recording, config: FloorCamConfig | None = None) -> None:
        self.recording = recording
        self.config = config or FloorCamConfig()
        self.scene: Scene = load_scene(recording)
        self._debris_outlines: dict[int, FloatArray] | None = None
        self._debris_colours: dict[int, FloatArray] = {}
        self._build_rays()

        dirt0 = recording.dirt_at(0.0)
        positive = dirt0[dirt0 > 0] if dirt0.size else np.zeros(0)
        self._dirt_max = float(np.percentile(positive, 92)) if positive.size else 1.0

    # ------------------------------------------------------------------
    def _build_rays(self) -> None:
        """Ground coordinates, in the camera's frame, for every output pixel.

        Computed once: the pattern of where each pixel lands on the floor never
        changes relative to the camera -- only where that pattern is placed in
        the world.
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
    def robot_pose(self, index: int) -> tuple[float, float, float]:
        """Where the robot is at this frame."""
        frames = self.recording.frames
        index = int(np.clip(index, 0, self.recording.n_frames - 1))
        return (
            float(frames["x"][index]),
            float(frames["y"][index]),
            float(frames["heading"][index]),
        )

    def camera_pose(self, index: int) -> tuple[float, float, float]:
        """Where the camera is. Bolted to the robot unless overridden."""
        return self.robot_pose(index)

    def draw_overlays(self, image: FloatArray, index: int) -> None:
        """Anything drawn in front of the floor. Nothing, by default."""

    # ------------------------------------------------------------------
    def frame(self, index: int) -> NDArray[np.float64]:
        """Render one frame as an RGB array in ``[0, 1]``."""
        rec = self.recording
        index = int(np.clip(index, 0, rec.n_frames - 1))
        t = float(rec.frames["time"][index])
        cx, cy, yaw = self.camera_pose(index)

        # Place the ray pattern in the world.
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        world_x = cx + self._ahead * cos_h - self._lateral * sin_h
        world_y = cy + self._ahead * sin_h + self._lateral * cos_h

        grid = self.scene.grid
        rows = np.clip(((world_y - grid.miny) / grid.cell).astype(int), 0, grid.nrows - 1)
        cols = np.clip(((world_x - grid.minx) / grid.cell).astype(int), 0, grid.ncols - 1)

        inside = self.scene.navigable[rows, cols] & ~self.sky
        dirt = np.asarray(rec.dirt_at(t, interpolate=True))[rows, cols]

        image = self._paint(dirt, inside, world_x, world_y)
        self._draw_debris(image, rec, t, cx, cy, yaw)
        self.draw_overlays(image, index)
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
        floor = np.array(rgb(PALETTE["shallow"]))
        filth = np.array(rgb(PALETTE["dirt"]))
        water = np.array(rgb(PALETTE["deep"]))

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

    # ------------------------------------------------------------------
    def project(
        self, ahead: FloatArray, lateral: FloatArray, height: float | FloatArray = 0.0
    ) -> tuple[FloatArray, FloatArray]:
        """Points in the camera's frame to pixel coordinates.

        The exact inverse of :meth:`_build_rays`, generalised to points off the
        floor: ``height`` is metres above it, so a point on top of the robot
        projects where the top of the robot should be drawn. Solving the ray
        equations for the parameter gives ``t = ahead*cos(pitch) -
        rise*sin(pitch)`` with ``rise = height - camera_height``, and the screen
        coordinates fall out of that. At ``height=0`` it reduces exactly to the
        floor-only form it replaces.
        """
        cfg = self.config
        half_w = np.tan(cfg.fov / 2.0)
        half_h = half_w * cfg.aspect()
        cos_p, sin_p = np.cos(cfg.pitch), np.sin(cfg.pitch)

        rise = np.asarray(height, dtype=float) - cfg.camera_height
        t = np.maximum(ahead * cos_p - rise * sin_p, 1e-6)
        screen_x = lateral / t
        screen_y = (ahead * sin_p + rise * cos_p) / t
        cols = (screen_x / half_w * 0.5 + 0.5) * (cfg.width - 1)
        rows = (0.5 - screen_y / half_h * 0.5) * (cfg.height - 1)
        return cols, rows

    def to_camera(
        self, world_x: FloatArray, world_y: FloatArray, cx: float, cy: float, yaw: float
    ) -> tuple[FloatArray, FloatArray]:
        """World points to the camera's ahead/lateral frame."""
        cos_h, sin_h = np.cos(-yaw), np.sin(-yaw)
        dx, dy = np.asarray(world_x) - cx, np.asarray(world_y) - cy
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    # ------------------------------------------------------------------
    def _draw_debris(
        self, image: FloatArray, rec: Recording, t: float, cx: float, cy: float, yaw: float
    ) -> None:
        """Leaves and twigs, as their own outlines lying on the floor.

        Each item's silhouette is built in world coordinates and every vertex
        put through the same projection as the floor, so a leaf a hand's width
        from the bumper is a foreshortened shape filling the frame and the same
        leaf three metres out is a fleck. Drawing them as flat discs threw that
        away and made a soaked oak leaf indistinguishable from a twig -- which
        matters, because one of those is about to jam the intake.
        """
        debris = rec.debris_at(t, interpolate=True)
        if not debris.size:
            return
        # `collected` is fractional on the interpolated reading: how far
        # through the interval in which the item was picked up. Anything not
        # fully collected is still drawn, fading as it goes.
        indices = np.nonzero(debris[:, 4] < 1.0)[0]
        if not indices.size:
            return

        cfg = self.config
        ahead, _ = self.to_camera(debris[indices, 0], debris[indices, 1], cx, cy, yaw)

        visible = (ahead > 0.05) & (ahead < cfg.far)
        if not visible.any():
            return
        # Painter's algorithm: without it a leaf four metres out can be drawn
        # over one under the bumper.
        order = np.argsort(-ahead[visible])
        selected = indices[visible][order]

        offsets = self._outlines()
        for item in selected:
            offset = offsets.get(int(item))
            if offset is None:
                continue
            # Placed at where the item is now, not where it started: the robot
            # shoves oversized debris around, and the survivors of a run have
            # typically moved further than their own length.
            polygon = offset + debris[item, 0:2]
            a, lat = self.to_camera(polygon[:, 0], polygon[:, 1], cx, cy, yaw)
            cols, rows = self.project(a, lat)
            distance = float(np.hypot(*(polygon.mean(axis=0) - (cx, cy))))
            self.fill_polygon(
                image,
                cols,
                rows,
                self._debris_colours[int(item)],
                distance,
                alpha=float(np.clip(1.0 - debris[item, 4], 0.0, 1.0)),
            )

    def _outlines(self) -> dict[int, FloatArray]:
        """Outlines about each item's own centre, built once.

        Shape, rotation, scale and colour are fixed per item and worth caching.
        The centre is not -- see :meth:`_draw_debris`.
        """
        if self._debris_outlines is not None:
            return self._debris_outlines

        from zimablue.replay.debris_shapes import debris_colour, debris_offsets

        first = self.recording.debris_at(0.0)
        self._debris_outlines = {}
        self._debris_colours = {}
        if first.size:
            names = self.recording.debris_type_names()
            types = np.clip(first[:, 5].astype(int), 0, max(len(names) - 1, 0))
            kinds = [names[k] for k in types]
            offsets = debris_offsets(first[:, 3], kinds, np.arange(len(first)))
            for i, (offset, kind) in enumerate(zip(offsets, kinds, strict=True)):
                self._debris_outlines[i] = offset
                self._debris_colours[i] = np.array(rgb(debris_colour(kind, i)))
        return self._debris_outlines

    # ------------------------------------------------------------------
    def fill_polygon(
        self,
        image: FloatArray,
        cols: FloatArray,
        rows: FloatArray,
        colour: FloatArray,
        distance: float,
        *,
        alpha: float = 1.0,
        fade: float | None = None,
    ) -> None:
        """Scanline-fill a screen polygon, faded by how far away it is.

        An even-odd crossing test over the polygon's bounding box. Small enough
        to keep the module free of a drawing library, and the boxes are a few
        dozen pixels except for whatever is directly under the camera.
        """
        cfg = self.config
        cols = np.asarray(cols, dtype=float)
        rows = np.asarray(rows, dtype=float)
        if not (np.isfinite(cols).all() and np.isfinite(rows).all()):
            return
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
        blend = (1.0 - min(distance / cfg.far, 1.0) * 0.7 if fade is None else fade) * alpha
        patch = image[r0 : r1 + 1, c0 : c1 + 1]
        patch[inside] = colour * blend + patch[inside] * (1.0 - blend)
