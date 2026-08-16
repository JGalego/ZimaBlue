"""Render a recorded run as a 3D scene.

**This draws the run in three dimensions; it does not simulate in three.** The
motion comes from ``Fast2DBackend`` exactly as before. What is genuinely 3D is
the *geometry*: the floor is a surface built from the pool's depth model, the
walls are extruded from its boundary, and the robot sits at the local floor
depth rather than on a flat plane. When the cleaner works the deep end of a
sloped pool, it is two metres lower on screen than at the shallow end, because
the depth model says so.

That distinction is worth keeping sharp. A 3D *backend* -- buoyancy, contact,
wall climbing, cameras -- is designed but not built; see
``docs/architecture.md``. This module is the visualisation half of that story,
and it is useful on its own: the depth channel has been recorded since the
first release and until now only ever appeared as a number in the HUD.

Every 3D view is built from the recording's embedded pool geometry, so it works
on any ``.zbr`` -- including ones written before this module existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zimablue.replay.renderer import PALETTE, Scene, load_scene

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["Scene3D", "export_3d_frames", "export_3d_movie", "render_3d"]

FloatArray = NDArray[np.float64]

WALL_PANELS = 72
"""Target number of wall panels around the boundary.

A *count*, not a stride. A fixed stride silently destroys simple shapes: the
rectangular pool's boundary has four segments, so a stride of six produced a
single degenerate panel and the pool rendered with no walls at all. Deriving
the stride from the ring length instead means the 512-segment kidney curve gets
decimated and the rectangle keeps all four sides."""

FLOOR_CELL = 0.16
"""Floor mesh resolution in metres. Coarser than the dirt raster on purpose --
this is a surface plot rebuilt every frame, and the dirt is resampled onto it."""


def _dirt_colours(dirt: FloatArray, mask: NDArray[np.bool_]) -> NDArray:
    """Blend clean floor toward dirt brown by local dirt concentration."""
    from matplotlib.colors import to_rgba

    clean = np.array(to_rgba(PALETTE["shallow"]))
    filthy = np.array(to_rgba(PALETTE["dirt"]))
    peak = float(np.nanmax(dirt)) if dirt.size else 0.0
    weight = np.clip(dirt / peak, 0.0, 1.0) ** 0.55 if peak > 0 else np.zeros_like(dirt)

    colours = (
        clean[None, None, :] * (1 - weight[..., None]) + filthy[None, None, :] * weight[..., None]
    )
    # Outside the pool the surface is not drawn at all.
    colours[..., 3] = np.where(mask, 0.97, 0.0)
    return colours


@dataclass
class Scene3D:
    """Static 3D geometry derived once from a recording."""

    scene: Scene
    xs: FloatArray
    ys: FloatArray
    floor_z: FloatArray
    mask: NDArray[np.bool_]
    walls: list[NDArray]
    surface_ring: FloatArray
    max_depth: float
    initial_dirt: float

    @classmethod
    def build(cls, recording: Recording) -> Scene3D:
        scene = load_scene(recording)
        pool = scene.pool

        grid = pool.grid(FLOOR_CELL)
        xs, ys = grid.cell_centers()
        mask = pool.navigable_mask(FLOOR_CELL)
        depth = pool.depth_at(xs, ys)
        # Floor sits at negative z, water surface at zero: the natural frame for
        # a pool, and it makes a sloped basin read correctly without inverting.
        floor_z = np.where(mask, -depth, np.nan)

        ring = np.asarray(pool.boundary.exterior.coords, dtype=float)
        walls = []
        step = max(1, (len(ring) - 1) // WALL_PANELS)
        for i in range(0, len(ring) - 1, step):
            j = min(i + step, len(ring) - 1)
            (x0, y0), (x1, y1) = ring[i], ring[j]
            d0 = float(pool.depth_at(x0, y0))
            d1 = float(pool.depth_at(x1, y1))
            walls.append(np.array([[x0, y0, -d0], [x1, y1, -d1], [x1, y1, 0.0], [x0, y0, 0.0]]))

        first = recording.dirt_at(0.0)
        return cls(
            scene=scene,
            xs=xs,
            ys=ys,
            floor_z=floor_z,
            mask=mask,
            walls=walls,
            surface_ring=ring,
            max_depth=float(pool.max_depth),
            initial_dirt=float(np.asarray(first).sum()) if first.size else 0.0,
        )

    def dirt_on_floor(self, recording: Recording, t: float) -> FloatArray:
        """Resample the dirt raster onto the coarser floor mesh."""
        dirt = recording.dirt_at(t)
        if dirt.size == 0:
            return np.zeros_like(self.floor_z)
        fine = self.scene.grid
        rows = np.clip(((self.ys - fine.miny) / fine.cell).astype(int), 0, fine.nrows - 1)
        cols = np.clip(((self.xs - fine.minx) / fine.cell).astype(int), 0, fine.ncols - 1)
        return np.asarray(dirt)[rows, cols]


def _robot_box(
    x: float, y: float, z: float, heading: float, length: float, width: float, height: float
) -> list[NDArray]:
    """Six faces of an oriented box sitting on the floor."""
    hl, hw = length / 2, width / 2
    corners = np.array(
        [
            [-hl, -hw, 0.0],
            [hl, -hw, 0.0],
            [hl, hw, 0.0],
            [-hl, hw, 0.0],
            [-hl, -hw, height],
            [hl, -hw, height],
            [hl, hw, height],
            [-hl, hw, height],
        ]
    )
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]])
    pts = corners @ rotation.T + np.array([x, y, z])
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],  # front (heading +x in body frame)
        [0, 3, 7, 4],
    ]
    return [pts[face] for face in faces]


def render_3d(
    recording: Recording,
    index: int,
    *,
    ax: Any = None,
    geometry: Scene3D | None = None,
    elev: float = 34.0,
    azim: float = -58.0,
    trail_seconds: float = 240.0,
    show_water: bool = True,
    robot_scale: float = 2.4,
    zoom: float = 1.18,
) -> Any:
    """Draw one frame in 3D onto ``ax`` (created if not supplied)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    geo = geometry or Scene3D.build(recording)
    frames = recording.frames
    index = int(np.clip(index, 0, recording.n_frames - 1))
    t = float(frames["time"][index])

    if ax is None:
        figure = plt.figure(figsize=(9.0, 5.6), facecolor=PALETTE["panel"])
        ax = figure.add_subplot(111, projection="3d")
    ax.clear()
    ax.set_facecolor(PALETTE["panel"])

    # -- floor, coloured by remaining dirt ---------------------------------
    colours = _dirt_colours(geo.dirt_on_floor(recording, t), geo.mask)
    ax.plot_surface(
        geo.xs,
        geo.ys,
        geo.floor_z,
        facecolors=colours,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=True,
    )

    # -- walls --------------------------------------------------------------
    ax.add_collection3d(
        Poly3DCollection(
            geo.walls,
            facecolor=PALETTE["mid"],
            edgecolor="none",
            alpha=0.30,
        )
    )

    # -- water surface ------------------------------------------------------
    if show_water:
        ring = geo.surface_ring
        surface = [np.column_stack([ring[:, 0], ring[:, 1], np.zeros(len(ring))])]
        ax.add_collection3d(
            Poly3DCollection(
                surface, facecolor=PALETTE["shallow"], edgecolor=PALETTE["foam"], alpha=0.13
            )
        )

    # -- path, drawn on the floor it was driven over ------------------------
    start = max(0, index - int(trail_seconds / max(recording.frame_dt, 1e-6)))
    px = frames["x"][start : index + 1]
    py = frames["y"][start : index + 1]
    pz = -frames["depth"][start : index + 1] + 0.02
    if len(px) > 1:
        ax.plot(px, py, pz, color=PALETTE["trail"], linewidth=2.0, alpha=0.9, zorder=15)

    # -- robot --------------------------------------------------------------
    x, y = float(frames["x"][index]), float(frames["y"][index])
    heading = float(frames["heading"][index])
    floor = -float(frames["depth"][index])
    box = _robot_box(x, y, floor, heading, geo.scene.robot_length, geo.scene.robot_width, 0.26)
    ax.add_collection3d(
        Poly3DCollection(box, facecolor=PALETTE["hull"], edgecolor=PALETTE["accent"], linewidth=0.7)
    )

    # -- framing ------------------------------------------------------------
    minx, miny, maxx, maxy = geo.scene.pool.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(-geo.max_depth * 1.05, max(0.6, geo.max_depth * 0.25))
    # Horizontal scale is true; vertical is exaggerated about 3.6x. A 12 m pool
    # 2 m deep renders as a pancake at true scale, and depth is the entire
    # reason for looking at it this way. Called out here so nobody reads a
    # slope off the picture and believes the gradient.
    ax.set_box_aspect((maxx - minx, maxy - miny, geo.max_depth * 3.6), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    scenario = recording.manifest.get("scenario", {})
    name = scenario.get("name", "run") if isinstance(scenario, dict) else str(scenario)
    removed = float(frames["dirt_collected"][index]) if "dirt_collected" in frames else 0.0
    travelled = float(frames["distance"][index]) if "distance" in frames else 0.0
    # Coverage is the 2D view's job -- it owns the visit grid. Here the honest
    # per-frame numbers are the ones actually recorded as channels.
    ax.set_title(
        f"{name}   {int(t // 60):02d}:{int(t % 60):02d}   "
        f"{removed / max(geo.initial_dirt, 1.0):.0%} of the dirt   {travelled:.0f} m driven",
        color=PALETTE["ink"],
        fontsize=9,
        pad=-4,
    )
    return ax


def export_3d_movie(
    recording: Recording,
    path: str | Path,
    *,
    speed: float = 260.0,
    fps: int = 14,
    dpi: int = 58,
    orbit: float = 50.0,
    elev: float = 34.0,
) -> Path:
    """Render the run as an orbiting 3D animation.

    ``orbit`` is how many degrees the camera swings across the whole run. A
    slow drift gives the eye parallax, which is what makes a rendered 3D scene
    read as solid rather than as a flat painting.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    geo = Scene3D.build(recording)
    step = max(1, round(speed / (fps * max(recording.frame_dt, 1e-6))))
    indices = list(range(0, recording.n_frames, step))

    figure = plt.figure(figsize=(9.0, 5.6), facecolor=PALETTE["panel"])
    ax = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0)

    def draw(frame_number: int) -> tuple:
        fraction = frame_number / max(len(indices) - 1, 1)
        render_3d(
            recording,
            indices[frame_number],
            ax=ax,
            geometry=geo,
            elev=elev,
            azim=-58.0 + orbit * fraction,
        )
        return ()

    animation = FuncAnimation(figure, draw, frames=len(indices), blit=False)
    animation.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return path


def export_3d_frames(
    recording: Recording,
    path: str | Path,
    *,
    count: int = 4,
    dpi: int = 100,
    elev: float = 34.0,
) -> Path:
    """A contact sheet of 3D views across the run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geo = Scene3D.build(recording)

    columns = min(count, 2)
    rows = int(np.ceil(count / columns))
    figure = plt.figure(figsize=(6.8 * columns, 4.0 * rows), facecolor=PALETTE["panel"])
    for i, index in enumerate(np.linspace(0, recording.n_frames - 1, count).astype(int)):
        ax = figure.add_subplot(rows, columns, i + 1, projection="3d")
        render_3d(
            recording,
            int(index),
            ax=ax,
            geometry=geo,
            elev=elev,
            azim=-58.0 + 12.0 * i,
        )
    figure.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0, wspace=0.0, hspace=0.06)
    figure.savefig(path, dpi=dpi, facecolor=PALETTE["panel"])
    plt.close(figure)
    return path
