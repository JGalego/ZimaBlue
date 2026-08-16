"""Rendering a recorded run.

The brief for this module is "make it obvious what the robot is doing", and the
failure mode to avoid is a debugging spreadsheet.  So: the pool is water-blue,
dirt darkens it, the cleaned trail is bright, and the HUD is a thin strip of
only the numbers a viewer actually tracks.

matplotlib is an optional dependency (``pip install zimablue[viz]``) and is
imported lazily, inside functions, so that headless batch runs never pay for it
and ``import zimablue`` never pulls in a GUI stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zimablue.geometry import Grid
from zimablue.pool import Pool
from zimablue.recording import Recording

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["PALETTE", "ReplayRenderer", "load_scene"]

PALETTE = {
    "deep": "#053a68",
    "mid": "#0e6cb2",
    "shallow": "#54c8ea",
    "foam": "#c8f1fd",
    "coping": "#e9eff4",
    "hull": "#16212e",
    "accent": "#3ddcff",
    "trail": "#7fe9ff",
    "dirt": "#6b5735",
    "warn": "#ffb648",
    "bad": "#ff6b6b",
    "good": "#7ee787",
    "ink": "#dbe7f0",
    "panel": "#08111b",
}
"""Shared with ``tools/make_logo.py`` in spirit: same water, same accent, so the
replay and the project's identity look like the same thing."""


@dataclass
class Scene:
    """Static per-run context the renderer needs, derived once from a recording."""

    pool: Pool
    grid: Grid
    navigable: np.ndarray
    robot_length: float
    robot_width: float
    swath: float
    sonar_angles: tuple[float, ...]
    sonar_max_range: float


def load_scene(recording: Recording) -> Scene:
    """Rebuild the pool and robot geometry embedded in a recording.

    Reads the manifest, not the live presets -- a recording must stay renderable
    after the preset it came from has changed.
    """
    manifest = recording.manifest
    pool = Pool.from_dict(manifest["pool_config"])
    cell = float(manifest.get("cell", 0.10))
    robot_cfg = manifest.get("robot_config", {})
    chassis = robot_cfg.get("chassis", {})
    cleaning = robot_cfg.get("cleaning", {})

    swath = max(
        cleaning.get("brush", {}).get("width", 0.34),
        cleaning.get("pump", {}).get("intake_width", 0.3),
    )
    angles: tuple[float, ...] = ()
    max_range = 3.0
    for sensor in robot_cfg.get("sensors", []):
        if sensor.get("kind") == "sonar":
            angles = tuple(sensor.get("params", {}).get("beam_angles", ()))
            max_range = float(sensor.get("params", {}).get("max_range", 3.0))
            break

    return Scene(
        pool=pool,
        grid=pool.grid(cell),
        navigable=pool.navigable_mask(cell),
        robot_length=float(chassis.get("length", 0.42)),
        robot_width=float(chassis.get("width", 0.38)),
        swath=float(swath),
        sonar_angles=angles,
        sonar_max_range=max_range,
    )


class ReplayRenderer:
    """Draws one frame of a recording onto a matplotlib figure.

    Built once, then :meth:`draw` is called per frame; every artist is created
    up front and only its data is updated, which is what makes scrubbing feel
    immediate rather than redrawing the world each time.
    """

    def __init__(
        self,
        recording: Recording,
        *,
        figsize: tuple[float, float] = (12.0, 7.4),
        show_sensors: bool = True,
        show_trail: bool = True,
        trail_seconds: float = 90.0,
        dpi: int = 100,
    ) -> None:
        self.recording = recording
        self.scene = load_scene(recording)
        self.show_sensors = show_sensors
        self.show_trail = show_trail
        self.trail_seconds = trail_seconds
        self._frame_origin: tuple[float, float, float] | None = None
        self._ghost_world: tuple[float, float] | None = None

        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg",):
            pass
        import matplotlib.pyplot as plt

        self.fig: Figure = plt.figure(figsize=figsize, dpi=dpi, facecolor=PALETTE["panel"])
        # Pool on top, a thin HUD strip below: the numbers support the picture,
        # they do not compete with it.
        gridspec = self.fig.add_gridspec(
            2,
            1,
            height_ratios=[7.0, 1.25],
            hspace=0.06,
            left=0.03,
            right=0.985,
            top=0.965,
            bottom=0.04,
        )
        self.ax: Axes = self.fig.add_subplot(gridspec[0])
        self.hud: Axes = self.fig.add_subplot(gridspec[1])
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection

        scene = self.scene
        rec = self.recording
        ax = self.ax
        ax.set_facecolor(PALETTE["panel"])
        minx, miny, maxx, maxy = scene.pool.bounds
        pad = 0.35
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- pool body ---------------------------------------------------
        outline = np.asarray(scene.pool.boundary.exterior.coords)
        ax.add_patch(
            mpatches.Polygon(
                outline,
                closed=True,
                facecolor="none",
                edgecolor=PALETTE["coping"],
                linewidth=7,
                joinstyle="round",
                zorder=1,
            )
        )
        # Everything raster-based is clipped to the true pool outline. Without
        # this the 10 cm cells show as a staircase fringe against the smooth
        # coping, which reads as a rendering bug rather than as resolution.
        self._water_clip = mpatches.Polygon(
            outline, closed=True, transform=ax.transData, facecolor="none", edgecolor="none"
        )
        ax.add_patch(self._water_clip)
        # Depth shading: the water gets darker where it is deeper, which is the
        # cue that tells a viewer the pool has a shape in the third dimension.
        xs, ys = scene.grid.cell_centers()
        depth = np.asarray(scene.pool.depth_at(xs, ys), dtype=float)
        inside = scene.navigable
        self._depth_image = ax.imshow(
            depth,
            extent=scene.grid.extent,
            origin="lower",
            cmap=_water_cmap(),
            vmin=float(depth[inside].min()) if inside.any() else 0.0,
            vmax=float(depth[inside].max()) if inside.any() else 1.0,
            interpolation="bilinear",
            zorder=2,
        )
        self._depth_image.set_clip_path(self._water_clip)

        # --- cleaned swath ---------------------------------------------------
        # The single most important thing to show: where the robot has actually
        # been. Rendered under the dirt so a cleaned-but-still-dirty patch (the
        # brush-off case) still reads as dirty.
        self._covered_image = ax.imshow(
            np.zeros((*scene.grid.shape, 4)),
            extent=scene.grid.extent,
            origin="lower",
            zorder=2.5,
            interpolation="bilinear",
        )
        self._covered_image.set_clip_path(self._water_clip)
        self._first_visit = self._build_first_visit()

        # --- dirt ----------------------------------------------------------
        dirt0 = rec.dirt_at(0.0)
        self._dirt_max = float(np.percentile(dirt0[dirt0 > 0], 92)) if np.any(dirt0 > 0) else 1.0
        self._dirt_image = ax.imshow(
            _dirt_alpha(dirt0, scene.navigable, self._dirt_max),
            extent=scene.grid.extent,
            origin="lower",
            zorder=3,
            interpolation="bilinear",
        )
        self._dirt_image.set_clip_path(self._water_clip)

        # --- features --------------------------------------------------------
        for feature in scene.pool.features:
            footprint = feature.footprint
            if footprint is not None and not footprint.is_empty:
                ax.add_patch(
                    mpatches.Polygon(
                        np.asarray(footprint.exterior.coords),
                        closed=True,
                        facecolor="#243447",
                        edgecolor=PALETTE["coping"],
                        linewidth=1.2,
                        zorder=4,
                    )
                )
            position = getattr(feature, "position", None)
            if position is not None:
                ax.plot(
                    *position,
                    marker="o",
                    markersize=4,
                    color=PALETTE["coping"],
                    alpha=0.4,
                    zorder=4,
                )

        # --- debris -----------------------------------------------------------
        self._debris = ax.scatter(
            [], [], s=[], c="#a05a2c", edgecolors="#5d3316", linewidths=0.5, zorder=5
        )

        # --- trail ------------------------------------------------------------
        self._trail = LineCollection([], linewidths=0, zorder=6)
        ax.add_collection(self._trail)

        # --- sonar ------------------------------------------------------------
        self._rays = LineCollection(
            [], colors=PALETTE["accent"], linewidths=1.0, alpha=0.55, linestyles=":", zorder=7
        )
        ax.add_collection(self._rays)

        # --- robot -------------------------------------------------------------
        self._body = mpatches.FancyBboxPatch(
            (-scene.robot_length / 2, -scene.robot_width / 2),
            scene.robot_length,
            scene.robot_width,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=PALETTE["hull"],
            edgecolor="#05090e",
            linewidth=1.2,
            zorder=9,
        )
        ax.add_patch(self._body)
        self._nose = ax.plot([], [], color=PALETTE["accent"], linewidth=2.4, zorder=10)[0]
        self._brush = ax.plot([], [], marker="o", markersize=6, color=PALETTE["accent"], zorder=10)[
            0
        ]
        self._contact = ax.plot(
            [], [], marker="*", markersize=16, color=PALETTE["bad"], linestyle="none", zorder=11
        )[0]

        # If the controller published a pose estimate, draw it as a ghost. Two
        # marks that drift apart on screen say more about dead reckoning than a
        # column of numbers ever does.
        self._has_estimate = {"ctl.est_x", "ctl.est_y"} <= set(rec.frames)
        self._ghost = ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=PALETTE["warn"],
            markeredgewidth=1.6,
            linestyle="none",
            zorder=8,
        )[0]
        self._ghost_heading = ax.plot(
            [], [], color=PALETTE["warn"], linewidth=1.4, alpha=0.8, zorder=8
        )[0]
        self._error_line = ax.plot(
            [], [], color=PALETTE["warn"], linewidth=1.0, alpha=0.5, linestyle="--", zorder=8
        )[0]

        self._title = ax.text(
            0.012,
            0.975,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=PALETTE["ink"],
            fontsize=11,
            family="monospace",
            zorder=12,
        )
        self._banner = ax.text(
            0.5,
            0.055,
            "",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=PALETTE["warn"],
            fontsize=13,
            family="monospace",
            weight="bold",
            zorder=12,
        )
        self._build_hud()

    # ------------------------------------------------------------------
    def _build_hud(self) -> None:
        hud = self.hud
        hud.set_facecolor(PALETTE["panel"])
        hud.set_xlim(0, 1)
        hud.set_ylim(0, 1)
        hud.set_xticks([])
        hud.set_yticks([])
        for spine in hud.spines.values():
            spine.set_visible(False)

        self._bars: dict[str, Any] = {}
        labels = [
            ("coverage", PALETTE["accent"], "where it drove"),
            ("dirt removed", PALETTE["good"], "what it cleaned"),
            ("battery", PALETTE["warn"], ""),
            ("filter", PALETTE["bad"], ""),
        ]
        import matplotlib.patches as mpatches

        width = 0.205
        for i, (label, colour, note) in enumerate(labels):
            x0 = 0.03 + i * 0.245
            hud.text(
                x0, 0.70, label, color=PALETTE["ink"], fontsize=9, family="monospace", alpha=0.75
            )
            # Value sits above its own bar and right-aligned to it. Putting it
            # after the bar left it butting against the *next* label, so every
            # number appeared to belong to the wrong meter.
            value = hud.text(
                x0 + width,
                0.70,
                "",
                color=colour,
                fontsize=10,
                family="monospace",
                ha="right",
                weight="bold",
            )
            if note:
                hud.text(
                    x0,
                    0.10,
                    note,
                    color=PALETTE["ink"],
                    fontsize=7.5,
                    family="monospace",
                    alpha=0.4,
                )
            hud.add_patch(
                mpatches.Rectangle((x0, 0.36), width, 0.24, facecolor="#16222f", edgecolor="none")
            )
            bar = mpatches.Rectangle((x0, 0.36), 0.0, 0.24, facecolor=colour, edgecolor="none")
            hud.add_patch(bar)
            self._bars[label] = (bar, value, x0, width)

        self._hud_note = hud.text(
            0.72, 0.10, "", color=PALETTE["ink"], fontsize=8, family="monospace", alpha=0.55
        )

    # ------------------------------------------------------------------
    def draw(self, index: int) -> None:
        """Render frame ``index``."""
        rec = self.recording
        scene = self.scene
        index = int(np.clip(index, 0, rec.n_frames - 1))
        f = rec.frames
        t = float(f["time"][index])
        x, y, heading = float(f["x"][index]), float(f["y"][index]), float(f["heading"][index])

        # Dirt and debris come from the nearest keyframe at or before t.
        self._dirt_image.set_data(_dirt_alpha(rec.dirt_at(t), scene.navigable, self._dirt_max))
        self._covered_image.set_data(_covered_alpha(self._first_visit <= index, scene.navigable))
        debris = rec.debris_at(t)
        if debris.size:
            active = debris[:, 4] < 0.5
            self._debris.set_offsets(debris[active, :2] if active.any() else np.zeros((0, 2)))
            self._debris.set_sizes(debris[active, 3] * 900.0 if active.any() else np.zeros(0))

        # Trail: a fading window rather than the whole path, so the recent
        # behaviour stays legible on a long run.
        if self.show_trail:
            self._update_trail(index, t)

        # Robot.
        import matplotlib.transforms as mtransforms

        transform = mtransforms.Affine2D().rotate(heading).translate(x, y) + self.ax.transData
        self._body.set_transform(transform)
        nose = scene.robot_length * 0.62
        self._nose.set_data([x, x + np.cos(heading) * nose], [y, y + np.sin(heading) * nose])
        self._brush.set_data(
            [x + np.cos(heading) * nose * 0.85], [y + np.sin(heading) * nose * 0.85]
        )

        # Sonar rays, drawn at their measured length.
        if self.show_sensors and scene.sonar_angles:
            segments = []
            for i, angle in enumerate(scene.sonar_angles):
                key = f"sonar.beam_{i}"
                if key not in f:
                    continue
                r = float(f[key][index])
                if not np.isfinite(r):
                    continue
                a = heading + angle
                segments.append([(x, y), (x + np.cos(a) * r, y + np.sin(a) * r)])
            self._rays.set_segments(segments)

        if self._has_estimate:
            self._draw_estimate(index, x, y)

        contacts = int(f["contacts"][index]) if "contacts" in f else 0
        self._contact.set_data([x] if contacts else [], [y] if contacts else [])

        self._update_text(index, t)

    # ------------------------------------------------------------------
    def _draw_estimate(self, index: int, x: float, y: float) -> None:
        """Draw where the robot *thinks* it is, and the error to where it is.

        The estimate lives in the controller's own frame, anchored at the start
        pose, so it is rotated into world coordinates for display. Without that
        the ghost would sit in the wrong place for reasons that have nothing to
        do with estimation quality.
        """
        f = self.recording.frames
        if self._frame_origin is None:
            self._frame_origin = (float(f["x"][0]), float(f["y"][0]), float(f["heading"][0]))
        ox, oy, oh = self._frame_origin
        ex, ey = float(f["ctl.est_x"][index]), float(f["ctl.est_y"][index])
        if not (np.isfinite(ex) and np.isfinite(ey)):
            return
        wx = ox + ex * np.cos(oh) - ey * np.sin(oh)
        wy = oy + ex * np.sin(oh) + ey * np.cos(oh)
        self._ghost.set_data([wx], [wy])
        self._ghost_world = (wx, wy)

        if "ctl.est_heading" in f:
            heading = float(f["ctl.est_heading"][index]) + oh
            nose = self.scene.robot_length * 0.7
            self._ghost_heading.set_data(
                [wx, wx + np.cos(heading) * nose], [wy, wy + np.sin(heading) * nose]
            )
        self._error_line.set_data([x, wx], [y, wy])

    def _update_trail(self, index: int, t: float) -> None:
        f = self.recording.frames
        times = f["time"]
        start = int(np.searchsorted(times, t - self.trail_seconds))
        start = max(0, min(start, index))
        xs = f["x"][start : index + 1]
        ys = f["y"][start : index + 1]
        if xs.size < 2:
            self._trail.set_segments([])
            return
        points = np.column_stack([xs, ys])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        # Older segments fade and thin out; the head of the trail is brightest.
        age = np.linspace(0.0, 1.0, len(segments))
        rgba = np.zeros((len(segments), 4))
        rgba[:, :3] = _hex_rgb(PALETTE["trail"])
        rgba[:, 3] = 0.06 + 0.78 * age
        # matplotlib's stubs are narrower than what LineCollection accepts;
        # arrays are the documented input for all three of these.
        self._trail.set_segments(list(segments))
        self._trail.set_linewidth((0.7 + 3.2 * age).tolist())
        self._trail.set_color(rgba)  # type: ignore[arg-type]

    def _update_text(self, index: int, t: float) -> None:
        f = self.recording.frames
        manifest = self.recording.manifest
        scenario = manifest.get("scenario", {})

        coverage = self._coverage_at(index)
        dirt_removed = self._dirt_removed_at(t)
        battery = float(f["battery"][index]) if "battery" in f else 0.0
        filter_load = float(f["filter_load"][index]) if "filter_load" in f else 0.0
        capacity = (
            manifest.get("robot_config", {})
            .get("cleaning", {})
            .get("filter", {})
            .get("capacity", 900.0)
        )

        self._title.set_text(
            f"{scenario.get('pool', '?')} · {scenario.get('robot', '?')} · "
            f"{scenario.get('dirt', '?')} · seed {manifest.get('seed')}\n"
            f"{_clock(t)}   {float(f['distance'][index]):6.1f} m travelled"
        )

        flags = []
        if int(f["stuck"][index]) if "stuck" in f else 0:
            flags.append("STUCK")
        if filter_load >= capacity:
            flags.append("FILTER FULL")
        if battery <= 0.06:
            flags.append("BATTERY EMPTY")
        self._banner.set_text("   ".join(flags))

        for label, value in (
            ("coverage", coverage),
            ("dirt removed", dirt_removed),
            ("battery", battery),
            ("filter", min(filter_load / max(capacity, 1e-9), 1.0)),
        ):
            bar, text, _x0, width = self._bars[label]
            bar.set_width(width * float(np.clip(value, 0.0, 1.0)))
            text.set_text(f"{value * 100:4.0f}%")

        note = (
            f"controller {scenario.get('controller', '?')}   "
            f"frame {index + 1}/{self.recording.n_frames}"
        )
        if self._has_estimate and self._ghost_world is not None:
            error = float(
                np.hypot(
                    self._ghost_world[0] - float(f["x"][index]),
                    self._ghost_world[1] - float(f["y"][index]),
                )
            )
            note = f"estimate off by {error:4.2f} m   " + note
        self._hud_note.set_text(note)

    def _build_first_visit(self) -> np.ndarray:
        """Frame index at which each cell was first covered, or a large sentinel.

        One pass over the run. From this both the coverage overlay and the
        coverage percentage are exact at any frame, including after scrubbing
        backwards -- accumulating forward only would make a rewind show
        coverage the robot has not achieved yet.
        """
        scene = self.scene
        f = self.recording.frames
        n = self.recording.n_frames
        first = np.full(scene.grid.shape, np.iinfo(np.int32).max, dtype=np.int32)
        radius = 0.5 * scene.swath
        for i in range(n):
            x, y = float(f["x"][i]), float(f["y"][i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            window = scene.grid.window(x, y, radius)
            if window is None:
                continue
            patch = window.view(first)
            fresh = window.mask & (patch == np.iinfo(np.int32).max)
            patch[fresh] = i
        first[~scene.navigable] = np.iinfo(np.int32).max
        return first

    def _coverage_at(self, index: int) -> float:
        total = max(int(self.scene.navigable.sum()), 1)
        return float((self._first_visit <= index).sum()) / total

    def _dirt_removed_at(self, t: float) -> float:
        rec = self.recording
        if rec.dirt_keyframes.size == 0:
            return 0.0
        initial = float(rec.dirt_keyframes[0].sum())
        if initial <= 0:
            return 0.0
        now = float(rec.dirt_at(t).sum())
        return float(np.clip((initial - now) / initial, 0.0, 1.0))


# ----------------------------------------------------------------------
def _water_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "zimablue_water", [PALETTE["shallow"], PALETTE["mid"], PALETTE["deep"]]
    )


def _dirt_cmap():
    """A flat water base for the before/after panels of the summary."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("zimablue_base", [PALETTE["mid"], PALETTE["mid"]])


def _hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _dirt_alpha(
    dirt: np.ndarray, navigable: np.ndarray, vmax: float, *, strength: float = 0.85
) -> np.ndarray:
    """Dirt as a semi-transparent brown overlay on the water.

    An alpha layer rather than a colormap so the depth shading stays visible
    underneath: the viewer should be able to see both how deep and how dirty a
    patch is at once.
    """
    if dirt.size == 0 or dirt.shape != navigable.shape:
        return np.zeros((*navigable.shape, 4), dtype=float)
    intensity = np.clip(dirt / max(vmax, 1e-9), 0.0, 1.0)
    rgba = np.zeros((*dirt.shape, 4), dtype=float)
    rgba[..., :3] = _hex_rgb(PALETTE["dirt"])
    rgba[..., 3] = _smooth(np.where(navigable, intensity * strength, 0.0))
    return rgba


def _covered_alpha(covered: np.ndarray, navigable: np.ndarray) -> np.ndarray:
    """The cleaned swath as a pale wash over the water.

    The alpha is blurred before display: a binary mask stretched over 10 cm
    cells reads as a staircase, and the swath is a soft thing in reality.
    """
    alpha = np.where(covered & navigable, 0.34, 0.0)
    alpha = _smooth(alpha)
    rgba = np.zeros((*covered.shape, 4), dtype=float)
    rgba[..., :3] = _hex_rgb(PALETTE["foam"])
    rgba[..., 3] = alpha
    return rgba


def _smooth(a: np.ndarray) -> np.ndarray:
    """Cheap separable 3x3 blur, used to take the edge off raster overlays."""
    padded = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out += padded[dr : dr + a.shape[0], dc : dc + a.shape[1]]
    return out / 9.0


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"
