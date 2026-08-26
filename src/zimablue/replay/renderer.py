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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from zimablue.geometry import Grid
from zimablue.pool import Pool
from zimablue.recording import Recording
from zimablue.robot.design import CleanerDesign, make_design

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["PALETTE", "ReplayRenderer", "load_scene", "pile_range"]

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
    "silt": "#33270f",
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
    design: CleanerDesign = field(default_factory=lambda: make_design(None))
    """How the cleaner is drawn. Recordings written before designs existed do
    not carry one and get the default."""

    robot_height: float = 0.26
    fleet_geometry: tuple[tuple[float, float, CleanerDesign], ...] = ()
    """Length, width and design for every recorded fleet member."""


def load_scene(recording: Recording) -> Scene:
    """Rebuild the pool and robot geometry embedded in a recording.

    Reads the manifest, not the live presets -- a recording must stay renderable
    after the preset it came from has changed.
    """
    manifest = recording.manifest
    if not manifest.get("pool_config"):
        # A recording written on a robot carries no pool unless the runtime was
        # told one -- a cleaner does not know the shape of what it is in. Say
        # so, rather than failing inside Pool.from_dict on a None.
        raise ValueError(
            "this recording has no pool geometry, so there is nothing to draw it "
            "against. Recordings written by zimablue.hardware only carry a pool "
            "if the runtime was constructed with pool=..., because a robot does "
            "not know the shape of the pool it is in. Pass the pool you believe "
            "you are in, or attach one to the manifest before replaying."
        )
    pool = Pool.from_dict(manifest["pool_config"])
    cell = float(manifest.get("cell", 0.10))
    robot_cfg = manifest.get("robot_config", {})
    robot_configs = manifest.get("robot_configs") or [robot_cfg]
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

    design_cfg = robot_cfg.get("design")
    fleet_geometry = []
    for config in robot_configs:
        member_chassis = config.get("chassis", {})
        member_design = config.get("design")
        fleet_geometry.append(
            (
                float(member_chassis.get("length", 0.42)),
                float(member_chassis.get("width", 0.38)),
                CleanerDesign.from_dict(member_design) if member_design else make_design(None),
            )
        )
    return Scene(
        pool=pool,
        grid=pool.grid(cell),
        navigable=pool.navigable_mask(cell),
        robot_length=float(chassis.get("length", 0.42)),
        robot_width=float(chassis.get("width", 0.38)),
        robot_height=float(chassis.get("height", 0.26)),
        swath=float(swath),
        sonar_angles=angles,
        sonar_max_range=max_range,
        design=CleanerDesign.from_dict(design_cfg) if design_cfg else make_design(None),
        fleet_geometry=tuple(fleet_geometry),
    )


FLEET_COLOURS = ("#3ddcff", "#ffd166", "#ff7ab6", "#8affc1", "#c9a7ff", "#ff9f6e")
"""Ring and trail colours, one per robot. Six is more than any pool needs."""


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
        ax: Axes | None = None,
    ) -> None:
        self.recording = recording
        self.scene = load_scene(recording)
        fleet = recording.manifest.get("fleet") or {}
        self.fleet_size = int(fleet.get("count", 1))
        # A fleet recording carries r0.x alongside a flat copy of the same
        # channel; single-robot recordings carry only the flat one.
        self.prefixes = [f"r{i}." for i in range(self.fleet_size)] if self.fleet_size > 1 else [""]
        self.show_sensors = show_sensors
        self.show_trail = show_trail
        self.trail_seconds = trail_seconds
        self._frame_origin: tuple[float, float, float] | None = None
        self._ghost_world: tuple[float, float] | None = None

        import matplotlib.pyplot as plt

        self.fig: Figure
        self.ax: Axes
        self.hud: Axes | None

        if ax is not None:
            # Drawing into someone else's axes: a dirt-cam side panel, a
            # notebook grid, a figure of several pools. No HUD strip -- the
            # host owns the layout and a stolen row of meters would fight it.
            figure = ax.get_figure()
            if figure is None:  # pragma: no cover - detached axes
                raise ValueError("the axes passed as ax= is not attached to a figure")
            self.ax = ax
            self.fig = figure  # type: ignore[assignment]
            self.hud = None
            self._build()
            return

        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=PALETTE["panel"])
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
        self.ax = self.fig.add_subplot(gridspec[0])
        self.hud = self.fig.add_subplot(gridspec[1])
        self._build()

    # ------------------------------------------------------------------
    def _build_debris(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Outlines and colours for every debris item, computed once.

        The outline is built about the item's own centre, not at a position.
        Shape, rotation, scale and colour are fixed per item; where it *is* is
        not -- the robot shoves anything too big for the intake out of the way,
        and over a run the typical survivor travels further than its own
        length. Baking the first frame's centre in here drew every leaf where
        it started for the whole run.
        """
        import matplotlib.colors as mcolors

        from zimablue.replay.debris_shapes import debris_colour, debris_offsets

        first = self.recording.debris_at(0.0)
        if not first.size:
            return [], np.zeros((0, 4))

        names = self.recording.debris_type_names()
        types = np.clip(first[:, 5].astype(int), 0, max(len(names) - 1, 0))
        kinds = [names[k] for k in types]
        offsets = debris_offsets(first[:, 3], kinds, np.arange(len(first)))
        colours = mcolors.to_rgba_array([debris_colour(k, i) for i, k in enumerate(kinds)])
        return offsets, colours

    # ------------------------------------------------------------------
    def _build(self) -> None:
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection, PolyCollection

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
            # Same reason as the dirt layer: no interpolation, because the
            # cleaned swath is exactly as wide as the robot, not wider.
            interpolation="nearest",
        )
        self._covered_image.set_clip_path(self._water_clip)
        self._first_visit = self._build_first_visit()

        # --- dirt ----------------------------------------------------------
        dirt0 = rec.dirt_at(0.0)
        self._dirt_max = float(np.percentile(dirt0[dirt0 > 0], 92)) if np.any(dirt0 > 0) else 1.0
        self._dirt_pile = pile_range(rec, self._dirt_max)
        self._dirt_image = ax.imshow(
            _dirt_alpha(dirt0, scene.navigable, self._dirt_max, pile=self._dirt_pile),
            extent=scene.grid.extent,
            origin="lower",
            zorder=3,
            # Nearest, not bilinear: interpolating between cells smears the
            # cleaned swath outward again at display resolution.
            interpolation="nearest",
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
        # Real silhouettes at true scale rather than markers. A marker is sized
        # in points, so it stays the same size on screen while the pool is
        # zoomed -- and a 9 cm leaf drawn the same size as a 25 cm frond tells
        # you nothing about why one jams the intake and the other does not.
        self._debris_offsets, self._debris_colours = self._build_debris()
        self._debris = PolyCollection(
            [], facecolors=[], edgecolors="#3d2412", linewidths=0.35, zorder=5
        )
        ax.add_collection(self._debris)

        # --- trail ------------------------------------------------------------
        self._trail = LineCollection([], linewidths=0, zorder=6)
        ax.add_collection(self._trail)

        # --- sonar ------------------------------------------------------------
        self._rays = LineCollection(
            [], colors=PALETTE["accent"], linewidths=1.0, alpha=0.55, linestyles=":", zorder=7
        )
        ax.add_collection(self._rays)

        # --- robot -------------------------------------------------------------
        # Drawn from the cleaner's design rather than as a generic box, so a
        # domed suction unit and a quad-brush commercial machine are telling
        # apart at a glance. Every piece is one patch in the robot's own frame,
        # moved by a single affine transform per frame.
        design = scene.design
        scale = np.array([scene.robot_length, scene.robot_width])
        self._parts = []
        hull_patch = None
        for part in design.drawable():
            patch = mpatches.Polygon(
                np.asarray(part.outline, dtype=float) * scale,
                closed=True,
                facecolor=part.colour,
                edgecolor="#05090e" if part.name == "hull" else "none",
                linewidth=1.2 if part.name == "hull" else 0.0,
                alpha=part.alpha,
                zorder=9 + 0.01 * (part.z + 100),
            )
            ax.add_patch(patch)
            # Parts are clipped to the hull, so a brush bar drawn a little wide
            # reads as reaching the edge of the machine rather than floating
            # off it. It also means a design can be written with round numbers
            # instead of solving for where the hull curve is at each station.
            if hull_patch is None:
                hull_patch = patch
            else:
                patch.set_clip_path(hull_patch)
            self._parts.append(patch)

        # The rest of the fleet. Each gets the same design and a coloured ring
        # rather than a recoloured hull: the designs exist to be told apart
        # from each other, and repainting them to tell robots apart would
        # throw that away. The ring and the trail carry the identity.
        self._crew: list[dict[str, Any]] = []
        for index in range(1, self.fleet_size):
            colour = FLEET_COLOURS[index % len(FLEET_COLOURS)]
            parts = []
            hull = None
            if index < len(scene.fleet_geometry):
                member_length, member_width, member_design = scene.fleet_geometry[index]
            else:
                member_length, member_width, member_design = (
                    scene.robot_length,
                    scene.robot_width,
                    design,
                )
            member_scale = np.array([member_length, member_width])
            for part in member_design.drawable():
                patch = mpatches.Polygon(
                    np.asarray(part.outline, dtype=float) * member_scale,
                    closed=True,
                    facecolor=part.colour,
                    edgecolor="#05090e" if part.name == "hull" else "none",
                    linewidth=1.2 if part.name == "hull" else 0.0,
                    alpha=part.alpha,
                    zorder=9 + 0.01 * (part.z + 100),
                )
                ax.add_patch(patch)
                if hull is None:
                    hull = patch
                else:
                    patch.set_clip_path(hull)
                parts.append(patch)
            ring = mpatches.Circle(
                (0.0, 0.0),
                0.5 * max(member_length, member_width) * 1.15,
                fill=False,
                edgecolor=colour,
                linewidth=1.6,
                alpha=0.9,
                zorder=11,
            )
            ax.add_patch(ring)
            trail = LineCollection([], colors=colour, linewidths=1.4, alpha=0.5, zorder=6)
            ax.add_collection(trail)
            label = ax.text(
                0.0,
                0.0,
                str(index),
                color=colour,
                fontsize=7,
                family="monospace",
                ha="center",
                va="center",
                zorder=12,
            )
            self._crew.append({"parts": parts, "ring": ring, "trail": trail, "label": label})

        self._lead_ring: Any = None
        self._lead_label: Any = None
        if self.fleet_size > 1:
            self._lead_ring = mpatches.Circle(
                (0.0, 0.0),
                0.5 * max(scene.robot_length, scene.robot_width) * 1.15,
                fill=False,
                edgecolor=FLEET_COLOURS[0],
                linewidth=1.6,
                alpha=0.9,
                zorder=11,
            )
            ax.add_patch(self._lead_ring)
            self._lead_label = ax.text(
                0.0,
                0.0,
                "0",
                color=FLEET_COLOURS[0],
                fontsize=7,
                family="monospace",
                ha="center",
                va="center",
                zorder=12,
            )

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
        if self.hud is not None:
            self._build_hud()

    # ------------------------------------------------------------------
    def _update_title_only(self, index: int, t: float) -> None:
        """A bare label, for when this renderer is a panel in someone else's
        figure: the host owns the clock and the meters, and repeating them here
        just crowds the pool."""
        self._title.set_text("top down")
        self._title.set_fontsize(9)
        self._title.set_alpha(0.55)
        self._banner.set_text("")

    def _build_hud(self) -> None:
        hud = self.hud
        if hud is None:  # pragma: no cover - guarded by the caller
            return
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
            if label == "dirt removed":
                # A tick where the bar physically cannot go past, because the
                # pool holds debris this intake cannot swallow. Without it a
                # run that removed everything it could looks like a run that
                # gave up at 92%.
                self._ceiling = hud.plot([], [], color=PALETTE["warn"], linewidth=1.4, zorder=6)[0]

        # Right-aligned to the last bar. Left-aligned it grew off the edge of
        # the figure as soon as the frame counter reached five digits.
        self._hud_note = hud.text(
            0.97,
            0.10,
            "",
            color=PALETTE["ink"],
            fontsize=8,
            family="monospace",
            alpha=0.55,
            ha="right",
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

        # Dirt and debris are keyframed every ten simulated seconds and blended
        # between, so the field creeps rather than stepping and a shoved leaf
        # slides rather than teleporting.
        self._dirt_image.set_data(
            _dirt_alpha(
                rec.dirt_at(t, interpolate=True),
                scene.navigable,
                self._dirt_max,
                pile=self._dirt_pile,
            )
        )
        self._covered_image.set_data(_covered_alpha(self._first_visit <= index, scene.navigable))
        debris = rec.debris_at(t, interpolate=True)
        if debris.size and self._debris_offsets:
            # `collected` is fractional here: it is how far through the
            # interval in which the item was picked up, which fades it out
            # over that interval rather than winking it away between frames.
            active = np.nonzero(debris[:, 4] < 1.0)[0]
            centres = debris[active, 0:2]
            placed = zip(active, centres, strict=True)
            self._debris.set_verts([self._debris_offsets[i] + centre for i, centre in placed])
            # Alpha goes into the colours rather than through set_alpha, which
            # keeps the previous frame's array and re-applies it to a different
            # number of items on the next one.
            faces = self._debris_colours[active].copy()
            faces[:, 3] = np.clip(1.0 - debris[active, 4], 0.0, 1.0)
            self._debris.set_facecolor([tuple(face) for face in faces])

        # Trail: a fading window rather than the whole path, so the recent
        # behaviour stays legible on a long run.
        if self.show_trail:
            self._update_trail(index, t)

        # Robot.
        import matplotlib.transforms as mtransforms

        transform = mtransforms.Affine2D().rotate(heading).translate(x, y) + self.ax.transData
        for patch in self._parts:
            patch.set_transform(transform)
        nose = scene.robot_length * 0.62
        self._nose.set_data([x, x + np.cos(heading) * nose], [y, y + np.sin(heading) * nose])
        self._brush.set_data(
            [x + np.cos(heading) * nose * 0.85], [y + np.sin(heading) * nose * 0.85]
        )

        if self._lead_ring is not None:
            self._lead_ring.center = (x, y)
            self._lead_label.set_position((x, y))
        for member, mate in enumerate(self._crew, start=1):
            prefix = self.prefixes[member]
            mx, my = float(f[f"{prefix}x"][index]), float(f[f"{prefix}y"][index])
            mh = float(f[f"{prefix}heading"][index])
            mate_transform = mtransforms.Affine2D().rotate(mh).translate(mx, my) + self.ax.transData
            for patch in mate["parts"]:
                patch.set_transform(mate_transform)
            mate["ring"].center = (mx, my)
            mate["label"].set_position((mx, my))
            if self.show_trail:
                self._set_trail(mate["trail"], prefix, index, t, fade=False)

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
        self._set_trail(self._trail, self.prefixes[0], index, t, fade=True)

    def _set_trail(self, artist: Any, prefix: str, index: int, t: float, *, fade: bool) -> None:
        """A fading window of one robot's recent path.

        ``fade`` is off for the rest of the fleet: their trails are already
        distinguished by colour, and giving every one of them a bright head
        makes a three-robot pool look like a firework.
        """
        f = self.recording.frames
        times = f["time"]
        start = int(np.searchsorted(times, t - self.trail_seconds))
        start = max(0, min(start, index))
        xs = f[f"{prefix}x"][start : index + 1]
        ys = f[f"{prefix}y"][start : index + 1]
        if xs.size < 2:
            artist.set_segments([])
            return
        points = np.column_stack([xs, ys])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        artist.set_segments(list(segments))
        if not fade:
            return
        # Older segments fade and thin out; the head of the trail is brightest.
        age = np.linspace(0.0, 1.0, len(segments))
        rgba = np.zeros((len(segments), 4))
        rgba[:, :3] = _hex_rgb(PALETTE["trail"])
        rgba[:, 3] = 0.06 + 0.78 * age
        # matplotlib's stubs are narrower than what LineCollection accepts;
        # arrays are the documented input for all three of these.
        artist.set_linewidth((0.7 + 3.2 * age).tolist())
        artist.set_color(rgba)

    def _update_text(self, index: int, t: float) -> None:
        f = self.recording.frames
        if self.hud is None:
            self._update_title_only(index, t)
            return
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

        debris = self._debris_tally(t)
        if debris is not None:
            collected, total, oversize, ceiling = debris
            bar, _text, x0, width = self._bars["dirt removed"]
            if ceiling < 0.999:
                self._ceiling.set_data([x0 + width * ceiling] * 2, [0.32, 0.64])

        note = (
            f"controller {scenario.get('controller', '?')}   "
            f"frame {index + 1}/{self.recording.n_frames}"
        )
        if debris is not None and total:
            collected, total, oversize, ceiling = debris
            note = (
                f"debris {collected}/{total}"
                + (f", {oversize} too big" if oversize else "")
                + "   "
                + note
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

    def _debris_tally(self, t: float) -> tuple[int, int, int, float] | None:
        """Collected, total, oversize, and the ceiling they impose.

        Read from the debris snapshot rather than the run's final metrics, so
        it is right at every frame while scrubbing. The intake size comes from
        the robot embedded in the recording -- the same pool with a wider
        intake has a different ceiling, which is the point.
        """
        items = self.recording.debris_at(t)
        if not items.size:
            return None
        limit = float(
            self.recording.manifest.get("robot_config", {})
            .get("cleaning", {})
            .get("pump", {})
            .get("max_debris_size", 0.09)
        )
        collected = int((items[:, 4] > 0.5).sum())
        # Oversize *and still in the water*: an oversize floater the skimmer
        # took stops being anyone's ceiling, matching the metric.
        oversize_mask = (items[:, 3] > limit) & (items[:, 4] < 0.5)
        stuck_mass = float(items[oversize_mask, 2].sum())
        total_dirt = (
            float(self.recording.dirt_keyframes[0].sum())
            if len(self.recording.dirt_keyframes)
            else 0.0
        )
        # The dirt rasters do not include debris mass, so the denominator is
        # the pool's whole initial load: field plus items.
        initial = total_dirt + float(items[:, 2].sum())
        ceiling = 1.0 - stuck_mass / initial if initial > 0 else 1.0
        return collected, len(items), int(oversize_mask.sum()), ceiling

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
        for prefix in self.prefixes:
            for i in range(n):
                x, y = float(f[f"{prefix}x"][i]), float(f[f"{prefix}y"][i])
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
        """Fraction of the pool's dirt taken out, debris included.

        The raster keyframes hold the *field* only, so a bar built from them
        alone ignores a quarter of what is in an autumn pool and reads several
        points below the run's own metric. Leaves are dirt.
        """
        rec = self.recording
        if rec.dirt_keyframes.size == 0:
            return 0.0
        items = rec.debris_at(t)
        loose = float(items[items[:, 4] < 0.5, 2].sum()) if items.size else 0.0
        every = float(items[:, 2].sum()) if items.size else 0.0
        initial = float(rec.dirt_keyframes[0].sum()) + every
        if initial <= 0:
            return 0.0
        now = float(rec.dirt_at(t).sum()) + loose
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


def pile_range(recording: Recording, vmax: float) -> float:
    """How many times ``vmax`` the heaviest cell of a run reaches.

    Calibrating the overlay on the first frame alone was fine when dirt only
    ever went away. It accumulates now -- the returns sweep it into a heap at
    the drain -- so the scale has to be told how far the run actually goes, or
    the heap is one flat saturated blob whose edge is the only part that moves.
    """
    if recording.dirt_keyframes.size == 0:
        return 8.0
    heaviest = float(recording.dirt_keyframes.sum(axis=1, dtype=np.float32).max())
    return float(np.clip(heaviest / max(vmax, 1e-9), 2.0, 64.0))


def _dirt_alpha(
    dirt: np.ndarray,
    navigable: np.ndarray,
    vmax: float,
    *,
    pile: float = 8.0,
    strength: float = 0.85,
) -> np.ndarray:
    """Dirt as a semi-transparent brown overlay on the water.

    An alpha layer rather than a colormap so the depth shading stays visible
    underneath: the viewer should be able to see both how deep and how dirty a
    patch is at once.

    Alpha alone runs out. ``vmax`` is a percentile of the dirt a run *starts*
    with, and the returns sweep the floor into heaps an order of magnitude
    above that -- more than twenty times, in a kidney -- so a pile pins the
    alpha at maximum across its whole width and only its edge appears to move.
    Past ``vmax`` the colour darkens towards silt instead, which keeps the
    heap's growth visible without touching how anything below ``vmax`` looks.
    ``pile`` is how many times ``vmax`` reaches the darkest silt, and callers
    take it from the heaviest cell the run ever holds so the whole range is
    used rather than guessed at.
    """
    if dirt.size == 0 or dirt.shape != navigable.shape:
        return np.zeros((*navigable.shape, 4), dtype=float)
    ratio = dirt / max(vmax, 1e-9)
    intensity = np.clip(ratio, 0.0, 1.0)
    heaped = np.clip((ratio - 1.0) / max(pile - 1.0, 1e-9), 0.0, 1.0)[..., None]
    rgba = np.zeros((*dirt.shape, 4), dtype=float)
    light = np.array(_hex_rgb(PALETTE["dirt"]))
    dark = np.array(_hex_rgb(PALETTE["silt"]))
    rgba[..., :3] = light * (1.0 - heaped) + dark * heaped
    rgba[..., 3] = np.where(navigable, intensity * strength, 0.0)
    return rgba


def _covered_alpha(covered: np.ndarray, navigable: np.ndarray) -> np.ndarray:
    """The cleaned swath as a pale wash over the water.

    Deliberately *not* blurred. Softening the edge looked better and lied: a
    3x3 blur over 10 cm cells spreads the wash 10 cm past the swath on every
    side, which rendered 28% more area as covered than the robot had actually
    driven over -- 9.5 m2 of phantom coverage in a 59 m2 pool, and a cleaned
    lane drawn at 1.4x its true width.

    That is the one error this renderer must not make. The project exists to
    separate where the robot drove from what it removed, and an overlay that
    quietly widens the swath argues the opposite case. The cells are 10 cm and
    the staircase is the truth.
    """
    alpha = np.where(covered & navigable, 0.34, 0.0)
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
