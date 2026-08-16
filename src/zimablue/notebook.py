"""Interactive pool preview for Jupyter.

    import zimablue as zb
    zb.preview("kidney")          # drag to rotate, scroll to zoom

The 3D replay in :mod:`zimablue.replay.scene3d` renders through matplotlib,
which means every rotation is a re-render and a notebook cell shows a still.
This module takes the other route: build the pool's geometry in Python, hand it
to the browser as JSON, and let a small canvas renderer project it. Dragging
then costs a matrix multiply over a few thousand vertices instead of a round
trip to the kernel, so the pool turns at frame rate.

The payoff beyond smoothness is reach. There is no ``ipywidgets``, no
``ipympl``, no widget state and no kernel involved, so the same output works in
JupyterLab, classic Notebook, VS Code and Colab, and it keeps working in an
exported HTML file after the kernel is gone. The cost is that the renderer is
mine: painter's algorithm, flat shading, no z-buffer. For a pool -- a basin
with no self-intersecting geometry -- that is enough, and interpenetrating
faces would be a bug in the mesh rather than a limit of the sort.

``preview`` accepts a pool, a preset name, a finished run or a recording. Given
a run it draws where the dirt ended up and the path that was driven, which
makes it the fastest way to see why a corner scored badly.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zimablue.pool import Drain, Obstacle, Pool, Return, Skimmer, Stairs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["PoolPreview", "preview"]

FloatArray = NDArray[np.float64]

COLOURS = {
    "shallow": (0x54, 0xC8, 0xEA),
    "deep": (0x0A, 0x4C, 0x86),
    "dirt": (0x6B, 0x57, 0x35),
    "wall": (0x9F, 0xC4, 0xDA),
    "coping": (0xE9, 0xEF, 0xF4),
    "obstacle": (0x35, 0x4C, 0x60),
    "stairs": (0xB9, 0xD3, 0xE2),
    "trail": (0x7F, 0xE9, 0xFF),
    "hull": (0x16, 0x21, 0x2E),
    "drain": (0x1B, 0x2C, 0x3C),
    "jet": (0x3D, 0xDC, 0xFF),
    "panel": (0x08, 0x11, 0x1B),
}

DEPTH_EXAGGERATION = 2.6
"""How much the vertical axis is stretched.

A 12 m pool 1.8 m deep is a pancake at true scale, and depth is most of what
this view exists to show. Stated here, and on the page itself, so nobody reads
a gradient off the picture."""


Colour = tuple[float, float, float]


def _hex(rgb: Colour) -> str:
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix(a: Colour, b: Colour, t: float) -> Colour:
    t = float(np.clip(t, 0.0, 1.0))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _polygons(geometry: Any) -> list[Any]:
    """Every Polygon in a Shapely result, whatever shape the result took.

    Clipping a square against a pool outline can return a Polygon, an empty
    geometry, or -- where the outline pinches through the cell twice -- a
    GeometryCollection with lines in it.
    """
    from shapely.geometry import Polygon

    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    return [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]


def _triangles(polygon: Any) -> list[NDArray[np.float64]]:
    """Triangulate a polygon, keeping only the parts actually inside it.

    Fanning from a vertex is fine for a rectangle and wrong for a kidney: the
    concave notch throws slivers across open water. Shapely's Delaunay covers
    the convex hull, so the triangles that bridge the notch are discarded by
    testing each centroid against the polygon.
    """
    from shapely.geometry import Point
    from shapely.ops import triangulate
    from shapely.prepared import prep

    inside = prep(polygon)
    kept = []
    for tri in triangulate(polygon):
        c = tri.centroid
        if inside.contains(Point(c.x, c.y)):
            kept.append(np.asarray(tri.exterior.coords, dtype=float)[:3])
    return kept


def _normal(corners: list[tuple[float, float, float]]) -> list[float]:
    """Unit face normal, computed here rather than in the browser.

    Shading from a *world* normal is what separates a lit surface from a
    speckled one. Deriving it in the renderer from projected screen coordinates
    only recovers the face's apparent area, so identical floor tiles come out
    at different brightnesses purely because of where they sit in frame.
    """
    a, b, c = (np.asarray(corners[i], dtype=float) for i in (0, 1, 2))
    n = np.cross(b - a, c - a)
    length = float(np.linalg.norm(n))
    if length < 1e-12:  # degenerate quad, from a triangle fan cap
        return [0.0, 0.0, 1.0]
    # Vertical exaggeration tilts every sloped face; matching it here keeps the
    # shading consistent with the shape actually on screen.
    n = n / length
    n[2] /= DEPTH_EXAGGERATION
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return [round(float(v), 4) for v in n]


@dataclass
class _Mesh:
    """Vertices and quads, in the form the browser wants them."""

    vertices: list[list[float]] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    lit: list[int] = field(default_factory=list)
    normals: list[list[float]] = field(default_factory=list)

    def add_quad(
        self,
        corners: list[tuple[float, float, float]],
        colour: str,
        *,
        alpha: float = 1.0,
        lit: bool = True,
    ) -> None:
        base = len(self.vertices)
        for point in corners:
            self.vertices.append([round(float(c), 3) for c in point])
        self.faces.append([base, base + 1, base + 2, base + 3])
        self.colours.append(colour)
        self.alphas.append(round(alpha, 3))
        self.lit.append(1 if lit else 0)
        self.normals.append(_normal(corners))

    def add_polygon(
        self,
        points: list[tuple[float, float, float]],
        colour: str,
        *,
        alpha: float = 1.0,
        lit: bool = True,
    ) -> None:
        """Any convex-enough ring, as a fan of quads from its first vertex.

        Boundary cells clipped against the pool outline come back as polygons
        of three to seven vertices. Fanning them keeps a single face type in
        the payload, so the renderer stays one loop over quads; a fan of a
        mildly concave ring can stray outside it, which at a tenth of a cell is
        below the resolution anyone reads off this view.
        """
        if len(points) < 3:
            return
        for i in range(1, len(points) - 1):
            self.add_quad(
                [points[0], points[i], points[i + 1], points[i + 1]],
                colour,
                alpha=alpha,
                lit=lit,
            )


class PoolPreview:
    """A pool you can turn around, rendered in the browser.

    Displaying the object in a notebook cell is all it takes -- Jupyter calls
    ``_repr_html_``. :meth:`save` writes the same thing as a standalone file,
    which is useful for sharing a pool with someone who does not have the
    package installed.
    """

    def __init__(
        self,
        pool: Pool,
        *,
        cell: float = 0.3,
        dirt: FloatArray | None = None,
        path: FloatArray | None = None,
        title: str | None = None,
        subtitle: str = "",
        size: tuple[int, int] = (760, 460),
    ) -> None:
        self.pool = pool
        self.cell = cell
        self.title = title or pool.name
        self.subtitle = subtitle
        self.size = size
        self._mesh = _Mesh()
        self._build_floor(dirt)
        self._build_walls()
        self._build_features()
        self._build_surface()
        self.path = self._thin_path(path)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _build_floor(self, dirt: FloatArray | None) -> None:
        """A tile per navigable cell, coloured by depth and by dirt.

        Cells that straddle the pool outline are clipped against it rather than
        being kept or dropped whole. A raster floor inside a smooth wall leaves
        a staircase gap you can see straight through, and at the resolution
        that keeps the payload small the gap is a third of a metre wide.

        Corner depths are sampled individually rather than taking the cell
        centre four times, so a sloped floor comes out as a slope instead of a
        flight of steps.
        """
        from shapely.geometry import box as shapely_box
        from shapely.prepared import prep

        pool = self.pool
        grid = pool.grid(self.cell)
        navigable = pool.navigable
        inside = prep(navigable)

        # Corner lattice: one more row and column than there are cells.
        cx = grid.minx + np.arange(grid.ncols + 1) * grid.cell
        cy = grid.miny + np.arange(grid.nrows + 1) * grid.cell
        gx, gy = np.meshgrid(cx, cy)
        gz = -np.asarray(pool.depth_at(gx, gy), dtype=float)

        dirt_weight = self._dirt_weight(dirt, grid)
        max_depth = max(float(pool.max_depth), 1e-6)

        for row in range(grid.nrows):
            for col in range(grid.ncols):
                tile = shapely_box(cx[col], cy[row], cx[col + 1], cy[row + 1])
                if not inside.intersects(tile):
                    continue

                filth = float(dirt_weight[row, col]) if dirt_weight is not None else 0.0
                depth = -float(
                    np.mean(
                        [gz[row, col], gz[row, col + 1], gz[row + 1, col + 1], gz[row + 1, col]]
                    )
                )
                base = _mix(COLOURS["shallow"], COLOURS["deep"], depth / max_depth)
                colour = _hex(_mix(base, COLOURS["dirt"], filth))

                if inside.contains(tile):
                    self._mesh.add_quad(
                        [
                            (gx[row, col], gy[row, col], gz[row, col]),
                            (gx[row, col + 1], gy[row, col + 1], gz[row, col + 1]),
                            (gx[row + 1, col + 1], gy[row + 1, col + 1], gz[row + 1, col + 1]),
                            (gx[row + 1, col], gy[row + 1, col], gz[row + 1, col]),
                        ],
                        colour,
                    )
                    continue

                for piece in _polygons(navigable.intersection(tile)):
                    if piece.area < 1e-4:
                        continue
                    ring = np.asarray(piece.exterior.coords, dtype=float)[:-1]
                    z = -np.asarray(pool.depth_at(ring[:, 0], ring[:, 1]), dtype=float)
                    self._mesh.add_polygon(
                        [
                            (float(x), float(y), float(zz))
                            for (x, y), zz in zip(ring, z, strict=True)
                        ],
                        colour,
                    )

    def _dirt_weight(self, dirt: FloatArray | None, grid: Any) -> FloatArray | None:
        """Resample a dirt raster onto the preview grid, normalised to [0, 1]."""
        if dirt is None or np.asarray(dirt).size == 0:
            return None
        dirt = np.asarray(dirt, dtype=float)
        fine = self.pool.grid()
        centres_x, centres_y = grid.cell_centers()
        rows = np.clip(((centres_y - fine.miny) / fine.cell).astype(int), 0, fine.nrows - 1)
        cols = np.clip(((centres_x - fine.minx) / fine.cell).astype(int), 0, fine.ncols - 1)
        sampled = dirt[rows, cols]
        positive = sampled[sampled > 0]
        if not positive.size:
            return np.zeros_like(sampled)
        # A high percentile rather than the maximum: a single hot cell should
        # not wash the whole floor out to clean blue.
        peak = float(np.percentile(positive, 96))
        return np.clip(sampled / max(peak, 1e-9), 0.0, 1.0) ** 0.6

    def _build_walls(self) -> None:
        """Extrude the boundary from the floor up to the waterline."""
        ring = np.asarray(self.pool.boundary.exterior.coords, dtype=float)
        step = max(1, (len(ring) - 1) // 96)
        for i in range(0, len(ring) - 1, step):
            j = min(i + step, len(ring) - 1)
            (x0, y0), (x1, y1) = ring[i], ring[j]
            z0 = -float(self.pool.depth_at(x0, y0))
            z1 = -float(self.pool.depth_at(x1, y1))
            self._mesh.add_quad(
                [(x0, y0, z0), (x1, y1, z1), (x1, y1, 0.0), (x0, y0, 0.0)],
                _hex(COLOURS["wall"]),
                alpha=0.34,
            )
            # A coping band above the waterline: the pool needs a rim or it
            # looks like a hole rather than a container.
            self._mesh.add_quad(
                [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, 0.09), (x0, y0, 0.09)],
                _hex(COLOURS["coping"]),
                alpha=0.85,
                lit=False,
            )

    def _build_features(self) -> None:
        """Obstacles and steps as solids; hydraulics as floor markers."""
        for feature in self.pool.features:
            if isinstance(feature, Obstacle):
                self._extrude(feature.polygon, COLOURS["obstacle"], height=feature.height)
            elif isinstance(feature, Stairs):
                self._extrude(feature.polygon, COLOURS["stairs"], height=0.0, alpha=0.75)
            elif isinstance(feature, Drain | Return | Skimmer):
                colour = COLOURS["drain"] if isinstance(feature, Drain) else COLOURS["jet"]
                self._marker(feature.position, colour)

    def _extrude(self, polygon: Any, colour: Colour, *, height: float, alpha: float = 1.0) -> None:
        if polygon is None or polygon.is_empty:
            return
        ring = np.asarray(polygon.exterior.coords, dtype=float)
        floor = [(float(x), float(y), -float(self.pool.depth_at(x, y))) for x, y in ring[:-1]]
        if len(floor) < 3:
            return
        top = [(x, y, z + height) for x, y, z in floor]

        # Sides.
        for i in range(len(floor)):
            j = (i + 1) % len(floor)
            self._mesh.add_quad([floor[i], floor[j], top[j], top[i]], _hex(colour), alpha=alpha)
        # Cap.
        for tri in _triangles(polygon):
            z = [-float(self.pool.depth_at(x, y)) + height for x, y in tri]
            self._mesh.add_quad(
                [
                    (tri[0][0], tri[0][1], z[0]),
                    (tri[1][0], tri[1][1], z[1]),
                    (tri[2][0], tri[2][1], z[2]),
                    (tri[2][0], tri[2][1], z[2]),
                ],
                _hex(colour),
                alpha=alpha,
            )

    def _marker(self, position: tuple[float, float], colour: Colour) -> None:
        x, y = float(position[0]), float(position[1])
        z = -float(self.pool.depth_at(x, y)) + 0.01
        r = 0.16
        self._mesh.add_quad(
            [(x - r, y - r, z), (x + r, y - r, z), (x + r, y + r, z), (x - r, y + r, z)],
            _hex(colour),
            lit=False,
        )

    def _build_surface(self) -> None:
        """The waterline, as a translucent lid over the whole boundary."""
        colour = _hex(COLOURS["shallow"])
        for tri in _triangles(self.pool.boundary):
            self._mesh.add_quad(
                [
                    (tri[0][0], tri[0][1], 0.0),
                    (tri[1][0], tri[1][1], 0.0),
                    (tri[2][0], tri[2][1], 0.0),
                    (tri[2][0], tri[2][1], 0.0),
                ],
                colour,
                alpha=0.13,
                lit=False,
            )

    def _thin_path(self, path: FloatArray | None) -> list[list[float]]:
        """Decimate a driven path to something a canvas can stroke each frame."""
        if path is None:
            return []
        pts = np.asarray(path, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            return []
        stride = max(1, len(pts) // 1200)
        pts = pts[::stride]
        z = -np.asarray(self.pool.depth_at(pts[:, 0], pts[:, 1]), dtype=float) + 0.03
        return [
            [round(float(x), 3), round(float(y), 3), round(float(zz), 3)]
            for (x, y), zz in zip(pts, z, strict=True)
        ]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        """The scene as plain data -- everything the JS renderer needs."""
        minx, miny, maxx, maxy = self.pool.bounds
        pool = self.pool
        return {
            "vertices": self._mesh.vertices,
            "faces": self._mesh.faces,
            "colours": self._mesh.colours,
            "alphas": self._mesh.alphas,
            "lit": self._mesh.lit,
            "normals": self._mesh.normals,
            "path": self.path,
            "pathColour": _hex(COLOURS["trail"]),
            "centre": [(minx + maxx) / 2, (miny + maxy) / 2, -pool.max_depth / 2],
            "span": float(max(maxx - minx, maxy - miny, pool.max_depth)),
            "zScale": DEPTH_EXAGGERATION,
            "background": _hex(COLOURS["panel"]),
        }

    def to_html(self) -> str:
        """A self-contained fragment: markup, style, data and renderer."""
        width, height = self.size
        pool = self.pool
        facts = (
            f"{pool.floor_area:.0f} m² floor · {pool.max_depth:.1f} m deep · "
            f"{len(pool.features)} feature{'s' if len(pool.features) != 1 else ''} · "
            f"{DEPTH_EXAGGERATION:g}x vertical"
        )
        return _TEMPLATE.format(
            uid=f"zb{abs(hash((self.title, width, height, len(self._mesh.faces)))):x}",
            width=width,
            height=height,
            title=html.escape(self.title),
            subtitle=html.escape(self.subtitle or facts),
            data=json.dumps(self.payload(), separators=(",", ":")),
        )

    def _repr_html_(self) -> str:  # pragma: no cover - exercised by Jupyter
        return self.to_html()

    def save(self, path: str | Path) -> Path:
        """Write a standalone HTML page. Opens in any browser, no kernel."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        page = (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(self.title)}</title>"
            f"<body style='margin:0;background:{_hex(COLOURS['panel'])}'>"
            f"{self.to_html()}</body>"
        )
        path.write_text(page, encoding="utf-8")
        return path


def preview(
    subject: Any,
    *,
    cell: float = 0.3,
    show_dirt: bool = True,
    show_path: bool = True,
    **kwargs: Any,
) -> PoolPreview:
    """Build a preview from whatever you happen to be holding.

    Accepts a :class:`~zimablue.pool.Pool`, a preset name, a
    :class:`~zimablue.recording.Recording`, or the result of a run. Given
    anything with a run attached, the floor is tinted with the dirt that was
    left behind and the driven path is drawn on it.
    """
    pool, dirt, path, title = _unpack(subject, show_dirt=show_dirt, show_path=show_path)
    kwargs.setdefault("title", title)
    return PoolPreview(pool, cell=cell, dirt=dirt, path=path, **kwargs)


def _unpack(
    subject: Any, *, show_dirt: bool, show_path: bool
) -> tuple[Pool, FloatArray | None, FloatArray | None, str | None]:
    from zimablue.pool import POOL_PRESETS

    if isinstance(subject, str):
        return POOL_PRESETS.create(subject), None, None, subject
    if isinstance(subject, Pool):
        return subject, None, None, subject.name

    # A SimulationResult carries its recording; a Recording carries itself.
    recording: Recording | None = getattr(subject, "recording", None) or subject
    if recording is None or not hasattr(recording, "frames"):
        raise TypeError(
            f"cannot preview {type(subject).__name__} -- "
            "pass a Pool, a preset name, a Recording or a run result"
        )

    from zimablue.replay.renderer import load_scene

    scene = load_scene(recording)
    dirt = None
    if show_dirt and recording.n_frames:
        dirt = np.asarray(recording.dirt_at(float(recording.frames["time"][-1])))
        if dirt.size == 0:
            dirt = None
    path = None
    if show_path and recording.n_frames:
        path = np.column_stack([recording.frames["x"], recording.frames["y"]])

    scenario = recording.manifest.get("scenario", {})
    name = scenario.get("name") if isinstance(scenario, dict) else None
    return scene.pool, dirt, path, name or scene.pool.name


# ----------------------------------------------------------------------
# The browser half. Painter's algorithm over a few thousand quads: rotate,
# project, sort back to front, fill. Everything is scoped to a unique id so
# several previews can coexist in one notebook.
# ----------------------------------------------------------------------
_TEMPLATE = """
<div id="{uid}" style="width:{width}px;max-width:100%;font-family:ui-monospace,
     SFMono-Regular,Menlo,monospace;color:#dbe7f0;background:#08111b;
     border-radius:10px;padding:10px 12px 8px;box-sizing:border-box">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px">
    <strong style="font-size:13px;letter-spacing:.02em">{title}</strong>
    <span style="font-size:10.5px;opacity:.55">{subtitle}</span>
  </div>
  <canvas width="{width}" height="{height}"
          style="width:100%;height:auto;display:block;margin-top:8px;cursor:grab;
                 touch-action:none;border-radius:6px"></canvas>
  <div style="font-size:10px;opacity:.42;margin-top:6px">
    drag to rotate · scroll to zoom · shift-drag to pan · double-click to reset
  </div>
</div>
<script>
(function() {{
  var root = document.getElementById("{uid}");
  if (!root || root.dataset.ready) return;
  root.dataset.ready = "1";
  var S = {data};
  var canvas = root.querySelector("canvas");
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var view = {{yaw: -0.9, pitch: 0.62, dist: 1.02, panX: 0, panY: 0}};
  var home = Object.assign({{}}, view);
  // Light fixed to the camera, not to the pool: turning the pool should show
  // you its shape, not sweep a shadow across it.
  var light = normalise([-0.35, -0.45, 0.82]);

  function normalise(v) {{
    var n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  }}

  // Camera space, then a perspective divide. Runs once per frame over the
  // whole mesh, which is why the mesh is quads and not triangles: half as many
  // faces to sort.
  function project(points) {{
    var cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    var cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    var c = S.centre, k = S.zScale, span = S.span;
    var scale = Math.min(W, H) / (span * view.dist);
    var eye = span * 2.6;
    var out = new Float64Array(points.length * 3);
    for (var i = 0; i < points.length; i++) {{
      var v = points[i];
      var x = v[0] - c[0], y = v[1] - c[1], z = (v[2] - c[2]) * k;
      var rx = x * cy - y * sy;
      var ry = x * sy + y * cy;
      var ry2 = ry * cp - z * sp;
      var rz = ry * sp + z * cp;
      var persp = eye / Math.max(eye - ry2, span * 0.25);
      out[i * 3] = W / 2 + (rx * persp) * scale + view.panX;
      out[i * 3 + 1] = H / 2 - (rz * persp) * scale + view.panY;
      out[i * 3 + 2] = ry2;
    }}
    return out;
  }}

  function shade(hex, amount) {{
    var n = parseInt(hex.slice(1), 16);
    var r = Math.min(255, ((n >> 16) & 255) * amount);
    var g = Math.min(255, ((n >> 8) & 255) * amount);
    var b = Math.min(255, (n & 255) * amount);
    return "rgb(" + (r | 0) + "," + (g | 0) + "," + (b | 0) + ")";
  }}

  // One entry per drawable -- a mesh face, or a segment of the driven path.
  // Both go through the same sort so the trail is hidden by the near wall
  // instead of being painted over the top of the whole pool.
  var order = new Array(S.faces.length + Math.max(0, S.path.length - 1));

  function draw() {{
    var p = project(S.vertices);
    var pp = S.path.length > 1 ? project(S.path) : null;
    var cyaw = Math.cos(view.yaw), syaw = Math.sin(view.yaw);
    var cpitch = Math.cos(view.pitch), spitch = Math.sin(view.pitch);
    ctx.fillStyle = S.background;
    ctx.fillRect(0, 0, W, H);

    // Painter's algorithm. Depth key is the centroid in camera space -- exact
    // enough for a basin, whose faces never pass through each other.
    var count = 0;
    for (var f = 0; f < S.faces.length; f++) {{
      var q = S.faces[f], d = 0;
      for (var j = 0; j < 4; j++) d += p[q[j] * 3 + 2];
      order[count++] = [d * 0.25, f];
    }}
    if (pp) {{
      for (var s = 0; s < S.path.length - 1; s++) {{
        order[count++] = [(pp[s * 3 + 2] + pp[s * 3 + 5]) * 0.5, ~s];
      }}
    }}
    order.length = count;
    order.sort(function(a, b) {{ return b[0] - a[0]; }});

    for (var k = 0; k < order.length; k++) {{
      var fi = order[k][1];
      if (fi < 0) {{
        // Bitwise-complemented index: a path segment, not a face.
        var s0 = ~fi;
        ctx.strokeStyle = S.pathColour;
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(pp[s0 * 3], pp[s0 * 3 + 1]);
        ctx.lineTo(pp[s0 * 3 + 3], pp[s0 * 3 + 4]);
        ctx.stroke();
        continue;
      }}
      var face = S.faces[fi];
      var ax = p[face[0] * 3], ay = p[face[0] * 3 + 1];
      var bx = p[face[1] * 3], by = p[face[1] * 3 + 1];
      var cx = p[face[2] * 3], cyy = p[face[2] * 3 + 1];
      var dx = p[face[3] * 3], dy = p[face[3] * 3 + 1];

      var tone = 1;
      if (S.lit[fi]) {{
        // The face normal, rotated into camera space and dotted with a light
        // that rides along with the camera.
        var n = S.normals[fi];
        var nrx = n[0] * cyaw - n[1] * syaw;
        var nry0 = n[0] * syaw + n[1] * cyaw;
        var nry = nry0 * cpitch - n[2] * spitch;
        var nrz = nry0 * spitch + n[2] * cpitch;
        var lambert = Math.abs(nrx * light[0] + nry * light[1] + nrz * light[2]);
        tone = 0.58 + 0.62 * lambert;
      }}
      var fill = shade(S.colours[fi], tone);
      ctx.globalAlpha = S.alphas[fi];
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.lineTo(cx, cyy);
      ctx.lineTo(dx, dy);
      ctx.closePath();
      ctx.fill();
      // Stroke each opaque face in its own fill colour. Canvas antialiases
      // every edge against what is behind it, so abutting tiles leave a pale
      // hairline between them and the floor reads as graph paper. Translucent
      // faces are left unstroked: there the doubled alpha along a shared edge
      // is itself the seam, and a picket fence is worse than a hairline.
      if (S.alphas[fi] > 0.985) {{
        ctx.strokeStyle = fill;
        ctx.lineWidth = 1;
        ctx.stroke();
      }}
    }}
    ctx.globalAlpha = 1;
  }}

  var pending = false;
  function schedule() {{
    if (pending) return;
    pending = true;
    requestAnimationFrame(function() {{ pending = false; draw(); }});
  }}

  var dragging = false, lastX = 0, lastY = 0, panning = false;
  canvas.addEventListener("pointerdown", function(e) {{
    dragging = true;
    panning = e.shiftKey;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  }});
  canvas.addEventListener("pointermove", function(e) {{
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (panning) {{
      view.panX += dx * (W / canvas.clientWidth);
      view.panY += dy * (W / canvas.clientWidth);
    }} else {{
      view.yaw += dx * 0.008;
      // Clamped short of straight down: at the pole the yaw axis degenerates
      // and the pool spins about nothing.
      view.pitch = Math.max(-0.05, Math.min(1.45, view.pitch + dy * 0.006));
    }}
    schedule();
  }});
  function release(e) {{
    dragging = false;
    canvas.style.cursor = "grab";
  }}
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", function(e) {{
    e.preventDefault();
    view.dist = Math.max(0.24, Math.min(3.2, view.dist * (1 + e.deltaY * 0.0016)));
    schedule();
  }}, {{passive: false}});
  canvas.addEventListener("dblclick", function() {{
    view = Object.assign({{}}, home);
    schedule();
  }});

  draw();
}})();
</script>
"""
