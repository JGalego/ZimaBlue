"""Standard pool geometries.

Six shapes that between them exercise everything the simulator has to handle:
straight walls, a sloped floor, a concave corner, curvature, no corners at all,
and a blocking feature.  Dimensions are ordinary residential-pool sizes in
metres.

Each preset is an independent factory returning a fully-formed :class:`Pool`.
There is deliberately no shared "build a pool" conditional -- adding a shape
means adding a function, not extending a branch.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as shapely_box

from zimablue.pool.depth import CompositeDepth, ConstantDepth, PlaneSlopeDepth
from zimablue.pool.features import Drain, Obstacle, Return, Skimmer, Stairs
from zimablue.pool.pool import Pool
from zimablue.registry import Registry

__all__ = ["POOL_PRESETS", "make_pool"]

POOL_PRESETS: Registry[Pool] = Registry("pool")


def _ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 256) -> Polygon:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)]))


def _smooth_ring(polygon: Polygon, harmonics: int = 12, n: int = 512) -> Polygon:
    """Replace a boundary with a low-order Fourier curve through it.

    Boolean operations on ellipses leave cusps where the operands cross, and a
    buffer fillet only trades a corner for a curvature jump.  Resampling the
    ring at uniform arc length and keeping the first ``harmonics`` Fourier
    coefficients of x(s) and y(s) instead yields a curve that is smooth
    everywhere by construction -- a trigonometric polynomial has no corners.

    Free curvature also matters physically: a wall follower that meets a corner
    behaves differently from one tracing a smooth curve, and a kidney pool
    really is smooth.
    """
    ring = LineString(polygon.exterior.coords)
    stations = np.linspace(0.0, ring.length, n, endpoint=False)
    points = np.array([ring.interpolate(float(s)).coords[0] for s in stations])
    smoothed = []
    for axis in (0, 1):
        spectrum = np.fft.rfft(points[:, axis])
        spectrum[harmonics + 1 :] = 0.0
        smoothed.append(np.fft.irfft(spectrum, n))
    return Polygon(np.column_stack(smoothed))


@POOL_PRESETS.register("rectangular")
def rectangular(length: float = 10.0, width: float = 5.0, depth: float = 1.6) -> Pool:
    """A plain 10 x 5 m rectangular pool with a flat floor."""
    return Pool(
        boundary=shapely_box(0.0, 0.0, length, width),
        depth=ConstantDepth(depth),
        name="rectangular",
        material="plaster",
        features=(
            Drain("main_drain", position=(length / 2, width / 2), radius=0.25, flow_rate=0.15),
            Return("return_a", position=(0.15, width * 0.25), direction=(1.0, 0.0)),
            Return("return_b", position=(0.15, width * 0.75), direction=(1.0, 0.0)),
            Skimmer("skimmer", position=(length - 0.2, width / 2)),
        ),
    )


@POOL_PRESETS.register("sloped")
def sloped(length: float = 12.0, width: float = 5.0) -> Pool:
    """Rectangular pool ramping from a 1.0 m shallow end to a 2.4 m deep end."""
    return Pool(
        boundary=shapely_box(0.0, 0.0, length, width),
        depth=PlaneSlopeDepth(
            shallow=1.0, deep=2.4, origin=(0.0, 0.0), direction=(1.0, 0.0), length=length
        ),
        name="sloped",
        material="plaster",
        features=(
            Drain("deep_drain", position=(length - 1.5, width / 2), radius=0.3, flow_rate=0.2),
            Return("return_a", position=(0.15, width * 0.3), direction=(1.0, 0.0)),
            Skimmer("skimmer", position=(0.3, width - 0.3)),
        ),
    )


@POOL_PRESETS.register("l_shaped")
def l_shaped() -> Pool:
    """An L: a 12 x 5 m main leg with a 5 x 4 m wing, i.e. one concave corner.

    The concave corner is the interesting part -- naive boustrophedon coverage
    strands area behind it, which is exactly the failure the metrics should
    surface.
    """
    boundary = Polygon([(0, 0), (12, 0), (12, 5), (5, 5), (5, 9), (0, 9)])
    return Pool(
        boundary=boundary,
        depth=CompositeDepth(
            base=ConstantDepth(1.5),
            regions=((shapely_box(0, 5, 5, 9), ConstantDepth(1.1)),),
        ),
        name="l_shaped",
        material="tile",
        features=(
            Drain("main_drain", position=(8.0, 2.5), radius=0.25, flow_rate=0.15),
            Return("return_a", position=(0.15, 2.0), direction=(1.0, 0.0)),
            Return("return_b", position=(2.5, 8.85), direction=(0.0, -1.0)),
            Skimmer("skimmer", position=(11.7, 4.7)),
        ),
    )


@POOL_PRESETS.register("kidney")
def kidney() -> Pool:
    """A kidney-bean pool: a 12.5 x 6.4 m bean with one concave long side.

    Composed from ellipse booleans rather than a hand-typed vertex list: a body
    ellipse, a lobe that fattens the deep end, a closing buffer to erase the
    waist where the two meet, and a large offset circle that scoops the shallow
    side.  The result is then passed through :func:`_smooth_ring`, so the final
    boundary is a smooth curve with no cusps left over from the booleans.

    The concavity is the point: a boustrophedon planner that treats the pool as
    a single cell wastes travel crossing the scoop, and coverage metrics show it.
    """
    body = _ellipse(6.0, 3.9, 6.0, 3.1).union(_ellipse(9.6, 4.5, 2.9, 2.7))
    body = body.buffer(0.5).buffer(-0.5)  # closing: erase the waist
    scoop = _ellipse(4.6, 9.3, 4.3, 4.3)
    boundary = body.difference(scoop)
    if boundary.geom_type == "MultiPolygon":  # pragma: no cover - defensive
        boundary = max(boundary.geoms, key=lambda g: g.area)
    boundary = _smooth_ring(Polygon(boundary.exterior), harmonics=12)

    return Pool(
        boundary=boundary,
        depth=PlaneSlopeDepth(
            shallow=1.0, deep=2.0, origin=(0.0, 0.0), direction=(1.0, 0.0), length=12.0
        ),
        name="kidney",
        material="plaster",
        features=(
            Drain("main_drain", position=(9.4, 4.2), radius=0.25, flow_rate=0.18),
            Return("return_a", position=(1.4, 3.6), direction=(1.0, 0.1)),
            Return("return_b", position=(11.8, 4.6), direction=(-1.0, -0.2)),
            Skimmer("skimmer", position=(3.0, 1.4)),
        ),
    )


@POOL_PRESETS.register("oval")
def oval(length: float = 11.0, width: float = 6.0, depth: float = 1.5) -> Pool:
    """A smooth oval -- no corners anywhere, which stresses wall following."""
    return Pool(
        boundary=_ellipse(length / 2, width / 2, length / 2, width / 2),
        depth=ConstantDepth(depth),
        name="oval",
        material="fiberglass",
        features=(
            Drain("main_drain", position=(length / 2, width / 2), radius=0.25, flow_rate=0.15),
            Return("return_a", position=(1.0, width / 2), direction=(1.0, 0.0)),
            Skimmer("skimmer", position=(length - 1.0, width / 2)),
        ),
    )


@POOL_PRESETS.register("stairs")
def stairs() -> Pool:
    """Rectangular pool with a corner stair block and a ladder foot.

    Both features are blocking, so navigable area is strictly less than the
    boundary area -- a good check that coverage is measured against the right
    denominator.
    """
    length, width = 10.0, 5.0
    stair_block = shapely_box(0.0, 0.0, 2.2, 2.2)
    return Pool(
        boundary=shapely_box(0.0, 0.0, length, width),
        depth=PlaneSlopeDepth(
            shallow=1.1, deep=2.0, origin=(0.0, 0.0), direction=(1.0, 0.0), length=length
        ),
        name="stairs",
        material="vinyl",
        features=(
            Stairs("entry_stairs", polygon=stair_block, steps=3, top_depth=0.3, bottom_depth=1.1),
            Obstacle("ladder_foot", polygon=shapely_box(9.4, 3.6, 9.8, 4.0), height=0.4),
            Drain("main_drain", position=(7.5, 2.5), radius=0.25, flow_rate=0.16),
            Return("return_a", position=(0.15, 4.0), direction=(1.0, 0.0)),
            Skimmer("skimmer", position=(9.7, 0.4)),
        ),
    )


def make_pool(name: str, **kwargs: object) -> Pool:
    """Build a pool preset by name.

    >>> make_pool("kidney").name
    'kidney'
    """
    return POOL_PRESETS.create(name, **kwargs)
