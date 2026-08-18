"""Standard pool geometries.

Shapes that between them exercise everything the simulator has to handle:
straight walls, a sloped floor, a concave corner, curvature, no corners at all,
and a blocking feature.  Dimensions are ordinary residential-pool sizes in
metres.

Each preset is an independent factory returning a fully-formed :class:`Pool`.
There is deliberately no shared "build a pool" conditional -- adding a shape
means adding a function, not extending a branch.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from zimablue.geometry import smooth_ring
from zimablue.pool.depth import CompositeDepth, ConstantDepth, PlaneSlopeDepth
from zimablue.pool.features import Drain, Obstacle, Return, Skimmer, Stairs
from zimablue.pool.pool import Pool
from zimablue.registry import Registry

__all__ = ["POOL_PRESETS", "make_pool"]

POOL_PRESETS: Registry[Pool] = Registry("pool")


def _ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 256) -> Polygon:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)]))


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
    side.  The result is then passed through :func:`~zimablue.geometry.smooth_ring`, so the final
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
    boundary = smooth_ring(Polygon(boundary.exterior), harmonics=12)

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


@POOL_PRESETS.register("stadium")
def stadium(straight: float = 6.0, radius: float = 2.5, depth: float = 1.5) -> Pool:
    """Bunimovich's stadium: a rectangle capped with two half-discs.

    Not an ordinary pool shape. It is here because the billiard flow inside it
    is *provably* chaotic and ergodic -- the flat sides defocus a bundle of
    parallel trajectories faster than the curved ends refocus it, so nearby
    paths separate exponentially and almost every trajectory eventually comes
    arbitrarily close to every point.

    Which makes it the control case for the question this project keeps asking.
    A robot that bounces at random covers a stadium well *because of the room's
    shape*, not because of anything the controller did. Compare it against the
    rectangle, where the same controller can be trapped on a closed orbit
    forever, and you can separate the two contributions.

    Bunimovich, L. A. (1979). On the ergodic properties of nowhere dispersing
    billiards. *Communications in Mathematical Physics, 65*(3), 295-312.
    """
    left = _ellipse(0.0, 0.0, radius, radius)
    right = _ellipse(straight, 0.0, radius, radius)
    middle = shapely_box(0.0, -radius, straight, radius)
    outline = left.union(middle).union(right)
    # Shift into the positive quadrant, where every other pool lives.
    boundary = Polygon(np.asarray(outline.exterior.coords) + np.array([radius, radius]))
    return Pool(
        boundary=boundary,
        depth=ConstantDepth(depth),
        name="stadium",
        material="tile",
        features=(
            Drain(
                "main_drain",
                position=(radius + straight / 2, radius),
                radius=0.25,
                flow_rate=0.15,
            ),
        ),
    )


@POOL_PRESETS.register("mushroom")
def mushroom(
    cap_radius: float = 3.2, stem_width: float = 1.4, stem_length: float = 3.0, depth: float = 1.5
) -> Pool:
    """A half-disc cap on a rectangular stem. The trap, made of geometry.

    Bunimovich's mushroom has a *mixed* phase space: a set of integrable
    trajectories that stay in the cap forever, and a chaotic set that visits
    both cap and stem, sharply divided with nothing in between. It is the
    cleanest physical demonstration that where a robot ends up can be decided
    by the room rather than by the algorithm.

    A cleaner started on the wrong side of that divide will never enter the
    stem, however long it runs and however good its random number generator
    is. No amount of coverage statistics on the cap will reveal that; only
    looking at the stem will.

    It is also not a fantasy shape -- an L-shaped pool with a narrow neck to a
    spa is the same topology with corners.

    Bunimovich, L. A. (2001). Mushrooms and other billiards with divided phase
    space. *Chaos, 11*(4), 802-808.
    """
    cap = _ellipse(cap_radius, cap_radius, cap_radius, cap_radius).intersection(
        shapely_box(0.0, cap_radius, 2 * cap_radius, 2 * cap_radius)
    )
    stem = shapely_box(
        cap_radius - stem_width / 2,
        cap_radius - stem_length,
        cap_radius + stem_width / 2,
        cap_radius + 0.05,
    )
    boundary = Polygon(cap.union(stem).exterior)
    return Pool(
        boundary=boundary,
        depth=ConstantDepth(depth),
        name="mushroom",
        material="tile",
        features=(
            Drain("main_drain", position=(cap_radius, cap_radius + 0.8), radius=0.2, flow_rate=0.1),
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
