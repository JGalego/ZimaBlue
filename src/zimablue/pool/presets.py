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
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as shapely_box

from zimablue.pool.depth import CompositeDepth, ConstantDepth, PlaneSlopeDepth
from zimablue.pool.features import Drain, Obstacle, Return, Skimmer, Stairs
from zimablue.pool.pool import Pool
from zimablue.registry import Registry

__all__ = ["POOL_PRESETS", "make_pool"]

POOL_PRESETS: Registry[Pool] = Registry("pool")

FloatArray = NDArray[np.float64]


def _ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 256) -> Polygon:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)]))


def _arc(
    centre: tuple[float, float],
    radius: float,
    start: tuple[float, float],
    end: tuple[float, float],
    ccw: bool,
    step: float = 0.06,
) -> FloatArray:
    """Points along the circular arc of ``radius`` about ``centre``.

    Runs from ``start`` to ``end`` the short way round if ``ccw`` matches the
    sense of the turn, the long way otherwise.  Both endpoints are assumed to
    lie on the circle; only their bearings are used.

    ``step`` is a spacing in metres rather than a point count, so arcs of very
    different radii come out at the same density and the ring is uniform.
    """
    cx, cy = centre
    a0 = np.arctan2(start[1] - cy, start[0] - cx)
    a1 = np.arctan2(end[1] - cy, end[0] - cx)
    sweep = (a1 - a0) % (2 * np.pi) if ccw else -((a0 - a1) % (2 * np.pi))
    t = np.linspace(a0, a0 + sweep, max(8, int(round(abs(sweep) * radius / step))))
    return np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])


def _circles_meet(
    c1: tuple[float, float], r1: float, c2: tuple[float, float], r2: float, upper: bool
) -> tuple[float, float]:
    """One of the two points where circles ``(c1, r1)`` and ``(c2, r2)`` cross.

    ``upper`` picks the higher of the pair.  Raises if the circles are nested,
    disjoint or concentric, which is how an impossible set of kidney radii
    reports itself instead of quietly producing a self-crossing outline.
    """
    (ax, ay), (bx, by) = c1, c2
    dx, dy = bx - ax, by - ay
    d = float(np.hypot(dx, dy))
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        raise ValueError(
            f"no arc chain closes for radii {r1:.3g} and {r2:.3g} at centres {d:.3g} apart"
        )
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = float(np.sqrt(max(r1 * r1 - a * a, 0.0)))
    mx, my = ax + a * dx / d, ay + a * dy / d
    low = (mx + h * dy / d, my - h * dx / d)
    high = (mx - h * dy / d, my + h * dx / d)
    if (high[1] > low[1]) != upper:
        low, high = high, low
    return high if upper else low


def _towards(
    centre: tuple[float, float], target: tuple[float, float], radius: float
) -> tuple[float, float]:
    """The point ``radius`` from ``centre`` on the line to ``target``.

    A negative ``radius`` steps the other way.  Where two circles are tangent
    this is the point they touch, which is where one arc of the chain hands
    over to the next.
    """
    vx, vy = target[0] - centre[0], target[1] - centre[1]
    norm = float(np.hypot(vx, vy))
    return (centre[0] + radius * vx / norm, centre[1] + radius * vy / norm)


def _chord_point(polygon: Polygon, x: float, fraction: float) -> tuple[float, float]:
    """The point ``fraction`` of the way up the pool's vertical chord at ``x``.

    Lets a preset place a drain "in the middle of the deep end" and have that
    still mean the middle after its dimensions change.
    """
    _, miny, _, maxy = polygon.bounds
    chord = polygon.intersection(LineString([(x, miny - 1.0), (x, maxy + 1.0)]))
    if chord.is_empty:
        raise ValueError(f"x = {x:.3g} is outside the pool")
    if chord.geom_type != "LineString":  # a concave section can cut two chords
        chord = max(chord.geoms, key=lambda part: part.length)
    (_, y0), (_, y1) = chord.coords[0], chord.coords[-1]
    return (x, y0 + fraction * (y1 - y0))


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
def kidney(
    length: float = 12.5,
    shallow_lobe: float = 2.261,
    deep_lobe: float = 2.643,
    lobe_spacing: float = 7.596,
    lobe_rise: float = 1.109,
    belly_radius: float = 13.023,
    scoop_radius: float = 2.694,
    shallow: float = 1.0,
    deep: float = 1.8,
) -> Pool:
    """A kidney pool: 12.5 x 6.4 m, 54 m2, one concave long side.

    The outline is a chain of four circular arcs, which is how a kidney is
    actually set out on site -- stakes at the two lobe centres, a line for the
    long belly arc, a fourth for the scoop bitten out of the top:

    * ``shallow_lobe`` and ``deep_lobe`` are the radii of the two ends,
    * ``lobe_spacing`` and ``lobe_rise`` place the deep lobe's centre relative
      to the shallow one,
    * ``belly_radius`` is the long arc that runs under both lobes,
    * ``scoop_radius`` is the concave arc between them.

    Successive arcs meet where their circles are tangent, so the tangent is
    continuous the whole way round and there are no corners.  Curvature does
    jump at each join -- a real pool shell is no different, and it is a fairer
    test of a wall follower than a curve with no jumps at all.

    Everything is then scaled uniformly so the bounding length is ``length``.
    Uniform scaling maps circles to circles, so the arcs stay exact at any
    size; the defaults give 12.5 x 6.4 m over 54 m2, an 18 x 36 ft kidney.

    The floor is flat under the shallow end, ramps through a hopper over the
    middle 45% of the length, and is flat again under the deep end, which is
    where the drain goes and where the returns sweep everything.

    The concavity is the point.  A boustrophedon planner that treats the pool
    as a single cell wastes travel crossing the scoop, and the coverage
    metrics show it.

    An earlier version of this preset was a union of ellipses passed through a
    Fourier smoother.  It looked right and measured right, but it had no
    parameters, so there was no way to ask for a bigger kidney or a deeper
    scoop, and no statement of what the shape was supposed to be that a test
    could check.
    """
    if deep_lobe <= 0 or shallow_lobe <= 0 or scoop_radius <= 0:
        raise ValueError("lobe and scoop radii must be positive")
    if belly_radius <= max(shallow_lobe, deep_lobe):
        raise ValueError("belly_radius must exceed both lobe radii")

    shallow_centre = (0.0, 0.0)
    deep_centre = (lobe_spacing, lobe_rise)
    # Centres of the two long arcs: the belly contains both lobes (internal
    # tangency, hence the difference of radii) and the scoop excludes them.
    belly = _circles_meet(
        shallow_centre, belly_radius - shallow_lobe, deep_centre, belly_radius - deep_lobe, True
    )
    scoop = _circles_meet(
        shallow_centre, scoop_radius + shallow_lobe, deep_centre, scoop_radius + deep_lobe, True
    )
    belly_shallow = _towards(shallow_centre, belly, -shallow_lobe)
    belly_deep = _towards(deep_centre, belly, -deep_lobe)
    scoop_deep = _towards(deep_centre, scoop, deep_lobe)
    scoop_shallow = _towards(shallow_centre, scoop, shallow_lobe)
    ring = np.vstack(
        [
            _arc(belly, belly_radius, belly_shallow, belly_deep, ccw=True),
            _arc(deep_centre, deep_lobe, belly_deep, scoop_deep, ccw=True),
            _arc(scoop, scoop_radius, scoop_deep, scoop_shallow, ccw=False),
            _arc(shallow_centre, shallow_lobe, scoop_shallow, belly_shallow, ccw=True),
        ]
    )
    ring -= ring.min(axis=0)
    ring *= length / ring[:, 0].max()
    boundary = Polygon(ring)
    if not boundary.is_valid or not boundary.exterior.is_simple:
        raise ValueError("those radii give a self-crossing outline, not a kidney")

    hopper_start, hopper_end = 0.35 * length, 0.80 * length
    return Pool(
        boundary=boundary,
        depth=PlaneSlopeDepth(
            shallow=shallow,
            deep=deep,
            origin=(hopper_start, 0.0),
            direction=(1.0, 0.0),
            length=hopper_end - hopper_start,
        ),
        name="kidney",
        material="plaster",
        features=(
            # In the middle of the deep flat, which is where a main drain goes
            # and where the floor sends everything anyway. It used to sit
            # three metres short of the deep end, so the returns swept dirt
            # straight past it and piled it against the far wall, outside the
            # robot's reach.
            Drain(
                "main_drain",
                position=_chord_point(boundary, 0.88 * length, 0.5),
                radius=0.25,
                flow_rate=0.18,
            ),
            # Both returns sit at the shallow end and push the length of the
            # pool towards the drain. The preset used to put one at each end
            # aimed at the other, which is not how anyone plumbs a pool: the
            # two jets met in the middle and dirt piled onto the stagnation
            # line until it read as floor the robot had missed. Sweeping one
            # way puts the pile where it belongs, on the drain -- which is
            # what every other preset here already did.
            Return(
                "return_a",
                position=_chord_point(boundary, 0.11 * length, 0.28),
                direction=(1.0, 0.12),
            ),
            Return(
                "return_b",
                position=_chord_point(boundary, 0.19 * length, 0.78),
                direction=(1.0, -0.12),
            ),
            Skimmer("skimmer", position=_chord_point(boundary, 0.24 * length, 0.04)),
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
