"""Pool geometry."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

from zimablue.pool import (
    POOL_PRESETS,
    CompositeDepth,
    ConstantDepth,
    Obstacle,
    PlaneSlopeDepth,
    Pool,
    make_pool,
)


@pytest.mark.parametrize("name", POOL_PRESETS.names())
def test_every_preset_is_usable(name):
    pool = make_pool(name)
    assert pool.floor_area > 5.0
    assert pool.max_depth > 0.5
    assert pool.perimeter_length > 10.0
    assert pool.collision_segments.shape[1] == 4
    assert pool.navigable_mask().any()


@pytest.mark.parametrize("name", POOL_PRESETS.names())
def test_start_pose_is_inside_and_clear(name):
    pool = make_pool(name)
    x, y, _heading = pool.start_pose(clearance=0.3)
    assert pool.contains(x, y)
    distance, _, _, _ = pool.nearest_wall(x, y)
    assert distance >= 0.25, "start pose should not be jammed against a wall"


@pytest.mark.parametrize("name", POOL_PRESETS.names())
def test_serialisation_round_trip(name):
    original = make_pool(name)
    restored = Pool.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.floor_area == pytest.approx(original.floor_area, rel=1e-9)
    assert restored.max_depth == pytest.approx(original.max_depth)
    assert len(restored.features) == len(original.features)


def test_blocking_features_reduce_navigable_area():
    plain = Pool(shapely_box(0, 0, 10, 5), 1.5)
    blocked = Pool(
        shapely_box(0, 0, 10, 5),
        1.5,
        features=(Obstacle("block", polygon=shapely_box(1, 1, 3, 3)),),
    )
    assert blocked.floor_area == pytest.approx(plain.floor_area - 4.0)
    assert not blocked.contains(2.0, 2.0)


def test_depth_models():
    assert ConstantDepth(1.5).depth_at(0.0, 0.0) == pytest.approx(1.5)

    slope = PlaneSlopeDepth(1.0, 2.0, origin=(0, 0), direction=(1, 0), length=10.0)
    assert slope.depth_at(0.0, 0.0) == pytest.approx(1.0)
    assert slope.depth_at(10.0, 0.0) == pytest.approx(2.0)
    assert slope.depth_at(5.0, 0.0) == pytest.approx(1.5)
    # Clamped outside the ramp rather than extrapolating.
    assert slope.depth_at(20.0, 0.0) == pytest.approx(2.0)

    composite = CompositeDepth(
        base=ConstantDepth(2.0), regions=((shapely_box(0, 0, 1, 1), ConstantDepth(0.5)),)
    )
    assert composite.depth_at(0.5, 0.5) == pytest.approx(0.5)
    assert composite.depth_at(5.0, 5.0) == pytest.approx(2.0)


def test_raycast_hits_walls_at_the_right_distance():
    pool = Pool(shapely_box(0, 0, 10, 6), 1.5)
    # From the middle, facing +x, the wall is 5 m away.
    ranges = pool.raycast((5.0, 3.0), np.array([0.0, np.pi / 2, np.pi]), 20.0)
    assert ranges[0] == pytest.approx(5.0)
    assert ranges[1] == pytest.approx(3.0)
    assert ranges[2] == pytest.approx(5.0)


def test_raycast_reports_max_range_when_nothing_is_hit():
    pool = Pool(shapely_box(0, 0, 100, 100), 1.5)
    assert pool.raycast((50.0, 50.0), np.array([0.0]), 2.0)[0] == pytest.approx(2.0)


def test_unknown_preset_lists_alternatives():
    with pytest.raises(KeyError, match="kidny"):
        make_pool("kidny")


@pytest.mark.parametrize("name", POOL_PRESETS.names())
def test_depth_preserves_input_shape(name):
    """Scalar in, scalar out.

    A composite-depth pool once returned a 1-element array for scalar input,
    which NumPy will not convert back to a float -- so the l_shaped preset
    crashed on the first simulation step while every other pool worked.
    """
    pool = make_pool(name)
    x, y, _ = pool.start_pose()
    scalar = pool.depth_at(x, y)
    assert scalar.shape == ()
    assert float(scalar) > 0.0

    grid = np.array([[x, x], [x, x]])
    assert pool.depth_at(grid, grid).shape == (2, 2)
    assert pool.depth_at(np.array([x, x]), np.array([y, y])).shape == (2,)


def test_composite_depth_regions_apply_at_every_shape():
    model = CompositeDepth(
        base=ConstantDepth(2.0), regions=((shapely_box(0, 0, 1, 1), ConstantDepth(0.5)),)
    )
    assert float(model.depth_at(0.5, 0.5)) == pytest.approx(0.5)
    xs = np.array([0.5, 5.0])
    assert model.depth_at(xs, np.array([0.5, 5.0])) == pytest.approx([0.5, 2.0])


def _waist(polygon, samples=400):
    """Narrowest vertical chord in the middle 60%, over the widest anywhere."""
    from shapely.geometry import LineString

    minx, miny, maxx, maxy = polygon.bounds
    xs = np.linspace(minx, maxx, samples)
    spans = np.array(
        [polygon.intersection(LineString([(x, miny - 1), (x, maxy + 1)])).length for x in xs]
    )
    middle = (xs > minx + 0.2 * (maxx - minx)) & (xs < maxx - 0.2 * (maxx - minx))
    return spans[middle].min() / spans.max()


def test_the_kidney_is_the_size_it_says_it_is():
    """18 x 36 ft, which is a standard kidney, and 54 m2 of floor."""
    pool = make_pool("kidney")
    minx, miny, maxx, maxy = pool.boundary.bounds
    assert (minx, miny) == (0.0, 0.0)
    assert maxx == pytest.approx(12.5)
    assert maxy == pytest.approx(6.4, abs=0.1)
    assert pool.floor_area == pytest.approx(54.0, abs=1.0)


def test_the_kidney_is_actually_concave():
    """The scoop is the whole reason this preset exists.

    Solidity says how much of the convex hull the pool fills; the waist says
    how far in the scoop bites.  Both have to hold, because a shape can be
    dented at one end and still be a fat oval in the middle.
    """
    boundary = make_pool("kidney").boundary
    assert boundary.area / boundary.convex_hull.area == pytest.approx(0.89, abs=0.02)
    assert _waist(boundary) == pytest.approx(0.68, abs=0.03)


def test_the_kidney_scales_without_losing_its_shape():
    """Uniform scaling maps circles to circles, so the arcs survive it."""
    small, large = make_pool("kidney", length=8.0), make_pool("kidney", length=16.0)
    assert large.boundary.bounds[2] == pytest.approx(16.0)
    assert large.floor_area / small.floor_area == pytest.approx((16.0 / 8.0) ** 2, rel=1e-3)
    assert _waist(large.boundary) == pytest.approx(_waist(small.boundary), abs=0.01)


def test_a_deeper_scoop_takes_a_bigger_bite():
    shallow_scoop = make_pool("kidney", scoop_radius=4.0).boundary
    deep_scoop = make_pool("kidney", scoop_radius=2.0).boundary
    assert _waist(deep_scoop) < _waist(shallow_scoop)


def test_radii_that_cannot_close_say_so():
    with pytest.raises(ValueError, match="belly_radius must exceed"):
        make_pool("kidney", belly_radius=2.0)
    with pytest.raises(ValueError, match="no arc chain closes"):
        make_pool("kidney", lobe_spacing=40.0)


def test_the_kidney_floor_is_flat_shallow_sloped_then_flat_deep():
    """A hopper, not a single ramp from wall to wall.

    Pools are built this way, and it matters here: the drain sits on the deep
    flat, so a robot crossing it is not perpetually rolling downhill.
    """
    pool = make_pool("kidney")
    y = pool.boundary.centroid.y
    at = lambda x: float(pool.depth_at(x, y))  # noqa: E731
    assert at(0.5) == pytest.approx(at(4.0)) == pytest.approx(1.0)
    assert at(10.5) == pytest.approx(at(12.0)) == pytest.approx(1.8)
    assert at(4.0) < at(6.5) < at(10.0)


def test_the_kidney_drain_sits_on_the_deep_flat():
    """Otherwise the floor sweeps dirt straight past it, which it used to."""
    pool = make_pool("kidney")
    drain = next(f for f in pool.features if f.name == "main_drain")
    assert pool.boundary.contains(Point(drain.position))
    assert float(pool.depth_at(*drain.position)) == pytest.approx(pool.max_depth)


@pytest.mark.parametrize("length", [8.0, 12.5, 18.0])
def test_every_kidney_feature_lands_in_the_water(length):
    pool = make_pool("kidney", length=length)
    for feature in pool.features:
        assert pool.boundary.distance(Point(feature.position)) == 0.0, feature.name
