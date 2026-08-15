"""Pool geometry."""

from __future__ import annotations

import numpy as np
import pytest
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
