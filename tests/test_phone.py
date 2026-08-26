"""Multi-view phone pool reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

import zimablue as zb
from zimablue.imaging import PoolTrace, Region


def make_trace(boundary: Polygon) -> PoolTrace:
    return PoolTrace(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        mask=np.ones((2, 2), dtype=bool),
        outline_px=np.asarray(boundary.exterior.coords),
        boundary=boundary,
        regions=[Region(4, (0.5, 0.5), False)],
    )


def test_phone_fusion_uses_polygon_majority_not_an_average_box():
    traces = [
        make_trace(box(0.0, 0.0, 10.0, 5.0)),
        make_trace(box(0.1, 0.0, 10.1, 5.0)),
        make_trace(box(0.0, 0.1, 10.0, 5.1)),
    ]

    result = zb.fuse_phone_traces(traces)

    assert result.quorum == 2
    assert result.boundary.area == pytest.approx(49.99)
    assert result.confidence > 0.96
    assert result.area_variation == pytest.approx(0.0)


def test_phone_fusion_rejects_views_without_shared_geometry():
    with pytest.raises(ValueError, match="no quorum overlap"):
        zb.fuse_phone_traces([make_trace(box(0, 0, 1, 1)), make_trace(box(2, 2, 3, 3))])


def test_phone_depth_fit_recovers_a_measured_plane():
    boundary = box(0.0, 0.0, 10.0, 4.0)
    observations = [
        zb.DepthObservation(x, y, 1.0 + 0.1 * x + 0.05 * y)
        for x, y in [(0, 0), (10, 0), (0, 4), (10, 4), (5, 2)]
    ]

    depth = zb.fit_phone_depth(boundary, observations)

    assert float(depth.depth_at(2.0, 3.0)) == pytest.approx(1.35)
    assert float(depth.depth_at(10.0, 4.0)) == pytest.approx(2.2)


def test_phone_depth_fit_keeps_constant_measurements_constant():
    depth = zb.fit_phone_depth(
        box(0, 0, 2, 2),
        [zb.DepthObservation(0, 0, 1.7), zb.DepthObservation(2, 2, 1.7)],
    )
    assert isinstance(depth, zb.ConstantDepth)
    assert depth.max_depth == pytest.approx(1.7)


def test_phone_views_are_rectified_and_fused_end_to_end():
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    first[10:90, 15:85] = (20, 130, 210)
    second = np.zeros((100, 100, 3), dtype=np.uint8)
    second[12:88, 16:84] = (20, 130, 210)
    corners = ((0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0))
    views = [
        zb.PhoneView(first, corners, (10.0, 10.0), sample=(50, 50)),
        zb.PhoneView(second, corners, (10.0, 10.0), sample=(50, 50)),
    ]

    pool = zb.pool_from_phones(views, depth=1.8, closing=0, close_gaps=0, grow=0)

    assert 50.0 < pool.floor_area < 70.0
    assert pool.max_depth == pytest.approx(1.8)
