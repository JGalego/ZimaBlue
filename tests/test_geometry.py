"""The shared geometry primitives.

Everything here is pure: a polygon in, an array out. The simulator exercises
the common paths a few million times a run, so what is worth writing down is
the edges it does not reach -- an empty polygon, a cleaning head that misses
the raster entirely, an outline whose corners belong to a pixel grid rather
than to a pool.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from zimablue.geometry import Grid, polygon_segments, smooth_ring, wrap_angle


# ----------------------------------------------------------------------
def test_smoothing_keeps_the_shape_it_was_given():
    """A circle is already smooth, so smoothing must not shrink it."""
    circle = shapely_box(0, 0, 4, 4).centroid.buffer(2.0, quad_segs=64)
    smoothed = smooth_ring(circle, harmonics=12)
    assert smoothed.area == pytest.approx(circle.area, rel=0.01)
    assert smoothed.is_valid


def test_smoothing_rounds_off_corners_that_belong_to_the_pixel_grid():
    """A traced outline is a staircase; a real pool shell is not.

    Free curvature is physical, not cosmetic: a wall follower meeting a corner
    behaves differently from one tracing a curve.
    """
    staircase = Polygon([(0, 0), (2, 0), (2, 1), (4, 1), (4, 3), (0, 3)]).buffer(0)
    smoothed = smooth_ring(staircase, harmonics=6)
    # A low harmonic count cannot represent a sharp step, so the total turning
    # at the vertices falls even though the outline stays roughly in place.
    assert smoothed.area == pytest.approx(staircase.area, rel=0.25)
    assert len(smoothed.exterior.coords) > len(staircase.exterior.coords)


def test_more_harmonics_stay_closer_to_the_original():
    square = shapely_box(0, 0, 4, 4)
    loose = smooth_ring(square, harmonics=3)
    tight = smooth_ring(square, harmonics=40)
    assert abs(tight.area - square.area) < abs(loose.area - square.area)


# ----------------------------------------------------------------------
def test_a_polygon_becomes_one_row_per_edge():
    segments = polygon_segments(shapely_box(0, 0, 2, 1))
    assert segments.shape == (4, 4)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    assert sorted(lengths) == pytest.approx([1.0, 1.0, 2.0, 2.0])


def test_holes_are_edges_too():
    """A raycast that ignores the island hits the far wall through it."""
    with_hole = shapely_box(0, 0, 10, 10).difference(shapely_box(4, 4, 6, 6))
    assert len(polygon_segments(with_hole)) == 8


def test_an_empty_polygon_is_an_empty_array_not_a_crash():
    """Shape matters: callers index columns of the result unconditionally."""
    segments = polygon_segments(Polygon())
    assert segments.shape == (0, 4)
    assert segments.dtype == float


def test_a_degenerate_ring_contributes_no_zero_length_edges():
    """A repeated vertex is not an edge, and a zero-length one has no normal."""
    repeated = Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 0)])
    segments = polygon_segments(repeated)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    assert np.all(lengths > 0)


# ----------------------------------------------------------------------
@pytest.fixture
def grid() -> Grid:
    return Grid.covering(shapely_box(0.0, 0.0, 4.0, 3.0).bounds, cell=0.5)


def test_a_window_is_the_smallest_block_containing_the_disc(grid):
    window = grid.window(2.0, 1.5, 0.5)
    assert window is not None
    rows, cols = window.rows, window.cols
    assert window.mask.shape == (rows.stop - rows.start, cols.stop - cols.start)
    assert window.count > 0, "the disc's own centre cell must be inside it"
    # The window is a view, so writing through it reaches the raster.
    layer = np.zeros(grid.shape)
    view = window.view(layer)
    view[window.mask] = 1.0
    assert layer.sum() == window.count


def test_a_disc_off_the_edge_of_the_grid_has_no_window(grid):
    """None, not an empty array: every hot-loop consumer checks for it."""
    assert grid.window(-50.0, -50.0, 0.1) is None
    assert grid.window(500.0, 1.5, 0.1) is None


def test_a_window_is_clipped_to_the_grid_rather_than_running_off_it(grid):
    """A disc at the corner overhangs the raster on two sides."""
    window = grid.window(0.0, 0.0, 1.0)
    assert window is not None
    assert window.rows.start == 0 and window.cols.start == 0
    assert window.rows.stop <= grid.nrows and window.cols.stop <= grid.ncols
    assert window.view(np.zeros(grid.shape)).shape == window.mask.shape


def test_two_grids_with_the_same_geometry_share_one_cached_meshgrid():
    """Grid is frozen and hashes by value; the cache is keyed on that."""
    a = Grid.covering((0, 0, 4, 3), cell=0.5)
    b = Grid.covering((0, 0, 4, 3), cell=0.5)
    assert a == b
    assert a.cell_centers()[0] is b.cell_centers()[0]
    # Read-only, so a caller cannot corrupt what the next one reads.
    with pytest.raises(ValueError):
        a.cell_centers()[0][0, 0] = 99.0


# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (np.pi / 2, np.pi / 2),
        # The interval is half-open, so exactly +pi comes back as -pi. The
        # docstring used to claim the opposite, which is the sort of thing
        # somebody eventually relies on.
        (np.pi, -np.pi),
        (3 * np.pi, -np.pi),
        (-3 * np.pi, -np.pi),
        (2 * np.pi, 0.0),
    ],
)
def test_angles_wrap_into_one_turn(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected)


def test_wrapping_works_on_an_array_and_keeps_its_shape():
    angles = np.array([[0.0, 7.0], [-7.0, 100.0]])
    wrapped = np.asarray(wrap_angle(angles))
    assert wrapped.shape == angles.shape
    assert np.all(np.abs(wrapped) <= np.pi + 1e-12)
    # Wrapping changes the label, never the direction.
    assert np.allclose(np.cos(wrapped), np.cos(angles))
    assert np.allclose(np.sin(wrapped), np.sin(angles))
