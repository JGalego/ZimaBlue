"""Debris silhouettes."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.recording import Recording
from zimablue.replay.debris_shapes import (
    DEBRIS_PALETTE,
    debris_colour,
    debris_offsets,
    debris_outline,
    debris_polygons,
)
from zimablue.simulation import Simulation

KINDS = ["leaves", "twigs"]


def _area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _self_intersects(polygon: np.ndarray) -> bool:
    """Any two non-adjacent edges crossing.

    An even-odd fill draws a self-intersecting ring with a hole in it, which is
    how the first forked twig came out.
    """

    def crosses(p, q, r, s) -> bool:
        def side(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        return bool(side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0)

    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if crosses(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return True
    return False


@pytest.mark.parametrize("kind", [*KINDS, "floating", "something_unheard_of"])
def test_outlines_are_closed_fillable_rings(kind: str):
    for index in range(6):
        polygon = debris_outline(kind, index)
        assert polygon.ndim == 2 and polygon.shape[1] == 2
        assert _area(polygon) > 0.005, f"{kind} outline encloses nothing"
        assert not _self_intersects(polygon), f"{kind} outline crosses itself"


def test_a_leaf_is_longer_than_it_is_wide():
    """The first version was a symmetric almond, which is not a leaf."""
    for index in range(6):
        polygon = debris_outline("leaves", index)
        length, width = np.ptp(polygon, axis=0)
        assert length > 2.2 * width


def test_a_leaf_has_a_stem_behind_the_blade():
    for index in range(6):
        polygon = debris_outline("leaves", index)
        assert polygon[:, 0].min() < -0.1, "no stem"


def test_choices_are_stable_across_processes():
    """Two renders of the same recording must give the same picture, so the
    per-item choices cannot come from Python's per-process salted hash."""
    assert debris_outline("leaves", 17) is debris_outline("leaves", 17)
    assert debris_colour("leaves", 17) == debris_colour("leaves", 17)
    # Spot-checked values, which is the only way to catch a salt creeping in.
    assert debris_colour("leaves", 0) in DEBRIS_PALETTE["leaves"]
    assert debris_colour("twigs", 3) in DEBRIS_PALETTE["twigs"]


def test_items_differ_from_each_other():
    colours = {debris_colour("leaves", i) for i in range(40)}
    shapes = {id(debris_outline("leaves", i)) for i in range(40)}
    assert len(colours) > 2, "a drift of leaves should not be one flat brown"
    assert len(shapes) > 1


def test_polygons_are_placed_and_scaled():
    x = np.array([2.0, 5.0])
    y = np.array([1.0, 4.0])
    size = np.array([0.09, 0.18])
    polygons = debris_polygons(x, y, size, ["leaves", "twigs"], np.arange(2))
    assert len(polygons) == 2
    for polygon, cx, cy, length in zip(polygons, x, y, size, strict=True):
        assert abs(polygon[:, 0].mean() - cx) < length
        assert abs(polygon[:, 1].mean() - cy) < length
        span = float(np.hypot(*np.ptp(polygon, axis=0)))
        assert 0.5 * length < span < 2.0 * length


def test_the_recording_carries_which_kind_each_item_is(tmp_path):
    """Without the type column a replay can only draw anonymous blobs."""
    result = Simulation(pool="kidney", dirt="autumn", seed=4).run(seconds=30)
    recording = Recording.load(result.save(tmp_path / "autumn.zbr"))

    debris = recording.debris_at(0.0)
    assert debris.shape[1] == 6
    names = recording.debris_type_names()
    assert "leaves" in names and "twigs" in names
    kinds = set(debris[:, 5].astype(int))
    assert len(kinds) > 1, "an autumn pool has more than one sort of debris"
    assert max(kinds) < len(names)


def test_a_recording_without_the_type_column_still_opens():
    """Old .zbr files predate it; padding beats refusing to read them."""
    recording = Recording(
        manifest={},
        frames={"time": np.zeros(1)},
        dirt_times=np.zeros(1, dtype=np.float32),
        debris_keyframes=np.zeros((1, 3, 5), dtype=np.float32),
    )
    debris = recording.debris_at(0.0)
    assert debris.shape == (3, 6)
    assert recording.debris_type_names() == ["leaves"]


def test_an_offset_is_an_outline_about_its_own_centre():
    size = np.array([0.09, 0.05])
    indices = np.arange(2)
    offsets = debris_offsets(size, ["leaves", "twigs"], indices)
    placed = debris_polygons(
        np.array([1.0, 5.0]), np.array([2.0, 3.0]), size, ["leaves", "twigs"], indices
    )
    for offset, polygon, centre in zip(offsets, placed, [(1.0, 2.0), (5.0, 3.0)], strict=True):
        assert offset + np.asarray(centre) == pytest.approx(polygon)
        # Centred, so it can be moved by adding wherever the item is now.
        assert abs(offset[:, 0].mean()) < float(size.max())
        assert abs(offset[:, 1].mean()) < float(size.max())


def test_debris_is_drawn_where_it_is_now_not_where_it_started(tmp_path):
    """Oversized debris gets shoved around, and every view used to ignore that.

    The outlines were built once from the first frame and never moved, on a
    stated assumption that debris settles and only ever disappears. The robot
    pushes anything too big for the intake out of the way, so the items that
    survive a run are exactly the ones that travel -- and they were drawn where
    they had been at t=0 for the whole replay.
    """
    from zimablue.replay.renderer import ReplayRenderer

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    result = Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=4).run(seconds=240)
    recording = Recording.load(result.save(tmp_path / "shoved.zbr"))

    start = recording.debris_at(0.0)
    end = recording.debris_at(recording.duration)
    still_here = end[:, 4] < 0.5
    moved = np.hypot(end[:, 0] - start[:, 0], end[:, 1] - start[:, 1])
    item = int(np.argmax(np.where(still_here, moved, -1.0)))
    assert moved[item] > end[item, 3], "no item moved further than its own size to test with"

    renderer = ReplayRenderer(recording)
    renderer.draw(recording.n_frames - 1)
    centres = np.array([path.vertices.mean(axis=0) for path in renderer._debris.get_paths()])
    assert centres.size, "nothing was drawn to check"

    now, was = end[item, 0:2], start[item, 0:2]
    # The outline nearest where the item is now is that item's, and it is a
    # whole displacement away from where the frozen version would have put it.
    drawn_at = centres[np.argmin(np.hypot(*(centres - now).T))]
    assert np.hypot(*(drawn_at - now)) < end[item, 3]
    assert np.hypot(*(drawn_at - was)) == pytest.approx(moved[item], rel=0.05)
