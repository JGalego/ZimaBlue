"""The interactive pool preview."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from shapely.geometry import box as shapely_box

import zimablue as zb
from zimablue.notebook import PoolPreview, preview
from zimablue.recording import Recording
from zimablue.simulation import Simulation


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Recording:
    result = Simulation(pool="kidney", dirt="autumn", seed=3).run(seconds=90)
    return Recording.load(result.save(tmp_path_factory.mktemp("zbr") / "kidney.zbr"))


def _payload(html: str) -> dict:
    """Pull the scene back out of the rendered page."""
    match = re.search(r"var S = (\{.*?\});\n", html, re.DOTALL)
    assert match, "the page should carry its scene as JSON"
    return json.loads(match.group(1))


def test_preview_needs_no_matplotlib():
    """The point of rendering in the browser: no plotting stack involved.

    Checked in a fresh interpreter, because by the time the rest of this suite
    has run something else will have imported matplotlib anyway.
    """
    import subprocess
    import sys

    code = (
        "import sys, zimablue as zb; zb.preview('rectangular').to_html();"
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_a_preset_name_is_enough():
    view = preview("kidney")
    assert isinstance(view, PoolPreview)
    assert view.title == "kidney"
    assert view._mesh.faces, "the pool should have geometry"


def test_every_face_indexes_real_vertices():
    view = preview("rectangular")
    n = len(view._mesh.vertices)
    assert all(0 <= i < n for face in view._mesh.faces for i in face)
    # The payload arrays are parallel; a mismatch silently mis-colours faces.
    mesh = view._mesh
    assert len(mesh.faces) == len(mesh.colours) == len(mesh.alphas)
    assert len(mesh.faces) == len(mesh.lit) == len(mesh.normals)


def test_normals_are_unit_length():
    view = preview("sloped")
    lengths = np.linalg.norm(np.asarray(view._mesh.normals), axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def _horizontal_area(view: PoolPreview) -> float:
    """Total footprint of the faces that lie flat.

    Walls and coping are vertical, so they project to no horizontal area at
    all; what survives is the floor. That makes the sum directly comparable to
    the pool's own navigable area.
    """
    verts = np.asarray(view._mesh.vertices)
    total = 0.0
    for face in view._mesh.faces:
        pts = verts[face]
        if pts[:, 2].max() > -0.005:  # the waterline lid and the coping
            continue
        x, y = pts[:, 0], pts[:, 1]
        total += 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return total


def test_the_floor_is_clipped_to_the_pool_not_to_the_grid():
    """A raster floor inside a curved wall leaves a gap you can see through.

    Whole cells only would overshoot by the sliver of every boundary cell that
    the outline cuts off -- 1.3% of a kidney at this resolution, and visible as
    a staircase against a smooth wall. Measured on a bare boundary because
    drains and returns are drawn as floor patches and would muddy the sum.
    """
    kidney = zb.make_pool("kidney")
    pool = zb.Pool(boundary=kidney.boundary, depth=kidney.depth_model, name="bare")
    view = PoolPreview(pool, cell=0.4)
    assert _horizontal_area(view) == pytest.approx(pool.floor_area, rel=0.002)


def test_depth_shows_up_as_height():
    view = preview("sloped")
    z = np.asarray(view._mesh.vertices)[:, 2]
    assert z.min() < -2.0, "the deep end should sit well below the waterline"
    assert z.max() >= 0.0, "the coping should sit at or above it"


def test_features_add_geometry():
    plain = zb.Pool(boundary=shapely_box(0, 0, 6, 4), depth=zb.ConstantDepth(1.2), name="plain")
    with_island = zb.Pool(
        boundary=shapely_box(0, 0, 6, 4),
        depth=zb.ConstantDepth(1.2),
        name="island",
        features=(zb.Obstacle("rock", polygon=shapely_box(2, 1.5, 3, 2.5), height=0.6),),
    )
    # The island is carved out of the floor and added back as a solid, so the
    # navigable footprint drops by its area.
    assert _horizontal_area(PoolPreview(plain)) - _horizontal_area(
        PoolPreview(with_island)
    ) == pytest.approx(0.0, abs=0.05)
    assert len(PoolPreview(with_island)._mesh.faces) != len(PoolPreview(plain)._mesh.faces)


def test_a_recording_brings_its_dirt_and_its_path(recording):
    view = preview(recording)
    assert len(view.path) > 10, "the driven path should be carried through"
    plain = preview(recording, show_dirt=False, show_path=False)
    assert plain.path == []
    # Dirt tints the floor, so the two should not agree on colours.
    assert view._mesh.colours != plain._mesh.colours


def test_the_page_is_self_contained_and_scoped():
    html = preview("kidney").to_html()
    assert "<canvas" in html and "<script>" in html
    assert "http://" not in html and "https://" not in html, "no external resources"
    uid = re.search(r'<div id="(zb[0-9a-f]+)"', html)
    assert uid, "the container needs an id so two previews cannot collide"
    assert html.count(uid.group(1)) >= 2, "the script must look up its own container"


def test_the_payload_round_trips_as_json():
    payload = _payload(preview("rectangular").to_html())
    assert len(payload["faces"]) == len(payload["colours"])
    assert len(payload["normals"]) == len(payload["faces"])
    assert payload["span"] > 0
    assert payload["zScale"] > 1.0


def test_repr_html_is_what_jupyter_calls():
    view = preview("kidney")
    assert view._repr_html_() == view.to_html()


def test_save_writes_a_standalone_page(tmp_path):
    out = preview("kidney").save(tmp_path / "pool.html")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<canvas" in text


def test_an_unrenderable_object_says_so():
    with pytest.raises(TypeError, match="cannot preview"):
        preview(42)
