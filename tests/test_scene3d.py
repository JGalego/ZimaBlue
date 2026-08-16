"""The 3D replay renderer."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.recording import Recording
from zimablue.replay.scene3d import Scene3D, export_3d_frames, export_3d_movie, render_3d
from zimablue.simulation import Simulation

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Recording:
    result = Simulation(pool="sloped", dirt="light_sediment", seed=5).run(seconds=90)
    path = result.save(tmp_path_factory.mktemp("zbr") / "sloped.zbr")
    return Recording.load(path)


def test_geometry_is_built_from_the_recording(recording):
    """Not from the live preset -- an old recording must still render."""
    geo = Scene3D.build(recording)
    assert geo.walls, "a pool with walls should produce wall panels"
    assert geo.max_depth > 1.0
    assert geo.floor_z.shape == geo.xs.shape


def test_a_simple_boundary_keeps_every_wall():
    """A fixed stride once collapsed the four-sided rectangle to one panel,
    and the pool rendered with no walls at all."""
    result = Simulation(pool="rectangular", seed=1).run(seconds=20)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recording.load(result.save(Path(tmp) / "r.zbr"))
    geo = Scene3D.build(rec)
    assert len(geo.walls) == 4


def test_floor_follows_the_depth_model(recording):
    """The whole point of the 3D view: a sloped pool must actually slope."""
    geo = Scene3D.build(recording)
    inside = np.isfinite(geo.floor_z)
    depths = -geo.floor_z[inside]
    assert depths.min() == pytest.approx(1.0, abs=0.15)
    assert depths.max() == pytest.approx(2.4, abs=0.15)

    # Deeper toward +x, which is how the `sloped` preset is defined.
    left = geo.floor_z[inside & (geo.xs < geo.xs[inside].mean())].mean()
    right = geo.floor_z[inside & (geo.xs > geo.xs[inside].mean())].mean()
    assert right < left


def test_floor_is_not_drawn_outside_the_pool(recording):
    geo = Scene3D.build(recording)
    assert np.isnan(geo.floor_z[~geo.mask]).all()


def test_render_produces_an_axes(recording):
    ax = render_3d(recording, index=recording.n_frames // 2)
    assert ax.get_title()
    assert ax.name == "3d"


def test_render_clamps_out_of_range_indices(recording):
    assert render_3d(recording, index=10_000) is not None
    assert render_3d(recording, index=-5) is not None


def test_export_frames_and_movie(recording, tmp_path):
    sheet = export_3d_frames(recording, tmp_path / "sheet.png", count=2, dpi=40)
    assert sheet.exists() and sheet.stat().st_size > 5_000

    gif = export_3d_movie(recording, tmp_path / "run.gif", speed=400.0, fps=6, dpi=36)
    assert gif.exists() and gif.stat().st_size > 5_000


def test_frame_dt_comes_from_the_timestamps(recording):
    assert recording.frame_dt == pytest.approx(0.02, abs=1e-6)
