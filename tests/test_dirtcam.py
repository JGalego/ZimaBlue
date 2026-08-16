"""The bumper view."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.recording import Recording
from zimablue.replay.dirtcam import (
    DirtCam,
    DirtCamConfig,
    export_dirtcam,
    export_dirtcam_frames,
    render_dirtcam,
)
from zimablue.simulation import Simulation

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Recording:
    result = Simulation(pool="kidney", dirt="autumn", seed=11).run(seconds=120)
    return Recording.load(result.save(tmp_path_factory.mktemp("zbr") / "kidney.zbr"))


@pytest.fixture(scope="module")
def cam(recording) -> DirtCam:
    return DirtCam(recording)


def test_a_frame_is_a_finite_rgb_image(cam):
    frame = cam.frame(0)
    assert frame.shape == (cam.config.height, cam.config.width, 3)
    # The horizon divides by a quantity that reaches zero. An inf there
    # propagates as a NaN through the whole frame rather than as one bad pixel.
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0
    assert frame.max() <= 1.0


def test_the_horizon_is_where_the_rays_stop_meeting_the_floor(cam):
    """Sky at the top, floor at the bottom, and the boundary in between."""
    sky = cam.sky
    assert sky[0].all(), "the top row looks above the floor plane"
    assert not sky[-1].any(), "the bottom row looks at the floor under the bumper"
    # Monotone down each column: once a ray hits the floor, lower ones do too.
    for col in range(0, cam.config.width, 40):
        column = sky[:, col]
        assert not np.any(column[1:] & ~column[:-1]), "sky reappearing below the floor"


def test_the_near_field_is_metres_and_the_far_field_is_beyond_range(cam):
    inside = ~cam.sky
    assert cam._distance[inside].max() <= cam.config.far + 1e-9
    # A camera 18 cm off the floor sees the ground start right under itself.
    assert cam._distance[-1].min() < 0.5


def test_index_is_clamped_rather_than_raising(cam, recording):
    assert np.allclose(cam.frame(-5), cam.frame(0))
    assert np.allclose(cam.frame(10**9), cam.frame(recording.n_frames - 1))


def test_cleaning_makes_the_floor_lighter(cam, recording):
    """The whole argument of the view: dirt at the start, tile at the end."""
    first = cam.frame(0).mean()
    last = cam.frame(recording.n_frames - 1).mean()
    assert last > first


def test_config_changes_the_geometry(recording):
    wide = DirtCam(recording, DirtCamConfig(width=96, height=64, far=2.0))
    assert wide.frame(0).shape == (64, 96, 3)
    assert wide._distance[~wide.sky].max() <= 2.0 + 1e-9


def test_render_to_an_axes(recording):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    assert render_dirtcam(recording, 10, ax=ax) is ax
    assert ax.images, "nothing was drawn"
    plt.close(figure)


def test_export_a_contact_sheet(recording, tmp_path):
    out = export_dirtcam_frames(recording, tmp_path / "sheet.png", count=2, dpi=50)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("with_map", [True, False])
def test_export_an_animation(recording, tmp_path, with_map):
    out = export_dirtcam(
        recording,
        tmp_path / f"cam_{with_map}.gif",
        speed=600.0,
        fps=6,
        dpi=40,
        with_map=with_map,
    )
    assert out.exists() and out.stat().st_size > 0
