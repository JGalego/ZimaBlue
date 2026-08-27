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


def test_pool_walls_are_vertical_tiled_surfaces(cam):
    centre = cam.scene.pool.boundary.centroid
    target = cam._wall_ring[
        np.argmin(np.hypot(cam._wall_ring[:, 0] - centre.x, cam._wall_ring[:, 1] - centre.y))
    ]
    yaw = float(np.arctan2(target[1] - centre.y, target[0] - centre.x))

    wall, colour = cam._wall_layer(centre.x, centre.y, yaw)

    assert wall.any(), "a camera aimed at the boundary should see a wall"
    assert (wall & cam.sky).any(), "the wall should rise above the floor horizon"
    assert np.ptp(colour[wall], axis=0).max() > 0.05, "lighting and grout should shape the wall"


def test_debris_is_drawn_as_its_own_outline(cam, recording):
    """Not a disc. A leaf and a twig have to be told apart."""
    outlines = cam._outlines()
    assert outlines, "the run should have debris"
    first = recording.debris_at(0.0)
    assert len(outlines) == len(first)
    for polygon in outlines.values():
        assert polygon.ndim == 2 and polygon.shape[1] == 2
        assert len(polygon) >= 8, "an outline of three points is a triangle, not a leaf"

    # True scale: an outline is about as long as the item it stands for.
    for index, polygon in outlines.items():
        span = float(np.hypot(*(polygon.max(axis=0) - polygon.min(axis=0))))
        assert 0.5 * first[index, 3] < span < 2.0 * first[index, 3]


def test_near_debris_is_bigger_on_screen_than_far_debris(recording):
    """Foreshortening is the whole reason for projecting the outline."""
    from zimablue.replay.dirtcam import DirtCam

    cam = DirtCam(recording)
    outline = next(iter(cam._outlines().values()))
    extent = np.ptp(outline, axis=0).max()

    near = cam.project(np.array([0.4, 0.4 + extent]), np.array([0.0, 0.0]))
    far = cam.project(np.array([3.0, 3.0 + extent]), np.array([0.0, 0.0]))
    assert abs(near[1][1] - near[1][0]) > abs(far[1][1] - far[1][0])


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


# ----------------------------------------------------------------------
# How far apart displayed frames are in simulated time
# ----------------------------------------------------------------------
def test_a_window_covers_what_it_says_and_is_never_empty(recording):
    from zimablue.replay.floorcam import frame_window

    times = np.asarray(recording.frames["time"], dtype=float)
    picked = frame_window(recording, start=20.0, seconds=30.0, step=5)
    assert picked, "an empty window gives FuncAnimation nothing to draw"
    assert 20.0 <= times[picked[0]] < 20.0 + recording.frame_dt * 5
    assert times[picked[-1]] <= 50.0 + recording.frame_dt * 5
    assert np.all(np.diff(picked) == 5)

    whole = frame_window(recording, start=0.0, seconds=None, step=1)
    assert len(whole) == recording.n_frames
    # Past the end of the run, rather than raising or returning nothing.
    assert frame_window(recording, start=1e6, seconds=10.0, step=1) == [recording.n_frames - 1]


def test_the_close_up_cameras_default_to_a_rate_you_can_follow():
    """``speed / fps`` is the simulated time between displayed frames.

    The robot covers its own length in about a second. Past two or three
    seconds a frame a close-up camera shows a floor that is already swept
    rather than being swept, and dirt reads as popping into existence -- which
    is what the dirt cam did at 440x, forty seconds of pool between one frame
    and the next. The top-down view is exempt and stays fast: it shows the
    whole pool, and nothing in it moves a body length between frames.
    """
    import inspect

    from zimablue.replay import export_chasecam, export_dirtcam

    for export in (export_dirtcam, export_chasecam):
        args = inspect.signature(export).parameters
        speed = args["speed"].default
        fps = args["fps"].default
        assert speed / fps <= 3.0, f"{export.__name__} defaults to {speed / fps:.0f} s per frame"


def test_a_window_renders_fewer_frames_than_the_whole_run(recording, tmp_path):
    from PIL import Image

    whole = export_dirtcam(
        recording, tmp_path / "whole.gif", speed=24.0, fps=12, with_map=False, dpi=30
    )
    part = export_dirtcam(
        recording,
        tmp_path / "part.gif",
        speed=24.0,
        fps=12,
        with_map=False,
        dpi=30,
        seconds=30.0,
    )
    assert Image.open(part).n_frames < Image.open(whole).n_frames
    assert Image.open(part).n_frames > 1
