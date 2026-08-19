"""The chase camera, and the cleaner designs it exists to show off.

What is being defended here: that the projection is right -- a camera
behind the robot has to put the robot in front of it, at a size that shrinks
with distance, sitting on the floor rather than floating over it -- and that a
design is *only* a drawing: swapping one must move pixels and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb

pytest.importorskip("matplotlib")

from zimablue.replay.chasecam import (
    ChaseCam,
    ChaseCamConfig,
    export_chasecam,
    export_chasecam_frames,
    render_chasecam,
)
from zimablue.robot import DESIGNS, CleanerDesign, Part, make_design
from zimablue.robot.design import ellipse, rounded_rect, teardrop


@pytest.fixture(scope="module")
def recording():
    return (
        zb.Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=2)
        .run(seconds=45)
        .recording
    )


@pytest.fixture(scope="module")
def cam(recording):
    return ChaseCam(recording, ChaseCamConfig(width=160, height=90))


@pytest.fixture(scope="module")
def turning():
    """A run that actually turns.

    ``baseline_coverage`` drives long straight lanes -- over 45 seconds it
    changes heading by nineteen degrees in total, which is not enough to
    measure a camera lag against and is why the first version of that test
    failed on a working feature.
    """
    return (
        zb.Simulation(pool="kidney", robot="tracked", controller="random_bounce", seed=5)
        .run(seconds=60)
        .recording
    )


# -- the camera -----------------------------------------------------------


def test_a_frame_is_a_finite_rgb_image(cam):
    image = cam.frame(500)
    assert image.shape == (90, 160, 3)
    assert np.isfinite(image).all()
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_the_camera_sits_behind_the_robot(cam):
    x, y, heading = cam.robot_pose(400)
    cx, cy, _ = cam.camera_pose(400)
    behind = (x - cx) * np.cos(heading) + (y - cy) * np.sin(heading)
    assert behind > 0, "the camera must be behind the robot, or it cannot see it"
    assert np.hypot(x - cx, y - cy) == pytest.approx(cam.cfg.distance, rel=0.05)


def test_the_camera_heading_lags_the_robot(turning):
    """Rigidly bolting the camera makes a turn read as the pool spinning.

    Measured where the lag actually does something -- during the turns. A
    coverage controller drives straight most of the time, so the *mean* error
    over a run is near zero however laggy the camera is, which is why the first
    version of this test failed while the feature worked.
    """
    lagged = ChaseCam(turning, ChaseCamConfig(lag=0.5))
    rigid = ChaseCam(turning, ChaseCamConfig(lag=0.0))
    robot = np.unwrap(turning.frames["heading"])

    assert np.abs(rigid._heading_track() - robot).max() == pytest.approx(0.0, abs=1e-12)

    track = lagged._heading_track()
    assert np.abs(track - robot).max() > 0.1, "the camera should trail well behind in a turn"
    # And the reason to want that: the camera's own motion is smoother than the
    # robot's, which is what stops a turn reading as the pool rotating.
    assert np.abs(np.diff(track)).max() < np.abs(np.diff(robot)).max()


def test_the_lag_filter_does_not_whip_round_at_the_wrap(turning):
    """Filtering a wrapped angle sends the camera the long way round at +/-pi."""
    track = ChaseCam(turning, ChaseCamConfig(lag=0.3))._heading_track()
    assert np.abs(np.diff(track)).max() < 0.5, "a single frame should never swing that far"


def test_a_point_on_the_floor_ahead_projects_below_the_horizon(cam):
    cols, rows = cam.project(np.array([2.0]), np.array([0.0]), height=0.0)
    assert 0 <= cols[0] <= cam.config.width
    assert rows[0] > cam.config.height * 0.4, "floor ahead belongs in the lower half"


def test_projection_reduces_to_the_floor_only_form(cam):
    """The generalised projection replaced a floor-only one, and has to agree
    with it exactly where they overlap -- otherwise every debris outline moved."""
    cfg = cam.config
    ahead = np.array([0.5, 1.0, 2.5, 4.0])
    lateral = np.array([0.0, 0.3, -0.7, 0.2])

    half_w = np.tan(cfg.fov / 2.0)
    half_h = half_w * cfg.aspect()
    cos_p, sin_p = np.cos(cfg.pitch), np.sin(cfg.pitch)
    t = np.maximum(ahead * cos_p + cfg.camera_height * sin_p, 1e-6)
    old_cols = (lateral / t / half_w * 0.5 + 0.5) * (cfg.width - 1)
    old_rows = (0.5 - (sin_p - cfg.camera_height / t) / cos_p / half_h * 0.5) * (cfg.height - 1)

    cols, rows = cam.project(ahead, lateral, height=0.0)
    assert cols == pytest.approx(old_cols)
    assert rows == pytest.approx(old_rows)


def test_something_taller_projects_higher_up_the_frame(cam):
    _, floor = cam.project(np.array([1.5]), np.array([0.0]), height=0.0)
    _, top = cam.project(np.array([1.5]), np.array([0.0]), height=0.3)
    assert top[0] < floor[0], "rows count downward, so higher up means a smaller row"


def test_the_robot_is_drawn_and_is_bigger_when_the_camera_is_closer(recording):
    """The clearest signal that the machine is really being projected."""
    index = 500

    def painted(distance: float) -> int:
        """Pixels the robot covers: this frame against the same frame without it.

        The baseline has to come from a camera at the *same* distance -- move
        the camera and the floor moves too, so diffing against one fixed
        viewpoint measures the floor sliding rather than the robot's size.
        """
        config = ChaseCamConfig(width=160, height=90, distance=distance)
        with_robot = ChaseCam(recording, config).frame(index)
        blank_cam = ChaseCam(recording, config)
        blank_cam.draw_overlays = lambda image, i: None  # type: ignore[method-assign]
        return int((np.abs(with_robot - blank_cam.frame(index)).sum(axis=2) > 0.02).sum())

    near, far = painted(0.6), painted(2.5)
    assert near > 0, "nothing was drawn over the floor at all"
    assert near > far, "the robot should shrink as the camera pulls back"


def test_index_is_clamped_rather_than_raising(cam, recording):
    assert cam.frame(-50).shape == cam.frame(0).shape
    assert cam.frame(recording.n_frames + 1000).shape == cam.frame(0).shape


def test_each_design_paints_a_different_picture(recording):
    """If two designs render identically the whole feature is decorative."""
    config = ChaseCamConfig(width=140, height=80)
    frames = {}
    for name in DESIGNS.names():
        recording.manifest["robot_config"]["design"] = make_design(name).to_dict()
        frames[name] = ChaseCam(recording, config).frame(400)

    names = sorted(frames)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            difference = float(np.abs(frames[a] - frames[b]).max())
            assert difference > 0.01, f"{a} and {b} render the same"


def test_render_to_an_axes(recording):
    import matplotlib

    matplotlib.use("Agg")
    ax = render_chasecam(recording, 200, config=ChaseCamConfig(width=120, height=68))
    assert ax.images


def test_export_a_contact_sheet(recording, tmp_path):
    path = export_chasecam_frames(
        recording, tmp_path / "sheet.png", count=2, config=ChaseCamConfig(width=120, height=68)
    )
    assert path.exists() and path.stat().st_size > 0


def test_export_an_animation(recording, tmp_path):
    path = export_chasecam(
        recording,
        tmp_path / "chase.gif",
        speed=400.0,
        fps=6,
        dpi=40,
        config=ChaseCamConfig(width=120, height=68),
    )
    assert path.exists() and path.stat().st_size > 0


# -- the designs ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DESIGNS.names()))
def test_every_design_is_well_formed(name):
    design = make_design(name)
    assert design.description, f"{name} should say what it is"
    assert np.abs(design.body).max() <= 0.5 + 1e-9, "the hull must stay in the unit box"
    assert len(design.drawable()) == len(design.parts) + 1
    assert design.drawable()[0].name == "hull", "the hull is drawn first, under everything"


@pytest.mark.parametrize("name", sorted(DESIGNS.names()))
def test_every_design_survives_a_round_trip(name):
    """Designs ride inside the recording, so they have to serialise."""
    design = make_design(name)
    restored = CleanerDesign.from_dict(design.to_dict())
    assert restored.name == design.name
    assert restored.dome == design.dome
    assert np.allclose(restored.body, design.body, atol=1e-4)
    assert [p.name for p in restored.parts] == [p.name for p in design.parts]


@pytest.mark.parametrize("primitive", [teardrop, ellipse, rounded_rect])
def test_the_outline_primitives_do_not_self_intersect(primitive):
    """A wrongly wound teardrop renders as a bow tie, which is exactly what the
    first version of it did."""
    from shapely.geometry import Polygon

    polygon = Polygon(primitive())
    assert polygon.is_valid, "the outline crosses itself"
    assert polygon.area > 0.4, "it should fill most of its bounding box"


def test_the_nose_of_a_teardrop_is_the_rounded_end():
    points = teardrop()
    nose = points[np.argmax(points[:, 0])]
    tail = points[np.argmin(points[:, 0])]
    assert abs(nose[1]) < 0.05, "the nose should be a point on the centreline"
    assert abs(tail[1]) > 0.05, "the tail should be a flat edge, not a point"


def test_a_design_is_scaled_by_whatever_chassis_it_is_on():
    design = make_design("domed")
    small = design.scaled(0.30, 0.28)
    large = design.scaled(0.60, 0.56)
    assert np.ptp(large[:, 0]) == pytest.approx(2 * np.ptp(small[:, 0]))


def test_placing_a_design_rotates_and_translates_it():
    design = make_design("tracked")
    placed = design.place(design.body, 0.6, 0.3, x=3.0, y=-1.0, heading=np.pi / 2)

    # The bounding box centre, not the mean of the vertices: a teardrop's
    # vertices bunch around the rounded nose, so their mean sits forward of the
    # body origin and moves when the shape is rotated.
    centre = (placed.min(axis=0) + placed.max(axis=0)) / 2
    assert centre == pytest.approx([3.0, -1.0], abs=0.02)
    # A quarter turn puts the long axis across y.
    assert np.ptp(placed[:, 1]) == pytest.approx(0.6, abs=0.01)
    assert np.ptp(placed[:, 0]) == pytest.approx(0.3, abs=0.01)


def test_a_design_outside_the_unit_box_is_rejected():
    with pytest.raises(ValueError, match="scaled by the chassis"):
        CleanerDesign(name="huge", body=np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]))


def test_a_design_needs_at_least_a_triangle():
    with pytest.raises(ValueError, match="n >= 3"):
        CleanerDesign(name="line", body=np.array([[0.0, 0.0], [0.4, 0.0]]))


def test_a_design_changes_nothing_about_the_physics():
    """The headline promise of the whole module. A drawing that moved the
    numbers would be the worst kind of bug: invisible and everywhere."""
    plain = zb.Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=11).run(seconds=25)

    fancy_robot = zb.make_robot("tracked")
    fancy_robot.design = make_design("quad_brush")
    fancy = zb.Simulation(pool="kidney", robot=fancy_robot, dirt="autumn", seed=11).run(seconds=25)

    assert fancy.metrics.coverage == plain.metrics.coverage
    assert fancy.metrics.dirt_removed_fraction == plain.metrics.dirt_removed_fraction
    assert fancy.state.x == plain.state.x and fancy.state.y == plain.state.y


def test_an_unknown_design_name_lists_the_real_ones():
    with pytest.raises(KeyError, match="quad_brush"):
        make_design("go_faster_stripes")


def test_a_robot_carries_its_design_into_the_recording():
    robot = zb.make_robot("heavy_duty")
    assert robot.design.name == "heavy_duty"
    restored = type(robot).from_dict(robot.to_dict())
    assert restored.design.name == "heavy_duty"


def test_a_recording_without_a_design_still_replays(recording):
    """Every .zbr written before designs existed. They must not stop opening."""
    from zimablue.replay.renderer import load_scene

    stripped = recording.manifest["robot_config"].pop("design", None)
    try:
        scene = load_scene(recording)
        assert scene.design.name == "tracked", "the default should fill in"
    finally:
        if stripped is not None:
            recording.manifest["robot_config"]["design"] = stripped


def test_a_part_can_be_built_by_hand():
    """The point of the module: matching a real machine you have measured."""
    design = CleanerDesign(
        name="mine",
        body=rounded_rect(0.5, 0.42, radius=0.1),
        parts=(Part(rounded_rect(0.2, 0.2, radius=0.05), colour="#ff0000", z=2, lift=0.5),),
    )
    assert len(design.drawable()) == 2
    assert design.drawable()[-1].colour == "#ff0000"
