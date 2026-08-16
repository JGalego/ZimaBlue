"""Tracing a pool out of a photograph.

Every image here is synthesised from a real preset pool, so the truth is known
exactly and the whole pipeline can be checked end to end: render a pool into a
picture, trace it back, and compare. No fixtures to keep in the repository, and
no "looks about right".
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.imaging import PoolTrace, pool_from_image, trace_pool

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

TRUTH = zb.make_pool("kidney")
SIZE = (900, 620)


def _scene(*, oblique: bool = False, glare: bool = True, seed: int = 0):
    """A photograph of a kidney pool, with a background that fights back.

    Sky along the top (blue, and in the oblique shot genuinely larger than the
    foreshortened pool), a blue parasol, a blue towel, and sun glare on the
    water. Returns the image and a world-to-pixel map so a test can name real
    landmarks.
    """
    rng = np.random.default_rng(seed)
    w, h = SIZE
    image = Image.new("RGB", SIZE, (120, 128, 110))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w, int(h * 0.20)], fill=(126, 176, 214))  # sky
    draw.rectangle([0, int(h * 0.20), w, int(h * 0.40)], fill=(96, 122, 74))  # grass
    draw.rectangle([0, int(h * 0.40), w, h], fill=(196, 186, 168))  # decking

    ring = np.asarray(TRUTH.boundary.exterior.coords)
    minx, miny, maxx, maxy = TRUTH.boundary.bounds
    pad, top, bottom = 0.09 * w, 0.44 * h, 0.96 * h
    # One scale for both axes, or the photo is of a pool that does not exist.
    scale = min((w - 2 * pad) / (maxx - minx), (bottom - top) / (maxy - miny))

    def tilt(qx, qy):
        if not oblique:
            return qx, qy
        # A denominator linear in y is what makes this a real perspective
        # rather than a squeeze that merely looks like one -- so a homography
        # can invert it exactly, and the test measures the code not the fake.
        d = 1.0 + 0.0028 * (bottom - qy)
        return (qx - w / 2) / d + w / 2, (qy - bottom) / d + bottom

    def project(points):
        points = np.asarray(points, dtype=float)
        return np.column_stack(
            tilt(pad + (points[:, 0] - minx) * scale, bottom - (points[:, 1] - miny) * scale)
        )

    outline = project(ring)
    draw.polygon([tuple(p) for p in outline], fill=(38, 122, 176))
    draw.ellipse([int(w * 0.03), int(h * 0.22), int(w * 0.22), int(h * 0.36)], fill=(46, 110, 168))
    draw.rectangle(
        [int(w * 0.86), int(h * 0.24), int(w * 0.97), int(h * 0.34)], fill=(60, 132, 190)
    )

    array = np.asarray(image).astype(np.float64)
    if glare:
        ys, xs = np.mgrid[0:h, 0:w]
        hot = np.exp(-(((xs - w * 0.58) ** 2) / 4200 + ((ys - h * 0.66) ** 2) / 900))
        array += (hot * 150)[..., None]
    array += rng.normal(0, 7, array.shape)
    blurred = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)
    )
    return blurred, project


def _seed_pixel(project) -> tuple[int, int]:
    point = TRUTH.boundary.representative_point()
    return tuple(int(v) for v in project([(point.x, point.y)])[0])


def _reference_rectangle(project, w: float = 6.0, h: float = 4.0):
    corners = project([(1.0, 0.5), (1.0 + w, 0.5), (1.0 + w, 0.5 + h), (1.0, 0.5 + h)])
    return tuple(tuple(p) for p in corners), (w, h)


# ----------------------------------------------------------------------
def test_a_top_down_photo_recovers_the_pool():
    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    traced = trace_pool(image, sample=_seed_pixel(project), width=maxx - minx)
    assert traced.area == pytest.approx(TRUTH.floor_area, rel=0.03)


def test_perspective_is_worth_correcting():
    """The headline claim: an oblique shot is wrong by tens of percent."""
    image, project = _scene(oblique=True)
    minx, _, maxx, _ = TRUTH.boundary.bounds
    seed = _seed_pixel(project)

    naive = trace_pool(image, sample=seed, width=maxx - minx)
    corrected = trace_pool(image, sample=seed, corners=_reference_rectangle(project))

    assert abs(naive.area / TRUTH.floor_area - 1) > 0.15
    assert corrected.area == pytest.approx(TRUTH.floor_area, rel=0.03)
    assert "perspective" in corrected.scale


def test_the_seed_beats_the_sky():
    """In an oblique shot the sky really is larger than the pool.

    Colour cannot separate them -- a pool and a summer sky are near enough the
    same hue -- so the seed picks by connectivity instead. Compared in pixels,
    because which region was chosen is the question, not what it scales to.
    """
    image, project = _scene(oblique=True)
    minx, _, maxx, _ = TRUTH.boundary.bounds
    seed = _seed_pixel(project)

    guessing = trace_pool(image, width=maxx - minx)
    seeded = trace_pool(image, sample=seed, width=maxx - minx)

    sky = guessing.outline_px[:, 1].mean()
    water = seeded.outline_px[:, 1].mean()
    assert water > sky, "unseeded should land on the sky, seeded on the pool"
    # The seed pixel is inside what the seeded trace outlined, and outside the
    # other one.
    from shapely.geometry import Point
    from shapely.geometry import Polygon as ShapelyPolygon

    scaled = Point(seed[0] * 0.8, seed[1] * 0.8)
    assert ShapelyPolygon(seeded.outline_px).contains(scaled)
    assert not ShapelyPolygon(guessing.outline_px).contains(scaled)


def test_glare_costs_nothing():
    """Sun on the water must not shrink the pool.

    A highlight at the water's edge is a bite out of the outline rather than a
    hole in it, and an early version lost 15% of the area to one.
    """
    lit, project = _scene(oblique=True, glare=True)
    shaded, _ = _scene(oblique=True, glare=False)
    seed = _seed_pixel(project)
    corners = _reference_rectangle(project)

    with_sun = trace_pool(lit, sample=seed, corners=corners)
    without = trace_pool(shaded, sample=seed, corners=corners)
    assert with_sun.area == pytest.approx(without.area, rel=0.03)
    assert with_sun.area == pytest.approx(TRUTH.floor_area, rel=0.03)


def test_a_swimmer_in_the_pool_is_not_a_hole_in_it():
    image, project = _scene(glare=False)
    draw = ImageDraw.Draw(image)
    middle = project([TRUTH.boundary.representative_point().coords[0]])[0]
    draw.ellipse(
        [middle[0] - 22, middle[1] - 14, middle[0] + 22, middle[1] + 14], fill=(210, 60, 55)
    )

    minx, _, maxx, _ = TRUTH.boundary.bounds
    traced = trace_pool(image, sample=(int(middle[0]) + 60, int(middle[1])), width=maxx - minx)
    assert traced.area == pytest.approx(TRUTH.floor_area, rel=0.03)
    assert len(traced.boundary.interiors) == 0


# ----------------------------------------------------------------------
def test_scale_is_required():
    image, _ = _scene()
    with pytest.raises(ValueError, match="does not contain its own scale"):
        trace_pool(image)


def test_two_scales_are_also_refused():
    image, _ = _scene()
    with pytest.raises(ValueError, match="exactly one"):
        trace_pool(image, width=8.0, metres_per_pixel=0.03)


def test_the_scale_routes_agree():
    """width, metres_per_pixel and reference are three ways to say one thing."""
    image, project = _scene()
    seed = _seed_pixel(project)
    minx, _, maxx, _ = TRUTH.boundary.bounds

    by_width = trace_pool(image, sample=seed, width=maxx - minx)
    # A 5 m span of decking, measured off the same projection.
    a, b = project([(0.0, 0.2), (5.0, 0.2)])
    by_reference = trace_pool(image, sample=seed, reference=(tuple(a), tuple(b), 5.0))
    assert by_reference.area == pytest.approx(by_width.area, rel=0.05)


def test_seeding_the_sky_traces_the_sky():
    """No cleverness here on purpose: you get the region you pointed at.

    The sky runs off the frame, and the warning says so -- which is the signal
    that the seed was in the wrong place.
    """
    image, _ = _scene()
    traced = trace_pool(image, sample=(450, 20), width=10.0)
    assert any("edge of the frame" in w for w in traced.warnings)


def test_a_sample_outside_the_image_says_so():
    image, _ = _scene()
    with pytest.raises(ValueError, match="outside"):
        trace_pool(image, sample=(5000, 5000), width=10.0)


def test_asking_for_a_region_that_is_not_there():
    image, _ = _scene()
    with pytest.raises(ValueError, match=r"only .* were found"):
        trace_pool(image, region=99, width=10.0)


# ----------------------------------------------------------------------
def test_the_trace_reports_what_it_saw():
    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    traced = trace_pool(image, sample=_seed_pixel(project), width=maxx - minx)

    assert isinstance(traced, PoolTrace)
    assert len(traced.regions) > 1, "the decoys should be listed, not silently dropped"
    assert all(r.pixels > 0 for r in traced.regions)
    summary = traced.summary()
    assert "m2" in summary and "scale" in summary


def test_a_pool_cut_off_by_the_frame_warns():
    image, _project = _scene()
    cropped = image.crop((0, 0, SIZE[0] // 2, SIZE[1]))
    traced = trace_pool(cropped, sample=(260, 470), width=6.0)
    assert any("edge of the frame" in w for w in traced.warnings)


def test_the_pool_is_usable():
    """The point of all of it: something you can actually run a cleaner in."""
    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    pool = pool_from_image(
        image, sample=_seed_pixel(project), width=maxx - minx, depth=1.7, name="from_photo"
    )
    assert pool.name == "from_photo"
    assert pool.max_depth == pytest.approx(1.7)
    assert pool.boundary.is_valid
    # It starts at the origin like every preset, so scenarios do not care where
    # in the photograph it happened to be.
    assert pool.bounds[0] == pytest.approx(0.0, abs=1e-6)
    assert pool.bounds[1] == pytest.approx(0.0, abs=1e-6)

    result = zb.Simulation(pool=pool, dirt="light_sediment", seed=1, record=False).run(seconds=45)
    assert result.metrics.coverage > 0.0


def test_a_numpy_array_works_as_well_as_a_path(tmp_path):
    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    seed = _seed_pixel(project)

    path = tmp_path / "pool.png"
    image.save(path)
    from_path = trace_pool(path, sample=seed, width=maxx - minx)
    from_array = trace_pool(np.asarray(image), sample=seed, width=maxx - minx)
    assert from_array.area == pytest.approx(from_path.area, rel=1e-6)


def test_downscaling_does_not_move_the_answer():
    """Scale arguments are given in the original image's pixels."""
    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    seed = _seed_pixel(project)
    small = trace_pool(image, sample=seed, width=maxx - minx, max_size=360)
    large = trace_pool(image, sample=seed, width=maxx - minx, max_size=900)
    assert small.area == pytest.approx(large.area, rel=0.05)


def test_metres_per_pixel_is_read_at_the_original_resolution():
    image, project = _scene()
    seed = _seed_pixel(project)
    coarse = trace_pool(image, sample=seed, metres_per_pixel=0.02, max_size=300)
    fine = trace_pool(image, sample=seed, metres_per_pixel=0.02, max_size=720)
    assert coarse.area == pytest.approx(fine.area, rel=0.05)


def test_the_overlay_draws_something():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image, project = _scene()
    minx, _, maxx, _ = TRUTH.boundary.bounds
    traced = trace_pool(image, sample=_seed_pixel(project), width=maxx - minx)
    figure, ax = plt.subplots()
    assert traced.overlay(ax=ax) is ax
    assert ax.images and ax.lines
    plt.close(figure)
