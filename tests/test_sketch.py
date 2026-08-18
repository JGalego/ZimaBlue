"""Pools from drawings.

The fixtures are synthesised rather than photographed, which means every test
knows the answer it should get. A drawn ellipse of known size has a known area,
so the trace can be graded rather than merely inspected -- and the failure
modes a real sketch has (a lifted pen, clutter inside the outline, a shadow
across the page) are put in on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.imaging import trace_pool
from zimablue.sketch import SketchSegmenter, pool_from_sketch, trace_sketch

pytest.importorskip("PIL")

W, H = 640, 440


def draw(
    *,
    gaps: tuple[tuple[int, int], ...] = (),
    clutter: bool = False,
    shadow: bool = False,
    light_on_dark: bool = False,
    width: int = 4,
    rx: int = 200,
    ry: int = 130,
):
    """A closed elliptical outline, optionally sabotaged in the usual ways."""
    from PIL import Image, ImageDraw

    paper, ink = (
        ((18, 18, 22), (240, 240, 235)) if light_on_dark else ((248, 245, 238), (35, 32, 28))
    )
    image = Image.new("RGB", (W, H), paper)
    pen = ImageDraw.Draw(image)

    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    xs, ys = W / 2 + rx * np.cos(t), H / 2 + ry * np.sin(t)
    for i in range(len(t)):
        if any(lo <= i < hi for lo, hi in gaps):
            continue
        pen.line(
            [(xs[i], ys[i]), (xs[(i + 1) % len(t)], ys[(i + 1) % len(t)])], fill=ink, width=width
        )

    if clutter:
        pen.line([(W / 2 - 80, H / 2), (W / 2 + 80, H / 2)], fill=ink, width=3)
        pen.text((W / 2 - 30, H / 2 - 30), "deep end", fill=ink)
        pen.ellipse([W / 2 + 20, H / 2 + 20, W / 2 + 70, H / 2 + 60], outline=ink, width=3)

    array = np.asarray(image, dtype=float)
    if shadow:
        gradient = (
            np.linspace(1.0, 0.55, W)[None, :, None] * np.linspace(1.0, 0.8, H)[:, None, None]
        )
        array = array * gradient
    return np.clip(array, 0, 255).astype(np.uint8)


def area_of(image, **kwargs):
    trace = trace_pool(image, segmenter=SketchSegmenter(**kwargs), width=10.0, glare=False)
    return trace.pool().boundary.area


# -- the basic case --------------------------------------------------------


def test_a_clean_outline_is_filled():
    mask = SketchSegmenter()(draw(), None)
    assert 0.2 < mask.mean() < 0.5, "an ellipse filling half the frame should fill about a third"
    assert mask[H // 2, W // 2], "the centre of the ellipse is inside it"
    assert not mask[2, 2], "the corner of the page is not"


def test_the_traced_area_is_about_right():
    """A 400x260 px ellipse scaled to 10 m across has a knowable area.

    pi*a*b with a = 5 m and b = 5*130/200 = 3.25 m is 51 m2. Anything within a
    few percent means the fill, the trace and the scale all agree.
    """
    assert area_of(draw()) == pytest.approx(np.pi * 5.0 * 3.25, rel=0.08)


def test_it_works_on_a_dark_background():
    mask = SketchSegmenter(dark_ink=False)(draw(light_on_dark=True), None)
    assert mask[H // 2, W // 2]
    assert not mask[2, 2]


# -- the three things that make a drawing hard -----------------------------


def test_bridging_closes_a_lifted_pen():
    broken = draw(gaps=((40, 48), (150, 156)))
    assert 0.2 < SketchSegmenter()(broken, None).mean() < 0.5


def test_an_unbridged_gap_is_an_error_rather_than_an_empty_pool():
    """The two fills fail in opposite directions, which is why border filling
    is the default.

    A border fill cannot leak outward -- the border is where it starts. What it
    does instead is reach *through* the gap into the interior, leaving almost
    nothing, and "almost nothing" is detectable: an enclosed outline fills many
    times what the line itself covers.
    """
    broken = draw(gaps=((40, 48), (150, 156)))
    with pytest.raises(ValueError, match="does not enclose anything"):
        SketchSegmenter(bridge=0.0)(broken, None)

    with pytest.raises(ValueError, match="not closed"):
        SketchSegmenter(bridge=0.002)(draw(gaps=((30, 90),)), None)


def test_a_seed_fill_leaks_through_the_gap_instead():
    """The failure the default avoids: a pool the size of the photograph,
    which looks like a scale error and gets debugged as one."""
    broken = draw(gaps=((40, 48),))
    leaked = SketchSegmenter(bridge=0.0, fill_from_border=False)(broken, (W // 2, H // 2))
    assert leaked.mean() > 0.9


def test_clutter_inside_the_outline_is_part_of_the_pool():
    """An arrow drawn inside the pool is still inside the pool.

    Filling outward from a seed would stop at the arrow and return a shape with
    a bite out of it; filling inward from the page border cannot.
    """
    plain = area_of(draw())
    cluttered = area_of(draw(clutter=True))
    assert cluttered == pytest.approx(plain, rel=0.05)


def test_a_shadow_across_the_page_does_not_become_ink():
    """A phone photo of paper has a gradient. A global threshold either loses
    the line in the dark corner or calls the shadow ink."""
    assert area_of(draw(shadow=True)) == pytest.approx(area_of(draw()), rel=0.08)


def test_a_global_threshold_would_have_failed_on_the_shadow():
    """Pins the reason the local window exists, so nobody simplifies it away.

    With the window as wide as the image the estimate is global, and the
    gradient swamps the line.
    """
    shadowed = draw(shadow=True)
    local = SketchSegmenter().ink_mask(shadowed)
    with np.errstate(invalid="ignore"):
        globalish = SketchSegmenter(window=2.0).ink_mask(shadowed)
    assert local.mean() < 0.1, "the line is a small share of the page"
    assert globalish.mean() > local.mean() * 2, "a global threshold should smear"


# -- refusing to guess ------------------------------------------------------


def test_a_blank_page_is_a_clear_error():
    blank = np.full((H, W, 3), 250, dtype=np.uint8)
    with pytest.raises(ValueError, match="not a drawing"):
        SketchSegmenter()(blank, None)


def test_a_textured_photograph_is_a_clear_error():
    """Local thresholding is remarkably hard to fool -- a big black shape has a
    black local background and reads as no ink at all. What does trip the
    ceiling is fine texture: grass, gravel, a halftone scan."""
    checker = np.indices((H, W)).sum(axis=0) % 2
    textured = (checker[..., None] * np.uint8(255)).repeat(3, axis=2).astype(np.uint8)
    with pytest.raises(ValueError, match="photograph or a"):
        SketchSegmenter()(textured, None)


def test_seed_filling_needs_a_seed():
    with pytest.raises(ValueError, match="needs sample"):
        SketchSegmenter(fill_from_border=False)(draw(), None)


def test_a_seed_on_the_line_is_a_clear_error():
    with pytest.raises(ValueError, match="on the line itself"):
        SketchSegmenter(fill_from_border=False)(draw(), (W // 2, H // 2 - 130))


def test_both_fills_agree_when_the_outline_is_closed():
    """Interior clutter that does not span the pool is walked around by either
    fill, so on a well-drawn sketch the choice does not matter. It matters when
    the drawing is bad, which is the case worth defaulting for."""
    from_border = SketchSegmenter()(draw(clutter=True), None)
    from_seed = SketchSegmenter(fill_from_border=False)(draw(clutter=True), (W // 2, H // 2 - 60))
    assert from_seed.mean() == pytest.approx(from_border.mean(), rel=0.02)


# -- the whole pipeline -----------------------------------------------------


def test_a_sketch_becomes_a_pool_that_can_be_cleaned(tmp_path):
    from PIL import Image

    path = tmp_path / "napkin.png"
    Image.fromarray(draw(gaps=((40, 47),), clutter=True, shadow=True)).save(path)

    pool = pool_from_sketch(path, width=9.0, depth=1.5)
    assert pool.boundary.is_valid
    assert pool.boundary.area > 10.0

    result = zb.Simulation(pool=pool, controller="baseline_coverage", seed=1).run(minutes=4)
    assert result.metrics.coverage > 0.15


def test_the_defaults_differ_from_a_photograph(tmp_path):
    """Glare repair is meaningless on a drawing, and a hand-drawn line is
    wobbly at a scale nobody meant."""
    from PIL import Image

    path = tmp_path / "s.png"
    Image.fromarray(draw()).save(path)

    sketched = trace_sketch(path, width=9.0)
    photographic = trace_pool(path, segmenter=SketchSegmenter(), width=9.0, glare=False)
    assert len(sketched.pool().boundary.exterior.coords) <= len(
        photographic.pool().boundary.exterior.coords
    )


def test_sketch_settings_pass_through_trace_sketch(tmp_path):
    from PIL import Image

    path = tmp_path / "s.png"
    Image.fromarray(draw(light_on_dark=True)).save(path)
    trace = trace_sketch(path, width=9.0, dark_ink=False)
    assert trace.pool().boundary.area > 10.0


def test_the_scale_is_still_required(tmp_path):
    from PIL import Image

    path = tmp_path / "s.png"
    Image.fromarray(draw()).save(path)
    with pytest.raises(ValueError):
        trace_sketch(path)


# -- the helpers ------------------------------------------------------------


def test_dilate_and_erode_are_inverse_on_a_solid_block():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    from zimablue.sketch import _dilate, _erode

    assert np.array_equal(_erode(_dilate(mask, 3), 3), mask)


def test_the_box_blur_matches_a_direct_mean():
    from zimablue.sketch import _box_blur

    rng = np.random.default_rng(1)
    image = rng.random((40, 50))
    blurred = _box_blur(image, 2)
    padded = np.pad(image, 2, mode="edge")
    assert blurred[10, 10] == pytest.approx(padded[10 : 10 + 5, 10 : 10 + 5].mean())


def test_the_interior_fill_does_not_wrap_around_the_page():
    """np.roll wraps, which would join the left of the page to the right and
    let the fill escape through the edge."""
    from zimablue.sketch import _interior

    ring = np.zeros((40, 40), dtype=bool)
    ring[5:35, 5] = ring[5:35, 34] = True
    ring[5, 5:35] = ring[34, 5:35] = True
    inside = _interior(ring)
    assert inside[20, 20]
    assert not inside[0, 0] and not inside[0, 39] and not inside[39, 0]
