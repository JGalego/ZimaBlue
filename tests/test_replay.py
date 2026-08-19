"""Replay rendering. Headless: these assert that it draws, not how it looks."""

from __future__ import annotations

from itertools import pairwise

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

import numpy as np

from zimablue.replay import (
    SPEEDS,
    ReplayRenderer,
    export_frames,
    export_summary,
    load_scene,
)
from zimablue.replay.renderer import PALETTE, _hex_rgb


@pytest.fixture(scope="module")
def autumn_recording(tmp_path_factory):
    """A run with dirt that piles up, which light sediment does not do."""
    import zimablue as zb

    result = zb.Simulation(pool="kidney", dirt="autumn", seed=11).run(seconds=180)
    return zb.Recording.load(result.save(tmp_path_factory.mktemp("zbr") / "autumn.zbr"))


def test_scene_is_rebuilt_from_the_recording(short_run):
    scene = load_scene(short_run.recording)
    assert scene.pool.floor_area == pytest.approx(50.0, rel=1e-6)
    assert scene.swath > 0.2
    assert len(scene.sonar_angles) == 3


def test_renderer_draws_first_and_last_frames(short_run):
    renderer = ReplayRenderer(short_run.recording)
    renderer.draw(0)
    renderer.draw(short_run.recording.n_frames - 1)
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_seeking_is_order_independent(short_run):
    """Scrubbing backwards must not show coverage the robot has not reached."""
    renderer = ReplayRenderer(short_run.recording)
    last = short_run.recording.n_frames - 1
    renderer.draw(last)
    forward = renderer._coverage_at(last // 2)
    renderer.draw(0)
    backward = renderer._coverage_at(last // 2)
    assert forward == backward
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_coverage_curve_is_monotonic(short_run):
    renderer = ReplayRenderer(short_run.recording)
    values = [renderer._coverage_at(i) for i in range(0, short_run.recording.n_frames, 100)]
    assert all(b >= a - 1e-9 for a, b in pairwise(values))
    assert values[-1] == pytest.approx(short_run.metrics.coverage, abs=0.02)
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_export_frames_writes_images(short_run, tmp_path):
    written = export_frames(short_run.recording, tmp_path, count=3)
    assert len(written) == 3
    assert all(p.exists() and p.stat().st_size > 5000 for p in written)


def test_export_summary_writes_an_image(short_run, tmp_path):
    path = export_summary(short_run.recording, tmp_path / "summary.png")
    assert path.exists() and path.stat().st_size > 10000


def test_speeds_cover_the_documented_range():
    assert 0.25 in SPEEDS and 10.0 in SPEEDS
    assert list(SPEEDS) == sorted(SPEEDS)


def test_dirt_overlay_shrinks_as_the_run_proceeds(short_run):
    rec = short_run.recording
    assert rec.dirt_at(rec.duration).sum() <= rec.dirt_at(0.0).sum()


def test_renderer_handles_a_pool_with_obstacles():
    from zimablue.simulation import Simulation

    result = Simulation(pool="stairs", seed=1).run(seconds=30)
    renderer = ReplayRenderer(result.recording)
    renderer.draw(10)
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)


def test_the_hud_dirt_bar_agrees_with_the_run_that_produced_it(tmp_path):
    """They disagreed by seven points, and the bar was the wrong one.

    The dirt keyframes hold the raster field only. An autumn pool keeps about
    a quarter of its mass in leaves and twigs, so a bar built from the
    keyframes alone read several points below the run's own dirt-removed
    metric -- and the two numbers sat on the same screen.
    """
    import zimablue as zb
    from zimablue.replay.renderer import ReplayRenderer

    result = zb.Simulation(
        pool="kidney", robot="tracked", dirt="autumn", controller="baseline_coverage", seed=42
    ).run(minutes=3)
    recording = zb.Recording.load(result.save(tmp_path / "hud.zbr"))
    renderer = ReplayRenderer(recording)
    last = recording.n_frames - 1
    shown = renderer._dirt_removed_at(float(recording.frames["time"][last]))
    assert shown == pytest.approx(result.metrics.dirt_removed_fraction, abs=0.01)


def test_the_hud_reports_debris_the_intake_cannot_take(tmp_path):
    import zimablue as zb
    from zimablue.replay.renderer import ReplayRenderer

    result = zb.Simulation(
        pool="kidney", robot="tracked", dirt="autumn", controller="baseline_coverage", seed=42
    ).run(minutes=2)
    recording = zb.Recording.load(result.save(tmp_path / "hud2.zbr"))
    renderer = ReplayRenderer(recording)
    tally = renderer._debris_tally(float(recording.frames["time"][-1]))
    assert tally is not None
    _, total, oversize, ceiling = tally
    assert total == (
        result.metrics.debris_collected
        + result.metrics.debris_skimmed
        + result.metrics.debris_remaining
    )
    assert oversize == result.metrics.debris_oversize
    assert ceiling == pytest.approx(result.metrics.dirt_ceiling, abs=0.01)


def test_a_heap_of_dirt_is_drawn_as_more_than_a_flat_patch():
    """The overlay is calibrated on the dirt a run *starts* with.

    Dirt accumulates now -- the returns sweep the floor into a heap at the
    drain, more than twenty times the initial 92nd percentile in a kidney -- so
    alpha alone pins at maximum across the whole heap and only its edge appears
    to move. Past the scale the colour darkens instead.
    """
    from zimablue.replay.renderer import _dirt_alpha

    navigable = np.ones((2, 2), dtype=bool)
    vmax = 0.2
    shades = [
        _dirt_alpha(np.full((2, 2), grams), navigable, vmax, pile=10.0)[0, 0]
        for grams in (0.2, 0.6, 1.2, 2.0)
    ]
    alphas = [float(shade[3]) for shade in shades]
    brightness = [float(shade[:3].sum()) for shade in shades]

    assert alphas == pytest.approx([alphas[0]] * 4), "alpha saturates, as it must"
    assert brightness == sorted(brightness, reverse=True), "but the colour keeps going"
    assert brightness[0] > brightness[-1] * 1.5


def test_below_the_scale_nothing_about_the_shading_changed():
    """The darkening must not touch ordinary dirt, only heaps."""
    from zimablue.replay.renderer import _dirt_alpha

    navigable = np.ones((2, 2), dtype=bool)
    plain = _dirt_alpha(np.full((2, 2), 0.1), navigable, 0.2, pile=10.0)
    assert plain[0, 0, :3] == pytest.approx(_hex_rgb(PALETTE["dirt"]))
    assert float(plain[0, 0, 3]) == pytest.approx(0.5 * 0.85)


def test_the_scale_is_told_how_far_the_run_goes(autumn_recording):
    """Calibrating on frame zero alone is what left the heap flat."""
    from zimablue.replay.renderer import pile_range

    field = autumn_recording.dirt_at(0.0)
    vmax = float(np.percentile(field[field > 0], 92))
    heaviest = float(autumn_recording.dirt_keyframes.sum(axis=1, dtype=np.float32).max())
    assert pile_range(autumn_recording, vmax) == pytest.approx(min(max(heaviest / vmax, 2.0), 64.0))
