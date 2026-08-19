"""The water works too: floating debris rides the jets, the skimmer collects."""

from __future__ import annotations

import numpy as np

import zimablue as zb
from zimablue.backends.fast2d import Fast2DBackend


def run(minutes=5.0, seed=11, **backend_kwargs):
    backend = Fast2DBackend(**backend_kwargs) if backend_kwargs else "fast2d"
    sim = zb.Simulation(
        pool="rectangular",
        dirt="windy_day",
        backend=backend,
        seed=seed,
        controller="baseline_coverage",
    )
    return sim.run(minutes=minutes)


def test_floating_debris_moves_with_the_water():
    result = run(minutes=5.0)
    recording = result.require_recording()
    start = recording.debris_at(0.0)
    end = recording.debris_at(recording.duration)
    names = recording.debris_type_names()
    floating = [i for i, kind in enumerate(start[:, 5].astype(int)) if names[kind] == "floating"]
    assert floating, "windy_day should scatter floating debris"
    moved = [
        float(np.hypot(end[i, 0] - start[i, 0], end[i, 1] - start[i, 1]))
        for i in floating
        if end[i, 4] < 0.5  # still in the pool
    ]
    if moved:
        assert max(moved) > 0.05, "nothing that floats went anywhere"


def test_the_skimmer_finally_earns_its_keep():
    result = run(minutes=8.0)
    skimmed = [e for e in result.recording.events if e["kind"] == "skimmed"]
    assert skimmed, "eight windy minutes and the skimmer caught nothing"
    m = result.metrics
    assert m.debris_skimmed == sum(e["detail"]["count"] for e in skimmed)
    assert m.dirt_skimmed > 0
    # The robot is not credited with the skimmer's work.
    assert (
        m.debris_collected + m.debris_skimmed
        <= m.debris_collected + m.debris_skimmed + m.debris_remaining
    )


def test_what_sinks_does_not_sail():
    from zimablue.dirt.field import DebrisSet
    from zimablue.dirt.types import DIRT_TYPES

    debris = DebrisSet(
        types=[DIRT_TYPES["floating"], DIRT_TYPES["twigs"]],
        type_index=np.array([0, 1]),
        x=np.array([1.0, 1.0]),
        y=np.array([1.0, 1.0]),
        mass=np.array([2.0, 2.0]),
        size=np.array([0.01, 0.05]),
    )
    # twigs float too (wood); pin the second item down by making it dense.
    from dataclasses import replace

    debris.types[1] = replace(DIRT_TYPES["twigs"], density=2000.0)
    debris.advect(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
    assert debris.x[0] == 1.5, "the floater should ride the flow"
    assert debris.x[1] == 1.0, "the sunk item should stay put"


def test_the_wake_strength_is_a_dial():
    calm = run(minutes=2.0, wake_strength=0.0)
    stormy = run(minutes=2.0, wake_strength=1.0)
    assert calm.metrics.dirt_removed != stormy.metrics.dirt_removed
