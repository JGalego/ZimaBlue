"""The determinism contract.

Same ZimaBlue version + same platform + same scenario + same seed
=> bit-identical recording.

This is the promise the whole testbed rests on, so it is tested at every layer
that could break it, not just end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.dirt import make_dirt
from zimablue.pool import make_pool
from zimablue.rng import RngTree
from zimablue.scenarios import load_scenario
from zimablue.simulation import Simulation


def _frames(seed: int, seconds: float = 45.0, **kwargs):
    sim = Simulation(pool="kidney", dirt="autumn", seed=seed, **kwargs)
    return sim.run(seconds=seconds).recording.frames


def test_named_streams_are_stable_and_independent():
    tree = RngTree(42)
    assert RngTree(42).stream("dirt").random() == tree.stream("dirt").random()
    # A stream's values depend only on the root seed and its own name, so
    # asking for a new stream cannot shift an existing one.
    fresh = RngTree(42)
    fresh.stream("slip")
    fresh.stream("sensor:imu")
    assert fresh.fresh("dirt").random() == RngTree(42).stream("dirt").random()


def test_different_stream_names_diverge():
    tree = RngTree(7)
    assert tree.stream("a").random() != tree.stream("b").random()


def test_rng_rejects_bad_seeds():
    with pytest.raises(TypeError):
        RngTree("42")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RngTree(-1)


def test_dirt_generation_is_reproducible():
    pool = make_pool("kidney")
    spec = make_dirt("autumn")
    first = spec.build(pool, RngTree(42).stream("dirt"))
    second = spec.build(pool, RngTree(42).stream("dirt"))
    for name, layer in first.field.layers.items():
        assert np.array_equal(layer, second.field.layers[name])
    assert np.array_equal(first.debris.x, second.debris.x)
    assert np.array_equal(first.debris.mass, second.debris.mass)


def test_same_seed_gives_bit_identical_frames():
    a, b = _frames(42), _frames(42)
    assert set(a) == set(b)
    for channel in a:
        assert np.array_equal(a[channel], b[channel], equal_nan=True), channel


def test_different_seeds_diverge():
    a, c = _frames(42), _frames(43)
    assert not np.array_equal(a["x"], c["x"])


def test_metrics_are_reproducible():
    def metrics(seed):
        return (
            Simulation(pool="kidney", dirt="autumn", seed=seed, record=False)
            .run(seconds=45)
            .metrics.to_dict()
        )

    first, second = metrics(5), metrics(5)
    for key, value in first.items():
        assert second[key] == value, key


def test_scenario_runs_are_reproducible(tmp_path):
    scenario = load_scenario("scenarios/kidney.yaml")
    scenario.duration = 45.0
    first = scenario.run(seed=1).recording.frames
    second = scenario.run(seed=1).recording.frames
    for channel in first:
        assert np.array_equal(first[channel], second[channel], equal_nan=True), channel


def test_recording_survives_a_save_load_cycle_unchanged(tmp_path):
    recording = Simulation(pool="rectangular", seed=3).run(seconds=30).recording
    path = recording.save(tmp_path / "run.zbr")
    reloaded = type(recording).load(path)
    for channel, values in recording.frames.items():
        assert np.array_equal(values, reloaded.frames[channel], equal_nan=True), channel
    assert reloaded.manifest["seed"] == recording.manifest["seed"]
    assert len(reloaded.events) == len(recording.events)


def test_dirt_field_conserves_mass_under_transport():
    pool = make_pool("kidney")
    state = make_dirt("autumn").build(pool, RngTree(1).stream("dirt"))
    before = state.field.total()
    vx, vy = pool.flow_grid()
    for _ in range(5):
        state.field.drift(vx, vy, 1.0)
    assert state.field.total() == pytest.approx(before, rel=1e-9)
