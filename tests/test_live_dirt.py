"""A pool that keeps getting dirtier while the robot works."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.dirt import DirtSpec
from zimablue.dirt.generators import LayerSpec


def test_deposition_grows_the_field_and_the_denominator():
    from zimablue.dirt.field import DirtField
    from zimablue.geometry import Grid

    grid = Grid(minx=0.0, miny=0.0, nrows=10, ncols=10, cell=0.1)
    mask = np.ones((10, 10), dtype=bool)
    field = DirtField(grid, mask)
    field.add_layer("sediment", np.full((10, 10), 1.0))
    field.freeze_initial()
    field.attach_source("sediment", np.full((10, 10), 0.01))

    added = field.deposit(10.0)
    assert added == pytest.approx(10.0)
    assert field.deposited_total == pytest.approx(10.0)
    assert field.total() == pytest.approx(100.0 + 10.0)


def test_a_live_pool_cannot_score_above_one():
    from zimablue.dirt.field import DirtField, DirtState
    from zimablue.geometry import Grid

    grid = Grid(minx=0.0, miny=0.0, nrows=4, ncols=4, cell=0.1)
    mask = np.ones((4, 4), dtype=bool)
    field = DirtField(grid, mask)
    field.add_layer("sediment", np.full((4, 4), 1.0))
    field.freeze_initial()
    field.attach_source("sediment", np.full((4, 4), 0.1))
    state = DirtState(field)

    field.deposit(5.0)
    # Remove everything currently there: with a frozen denominator this would
    # read as more than 100% removed.
    for layer in field.layers.values():
        layer[:] = 0.0
    assert state.removed_fraction == pytest.approx(1.0)


def test_the_spec_round_trips_its_live_parts():
    spec = DirtSpec(
        name="live",
        layers=(LayerSpec("sediment", grams_per_m2=2.0, grams_per_m2_per_hour=6.0),),
        stir_interval=30.0,
        stir_strength=0.4,
    )
    rebuilt = DirtSpec.from_dict(spec.to_dict())
    assert rebuilt.layers[0].grams_per_m2_per_hour == 6.0
    assert rebuilt.stir_interval == 30.0
    assert rebuilt.stir_strength == 0.4


def test_pool_party_rains_and_stirs():
    result = zb.Simulation(pool="rectangular", dirt="pool_party", seed=8).run(minutes=3)
    m = result.metrics
    assert m.dirt_deposited > 0, "three party minutes should deposit something"
    assert 0.0 <= m.dirt_removed_fraction <= 1.0
    # Deposition is part of the recorded story: the field at the end holds
    # mass that was not there at the start, in cells the robot never visited.
    assert m.dirt_deposited == pytest.approx(3 / 60 * 14.0 * 50.0, rel=0.15)


def test_the_party_is_reproducible():
    def run():
        return zb.Simulation(pool="rectangular", dirt="pool_party", seed=8).run(minutes=2)

    first, second = run(), run()
    assert first.metrics.dirt_removed == second.metrics.dirt_removed
    assert first.metrics.dirt_deposited == second.metrics.dirt_deposited


def test_stirring_moves_dirt_without_creating_any():
    def run(interval):
        spec = DirtSpec(
            name="stir_test",
            layers=(LayerSpec("sediment", grams_per_m2=8.0, patterns=("patchy",)),),
            stir_interval=interval,
            stir_strength=0.8,
        )
        return zb.Simulation(pool="rectangular", dirt=spec, seed=6).run(minutes=2)

    calm = run(0.0)
    stirred = run(20.0)
    calm_grid = calm.spatial.remaining_dirt
    stirred_grid = stirred.spatial.remaining_dirt
    assert not np.allclose(calm_grid, stirred_grid), "six stirs should leave a mark"
    # Stirring redistributes; it does not create or destroy.
    assert stirred.metrics.dirt_deposited == 0.0
