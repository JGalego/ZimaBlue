"""The dirt raster, at its edges.

The simulator drives DirtField a few hundred thousand times a run and
reaches the common paths by brute force. What it does not reach is the
arithmetic's boundaries: an empty field, a layer added twice, a scrub that
lifts nothing, debris with no buoyant items in it. Those are guards, and a
guard nothing tests is a guard nobody notices removing.
"""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.dirt.field import DebrisSet, DirtField, DirtState
from zimablue.geometry import Grid

CELL = 0.5


@pytest.fixture
def grid() -> Grid:
    return Grid.covering((0.0, 0.0, 4.0, 4.0), cell=CELL)


@pytest.fixture
def field(grid) -> DirtField:
    return DirtField(grid, mask=np.ones(grid.shape, dtype=bool))


# ----------------------------------------------------------------------
def test_an_empty_field_still_answers_every_question(field):
    """Nothing is dirty yet; the shapes still have to be right.

    Every consumer indexes the grid unconditionally, so an empty field that
    returns a zero-length array instead of a zero-filled one is a crash in
    the renderer rather than a blank pool.
    """
    assert field.total() == 0.0
    assert field.total_grid().shape == field.grid.shape
    assert not field.total_grid().any()
    assert field.concentration().shape == field.grid.shape
    assert field.snapshot().shape == (0, *field.grid.shape)
    assert field.layer_names() == []
    assert field.type_specs() == []


def test_concentration_is_mass_per_square_metre_not_per_cell(field, grid):
    """A resolution-independent view: halving the cell must not double it."""
    values = np.zeros(grid.shape)
    values[0, 0] = 10.0
    field.add_layer("sediment", values)
    assert field.concentration()[0, 0] == pytest.approx(10.0 / grid.cell_area)


def test_adding_the_same_layer_twice_accumulates_rather_than_replaces(field, grid):
    values = np.full(grid.shape, 2.0)
    field.add_layer("sediment", values)
    field.add_layer("sediment", values)
    assert field.total() == pytest.approx(2 * 2.0 * grid.shape[0] * grid.shape[1])
    assert field.layer_names() == ["sediment"]


def test_a_layer_the_wrong_shape_is_refused_with_both_shapes(field):
    with pytest.raises(ValueError, match="expected"):
        field.add_layer("sediment", np.ones((2, 2)))


def test_dirt_outside_the_pool_is_dropped_not_stored(grid):
    """The mask is the navigable water; mass outside it can never be cleaned
    and would sit in the denominator of every coverage figure forever."""
    mask = np.zeros(grid.shape, dtype=bool)
    mask[0, 0] = True
    field = DirtField(grid, mask=mask)
    field.add_layer("sediment", np.full(grid.shape, 1.0))
    assert field.total() == pytest.approx(1.0)


def test_negative_dirt_is_clamped_rather_than_subtracted(field, grid):
    values = np.full(grid.shape, -5.0)
    field.add_layer("sediment", values)
    assert field.total() == 0.0


# ----------------------------------------------------------------------
def test_a_source_creates_the_layer_it_will_deposit_into(field):
    """Attaching a source before any dirt exists must not KeyError later."""
    field.attach_source("sediment", np.full(field.grid.shape, 0.1))
    assert field.layer_names() == ["sediment"]
    assert field.total() == 0.0

    deposited = field.deposit(10.0)
    assert deposited > 0
    assert field.total() == pytest.approx(deposited)
    assert field.deposited_total == pytest.approx(deposited)


def test_two_sources_of_one_type_add_up(field):
    rate = np.full(field.grid.shape, 0.1)
    field.attach_source("sediment", rate)
    field.attach_source("sediment", rate)
    assert field.deposit(1.0) == pytest.approx(2 * 0.1 * field.grid.nrows * field.grid.ncols)


def test_deposition_outside_the_pool_is_refused(grid):
    mask = np.zeros(grid.shape, dtype=bool)
    field = DirtField(grid, mask=mask)
    field.attach_source("sediment", np.full(grid.shape, 1.0))
    assert field.deposit(60.0) == 0.0


# ----------------------------------------------------------------------
def test_density_off_the_grid_is_zero_not_an_error(field, grid):
    field.add_layer("sediment", np.full(grid.shape, 1.0))
    assert field.density_at(-100.0, -100.0, 0.3) == 0.0


def test_density_reads_the_dirt_under_the_head(field, grid):
    values = np.zeros(grid.shape)
    values[2, 2] = 8.0
    field.add_layer("sediment", values)
    xs, ys = grid.cell_centers()
    assert field.density_at(float(xs[2, 2]), float(ys[2, 2]), CELL) > 0.0


def test_scrubbing_a_clean_patch_removes_nothing(field, grid):
    """The zero-total guard: without it this divides by the mass it did not
    find."""
    field.add_layer("sediment", np.zeros(grid.shape))
    window = grid.window(2.0, 2.0, 0.5)
    assert field.remove_window(window, {"sediment": 1.0}) == {}


def test_scrubbing_takes_the_fraction_it_was_asked_for(field, grid):
    field.add_layer("sediment", np.full(grid.shape, 10.0))
    before = field.total()
    window = grid.window(2.0, 2.0, 0.6)
    taken = field.remove_window(window, {"sediment": 0.5})
    assert sum(taken.values()) == pytest.approx(before - field.total())
    assert 0 < sum(taken.values()) < before


def test_disturbing_a_clean_patch_lifts_nothing(field, grid):
    field.add_layer("sediment", np.zeros(grid.shape))
    window = grid.window(2.0, 2.0, 0.5)
    field.disturb_window(window, strength=1.0)
    assert field.total() == 0.0


def test_disturbance_moves_dirt_without_creating_or_destroying_it(field, grid):
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sand", values)
    before = field.total()
    field.disturb_window(grid.window(2.0, 2.0, 1.0), strength=1.0)
    assert field.total() == pytest.approx(before, rel=1e-9), "a wake is not a drain"


def test_no_strength_is_no_disturbance(field, grid):
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sand", values)
    field.disturb_window(grid.window(2.0, 2.0, 1.0), strength=0.0)
    assert field.total_grid()[4, 4] == pytest.approx(100.0)


def test_drift_over_no_time_changes_nothing(field, grid):
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sediment", values)
    flow = np.full(grid.shape, 0.2)
    field.drift(flow, flow, dt=0.0)
    assert field.total_grid()[4, 4] == pytest.approx(100.0)


def test_drift_in_still_water_moves_nothing(field, grid):
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sediment", values)
    still = np.zeros(grid.shape)
    field.drift(still, still, dt=1.0)
    assert field.total_grid()[4, 4] == pytest.approx(100.0)


def test_drift_conserves_mass_rather_than_flushing_it_out_of_the_pool(field, grid):
    """Mass that would leave goes back to the sender; rolling would wrap it
    round to the far wall instead."""
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sediment", values)
    flow = np.full(grid.shape, 0.5)
    for _ in range(20):
        field.drift(flow, flow, dt=0.5)
    assert field.total() == pytest.approx(100.0, rel=1e-9)


def test_sand_is_too_heavy_to_drift(field, grid):
    """Only the fraction light enough to resuspend moves with the water."""
    values = np.zeros(grid.shape)
    values[4, 4] = 100.0
    field.add_layer("sand", values)
    flow = np.full(grid.shape, 0.5)
    field.drift(flow, flow, dt=1.0)
    assert field.total_grid()[4, 4] == pytest.approx(100.0)


def test_an_unstable_spread_is_refused_rather_than_diverging(field, grid):
    field.add_layer("sediment", np.full(grid.shape, 1.0))
    flow = np.full(grid.shape, 0.2)
    with pytest.raises(ValueError, match=r"\[0, 0.25\)"):
        field.drift(flow, flow, dt=1.0, spread=0.3)


# ----------------------------------------------------------------------
def test_a_field_with_no_dirt_at_all_reads_as_entirely_removed(grid):
    """Nothing to clean is not a failure to clean; the fraction is a ratio
    whose denominator is zero."""
    field = DirtField(grid, mask=np.ones(grid.shape, dtype=bool))
    state = DirtState(field, DebrisSet())
    assert state.budget_mass == 0.0
    assert state.removed_fraction == 1.0
    assert state.removed_mass == 0.0


def test_a_debris_set_with_no_items_is_still_answerable():
    debris = DebrisSet()
    assert len(debris) == 0
    assert debris.remaining_mass == 0.0
    assert debris.collected_count == 0
    assert debris.buoyant.shape == (0,)
    assert debris.type_names() == []
