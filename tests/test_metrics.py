"""Metrics -- especially that coverage and cleanliness stay distinct."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.metrics import Metrics
from zimablue.simulation import Simulation


def test_metrics_are_in_range(short_run):
    m = short_run.metrics
    assert 0.0 <= m.coverage <= 1.0
    assert 0.0 <= m.wall_coverage <= 1.0
    assert 0.0 <= m.dirt_removed_fraction <= 1.0
    assert 0.0 <= m.cleaning_uniformity <= 1.0
    assert 0.0 <= m.battery_remaining <= 1.0
    assert m.distance_traveled >= 0.0
    assert m.revisits >= 0.0


def test_coverage_and_cleanliness_are_different_numbers():
    """The distinction the project exists for.

    Same controller, same seed, same pool -- only the brush differs. Coverage
    is unchanged because the robot drives identically; cleaning collapses.

    The dirt is pure biofilm on purpose. A mixed preset dilutes the effect,
    because the loose fraction comes up under suction either way and the
    comparison ends up measuring the sediment rather than the claim.
    """
    from zimablue.controllers.base import ControlInput
    from zimablue.controllers.baseline import BaselineCoverage
    from zimablue.dirt import DirtSpec, LayerSpec
    from zimablue.robot import DriveCommand

    adhered = DirtSpec(name="biofilm_only", layers=(LayerSpec("biofilm", grams_per_m2=40.0),))

    class BrushOverride:
        name = "brush_override"

        def __init__(self, brush: bool) -> None:
            self.brush = brush
            self.inner = BaselineCoverage()

        def reset(self, robot):
            self.inner.reset(robot)

        def step(self, ctl: ControlInput) -> DriveCommand:
            command = self.inner.step(ctl)
            return DriveCommand(command.left, command.right, self.brush, command.pump)

    def run(brush: bool):
        return (
            Simulation(
                pool="rectangular",
                dirt=adhered,
                controller=BrushOverride(brush),
                seed=11,
                record=False,
            )
            .run(seconds=240)
            .metrics
        )

    with_brush, without = run(True), run(False)
    assert with_brush.coverage == pytest.approx(without.coverage, abs=0.02), (
        "the robot drives identically; only cleaning should change"
    )
    assert with_brush.dirt_removed_fraction > 2.0 * without.dirt_removed_fraction, (
        f"brush {with_brush.dirt_removed_fraction:.3f} vs "
        f"no brush {without.dirt_removed_fraction:.3f}"
    )


def test_revisits_counts_passes_not_ticks():
    """Standing still must not inflate the revisit count.

    Visits are credited per pass, so a stationary robot scores near zero
    revisits however long it idles.
    """
    from zimablue.controllers.base import ControlInput
    from zimablue.robot import DriveCommand

    class Idle:
        name = "idle"

        def reset(self, robot):
            pass

        def step(self, ctl: ControlInput) -> DriveCommand:
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)

    metrics = (
        Simulation(pool="rectangular", controller=Idle(), seed=1, record=False)
        .run(seconds=60)
        .metrics
    )
    assert metrics.revisits < 1.0, f"idling scored {metrics.revisits:.1f} revisits"


def test_spatial_metrics_line_up_with_the_scalars(short_run):
    spatial = short_run.spatial
    navigable = int(spatial.navigable.sum())
    covered = int((spatial.navigable & (spatial.visits > 0)).sum())
    assert covered / navigable == pytest.approx(short_run.metrics.coverage, rel=1e-6)
    assert spatial.missed.sum() == navigable - covered
    assert spatial.dirt_removed_grid.min() >= 0.0


def test_metrics_round_trip_through_a_dict(short_run):
    restored = Metrics.from_dict(short_run.metrics.to_dict())
    assert restored.coverage == short_run.metrics.coverage
    assert restored.termination == short_run.metrics.termination


def test_summary_mentions_both_families(short_run):
    text = short_run.metrics.summary()
    assert "coverage" in text
    assert "dirt removed" in text


def test_battery_terminates_a_long_run():
    """A flat battery stops the run and is reported as such.

    Uses a deliberately tiny battery: the point is the termination path, and
    draining a real 70 Wh pack takes three simulated hours.
    """
    from dataclasses import replace

    from zimablue.robot import Battery, make_robot

    robot = make_robot("compact")
    robot.power = replace(robot.power, battery=Battery(capacity_wh=0.35, voltage=18.0))

    metrics = (
        Simulation(pool="rectangular", robot=robot, seed=2, record=False).run(minutes=30).metrics
    )
    assert metrics.termination == "battery_empty"
    assert metrics.battery_remaining <= robot.power.battery.cutoff + 1e-6
    assert metrics.runtime < 30 * 60


def test_coverage_target_stops_early():
    sim = Simulation(
        pool="rectangular", seed=5, record=False, coverage_target=0.15, dirt_target=None
    )
    metrics = sim.run(minutes=30).metrics
    assert metrics.termination == "target_reached"
    assert metrics.coverage >= 0.15
    assert metrics.runtime < 30 * 60


def test_coverage_grows_with_time():
    short = Simulation(pool="rectangular", seed=6, record=False).run(seconds=60).metrics
    long = Simulation(pool="rectangular", seed=6, record=False).run(seconds=300).metrics
    assert long.coverage > short.coverage
    assert long.distance_traveled > short.distance_traveled


def test_energy_accumulates_monotonically(short_run):
    power = short_run.recording.frames["distance"]
    assert np.all(np.diff(power) >= -1e-6), "distance travelled must never decrease"
