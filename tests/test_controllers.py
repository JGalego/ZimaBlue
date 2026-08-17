"""Controllers, including the rule that they never see ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.controllers import BaselineCoverage, DirtOracle, LawnmowerOracle, Phase, RandomBounce
from zimablue.controllers.base import CONTROLLERS
from zimablue.simulation import Simulation

ORACLES = ("lawnmower_oracle", "dirt_oracle")
"""The controllers that read ground truth, and so are bounds rather than
controllers. Everything else has to work from sensors alone."""


def test_every_registered_controller_drives_the_robot():
    for name in CONTROLLERS.names():
        sim = Simulation(
            pool="rectangular",
            controller=name,
            seed=1,
            record=False,
            expose_truth=name in ORACLES,
        )
        result = sim.run(seconds=60)
        assert result.metrics.distance_traveled > 1.0, f"{name} barely moved"


def test_controllers_do_not_receive_ground_truth_by_default():
    """The property that keeps coverage numbers meaningful."""
    seen = []

    class Spy:
        name = "spy"

        def reset(self, robot):
            pass

        def step(self, ctl):
            from zimablue.robot import DriveCommand

            seen.append(ctl.truth)
            return DriveCommand(0.2, 0.2)

    Simulation(pool="rectangular", controller=Spy(), seed=1, record=False).run(seconds=5)
    assert seen and all(t is None for t in seen)


def test_oracle_refuses_to_run_without_ground_truth():
    sim = Simulation(
        pool="rectangular",
        controller="lawnmower_oracle",
        seed=1,
        record=False,
        expose_truth=False,
    )
    with pytest.raises(RuntimeError, match="expose_truth"):
        sim.run(seconds=5)


def test_oracle_beats_the_baseline_on_coverage():
    """The oracle is the upper bound; if it is not ahead, something is wrong."""

    def coverage(controller, truth):
        return (
            Simulation(
                pool="rectangular", controller=controller, seed=3, record=False, expose_truth=truth
            )
            .run(minutes=10)
            .metrics.coverage
        )

    assert coverage("lawnmower_oracle", True) > coverage("baseline_coverage", False)


def test_baseline_reaches_its_lane_phase():
    controller = BaselineCoverage()
    sim = Simulation(pool="rectangular", controller=controller, seed=1, record=False)
    sim.run(minutes=8)
    assert controller.phase is not Phase.PERIMETER, "should leave the perimeter pass"


def test_baseline_recovers_rather_than_grinding():
    """A wedged robot must back out, not spin against the wall forever."""
    controller = BaselineCoverage()
    result = Simulation(pool="l_shaped", controller=controller, seed=9, record=False).run(minutes=8)
    assert result.metrics.distance_traveled > 40.0


def test_heading_estimator_integrates_and_drifts():
    from zimablue.controllers.baseline import HeadingEstimator

    estimator = HeadingEstimator()
    for i in range(100):
        estimator.update(i * 0.01, 1.0)
    assert estimator.heading == pytest.approx(0.99, abs=0.02)


def test_random_bounce_is_seeded():
    def run(seed):
        return (
            Simulation(pool="rectangular", controller=RandomBounce(seed=seed), seed=1, record=False)
            .run(seconds=60)
            .metrics.coverage
        )

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_a_custom_controller_needs_no_zimablue_changes():
    """The extension point: satisfy the protocol, pass the instance."""
    from zimablue.robot import DriveCommand

    class Spiral:
        name = "spiral"

        def reset(self, robot):
            self.t = 0.0

        def step(self, ctl):
            self.t += ctl.dt
            top = ctl.robot.locomotion.max_speed
            return DriveCommand(left=top * 0.6, right=top * (0.3 + 0.3 * np.sin(self.t / 8)))

    result = Simulation(pool="oval", controller=Spiral(), seed=2, record=False).run(seconds=120)
    assert result.metrics.coverage > 0.0
    assert result.metrics.distance_traveled > 5.0


def test_the_dirt_oracle_refuses_to_run_without_ground_truth():
    sim = Simulation(pool="rectangular", controller="dirt_oracle", seed=1, record=False)
    with pytest.raises(RuntimeError, match="expose_truth"):
        sim.run(seconds=5)


def test_the_two_oracles_disagree_about_which_run_was_better():
    """The whole thesis, as an assertion.

    One oracle is optimal at driving and the other is greedy about cleaning.
    If they ranked the same way, coverage would be a proxy for cleanliness and
    there would be nothing here worth simulating.

    Eight minutes, deliberately: greedy is myopic and a systematic sweep
    catches it somewhere past twenty. That crossover is documented in
    ``DirtOracle`` rather than asserted here, because pinning it would cost
    four half-hour runs on every CI job.
    """

    def run(controller):
        return Simulation(
            pool="kidney",
            dirt="autumn",
            controller=controller,
            seed=42,
            record=False,
            expose_truth=True,
        ).run(minutes=8)

    lawnmower = run("lawnmower_oracle").metrics
    greedy = run("dirt_oracle").metrics

    assert lawnmower.coverage > greedy.coverage, "the lawnmower should cover more"
    assert greedy.dirt_removed_fraction > lawnmower.dirt_removed_fraction, (
        "and the greedy one should clean more"
    )


def test_the_dirt_oracle_goes_where_the_dirt_is():
    """It should end up in the dirtiest part of the pool, not the middle."""
    sim = Simulation(
        pool="rectangular",
        dirt="autumn",
        controller="dirt_oracle",
        seed=7,
        record=False,
        expose_truth=True,
    )
    start = sim.world.dirt.field.total_grid().copy()
    grid = sim.world.dirt.field.grid
    xs, ys = grid.cell_centers()
    row, col = np.unravel_index(int(np.argmax(start)), start.shape)
    dirtiest = (float(xs[row, col]), float(ys[row, col]))

    result = sim.run(minutes=6)
    removed = start - result.world.dirt.field.total_grid()
    # Most of what it took came from near where the dirt was thickest.
    row, col = np.unravel_index(int(np.argmax(removed)), removed.shape)
    worked = (float(xs[row, col]), float(ys[row, col]))
    assert np.hypot(worked[0] - dirtiest[0], worked[1] - dirtiest[1]) < 3.0


def test_the_travel_cost_trades_driving_for_dirt():
    def run(travel_cost):
        return (
            Simulation(
                pool="kidney",
                dirt="autumn",
                controller=DirtOracle(travel_cost=travel_cost),
                seed=42,
                record=False,
                expose_truth=True,
            )
            .run(minutes=8)
            .metrics
        )

    greedy = run(0.0)
    thrifty = run(1.0)
    assert thrifty.distance_traveled < greedy.distance_traveled


def test_oracle_is_listed_but_flagged_in_its_docstring():
    assert "not a legitimate controller" in (
        LawnmowerOracle.__doc__ or ""
    ).lower() or "not a legitimate controller" in (LawnmowerOracle.__doc__ or "")


def test_the_dirt_oracle_stops_when_the_pool_is_clean():
    """No dirt left, nowhere to go. It should park, not drive at nothing."""
    from zimablue.robot import DriveCommand

    sim = Simulation(
        pool="rectangular", controller="dirt_oracle", seed=1, record=False, expose_truth=True
    )
    for layer in sim.world.dirt.field.layers:
        sim.world.dirt.field.layers[layer][:] = 0.0

    sim.step()
    assert sim._last_command == DriveCommand.stop()
