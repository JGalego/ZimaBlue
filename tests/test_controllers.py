"""Controllers, including the rule that they never see ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from zimablue.controllers import BaselineCoverage, LawnmowerOracle, Phase, RandomBounce
from zimablue.controllers.base import CONTROLLERS
from zimablue.simulation import Simulation


def test_every_registered_controller_drives_the_robot():
    for name in CONTROLLERS.names():
        sim = Simulation(
            pool="rectangular",
            controller=name,
            seed=1,
            record=False,
            expose_truth=(name == "lawnmower_oracle"),
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


def test_oracle_is_listed_but_flagged_in_its_docstring():
    assert "not a legitimate controller" in (
        LawnmowerOracle.__doc__ or ""
    ).lower() or "not a legitimate controller" in (LawnmowerOracle.__doc__ or "")
