#!/usr/bin/env python3
"""Write your own autonomy stack and benchmark it against the shipped ones.

    python examples/custom_controller.py
    python examples/custom_controller.py --minutes 5

A controller is one class with two methods. It sees sensor readings only --
never ground-truth pose -- which is what keeps the coverage numbers honest.
"""

from __future__ import annotations

import argparse

import numpy as np

import zimablue as zb
from zimablue.controllers import ControlInput
from zimablue.geometry import wrap_angle


class SpiralOut:
    """Drive an expanding spiral; on contact, turn away and start a new one.

    Not a good coverage strategy -- but a complete, working one in 30 lines,
    which is the point of the exercise.
    """

    name = "spiral_out"

    def __init__(self, growth: float = 0.02) -> None:
        self.growth = growth

    def reset(self, robot: zb.Cleaner) -> None:
        self.curvature = 1.6
        self.heading = 0.0
        self.avoid_until = 0.0
        self._last_time = 0.0

    def step(self, ctl: ControlInput) -> zb.DriveCommand:
        top = ctl.robot.locomotion.max_speed
        if ctl.battery <= ctl.robot.power.battery.cutoff:
            return zb.DriveCommand.stop()

        # Integrate the gyro for heading. It drifts; that is realistic.
        imu = ctl.reading("imu")
        dt = max(ctl.time - self._last_time, 0.0)
        self._last_time = ctl.time
        if imu is not None and imu.valid:
            self.heading = float(wrap_angle(self.heading + imu[2] * dt))

        contact = ctl.reading("contact")
        bumped = bool(contact is not None and contact.valid and np.any(contact.values > 0.5))
        if bumped or ctl.extras.get("stuck", 0.0) > 0.5:
            self.avoid_until = ctl.time + 1.8
            self.curvature = 1.6  # restart the spiral somewhere new

        if ctl.time < self.avoid_until:
            return zb.DriveCommand(left=-top * 0.5, right=top * 0.5)

        # Unwind the spiral: curvature decays, so the radius grows.
        self.curvature = max(self.curvature - self.growth * dt, 0.0)
        return zb.DriveCommand.from_body(top * 0.8, self.curvature, ctl.robot.locomotion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=20.0)
    args = parser.parse_args()

    print(f"{args.minutes:g} simulated minutes in a kidney pool, seed 42.\n")
    print(f"{'controller':>18s}  {'coverage':>9s}  {'dirt':>6s}  {'revisits':>9s}")

    contenders = [
        ("random_bounce", "random_bounce", False),
        ("baseline_coverage", "baseline_coverage", False),
        ("spiral_out (ours)", SpiralOut(), False),
        ("lawnmower_oracle", "lawnmower_oracle", True),
    ]
    for label, controller, truth in contenders:
        result = zb.Simulation(
            pool="kidney",
            dirt="light_sediment",
            controller=controller,
            seed=42,
            record=False,
            expose_truth=truth,
        ).run(minutes=args.minutes)
        m = result.metrics
        print(f"{label:>18s}  {m.coverage:8.0%}  {m.dirt_removed_fraction:5.0%}  {m.revisits:9.2f}")

    print(
        "\nlawnmower_oracle drives from ground truth, so it is an upper bound "
        "rather than a competitor."
    )


if __name__ == "__main__":
    main()
