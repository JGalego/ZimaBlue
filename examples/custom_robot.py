#!/usr/bin/env python3
"""Build a cleaner from components, and break one of its sensors on purpose.

    python examples/custom_robot.py

Two things worth noticing:

* No ZimaBlue file was edited to add this robot. A cleaner is a composition of
  components, so a new design is a call, not a patch.
* The brush is what removes adhered dirt. Turn it down and coverage barely
  moves while cleaning collapses -- which is the distinction the whole library
  is built around.
"""

from __future__ import annotations

import numpy as np

import zimablue as zb


def build_cleaner(*, brush_aggressiveness: float, name: str) -> zb.Cleaner:
    """A custom mid-size tracked cleaner with a configurable brush."""
    motor = zb.Motor(max_speed=0.30, max_accel=1.0, efficiency=0.7)
    return zb.Cleaner(
        name=name,
        chassis=zb.Chassis(length=0.45, width=0.40, mass=10.5, displacement=0.0082),
        locomotion=zb.Locomotion(
            left=zb.DriveUnit(name="left", motor=motor, tracked=True),
            right=zb.DriveUnit(name="right", motor=motor, tracked=True),
            track_width=0.35,
            traction=0.92,
        ),
        cleaning=zb.CleaningSystem(
            brush=zb.Brush(width=0.38, rpm=100.0, aggressiveness=brush_aggressiveness),
            pump=zb.Pump(flow_rate=5.0, power=48.0, intake_width=0.32),
            filter=zb.Filter(capacity=1200.0, mesh=45e-6),
        ),
        power=zb.PowerSystem(battery=zb.Battery(capacity_wh=140.0)),
        sensors=[
            zb.Encoder(),
            zb.IMU(),
            zb.PressureSensor(max_depth=3.0),
            zb.ContactSensor(),
            zb.Sonar(beam_angles=(0.0, np.deg2rad(45), np.deg2rad(-45)), max_range=3.5),
        ],
    )


def main() -> None:
    print("Same pool, same dirt, same controller. Only the brush differs.\n")
    print(f"{'brush':>10s}  {'coverage':>9s}  {'dirt removed':>13s}")

    for aggressiveness, label in ((1.4, "strong"), (1.0, "normal"), (0.15, "worn out")):
        robot = build_cleaner(brush_aggressiveness=aggressiveness, name=f"custom_{label}")
        result = zb.Simulation(
            pool="l_shaped", robot=robot, dirt="neglected_pool", seed=42, record=False
        ).run(minutes=20)
        print(
            f"{label:>10s}  {result.metrics.coverage:8.0%}  "
            f"{result.metrics.dirt_removed_fraction:12.0%}"
        )

    # Now break a sensor part-way through, and watch the navigation suffer.
    print("\nSame robot, but the sonar starts lying five minutes in:")
    robot = build_cleaner(brush_aggressiveness=1.0, name="faulty")
    robot.sensors.sonar.inject_fault(
        bias=0.9,                  # reports walls almost a metre further away
        dropout_probability=0.15,
        start_time=300.0,
        label="sonar_drift",
    )
    result = zb.Simulation(
        pool="l_shaped", robot=robot, dirt="neglected_pool", seed=42, record=False
    ).run(minutes=20)
    print(
        f"{'faulty':>10s}  {result.metrics.coverage:8.0%}  "
        f"{result.metrics.dirt_removed_fraction:12.0%}   "
        f"collisions {result.metrics.collisions}"
    )


if __name__ == "__main__":
    main()
