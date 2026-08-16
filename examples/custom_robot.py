#!/usr/bin/env python3
"""Build a cleaner from components, and break one of its sensors on purpose.

    python examples/custom_robot.py
    python examples/custom_robot.py --minutes 5

Two things worth noticing:

* No ZimaBlue file was edited to add this robot. A cleaner is a composition of
  components, so a new design is a call, not a patch.
* The brush is what removes adhered dirt. Turn it down and coverage barely
  moves while cleaning collapses -- which is the distinction the whole library
  is built around.
* Surface material changes how much the brush matters, in a way that is not
  obvious until you measure it.
* Sensor faults barely affect the baseline controller -- because the baseline
  barely uses its sensors. Worth knowing before trusting it.
"""

from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=20.0)
    args = parser.parse_args()

    # Dirt dominated by adhered growth. A mixed preset would bury the effect:
    # most of its mass is loose sediment that comes up under suction whatever
    # the brush is doing, so the totals barely move.
    adhered = zb.DirtSpec(
        name="growth",
        layers=(
            zb.LayerSpec("algae", grams_per_m2=45.0, patterns=("patchy",)),
            zb.LayerSpec("biofilm", grams_per_m2=20.0, patterns=("edges",)),
        ),
    )

    print("Same pool, same dirt, same controller. Only the brush differs.\n")
    print(f"{'brush':>10s}  {'coverage':>9s}  {'algae':>7s}  {'biofilm':>8s}")

    for aggressiveness, label in ((1.4, "strong"), (1.0, "normal"), (0.15, "worn out")):
        robot = build_cleaner(brush_aggressiveness=aggressiveness, name=f"custom_{label}")
        metrics = (
            zb.Simulation(pool="rectangular", robot=robot, dirt=adhered, seed=42, record=False)
            .run(minutes=args.minutes)
            .metrics
        )
        print(
            f"{label:>10s}  {metrics.coverage:8.0%}  "
            f"{_removed(metrics, 'algae'):6.1%}  {_removed(metrics, 'biofilm'):7.1%}"
        )

    print(
        "\nCoverage is identical -- the robot drives the same route either way.\n"
        "Only the cleaning changes, and only for the dirt that is bonded down."
    )

    # Surface matters too, and the model says something non-obvious about it.
    print("\nThe same brush on different surfaces (algae removed):")
    print(f"{'surface':>12s}  {'strong':>8s}  {'worn out':>9s}")
    for surface in ("concrete", "plaster", "vinyl", "tile"):
        row = []
        for aggressiveness in (1.4, 0.15):
            pool = zb.make_pool("rectangular")
            pool.material = zb.get_material(surface)
            robot = build_cleaner(brush_aggressiveness=aggressiveness, name="x")
            metrics = (
                zb.Simulation(pool=pool, robot=robot, dirt=adhered, seed=42, record=False)
                .run(minutes=args.minutes)
                .metrics
            )
            row.append(_removed(metrics, "algae"))
        print(f"{surface:>12s}  {row[0]:7.1%}  {row[1]:8.1%}")

    print(
        "\nOn rough concrete the bond is strongest, so the brush matters most.\n"
        "On smooth tile more of the growth is loose to begin with and suction\n"
        "alone gets further -- which is why cleaner makers match brush material\n"
        "to surface."
    )

    # Finally: break a sensor part-way through. The result here is not the one
    # you might expect, and it is worth reporting rather than hiding.
    print("\nInjecting sensor faults five minutes in:")
    print(f"{'sensor':>12s}  {'coverage':>9s}  {'collisions':>11s}")
    faults = (
        ("none", None),
        ("gyro drift", lambda r: r.sensors.imu.inject_fault(bias=0.03, start_time=300.0)),
        ("sonar stuck", lambda r: r.sensors.sonar.inject_fault(stuck=True, start_time=300.0)),
        ("sonar long", lambda r: r.sensors.sonar.inject_fault(bias=0.9, start_time=300.0)),
    )
    for label, inject in faults:
        robot = build_cleaner(brush_aggressiveness=1.0, name=label)
        if inject is not None:
            inject(robot)
        metrics = (
            zb.Simulation(pool="l_shaped", robot=robot, dirt=adhered, seed=42, record=False)
            .run(minutes=args.minutes)
            .metrics
        )
        print(f"{label:>12s}  {metrics.coverage:8.0%}  {metrics.collisions:11d}")

    print(
        "\nBarely a difference -- and that is the finding. The baseline controller\n"
        "is a reflex agent: it steers on bump switches and only glances at the\n"
        "sonar, so corrupting the sonar changes little and a drifting gyro just\n"
        "reshuffles which parts of the pool it misses. A controller that built a\n"
        "map, or fused these sensors into a pose estimate, would degrade sharply\n"
        "here. That is the point of having the fault machinery before having the\n"
        "algorithm that depends on it."
    )


def _removed(metrics: zb.Metrics, dirt: str) -> float:
    """Fraction of one dirt type removed over the run."""
    gone = metrics.removed_by_type.get(dirt, 0.0)
    left = metrics.dirt_by_type.get(dirt, 0.0)
    return gone / max(gone + left, 1e-9)


if __name__ == "__main__":
    main()
