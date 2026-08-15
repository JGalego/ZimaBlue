"""Standard cleaner configurations.

Three points on the trade-off curve, each with a full sensor suite so that
comparing controllers across robots does not accidentally compare sensor sets.
"""

from __future__ import annotations

import numpy as np

from zimablue.registry import Registry
from zimablue.robot.cleaner import Cleaner
from zimablue.robot.components import (
    Battery,
    Brush,
    Chassis,
    CleaningSystem,
    DriveUnit,
    Filter,
    Locomotion,
    Motor,
    PowerSystem,
    Pump,
)
from zimablue.sensors.models import IMU, ContactSensor, Encoder, PressureSensor, Sonar

__all__ = ["ROBOT_PRESETS", "make_robot"]

ROBOT_PRESETS: Registry[Cleaner] = Registry("robot")


def _standard_sensors(max_depth: float = 5.0, sonar_range: float = 3.0) -> list:
    """The sensor set every preset carries.

    Three sonar beams -- ahead and +/-40 degrees -- is the minimum that lets a
    wall follower tell "wall ahead" from "wall alongside".
    """
    return [
        Encoder(),
        IMU(),
        PressureSensor(max_depth=max_depth),
        ContactSensor(),
        Sonar(beam_angles=(0.0, np.deg2rad(40.0), np.deg2rad(-40.0)), max_range=sonar_range),
    ]


@ROBOT_PRESETS.register("tracked")
def tracked() -> Cleaner:
    """The reference robot: a mid-size tracked differential-drive cleaner.

    Tracks, a 34 cm brush, a 900 g filter and a 120 Wh battery -- enough to
    cover a typical residential pool in one charge, which makes battery
    limits visible in metrics without dominating every run.
    """
    motor = Motor(max_speed=0.32, max_accel=0.9, efficiency=0.72, idle_power=0.6)
    return Cleaner(
        name="tracked",
        chassis=Chassis(length=0.42, width=0.38, height=0.26, mass=9.5, displacement=0.0075),
        locomotion=Locomotion(
            left=DriveUnit(name="left", motor=motor, tracked=True, contact_length=0.22),
            right=DriveUnit(name="right", motor=motor, tracked=True, contact_length=0.22),
            track_width=0.33,
            traction=0.9,
            turn_resistance=0.35,
        ),
        cleaning=CleaningSystem(
            brush=Brush(width=0.34, rpm=90.0, aggressiveness=1.0, power=12.0),
            pump=Pump(flow_rate=4.5, power=45.0, intake_width=0.30, max_debris_size=0.09),
            filter=Filter(capacity=900.0, mesh=60e-6),
        ),
        power=PowerSystem(battery=Battery(capacity_wh=120.0, voltage=24.0), electronics_power=2.5),
        sensors=_standard_sensors(),
    )


@ROBOT_PRESETS.register("compact")
def compact() -> Cleaner:
    """A small, cheap, wheeled unit: quick and nimble, weak and short-legged.

    Narrow swath and a small filter, so it fills up and needs many more passes.
    Useful as the "this should score worse" control in benchmarks.
    """
    motor = Motor(max_speed=0.26, max_accel=1.2, efficiency=0.65, idle_power=0.4, stall_force=28.0)
    return Cleaner(
        name="compact",
        chassis=Chassis(length=0.33, width=0.30, height=0.22, mass=6.0, displacement=0.0048),
        locomotion=Locomotion(
            left=DriveUnit(name="left", motor=motor, tracked=False, contact_length=0.10),
            right=DriveUnit(name="right", motor=motor, tracked=False, contact_length=0.10),
            track_width=0.26,
            traction=0.7,
            turn_resistance=0.15,
        ),
        cleaning=CleaningSystem(
            brush=Brush(width=0.24, rpm=70.0, aggressiveness=0.75, power=7.0),
            pump=Pump(flow_rate=2.8, power=28.0, intake_width=0.20, max_debris_size=0.06),
            filter=Filter(capacity=400.0, mesh=90e-6),
        ),
        power=PowerSystem(battery=Battery(capacity_wh=70.0, voltage=18.0), electronics_power=1.8),
        sensors=_standard_sensors(max_depth=3.0, sonar_range=2.0),
    )


@ROBOT_PRESETS.register("heavy_duty")
def heavy_duty() -> Cleaner:
    """A large twin-brush commercial unit: wide, strong, thirsty.

    Two brushes and a fine 40 um filter, so it removes far more of the fine
    fraction -- the clearest demonstration that coverage and cleanliness are
    different measurements.
    """
    motor = Motor(max_speed=0.36, max_accel=0.7, efficiency=0.78, idle_power=1.0, stall_force=80.0)
    return Cleaner(
        name="heavy_duty",
        chassis=Chassis(length=0.55, width=0.48, height=0.32, mass=16.0, displacement=0.0125),
        locomotion=Locomotion(
            left=DriveUnit(name="left", motor=motor, tracked=True, contact_length=0.30),
            right=DriveUnit(name="right", motor=motor, tracked=True, contact_length=0.30),
            track_width=0.42,
            traction=1.0,
            turn_resistance=0.5,
        ),
        cleaning=CleaningSystem(
            brush=Brush(width=0.46, rpm=110.0, aggressiveness=1.3, power=18.0),
            roller=Brush(width=0.46, rpm=110.0, aggressiveness=1.1, power=16.0),
            pump=Pump(flow_rate=7.0, power=70.0, intake_width=0.42, max_debris_size=0.13),
            filter=Filter(capacity=2000.0, mesh=40e-6),
        ),
        power=PowerSystem(battery=Battery(capacity_wh=220.0, voltage=36.0), electronics_power=3.5),
        sensors=_standard_sensors(max_depth=6.0, sonar_range=4.0),
    )


def make_robot(name: str, **kwargs: object) -> Cleaner:
    """Build a robot preset by name.

    >>> make_robot("tracked").name
    'tracked'
    """
    return ROBOT_PRESETS.create(name, **kwargs)
