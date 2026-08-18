"""The cleaner: chassis, locomotion, cleaning head, power and sensors."""

from __future__ import annotations

from zimablue.robot.cleaner import Cleaner
from zimablue.robot.command import DriveCommand
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
from zimablue.robot.design import DESIGNS, CleanerDesign, Part, make_design
from zimablue.robot.presets import ROBOT_PRESETS, make_robot

__all__ = [
    "DESIGNS",
    "ROBOT_PRESETS",
    "Battery",
    "Brush",
    "Chassis",
    "Cleaner",
    "CleanerDesign",
    "CleaningSystem",
    "DriveCommand",
    "DriveUnit",
    "Filter",
    "Locomotion",
    "Motor",
    "Part",
    "PowerSystem",
    "Pump",
    "make_design",
    "make_robot",
]
