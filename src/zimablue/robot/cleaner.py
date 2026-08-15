"""The :class:`Cleaner` -- a composition of components, not a subclass tree.

A user builds a robot the way the README shows::

    robot = zb.Cleaner(
        chassis=zb.Chassis(length=0.5, mass=12.0),
        locomotion=zb.Locomotion(track_width=0.4),
        sensors=[zb.IMU(), zb.Encoder(), zb.Sonar(beam_angles=(0.0, 0.5))],
    )

Nothing in ZimaBlue needs editing to support a new robot: the simulator only
ever asks the cleaner for its components' parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from zimablue.robot.components import (
    Chassis,
    CleaningSystem,
    Locomotion,
    PowerSystem,
)
from zimablue.sensors.base import Sensor
from zimablue.sensors.suite import SensorSuite

__all__ = ["Cleaner"]


class Cleaner:
    """A robotic pool cleaner."""

    def __init__(
        self,
        *,
        name: str = "cleaner",
        chassis: Chassis | None = None,
        locomotion: Locomotion | None = None,
        cleaning: CleaningSystem | None = None,
        power: PowerSystem | None = None,
        sensors: list[Sensor] | tuple[Sensor, ...] | SensorSuite | None = None,
    ) -> None:
        self.name = name
        self.chassis = chassis if chassis is not None else Chassis()
        self.locomotion = locomotion if locomotion is not None else Locomotion()
        self.cleaning = cleaning if cleaning is not None else CleaningSystem()
        self.power = power if power is not None else PowerSystem()
        if isinstance(sensors, SensorSuite):
            self.sensors = sensors
        else:
            self.sensors = SensorSuite(list(sensors) if sensors else [])

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------
    @property
    def radius(self) -> float:
        """Collision radius, m."""
        return self.chassis.radius

    @property
    def swath_width(self) -> float:
        """Width cleaned per pass, m."""
        return self.cleaning.swath_width

    def footprint(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> Polygon:
        """Hull outline at a pose, for rendering and area queries."""
        half_l = self.chassis.length / 2
        half_w = self.chassis.width / 2
        base = shapely_box(-half_l, -half_w, half_l, half_w)
        return translate(rotate(base, heading, origin=(0, 0), use_radians=True), x, y)

    def cleaning_footprint(self, x: float, y: float, heading: float) -> tuple[float, float, float]:
        """Centre and radius of the cleaned disc for this pose.

        A disc rather than a rectangle: at 10 cm raster resolution the
        difference is under one cell, and a disc costs one distance comparison
        instead of a polygon test on every cell, every tick.
        """
        return (x, y, 0.5 * self.swath_width)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def power_draw(
        self,
        speed_left: float,
        speed_right: float,
        load_left: float,
        load_right: float,
        *,
        brush_on: bool,
        pump_duty: float,
    ) -> float:
        """Total instantaneous electrical draw, W."""
        motors = self.locomotion.left.motor.power_draw(
            speed_left, load_left
        ) + self.locomotion.right.motor.power_draw(speed_right, load_right)
        brush = self.cleaning.total_agitation_power if brush_on else 0.0
        pump = self.cleaning.pump.power * float(np.clip(pump_duty, 0.0, 1.0))
        return motors + brush + pump + self.power.electronics_power

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chassis": self.chassis.to_dict(),
            "locomotion": self.locomotion.to_dict(),
            "cleaning": self.cleaning.to_dict(),
            "power": self.power.to_dict(),
            "sensors": self.sensors.specs(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cleaner:
        return cls(
            name=data.get("name", "cleaner"),
            chassis=Chassis(**data["chassis"]),
            locomotion=Locomotion.from_dict(data["locomotion"]),
            cleaning=CleaningSystem.from_dict(data["cleaning"]),
            power=PowerSystem.from_dict(data["power"]),
            sensors=SensorSuite.from_specs(data.get("sensors", [])),
        )

    def describe(self) -> str:
        """A short human-readable summary, used by the CLI."""
        return (
            f"{self.name}: {self.chassis.length * 100:.0f}x{self.chassis.width * 100:.0f} cm, "
            f"{self.chassis.mass:.1f} kg, {self.locomotion.max_speed:.2f} m/s, "
            f"swath {self.swath_width * 100:.0f} cm, "
            f"battery {self.power.battery.capacity_wh:.0f} Wh, "
            f"sensors: {', '.join(self.sensors) or 'none'}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Cleaner(name={self.name!r}, sensors={list(self.sensors)})"
