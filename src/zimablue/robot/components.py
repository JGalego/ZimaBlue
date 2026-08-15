"""Cleaner components.

Every component is a frozen dataclass of physical parameters plus, where it
earns its keep, a small amount of behaviour (a motor clamps a command; a filter
reports how clogged it is).  They hold no simulation state -- that lives in the
backend's ``SimState`` -- which is what lets one ``Cleaner`` description be run
many times, in parallel, deterministically.

Numbers are plausible mid-range residential-cleaner figures, not a specific
product's specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Battery",
    "Brush",
    "Chassis",
    "CleaningSystem",
    "DriveUnit",
    "Filter",
    "Locomotion",
    "Motor",
    "PowerSystem",
    "Pump",
]


@dataclass(frozen=True)
class Chassis:
    """Hull dimensions and mass."""

    length: float = 0.42
    width: float = 0.38
    height: float = 0.26
    mass: float = 9.5
    """Dry mass, kg."""

    displacement: float = 0.0075
    """Displaced volume, m^3. Near-neutral buoyancy is what keeps a cleaner
    tractable on walls; this feeds the effective normal load."""

    def __post_init__(self) -> None:
        for name in ("length", "width", "height", "mass", "displacement"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Chassis.{name} must be positive")

    @property
    def radius(self) -> float:
        """Circumscribed radius used for broad-phase collision, m."""
        return 0.5 * float(np.hypot(self.length, self.width))

    @property
    def submerged_weight(self) -> float:
        """Apparent weight underwater, N. Drives traction.

        Buoyancy cancels most of the dry weight, which is why a pool cleaner
        needs either high displacement compensation or suction to grip a wall.
        """
        water_density = 997.0
        gravity = 9.80665
        return max(0.5, (self.mass - water_density * self.displacement) * gravity)

    def to_dict(self) -> dict[str, float]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "mass": self.mass,
            "displacement": self.displacement,
        }


@dataclass(frozen=True)
class Motor:
    """A drive motor, described at the track surface rather than at the shaft.

    Working in linear units removes a gear-ratio conversion from the hot loop
    without losing anything the 2D backend can observe.
    """

    max_speed: float = 0.32
    """No-load track speed, m/s."""

    max_accel: float = 0.9
    """Peak acceleration of the track surface, m/s^2."""

    efficiency: float = 0.72
    """Electrical to mechanical conversion, 0-1."""

    idle_power: float = 0.6
    """Draw when energised but not moving, W."""

    stall_force: float = 45.0
    """Peak tractive force one side can deliver, N."""

    def clamp_speed(self, requested: float) -> float:
        return float(np.clip(requested, -self.max_speed, self.max_speed))

    def apply_limits(self, current: float, requested: float, dt: float) -> float:
        """Slew the current speed toward ``requested`` within acceleration limits."""
        target = self.clamp_speed(requested)
        max_delta = self.max_accel * dt
        return float(current + np.clip(target - current, -max_delta, max_delta))

    def power_draw(self, speed: float, load_force: float) -> float:
        """Electrical power for a given track speed and tractive load, W."""
        mechanical = abs(speed) * abs(load_force)
        return self.idle_power + mechanical / max(self.efficiency, 1e-3)

    def to_dict(self) -> dict[str, float]:
        return {
            "max_speed": self.max_speed,
            "max_accel": self.max_accel,
            "efficiency": self.efficiency,
            "idle_power": self.idle_power,
            "stall_force": self.stall_force,
        }


@dataclass(frozen=True)
class DriveUnit:
    """One side of the drivetrain: a motor plus its wheel or track."""

    name: str = "drive"
    motor: Motor = Motor()
    wheel_radius: float = 0.055
    tracked: bool = True
    """Tracks put more rubber down than wheels: better grip, worse turning."""

    contact_length: float = 0.22
    """Length of the ground contact patch, m. Longer resists turning."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "motor": self.motor.to_dict(),
            "wheel_radius": self.wheel_radius,
            "tracked": self.tracked,
            "contact_length": self.contact_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriveUnit:
        return cls(
            name=data["name"],
            motor=Motor(**data["motor"]),
            wheel_radius=float(data["wheel_radius"]),
            tracked=bool(data["tracked"]),
            contact_length=float(data["contact_length"]),
        )


@dataclass(frozen=True)
class Locomotion:
    """A differential drive: two independent sides plus the contact model."""

    left: DriveUnit = DriveUnit(name="left")
    right: DriveUnit = DriveUnit(name="right")
    track_width: float = 0.33
    """Lateral distance between the two contact patches, m."""

    traction: float = 0.9
    """Grip multiplier applied on top of the pool surface's own friction."""

    turn_resistance: float = 0.35
    """0-1. How much a long contact patch resists yaw, causing turn slip."""

    def __post_init__(self) -> None:
        if self.track_width <= 0:
            raise ValueError("Locomotion.track_width must be positive")
        if not 0.0 <= self.turn_resistance < 1.0:
            raise ValueError("Locomotion.turn_resistance must be in [0, 1)")

    @property
    def max_speed(self) -> float:
        return min(self.left.motor.max_speed, self.right.motor.max_speed)

    def to_body_velocity(self, v_left: float, v_right: float) -> tuple[float, float]:
        """Differential-drive forward kinematics: ``(v, omega)``."""
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / self.track_width
        return v, omega

    def to_wheel_speeds(self, v: float, omega: float) -> tuple[float, float]:
        """Inverse kinematics: body ``(v, omega)`` to left/right track speeds."""
        half = 0.5 * omega * self.track_width
        return v - half, v + half

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "track_width": self.track_width,
            "traction": self.traction,
            "turn_resistance": self.turn_resistance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Locomotion:
        return cls(
            left=DriveUnit.from_dict(data["left"]),
            right=DriveUnit.from_dict(data["right"]),
            track_width=float(data["track_width"]),
            traction=float(data["traction"]),
            turn_resistance=float(data["turn_resistance"]),
        )


@dataclass(frozen=True)
class Brush:
    """A rotating agitation brush.

    Only agitation: the brush loosens adhered dirt, it does not collect it.
    Collection is the pump's job, which is why a brush-less robot can drive over
    algae all day and remove almost none of it.
    """

    width: float = 0.34
    """Swept width, m."""

    rpm: float = 90.0
    aggressiveness: float = 1.0
    """Scales how much adhered dirt one pass releases."""

    power: float = 12.0
    """Electrical draw when running, W."""

    def agitation(self, speed: float) -> float:
        """Dimensionless agitation strength at a given travel speed.

        Brushing depends on brush rotation, not travel: driving faster spreads
        the same agitation over more area, so effectiveness per unit area falls.
        """
        rotation = self.rpm / 90.0
        pace = 1.0 / (1.0 + max(speed, 0.0) / 0.25)
        return self.aggressiveness * rotation * pace

    def to_dict(self) -> dict[str, float]:
        return {
            "width": self.width,
            "rpm": self.rpm,
            "aggressiveness": self.aggressiveness,
            "power": self.power,
        }


@dataclass(frozen=True)
class Pump:
    """The impeller that draws water and debris through the intake."""

    flow_rate: float = 4.5
    """Volumetric flow, L/s."""

    power: float = 45.0
    """Draw at full duty, W."""

    intake_width: float = 0.30
    """Width of the intake mouth, m."""

    max_debris_size: float = 0.09
    """Largest debris the intake will swallow, m. Bigger items are pushed
    around instead, and repeatedly bumping one is a real failure mode."""

    def suction(self, duty: float, clog: float) -> float:
        """Effective suction, 0-1, given pump duty and filter clog fraction.

        Suction falls off with clogging rather than stopping abruptly: a
        half-full filter still works, a full one barely does.
        """
        duty = float(np.clip(duty, 0.0, 1.0))
        return duty * float(np.clip(1.0 - clog**2, 0.0, 1.0))

    def to_dict(self) -> dict[str, float]:
        return {
            "flow_rate": self.flow_rate,
            "power": self.power,
            "intake_width": self.intake_width,
            "max_debris_size": self.max_debris_size,
        }


@dataclass(frozen=True)
class Filter:
    """The basket or cartridge that retains what the pump collects."""

    capacity: float = 900.0
    """Retained mass before the filter is full, grams."""

    mesh: float = 60e-6
    """Mesh aperture, m. Particles finer than this pass straight through --
    which is why a cleaner can run all day and leave the water hazy."""

    def retains(self, particle_size: float) -> float:
        """Fraction of particles of a given size the mesh retains.

        A soft transition around the mesh size rather than a hard cut-off:
        real media capture some fraction of nominally-passing particles.
        """
        ratio = particle_size / self.mesh
        return float(np.clip(ratio**2 / (1.0 + ratio**2), 0.0, 1.0))

    def clog_fraction(self, load: float) -> float:
        return float(np.clip(load / self.capacity, 0.0, 1.0))

    def to_dict(self) -> dict[str, float]:
        return {"capacity": self.capacity, "mesh": self.mesh}


@dataclass(frozen=True)
class CleaningSystem:
    """Brush, intake, pump and filter as one subsystem."""

    brush: Brush = Brush()
    pump: Pump = Pump()
    filter: Filter = Filter()
    roller: Brush | None = None
    """Optional second (rear) brush."""

    @property
    def swath_width(self) -> float:
        """Width actually cleaned per pass, m.

        The wider of brush and intake: the brush loosens over its own width and
        the pump collects over its own, so a pass affects the union.
        """
        widths = [self.brush.width, self.pump.intake_width]
        if self.roller is not None:
            widths.append(self.roller.width)
        return max(widths)

    @property
    def total_agitation_power(self) -> float:
        return self.brush.power + (self.roller.power if self.roller else 0.0)

    def agitation(self, speed: float) -> float:
        total = self.brush.agitation(speed)
        if self.roller is not None:
            total += 0.6 * self.roller.agitation(speed)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "brush": self.brush.to_dict(),
            "pump": self.pump.to_dict(),
            "filter": self.filter.to_dict(),
            "roller": self.roller.to_dict() if self.roller else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleaningSystem:
        return cls(
            brush=Brush(**data["brush"]),
            pump=Pump(**data["pump"]),
            filter=Filter(**data["filter"]),
            roller=Brush(**data["roller"]) if data.get("roller") else None,
        )


@dataclass(frozen=True)
class Battery:
    """Energy store."""

    capacity_wh: float = 120.0
    voltage: float = 24.0
    initial_charge: float = 1.0
    """State of charge at t=0, 0-1."""

    cutoff: float = 0.05
    """State of charge at which the robot stops, 0-1."""

    def __post_init__(self) -> None:
        if self.capacity_wh <= 0:
            raise ValueError("Battery.capacity_wh must be positive")
        if not 0.0 < self.initial_charge <= 1.0:
            raise ValueError("Battery.initial_charge must be in (0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {
            "capacity_wh": self.capacity_wh,
            "voltage": self.voltage,
            "initial_charge": self.initial_charge,
            "cutoff": self.cutoff,
        }


@dataclass(frozen=True)
class PowerSystem:
    """Battery plus the loads that are not the drive or cleaning subsystems."""

    battery: Battery = Battery()
    electronics_power: float = 2.5
    """Always-on draw of the controller and sensors, W."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery": self.battery.to_dict(),
            "electronics_power": self.electronics_power,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PowerSystem:
        return cls(
            battery=Battery(**data["battery"]),
            electronics_power=float(data["electronics_power"]),
        )
