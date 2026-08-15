"""Dirt types.

Pool contamination is not one substance, and the differences are what make
"did it clean?" a harder question than "did it drive there?".  A dirt type is a
bundle of physical properties; the cleaning model reads those properties rather
than branching on the type's name, so a user-defined dirt type works everywhere
a built-in one does.

Settling velocity is *derived* from particle size and density rather than tuned
per type, which keeps the presets physically ordered relative to each other.
Flocculation and hindered settling are known effects that are deliberately not
modelled -- see ``docs/research.md`` section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DIRT_TYPES",
    "DirtType",
    "get_dirt_type",
    "settling_velocity",
    "stokes_settling_velocity",
]

WATER_DENSITY = 997.0
WATER_VISCOSITY = 8.9e-4
"""Dynamic viscosity of fresh water near 25 C, Pa*s."""

GRAVITY = 9.80665


def stokes_settling_velocity(
    particle_size: float,
    density: float,
    *,
    fluid_density: float = WATER_DENSITY,
    viscosity: float = WATER_VISCOSITY,
) -> float:
    """Terminal settling velocity in m/s from Stokes' law.

    ``w = g * d^2 * (rho_p - rho_f) / (18 * mu)``

    Positive is sinking.  Valid only in creeping flow (particle Reynolds number
    below ~1), which for water means roughly d < 100 um.  Above that it
    overpredicts badly -- 350 um sand comes out at 124 mm/s against a measured
    ~45 mm/s.  Use :func:`settling_velocity` unless you specifically want the
    Stokes result.
    """
    return float(GRAVITY * particle_size**2 * (density - fluid_density) / (18.0 * viscosity))


def settling_velocity(
    particle_size: float,
    density: float,
    *,
    fluid_density: float = WATER_DENSITY,
    viscosity: float = WATER_VISCOSITY,
    c1: float = 20.0,
    c2: float = 1.1,
) -> float:
    """Terminal settling velocity in m/s, valid across all particle sizes.

    Uses the unified equation of Ferguson & Church (2004), *A simple universal
    equation for grain settling velocity*, J. Sedimentary Research 74(6):

    ``w = R g d^2 / (C1 nu + sqrt(0.75 C2 R g d^3))``

    where ``R`` is submerged specific gravity and ``nu`` is kinematic
    viscosity.  The first denominator term dominates for fine particles, where
    the expression reduces exactly to Stokes' law; the second dominates for
    coarse ones, giving the constant-drag (Newton) regime.  Defaults
    ``C1 = 20, C2 = 1.1`` are the paper's values for natural sand grains
    (smooth spheres would be 18 and 0.4).

    Positive is sinking, negative is rising: buoyant particles are computed
    from ``|R|`` and negated, since the correlation is posed for dense grains.

    Leaf-sized flat debris is outside the correlation's validity -- real leaves
    are plates, not grains, and flutter rather than settle. ZimaBlue tracks
    those as discrete items where the exact rise rate does not drive behaviour.
    """
    kinematic = viscosity / fluid_density
    r = (density - fluid_density) / fluid_density
    magnitude = abs(r)
    if magnitude == 0.0 or particle_size <= 0.0:
        return 0.0
    numerator = magnitude * GRAVITY * particle_size**2
    denominator = c1 * kinematic + np.sqrt(0.75 * c2 * magnitude * GRAVITY * particle_size**3)
    return float(np.sign(r) * numerator / denominator)


@dataclass(frozen=True)
class DirtType:
    """One kind of contamination."""

    name: str

    density: float
    """Bulk particle density, kg/m^3. Below water's ~997 it floats."""

    particle_size: float
    """Characteristic diameter, m."""

    adhesion: float
    """0-1. How strongly it bonds to the surface. Suction alone cannot lift
    high-adhesion dirt; it must be agitated loose first."""

    pickup_difficulty: float = 1.0
    """Multiplier on the effort required to collect it once loose."""

    resuspension: float = 0.2
    """0-1. Tendency to be kicked back into the water by passing flow or by the
    robot itself, then to settle again somewhere else."""

    discrete: bool = False
    """True for item-like debris (leaves, twigs) simulated individually rather
    than as a density field."""

    colour: str = "#8a7452"
    """Used by the replay renderer."""

    _settling_velocity: float | None = field(default=None, repr=False)

    @property
    def settling_velocity(self) -> float:
        """m/s, positive downward. Derived from size and density unless set."""
        if self._settling_velocity is not None:
            return self._settling_velocity
        return settling_velocity(self.particle_size, self.density)

    @property
    def buoyant(self) -> bool:
        return self.density < WATER_DENSITY

    @property
    def adhered(self) -> bool:
        """Whether this type needs mechanical agitation to release."""
        return self.adhesion >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "density": self.density,
            "particle_size": self.particle_size,
            "adhesion": self.adhesion,
            "pickup_difficulty": self.pickup_difficulty,
            "resuspension": self.resuspension,
            "discrete": self.discrete,
            "colour": self.colour,
            "settling_velocity": self.settling_velocity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirtType:
        return cls(
            name=data["name"],
            density=float(data["density"]),
            particle_size=float(data["particle_size"]),
            adhesion=float(data["adhesion"]),
            pickup_difficulty=float(data.get("pickup_difficulty", 1.0)),
            resuspension=float(data.get("resuspension", 0.2)),
            discrete=bool(data.get("discrete", False)),
            colour=data.get("colour", "#8a7452"),
            _settling_velocity=float(data["settling_velocity"])
            if "settling_velocity" in data
            else None,
        )


DIRT_TYPES: dict[str, DirtType] = {
    t.name: t
    for t in (
        # Fine, slow to settle, easily stirred back up, mostly slips the mesh.
        DirtType(
            name="sediment",
            density=1400.0,
            particle_size=20e-6,
            adhesion=0.1,
            pickup_difficulty=1.0,
            resuspension=0.6,
            colour="#9a8b6f",
        ),
        # Dense and fast-settling; ends up in low points and dead corners.
        DirtType(
            name="sand",
            density=2650.0,
            particle_size=350e-6,
            adhesion=0.05,
            pickup_difficulty=1.3,
            resuspension=0.15,
            colour="#c2a878",
        ),
        # Adhered growth: suction does nothing without the brush.
        DirtType(
            name="algae",
            density=1100.0,
            particle_size=15e-6,
            adhesion=0.85,
            pickup_difficulty=1.6,
            resuspension=0.05,
            colour="#5f8f4e",
        ),
        # Biofilm: the worst case -- bonded, slimy, needs repeated passes.
        DirtType(
            name="biofilm",
            density=1050.0,
            particle_size=8e-6,
            adhesion=0.95,
            pickup_difficulty=2.2,
            resuspension=0.02,
            colour="#6b7f5a",
        ),
        # Discrete items.
        DirtType(
            name="leaves",
            density=940.0,
            particle_size=45e-3,
            adhesion=0.0,
            pickup_difficulty=1.1,
            resuspension=0.5,
            discrete=True,
            colour="#a05a2c",
        ),
        DirtType(
            name="twigs",
            density=780.0,
            particle_size=70e-3,
            adhesion=0.0,
            pickup_difficulty=1.8,
            resuspension=0.3,
            discrete=True,
            colour="#6b4423",
        ),
        # Floats on the surface; the skimmer's problem, not the robot's.
        DirtType(
            name="floating",
            density=650.0,
            particle_size=3e-3,
            adhesion=0.0,
            pickup_difficulty=1.0,
            resuspension=0.9,
            discrete=True,
            colour="#d8c98a",
        ),
    )
}


def get_dirt_type(name: str | DirtType) -> DirtType:
    """Look a dirt type up by name, or pass a ``DirtType`` straight through."""
    if isinstance(name, DirtType):
        return name
    try:
        return DIRT_TYPES[name]
    except KeyError:
        raise KeyError(f"unknown dirt type {name!r}; available: {sorted(DIRT_TYPES)}") from None
