"""Pool surface materials.

Material affects three things the simulator cares about: how well tracks grip
(``friction``), how effectively a brush dislodges adhered dirt (``brush_gain``),
and how strongly dirt sticks in the first place (``adhesion_factor``).  Values
are relative, ordered by the qualitative behaviour reported in cleaner
literature (rubber brushes for rough plaster/concrete, soft foam for smooth
vinyl and fiberglass) rather than measured coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MATERIALS", "SurfaceMaterial", "get_material"]


@dataclass(frozen=True)
class SurfaceMaterial:
    """Physical character of a pool surface."""

    name: str
    friction: float
    """Track-to-surface traction coefficient. Higher grips better."""

    brush_gain: float
    """Multiplier on brush agitation effectiveness."""

    adhesion_factor: float
    """Multiplier on how strongly adhered dirt bonds to this surface."""

    roughness: float
    """Surface roughness in arbitrary 0-1 units; drives fine-sediment retention."""

    def __post_init__(self) -> None:
        for field_name in ("friction", "brush_gain", "adhesion_factor", "roughness"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{self.name}.{field_name} must be positive, got {value}")


MATERIALS: dict[str, SurfaceMaterial] = {
    m.name: m
    for m in (
        # Rough, porous; grips well, holds sediment, algae bonds hard.
        SurfaceMaterial(
            "plaster", friction=0.85, brush_gain=1.0, adhesion_factor=1.0, roughness=0.6
        ),
        SurfaceMaterial(
            "concrete", friction=0.95, brush_gain=1.1, adhesion_factor=1.25, roughness=0.8
        ),
        # Smooth; poor grip (slip matters), dirt releases easily.
        SurfaceMaterial("tile", friction=0.55, brush_gain=0.85, adhesion_factor=0.6, roughness=0.2),
        SurfaceMaterial(
            "vinyl", friction=0.65, brush_gain=0.75, adhesion_factor=0.7, roughness=0.3
        ),
        SurfaceMaterial(
            "fiberglass", friction=0.6, brush_gain=0.8, adhesion_factor=0.55, roughness=0.15
        ),
    )
}


def get_material(name: str | SurfaceMaterial) -> SurfaceMaterial:
    """Look a material up by name, or pass a ``SurfaceMaterial`` straight through."""
    if isinstance(name, SurfaceMaterial):
        return name
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown pool material {name!r}; available: {sorted(MATERIALS)}") from None
