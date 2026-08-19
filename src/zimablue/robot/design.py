"""What a cleaner *looks* like, as distinct from what it does.

Until now a robot was drawn as a rounded rectangle with a line for a nose,
which is fine for reading a trajectory and useless for recognising a machine.
Real cleaners have shapes you can tell apart across a car park: a low wide
scrubber with a full-width roller, a domed suction unit with a single central
intake, a squat commercial box with a brush at each corner.  Those differences
are the first thing anybody notices and the last thing a simulator usually
models.

A :class:`CleanerDesign` is a silhouette and a handful of parts, in normalised
body coordinates.  What follows from that:

**It is purely cosmetic.** Physics reads :class:`~zimablue.robot.Chassis`, and
only ``Chassis``. Collision uses the hull rectangle, cleaning uses the swath
width, traction uses the mass. Swapping a design changes every rendered pixel
and not one number in the metrics -- which is the point, because a drawing that
silently changed the answers would be a trap. If you want a robot that behaves
differently, change its components.

**It is size-independent.** Coordinates run ``-0.5..0.5`` along each axis and
are scaled by the chassis at draw time, so any design fits any robot. A domed
shape on a heavy-duty chassis is a big domed machine, not a small one floating
in a large bounding box.

The presets are archetypes rather than particular products. Named by form --
``domed``, ``flat_scrubber``, ``quad_brush`` -- because the form is the useful
abstraction and because a library has no business shipping traced outlines of
somebody's industrial design. To match a specific machine, measure it and build
a design; that is what :class:`CleanerDesign` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.registry import Registry

__all__ = ["DESIGNS", "CleanerDesign", "Part", "make_design"]

FloatArray = NDArray[np.float64]

# The default livery. Four steps far enough apart to read at the size a robot
# occupies in a top-down pool -- about forty pixels. The first attempt used
# four near-identical dark blues, which was tasteful and rendered every design
# as the same dark blob with one cyan mark on it.
HULL = "#2a3c52"
"""Main body."""

TRIM = "#4a6b8e"
"""Raised structure: domes, decks, housings."""

DARK = "#0d151f"
"""Recesses and things that touch the floor: tracks, intakes, skirts."""

ACCENT = "#3ddcff"
"""The parts that move. One bright colour, used sparingly, so that at a glance
you can see which end the brushes are on."""


@dataclass(frozen=True)
class Part:
    """One drawn piece, in normalised body coordinates.

    ``outline`` is an ``(n, 2)`` array with x forward and y to port, each in
    ``-0.5..0.5``. ``lift`` is the part's height above the floor as a fraction
    of the chassis height, used by the chase camera to give the machine some
    thickness; the top-down view ignores it.
    """

    outline: FloatArray
    colour: str = TRIM
    z: int = 0
    """Draw order within the design. Higher is drawn later, so on top."""

    lift: float = 0.0
    alpha: float = 1.0
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "outline": np.asarray(self.outline, dtype=float).round(4).tolist(),
            "colour": self.colour,
            "z": self.z,
            "lift": self.lift,
            "alpha": self.alpha,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Part:
        return cls(
            outline=np.asarray(data["outline"], dtype=float),
            colour=data.get("colour", TRIM),
            z=int(data.get("z", 0)),
            lift=float(data.get("lift", 0.0)),
            alpha=float(data.get("alpha", 1.0)),
            name=data.get("name", ""),
        )


@dataclass(frozen=True)
class CleanerDesign:
    """A cleaner's silhouette and its visible parts."""

    name: str
    body: FloatArray
    """Hull outline, normalised. The thing you see from above."""

    parts: tuple[Part, ...] = ()
    hull: str = HULL
    trim: str = TRIM
    accent: str = ACCENT
    dome: float = 0.55
    """How tall the machine reads in the chase camera, as a fraction of the
    chassis height. Not used by any physics -- ``Chassis.height`` is the real
    number, and this is how much of it the silhouette is extruded by."""

    description: str = ""

    def __post_init__(self) -> None:
        body = np.asarray(self.body, dtype=float)
        if body.ndim != 2 or body.shape[1] != 2 or len(body) < 3:
            raise ValueError(f"design {self.name!r}: body must be an (n, 2) array with n >= 3")
        if np.abs(body).max() > 1.0:
            raise ValueError(
                f"design {self.name!r}: body coordinates run -0.5..0.5 and are scaled by the "
                f"chassis at draw time; got a maximum of {np.abs(body).max():.2f}"
            )
        object.__setattr__(self, "body", body)

    # ------------------------------------------------------------------
    def scaled(self, length: float, width: float) -> FloatArray:
        """The hull outline in metres, in the robot's own frame."""
        return self.body * np.array([length, width])

    def place(
        self, outline: FloatArray, length: float, width: float, x: float, y: float, heading: float
    ) -> FloatArray:
        """A normalised outline, in metres, at a world pose."""
        points = np.asarray(outline, dtype=float) * np.array([length, width])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return points @ rotation.T + np.array([x, y])

    def drawable(self) -> list[Part]:
        """Body plus parts, in draw order, as one list.

        The hull is a part like any other once it is time to draw; keeping it a
        separate field is for the callers that need the silhouette alone -- a
        shadow, a footprint, a chase camera's occlusion test.
        """
        hull = Part(self.body, colour=self.hull, z=-100, lift=0.0, name="hull")
        return [hull, *sorted(self.parts, key=lambda p: p.z)]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": np.asarray(self.body, dtype=float).round(4).tolist(),
            "parts": [part.to_dict() for part in self.parts],
            "hull": self.hull,
            "trim": self.trim,
            "accent": self.accent,
            "dome": self.dome,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleanerDesign:
        return cls(
            name=data.get("name", "custom"),
            body=np.asarray(data["body"], dtype=float),
            parts=tuple(Part.from_dict(p) for p in data.get("parts", [])),
            hull=data.get("hull", HULL),
            trim=data.get("trim", TRIM),
            accent=data.get("accent", ACCENT),
            dome=float(data.get("dome", 0.55)),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CleanerDesign(name={self.name!r}, parts={len(self.parts)})"


# ----------------------------------------------------------------------
# Shape helpers
#
# Building the outlines from a few primitives rather than pasting coordinate
# tables: a wall of numbers is unreadable and nobody would ever edit one.
# ----------------------------------------------------------------------
def rounded_rect(
    half_length: float = 0.5, half_width: float = 0.5, radius: float = 0.12, steps: int = 6
) -> FloatArray:
    """A rectangle with rounded corners, counter-clockwise from the nose."""
    radius = min(radius, half_length, half_width)
    corners = [
        (half_length - radius, half_width - radius, 0.0),
        (-(half_length - radius), half_width - radius, np.pi / 2),
        (-(half_length - radius), -(half_width - radius), np.pi),
        (half_length - radius, -(half_width - radius), 3 * np.pi / 2),
    ]
    points: list[tuple[float, float]] = []
    for cx, cy, start in corners:
        angles = np.linspace(start, start + np.pi / 2, steps)
        points.extend(zip(cx + radius * np.cos(angles), cy + radius * np.sin(angles), strict=True))
    return np.asarray(points, dtype=np.float64)


def ellipse(half_length: float = 0.5, half_width: float = 0.5, steps: int = 28) -> FloatArray:
    angles = np.linspace(0.0, 2 * np.pi, steps, endpoint=False)
    return np.column_stack([half_length * np.cos(angles), half_width * np.sin(angles)])


def teardrop(
    half_length: float = 0.5, half_width: float = 0.5, blunt: float = 0.72, steps: int = 30
) -> FloatArray:
    """Round at the nose, squarer at the tail.

    The plan view of most tracked cleaners: the leading edge is curved so it
    rides up on things and the trailing edge is flat because that is where the
    cable and the filter go.
    """
    # Walk the nose arc from starboard round to port, then the tail back down
    # the port side. Getting the winding wrong here is not a subtle bug -- the
    # outline crosses itself and the hull renders as a bow tie.
    angles = np.linspace(-np.pi / 2, np.pi / 2, steps)
    nose = np.column_stack([half_length * np.cos(angles), half_width * np.sin(angles)])
    tail = np.array(
        [
            [-half_length * blunt, half_width * 0.92],
            [-half_length, half_width * 0.62],
            [-half_length, -half_width * 0.62],
            [-half_length * blunt, -half_width * 0.92],
        ]
    )
    return np.vstack([nose, tail])


def bar(x: float, half_width: float, thickness: float) -> FloatArray:
    """A lateral bar at ``x`` -- a brush roller, an intake slot, a bumper."""
    return np.array(
        [
            [x - thickness / 2, -half_width],
            [x + thickness / 2, -half_width],
            [x + thickness / 2, half_width],
            [x - thickness / 2, half_width],
        ]
    )


def pad(x: float, y: float, length: float, width: float) -> FloatArray:
    """A rectangular patch -- a track, a wheel, a hatch."""
    return np.array(
        [
            [x - length / 2, y - width / 2],
            [x + length / 2, y - width / 2],
            [x + length / 2, y + width / 2],
            [x - length / 2, y + width / 2],
        ]
    )


def disc(x: float, y: float, radius: float, steps: int = 16) -> FloatArray:
    angles = np.linspace(0.0, 2 * np.pi, steps, endpoint=False)
    return np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
DESIGNS: Registry[CleanerDesign] = Registry("design", entry_point_group="zimablue.designs")


def _tracks(inset: float = 0.44, length: float = 0.86, width: float = 0.11) -> list[Part]:
    return [
        Part(pad(0.0, side * inset, length, width), colour=DARK, z=5, name=f"track_{i}")
        for i, side in enumerate((1, -1))
    ]


@DESIGNS.register("tracked")
def tracked() -> CleanerDesign:
    """The default. Tracks down both flanks, one brush roller at the nose."""
    return CleanerDesign(
        name="tracked",
        body=teardrop(),
        parts=(
            *_tracks(),
            Part(bar(0.38, 0.34, 0.08), colour=ACCENT, z=10, lift=0.05, name="brush"),
            Part(rounded_rect(0.26, 0.30, radius=0.08), colour=TRIM, z=6, lift=0.9, name="top"),
            Part(bar(-0.30, 0.24, 0.12), colour=DARK, z=8, lift=0.45, name="filter"),
        ),
        dome=0.6,
        description="Tracked hull with a leading brush roller. The shipped default.",
    )


@DESIGNS.register("compact")
def compact() -> CleanerDesign:
    """Small and oval, with a single central intake. Residential above-ground."""
    return CleanerDesign(
        name="compact",
        body=ellipse(0.5, 0.5),
        parts=(
            Part(ellipse(0.34, 0.36), colour=TRIM, z=4, lift=0.85, name="shell"),
            Part(disc(0.0, 0.0, 0.19), colour=DARK, z=6, name="intake"),
            Part(disc(0.0, 0.0, 0.10), colour=ACCENT, z=8, lift=0.1, name="impeller"),
            Part(bar(0.32, 0.34, 0.07), colour=DARK, z=5, lift=0.05, name="scuff"),
        ),
        dome=0.75,
        description="Oval hull, one central intake. The small residential shape.",
    )


@DESIGNS.register("heavy_duty")
def heavy_duty() -> CleanerDesign:
    """Big, square and slab-sided, with twin rollers and a lifting handle."""
    return CleanerDesign(
        name="heavy_duty",
        body=rounded_rect(0.5, 0.5, radius=0.10),
        parts=(
            *_tracks(inset=0.42, length=0.92, width=0.15),
            Part(bar(0.42, 0.40, 0.08), colour=ACCENT, z=10, lift=0.05, name="brush_front"),
            Part(bar(-0.42, 0.40, 0.08), colour=ACCENT, z=10, lift=0.05, name="brush_rear"),
            Part(pad(0.0, 0.0, 0.46, 0.60), colour=TRIM, z=6, lift=0.8, name="body_top"),
            Part(bar(0.0, 0.16, 0.05), colour=DARK, z=12, lift=1.0, name="handle"),
        ),
        dome=0.9,
        description="Square commercial hull, brushes fore and aft, a carry handle.",
    )


@DESIGNS.register("domed")
def domed() -> CleanerDesign:
    """A rounded shell with a tall dome. The commonest consumer silhouette."""
    return CleanerDesign(
        name="domed",
        body=ellipse(0.5, 0.46),
        parts=(
            *_tracks(inset=0.38, length=0.7, width=0.09),
            Part(bar(0.34, 0.26, 0.10), colour=ACCENT, z=6, name="intake_slot"),
            Part(ellipse(0.36, 0.36), colour=TRIM, z=8, lift=0.7, name="dome"),
            Part(ellipse(0.22, 0.22), colour=HULL, z=9, lift=0.95, name="dome_top"),
            Part(ellipse(0.09, 0.09), colour=DARK, z=10, lift=1.0, name="port"),
        ),
        dome=1.0,
        description="Rounded shell under a tall dome, wide intake slot at the front.",
    )


@DESIGNS.register("flat_scrubber")
def flat_scrubber() -> CleanerDesign:
    """Low and wide, with a full-width roller. Built for floors, not walls."""
    return CleanerDesign(
        name="flat_scrubber",
        body=rounded_rect(0.42, 0.5, radius=0.16),
        parts=(
            Part(bar(0.28, 0.46, 0.14), colour=ACCENT, z=10, lift=0.05, name="roller"),
            Part(bar(-0.30, 0.44, 0.08), colour=DARK, z=8, lift=0.05, name="squeegee"),
            Part(rounded_rect(0.20, 0.34, radius=0.06), colour=TRIM, z=6, lift=0.5, name="deck"),
        ),
        dome=0.3,
        description="Low wide deck with a full-width roller. Floors only.",
    )


@DESIGNS.register("quad_brush")
def quad_brush() -> CleanerDesign:
    """Four corner brushes on a squat body. The commercial-pool workhorse."""
    corners = [
        Part(disc(sx * 0.34, sy * 0.36, 0.15), colour=ACCENT, z=4, alpha=0.9, name=f"brush_{i}")
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]
    return CleanerDesign(
        name="quad_brush",
        body=rounded_rect(0.44, 0.44, radius=0.14),
        parts=(
            *corners,
            Part(disc(0.0, 0.0, 0.24), colour=DARK, z=8, name="intake"),
            Part(rounded_rect(0.22, 0.22, radius=0.07), colour=TRIM, z=10, lift=0.8, name="top"),
        ),
        dome=0.65,
        description="Squat body with a rotating brush at each corner.",
    )


@DESIGNS.register("suction_disc")
def suction_disc() -> CleanerDesign:
    """A hockey puck with a hose. No motor of its own, no tracks, no brushes."""
    return CleanerDesign(
        name="suction_disc",
        body=ellipse(0.5, 0.5, steps=36),
        parts=(
            Part(disc(0.0, 0.0, 0.34), colour=DARK, z=4, name="skirt"),
            Part(disc(0.0, 0.0, 0.20), colour=TRIM, z=6, lift=0.3, name="bell"),
            Part(disc(0.0, 0.0, 0.10), colour=ACCENT, z=8, lift=0.4, name="throat"),
            Part(pad(-0.22, 0.0, 0.56, 0.13), colour=TRIM, z=10, lift=1.0, name="hose"),
        ),
        dome=0.45,
        description="Hydraulic disc driven by the filter pump. No onboard motor.",
    )


def make_design(design: CleanerDesign | str | None) -> CleanerDesign:
    """Resolve a design from a name, an object, or ``None`` for the default."""
    if design is None:
        return DESIGNS.create("tracked")
    if isinstance(design, CleanerDesign):
        return design
    return DESIGNS.create(design)
