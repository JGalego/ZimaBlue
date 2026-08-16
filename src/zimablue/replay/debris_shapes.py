"""What a piece of debris looks like.

Debris used to be drawn as a brown dot in both views, which is fine as a
position marker and useless as a picture: a soaked oak leaf and a snapped twig
behave differently under a brush, and a viewer trying to work out why the robot
keeps stalling in one corner cannot tell them apart.

The outlines here are cheap parametric silhouettes -- an ovate leaf with a
stem, an irregular twig, a small ellipse for anything else -- built once in a
canonical unit frame and then scaled, rotated and placed per item. Both the
top-down replay and the dirt cam draw from this module, so a leaf is the same
leaf whichever way you look at it.

Every per-item choice (which outline variant, how far it is rotated, its exact
colour) is derived from the item's own index through a fixed hash. It has to be
stable: the same leaf must not change species between frames, and re-rendering
the same recording twice must give the same picture.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["DEBRIS_PALETTE", "debris_colour", "debris_outline", "debris_polygons"]

FloatArray = NDArray[np.float64]

DEBRIS_PALETTE: dict[str, tuple[str, ...]] = {
    # Autumn leaves are not one colour. Sampling a small spread per item is
    # what stops a drift of leaves reading as a single brown smear.
    "leaves": ("#a8602c", "#8f4a22", "#c07a35", "#7d5a2a", "#b34a28", "#96682f"),
    "twigs": ("#5f4020", "#6b4423", "#4a3218", "#75512b"),
    "floating": ("#d8c98a", "#c9b877", "#e0d49b"),
}
DEFAULT_COLOURS = ("#8a7452", "#7a6647")


def _leaf(lobes: int, waist: float, taper: float) -> FloatArray:
    """An ovate leaf pointing along +x, in a unit frame.

    The blade is a beta profile, ``t**0.55 * (1 - t)**taper``: it swells fast
    from the base, peaks around a third of the way along and draws out to a
    point. A sine bulge is symmetric about the middle and gives an almond
    instead, which is the shape this started as and did not look like a leaf.
    ``lobes`` ripples the margin so no two leaves in a drift are identical.
    """
    t = np.linspace(0.0, 1.0, 30)
    rise = 0.55
    peak = rise / (rise + taper)
    width = waist * (t**rise * (1.0 - t) ** taper) / (peak**rise * (1.0 - peak) ** taper)
    width *= 1.0 + 0.09 * np.sin(lobes * np.pi * t)

    # The stem needs width. A single point behind the blade is a zero-area
    # spike, and a filled polygon renders it as nothing at all.
    stem_half = 0.016
    return np.vstack(
        [
            [[-0.26, stem_half]],
            np.column_stack([t, width]),
            np.column_stack([t[::-1], -width[::-1]]),
            [[-0.26, -stem_half]],
        ]
    )


def _twig(kinks: int, thickness: float, node: float, spread: float) -> FloatArray:
    """A bent stick along +x with one side shoot, in a unit frame.

    The side shoot is what makes it read as a twig rather than as a scratch on
    the lens. It grows from one side only: a symmetric fork has to bring both
    prongs back to a shared crotch, and getting that wrong leaves the ring
    self-intersecting, which an even-odd fill draws as a hole.
    """
    t = np.linspace(0.0, 1.0, 18)
    bend = 0.06 * np.sin(kinks * np.pi * t)
    half = thickness * (1.0 - 0.45 * t)
    upper = np.column_stack([t, bend + half])
    lower = np.column_stack([t, bend - half])

    # Where the shoot leaves the trunk, and where it rejoins a little further
    # along. Splicing between two trunk vertices keeps the ring simple.
    start = int(np.searchsorted(t, node))
    end = min(start + 2, len(t) - 1)
    root = upper[start]
    tip = np.array([root[0] + 0.30, root[1] + spread])
    shoot_half = thickness * 0.55
    shoot = np.array(
        [
            root,
            [tip[0] - shoot_half, tip[1] + shoot_half * 0.4],
            tip,
            [tip[0] - shoot_half * 1.6, tip[1] - shoot_half * 1.2],
            upper[end],
        ]
    )

    return np.vstack([upper[:start], shoot, upper[end + 1 :], lower[::-1]])


def _blob() -> FloatArray:
    """A plain ellipse, for debris that is neither leaf nor twig."""
    a = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    return np.column_stack([0.5 + 0.5 * np.cos(a), 0.34 * np.sin(a)])


def _variants(kind: str) -> tuple[FloatArray, ...]:
    if kind == "leaves":
        return tuple(
            _leaf(lobes, waist, taper)
            for lobes, waist, taper in ((4, 0.21, 0.85), (6, 0.17, 1.15), (3, 0.235, 0.95))
        )
    if kind == "twigs":
        # The shoot always grows the same way. Sending it the other way means
        # splicing into the opposite edge, and getting that wrong crosses the
        # trunk; the per-item rotation already puts shoots on both sides.
        return tuple(
            _twig(kinks, thick, node, spread)
            for kinks, thick, node, spread in (
                (2, 0.045, 0.55, 0.20),
                (3, 0.032, 0.42, 0.16),
                (1, 0.055, 0.68, 0.24),
            )
        )
    return (_blob(),)


_CACHE: dict[str, tuple[FloatArray, ...]] = {}


def debris_outline(kind: str, index: int) -> FloatArray:
    """The unit-frame outline for item ``index`` of ``kind``."""
    if kind not in _CACHE:
        _CACHE[kind] = _variants(kind)
    shapes = _CACHE[kind]
    return shapes[_hash(index, 7919) % len(shapes)]


def debris_colour(kind: str, index: int) -> str:
    palette = DEBRIS_PALETTE.get(kind, DEFAULT_COLOURS)
    return palette[_hash(index, 6607) % len(palette)]


def _hash(index: int, salt: int) -> int:
    """A tiny integer hash. Python's ``hash`` is salted per process, so using
    it here would make a re-render of the same recording look different."""
    return (int(index) * 2654435761 + salt) & 0x7FFFFFFF


def debris_polygons(
    x: FloatArray,
    y: FloatArray,
    size: FloatArray,
    kinds: list[str],
    indices: NDArray[np.int_],
) -> list[FloatArray]:
    """World-space outlines for a set of debris items.

    ``size`` is the item's long dimension in metres, so the result is at true
    scale -- a 9 cm leaf comes out 9 cm across.
    """
    polygons = []
    for i, (px, py, length, kind) in enumerate(
        zip(x, y, size, kinds, strict=True), start=int(indices[0]) if len(indices) else 0
    ):
        unit = debris_outline(kind, i)
        angle = _hash(i, 104729) / 0x7FFFFFFF * 2.0 * np.pi
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        scaled = unit * float(length)
        polygons.append(
            np.column_stack(
                [
                    px + scaled[:, 0] * cos_a - scaled[:, 1] * sin_a,
                    py + scaled[:, 0] * sin_a + scaled[:, 1] * cos_a,
                ]
            )
        )
    return polygons
