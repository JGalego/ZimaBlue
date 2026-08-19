"""Pools from drawings, not photographs.

    import zimablue as zb

    pool = zb.pool_from_sketch("napkin.jpg", width=9.0, depth=1.6)

Somebody sketches a pool on paper, on a whiteboard, or in a paint program, and
out comes a :class:`~zimablue.pool.Pool` to run a cleaner in.  It is the
fastest way to get a shape that is *nearly* yours into the simulator, which is
usually what you want before committing to measuring the real thing.

This is not a second copy of :mod:`zimablue.imaging`.  A sketch is a different
segmentation problem -- ink on paper rather than water among umbrellas -- but
everything downstream of "which pixels are pool" is identical: pick the region,
fill the holes, trace the outline, apply the scale, simplify, smooth.  So
:class:`SketchSegmenter` satisfies the same
:class:`~zimablue.imaging.Segmenter` protocol that the colour rules and
:class:`~zimablue.segment.SamSegmenter` do, and
:func:`~zimablue.imaging.trace_pool` does the rest unchanged::

    trace = zb.trace_pool("napkin.jpg", segmenter=zb.SketchSegmenter(), width=9.0)

What makes a drawing hard
-------------------------

Not the colours -- a sketch is usually the highest-contrast image you will ever
segment. The trouble is elsewhere:

**The line is not closed.** A hand-drawn loop has gaps where the pen lifted,
and a paint-program stroke drawn fast has gaps where the mouse jumped. The
stroke is dilated to bridge those before filling.

Which way it fails when the bridge is not enough depends on how you fill, and
the difference decided the default. Filling outward from a seed leaks through
the gap and returns the whole page -- a pool the size of the photograph, which
looks like a scale error and gets debugged for an hour. Filling inward from
the page border does the opposite: the border reaches through the gap into the
interior, so almost nothing is left and you get an empty pool. Failing to
nothing is much the better failure, and it is detectable -- if the filled area
is barely larger than the ink, the outline did not close, and that is an error
rather than a shrug.

**Interior clutter.** Sketches have arrows, dimension marks, a scribbled
"deep end", a coffee ring. Those are ink, and ink is not boundary. Filling
from the outside in rather than the inside out ignores them: the region taken
is everything the page's border cannot reach, so anything floating inside the
outline is part of the pool whether it is ink or not.

**Paper is not white.** Phone photos of paper have a shadow gradient across
them, and a global threshold either loses the line in the dark corner or turns
the shadow into ink. The threshold is computed locally.

What it will not do
-------------------

It finds *one closed outline*. A drawing with the pool and a separate spa
gives you the bigger one, and a drawing where the outline crosses itself gives
you whichever lobe is bigger. Neither is detectable from the mask alone, which
is why :meth:`~zimablue.imaging.PoolTrace.overlay` exists -- look at it.

And a sketch has no scale, exactly as a photograph has none, so ``width`` or
one of its siblings is still required. The difference is that with a sketch
you almost certainly mean ``width``: you know how long the pool is, and the
drawing is not to scale anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.imaging import require_pillow

__all__ = ["SketchSegmenter", "pool_from_sketch", "trace_sketch"]

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass
class SketchSegmenter:
    """Finds the region enclosed by a drawn outline.

    Satisfies :class:`~zimablue.imaging.Segmenter`, so it drops into
    :func:`~zimablue.imaging.trace_pool` wherever the colour rules or a SAM
    export would go.

    The defaults suit a dark line on light paper, which covers pencil, pen,
    marker and the default brush in every paint program. For a light line on a
    dark background -- a whiteboard photo inverted, a screenshot with a dark
    theme -- set ``dark_ink=False``.
    """

    dark_ink: bool = True
    """True when the line is darker than the page."""

    threshold: float = 0.12
    """How far from the local background a pixel must be to count as ink,
    as a fraction of full range. Lower catches faint pencil and also catches
    paper texture."""

    window: float = 0.08
    """Size of the neighbourhood the local background is estimated over, as a
    fraction of the image's smaller side.

    This is the setting that matters for a photo of paper. A global threshold
    across a shadow gradient either loses the line where the page is dark or
    calls the shadow ink; estimating the background locally removes the
    gradient before thresholding. Too small and the interior of a thick stroke
    becomes its own background and the stroke hollows out.
    """

    bridge: float = 0.012
    """How far to reach across a break in the line, as a fraction of the
    smaller side. About 1% closes the gaps a hand leaves and does not close a
    deliberate 20 cm inlet."""

    ink_fraction: tuple[float, float] = (0.001, 0.35)
    """Sanity bounds on how much of the image should be ink.

    Below the floor there is no drawing -- a blank page, or a threshold set too
    high. Above the ceiling the "ink" is a photograph or a filled shape, and
    the outline-tracing logic will return nonsense confidently. Both raise
    rather than guess.
    """

    fill_from_border: bool = True
    """Take everything the page border cannot reach.

    Why this is the default over flood filling outward from a seed:
    It needs no seed, so a drawing can be traced without being told where the
    inside is. And it fails safe: when the outline has a gap the bridge cannot
    close, this returns almost nothing and says so, where a seed fill leaks
    through the gap and returns the entire page as one enormous pool.
    """

    # Populated by __call__, for the caller who wants to know what happened.
    ink_share: float = 0.0
    bridged: int = 0
    filled_share: float = 0.0

    # ------------------------------------------------------------------
    def __call__(self, rgb: NDArray[np.uint8], sample: tuple[int, int] | None) -> BoolArray:
        ink = self.ink_mask(rgb)
        self.ink_share = float(ink.mean())
        low, high = self.ink_fraction
        if self.ink_share < low:
            raise ValueError(
                f"only {self.ink_share:.2%} of the image is ink, which is not a drawing. "
                f"Lower threshold= (currently {self.threshold}), or set dark_ink=False if "
                "the line is lighter than the background."
            )
        if self.ink_share > high:
            raise ValueError(
                f"{self.ink_share:.0%} of the image is ink, which is a photograph or a "
                "filled shape rather than an outline. Raise threshold=, or use "
                "trace_pool's colour rules if this is a photo."
            )

        span = min(rgb.shape[:2])
        self.bridged = max(round(self.bridge * span), 0)
        closed = _dilate(ink, self.bridged) if self.bridged else ink

        inside = _interior(closed) if self.fill_from_border else _flood_from(closed, sample)
        # The stroke itself belongs to the pool: a line drawn *on* the edge is
        # the edge, and excluding it shrinks every pool by the pen's width.
        inside |= closed

        # Undo the bridging dilation, or every pool comes out one bridge radius
        # too big -- about 1% of the image, which on a 10 m pool is 10 cm of
        # invented water all the way round.
        if self.bridged:
            inside = _erode(inside, self.bridged)

        self.filled_share = float(inside.mean())
        # An enclosed outline fills far more than the line itself covers -- the
        # test drawing here is eighteen times. Barely more than the ink means
        # the fill found no interior, which for a border fill is exactly what
        # an unclosed outline looks like.
        if self.filled_share < 2.0 * self.ink_share:
            raise ValueError(
                f"the outline does not enclose anything: it covers {self.ink_share:.1%} of "
                f"the image and the region inside it only {self.filled_share:.1%}. The line "
                f"is probably not closed -- raise bridge= (currently {self.bridge}) to reach "
                "further across the gaps."
            )
        return inside

    # ------------------------------------------------------------------
    def ink_mask(self, rgb: NDArray[np.uint8]) -> BoolArray:
        """Which pixels are line, after removing the background gradient."""
        grey = np.asarray(rgb, dtype=float).mean(axis=2) / 255.0
        span = min(grey.shape)
        radius = max(round(self.window * span), 1)
        background = _box_blur(grey, radius)
        difference = background - grey if self.dark_ink else grey - background
        return difference > self.threshold


# ----------------------------------------------------------------------
# Morphology and filling
# ----------------------------------------------------------------------
def _box_blur(image: FloatArray, radius: int) -> FloatArray:
    """Mean over a square window, via a summed-area table.

    Two cumulative sums and four lookups per pixel, independent of the window
    size -- which matters because the window here is a twelfth of the image and
    a naive convolution would be seconds rather than milliseconds.
    """
    padded = np.pad(image, radius + 1, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    rows, cols = image.shape
    r0 = np.arange(rows)
    c0 = np.arange(cols)
    top, bottom = r0[:, None], (r0 + size)[:, None]
    left, right = c0[None, :], (c0 + size)[None, :]
    total = (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )
    return total / (size * size)


def _dilate(mask: BoolArray, radius: int) -> BoolArray:
    """Grow by a disc of ``radius`` pixels, separably.

    A square dilation would be cheaper still, but it closes diagonal gaps at
    sqrt(2) times the stated reach, which makes ``bridge`` mean different
    things in different directions.
    """
    if radius <= 0:
        return mask
    grown = mask.copy()
    for offset in range(1, radius + 1):
        for axis in (0, 1):
            grown |= np.roll(grown, offset, axis=axis) | np.roll(grown, -offset, axis=axis)
    return grown


def _erode(mask: BoolArray, radius: int) -> BoolArray:
    """Shrink by ``radius``. Dilation of the complement, inverted."""
    if radius <= 0:
        return mask
    return ~_dilate(~mask, radius)


def _interior(closed: BoolArray) -> BoolArray:
    """Everything the image border cannot reach without crossing the line.

    A scanline flood fill from the border, iterated until nothing changes.
    Whatever the fill cannot reach is either inside the outline or is the
    outline, and both are the pool.
    """
    outside = np.zeros_like(closed)
    free = ~closed
    # Seed from every border pixel that is not itself ink.
    outside[0, :] = free[0, :]
    outside[-1, :] = free[-1, :]
    outside[:, 0] = free[:, 0]
    outside[:, -1] = free[:, -1]

    while True:
        grown = outside.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                grown |= np.roll(outside, shift, axis=axis)
        # Rolling wraps, which would leak across the image edges and join the
        # left of the page to the right. Clear the wrapped row and column.
        grown[0, :] &= outside[0, :] | np.roll(outside, -1, axis=0)[0, :]
        grown[-1, :] &= outside[-1, :] | np.roll(outside, 1, axis=0)[-1, :]
        grown &= free
        if np.array_equal(grown, outside):
            break
        outside = grown
    return ~outside & ~closed


def _flood_from(closed: BoolArray, sample: tuple[int, int] | None) -> BoolArray:
    """Everything reachable from ``sample`` without crossing the line."""
    if sample is None:
        raise ValueError(
            "fill_from_border=False needs sample=(x, y), a pixel inside the drawn outline"
        )
    x, y = int(sample[0]), int(sample[1])
    rows, cols = closed.shape
    if not (0 <= x < cols and 0 <= y < rows):
        raise ValueError(f"sample {sample} is outside the {cols}x{rows} image")
    if closed[y, x]:
        raise ValueError(f"sample {sample} landed on the line itself, not inside it")

    inside = np.zeros_like(closed)
    inside[y, x] = True
    free = ~closed
    while True:
        grown = inside.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                grown |= np.roll(inside, shift, axis=axis)
        grown &= free
        if np.array_equal(grown, inside):
            break
        inside = grown
    return inside


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------
def trace_sketch(image: Any, **kwargs: Any) -> Any:
    """Trace a pool out of a drawing. See :func:`~zimablue.imaging.trace_pool`.

    Every keyword of ``trace_pool`` works, plus the
    :class:`SketchSegmenter` settings, which are passed through by name.
    Defaults differ in two places and both are about drawings rather than
    photographs: glare repair is off, because a drawing has no specular
    highlights and the repair only has ink to work with; and the outline is
    simplified harder, because a hand-drawn line is wobbly at a scale nobody
    intended to mean anything.
    """
    from zimablue.imaging import trace_pool

    sketch_fields = {f for f in SketchSegmenter.__dataclass_fields__ if not f.startswith("_")}
    settings = {name: kwargs.pop(name) for name in list(kwargs) if name in sketch_fields}
    kwargs.setdefault("glare", False)
    kwargs.setdefault("simplify", 3.0)
    kwargs.setdefault("smooth_edges", 0.10)
    return trace_pool(image, segmenter=SketchSegmenter(**settings), **kwargs)


def pool_from_sketch(image: Any, *, depth: Any = 1.6, name: str = "sketched", **kwargs: Any) -> Any:
    """A :class:`~zimablue.pool.Pool` straight from a drawing."""
    require_pillow()
    return trace_sketch(image, **kwargs).pool(depth=depth, name=name)
