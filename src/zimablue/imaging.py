"""Build a pool from a photograph.

    import zimablue as zb

    pool = zb.pool_from_image("backyard.jpg", width=8.4, depth=1.6)

You point a phone at a pool, and out comes a :class:`~zimablue.pool.Pool` you
can run a cleaner in. What actually happens is three steps -- find the water,
trace its edge, and turn pixels into metres -- and each of them can be wrong in
a way the next one cannot detect, so this module is built to be *checked*
rather than trusted. :meth:`PoolTrace.overlay` draws what it found on top of
your photo, and that picture is the deliverable as much as the polygon is.

Scale is not optional
---------------------

A single photograph does not contain its own scale. Nothing in an image
distinguishes a 3 m plunge pool from a 30 m lap pool photographed from ten
times further away; the projection is identical. So one of these is required,
and there is no default:

``metres_per_pixel``
    You already know the ground resolution -- a satellite tile, say.
``width``
    The pool's real width, across the widest part of what gets traced.
``reference``
    Two pixels and the real distance between them: ``((x1, y1), (x2, y2), m)``.
    A diving board, a standard paving slab, a door.
``corners``
    Four pixels forming a real rectangle, and its size. This one also removes
    perspective, and is the one to use for a photo taken standing at the
    poolside.

The first three assume the camera looked straight down. A photo taken from
head height does not, and using them there returns a pool skewed by however
oblique the shot was -- narrower at the far end, and wrong by tens of percent.

Finding the water
-----------------

Backgrounds contain sky, blue umbrellas, blue tiling and blue cars. With
nothing to go on, the largest blue-ish region wins, which is usually the pool
and is sometimes the sky -- in a photo taken from the poolside the pool is
foreshortened and the sky is not, so the sky can genuinely be bigger.

**For a photograph, pass** ``sample=(x, y)`` **-- any pixel inside the
water.** Two things then change. The water's colour is read off the water
rather than assumed, which matters because pools go green and tiles go navy;
and the region *containing that pixel* is taken rather than the biggest one.
That second part is what settles the sky, which no colour rule can: a pool and
a summer sky are close enough in hue to be indistinguishable, but they are not
joined to each other.

Every candidate region is listed on the trace, and ``region=n`` picks one by
size if you would rather.

Sunlight is the other hazard. A specular highlight is white, so no rule that
looks for blue will find it; in the middle of the pool it leaves a hole, and
against the edge -- where the sun actually puts it -- it takes a bite out of
the outline. Both are repaired, and the repair is visible in the overlay.

What it cannot do
-----------------

Depth. A photograph of the surface says nothing about the floor beneath it, so
the depth model is yours to supply and defaults to a flat 1.5 m. If you know
the shallow and deep ends, hand in a
:class:`~zimablue.pool.PlaneSlopeDepth` and say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from zimablue.geometry import smooth_ring
from zimablue.pool import ConstantDepth, Pool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.pool.depth import DepthModel

__all__ = ["IMAGE_HINT", "PoolTrace", "Region", "pool_from_image", "require_pillow", "trace_pool"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

IMAGE_HINT = "Pillow is needed to read images. Install it with:  pip install 'zimablue[image]'"

MAX_SIZE = 720
"""Longest side the tracer works at, in pixels.

A pool outline has no detail that survives past this, and everything here is
either per-pixel or a flood fill, so halving the side quarters the work. The
scale arguments are still given in the original image's pixels.
"""


def require_pillow() -> None:
    """Raise an actionable error if Pillow is missing."""
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise ModuleNotFoundError(IMAGE_HINT) from exc


# ----------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """One candidate water region, in working-image pixels."""

    pixels: int
    centroid: tuple[float, float]
    touches_border: bool

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        edge = ", cut off by the frame" if self.touches_border else ""
        return f"Region({self.pixels} px at {self.centroid[0]:.0f},{self.centroid[1]:.0f}{edge})"


def _label(mask: BoolArray) -> NDArray[np.int32]:
    """Label 4-connected components, by runs rather than by pixels.

    A per-pixel flood fill in Python over a 720 px image is hundreds of
    thousands of iterations. Rows of a segmented photograph contain a handful
    of runs each, so unioning runs against the row above is the same answer for
    a thousandth of the work.
    """
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    previous: list[tuple[int, int, int]] = []
    for r in range(rows):
        row = mask[r]
        if not row.any():
            previous = []
            continue
        # Run starts and ends, from the transitions in the row.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        current: list[tuple[int, int, int]] = []
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            label = 0
            for pstart, pend, plabel in previous:
                if pstart < end and start < pend:
                    label = plabel if label == 0 else label
                    union(label, plabel)
            if label == 0:
                parent.append(len(parent))
                label = len(parent) - 1
            labels[r, start:end] = label
            current.append((start, end, label))
        previous = current

    if len(parent) == 1:
        return labels
    # Flatten the union-find, then renumber so labels are contiguous. Root 0 is
    # the background and sorts first, so it stays 0 without special-casing.
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    _unique, compact = np.unique(roots, return_inverse=True)
    return compact.astype(np.int32).reshape(-1)[labels]


def _regions(mask: BoolArray) -> tuple[NDArray[np.int32], list[Region]]:
    """Label ``mask`` and describe each component, largest first."""
    labels = _label(mask)
    count = int(labels.max())
    if count == 0:
        return labels, []

    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    found = []
    for index in range(1, count + 1):
        selected = labels == index
        pixels = int(selected.sum())
        if pixels == 0:
            continue
        ys, xs = np.nonzero(selected)
        found.append(
            (
                pixels,
                index,
                Region(
                    pixels=pixels,
                    centroid=(float(xs.mean()), float(ys.mean())),
                    touches_border=bool((selected & border).any()),
                ),
            )
        )
    found.sort(key=lambda item: -item[0])
    # Renumber so region 0 is the largest, which is the one callers index by.
    remap = np.zeros(count + 1, dtype=np.int32)
    for rank, (_pixels, index, _region) in enumerate(found):
        remap[index] = rank + 1
    return remap[labels], [region for _pixels, _index, region in found]


def _fill_holes(mask: BoolArray) -> BoolArray:
    """Close gaps enclosed by the mask -- a float, a swimmer, a ladder.

    Background that reaches the image border is outside; anything else is a
    hole. Without this a pool with anything in it traces as a pool full of
    islands.
    """
    labels = _label(~mask)
    if labels.max() == 0:
        return mask
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    outside = np.isin(labels, np.unique(border[border > 0]))
    return mask | ~(outside | mask)


def _dilate(mask: BoolArray, iterations: int = 1) -> BoolArray:
    """Grow the mask by one 4-neighbourhood per iteration."""
    out = mask
    for _ in range(iterations):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def _close(mask: BoolArray, radius: int) -> BoolArray:
    """Bridge gaps narrower than ``2 * radius`` without moving the outer edge."""
    if radius <= 0:
        return mask
    return ~_dilate(~_dilate(mask, radius), radius)


def _box3(a: FloatArray) -> FloatArray:
    """Sum over each pixel's 3x3 neighbourhood, edges included."""
    pad = ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad)
    out = np.zeros_like(a, dtype=float)
    for dy in range(3):
        for dx in range(3):
            out += p[dy : dy + a.shape[0], dx : dx + a.shape[1]]
    return out


def _grow_to_edges(
    mask: BoolArray, rgb: NDArray[np.uint8], *, passes: int, tolerance: float, floor: float
) -> BoolArray:
    """Creep the water outward while the colour keeps changing gradually.

    A pool does not end where a colour threshold ends. The last stretch before
    the coping is a hand's width of very shallow water over the edge, and it
    grades continuously from the water beside it: on the photo this was tuned
    against, hue runs 182 degrees in the middle to 161 at the rim without a
    single step worth calling an edge. Any global rule wide enough to hold both
    ends of that ramp is also wide enough to hold the coping, whose hue sits
    between them at 201.

    What separates the rim from the coping is not colour but *gradient*. The
    ramp is gradual and the coping arrives as a 36 degree jump, so a pixel is
    taken when it looks like the neighbours that already belong -- rather than
    like a sample taken metres away -- and the hard edge stops it dead. Missing
    that rim cost about 20 cm all the way round, which on a 25 m pool is 9% of
    the floor.

    ``passes`` bounds how far it can creep, so a leak through a gap in the
    coping costs a few pixels rather than the whole patio.
    """
    if passes <= 0:
        return mask

    colours = rgb.astype(np.float64)
    saturation = _to_hsv(rgb)[..., 1]
    grown = mask
    for _ in range(passes):
        frontier = _dilate(grown) & ~grown
        if not frontier.any():
            break
        count = _box3(grown.astype(float))
        neighbourhood = _box3(colours * grown[..., None])
        with np.errstate(divide="ignore", invalid="ignore"):
            local = neighbourhood / np.maximum(count, 1.0)[..., None]
        difference = np.linalg.norm(colours - local, axis=-1)
        accepted = frontier & (count > 0) & (difference <= tolerance) & (saturation >= floor)
        if not accepted.any():
            break
        grown = grown | accepted
    return grown


def _absorb_glare(
    mask: BoolArray, rgb: NDArray[np.uint8], *, value: float | None, surround: float
) -> BoolArray:
    """Take back the patches the sun blew out.

    A specular highlight on water is white, not blue, so no colour rule that
    finds water will find its glare. Filling holes recovers a highlight in the
    middle of the pool but not one against the edge, which is exactly where the
    sun puts it in an afternoon photo. That one is a bite out of the outline,
    open to the outside, and it cost 15% of the traced area in testing.

    The rule: a blob brighter than the water, not running off the frame, and
    whose own border is mostly water, is water. ``surround`` is that fraction,
    and it is what stops a white sun lounger at the pool edge going in too.

    The brightness threshold is read off the water rather than fixed, because a
    fixed one cuts through the middle of the highlight's gradient. The dim
    outer ring of the glare then belongs to neither the water nor the blown-out
    core, insulating one from the other so nothing is ever absorbed.
    """
    v = _to_hsv(rgb)[..., 2]
    threshold = float(np.percentile(v[mask], 97)) if value is None else value
    bright = ~mask & (v >= threshold)
    if not bright.any():
        return mask

    water_pixels = int(mask.sum())
    labels = _label(bright)
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    absorbed = mask.copy()
    for index in range(1, int(labels.max()) + 1):
        blob = labels == index
        # Sunlit decking is brighter than the water too, but it runs off the
        # frame and dwarfs the pool.
        if (blob & border).any() or int(blob.sum()) > 0.5 * water_pixels:
            continue
        ring = _dilate(blob) & ~blob
        touching = int(ring.sum())
        if touching and int((ring & mask).sum()) / touching >= surround:
            absorbed |= blob
    return absorbed


# ----------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------
def _to_hsv(rgb: NDArray[np.uint8]) -> FloatArray:
    """Hue in degrees, saturation and value in ``[0, 1]``."""
    a = rgb.astype(np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    high, low = a.max(axis=-1), a.min(axis=-1)
    span = high - low

    hue = np.zeros_like(high)
    with np.errstate(divide="ignore", invalid="ignore"):
        hue = np.where(high == r, (g - b) / span % 6.0, hue)
        hue = np.where(high == g, (b - r) / span + 2.0, hue)
        hue = np.where(high == b, (r - g) / span + 4.0, hue)
    hue = np.where(span < 1e-9, 0.0, hue) * 60.0
    saturation = np.where(high < 1e-9, 0.0, span / np.maximum(high, 1e-9))
    return np.stack([hue, saturation, high], axis=-1)


def _water_by_colour(
    rgb: NDArray[np.uint8], hue: tuple[float, float], saturation: float, value: float
) -> BoolArray:
    """Everything blue-to-cyan enough to be water."""
    hsv = _to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= hue[0]) & (h <= hue[1]) & (s >= saturation) & (v >= value)


def _water_by_sample(
    rgb: NDArray[np.uint8],
    sample: tuple[int, int],
    hue_tolerance: float,
    saturation_ratio: float,
    saturation_floor: float,
) -> BoolArray:
    """Everything the same *hue* as a patch around ``sample``, and as vivid.

    Reading the water's colour off the water beats asserting what colour water
    is. Pools go green, tiles go navy, and evening light moves everything.

    Matching on hue and saturation rather than on the colour as a whole is what
    lets one sample cover a pool from the deep end to the top step. Depth
    changes a pool's colour enormously in RGB -- on the photo this was tuned
    against, the deep end and the shallowest step are 100 apart out of 255 --
    while barely moving the hue, because it is the same water over the same
    plaster with more or less of it in between. Any distance measured in RGB
    therefore has to be opened up so far to reach the steps that it swallows
    the stone coping on the way, and no amount of tuning fixes that: on that
    photo the outermost step and the coping differ by *twelve*, which is less
    than two patches of open water differ from each other.

    Hue separates them cleanly -- water 181 degrees, coping 202 -- and
    saturation confirms it, since wet plaster is vivid and dry stone is not.
    Both are read off the sample rather than fixed, so a green pool or a navy
    one works the same way, and brightness is ignored entirely, which is what
    keeps sunlit and shaded water together in one region.
    """
    x, y = int(sample[0]), int(sample[1])
    rows, cols = rgb.shape[:2]
    if not (0 <= x < cols and 0 <= y < rows):
        raise ValueError(f"sample {sample} is outside the {cols}x{rows} image")

    half = max(2, min(rows, cols) // 100)
    patch = rgb[
        max(y - half, 0) : y + half + 1,
        max(x - half, 0) : x + half + 1,
    ]
    reference = _to_hsv(np.median(patch.reshape(-1, 3), axis=0).reshape(1, 1, 3).astype(np.uint8))

    hsv = _to_hsv(rgb)
    # Hue is an angle, so 359 and 1 are two degrees apart.
    delta = np.abs(hsv[..., 0] - reference[0, 0, 0])
    delta = np.minimum(delta, 360.0 - delta)

    # A near-grey pixel has no meaningful hue at all, so the saturation test
    # has to stand on its own rather than merely support the hue one.
    floor = max(saturation_floor, float(reference[0, 0, 1]) * saturation_ratio)
    return (delta <= hue_tolerance) & (hsv[..., 1] >= floor)


# ----------------------------------------------------------------------
# Pixels to metres
# ----------------------------------------------------------------------
def _homography(source: FloatArray, target: FloatArray) -> FloatArray:
    """The 3x3 transform taking four ``source`` points onto four ``target``.

    Direct linear transform: each correspondence gives two equations in the
    eight unknowns, so four points determine it exactly. This is what undoes
    the perspective of a photo taken from the poolside -- and it is applied to
    the traced outline rather than to the image, so nothing is resampled.
    """
    rows = []
    for (u, v), (x, y) in zip(source, target, strict=True):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y])
    solution, *_ = np.linalg.lstsq(np.array(rows), target.reshape(-1), rcond=None)
    return np.append(solution, 1.0).reshape(3, 3)


def _apply(matrix: FloatArray, points: FloatArray) -> FloatArray:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


# ----------------------------------------------------------------------
# The trace
# ----------------------------------------------------------------------
@dataclass
class PoolTrace:
    """What the tracer found, and enough context to judge whether it is right."""

    image: NDArray[np.uint8]
    """The working image -- downscaled from the original, RGB."""

    mask: BoolArray
    outline_px: FloatArray
    """The traced boundary in working-image pixels, one row per vertex."""

    boundary: Polygon
    """The boundary in metres, y up, translated so the pool starts at the origin."""

    regions: list[Region] = field(default_factory=list)
    chosen: int = 0
    scale: str = ""
    """How pixels became metres, in words -- worth printing next to the area."""

    warnings: list[str] = field(default_factory=list)

    @property
    def area(self) -> float:
        return float(self.boundary.area)

    def summary(self) -> str:
        minx, miny, maxx, maxy = self.boundary.bounds
        lines = [
            f"traced   {self.area:.1f} m2  ({maxx - minx:.1f} x {maxy - miny:.1f} m)",
            f"scale    {self.scale}",
            f"outline  {len(self.outline_px)} vertices from {self.regions[self.chosen].pixels}"
            " pixels",
        ]
        if len(self.regions) > 1:
            others = ", ".join(f"{r.pixels}px" for r in self.regions[1:4])
            lines.append(f"also saw {len(self.regions) - 1} other region(s): {others}")
        lines += [f"WARNING  {w}" for w in self.warnings]
        return "\n".join(lines)

    def pool(
        self,
        depth: DepthModel | float = 1.5,
        *,
        name: str = "traced",
        material: str = "plaster",
        features: tuple[Any, ...] = (),
        water: Any = None,
    ) -> Pool:
        """Turn the trace into a :class:`~zimablue.pool.Pool`.

        Depth is yours: a photograph of the surface says nothing about the
        floor under it.
        """
        return Pool(
            boundary=self.boundary,
            depth=ConstantDepth(float(depth)) if isinstance(depth, int | float) else depth,
            name=name,
            material=material,
            features=features,
            water=water,
        )

    def overlay(self, path: str | Path | None = None, *, ax: Any = None) -> Any:
        """Draw the outline over the photo. Look at this before trusting it.

        Automatic segmentation of an arbitrary snapshot is a guess, and the
        cheapest way to catch a wrong one -- the sky, a blue parasol, half the
        pool -- is to see it.
        """
        from zimablue.replay._deps import require_matplotlib

        require_matplotlib()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.imshow(self.image)
        ax.imshow(
            np.dstack([np.zeros((*self.mask.shape, 2)), self.mask * 1.0, self.mask * 0.35]),
            interpolation="nearest",
        )
        closed = np.vstack([self.outline_px, self.outline_px[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="#ff6b6b", linewidth=2.0)
        for index, region in enumerate(self.regions[:5]):
            if index == self.chosen:
                continue
            ax.annotate(
                f"region {index}",
                region.centroid,
                color="#ffb648",
                fontsize=8,
                ha="center",
            )
        ax.set_title(self.summary().splitlines()[0], fontsize=10)
        ax.set_axis_off()
        if path is not None:
            figure = ax.get_figure()
            figure.tight_layout()
            figure.savefig(path, dpi=130)
        return ax


def _load(image: Any, max_size: int) -> tuple[NDArray[np.uint8], float]:
    """Read anything image-shaped into an RGB array, and say how much it shrank."""
    require_pillow()
    from PIL import Image

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.dstack([array] * 3)
        picture = Image.fromarray(array[..., :3].astype(np.uint8))
    elif isinstance(image, (str, Path)):
        picture = Image.open(image)
    else:
        picture = image
    picture = picture.convert("RGB")

    factor = min(1.0, max_size / max(picture.size))
    if factor < 1.0:
        picture = picture.resize(
            (max(1, round(picture.width * factor)), max(1, round(picture.height * factor))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(picture, dtype=np.uint8), factor


def _trace_boundary(mask: BoolArray) -> FloatArray:
    """Moore-neighbour trace of the outer boundary, clockwise in pixel space.

    Marching squares would give a subpixel contour, but the mask came from a
    colour threshold on a photograph -- its edge is uncertain by a pixel or two
    anyway, and the outline is smoothed afterwards.
    """
    padded = np.pad(mask, 1)
    starts = np.argwhere(padded)
    if not len(starts):  # pragma: no cover - guarded by the caller
        raise ValueError("nothing to trace")
    start = tuple(starts[0])

    # Clockwise from west, in (row, col) offsets.
    neighbours = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    contour = [start]
    current = start
    backtrack = 0
    for _ in range(4 * padded.size):
        for step in range(1, 9):
            direction = (backtrack + step) % 8
            dr, dc = neighbours[direction]
            candidate = (current[0] + dr, current[1] + dc)
            if padded[candidate]:
                # Come into the next pixel from where we just were.
                backtrack = (direction + 4 + 1) % 8
                current = candidate
                break
        else:  # pragma: no cover - an isolated pixel
            break
        if current == start and len(contour) > 2:
            break
        contour.append(current)

    points = np.array(contour, dtype=float) - 1.0
    return points[:, ::-1]  # (row, col) -> (x, y)


def trace_pool(
    image: Any,
    *,
    # -- where the water is ------------------------------------------------
    sample: tuple[int, int] | None = None,
    region: int | None = None,
    hue_tolerance: float = 16.0,
    saturation_ratio: float = 0.22,
    saturation_floor: float = 0.07,
    grow: int = 8,
    grow_tolerance: float = 40.0,
    hue: tuple[float, float] = (150.0, 250.0),
    saturation: float = 0.16,
    value: float = 0.12,
    # -- how big it is -----------------------------------------------------
    metres_per_pixel: float | None = None,
    width: float | None = None,
    reference: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    corners: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
        tuple[float, float],
    ]
    | None = None,
    # -- cleaning up --------------------------------------------------------
    glare: bool = True,
    glare_value: float | None = None,
    glare_surround: float = 0.55,
    closing: int = 2,
    close_gaps: float = 0.35,
    smooth_edges: float = 0.0,
    # -- how smooth ---------------------------------------------------------
    simplify: float = 1.5,
    smooth: int | None = None,
    max_size: int = MAX_SIZE,
) -> PoolTrace:
    """Find a pool in ``image`` and measure it.

    Exactly one of ``metres_per_pixel``, ``width``, ``reference`` or
    ``corners`` is required -- a photograph carries no scale of its own, and
    inventing one would put every area and coverage number downstream quietly
    out by a factor nobody could see.

    Two ways to tidy the outline, and they are not interchangeable.
    ``smooth_edges`` is a radius in metres: a rolling ball that shaves the
    pixel staircase and fillets the corners by exactly that much, which is what
    a real pool's corners are anyway. ``smooth`` is a count of Fourier
    harmonics, which has no scale -- it suits a shape that genuinely is a curve
    and turns straight sides into curves to pay for the corners on one that is
    not. Both are off by default; reach for ``smooth_edges=0.15`` first.
    """
    scales = [
        metres_per_pixel is not None,
        width is not None,
        reference is not None,
        corners is not None,
    ]
    if sum(scales) != 1:
        raise ValueError(
            "give exactly one of metres_per_pixel, width, reference or corners -- "
            "a photograph does not contain its own scale, so there is no default"
        )

    rgb, factor = _load(image, max_size)
    rows, cols = rgb.shape[:2]

    seed: tuple[int, int] | None = None
    if sample is not None:
        seed = (round(sample[0] * factor), round(sample[1] * factor))
        mask = _water_by_sample(rgb, seed, hue_tolerance, saturation_ratio, saturation_floor)
    else:
        mask = _water_by_colour(rgb, hue, saturation, value)

    if not mask.any():
        raise ValueError(
            "found no water. Pass sample=(x, y) with a pixel inside the pool, or widen "
            "hue=/saturation=/value= if it is not blue"
        )

    labels, regions = _regions(mask)

    if region is None:
        # A seed says which region, and says it better than size does. Sky and
        # pool water are close enough in colour that a colour rule cannot
        # separate them, but they are not joined to each other, so the piece
        # the seed lands in is the answer. Without a seed, fall back to size.
        region = 0
        if seed is not None:
            at_seed = int(labels[seed[1], seed[0]])
            if at_seed == 0:
                raise ValueError(
                    f"sample {sample} did not land on water -- it matched no region. "
                    "Check the pixel, or raise hue_tolerance="
                )
            region = at_seed - 1
    if region >= len(regions):
        raise ValueError(f"asked for region {region} but only {len(regions)} were found")

    mask = labels == region + 1
    if grow and seed is not None:
        # Only with a seed: without one the colour rule is a guess already, and
        # growing a guess just makes a bigger one.
        # Only the absolute floor here, not the ratio-derived one the global
        # rule uses. Growth is already bounded twice over -- by the pass count
        # and by the local gradient -- so it can afford to reach into water too
        # shallow to be vivid, which is exactly the rim it is here to recover.
        mask = _grow_to_edges(
            mask, rgb, passes=grow, tolerance=grow_tolerance, floor=saturation_floor
        )
    if glare:
        # Before hole filling, because a highlight against the pool edge is a
        # bite out of the outline rather than a hole in it.
        mask = _absorb_glare(mask, rgb, value=glare_value, surround=glare_surround)
    mask = _fill_holes(_close(mask, closing))
    # Absorbing and closing can bridge to a neighbour, so keep the piece the
    # region actually was.
    merged, _ = _regions(mask)
    mask = merged == 1

    warnings: list[str] = []
    chosen = regions[region]
    if chosen.touches_border:
        warnings.append(
            "the water runs off the edge of the frame, so this is the part you "
            "photographed rather than the whole pool"
        )
    if chosen.pixels < 0.01 * rows * cols:
        warnings.append(f"only {chosen.pixels} pixels -- check the overlay before using this")
    if len(regions) > 1 and regions[1].pixels > 0.6 * chosen.pixels and sample is None:
        warnings.append(
            f"region 1 is nearly as big ({regions[1].pixels} px vs {chosen.pixels} px); if the "
            "wrong one was picked, pass sample=(x, y) or region=1"
        )

    outline_px = _trace_boundary(mask)
    pixel_polygon = Polygon(outline_px)
    if simplify > 0:
        pixel_polygon = pixel_polygon.simplify(simplify)
    if not pixel_polygon.is_valid or pixel_polygon.is_empty:
        pixel_polygon = pixel_polygon.buffer(0)
        if pixel_polygon.geom_type == "MultiPolygon":
            pixel_polygon = max(pixel_polygon.geoms, key=lambda g: g.area)

    ring = np.asarray(pixel_polygon.exterior.coords, dtype=float)[:-1]
    boundary, scale = _to_metres(ring, factor, rows, metres_per_pixel, width, reference, corners)

    if (close_gaps > 0 or smooth_edges > 0) and pixel_polygon.area > 0:
        # Closing belongs in metres -- see _close_notches -- but it is applied
        # to the pixel outline so that what the overlay draws is what was used.
        # The radius converts through the average linear scale, which for a
        # perspective-corrected trace varies a little across the frame and does
        # not matter at the size of the slots being removed.
        metres_per_px = float(np.sqrt(boundary.area / pixel_polygon.area))
        pixel_polygon = _close_notches(pixel_polygon, close_gaps / metres_per_px)
        if smooth_edges > 0:
            pixel_polygon = _round_edges(pixel_polygon, smooth_edges / metres_per_px)
        if simplify > 0:
            # Buffering replaces every corner with an arc of vertices; without
            # this the outline leaves with twice as many points as it arrived.
            pixel_polygon = pixel_polygon.simplify(simplify)
        ring = np.asarray(pixel_polygon.exterior.coords, dtype=float)[:-1]
        boundary, scale = _to_metres(
            ring, factor, rows, metres_per_pixel, width, reference, corners
        )

    if smooth:
        # In pixel space, again so the overlay shows what was used. Smoothing
        # is scale-free, so doing it here or after scaling is the same curve.
        pixel_polygon = smooth_ring(pixel_polygon, harmonics=int(smooth))
        ring = np.asarray(pixel_polygon.exterior.coords, dtype=float)[:-1]
        boundary, scale = _to_metres(
            ring, factor, rows, metres_per_pixel, width, reference, corners
        )

    minx, miny, _maxx, _maxy = boundary.bounds
    from shapely.affinity import translate

    boundary = translate(boundary, -minx, -miny)

    return PoolTrace(
        image=rgb,
        mask=mask,
        outline_px=np.asarray(pixel_polygon.exterior.coords, dtype=float),
        boundary=boundary,
        regions=regions,
        chosen=region,
        scale=scale,
        warnings=warnings,
    )


def _close_notches(boundary: Polygon, radius: float) -> Polygon:
    """Fill intrusions narrower than ``2 * radius`` metres.

    Every way a photograph goes wrong at the water's edge leaves the same
    signature: a slot poked into the outline that is far longer than it is
    wide. An underwater lamp does it, a hard shadow does it, a pool ladder
    does it. Chasing each cause through the colour rules is a losing game --
    on one photo the insulating shell was too bright for the water test, on the
    next it was too *yellow* for it -- but the shape of the damage is the same
    every time and a morphological closing removes it by definition.

    The threshold is set in metres, because that is the only place it means
    anything: pools do not have 30 cm slots in them, so anything narrower than
    that is the photograph rather than the pool. A genuine narrow waist -- a
    kidney's, or the neck on a figure-of-eight -- is metres wide and survives.

    Shapely's buffer does the closing exactly on the polygon, so there is no
    second rasterisation and no staircase to smooth away afterwards.
    """
    closed = boundary.buffer(radius, join_style=1).buffer(-radius, join_style=1)
    if closed.is_empty:  # pragma: no cover - only for a degenerate sliver
        return boundary
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda g: g.area)
    return Polygon(closed.exterior)


def _round_edges(boundary: Polygon, radius: float) -> Polygon:
    """Fillet the outline with a rolling ball of ``radius`` metres.

    The companion to :func:`_close_notches`. That one fills intrusions; this
    one shaves protrusions and rounds convex corners, and between them they are
    a morphological smoothing at a stated physical scale.

    Worth having because a traced outline carries two kinds of roughness that
    Fourier smoothing cannot tell apart. One is the pixel grid's staircase and
    the ragged pixel or two where the water meets the coping, which is noise.
    The other is the pool's actual corners. A rolling ball removes the first
    -- nothing survives that is smaller than the ball -- and merely *rounds*
    the second, by exactly the radius asked for, which is what a real pool's
    corners are anyway: struck with a radius, not mitred to a point.

    Fourier smoothing is global instead, so it has no scale and no way to spend
    its budget locally. Sixteen harmonics on a pool with straight sides turned
    the sides into curves to pay for the corners.
    """
    if radius <= 0:
        return boundary

    # Straighten first. A rolling ball rounds corners but rides over every
    # wobble on the way between them, and a traced edge that should be one
    # straight run arrives as thirty segments each a pixel out of line.
    # Collapsing anything smaller than the ball is the same judgement the ball
    # makes, applied to the runs instead of the corners.
    result = boundary.simplify(radius)

    # Then round both ways: opening takes the convex corners, closing takes the
    # concave ones. One without the other rounds half the outline and leaves
    # the rest mitred, which looks like a mistake rather than a finish.
    for op in (-radius, radius):
        stepped = result.buffer(op, join_style=1).buffer(-op, join_style=1)
        if stepped.is_empty:  # a pool thinner than the ball; leave it alone
            return boundary
        if stepped.geom_type == "MultiPolygon":
            stepped = max(stepped.geoms, key=lambda g: g.area)
        result = Polygon(stepped.exterior)
    return result


def _to_metres(
    ring: FloatArray,
    factor: float,
    rows: int,
    metres_per_pixel: float | None,
    width: float | None,
    reference: tuple[tuple[float, float], tuple[float, float], float] | None,
    corners: Any,
) -> tuple[Polygon, str]:
    """Map a pixel ring into world metres, y up."""
    if corners is not None:
        points, (rect_w, rect_h) = corners
        source = np.asarray(points, dtype=float) * factor
        # The four points trace the rectangle as it appears in the photo,
        # starting at whichever corner should land on the world origin.
        target = np.array([[0.0, 0.0], [rect_w, 0.0], [rect_w, rect_h], [0.0, rect_h]])
        matrix = _homography(source, target)
        world = _apply(matrix, ring)
        return Polygon(world), (
            f"perspective-corrected against a {rect_w:g} x {rect_h:g} m rectangle"
        )

    if reference is not None:
        (x1, y1), (x2, y2), metres = reference
        pixels = float(np.hypot(x2 - x1, y2 - y1)) * factor
        if pixels < 1e-6:
            raise ValueError("the two reference points are the same pixel")
        mpp = metres / pixels
        note = f"{metres:g} m across {pixels / factor:.0f} px of reference"
    elif width is not None:
        span = float(ring[:, 0].max() - ring[:, 0].min())
        if span < 1e-6:  # pragma: no cover - a degenerate trace
            raise ValueError("the traced outline has no width to scale by")
        mpp = width / span
        note = f"{width:g} m across the traced width"
    else:
        assert metres_per_pixel is not None
        # Given per *original* pixel, and the ring is in working pixels. The
        # image shrank by ``factor``, so each working pixel covers more ground,
        # not less -- multiplying here instead of dividing scaled the pool by
        # the square of the downscale and went unnoticed until a test compared
        # two ``max_size`` settings.
        mpp = metres_per_pixel / factor
        note = f"{metres_per_pixel:g} m per pixel"

    # Image rows increase downward and world y increases upward.
    world = np.column_stack([ring[:, 0] * mpp, (rows - ring[:, 1]) * mpp])
    return Polygon(world), f"{note}  ({mpp * factor:.4f} m/px of the original)"


def pool_from_image(
    image: Any,
    *,
    depth: DepthModel | float = 1.5,
    name: str = "traced",
    material: str = "plaster",
    **kwargs: Any,
) -> Pool:
    """Trace ``image`` and return the :class:`~zimablue.pool.Pool` directly.

    The short form. Keep the :class:`PoolTrace` instead when you want to look
    at :meth:`PoolTrace.overlay` first, which for a photograph you should.
    """
    trace = trace_pool(image, **kwargs)
    return trace.pool(depth, name=name, material=material)
