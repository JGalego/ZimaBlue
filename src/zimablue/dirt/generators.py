"""Deterministic dirt generation.

A :class:`DirtSpec` is a *declarative* description of how dirty a pool is --
which types, how much of each per square metre, and how they are distributed.
It is not a dirt field: it becomes one only when applied to a specific pool
with a specific RNG stream.  That separation is what lets one scenario file
run against six pool shapes and a thousand seeds.

Spatial patterns are composed by multiplying weight grids, so "leaves, blown
into the corners, patchily" is ``("patchy", "edges")`` rather than a bespoke
generator function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zimablue.dirt.field import DebrisSet, DirtField, DirtState
from zimablue.dirt.types import DirtType, get_dirt_type
from zimablue.registry import Registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.geometry import Grid
    from zimablue.pool import Pool

__all__ = [
    "DIRT_PRESETS",
    "PATTERNS",
    "DebrisSpec",
    "DirtSpec",
    "LayerSpec",
    "make_dirt",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PATTERNS = ("uniform", "patchy", "deep", "shallow", "edges", "corners", "centre")


# ---------------------------------------------------------------------------
# Spatial patterns
# ---------------------------------------------------------------------------
def _pattern_weights(
    name: str,
    pool: Pool,
    grid: Grid,
    mask: BoolArray,
    rng: np.random.Generator,
    *,
    patch_scale: float = 1.6,
    contrast: float = 1.2,
) -> FloatArray:
    """A non-negative weight grid for one named pattern.

    Weights are relative; the caller normalises so that a layer's total mass
    comes out exactly as specified regardless of pool size or pattern.
    """
    xs, ys = grid.cell_centers()

    if name == "uniform":
        return np.ones(grid.shape, dtype=float)

    if name == "patchy":
        # Sum of random Gaussian blobs: a smooth, seeded, scipy-free noise
        # field. Blob count scales with area so patch size stays physical.
        area = max(pool.floor_area, 1.0)
        count = int(np.clip(area / (patch_scale**2) * 1.5, 6, 220))
        minx, miny, maxx, maxy = pool.bounds
        cx = rng.uniform(minx, maxx, count)
        cy = rng.uniform(miny, maxy, count)
        amp = rng.uniform(0.4, 1.0, count)
        sigma = patch_scale * rng.uniform(0.5, 1.3, count)
        acc = np.zeros(grid.shape, dtype=float)
        for x0, y0, a, s in zip(cx, cy, amp, sigma, strict=True):
            acc += a * np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2.0 * s * s))
        acc /= max(acc.max(), 1e-12)
        return 0.15 + contrast * acc

    if name in ("deep", "shallow"):
        depths = pool.depth_grid(grid.cell)
        span = max(float(depths[mask].max() - depths[mask].min()), 1e-6)
        norm = (depths - float(depths[mask].min())) / span
        # Dense particles collect in low points; buoyant films gather shallow.
        return 0.2 + (norm if name == "deep" else 1.0 - norm) * 1.8

    if name in ("edges", "corners"):
        # Distance to the nearest wall, computed once by eroding the polygon
        # rather than per-cell against every segment.
        dist = _distance_to_wall(pool, grid, mask)
        near = np.exp(-dist / 0.5)
        if name == "edges":
            return 0.15 + 2.0 * near
        # Corners are where two walls are close at once: the interior distance
        # transform is small *and* the local wall direction changes. Squaring
        # the edge weight is a cheap stand-in that concentrates in corners.
        return 0.1 + 3.0 * near**2

    if name == "centre":
        dist = _distance_to_wall(pool, grid, mask)
        return 0.2 + 2.0 * (dist / max(dist.max(), 1e-6))

    raise ValueError(f"unknown dirt pattern {name!r}; available: {list(PATTERNS)}")


def _distance_to_wall(pool: Pool, grid: Grid, mask: BoolArray) -> FloatArray:
    """Approximate distance from each navigable cell to the nearest surface.

    Built by repeatedly shrinking the navigable polygon: each ring that
    survives an erosion of ``step`` is at least that far from a wall.  Cheaper
    and more robust than a per-cell segment search, and the 10 cm quantisation
    is below the raster resolution anyway.
    """
    from shapely import contains_xy

    xs, ys = grid.cell_centers()
    dist = np.zeros(grid.shape, dtype=float)
    step = grid.cell * 2.0
    region = pool.navigable
    for i in range(1, 25):
        region = pool.navigable.buffer(-step * i)
        if region.is_empty:
            break
        inside = np.asarray(contains_xy(region, xs, ys))
        if not inside.any():
            break
        dist = np.where(inside, step * i, dist)
    return np.where(mask, dist, 0.0)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSpec:
    """How much of one continuous dirt type there is, and where."""

    dirt: str
    grams_per_m2: float
    patterns: tuple[str, ...] = ("uniform",)
    """Multiplied together, so ``("patchy", "edges")`` means both at once."""

    patch_scale: float = 1.6
    contrast: float = 1.2

    grams_per_m2_per_hour: float = 0.0
    """Deposition while the run is going: leaves keep falling, pollen keeps
    landing. Spread with the same patterns as the initial mass, so a corner
    that starts dirty also *gets* dirty. Zero means the classic frozen pool."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dirt": self.dirt,
            "grams_per_m2": self.grams_per_m2,
            "patterns": list(self.patterns),
            "patch_scale": self.patch_scale,
            "contrast": self.contrast,
        }
        if self.grams_per_m2_per_hour:
            out["grams_per_m2_per_hour"] = self.grams_per_m2_per_hour
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerSpec:
        return cls(
            dirt=data["dirt"],
            grams_per_m2=float(data["grams_per_m2"]),
            patterns=tuple(data.get("patterns", ("uniform",))),
            patch_scale=float(data.get("patch_scale", 1.6)),
            contrast=float(data.get("contrast", 1.2)),
            grams_per_m2_per_hour=float(data.get("grams_per_m2_per_hour", 0.0)),
        )


@dataclass(frozen=True)
class DebrisSpec:
    """How many discrete items there are, and how big."""

    dirt: str
    per_100m2: float
    """Item count per 100 m^2 of floor, so pools of different sizes get
    comparable debris density."""

    mass_range: tuple[float, float] = (1.5, 6.0)
    """Grams per item."""

    size_range: tuple[float, float] = (0.03, 0.09)
    """Characteristic size in metres -- what decides whether the intake can
    swallow it."""

    patterns: tuple[str, ...] = ("uniform",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dirt": self.dirt,
            "per_100m2": self.per_100m2,
            "mass_range": list(self.mass_range),
            "size_range": list(self.size_range),
            "patterns": list(self.patterns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebrisSpec:
        return cls(
            dirt=data["dirt"],
            per_100m2=float(data["per_100m2"]),
            mass_range=tuple(data.get("mass_range", (1.5, 6.0))),
            size_range=tuple(data.get("size_range", (0.03, 0.09))),
            patterns=tuple(data.get("patterns", ("uniform",))),
        )


@dataclass(frozen=True)
class DirtSpec:
    """A reproducible description of how dirty a pool is."""

    name: str
    layers: tuple[LayerSpec, ...] = ()
    debris: tuple[DebrisSpec, ...] = ()
    description: str = ""

    stir_interval: float = 0.0
    """Seconds between somebody stirring the water -- a swimmer pushing off,
    a cannonball. Each stir resuspends loose dirt in one random patch. Zero
    means nobody is in the pool."""

    stir_strength: float = 0.6
    _types: dict[str, DirtType] = field(default_factory=dict, repr=False)
    """Custom dirt types referenced by name in ``layers``/``debris``."""

    def resolve(self, name: str) -> DirtType:
        return self._types.get(name) or get_dirt_type(name)

    def build(self, pool: Pool, rng: np.random.Generator, cell: float | None = None) -> DirtState:
        """Realise this spec as a concrete :class:`DirtState` for ``pool``.

        Given the same pool and the same generator state, the result is
        identical -- that is the whole contract.
        """
        from zimablue.pool import DEFAULT_CELL

        cell = DEFAULT_CELL if cell is None else cell
        grid = pool.grid(cell)
        mask = pool.navigable_mask(cell)
        dirt_field = DirtField(grid, mask)

        for layer in self.layers:
            dirt_type = self.resolve(layer.dirt)
            weights = np.ones(grid.shape, dtype=float)
            for pattern in layer.patterns:
                weights = weights * _pattern_weights(
                    pattern,
                    pool,
                    grid,
                    mask,
                    rng,
                    patch_scale=layer.patch_scale,
                    contrast=layer.contrast,
                )
            weights = np.where(mask, weights, 0.0)
            total = float(weights.sum())
            if total <= 0:
                continue
            target = layer.grams_per_m2 * pool.floor_area
            dirt_field.add_layer(dirt_type, weights * (target / total))
            if layer.grams_per_m2_per_hour > 0:
                rate = layer.grams_per_m2_per_hour * pool.floor_area / 3600.0
                dirt_field.attach_source(dirt_type, weights * (rate / total))

        debris = self._build_debris(pool, grid, mask, rng)
        dirt_field.freeze_initial()
        return DirtState(
            dirt_field,
            debris,
            stir_interval=self.stir_interval,
            stir_strength=self.stir_strength,
        )

    def _build_debris(
        self, pool: Pool, grid: Grid, mask: BoolArray, rng: np.random.Generator
    ) -> DebrisSet:
        types: list[DirtType] = []
        idx: list[int] = []
        xs_out: list[float] = []
        ys_out: list[float] = []
        masses: list[float] = []
        sizes: list[float] = []

        cell_xs, cell_ys = grid.cell_centers()
        for spec in self.debris:
            count = round(spec.per_100m2 * pool.floor_area / 100.0)
            if count <= 0:
                continue
            dirt_type = self.resolve(spec.dirt)
            types.append(dirt_type)
            type_id = len(types) - 1

            weights = np.ones(grid.shape, dtype=float)
            for pattern in spec.patterns:
                weights = weights * _pattern_weights(pattern, pool, grid, mask, rng)
            weights = np.where(mask, weights, 0.0)
            flat = weights.ravel()
            if flat.sum() <= 0:
                continue
            # Sample cells by weight, then jitter within the cell so items are
            # not visibly snapped to the raster.
            picks = rng.choice(flat.size, size=count, p=flat / flat.sum())
            jitter = rng.uniform(-0.5, 0.5, size=(count, 2)) * grid.cell
            xs_out.extend((cell_xs.ravel()[picks] + jitter[:, 0]).tolist())
            ys_out.extend((cell_ys.ravel()[picks] + jitter[:, 1]).tolist())
            masses.extend(rng.uniform(*spec.mass_range, size=count).tolist())
            sizes.extend(rng.uniform(*spec.size_range, size=count).tolist())
            idx.extend([type_id] * count)

        return DebrisSet(
            types=types,
            type_index=np.array(idx, dtype=int),
            x=np.array(xs_out, dtype=float),
            y=np.array(ys_out, dtype=float),
            mass=np.array(masses, dtype=float),
            size=np.array(sizes, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "layers": [layer.to_dict() for layer in self.layers],
            "debris": [d.to_dict() for d in self.debris],
        }
        if self.stir_interval:
            out["stir_interval"] = self.stir_interval
            out["stir_strength"] = self.stir_strength
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirtSpec:
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            layers=tuple(LayerSpec.from_dict(d) for d in data.get("layers", [])),
            debris=tuple(DebrisSpec.from_dict(d) for d in data.get("debris", [])),
            stir_interval=float(data.get("stir_interval", 0.0)),
            stir_strength=float(data.get("stir_strength", 0.6)),
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
DIRT_PRESETS: Registry[DirtSpec] = Registry("dirt", entry_point_group="zimablue.dirt")


@DIRT_PRESETS.register("clean")
def clean() -> DirtSpec:
    """A pool that was cleaned yesterday. The control condition."""
    return DirtSpec(
        name="clean",
        description="Freshly maintained; a trace of fine sediment only.",
        layers=(LayerSpec("sediment", grams_per_m2=1.5, patterns=("uniform",)),),
    )


@DIRT_PRESETS.register("light_sediment")
def light_sediment() -> DirtSpec:
    """A week of ordinary use: fine dust, slightly patchy, pooling low."""
    return DirtSpec(
        name="light_sediment",
        description="A week of normal use.",
        layers=(
            LayerSpec("sediment", grams_per_m2=12.0, patterns=("patchy", "deep")),
            LayerSpec("sand", grams_per_m2=4.0, patterns=("deep",)),
        ),
    )


@DIRT_PRESETS.register("autumn")
def autumn() -> DirtSpec:
    """Leaf fall: discrete debris that tests the intake, not just the brush.

    Leaves are large enough that some will not fit through the intake, so a
    robot can drive over them repeatedly and remove nothing -- the interesting
    failure this preset exists to produce.
    """
    return DirtSpec(
        name="autumn",
        description="Leaf fall over a still week.",
        layers=(
            LayerSpec("sediment", grams_per_m2=9.0, patterns=("patchy",)),
            LayerSpec("algae", grams_per_m2=5.0, patterns=("patchy", "edges")),
        ),
        debris=(
            DebrisSpec(
                "leaves",
                per_100m2=95.0,
                mass_range=(1.2, 5.5),
                size_range=(0.035, 0.11),
                patterns=("patchy", "deep"),
            ),
            DebrisSpec(
                "twigs",
                per_100m2=14.0,
                mass_range=(3.0, 12.0),
                size_range=(0.06, 0.16),
                patterns=("edges",),
            ),
        ),
    )


@DIRT_PRESETS.register("windy_day")
def windy_day() -> DirtSpec:
    """Wind-blown grit and floating litter piled against the lee wall."""
    return DirtSpec(
        name="windy_day",
        description="Grit and litter driven against the walls.",
        layers=(
            LayerSpec("sand", grams_per_m2=26.0, patterns=("edges", "patchy"), patch_scale=2.4),
            LayerSpec("sediment", grams_per_m2=8.0, patterns=("patchy",)),
        ),
        debris=(
            DebrisSpec(
                "leaves",
                per_100m2=45.0,
                mass_range=(0.8, 3.0),
                size_range=(0.03, 0.08),
                patterns=("edges",),
            ),
            DebrisSpec(
                "floating",
                per_100m2=70.0,
                mass_range=(0.1, 0.6),
                size_range=(0.004, 0.02),
                patterns=("edges", "shallow"),
            ),
        ),
    )


@DIRT_PRESETS.register("neglected_pool")
def neglected_pool() -> DirtSpec:
    """A month unattended: heavy adhered growth over everything.

    The hard case. Most of this mass is algae and biofilm, which suction alone
    cannot lift -- a robot with a weak brush will score high on coverage and
    low on dirt removed, which is the whole point of measuring both.
    """
    return DirtSpec(
        name="neglected_pool",
        description="A month unattended; heavy algae and biofilm.",
        layers=(
            LayerSpec("algae", grams_per_m2=42.0, patterns=("patchy", "edges"), contrast=1.6),
            LayerSpec("biofilm", grams_per_m2=18.0, patterns=("edges",)),
            LayerSpec("sediment", grams_per_m2=30.0, patterns=("patchy", "deep")),
            LayerSpec("sand", grams_per_m2=14.0, patterns=("deep", "corners")),
        ),
        debris=(
            DebrisSpec(
                "leaves",
                per_100m2=55.0,
                mass_range=(2.0, 8.0),
                size_range=(0.04, 0.12),
                patterns=("deep",),
            ),
        ),
    )


@DIRT_PRESETS.register("corner_heavy")
def corner_heavy() -> DirtSpec:
    """Everything piled in the corners: a direct test of edge coverage."""
    return DirtSpec(
        name="corner_heavy",
        description="Debris concentrated where a lawnmower path reaches last.",
        layers=(
            LayerSpec("sand", grams_per_m2=34.0, patterns=("corners",)),
            LayerSpec("sediment", grams_per_m2=10.0, patterns=("edges",)),
            LayerSpec("algae", grams_per_m2=8.0, patterns=("corners",)),
        ),
        debris=(
            DebrisSpec(
                "leaves",
                per_100m2=35.0,
                mass_range=(1.0, 4.0),
                size_range=(0.03, 0.09),
                patterns=("corners",),
            ),
        ),
    )


@DIRT_PRESETS.register("pool_party")
def pool_party() -> DirtSpec:
    """A pool in use. Dirt keeps arriving and the water keeps getting stirred,
    so 'done' is a rate you hold, not a state you reach."""
    return DirtSpec(
        name="pool_party",
        description="Sediment falling all afternoon, stirred by whoever is in the water.",
        layers=(
            LayerSpec(
                "sediment",
                grams_per_m2=6.0,
                patterns=("patchy",),
                grams_per_m2_per_hour=14.0,
            ),
            LayerSpec("sand", grams_per_m2=4.0, patterns=("edges",)),
        ),
        debris=(
            DebrisSpec(
                "floating",
                per_100m2=12.0,
                mass_range=(0.5, 1.5),
                size_range=(0.01, 0.03),
            ),
        ),
        stir_interval=45.0,
        stir_strength=0.5,
    )


@DIRT_PRESETS.register("random_debris")
def random_debris() -> DirtSpec:
    """A grab bag: every type present, patchily, for smoke-testing.

    Not a realistic pool -- a deliberately awkward one that exercises every
    branch of the cleaning model in a single short run.
    """
    return DirtSpec(
        name="random_debris",
        description="Every dirt type at once; a stress case, not a real pool.",
        layers=(
            LayerSpec("sediment", grams_per_m2=14.0, patterns=("patchy",), patch_scale=1.1),
            LayerSpec("sand", grams_per_m2=12.0, patterns=("patchy", "deep")),
            LayerSpec("algae", grams_per_m2=10.0, patterns=("patchy", "edges")),
            LayerSpec("biofilm", grams_per_m2=6.0, patterns=("corners",)),
        ),
        debris=(
            DebrisSpec("leaves", per_100m2=40.0, patterns=("patchy",)),
            DebrisSpec(
                "twigs",
                per_100m2=12.0,
                mass_range=(3.0, 14.0),
                size_range=(0.05, 0.18),
                patterns=("patchy",),
            ),
            DebrisSpec(
                "floating",
                per_100m2=25.0,
                mass_range=(0.1, 0.5),
                size_range=(0.003, 0.015),
                patterns=("shallow",),
            ),
        ),
    )


def make_dirt(name: str, **kwargs: object) -> DirtSpec:
    """Build a dirt preset by name.

    >>> make_dirt("autumn").name
    'autumn'
    """
    return DIRT_PRESETS.create(name, **kwargs)
