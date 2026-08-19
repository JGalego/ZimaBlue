"""Spatial dirt state: a continuous field plus discrete debris.

Two representations, because pools contain two genuinely different things:

* :class:`DirtField` -- a mass raster, one layer per dirt type, in grams per
  cell.  Right for sediment, sand, algae and biofilm, which are spread thin and
  continuous.
* :class:`DebrisSet` -- individually tracked items with a position, mass and
  size.  Right for leaves and twigs, which are big enough that whether one
  particular leaf fits through the intake is a real question.

Both are pure state containers.  Removal is driven by the cleaning model in
``zimablue.physics.cleaning``; the transport here (settling, resuspension,
drift toward drains) is the environment acting on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.dirt.types import DirtType, get_dirt_type
from zimablue.geometry import Grid, Window

__all__ = ["DebrisSet", "DirtField", "DirtState"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _box_blur(a: FloatArray) -> FloatArray:
    """3x3 mean filter by slicing.

    Hand-rolled to keep SciPy out of the required dependency set; at pool raster
    sizes (~50x100) this is fast enough to run every simulated second.
    """
    padded = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out += padded[dr : dr + a.shape[0], dc : dc + a.shape[1]]
    return out / 9.0


class DirtField:
    """Per-type dirt mass on a raster, in grams per cell."""

    def __init__(
        self,
        grid: Grid,
        mask: BoolArray,
        layers: dict[str, FloatArray] | None = None,
        types: dict[str, DirtType] | None = None,
    ) -> None:
        self.grid = grid
        self.mask = np.asarray(mask, dtype=bool)
        self.layers: dict[str, FloatArray] = {}
        self.types: dict[str, DirtType] = dict(types or {})
        for name, values in (layers or {}).items():
            self.add_layer(name, values)
        self._initial_total = self.total()
        self._initial_by_type = self.by_type()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_layer(self, dirt_type: str | DirtType, values: FloatArray) -> None:
        """Add mass of one type, creating the layer if needed."""
        dt = get_dirt_type(dirt_type)
        values = np.asarray(values, dtype=float)
        if values.shape != self.grid.shape:
            raise ValueError(
                f"dirt layer {dt.name!r} has shape {values.shape}, "
                f"expected {self.grid.shape} to match the pool grid"
            )
        values = np.where(self.mask, np.maximum(values, 0.0), 0.0)
        self.types[dt.name] = dt
        if dt.name in self.layers:
            self.layers[dt.name] = self.layers[dt.name] + values
        else:
            self.layers[dt.name] = values

    def freeze_initial(self) -> None:
        """Record the current state as the baseline for removal metrics."""
        self._initial_total = self.total()
        self._initial_by_type = self.by_type()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def total(self) -> float:
        """Total dirt mass in grams."""
        return float(sum(float(layer.sum()) for layer in self.layers.values()))

    def by_type(self) -> dict[str, float]:
        return {name: float(layer.sum()) for name, layer in self.layers.items()}

    @property
    def initial_total(self) -> float:
        return self._initial_total

    @property
    def initial_by_type(self) -> dict[str, float]:
        return dict(self._initial_by_type)

    def total_grid(self) -> FloatArray:
        """Sum over all layers -- what the replay renders as "how dirty"."""
        if not self.layers:
            return np.zeros(self.grid.shape, dtype=float)
        return np.sum(np.stack(list(self.layers.values())), axis=0)

    def concentration(self) -> FloatArray:
        """Dirt mass per square metre, for a resolution-independent view."""
        return self.total_grid() / self.grid.cell_area

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def remove_window(self, window: Window, fractions: dict[str, float]) -> dict[str, float]:
        """Remove a per-type fraction of the mass under ``window``.

        Returns the mass removed per type, in grams.  Fractions are clipped to
        ``[0, 1]``: a cleaning model can never remove more than is there.

        Window-scoped because this runs every tick over a patch the size of the
        cleaning head, while the raster covers the whole pool.
        """
        removed: dict[str, float] = {}
        for name, layer in self.layers.items():
            frac = float(np.clip(fractions.get(name, 0.0), 0.0, 1.0))
            if frac <= 0.0:
                continue
            patch = window.view(layer)
            taken = patch[window.mask] * frac
            total = float(taken.sum())
            if total <= 0:
                continue
            patch[window.mask] -= taken
            removed[name] = total
        return removed

    def disturb_window(self, window: Window, strength: float = 1.0) -> None:
        """Kick loose dirt out of ``window`` and let it resettle just outside it.

        This is why a robot driving fast through fine sediment leaves a haze
        behind it: mass is not destroyed, only moved.  Each type's
        ``resuspension`` controls how much is liable to shift.
        """
        if strength <= 0:
            return
        # Blur over the window padded by one cell, so lifted mass can land
        # outside the swath rather than settling straight back down.
        rows = slice(max(window.rows.start - 1, 0), window.rows.stop + 1)
        cols = slice(max(window.cols.start - 1, 0), window.cols.stop + 1)
        inner = (
            slice(
                window.rows.start - rows.start,
                window.rows.start - rows.start + window.mask.shape[0],
            ),
            slice(
                window.cols.start - cols.start,
                window.cols.start - cols.start + window.mask.shape[1],
            ),
        )
        local_mask = self.mask[rows, cols]

        for name, layer in self.layers.items():
            dirt = self.types[name]
            share = float(np.clip(dirt.resuspension * strength, 0.0, 1.0))
            if share <= 0:
                continue
            patch = layer[rows, cols]
            lifted = np.zeros_like(patch)
            lifted[inner][window.mask] = patch[inner][window.mask] * share
            total = float(lifted.sum())
            if total <= 0:
                continue
            spread = np.where(local_mask, _box_blur(lifted), 0.0)
            spread_sum = float(spread.sum())
            if spread_sum <= 0:
                continue
            # Renormalise: the blur leaks mass past the mask and the window edge.
            patch += spread * (total / spread_sum) - lifted

    def drift(
        self,
        flow_vx: FloatArray,
        flow_vy: FloatArray,
        dt: float,
        *,
        spread: float = 0.12,
    ) -> None:
        """Advect fine, easily-suspended dirt along the water flow.

        First-order upwind on the raster, applied only to the fraction of each
        layer that is light enough to move, plus a diffusion term. The flow
        field itself is a superposition heuristic rather than a solution, and
        this is a matching level of coarseness -- but three details are not
        negotiable, because each of them produced a visible artefact.

        **The upwind gate reads the source cell.** An earlier version wrote
        ``np.where(cx > 0, np.roll(mx, 1, axis=1), 0.0)``, which tests the
        velocity at the cell the mass is arriving *at*. Everywhere the flow is
        smooth the two agree; at a stagnation line, where the sign flips, the
        destination's velocity is already negative and the arriving mass was
        silently dropped.

        **Mass that would leave the pool goes back to the sender.** Rolling
        wraps around the array, and the old code let it, then clipped with the
        mask and rescaled the whole field by ``layer.sum() / out.sum()`` to put
        the total back. That is a *multiplicative* correction: it hands the
        most mass to whichever cell already has the most, which is exactly the
        cell at the middle of a convergence. Shifting without wrapping and
        returning blocked mass to its origin conserves the total by
        construction, and no cell is rewarded for being large.

        **Diffusion is required, not decorative.** Advection alone concentrates
        without bound wherever the flow converges: on a one-dimensional
        convergence test, four hundred steps of pure upwind put twenty times a
        cell's initial load into one cell. The kidney pool used to have two
        return jets pointing at each other, so this was not hypothetical -- it
        drew a brown stripe down the middle of the pool that looked like floor
        the robot had missed. ``spread`` of 0.08 holds
        the same test to about 17 times over three cells, which is a dirt line
        rather than a knife edge, and is what turbulence does.

        ``spread`` is one Laplacian step per drift call and must stay below
        0.25 for an explicit four-neighbour stencil to be stable.
        """
        if dt <= 0:
            return
        if not 0.0 <= spread < 0.25:
            raise ValueError(f"spread must be in [0, 0.25) for stability, got {spread}")
        cell = self.grid.cell
        for name, layer in self.layers.items():
            dirt = self.types[name]
            mobile = float(np.clip(dirt.resuspension - 0.3, 0.0, 1.0))
            if mobile <= 0:
                continue
            # Courant fraction moved per axis, capped for stability.
            cx = np.clip(flow_vx * dt / cell, -0.4, 0.4) * mobile
            cy = np.clip(flow_vy * dt / cell, -0.4, 0.4) * mobile
            moving = layer * (np.abs(cx) + np.abs(cy))
            if moving.sum() <= 0:
                continue
            weight_x = np.abs(cx) / np.maximum(np.abs(cx) + np.abs(cy), 1e-12)
            along_x = moving * weight_x
            along_y = moving - along_x

            out = layer.copy()
            for leaving, step, axis, amount in (
                (cx > 0, 1, 1, along_x),
                (cx < 0, -1, 1, along_x),
                (cy > 0, 1, 0, along_y),
                (cy < 0, -1, 0, along_y),
            ):
                # A cell may only send what somewhere can receive. Subtracting
                # the whole moving fraction first and putting back whatever
                # failed to land is the same idea and one step harder to get
                # right: the shift truncates at the *array* edge as well as at
                # the pool wall, and mass that fell off the array was gone
                # before there was anything left to give back.
                sent = np.where(leaving & (_shift(self.mask, -step, axis) > 0), amount, 0.0)
                out -= sent
                out += _shift(sent, step, axis)

            if spread > 0:
                out += spread * (_neighbour_sum(out) - out * self._neighbour_count)
            self.layers[name] = np.where(self.mask, out, 0.0)

    @property
    def _neighbour_count(self) -> FloatArray:
        """How many of each cell's four neighbours are inside the pool.

        Cached: the mask does not change during a run, and the diffusion step
        needs it every call to stay conservative at the wall -- a cell on the
        edge must only give away as much as it can actually give.
        """
        cached = getattr(self, "_neighbours", None)
        if cached is None:
            inside = self.mask.astype(float)
            cached = _neighbour_sum(inside)
            self._neighbours = cached
        return cached

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def snapshot(self) -> FloatArray:
        """Compact float16 array of all layers, ``(n_types, rows, cols)``.

        float16 because keyframes are a visualisation and analysis artefact,
        not the source of truth: the metrics that matter are computed from the
        live field and stored separately. Halving the width halves the biggest
        contributor to a recording's size, and across the range a cell actually
        covers -- from a fifth of a gram of ordinary sediment to the eight or
        so grams a heap at the drain reaches -- float16 resolves to a few
        milligrams at worst.

        It is a *storage* format and nothing more. Reading it back through
        :meth:`~zimablue.recording.Recording.dirt_at` widens to float32 first,
        because summing the layers or weighting two keyframes at this precision
        loses far more than the quantisation does.
        """
        if not self.layers:
            return np.zeros((0, *self.grid.shape), dtype=np.float16)
        return np.stack([self.layers[n] for n in self.layer_names()]).astype(np.float16)

    def layer_names(self) -> list[str]:
        """Layer names in a stable order -- recordings depend on this."""
        return sorted(self.layers)

    def type_specs(self) -> list[dict[str, Any]]:
        return [self.types[n].to_dict() for n in self.layer_names()]


class DebrisSet:
    """Discrete debris items, stored column-wise for cheap vector queries."""

    def __init__(
        self,
        types: list[DirtType] | None = None,
        type_index: NDArray[np.int_] | None = None,
        x: FloatArray | None = None,
        y: FloatArray | None = None,
        mass: FloatArray | None = None,
        size: FloatArray | None = None,
    ) -> None:
        self.types: list[DirtType] = list(types or [])
        n = 0 if x is None else len(x)
        self.type_index = (
            np.zeros(n, dtype=int) if type_index is None else np.asarray(type_index, dtype=int)
        )
        self.x = np.zeros(n) if x is None else np.asarray(x, dtype=float)
        self.y = np.zeros(n) if y is None else np.asarray(y, dtype=float)
        self.mass = np.zeros(n) if mass is None else np.asarray(mass, dtype=float)
        self.size = np.zeros(n) if size is None else np.asarray(size, dtype=float)
        self.collected = np.zeros(n, dtype=bool)
        self._initial_mass = float(self.mass.sum())

    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def active(self) -> BoolArray:
        return ~self.collected

    @property
    def remaining_mass(self) -> float:
        return float(self.mass[self.active].sum())

    @property
    def initial_mass(self) -> float:
        return self._initial_mass

    @property
    def collected_count(self) -> int:
        return int(self.collected.sum())

    def near(self, x: float, y: float, radius: float) -> BoolArray:
        """Active items whose centre is within ``radius`` of ``(x, y)``."""
        return self.active & ((self.x - x) ** 2 + (self.y - y) ** 2 <= radius * radius)

    def collect(self, selection: BoolArray) -> tuple[float, int]:
        """Mark items collected; returns ``(mass, count)``."""
        selection = selection & self.active
        count = int(selection.sum())
        if count == 0:
            return (0.0, 0)
        mass = float(self.mass[selection].sum())
        self.collected |= selection
        return (mass, count)

    def nudge(self, selection: BoolArray, dx: float, dy: float, inside: Any = None) -> None:
        """Push items that are too big to swallow out of the way.

        ``inside`` is an optional predicate taking ``(x, y)`` arrays and
        returning a boolean mask.  Without it, a leaf shoved along by the robot
        for long enough eventually ends up outside the pool -- which looks
        exactly as wrong as it sounds when you watch the replay.
        """
        selection = selection & self.active
        if not selection.any():
            return
        new_x = np.where(selection, self.x + dx, self.x)
        new_y = np.where(selection, self.y + dy, self.y)
        if inside is not None:
            allowed = np.asarray(inside(new_x, new_y))
            new_x = np.where(allowed, new_x, self.x)
            new_y = np.where(allowed, new_y, self.y)
        self.x, self.y = new_x, new_y

    def type_names(self) -> list[str]:
        return [t.name for t in self.types]

    def snapshot(self) -> FloatArray:
        """``(n, 6)`` float32: ``x, y, mass, size, collected, type``.

        ``type`` indexes :meth:`type_names`. It is here because a leaf and a
        twig are not the same object to look at, and without it a replay can
        only draw both as the same anonymous blob.
        """
        if len(self) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack(
            [
                self.x,
                self.y,
                self.mass,
                self.size,
                self.collected.astype(float),
                self.type_index.astype(float),
            ]
        ).astype(np.float32)


class DirtState:
    """Everything dirty in the pool: the field plus the debris."""

    def __init__(self, field: DirtField, debris: DebrisSet | None = None) -> None:
        self.field = field
        self.debris = debris if debris is not None else DebrisSet()

    @property
    def total_mass(self) -> float:
        return self.field.total() + self.debris.remaining_mass

    @property
    def initial_mass(self) -> float:
        return self.field.initial_total + self.debris.initial_mass

    @property
    def removed_mass(self) -> float:
        return max(0.0, self.initial_mass - self.total_mass)

    @property
    def removed_fraction(self) -> float:
        if self.initial_mass <= 0:
            return 1.0
        return self.removed_mass / self.initial_mass

    def summary(self) -> dict[str, float]:
        out = {"total_g": self.total_mass, "initial_g": self.initial_mass}
        out.update({f"{k}_g": v for k, v in self.field.by_type().items()})
        if len(self.debris):
            out["debris_items"] = float(len(self.debris) - self.debris.collected_count)
        return out


def _shift(a: Any, step: int, axis: int) -> FloatArray:
    """Shift by one cell without wrapping; zeros come in at the edge.

    ``np.roll`` wraps, which teleports dirt from one end of the pool to the
    other. At raster resolution that is a few grams appearing against the far
    wall every drift step.
    """
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    if axis == 1:
        if step > 0:
            out[:, step:] = a[:, :-step]
        else:
            out[:, :step] = a[:, -step:]
    else:
        if step > 0:
            out[step:, :] = a[:-step, :]
        else:
            out[:step, :] = a[-step:, :]
    return out


def _neighbour_sum(a: FloatArray) -> FloatArray:
    """Sum of the four edge-adjacent cells, zero outside the array."""
    return _shift(a, 1, 1) + _shift(a, -1, 1) + _shift(a, 1, 0) + _shift(a, -1, 0)
