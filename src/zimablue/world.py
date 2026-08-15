"""The world: a pool and everything dirty in it.

Kept separate from :class:`~zimablue.pool.Pool` because a pool is a static
description while a world has *state* that a run mutates.  Two simulations can
share one ``Pool`` and hold independent ``World`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zimablue.pool import DEFAULT_CELL, Pool

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from zimablue.dirt import DirtSpec, DirtState

__all__ = ["World"]


@dataclass
class World:
    """A pool plus its dirt state."""

    pool: Pool
    dirt: DirtState
    cell: float = DEFAULT_CELL
    """Raster resolution shared by the dirt field, coverage grid and metrics."""

    @classmethod
    def build(
        cls,
        pool: Pool,
        dirt_spec: DirtSpec,
        rng: np.random.Generator,
        cell: float = DEFAULT_CELL,
    ) -> World:
        """Generate a world by applying ``dirt_spec`` to ``pool``."""
        return cls(pool=pool, dirt=dirt_spec.build(pool, rng, cell), cell=cell)

    @property
    def water(self):
        return self.pool.water
