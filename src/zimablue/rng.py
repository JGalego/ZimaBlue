"""Deterministic random-number plumbing.

Every stochastic consumer in ZimaBlue (dirt generation, sensor noise, wheel
slip, fault triggering) draws from a *named* child stream of a single root
seed.  Names are folded into the ``SeedSequence`` spawn key, so a stream's
values depend only on the root seed and its own name -- adding a sixth sensor
never shifts the fifth sensor's noise sequence.

That property is what makes the determinism contract in ``docs/architecture.md``
survive ordinary refactoring.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

__all__ = ["RngTree", "name_to_key", "stable_hash"]

# Number of 32-bit words used to represent a stream name in the spawn key.
_KEY_WORDS = 4


def stable_hash(text: str) -> int:
    """Return a process-independent 128-bit hash of ``text``.

    ``hash()`` is salted per process (PYTHONHASHSEED), which would make seeded
    runs irreproducible across invocations.  BLAKE2b is not.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def name_to_key(name: str) -> tuple[int, ...]:
    """Fold a stream name into a tuple of 32-bit words for a spawn key."""
    value = stable_hash(name)
    return tuple((value >> (32 * i)) & 0xFFFF_FFFF for i in range(_KEY_WORDS))


class RngTree:
    """A root seed that hands out reproducible, independent named streams.

    >>> tree = RngTree(42)
    >>> a = tree.stream("dirt").random()
    >>> b = RngTree(42).stream("dirt").random()
    >>> a == b
    True

    Streams are memoised: asking for ``"dirt"`` twice returns the *same*
    generator, so a consumer that holds on to its generator keeps advancing one
    sequence rather than restarting it.
    """

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._seed

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator for ``name``, creating it on first use."""
        existing = self._streams.get(name)
        if existing is not None:
            return existing
        seq = np.random.SeedSequence(entropy=self._seed, spawn_key=name_to_key(name))
        generator = np.random.default_rng(seq)
        self._streams[name] = generator
        return generator

    def fresh(self, name: str) -> np.random.Generator:
        """Return a generator for ``name`` rewound to its start.

        Useful when a component must be able to regenerate an identical
        sequence (for example re-deriving a dirt field from a recording).
        """
        seq = np.random.SeedSequence(entropy=self._seed, spawn_key=name_to_key(name))
        generator = np.random.default_rng(seq)
        self._streams[name] = generator
        return generator

    def branch(self, name: str) -> RngTree:
        """A whole sub-tree, seeded from this one and this name.

        A stream is one generator; a branch is a new tree that can hand out
        streams of its own. What needs it is a fleet: each robot's backend
        wants its *own* ``"slip"``, and handing them all the same tree would
        have them drawing in turn from one sequence -- so the noise robot 2
        sees would depend on how many robots are ahead of it in the list, and
        adding a fourth robot would change the first three's runs.
        """
        seq = np.random.SeedSequence(entropy=self._seed, spawn_key=name_to_key(name))
        return RngTree(int(seq.generate_state(1, dtype=np.uint32)[0]))

    def names(self) -> Iterable[str]:
        """Names of the streams handed out so far (creation order)."""
        return tuple(self._streams)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"RngTree(seed={self._seed}, streams={len(self._streams)})"
