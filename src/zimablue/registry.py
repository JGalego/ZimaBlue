"""A very small named-factory registry.

Presets (pools, robots, dirt, controllers, backends) are all "a name maps to a
callable that builds the thing".  This is that, and nothing more: a dict, a
decorator, and a lookup whose error message lists the valid names.  No entry
points, no plugin discovery, no metaclasses -- those earn their complexity when
there is a second consumer, and there is not yet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to factory callables."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: register a factory under ``name``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self.kind} preset {name!r} is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def add(self, name: str, factory: Callable[..., T]) -> None:
        """Register a factory imperatively (for user code and tests)."""
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> T:
        """Build the preset called ``name``."""
        try:
            factory = self._factories[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} preset {name!r}. Available: {', '.join(self.names())}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Registry({self.kind!r}, {self.names()})"
