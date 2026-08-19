"""A very small named-factory registry, with plugin discovery.

Presets (pools, robots, dirt, controllers, planners, backends) are all "a name
maps to a callable that builds the thing".  This is that: a dict, a decorator,
and a lookup whose error message lists the valid names.

Registries that name an ``entry_point_group`` also accept plugins.  A separate
package declares a factory under that group::

    [project.entry-points."zimablue.planners"]
    lawnfair = "zimablue_lawnfair:make_planner"

and after ``pip install zimablue-lawnfair`` the name shows up in
``zimablue list``, in ``compare()``, and everywhere else the registry is
consulted -- no import of the plugin package required.  Discovery is lazy and
the plugin module is only imported when its name is actually built, so a
hundred installed plugins cost a metadata scan, not a hundred imports.

A plugin cannot shadow a built-in name, and the first of two plugins claiming
the same name wins; both cases warn rather than fail, because a broken
neighbour should not take the whole registry down with it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


def _entry_points(group: str):
    """Installed entry points for ``group``.

    A module-level seam rather than an inline call, so tests can substitute
    fake distributions without building real ``.dist-info`` directories.
    """
    from importlib.metadata import entry_points

    return entry_points(group=group)


class Registry(Generic[T]):
    """Maps names to factory callables."""

    def __init__(self, kind: str, *, entry_point_group: str | None = None) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._factories: dict[str, Callable[..., T]] = {}
        self._plugins: dict[str, object] = {}
        self._discovered = entry_point_group is None

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
        self._discover()
        factory = self._factories.get(name)
        if factory is None:
            entry = self._plugins.get(name)
            if entry is None:
                raise KeyError(
                    f"unknown {self.kind} preset {name!r}. Available: {', '.join(self.names())}"
                )
            factory = self._load(name, entry)
        return factory(**kwargs)

    def names(self) -> list[str]:
        """Registered names, built-in and plugin, sorted."""
        self._discover()
        return sorted(set(self._factories) | set(self._plugins))

    def __contains__(self, name: object) -> bool:
        self._discover()
        return name in self._factories or name in self._plugins

    def __len__(self) -> int:
        self._discover()
        return len(set(self._factories) | set(self._plugins))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Registry({self.kind!r}, {self.names()})"

    # -- plugins -------------------------------------------------------------
    def _discover(self) -> None:
        """Scan installed distributions for this registry's entry points, once.

        Import of the plugin module is deferred to :meth:`create`; discovery
        reads package metadata only.
        """
        if self._discovered:
            return
        self._discovered = True
        assert self.entry_point_group is not None
        for entry in _entry_points(self.entry_point_group):
            claimed = self._factories if entry.name in self._factories else self._plugins
            if entry.name in claimed:
                warnings.warn(
                    f"{self.kind} plugin {entry.name!r} from {_dist_name(entry)} ignored: "
                    f"the name is already taken",
                    stacklevel=3,
                )
                continue
            self._plugins[entry.name] = entry

    def _load(self, name: str, entry: object) -> Callable[..., T]:
        try:
            factory = entry.load()  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImportError(
                f"{self.kind} plugin {name!r} from {_dist_name(entry)} failed to import: {exc}"
            ) from exc
        if not callable(factory):
            raise TypeError(
                f"{self.kind} plugin {name!r} from {_dist_name(entry)} is not a factory: "
                f"the entry point must name a callable"
            )
        self._factories[name] = factory
        self._plugins.pop(name, None)
        return factory


def _dist_name(entry: object) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None)
    return str(name) if name else "an installed package"
