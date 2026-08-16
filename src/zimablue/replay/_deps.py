"""Optional-dependency guard for the drawing code.

Replay is an optional extra, so running it without matplotlib is an ordinary
mistake for a user to make. It deserves an instruction, not a traceback through
our internals -- and one that names the extra, since ``pip install matplotlib``
works but leaves the project's own dependency declaration wrong.

Its own module rather than ``replay/__init__`` so the submodules can import it
without a cycle.
"""

from __future__ import annotations

__all__ = ["VIZ_HINT", "require_matplotlib"]

VIZ_HINT = "matplotlib is needed to draw replays. Install it with:  pip install 'zimablue[viz]'"


def require_matplotlib() -> None:
    """Raise a actionable error if matplotlib is missing."""
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise ModuleNotFoundError(VIZ_HINT) from exc
