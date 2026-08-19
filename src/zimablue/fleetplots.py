"""Pictures of a fleet.

A single-robot run has one path and one coverage number, and the top-down
replay says almost everything about it. A fleet has neither: the interesting
quantities are all *relations* -- who covered what, who did more than their
share, where two robots went over the same tile, when each of them stopped
being useful. None of that is visible in a picture of three robots driving.

One function per view, and :func:`plot_fleet` assembles them:

:func:`plot_paths`
    Every robot's trajectory in its own colour, on the pool. The one view
    where a partition is obvious at a glance -- and where a partition that
    went wrong is even more obvious.
:func:`plot_territory`
    Which robot actually covered each square metre. Not the plan: the outcome.
    A partitioned fleet should reproduce its partition here, and where it does
    not is where the follower failed.
:func:`plot_overlap`
    How many different robots went over each cell -- it draws the waste.
:func:`plot_progress`
    Team coverage against time, with each robot's contribution stacked under
    it, so a robot that finished early and parked is visible as a line that
    goes flat while the others keep climbing.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from zimablue.replay._deps import require_matplotlib
from zimablue.replay.renderer import FLEET_COLOURS, PALETTE

__all__ = [
    "plot_fleet",
    "plot_overlap",
    "plot_paths",
    "plot_progress",
    "plot_territory",
]

INK = PALETTE["ink"]
FAINT = "#7f9db8"


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> Any:
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=INK, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#24384c")
    if title:
        ax.set_title(title, color=INK, fontsize=9, family="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, color=FAINT, fontsize=8, family="monospace")
    if ylabel:
        ax.set_ylabel(ylabel, color=FAINT, fontsize=8, family="monospace")
    return ax


def _colour(index: int) -> str:
    return FLEET_COLOURS[index % len(FLEET_COLOURS)]


def _outline(result: Any) -> np.ndarray:
    return np.asarray(result.world.pool.boundary.exterior.coords)


def _tracks(result: Any) -> list[np.ndarray]:
    """Each robot's path, from the recording if there is one."""
    recording = result.recording
    if recording is None:
        raise ValueError(
            "these plots read the trajectories out of the recording; "
            "build the fleet with Fleet(..., record=True)"
        )
    frames = recording.frames
    count = len(result.states)
    out = []
    for index in range(count):
        prefix = f"r{index}." if count > 1 else ""
        out.append(
            np.column_stack([np.asarray(frames[f"{prefix}x"]), np.asarray(frames[f"{prefix}y"])])
        )
    return out


# ----------------------------------------------------------------------
def plot_paths(result: Any, *, ax: Any = None, stride: int = 3) -> Any:
    """Every robot's trajectory, in its own colour."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.6), facecolor=PALETTE["panel"])
        ax = fig.add_subplot(111)
        standalone = True
    else:
        fig, standalone = ax.figure, False

    outline = _outline(result)
    ax.plot(outline[:, 0], outline[:, 1], color="#24384c", linewidth=1.4)
    for index, track in enumerate(_tracks(result)):
        ax.plot(
            track[::stride, 0],
            track[::stride, 1],
            color=_colour(index),
            linewidth=0.8,
            alpha=0.85,
            label=f"r{index} {result.metrics.robots[index].coverage:.0%}",
        )
        ax.scatter([track[0, 0]], [track[0, 1]], color=_colour(index), s=22, zorder=5)
    _style(ax, f"where each robot went -- team {result.metrics.team.coverage:.0%}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    legend = ax.legend(
        fontsize=6, loc="upper right", facecolor=PALETTE["panel"], edgecolor="#24384c"
    )
    for text in legend.get_texts():
        text.set_color(INK)
    if standalone:
        fig.tight_layout()
    return fig


def plot_territory(result: Any, *, ax: Any = None) -> Any:
    """Which robot covered each cell -- the partition that actually happened.

    Worth putting next to the one the partitioner drew. A share that a robot
    was given and did not reach shows up here as somebody else's colour, and
    that gap is the difference between dividing a pool and cleaning it.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.6), facecolor=PALETTE["panel"])
        ax = fig.add_subplot(111)
        standalone = True
    else:
        fig, standalone = ax.figure, False

    count = len(result.states)
    territory = np.asarray(result.territory, dtype=float)
    territory[~result.spatial.navigable] = np.nan
    colours = ListedColormap([_colour(i) for i in range(count)])
    norm = BoundaryNorm(np.arange(-0.5, count), count)
    grid = result.world.pool.grid(result.world.cell)
    shown = np.where(territory < 0, np.nan, territory)
    ax.imshow(
        shown, extent=grid.extent, origin="lower", cmap=colours, norm=norm, interpolation="nearest"
    )
    outline = _outline(result)
    ax.plot(outline[:, 0], outline[:, 1], color="#24384c", linewidth=1.4)
    owned = np.asarray(result.territory)
    claimed = max(int((owned >= 0).sum()), 1)
    shares = "  ".join(f"r{i} {int((owned == i).sum()) / claimed:.0%}" for i in range(count))
    _style(ax, f"territory -- {shares}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if standalone:
        fig.tight_layout()
    return fig


def plot_overlap(result: Any, *, ax: Any = None) -> Any:
    """How many robots went over each cell.

    Everything above one is work the fleet paid for twice. A good partition
    leaves a thin seam along its internal borders and nothing else; a
    cooperative fleet with no partition covers the pool in plaid.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.6), facecolor=PALETTE["panel"])
        ax = fig.add_subplot(111)
        standalone = True
    else:
        fig, standalone = ax.figure, False

    times = np.asarray(result.times_covered, dtype=float)
    times[~result.spatial.navigable] = np.nan
    times[times == 0] = np.nan
    grid = result.world.pool.grid(result.world.cell)
    from matplotlib.colors import BoundaryNorm, ListedColormap

    count = max(len(result.states), 2)
    # Discrete, not a gradient: "two robots" and "three robots" are counts, and
    # a continuous scale invites reading a value off a colour that has none.
    shades = ListedColormap(["#1f6f8b", "#e8734a", "#ffd166", "#ff5d8f", "#c9a7ff"][:count])
    image = ax.imshow(
        times,
        extent=grid.extent,
        origin="lower",
        cmap=shades,
        norm=BoundaryNorm(np.arange(0.5, count + 1.5), count),
        interpolation="nearest",
    )
    outline = _outline(result)
    ax.plot(outline[:, 0], outline[:, 1], color="#24384c", linewidth=1.4)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=range(1, count + 1))
    bar.ax.tick_params(colors=INK, labelsize=6)
    _style(ax, f"robots per cell -- {result.metrics.overlap:.0%} of the floor done twice or more")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if standalone:
        fig.tight_layout()
    return fig


def plot_progress(result: Any, *, ax: Any = None, samples: int = 120) -> Any:
    """Team coverage over time, with each robot's own curve under it."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.2), facecolor=PALETTE["panel"])
        ax = fig.add_subplot(111)
        standalone = True
    else:
        fig, standalone = ax.figure, False

    recording = result.recording
    pool = result.world.pool
    swath = 0.34
    count = len(result.states)
    prefixes = [f"r{i}." if count > 1 else "" for i in range(count)]
    for index, prefix in enumerate(prefixes):
        times, values = _curve_for(recording, pool, [prefix], swath, samples)
        ax.plot(
            times / 60.0,
            values * 100.0,
            color=_colour(index),
            linewidth=1.1,
            alpha=0.8,
            label=f"r{index}",
        )
    # The team curve is the union, not the sum: two robots over the same tile
    # is one covered tile, and adding the curves would claim otherwise.
    times, values = _curve_for(recording, pool, prefixes, swath, samples)
    ax.plot(times / 60.0, values * 100.0, color=INK, linewidth=2.0, label="team")
    _style(ax, "coverage over time", "minutes", "% of floor")
    legend = ax.legend(
        fontsize=6, loc="upper left", facecolor=PALETTE["panel"], edgecolor="#24384c"
    )
    for text in legend.get_texts():
        text.set_color(INK)
    if standalone:
        fig.tight_layout()
    return fig


def _curve_for(recording, pool, prefixes, swath, samples):
    """Coverage against time for one robot, or for several taken together."""
    grid = pool.grid(0.1)
    navigable = pool.navigable_mask(0.1)
    total = int(navigable.sum())
    frames = recording.frames
    time = np.asarray(frames["time"], dtype=float)
    tracks = [
        (
            np.clip(
                ((np.asarray(frames[f"{p}y"]) - grid.miny) / 0.1).astype(int), 0, grid.nrows - 1
            ),
            np.clip(
                ((np.asarray(frames[f"{p}x"]) - grid.minx) / 0.1).astype(int), 0, grid.ncols - 1
            ),
        )
        for p in prefixes
    ]
    reach = max(round(0.5 * swath / 0.1), 0)
    offsets = [
        (dr, dc)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if np.hypot(dr, dc) * 0.1 <= 0.5 * swath
    ]
    covered = np.zeros_like(navigable)
    edges = np.unique(np.linspace(0, len(time), samples + 1).astype(int))
    times, fractions = [], []
    for start, stop in pairwise(edges):
        for rows, cols in tracks:
            for dr, dc in offsets:
                covered[
                    np.clip(rows[start:stop] + dr, 0, grid.nrows - 1),
                    np.clip(cols[start:stop] + dc, 0, grid.ncols - 1),
                ] = True
        times.append(time[stop - 1])
        fractions.append((covered & navigable).sum() / total if total else 0.0)
    return np.asarray(times), np.asarray(fractions)


def plot_fleet(result: Any) -> Any:
    """All the views in one figure."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.4, 8.0), facecolor=PALETTE["panel"])
    grid = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.12)
    plot_paths(result, ax=fig.add_subplot(grid[0, 0]))
    plot_territory(result, ax=fig.add_subplot(grid[0, 1]))
    plot_overlap(result, ax=fig.add_subplot(grid[1, 0]))
    plot_progress(result, ax=fig.add_subplot(grid[1, 1]))
    fig.suptitle(
        f"{len(result.states)} robots -- team {result.metrics.team.coverage:.0%}, "
        f"overlap {result.metrics.overlap:.0%}, speedup {result.metrics.speedup:.2f}x",
        color=INK,
        fontsize=11,
        family="monospace",
    )
    return fig
