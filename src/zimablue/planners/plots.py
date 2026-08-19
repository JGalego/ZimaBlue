"""Pictures of a planner comparison, because one view is not enough.

A wide table is a reference, not an argument. These are the readings of it
that turned out to be worth having, and they answer different questions:

:func:`plot_matrix`
    All planners, all dimensions, rescaled so the best in each column is full
    brightness. Reads the *shape* of a planner -- which axes it wins and which
    it pays for -- rather than its rank.
:func:`plot_paths`
    The trajectories themselves, side by side. Every scalar in the table is a
    summary of these, and a summary that surprises you is usually a picture you
    have not looked at.
:func:`plot_curves`
    Coverage against time. Two planners that finish level can get there
    completely differently, and for a machine on a battery the shape of the
    curve is the product.
:func:`plot_tradeoff`
    Any two dimensions against each other, with the Pareto front drawn. This
    is where "there is no best planner" stops being a disclaimer and becomes
    a picture.

:func:`plot_comparison` puts all four in one figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zimablue.planners.compare import Comparison
from zimablue.replay._deps import require_matplotlib
from zimablue.replay.renderer import PALETTE

__all__ = [
    "plot_comparison",
    "plot_curves",
    "plot_matrix",
    "plot_paths",
    "plot_plans",
    "plot_tradeoff",
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


def _new(width: float, height: float) -> Any:
    require_matplotlib()
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(width, height), facecolor=PALETTE["panel"])


# ----------------------------------------------------------------------
def plot_matrix(comparison: Comparison, *, pool: str | None = None, ax: Any = None) -> Any:
    """Planners down, dimensions across, colour by rank and text by value.

    The colour is normalised *within a column*, so it says "best here" and
    never "good in absolute terms" -- if every planner is bad at something,
    one of them is still bright. The number printed in each cell is the real
    measurement, and that is the one to quote.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    planners = comparison.planners
    dimensions = comparison.dimensions
    grid = comparison.matrix(pool)
    if ax is None:
        fig = _new(1.05 * len(dimensions) + 2.6, 0.36 * len(planners) + 1.6)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(
        [f"{d.label}\n{'higher' if d.better > 0 else 'lower'}" for d in dimensions],
        fontsize=7,
        color=INK,
        family="monospace",
    )
    ax.set_yticks(range(len(planners)))
    ax.set_yticklabels(planners, fontsize=7, color=INK, family="monospace")
    for row, planner in enumerate(planners):
        for col, dimension in enumerate(dimensions):
            value = comparison.score(planner, dimension.key, pool)
            ax.text(
                col,
                row,
                dimension.format(value),
                ha="center",
                va="center",
                fontsize=6,
                family="monospace",
                color="#0d151f" if grid[row, col] > 0.55 else "#dbe7f2",
            )
    title = comparison.label + (f" -- {pool}" if pool else "")
    ax.set_title(
        f"{title} ({comparison.minutes:.0f} min)", color=INK, fontsize=10, family="monospace"
    )
    ax.set_xticks(np.arange(-0.5, len(dimensions)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(planners)), minor=True)
    ax.grid(which="minor", color=PALETTE["panel"], linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_paths(
    comparison: Comparison, *, pool: str | None = None, columns: int = 4, seed: int | None = None
) -> Any:
    """Small multiples of the trajectories, one panel per planner."""
    require_matplotlib()

    pool_name = pool or comparison.pools[0]
    trials = [t for t in comparison.select(pool=pool_name) if seed is None or t.seed == seed]
    chosen: dict[str, Any] = {}
    for trial in trials:
        chosen.setdefault(trial.planner, trial)
    names = [n for n in comparison.planners if n in chosen]

    rows = int(np.ceil(len(names) / columns))
    fig = _new(2.7 * columns, 2.5 * rows)
    outline = _outline(pool_name)
    for index, name in enumerate(names):
        ax = fig.add_subplot(rows, columns, index + 1)
        trial = chosen[name]
        if outline is not None:
            ax.plot(outline[:, 0], outline[:, 1], color="#24384c", linewidth=1.2)
        path = trial.path
        # Colour along the path so the order of the sweep is visible; two
        # planners can trace the same set of lines in very different orders.
        points = np.arange(len(path))
        ax.scatter(path[:, 0], path[:, 1], c=points, cmap="cool", s=0.6, linewidths=0, alpha=0.85)
        _style(ax, f"{name}  {trial.scores['coverage']:.0%}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"where they went -- {pool_name}", color=INK, fontsize=11, family="monospace")
    fig.tight_layout()
    return fig


def plot_curves(comparison: Comparison, *, pool: str | None = None, ax: Any = None) -> Any:
    """Coverage against time, every planner on one axes.

    The anytime question. A planner whose curve is a straight line to its final
    value is spending its whole run productively; one that flattens at half
    time has finished what it can reach and is going over it again.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    pool_name = pool or comparison.pools[0]
    if ax is None:
        fig = _new(7.0, 4.2)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    colours = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(comparison.planners)))
    for colour, name in zip(colours, comparison.planners, strict=True):
        trials = comparison.select(planner=name, pool=pool_name)
        if not trials:
            continue
        times, values = trials[0].curve
        ax.plot(times / 60.0, values * 100.0, color=colour, linewidth=1.4, label=name)
    ax.axhline(50, color=FAINT, linewidth=0.6, linestyle=":")
    _style(ax, f"coverage over time -- {pool_name}", "minutes", "% of floor")
    legend = ax.legend(
        fontsize=6, ncol=2, loc="upper left", facecolor=PALETTE["panel"], edgecolor="#24384c"
    )
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    return fig


def plot_tradeoff(
    comparison: Comparison,
    *,
    x: str = "efficiency",
    y: str = "coverage",
    pool: str | None = None,
    ax: Any = None,
) -> Any:
    """Two dimensions against each other, with the Pareto front joined up.

    Default axes are the two that most often disagree: a planner can buy
    coverage with overlap, and efficiency is the bill.
    """
    require_matplotlib()

    lookup = {d.key: d for d in comparison.dimensions}
    dim_x, dim_y = lookup[x], lookup[y]
    if ax is None:
        fig = _new(5.6, 4.4)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    points = []
    for name in comparison.planners:
        px = comparison.score(name, x, pool) * dim_x.scale
        py = comparison.score(name, y, pool) * dim_y.scale
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        points.append((px, py, name))

    for px, py, name in points:
        ax.scatter([px], [py], s=26, color=PALETTE.get("accent", "#3ddcff"), zorder=3)
        ax.annotate(
            name,
            (px, py),
            fontsize=6,
            color=INK,
            family="monospace",
            xytext=(4, 3),
            textcoords="offset points",
        )

    front = _pareto(points, dim_x.better, dim_y.better)
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color=FAINT,
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )
    _style(
        ax,
        "the trade-off",
        f"{dim_x.label} ({'higher' if dim_x.better > 0 else 'lower'} is better)",
        f"{dim_y.label} ({'higher' if dim_y.better > 0 else 'lower'} is better)",
    )
    fig.tight_layout()
    return fig


def _pareto(points, better_x: int, better_y: int):
    """The non-dominated set, sorted for drawing."""
    front = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and (other[0] - candidate[0]) * better_x >= 0
            and (other[1] - candidate[1]) * better_y >= 0
            and (other[0] != candidate[0] or other[1] != candidate[1])
            for other in points
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda p: p[0])


def plot_plans(pool: Any, planners=None, *, robot: Any = None, columns: int = 4) -> Any:
    """The offline plans themselves, before anyone has tried to drive them.

    Worth looking at separately from the trajectories: the difference between
    a plan and the path that resulted from following it is the follower's
    error, and seeing them side by side is the only way to tell a bad route
    from a badly driven one.
    """
    require_matplotlib()
    import zimablue as zb
    from zimablue.planners.base import PLANNERS, make_planner

    if isinstance(pool, str):
        pool = zb.make_pool(pool)
    robot = robot if robot is not None else zb.make_robot("tracked")
    names = list(planners or PLANNERS.names())

    rows = int(np.ceil(len(names) / columns))
    fig = _new(2.9 * columns, 2.7 * rows)
    outline = np.asarray(pool.boundary.exterior.coords)
    for index, name in enumerate(names):
        ax = fig.add_subplot(rows, columns, index + 1)
        ax.plot(outline[:, 0], outline[:, 1], color="#24384c", linewidth=1.2)
        try:
            plan = make_planner(name).plan(pool, robot)
        except Exception as error:  # pragma: no cover - a planner that cannot
            _style(ax, f"{name}: {type(error).__name__}")
            ax.set_aspect("equal")
            continue
        for cell in plan.cells:
            try:
                coords = np.asarray(cell.exterior.coords)
            except AttributeError:
                continue
            ax.fill(coords[:, 0], coords[:, 1], color="#16283a", zorder=0)
            ax.plot(coords[:, 0], coords[:, 1], color="#24384c", linewidth=0.4, zorder=1)
        way = plan.waypoints
        ax.plot(way[:, 0], way[:, 1], color=PALETTE.get("accent", "#3ddcff"), linewidth=0.7)
        _style(ax, f"{name}\n{plan.length:.0f} m, {np.degrees(plan.turns):.0f} deg")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_comparison(comparison: Comparison, *, pool: str | None = None) -> Any:
    """All the views in one figure."""
    require_matplotlib()

    pool_name = pool or comparison.pools[0]
    fig = _new(
        max(1.05 * len(comparison.dimensions) + 2.6, 12.0),
        0.36 * len(comparison.planners) + 7.0,
    )
    grid = fig.add_gridspec(2, 2, height_ratios=(0.36 * len(comparison.planners) + 1.2, 4.4))
    plot_matrix(comparison, pool=pool, ax=fig.add_subplot(grid[0, :]))
    plot_curves(comparison, pool=pool_name, ax=fig.add_subplot(grid[1, 0]))
    plot_tradeoff(comparison, pool=pool_name, ax=fig.add_subplot(grid[1, 1]))
    fig.tight_layout()
    return fig


def _outline(pool_name: str):
    try:
        import zimablue as zb

        return np.asarray(zb.make_pool(pool_name).boundary.exterior.coords)
    except Exception:  # pragma: no cover - a pool built from a file
        return None
