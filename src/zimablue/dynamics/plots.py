"""Pictures for the analyses. matplotlib only, and only when called.

Each function draws one figure for one analysis and returns it, so a caller can
retitle it, put it in a grid or save it.  Nothing here computes anything -- the
numbers come from the analysis objects, and if a plot disagrees with a printed
number that is a bug in the plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.dynamics.averaging import CleaningForecast
from zimablue.dynamics.ergodic import ErgodicScore
from zimablue.dynamics.lyapunov import Divergence
from zimablue.dynamics.returnmap import ReturnMap
from zimablue.dynamics.transfer import TransferOperator
from zimablue.replay._deps import require_matplotlib
from zimablue.replay.renderer import PALETTE

__all__ = [
    "plot_divergence",
    "plot_ergodic",
    "plot_forecast",
    "plot_return_map",
    "plot_transfer",
]

FloatArray = NDArray[np.float64]


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> Any:
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["ink"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#24384c")
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=10, family="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, color="#7f9db8", fontsize=8, family="monospace")
    if ylabel:
        ax.set_ylabel(ylabel, color="#7f9db8", fontsize=8, family="monospace")
    return ax


def _figure(ncols: int = 1, width: float = 6.0, height: float = 3.4) -> tuple[Any, Any]:
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=PALETTE["panel"])
    return fig, np.atleast_1d(axes)


# ----------------------------------------------------------------------
def plot_return_map(section: ReturnMap, *, tolerance: float = 0.03) -> Any:
    """The Poincaré section, and its recurrence plot.

    Left: every wall contact as a point on the unrolled perimeter against its
    incidence angle, coloured by when it happened. Structure here -- points
    piling onto a few spots, or lying along a curve -- is the robot repeating
    itself. A cloud is a robot that is not.

    Right: the recurrence plot. Stripes parallel to the diagonal are
    periodicity, and their spacing is the period.
    """
    fig, axes = _figure(2, width=5.0, height=4.2)

    ax = _style(axes[0], f"section: {section.source}", "arc length along wall (m)", "angle (deg)")
    if len(section):
        dots = ax.scatter(
            section.s,
            np.degrees(section.theta),
            c=section.time / 60.0,
            cmap="viridis",
            s=22,
            edgecolors="none",
        )
        bar = fig.colorbar(dots, ax=ax, pad=0.02)
        bar.set_label("minutes", color="#7f9db8", fontsize=8)
        bar.ax.tick_params(colors=PALETTE["ink"], labelsize=7)
    ax.set_xlim(0, section.perimeter)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color="#24384c", linewidth=0.8)

    for orbit in section.periodic_orbits(tolerance)[:4]:
        ax.plot(
            orbit.s,
            np.degrees(orbit.theta),
            marker="o",
            markersize=15,
            markerfacecolor="none",
            markeredgecolor=PALETTE["bad"] if orbit.attracting else PALETTE["warn"],
            markeredgewidth=1.6,
            linestyle="none",
        )

    ax = _style(axes[1], "recurrence", "contact number", "contact number")
    if len(section) > 1:
        ax.imshow(
            section.recurrence_matrix(tolerance),
            cmap="magma",
            origin="lower",
            interpolation="nearest",
        )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def plot_transfer(operator: TransferOperator, *, sets: int = 2) -> Any:
    """Invariant measure, spectrum, and the regions the robot rarely leaves."""
    fig, axes = _figure(3, width=4.4, height=3.8)

    ax = _style(axes[0], "invariant measure", "x (m)", "y (m)")
    measure = operator.to_grid(operator.invariant_measure())
    extent = (
        float(operator.centres[:, 0].min() - operator.cell / 2),
        float(operator.centres[:, 0].max() + operator.cell / 2),
        float(operator.centres[:, 1].min() - operator.cell / 2),
        float(operator.centres[:, 1].max() + operator.cell / 2),
    )
    image = ax.imshow(measure, origin="lower", cmap="magma", extent=extent, interpolation="nearest")
    fig.colorbar(image, ax=ax, pad=0.02).ax.tick_params(colors=PALETTE["ink"], labelsize=7)
    ax.set_aspect("equal")

    ax = _style(axes[1], "spectrum", "Re", "Im")
    values = operator.eigenvalues[: min(40, len(operator))]
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color="#24384c", linewidth=1.0)
    ax.scatter(np.real(values), np.imag(values), s=26, color=PALETTE["accent"], edgecolors="none")
    if values.size > 1:
        ax.scatter(
            [np.real(values[1])],
            [np.imag(values[1])],
            s=90,
            facecolors="none",
            edgecolors=PALETTE["warn"],
            linewidths=1.6,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    mixing = operator.mixing_time
    ax.text(
        0.5,
        -0.22,
        f"gap {operator.spectral_gap:.3f} · mixing "
        + ("never" if not np.isfinite(mixing) else f"{mixing:.0f} s"),
        transform=ax.transAxes,
        ha="center",
        color="#7f9db8",
        fontsize=8,
        family="monospace",
    )

    ax = _style(axes[2], f"almost-invariant sets ({sets})", "x (m)", "y (m)")
    labels = operator.almost_invariant_sets(sets)
    ax.imshow(
        operator.to_grid(labels.astype(float)),
        origin="lower",
        cmap="Set2",
        extent=extent,
        interpolation="nearest",
    )
    ax.set_aspect("equal")
    leak = operator.leak_rate(labels)
    ax.text(
        0.5,
        -0.22,
        "leak " + " · ".join(f"{v:.1%}" for _, v in sorted(leak.items())),
        transform=ax.transAxes,
        ha="center",
        color="#7f9db8",
        fontsize=8,
        family="monospace",
    )
    fig.suptitle(operator.source, color=PALETTE["ink"], family="monospace", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


# ----------------------------------------------------------------------
def plot_ergodic(scores: dict[str, ErgodicScore]) -> Any:
    """Ergodic score over time, one line per controller.

    The shape matters more than the endpoint. A line still falling at the right
    edge was cut short; one that turns back up has stopped serving the target
    and started wasting time, which is what a controller that finishes and
    parks looks like.
    """
    fig, axes = _figure(1, width=7.0, height=4.0)
    ax = _style(axes[0], "", "minutes", "ergodic score (lower is better)")
    colours = ["#3ddcff", "#ffd166", "#ef476f", "#06d6a0", "#c77dff", "#ff9f1c"]

    for (name, score), colour in zip(scores.items(), colours, strict=False):
        ax.plot(score.times / 60.0, score.history, color=colour, linewidth=1.8, label=name)
        best = int(np.argmin(score.history))
        ax.plot(
            [score.times[best] / 60.0],
            [score.history[best]],
            marker="v",
            markersize=7,
            color=colour,
            linestyle="none",
        )
    ax.set_yscale("log")
    target = next(iter(scores.values())).target if scores else ""
    ax.set_title(
        f"distance from the {target} distribution   (v = best)",
        color=PALETTE["ink"],
        fontsize=10,
        family="monospace",
    )
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink"])
    legend.get_frame().set_alpha(0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def plot_divergence(runs: dict[str, Divergence]) -> Any:
    """Separation between twins on a log axis, with the fitted growth."""
    fig, axes = _figure(1, width=7.0, height=4.0)
    ax = _style(axes[0], "", "minutes", "separation (m)")
    colours = ["#3ddcff", "#ffd166", "#ef476f", "#06d6a0", "#c77dff"]

    for (name, run), colour in zip(runs.items(), colours, strict=False):
        for trace in run.separation:
            ax.plot(run.time / 60.0, trace, color=colour, linewidth=0.6, alpha=0.30)
        ax.plot(
            run.time / 60.0,
            run.typical,
            color=colour,
            linewidth=2.2,
            label=f"{name}  λ={run.exponent():.3f}/s  {run.diverged:.0%} diverged",
        )
        ax.axhline(0.25 * run.pool_scale, color="#24384c", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_title(
        "how fast two runs a millimetre apart stop agreeing\n"
        "(thin = one twin, thick = median, dotted = a quarter of the pool)",
        color=PALETTE["ink"],
        fontsize=9,
        family="monospace",
    )
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink"], loc="lower right")
    legend.get_frame().set_alpha(0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def plot_forecast(forecasts: dict[str, CleaningForecast]) -> Any:
    """Predicted against actual dirt, with the fitting window shaded."""
    fig, axes = _figure(len(forecasts), width=4.0, height=3.6)
    colours = ["#3ddcff", "#ffd166", "#ef476f", "#06d6a0"]

    for ax, (name, forecast), colour in zip(axes, forecasts.items(), colours, strict=False):
        _style(ax, name, "minutes", "dirt remaining (g)")
        ax.axvspan(0, forecast.fitted_from / 60.0, color="#16222f", zorder=0)
        ax.plot(forecast.times / 60.0, forecast.actual, color=colour, linewidth=2.0, label="actual")
        ax.plot(
            forecast.times / 60.0,
            forecast.predicted,
            color=PALETTE["ink"],
            linewidth=1.4,
            linestyle="--",
            label="predicted",
        )
        ax.text(
            0.97,
            0.94,
            f"error {forecast.forecast_error:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#7f9db8",
            fontsize=8,
            family="monospace",
        )
        legend = ax.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink"], loc="lower left")
        legend.get_frame().set_alpha(0)
    fig.suptitle(
        "fitted on the shaded window, forecast beyond it",
        color=PALETTE["ink"],
        family="monospace",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
