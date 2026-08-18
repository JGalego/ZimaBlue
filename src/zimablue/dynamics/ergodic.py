"""The ergodic metric: one number for "spend time where the dirt is".

This project's argument is that coverage and cleanliness come apart, and it has
been making it with two separate numbers and a table showing they rank
controllers differently.  There is an established formalism that puts the
argument in one number instead, and it is a slight embarrassment that we got
this far without naming it.

Mathew and Mezić define the *ergodicity* of a trajectory against a target
distribution as a Sobolev-space distance between two things: the fraction of
time the trajectory has spent in each region, and the fraction of the target
distribution that lives there.  Expand both in a Fourier basis and the distance
is a weighted sum over modes::

    Phi(t) = sum_k  Lambda_k * | c_k(t) - mu_k |^2

where ``c_k`` are the trajectory's time-averaged basis coefficients, ``mu_k``
the target's, and ``Lambda_k = (1 + |k|^2)^-s`` with ``s = (n+1)/2`` weights
coarse structure above fine.  Zero means the trajectory has distributed itself
exactly as the target asks.

The choice that matters is the target.  Uniform gives you coverage, and it is
what every coverage planner implicitly optimises.  **Make the target the dirt
density and the metric becomes cleanliness** -- one scalar that says "spend
time in proportion to how dirty it is", falling as the robot does so.  The
disagreement this library keeps pointing at is then not two metrics but one
metric under two targets, which is a much sharper way to say it.

Why the weighting is not optional: without ``Lambda_k`` the sum is dominated by
high-frequency modes, and a trajectory would be judged on whether it visited
every square centimetre in the right proportion. Weighting coarse modes higher
says that being in the right half of the pool matters more than being in the
right centimetre, which is what anybody means by covering a pool.

Mathew, G., & Mezić, I. (2011). Metrics for ergodicity and design of ergodic
dynamics for multi-agent systems. *Physica D, 240*(4-5), 432-442.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["ErgodicScore", "ergodic_score", "target_measure"]

FloatArray = NDArray[np.float64]


@dataclass
class ErgodicScore:
    """How far a trajectory is from distributing itself like a target."""

    value: float
    """The metric at the end of the run. Lower is better; zero is perfect."""

    history: FloatArray
    """The metric over time, one sample per evaluation point.

    More useful than the endpoint. A trajectory that plateaus has stopped
    improving and the rest of the battery is being spent on ground it has
    already served; one still falling at the cutoff was interrupted.
    """

    times: FloatArray
    target: str
    modes: int
    coefficients: FloatArray
    """The trajectory's basis coefficients, ``(modes, modes)``."""

    target_coefficients: FloatArray

    @property
    def converged(self) -> bool:
        """Has the metric stopped falling?

        Compares the last fifth of the run against the fifth before it. False
        means the run was cut short of what this controller could have done.
        """
        if self.history.size < 10:
            return False
        fifth = max(self.history.size // 5, 1)
        recent = float(self.history[-fifth:].mean())
        before = float(self.history[-2 * fifth : -fifth].mean())
        return recent >= before * 0.98

    @property
    def best(self) -> float:
        """The lowest the score ever got."""
        return float(self.history.min()) if self.history.size else float("inf")

    @property
    def time_of_best(self) -> float:
        """When it got there, in seconds."""
        return float(self.times[int(np.argmin(self.history))]) if self.history.size else 0.0

    @property
    def wasted(self) -> float:
        """Fraction of the run spent *after* the score stopped improving.

        The number that catches a controller finishing early and parking. This
        metric is not monotone -- unlike coverage, which can only go up, and
        dirt removed, which can only go up. A robot that sits in one place
        makes its time-averaged distribution worse every second it sits there,
        and this says how much of the cycle went that way.

        Both shipped oracles score badly here, and correctly so: they drive a
        good path, finish, and stop. Coverage cannot see that and this can.
        """
        span = float(self.times[-1] - self.times[0]) if self.times.size > 1 else 0.0
        if span <= 0:
            return 0.0
        return float(max(0.0, (self.times[-1] - self.time_of_best) / span))

    def describe(self) -> str:
        state = "converged" if self.converged else "still improving"
        tail = ""
        if self.value > self.best * 1.1:
            tail = f", then got worse over the last {self.wasted:.0%} of the run"
        return (
            f"ergodic score vs {self.target}: {self.value:.4f} "
            f"(best {self.best:.4f} at {self.time_of_best:.0f}s, {state}{tail})"
        )


def target_measure(
    recording: Recording, target: str = "dirt", *, pool: Any = None
) -> tuple[FloatArray, Any]:
    """The distribution a trajectory is being asked to match.

    ``"dirt"`` uses the dirt field at the start of the run, so the metric asks
    "did you go where the mess was". ``"uniform"`` weights every navigable cell
    equally, which is coverage. ``"remaining"`` uses the dirt still there at
    the end, which scores how much of the *unfinished* work the robot was near
    -- useful for asking whether a controller failed by missing dirt or by
    finding it and not removing it.
    """
    from zimablue.replay.renderer import load_scene

    if pool is None:
        scene = load_scene(recording)
        pool = scene.pool
    else:
        scene = None
    cell = float(recording.manifest.get("cell", 0.1))
    navigable = pool.navigable_mask(cell)

    if target == "uniform":
        density = navigable.astype(float)
    elif target in ("dirt", "remaining"):
        when = 0.0 if target == "dirt" else float(recording.duration)
        density = np.asarray(recording.dirt_at(when), dtype=float)
        if density.shape != navigable.shape:
            raise ValueError(
                f"the dirt raster is {density.shape} and the navigable mask is "
                f"{navigable.shape}; they must come from the same cell size"
            )
        density = np.where(navigable, np.maximum(density, 0.0), 0.0)
    else:
        raise ValueError(f"unknown target {target!r}; use 'dirt', 'remaining' or 'uniform'")

    total = density.sum()
    if total <= 0:
        # A spotless pool has no dirt to weight by, and asking a controller to
        # match a distribution of nothing is not a question. Fall back to
        # uniform and let the caller see it in the label.
        density = navigable.astype(float)
        total = density.sum()
    return density / total, pool


def ergodic_score(
    recording: Recording,
    *,
    target: str = "dirt",
    modes: int = 8,
    samples: int = 240,
    pool: Any = None,
) -> ErgodicScore:
    """Score a recorded run against a target distribution.

    ``modes`` is how many Fourier modes per axis. Eight is plenty: the
    weighting suppresses the fine ones anyway, and the cost is quadratic.
    ``samples`` is how many points along the run the running score is
    evaluated at -- the coefficients are cumulative, so this only decides the
    resolution of the curve, not the answer at the end.
    """
    density, pool = target_measure(recording, target, pool=pool)
    minx, miny, maxx, maxy = pool.boundary.bounds
    span = np.array([max(maxx - minx, 1e-9), max(maxy - miny, 1e-9)])
    origin = np.array([minx, miny])

    frames = recording.frames
    path = np.column_stack(
        [np.asarray(frames["x"], dtype=float), np.asarray(frames["y"], dtype=float)]
    )
    keep = np.isfinite(path).all(axis=1)
    path, times = path[keep], np.asarray(frames["time"], dtype=float)[keep]
    if path.shape[0] < 2:
        raise ValueError("a trajectory needs at least two poses to be scored")

    normalised = np.clip((path - origin) / span, 0.0, 1.0)
    k = np.arange(modes)

    # Basis: separable cosines on the bounding box, each normalised to unit L2
    # norm so a mode's coefficient does not depend on which mode it is.
    norm = np.where(k == 0, 1.0, np.sqrt(0.5))
    cos_x = np.cos(np.pi * np.outer(normalised[:, 0], k)) / norm  # (t, kx)
    cos_y = np.cos(np.pi * np.outer(normalised[:, 1], k)) / norm

    # Trajectory coefficients, cumulative in time: c_k(t) = mean over [0, t].
    products = cos_x[:, :, None] * cos_y[:, None, :]  # (t, kx, ky)
    running = np.cumsum(products, axis=0) / np.arange(1, len(products) + 1)[:, None, None]

    # Target coefficients, over the raster.
    cell = float(recording.manifest.get("cell", 0.1))
    rows, cols = np.nonzero(density > 0)
    weights = density[rows, cols]
    cx = np.clip((minx + (cols + 0.5) * cell - minx) / span[0], 0.0, 1.0)
    cy = np.clip((miny + (rows + 0.5) * cell - miny) / span[1], 0.0, 1.0)
    target_coefficients = (
        (np.cos(np.pi * np.outer(cx, k)) / norm)[:, :, None]
        * (np.cos(np.pi * np.outer(cy, k)) / norm)[:, None, :]
        * weights[:, None, None]
    ).sum(axis=0)

    # Sobolev weights: coarse structure counts for more than fine.
    kx, ky = np.meshgrid(k, k, indexing="ij")
    weight = (1.0 + kx**2 + ky**2) ** (-1.5)

    at = np.unique(np.linspace(0, len(running) - 1, min(samples, len(running))).astype(int))
    history = np.array(
        [float((weight * (running[i] - target_coefficients) ** 2).sum()) for i in at]
    )

    return ErgodicScore(
        value=float(history[-1]),
        history=history,
        times=times[at],
        target=target,
        modes=modes,
        coefficients=running[-1],
        target_coefficients=target_coefficients,
    )
