"""Two timescales: a fast robot eating a slow field.

The cleaner moves at 0.3 m/s and the dirt field changes over tens of minutes.
That separation is the classical setting for averaging theory, and it makes a
prediction worth testing: on the slow timescale, what matters is not the
robot's exact path but the *fraction of time it spends in each place*.  Two
controllers with the same occupancy density should clean at the same rate
however differently they drive.

Take it literally.  Cell *i* holds mass ``m_i``; the robot passes over it at
some rate ``rho_i`` visits per second; each pass removes a fixed fraction.
Then::

    dm_i/dt = -eta * rho_i * m_i        =>      m_i(t) = m_i(0) * exp(-eta rho_i t)

and total dirt is a sum of exponentials with a *spread of rates*, one per cell.
That spread is the whole shape of a cleaning curve: the well-served cells empty
in the first few minutes and the rest of the cycle is spent waiting on the
badly-served ones. A single exponential cannot produce the long tail every
cleaning curve has; a distribution of them does it naturally.

Which is useful in two ways.  Fit ``eta`` from the first few minutes and you
can **predict the rest of the cycle** without simulating it -- if the
prediction holds, the occupancy density is a sufficient statistic and every
question about cleaning reduces to a question about where the robot spends its
time.  And it gives the greedy-versus-systematic result a mechanism:
``dirt_oracle`` chases the fast variable and wins early, ``baseline_coverage``
shapes the occupancy density and wins late.

Where it will break, stated in advance. Adhered dirt needs brush agitation and
does not come off at a constant fraction per pass, and the filter fills, and
fines pass through the mesh and settle again. So the fitted ``eta`` is an
effective rate absorbing all of that, and a prediction that holds is evidence
the lumping is fair over the horizon tested -- not that the mechanism is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["CleaningForecast", "forecast_cleaning", "occupancy"]

FloatArray = NDArray[np.float64]


def occupancy(
    recording: Recording,
    *,
    until: float | None = None,
    cell: float | None = None,
    swath: float | None = None,
    pool: Any = None,
) -> FloatArray:
    """Seconds per cell the cleaning head spent over it, up to ``until``.

    The head is a disc, not a point, so a pose contributes to every cell within
    half a swath of it. Counting only the cell under the centre would
    understate the rate by the ratio of the swath area to a cell -- about a
    factor of ten at this resolution, which would be absorbed into the fitted
    removal rate and make it meaningless as a physical number.
    """
    from zimablue.replay.renderer import load_scene

    if pool is None:
        scene = load_scene(recording)
        pool = scene.pool
    else:
        scene = None
    cell = cell if cell is not None else float(recording.manifest.get("cell", 0.1))
    if swath is None:
        swath = float(scene.swath) if scene is not None else 0.34

    grid = pool.grid(cell)
    frames = recording.frames
    time = np.asarray(frames["time"], dtype=float)
    stop = len(time) if until is None else int(np.searchsorted(time, until, "right"))
    stop = max(stop, 2)

    x = np.asarray(frames["x"], dtype=float)[:stop]
    y = np.asarray(frames["y"], dtype=float)[:stop]
    dt = float(recording.frame_dt)

    seconds = np.zeros((grid.nrows, grid.ncols), dtype=float)
    radius = max(round(0.5 * swath / cell), 0)
    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if np.hypot(dr, dc) * cell <= 0.5 * swath
    ]
    rows = np.clip(((y - grid.miny) / cell).astype(int), 0, grid.nrows - 1)
    cols = np.clip(((x - grid.minx) / cell).astype(int), 0, grid.ncols - 1)
    for dr, dc in offsets:
        np.add.at(
            seconds,
            (np.clip(rows + dr, 0, grid.nrows - 1), np.clip(cols + dc, 0, grid.ncols - 1)),
            dt,
        )
    return seconds


@dataclass
class CleaningForecast:
    """A prediction of the rest of the cycle, made from the start of it."""

    times: FloatArray
    predicted: FloatArray
    """Grams of dirt remaining, predicted."""

    actual: FloatArray
    fitted_from: float
    """Seconds of the run used to fit. Everything after is a genuine forecast."""

    rate: float
    """Effective removal rate, per second of head-over-cell contact.

    An effective number, not a physical one: it absorbs adhesion, brush
    agitation, filter losses and resuspension into a single constant.
    """

    @property
    def error(self) -> FloatArray:
        """Predicted minus actual, grams."""
        return self.predicted - self.actual

    @property
    def forecast_error(self) -> float:
        """Mean absolute error over the forecast region only, as a fraction.

        Fitting the first five minutes and then reporting the error over those
        same five minutes measures the fit, not the forecast. This measures
        only the part the model had not seen.
        """
        ahead = self.times > self.fitted_from
        if not ahead.any():
            return 0.0
        initial = max(float(self.actual[0]), 1e-9)
        return float(np.mean(np.abs(self.error[ahead])) / initial)

    def describe(self) -> str:
        return (
            f"fitted on the first {self.fitted_from / 60:.0f} min, "
            f"forecast error {self.forecast_error:.1%} of the initial load "
            f"(rate {self.rate:.4f}/s)"
        )


def forecast_cleaning(
    recording: Recording,
    *,
    fit_fraction: float = 0.25,
    samples: int = 60,
    pool: Any = None,
) -> CleaningForecast:
    """Fit a removal rate on the first part of a run and predict the rest.

    The test of the whole averaging idea. If a rate fitted on the first quarter
    predicts the remaining three quarters, then the occupancy density really is
    doing the work and the exact trajectory is a detail.

    The occupancy used for the forecast is the one measured over the *fitting
    window*, extrapolated linearly. That is the assumption being tested along
    with the rate: that the robot goes on distributing its time the way it
    started. A controller that changes strategy halfway through -- a coverage
    pass then a spot-clean -- should break this, and if it does, that is the
    measurement working.
    """
    duration = float(recording.duration)
    if duration <= 0:
        raise ValueError("this recording has no duration to forecast over")
    fit_until = max(duration * float(np.clip(fit_fraction, 0.05, 0.9)), recording.frame_dt * 10)

    dirt0 = np.asarray(recording.dirt_at(0.0), dtype=float)
    if dirt0.size == 0:
        raise ValueError("this recording has no dirt keyframes, so there is nothing to forecast")

    seconds = occupancy(recording, until=fit_until, pool=pool)
    if seconds.shape != dirt0.shape:
        raise ValueError(f"occupancy is {seconds.shape} and the dirt raster is {dirt0.shape}")
    # Per-second occupancy rate, assumed to continue at the same density.
    density = seconds / fit_until

    times = np.linspace(0.0, duration, samples)
    actual = np.array([float(np.asarray(recording.dirt_at(t)).sum()) for t in times])

    def total(rate: float, t: float) -> float:
        return float((dirt0 * np.exp(-rate * density * t)).sum())

    # One unknown, monotone in it: bisect rather than reach for an optimiser.
    fit_at = times[times <= fit_until]
    target = np.array([float(np.asarray(recording.dirt_at(t)).sum()) for t in fit_at])
    low, high = 1e-6, 10.0
    for _ in range(60):
        mid = np.sqrt(low * high)
        residual = sum(total(mid, t) - m for t, m in zip(fit_at, target, strict=True))
        if residual > 0:  # predicting too much dirt left: remove faster
            low = mid
        else:
            high = mid
    rate = float(np.sqrt(low * high))

    return CleaningForecast(
        times=times,
        predicted=np.array([total(rate, t) for t in times]),
        actual=actual,
        fitted_from=float(fit_until),
        rate=rate,
    )
