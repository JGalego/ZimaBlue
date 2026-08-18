"""How fast two nearly identical runs stop being identical.

Start the same robot in the same pool a millimetre apart and watch the gap.  If
it grows exponentially the system is chaotic, and the rate is a Lyapunov
exponent.  This matters here for a reason that has nothing to do with the word
"chaos" and everything to do with what a simulator is for.

**It bounds how far a prediction means anything.** A rollout is useful for
about ``1 / lambda`` seconds; past that the trajectory is a plausible sample
from the right distribution and not a forecast. Anyone training a policy on
this backend, or planning against a model of it, is entitled to know that
number rather than guessing.

**It measures the controller, not the pool.** Feed the same pool to different
controllers and the divergence rate is a property of the autonomy stack. The
expectation is a tension rather than a ranking: chaos mixes, so a *high* rate
should go with *better* coverage and *worse* repeatability. Whether that
tension shows up is the interesting part.

Two things make this cheap and honest here. The simulator is bit-reproducible,
so a run repeated with the same seed is identical to the last bit and any
divergence at all comes from the perturbation. And the perturbation can be
applied where it belongs -- to the initial pose -- rather than to the seed,
which would change the noise realisation and measure something else entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Divergence", "divergence"]

FloatArray = NDArray[np.float64]


@dataclass
class Divergence:
    """Separation between a reference run and its perturbed twins."""

    time: FloatArray
    separation: FloatArray
    """``(runs, samples)`` metres between each twin and the reference."""

    epsilon: float
    """How far apart they started, metres.

    There is a floor on what is worth asking for. Recordings store positions as
    float32, so separations below about a micron cannot be read back out of
    one at all.
    """

    pool_scale: float
    """A length the pool provides -- its diameter. Separation saturates here,
    because two robots in the same pool cannot get further apart than the pool,
    and the fit has to stop before it does."""

    @property
    def typical(self) -> FloatArray:
        """Median separation across twins, sample by sample.

        Median rather than mean, and it matters. The twins are not a tight
        bundle: pushed a millimetre along a lane the controller absorbs it,
        pushed across one it compounds, so the outcomes come out bimodal --
        some twins glued to the reference at 10 microns and others out at the
        far wall. A geometric mean of those reports a middle value that no twin
        ever occupied.
        """
        return np.median(self.separation, axis=0)

    def exponents(self, *, ceiling: float = 0.25) -> FloatArray:
        """One growth rate per twin, fitted before saturation.

        ``ceiling`` is the fraction of :attr:`pool_scale` at which each fit
        stops. Fitting past it measures the pool's diameter rather than the
        dynamics: the curve flattens because there is nowhere further to go,
        and a straight line through the flat part reads much shallower than
        the truth.
        """
        limit = ceiling * self.pool_scale
        rates = []
        for trace in self.separation:
            usable = np.flatnonzero((trace < limit) & (trace > self.epsilon))
            if usable.size < 5:
                rates.append(0.0)
                continue
            stop = usable[-1] + 1
            # A twin that starts exactly on the reference has zeros in it, and
            # log(0) is not a growth rate. Floor at the recording's own
            # precision rather than dropping the samples, which would shorten
            # the window the slope is fitted over.
            window = np.maximum(trace[:stop], 1e-9)
            rates.append(float(np.polyfit(self.time[:stop], np.log(window), 1)[0]))
        return np.asarray(rates, dtype=float)

    def exponent(self, **kwargs: float) -> float:
        """The median twin's growth rate, in inverse seconds."""
        rates = self.exponents(**kwargs)
        return float(np.median(rates)) if rates.size else 0.0

    @property
    def diverged(self) -> float:
        """Fraction of twins that ended up a quarter of a pool apart.

        The most robust thing this measurement produces, and the one to quote:
        it answers "does a millimetre matter?" with a probability rather than
        with a rate fitted to a bimodal sample.
        """
        if not self.separation.size:
            return 0.0
        reached = (self.separation > 0.25 * self.pool_scale).any(axis=1)
        return float(reached.mean())

    def time_to_diverge(self) -> float:
        """Median seconds for a twin to get a quarter of a pool away.

        ``inf`` if fewer than half of them ever do.
        """
        limit = 0.25 * self.pool_scale
        times = []
        for trace in self.separation:
            past = np.flatnonzero(trace > limit)
            times.append(float(self.time[past[0]]) if past.size else float("inf"))
        return float(np.median(times)) if times else float("inf")

    def horizon(self, **kwargs: float) -> float:
        """Seconds before a millimetre of uncertainty becomes a metre.

        The exponent in units somebody can use. Beyond this, a rollout of this
        simulator is a sample and not a prediction, and a plan built on one is
        planning against a coin.
        """
        rate = self.exponent(**kwargs)
        if rate <= 0:
            return float("inf")
        return float(np.log(1.0 / self.epsilon) / rate)

    def describe(self) -> str:
        rate = self.exponent()
        when = self.time_to_diverge()
        spread = f"{self.diverged:.0%} of twins ended a quarter-pool apart" + (
            f", median after {when:.0f} s" if np.isfinite(when) else ""
        )
        if rate <= 0:
            return f"median twin did not diverge; {spread}"
        return f"lambda = {rate:.4f}/s, horizon {self.horizon():.0f} s; {spread}"


def divergence(
    *,
    epsilon: float = 1e-3,
    runs: int = 6,
    minutes: float = 20.0,
    samples: int = 400,
    seed: int = 0,
    **simulation: Any,
) -> Divergence:
    """Run one reference and ``runs`` twins started ``epsilon`` metres away.

    Extra keyword arguments go straight to :class:`~zimablue.Simulation`, so
    this reads as an ordinary run with a sensitivity question attached::

        divergence(pool="kidney", controller="baseline_coverage", minutes=20)

    The twins are displaced in a ring of directions around the reference start,
    rather than all in one direction, because a robot pushed sideways off a
    lane and one pushed along it are not perturbed by the same amount in any
    sense the controller cares about.
    """
    import zimablue as zb

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if runs < 1:
        raise ValueError(f"need at least one twin, got {runs}")

    simulation.setdefault("pool", "kidney")
    simulation.setdefault("dirt", "light_sediment")
    baseline = zb.Simulation(seed=seed, record=True, **simulation)
    # The robot's *actual* start, not the first recorded frame. Frame zero is
    # written after the first step, so it is already a tick downstream -- and
    # it is float32, so reading the start pose off it displaced every twin by
    # a third of a millimetre before the perturbation was even applied.
    start = baseline.start_pose
    base = baseline.run(minutes=minutes).recording
    if base is None:  # pragma: no cover - record=True is forced above
        raise RuntimeError("the reference run produced no recording to compare against")
    x0, y0 = base.frames["x"], base.frames["y"]

    requested = simulation["pool"]
    pool = zb.make_pool(requested) if isinstance(requested, str) else requested
    minx, miny, maxx, maxy = pool.boundary.bounds
    pool_scale = float(np.hypot(maxx - minx, maxy - miny))

    angles = np.linspace(0.0, 2 * np.pi, runs, endpoint=False)
    traces: list[FloatArray] = []
    length = 0

    for angle in angles:
        pose = (
            start[0] + epsilon * float(np.cos(angle)),
            start[1] + epsilon * float(np.sin(angle)),
            start[2],
        )
        try:
            twin = zb.Simulation(seed=seed, start_pose=pose, record=True, **simulation).run(
                minutes=minutes
            )
        except ValueError:
            # The displaced start landed outside the navigable pool. Skip it
            # rather than nudging it back in, which would silently change the
            # perturbation size this whole measurement is scaled by.
            continue
        if twin.recording is None:  # pragma: no cover - record=True is forced above
            continue
        frames = twin.recording.frames
        n = min(len(frames["x"]), len(x0))
        traces.append(np.hypot(frames["x"][:n] - x0[:n], frames["y"][:n] - y0[:n]))
        length = n if length == 0 else min(length, n)

    if not traces:
        raise ValueError(
            f"every perturbed start at epsilon={epsilon} m fell outside the pool; "
            "use a smaller epsilon or a start further from the wall"
        )

    at = np.unique(np.linspace(0, length - 1, min(samples, length)).astype(int))
    return Divergence(
        time=np.asarray(base.frames["time"], dtype=float)[at],
        separation=np.array([trace[:length][at] for trace in traces], dtype=float),
        epsilon=float(epsilon),
        pool_scale=pool_scale,
    )


def compare(controllers: Sequence[str], **kwargs: Any) -> dict[str, Divergence]:
    """Divergence for several controllers over the same pool and seed."""
    return {name: divergence(controller=name, **kwargs) for name in controllers}
