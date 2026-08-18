"""The Poincaré section, and the periodic orbits it exposes.

A cleaner in a pool is a flow in a bounded domain, and the classical way to
study one is to stop looking at the whole trajectory and look only at where it
crosses a chosen surface.  The natural surface here is the wall: unroll the
pool's boundary to arc length ``s``, record the incidence angle ``theta`` at
every contact, and a twenty-minute run collapses from ninety thousand frames to
a few dozen points on a cylinder.

What that buys is the thing nobody measures.  A **fixed point of the return
map** -- a contact that recurs at the same place and the same angle -- is a
periodic orbit: the robot doing the same loop forever.  If it is *attracting*,
the robot falls into it and never leaves, and coverage stops improving while
every dashboard still says the machine is working.  That is the failure mode
owners actually complain about, and it is invisible in a coverage percentage
until it is far too late.

Everything here reads a finished recording. Contacts are already in the
``.zbr``; this is arithmetic on data we have, not another simulation.

A caveat, stated once and meant. The true state of this system is not
``(s, theta)``. The controller carries an EKF, an occupancy map and a lane
plan, so two contacts that agree on the wall can be followed by completely
different behaviour. A recurrence found here is therefore *evidence* of a
periodic orbit and not proof of one -- which is why the multiplier is
estimated from the data rather than asserted, and why
:meth:`ReturnMap.periodic_orbits` reports how many times a candidate repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = ["Orbit", "ReturnMap", "return_map"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Orbit:
    """A candidate periodic orbit found in the section."""

    period: int
    """How many wall contacts the robot makes before repeating."""

    repeats: int
    """How many times in a row it came back. Two is a coincidence; ten is not."""

    s: float
    """Arc length of the first contact, metres along the perimeter."""

    theta: float
    """Incidence there, radians from the outward normal. 0 is head-on."""

    start_time: float
    duration: float
    multiplier: float
    """How a small disturbance grows over one period.

    Below 1 the orbit is attracting: the robot is drawn onto this loop and
    will not leave without a kick. Above 1 it is repelling, and the run is
    passing through rather than settling. This is a local finite-difference
    estimate from the trajectory, not an eigenvalue of anything analytic.
    """

    @property
    def attracting(self) -> bool:
        return self.multiplier < 1.0

    def describe(self) -> str:
        kind = "attracting" if self.attracting else "repelling"
        return (
            f"period-{self.period} orbit, {self.repeats} repeats, {kind} "
            f"(multiplier {self.multiplier:.2f}), from t={self.start_time:.0f}s "
            f"for {self.duration:.0f}s"
        )


@dataclass
class ReturnMap:
    """Wall contacts as points on a cylinder, plus what they imply."""

    s: FloatArray
    """Arc length along the perimeter at each contact, metres."""

    theta: FloatArray
    """Incidence from the wall's outward normal, radians.

    0 is driving straight into the wall; +/-pi/2 is sliding along it. A cleaner that
    approaches every wall head-on and one that grazes them are doing very
    different things, and this is the axis that separates them.
    """

    time: FloatArray
    perimeter: float
    source: str = ""

    def __len__(self) -> int:
        return int(self.s.size)

    @property
    def rate(self) -> float:
        """Wall contacts per minute. A crude but honest measure of thrashing."""
        span = float(self.time[-1] - self.time[0]) if len(self) > 1 else 0.0
        return 60.0 * len(self) / span if span > 0 else 0.0

    # ------------------------------------------------------------------
    def _normalised(self) -> FloatArray:
        """Points scaled so a distance in ``s`` and one in ``theta`` compare.

        Arc length is metres over a perimeter of tens; the angle is radians
        over pi. Comparing them raw means the angle contributes nothing and
        every recurrence test is really a test on position alone.
        """
        return np.column_stack([self.s / max(self.perimeter, 1e-9), self.theta / np.pi])

    def separation(self, i: int, j: int) -> float:
        """Distance between two contacts on the cylinder, wrapping in ``s``."""
        points = self._normalised()
        ds = abs(points[i, 0] - points[j, 0])
        ds = min(ds, 1.0 - ds)  # the perimeter is a loop
        return float(np.hypot(ds, points[i, 1] - points[j, 1]))

    # ------------------------------------------------------------------
    def recurrence_matrix(self, tolerance: float = 0.03) -> NDArray[np.bool_]:
        """``R[i, j]`` -- did contact *j* land where contact *i* did?

        The recurrence plot of nonlinear time-series analysis. Diagonal stripes
        parallel to the main diagonal are periodicity; a solid block is the
        trajectory sitting still; scattered dots are a system that is mixing.
        """
        points = self._normalised()
        ds = np.abs(points[:, 0][:, None] - points[:, 0][None, :])
        ds = np.minimum(ds, 1.0 - ds)
        dtheta = points[:, 1][:, None] - points[:, 1][None, :]
        return np.hypot(ds, dtheta) < tolerance

    def periodic_orbits(
        self,
        tolerance: float = 0.03,
        *,
        max_period: int = 12,
        min_repeats: int = 3,
    ) -> list[Orbit]:
        """Candidate closed loops, longest-lived first.

        For each period *k*, look for runs of consecutive contacts where the
        section point at *i + k* lands within ``tolerance`` of the one at *i*.
        A run of length *r* means the robot went round the same loop *r* times.
        """
        if len(self) < 4:
            return []
        points = self._normalised()
        found: list[Orbit] = []

        for period in range(1, min(max_period, len(self) - 1) + 1):
            ds = np.abs(points[period:, 0] - points[:-period, 0])
            ds = np.minimum(ds, 1.0 - ds)
            close = np.hypot(ds, points[period:, 1] - points[:-period, 1]) < tolerance

            for start, length in _runs(close):
                if length < min_repeats:
                    continue
                stop = start + length + period - 1
                found.append(
                    Orbit(
                        period=period,
                        repeats=int(length),
                        s=float(self.s[start]),
                        theta=float(self.theta[start]),
                        start_time=float(self.time[start]),
                        duration=float(self.time[min(stop, len(self) - 1)] - self.time[start]),
                        multiplier=self._multiplier(start, stop, period),
                    )
                )

        # A period-6 orbit also satisfies the period-12 test. Keep the shortest
        # period that explains a stretch of the run, or every real loop is
        # reported once per multiple of itself.
        found.sort(key=lambda o: (o.period, -o.repeats))
        kept: list[Orbit] = []
        for orbit in found:
            window = (orbit.start_time, orbit.start_time + orbit.duration)
            if any(_overlaps(window, (k.start_time, k.start_time + k.duration)) for k in kept):
                continue
            kept.append(orbit)
        return sorted(kept, key=lambda o: -o.duration)

    def _multiplier(self, start: int, stop: int, period: int) -> float:
        """How the distance from the orbit grows over one period.

        Finite differences: take how far each contact sits from the one a
        period earlier, and see whether that gap shrinks or grows as the run
        proceeds. A geometric fit through the gaps is the multiplier.
        """
        gaps = np.array(
            [
                self.separation(i, i + period)
                for i in range(start, min(stop, len(self) - period))
                if i + period < len(self)
            ]
        )
        gaps = gaps[gaps > 1e-9]
        if gaps.size < 3:
            return 1.0
        # Fit log(gap) ~ a + b * n; the multiplier is exp(b).
        slope = float(np.polyfit(np.arange(gaps.size), np.log(gaps), 1)[0])
        return float(np.exp(np.clip(slope, -5.0, 5.0)))

    def trapped_fraction(self, tolerance: float = 0.03, **kwargs: Any) -> float:
        """Fraction of the run spent on an attracting periodic orbit.

        The headline number. If a fifth of a cleaning cycle is spent going
        round the same loop, that is a fifth of the battery spent covering
        ground already covered, and no coverage metric will say so.
        """
        span = float(self.time[-1] - self.time[0]) if len(self) > 1 else 0.0
        if span <= 0:
            return 0.0
        trapped = sum(o.duration for o in self.periodic_orbits(tolerance, **kwargs) if o.attracting)
        return float(min(trapped / span, 1.0))


def return_map(
    recording: Recording,
    *,
    pool: Any = None,
    debounce: float = 1.5,
    min_travel: float = 0.6,
) -> ReturnMap:
    """Build the Poincaré section from a recorded run.

    A contact is taken at the *rising edge* of the bump switches -- the moment
    the robot arrives at the wall. Using every frame in contact instead would
    count a robot leaning on a wall for three seconds as a hundred and fifty
    arrivals, and turn every wall-follow into a false periodic orbit.

    Rising edges alone are not enough either, and this is not a detail. In a
    real run, 69% of them fall less than half a second after the previous one:
    the robot bumps, backs off a few centimetres, and bumps again. Those are
    one arrival, and treating them as eight produced a section full of
    "period-1 orbits lasting zero seconds" -- chatter dressed up as dynamics.

    So a contact is new if it is more than ``debounce`` seconds after the last
    one *or* more than ``min_travel`` metres away along the perimeter. The
    second clause matters in a narrow channel, where the robot really does
    touch alternating walls a fraction of a second apart and both are real.
    """
    from zimablue.replay.renderer import load_scene

    frames = recording.frames
    if "contacts" not in frames:
        raise KeyError("this recording has no contact channel, so it has no wall crossings")

    pool = pool if pool is not None else load_scene(recording).pool
    touching = np.asarray(frames["contacts"], dtype=int) > 0
    rising = np.flatnonzero(touching & ~np.concatenate([[False], touching[:-1]]))
    rising = _debounce(
        rising,
        np.asarray(frames["time"], dtype=float),
        np.asarray(frames["x"], dtype=float),
        np.asarray(frames["y"], dtype=float),
        pool,
        debounce=debounce,
        min_travel=min_travel,
    )
    if rising.size == 0:
        return ReturnMap(
            s=np.zeros(0),
            theta=np.zeros(0),
            time=np.zeros(0),
            perimeter=pool.perimeter_length,
            source=str(recording.manifest.get("scenario", {}).get("name", "")),
        )

    x = np.asarray(frames["x"], dtype=float)[rising]
    y = np.asarray(frames["y"], dtype=float)[rising]
    heading = np.asarray(frames["heading"], dtype=float)[rising]

    ring = pool.boundary.exterior
    s = np.array([pool.project_to_perimeter(px, py) for px, py in zip(x, y, strict=True)])

    # Incidence angle against the local wall tangent, from a short chord of the
    # boundary either side of the contact. Differencing the ring's own vertices
    # would give whatever resolution the polygon happens to have, which for the
    # kidney is 256 points and for a rectangle is 4.
    step = max(pool.perimeter_length * 0.002, 1e-3)
    before = np.array([ring.interpolate((v - step) % ring.length).coords[0] for v in s])
    after = np.array([ring.interpolate((v + step) % ring.length).coords[0] for v in s])
    tangent = np.arctan2(after[:, 1] - before[:, 1], after[:, 0] - before[:, 0])

    # Incidence measured from the wall's *outward* normal: 0 is driving
    # straight into it, +/-90 degrees is sliding along it. Two earlier
    # conventions were both unreadable -- against the tangent, every contact
    # landed in the same half of the plot; against the inward normal, they all
    # sat near 180 degrees, because a robot arriving at a wall is by definition
    # pointing out of the pool.
    outward = tangent - np.pi / 2 if _counterclockwise(ring) else tangent + np.pi / 2
    offset = heading - outward
    theta = np.arctan2(np.sin(offset), np.cos(offset))
    return ReturnMap(
        s=s,
        theta=theta,
        time=np.asarray(frames["time"], dtype=float)[rising],
        perimeter=float(pool.perimeter_length),
        source=str(recording.manifest.get("scenario", {}).get("controller", "")),
    )


def _counterclockwise(ring: Any) -> bool:
    """Which way the boundary is wound, so the normal points out of the pool.

    Shapely does not promise an orientation, and getting it backwards flips
    every incidence angle by 180 degrees -- which looks plausible and is wrong.
    """
    coords = np.asarray(ring.coords)
    x, y = coords[:, 0], coords[:, 1]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))) < 0


def _debounce(
    indices: NDArray[np.int64],
    time: FloatArray,
    x: FloatArray,
    y: FloatArray,
    pool: Any,
    *,
    debounce: float,
    min_travel: float,
) -> NDArray[np.int64]:
    """Collapse a burst of re-triggers into the one arrival it really is."""
    if indices.size == 0 or debounce <= 0:
        return indices
    kept = [int(indices[0])]
    last_time = float(time[indices[0]])
    last_s = pool.project_to_perimeter(float(x[indices[0]]), float(y[indices[0]]))
    perimeter = float(pool.perimeter_length)

    for index in indices[1:]:
        now = float(time[index])
        here = pool.project_to_perimeter(float(x[index]), float(y[index]))
        along = abs(here - last_s)
        along = min(along, perimeter - along)
        if now - last_time >= debounce or along >= min_travel:
            kept.append(int(index))
            last_time, last_s = now, here
    return np.asarray(kept, dtype=np.int64)


def _runs(flags: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Start index and length of every run of ``True``."""
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b - a)) for a, b in zip(starts, stops, strict=True)]


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
