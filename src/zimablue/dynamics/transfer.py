"""The transfer operator: where the robot ends up, and how fast it forgets.

Chop the pool into cells and ask, for a lag ``tau``, how often a robot in cell
*i* is in cell *j* a moment ``tau`` later.  That matrix is a discretisation of
the Perron-Frobenius operator, and its spectrum answers three questions that
coverage percentages cannot.

**Where does it spend its time?** The leading eigenvector is the invariant
measure -- the occupancy density the run converges to. Not where the robot has
been, which is what a visit grid shows, but where it is *heading*, which is
what decides the rest of the cycle.

**How fast does it mix?** The second eigenvalue governs how quickly an
initial distribution relaxes to that invariant measure. The spectral gap
``1 - |lambda_2|`` is, literally, how fast a controller homogenises its
coverage. It is the number that turns "this controller is thorough" from an
impression into a rate.

**Where does it get stuck?** The eigenvectors just below the leading one pick
out *almost-invariant sets* -- regions the robot enters and rarely leaves.
This is the part that is a diagnosis rather than a score. On an L-shaped pool
it finds, without being told the pool is L-shaped, that the robot seldom
crosses between the arms.

Following Dellnitz & Junge's set-oriented numerics and Froyland's
almost-invariant sets, but estimated from trajectories rather than from a
known map, because a trajectory is what a recording contains.

One honest limitation. A Markov model assumes the next cell depends only on
the current one, and this robot has memory -- a lane plan, an occupancy map, a
heading. So the operator is a projection of a much larger system onto its
spatial coordinates, and the mixing rate it reports is the mixing rate *of that
projection*. It is a useful summary, not a complete description, and a run long
enough to visit each cell many times is what keeps it honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.recording import Recording

__all__ = ["TransferOperator", "transfer_operator"]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class TransferOperator:
    """A cell-to-cell transition matrix, and what its spectrum says."""

    matrix: FloatArray
    """Row-stochastic ``(n, n)``. Row *i* is where the robot goes from cell *i*."""

    centres: FloatArray
    """``(n, 2)`` cell centres in metres, for plotting and for naming a region."""

    counts: FloatArray
    """Raw visit counts per cell, before normalisation. Low counts mean a row
    of the matrix is guesswork, and :attr:`weak_rows` says how many."""

    cell: float
    lag: float
    """Seconds between the two samples of each transition."""

    shape: tuple[int, int]
    """Rows and columns of the grid the cells were cut from."""

    index: NDArray[np.int64]
    """``(rows, cols)`` grid of cell numbers, ``-1`` outside the pool or in a
    cell the robot never reached. Lets an analysis be put back on the floor it
    came from."""

    observed: float = 0.0
    """Total seconds of trajectory the matrix was estimated from."""

    unvisited: int = 0
    """Navigable cells the robot never entered, and which are therefore not in
    the matrix at all. A large number here means the spectrum describes a
    smaller pool than the one on the drawing."""

    source: str = ""

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def weak_rows(self) -> int:
        """Cells visited too rarely for their row to mean anything."""
        return int((self.counts < 5).sum())

    def eigen(self) -> tuple[ComplexArray, ComplexArray]:
        """Eigenvalues sorted by magnitude, descending, with their vectors.

        Complex, and not as a formality: a transfer matrix is not symmetric,
        and a controller that circulates around the pool puts a conjugate pair
        in the spectrum rather than a real eigenvalue. Everything downstream
        already took a real part or a magnitude; only the annotation claimed
        otherwise, and older numpy stubs let it.
        """
        values, vectors = np.linalg.eig(self.matrix.T)
        order = np.argsort(-np.abs(values))
        return values[order], vectors[:, order]

    def invariant_measure(self) -> FloatArray:
        """The occupancy density the run converges to, summing to one.

        The leading left eigenvector. Compare it against the visit grid: the
        visit grid is where the robot *has* been, and this is where it will be
        if the run continues. When they disagree the run has not converged, and
        the cleaning schedule is being decided by transients.
        """
        _, vectors = self.eigen()
        measure = np.abs(np.real(vectors[:, 0]))
        total = measure.sum()
        return measure / total if total > 0 else measure

    @property
    def eigenvalues(self) -> ComplexArray:
        return self.eigen()[0]

    @property
    def spectral_gap(self) -> float:
        """``1 - |lambda_2|``. How fast the controller homogenises coverage.

        Near 1: a fresh start is forgotten within one lag, and the robot is
        everywhere at once in a statistical sense. Near 0: the run is still
        carrying its initial condition around, which for a cleaner means the
        corner it started in is being cleaned and the far end is not.
        """
        values = self.eigenvalues
        return float(1.0 - abs(values[1])) if values.size > 1 else 1.0

    @property
    def mixing_time(self) -> float:
        """Seconds for a disturbance to decay by ``1/e``. ``inf`` if it never does.

        The spectral gap in units anybody can act on: a mixing time longer than
        the battery lasts means the pool never gets an even pass, whatever the
        coverage number says at the end.

        Check :attr:`reliable` before quoting it. A mixing time several times
        longer than the trajectory it was estimated from is an extrapolation,
        not a measurement -- the run never showed the operator what happens on
        that timescale.
        """
        values = self.eigenvalues
        if values.size < 2:
            return float("inf")
        second = abs(values[1])
        if second >= 1.0 - 1e-12:
            return float("inf")
        return float(-self.lag / np.log(max(second, 1e-12)))

    @property
    def reliable(self) -> bool:
        """Was the run long enough for the mixing time to mean anything?

        The estimate needs the trajectory to have actually mixed at least once.
        A twenty-five-minute run reporting a two-hour mixing time is telling
        you it never got there -- which is itself worth knowing, and is not the
        same as knowing the mixing time is two hours.
        """
        mixing = self.mixing_time
        if not np.isfinite(mixing):
            return False
        return self.observed >= 2.0 * mixing and self.weak_rows < 0.2 * len(self)

    # ------------------------------------------------------------------
    def almost_invariant_sets(self, count: int = 2) -> NDArray[np.int64]:
        """Split the pool into regions the robot rarely moves between.

        The sign structure of the sub-leading eigenvectors. An eigenvalue close
        to 1 that is *not* the leading one means there is a second, nearly
        conserved quantity, and its eigenvector says what: the cells it splits
        into positive and negative are two regions with little traffic across
        the boundary.

        Note what it finds, because it is not what you might expect. On the
        mushroom pool the geometric neck is at y = 3.25 and the partition falls
        near y = 2.7 -- lower, inside the stem. That is correct: the robot
        moves through the top of the stem freely and it is the bottom it cannot
        leave. This looks for where the traffic is thin, which is a different
        place from where the walls are.

        Returns a label per cell. Labels are arbitrary; the partition is the
        result.
        """
        if count < 2 or len(self) < count:
            return np.zeros(len(self), dtype=np.int64)
        _, vectors = self.eigen()
        # Skip the leading vector: it is the invariant measure and is positive
        # everywhere, so it separates nothing.
        features = np.real(vectors[:, 1:count])
        labels = np.zeros(len(self), dtype=np.int64)
        for column in range(features.shape[1]):
            labels = labels * 2 + (features[:, column] > 0).astype(np.int64)
        # Renumber so labels run 0..k-1 whatever the sign structure produced.
        _, labels = np.unique(labels, return_inverse=True)
        return labels.astype(np.int64)

    def leak_rate(self, labels: NDArray[np.int64]) -> dict[int, float]:
        """Per region, the share of transitions that leave it.

        The number behind the picture. A region with a 2% leak rate is one the
        robot escapes about once every fifty lags -- for a 10-second lag, once
        every eight minutes, which on a 30-minute cycle means it visits the
        rest of the pool three times.
        """
        rates: dict[int, float] = {}
        for label in np.unique(labels):
            inside = labels == label
            weight = self.counts[inside]
            if weight.sum() <= 0:
                rates[int(label)] = 0.0
                continue
            outward = self.matrix[np.ix_(inside, ~inside)].sum(axis=1)
            rates[int(label)] = float(np.average(outward, weights=weight))
        return rates

    def to_grid(self, values: FloatArray, fill: float = np.nan) -> FloatArray:
        """Put a per-cell quantity back on the pool's raster, for drawing."""
        grid = np.full(self.shape, fill, dtype=float)
        mask = self.index >= 0
        grid[mask] = values[self.index[mask]]
        return grid

    def summary(self) -> str:
        gap = self.spectral_gap
        mixing = self.mixing_time
        mixing_text = "never" if not np.isfinite(mixing) else f"{mixing:.0f} s"
        if not self.reliable and np.isfinite(mixing):
            mixing_text = f">{self.observed / 2:.0f} s (longer than the run can show)"
        missed = f", {self.unvisited} never reached" if self.unvisited else ""
        return (
            f"{len(self)} cells at {self.cell:.2f} m, lag {self.lag:.0f} s{missed} | "
            f"gap {gap:.3f}, mixing {mixing_text}"
        )


def transfer_operator(
    recordings: Recording | Sequence[Recording],
    *,
    cell: float = 0.75,
    lag: float = 10.0,
    pool: Any = None,
) -> TransferOperator:
    """Estimate the operator from one recording or a batch of them.

    ``cell`` is much coarser than the dirt raster on purpose. The matrix has a
    row per cell and needs many transitions out of each to be worth anything,
    so at 10 cm a half-hour run would give a matrix of twelve thousand rows and
    about two samples each -- noise with an eigendecomposition. Three quarters
    of a metre gives tens of cells and hundreds of samples.

    Pass a list to pool several seeds. That is the honest way to use this: one
    trajectory is one sample path, and the operator is a property of the
    controller, not of the path.
    """
    from zimablue.replay.renderer import load_scene

    runs: list[Recording] = [recordings] if isinstance(recordings, Recording) else list(recordings)
    if not runs:
        raise ValueError("no recordings to estimate an operator from")

    first = runs[0]
    pool = pool if pool is not None else load_scene(first).pool
    grid = pool.grid(cell)
    navigable = pool.navigable_mask(cell)

    index = np.full(navigable.shape, -1, dtype=np.int64)
    index[navigable] = np.arange(int(navigable.sum()))
    n = int(navigable.sum())
    if n < 4:
        raise ValueError(f"only {n} navigable cells at cell={cell} m; use a finer cell")

    matrix = np.zeros((n, n), dtype=float)
    counts = np.zeros(n, dtype=float)
    observed = 0.0

    for run in runs:
        frames = run.frames
        step = max(round(lag / max(run.frame_dt, 1e-9)), 1)
        rows = np.clip(
            ((np.asarray(frames["y"], dtype=float) - grid.miny) / cell).astype(int),
            0,
            grid.nrows - 1,
        )
        cols = np.clip(
            ((np.asarray(frames["x"], dtype=float) - grid.minx) / cell).astype(int),
            0,
            grid.ncols - 1,
        )
        cells = index[rows, cols]
        # A pose can land outside the navigable mask -- the hull overlaps a
        # wall by a centimetre during a contact. Drop those rather than
        # snapping them to a neighbour and inventing a transition.
        source, target = cells[:-step], cells[step:]
        valid = (source >= 0) & (target >= 0)
        np.add.at(matrix, (source[valid], target[valid]), 1.0)
        np.add.at(counts, source[valid], 1.0)
        observed += float(run.duration)

    # Cells the robot never entered are dropped, not kept as self-loops. This
    # is not tidying: an absorbing state contributes an eigenvalue of exactly
    # 1, so twenty-one unvisited cells in an L-shaped pool put twenty-one
    # spurious ones at the top of the spectrum and the real second eigenvalue
    # -- the whole point of the exercise -- disappeared underneath them. The
    # operator describes the dynamics that were *sampled*; where the robot
    # never went, there is no evidence to describe.
    visited = counts > 0
    unvisited = int((~visited).sum())
    if int(visited.sum()) < 4:
        raise ValueError(
            f"the robot reached only {int(visited.sum())} cells at cell={cell} m; "
            "either the run is too short or the cell size is too small"
        )
    matrix = matrix[np.ix_(visited, visited)]
    counts = counts[visited]
    # Dropping columns can empty a row whose only destination was a cell that
    # is itself unvisited-as-a-source. Rare, but it makes the matrix
    # substochastic; give those rows a self-loop rather than a zero row.
    empty = matrix.sum(axis=1) == 0
    matrix[empty, np.flatnonzero(empty)] = 1.0
    matrix /= matrix.sum(axis=1, keepdims=True)

    # Renumber the grid index so cell k of the matrix is cell k on the floor.
    renumber = np.full(n, -1, dtype=np.int64)
    renumber[np.flatnonzero(visited)] = np.arange(int(visited.sum()))
    reindexed = np.where(index >= 0, renumber[np.clip(index, 0, None)], -1)

    flat = np.flatnonzero(navigable.ravel())[visited]
    scenario = first.manifest.get("scenario", {})
    return TransferOperator(
        matrix=matrix,
        centres=np.column_stack(
            [
                grid.minx + (flat % grid.ncols + 0.5) * cell,
                grid.miny + (flat // grid.ncols + 0.5) * cell,
            ]
        ),
        counts=counts,
        cell=cell,
        lag=lag,
        shape=navigable.shape,
        index=reindexed,
        observed=observed,
        unvisited=unvisited,
        source=f"{scenario.get('pool', '?')} / {scenario.get('controller', '?')}",
    )
