"""Spectral multiscale coverage: steer to match a distribution, not a path.

Every other planner in this package answers "which cell next?". This one never
asks. It carries a running Fourier description of where the robot has *spent
its time*, compares it with a Fourier description of where it *should* spend
its time, and drives down the difference. Coverage is not the objective, it is
what happens when the two spectra agree.

The metric is Mathew and Mezic's (2011), and :mod:`zimablue.dynamics.ergodic`
already computes it -- that module scores a finished run, this one closes the
loop around it::

    phi(t) = sum_k Lambda_k |c_k(t) - mu_k|^2

``c_k`` is the time-average of the basis function ``f_k`` along the trajectory
so far, ``mu_k`` is its integral against the target distribution, and
``Lambda_k = (1 + |k|^2)^(-3/2)`` weights coarse modes above fine ones. That
weighting is the "multiscale" in the name and it is the reason the method
behaves sensibly when it is interrupted: it gets the broad strokes right first
and refines, so a run cut short at any point has done the most valuable half
of the work. A boustrophedon sweep cut in half has cleaned half a pool.

The control law is the steepest descent of ``phi`` for a first-order vehicle:
drive along ``-B``, where ``B_j = sum_k Lambda_k (c_k - mu_k) df_k/dx_j``. On a
differential drive that becomes a heading to hold, and the usual "slow down
while you turn" applies.

What it is not
--------------

It is not complete. There is no theorem here saying every square metre gets
visited, and with a uniform target the trajectory keeps redistributing time
forever rather than terminating. What it gives instead is a schedule: an
anytime, smooth, non-repeating path whose *density* converges to the one you
asked for. If the target is dirt rather than floor, it will spend its time
where the dirt is, without ever planning a route to it.
"""

from __future__ import annotations

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.controllers.systematic import MapCell
from zimablue.geometry import wrap_angle
from zimablue.planners.online import CCW, OnlineCoverage
from zimablue.robot import DriveCommand

__all__ = ["SpectralCoverage"]


class SpectralCoverage(OnlineCoverage):
    """SMC (Mathew & Mezic, 2011) over the map the robot builds as it goes.

    Shares the estimator, the occupancy grid and the bump recovery with every
    other online planner here, and replaces only the decision: instead of
    choosing a cell it chooses a heading, every tick.

    The domain is the bounding box of the floor observed so far, so it grows
    during the first minutes of a run. When it grows the spectrum has to be
    recomputed -- the basis functions changed -- which is why the trajectory is
    kept. It happens a handful of times per run and costs nothing.
    """

    name = "smc"

    def __init__(
        self,
        *,
        modes: int = 8,
        cruise: float = 0.95,
        turn_gain: float = 2.2,
        sample_interval: float = 0.25,
        target: str = "uniform",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if target not in ("uniform", "unvisited"):
            raise ValueError(f"target must be 'uniform' or 'unvisited', got {target!r}")
        self.modes = int(modes)
        self.cruise = float(cruise)
        self.smc_turn_gain = float(turn_gain)
        self.sample_interval = float(sample_interval)
        self.target = target

    # ------------------------------------------------------------------
    def begin(self) -> None:
        wave = np.arange(self.modes)
        self._k1, self._k2 = np.meshgrid(wave, wave, indexing="ij")
        # Coarse modes matter more. The exponent is (n+1)/2 for n dimensions,
        # which is what makes the metric a norm on a Sobolev space of negative
        # index -- the formal statement of "get the big picture right first".
        self._weight = (1.0 + self._k1**2 + self._k2**2) ** -1.5
        self._domain: tuple[float, float, float, float] | None = None
        self._trace: list[tuple[float, float]] = []
        self._sum = np.zeros((self.modes, self.modes))
        self._mu = np.zeros((self.modes, self.modes))
        self._last_sample = -1e9
        self._last_domain_check = -1e9
        self._phi = 0.0

    # ------------------------------------------------------------------
    def _act(self, ctl: ControlInput, pose) -> DriveCommand:
        top = ctl.robot.locomotion.max_speed
        cell = self.map.to_index(pose.x, pose.y)
        if cell != self.here:
            step = (cell[0] - self.here[0], cell[1] - self.here[1])
            if step in CCW:
                self.facing = step
            self.here = cell
        self.done.add(cell)

        if ctl.time - self._last_domain_check > 2.0:
            self._last_domain_check = ctl.time
            self._refresh_domain()

        if self._domain is None:
            # Nothing observed yet. Drive straight until the sonar or the
            # bumpers have given the box something to be.
            return DriveCommand(top * self.cruise, top * self.cruise, brush=True, pump=1.0)

        if ctl.time - self._last_sample >= self.sample_interval:
            self._last_sample = ctl.time
            self._trace.append((pose.x, pose.y))
            self._sum += self._basis(pose.x, pose.y)

        heading = self._descent(pose)
        heading = self._avoid(heading, pose)
        error = float(wrap_angle(heading - pose.heading))
        forward = top * self.cruise * float(np.clip(np.cos(error), 0.0, 1.0))
        return DriveCommand.from_body(forward, self.smc_turn_gain * error, ctl.robot.locomotion)

    def choose(self, here):  # pragma: no cover - SMC never picks a cell
        raise NotImplementedError("smc steers continuously; it has no cell to choose")

    # -- the spectrum --------------------------------------------------------
    def _basis(self, x: float, y: float) -> np.ndarray:
        x0, y0, lx, ly = self._domain
        u = np.clip((x - x0) / lx, 0.0, 1.0)
        v = np.clip((y - y0) / ly, 0.0, 1.0)
        return self._norm * np.cos(self._k1 * np.pi * u) * np.cos(self._k2 * np.pi * v)

    def _gradient(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, lx, ly = self._domain
        u = np.clip((x - x0) / lx, 0.0, 1.0)
        v = np.clip((y - y0) / ly, 0.0, 1.0)
        cu, cv = np.cos(self._k1 * np.pi * u), np.cos(self._k2 * np.pi * v)
        su, sv = np.sin(self._k1 * np.pi * u), np.sin(self._k2 * np.pi * v)
        dx = -self._norm * (self._k1 * np.pi / lx) * su * cv
        dy = -self._norm * (self._k2 * np.pi / ly) * cu * sv
        return dx, dy

    def _descent(self, pose) -> float:
        """The heading that reduces the ergodic metric fastest."""
        coefficients = self._sum / max(len(self._trace), 1)
        residual = self._weight * (coefficients - self._mu)
        self._phi = float((self._weight * (coefficients - self._mu) ** 2).sum())
        dx, dy = self._gradient(pose.x, pose.y)
        bx, by = float((residual * dx).sum()), float((residual * dy).sum())
        if abs(bx) < 1e-12 and abs(by) < 1e-12:
            return pose.heading
        return float(np.arctan2(-by, -bx))

    def _avoid(self, heading: float, pose) -> float:
        """Nudge the descent direction off a wall it points straight at.

        The control law knows nothing about obstacles -- SMC is stated for a
        vehicle in free space -- so the map has to veto. Sixteen candidate
        headings, nearest clear one wins.
        """
        if self._clear(heading, pose):
            return heading
        for offset in np.arange(1, 9) * (np.pi / 8):
            for sign in (1.0, -1.0):
                candidate = heading + sign * offset
                if self._clear(candidate, pose):
                    return float(candidate)
        return heading

    def _clear(self, heading: float, pose) -> bool:
        for step in (1.0, 2.0):
            probe = (
                pose.x + np.cos(heading) * step * self.cell,
                pose.y + np.sin(heading) * step * self.cell,
            )
            if self.map.state_at(*probe) == MapCell.WALL:
                return False
        return True

    # -- the domain and the target ------------------------------------------
    def _refresh_domain(self) -> None:
        """Re-fit the box to the observed floor, and the target inside it."""
        floor = np.argwhere(self.map.grid == MapCell.FREE)
        if len(floor) < 8:
            return
        rows, cols = floor[:, 0], floor[:, 1]
        margin = 1.0
        x0 = (cols.min() - self.map.origin) * self.cell - margin
        y0 = (rows.min() - self.map.origin) * self.cell - margin
        lx = (cols.max() - cols.min()) * self.cell + 2 * margin
        ly = (rows.max() - rows.min()) * self.cell + 2 * margin
        fresh = (x0, y0, max(lx, 1.0), max(ly, 1.0))

        moved = (
            self._domain is None
            or max(abs(a - b) for a, b in zip(fresh, self._domain, strict=True)) > 0.5 * self.cell
        )
        if moved:
            self._domain = fresh
            self._norm = self._normalisation()
            # The basis changed, so every coefficient did. This is why the
            # trajectory is kept rather than only its running sum.
            self._sum = np.zeros((self.modes, self.modes))
            for x, y in self._trace:
                self._sum += self._basis(x, y)
        self._retarget(floor)

    def _normalisation(self) -> np.ndarray:
        """``1/h_k``, so each basis function has unit norm over the box."""
        half = np.where(self._k1 == 0, 1.0, 0.5) * np.where(self._k2 == 0, 1.0, 0.5)
        return 1.0 / np.sqrt(half)

    def _retarget(self, floor: np.ndarray) -> None:
        """Fourier coefficients of the distribution we are aiming at."""
        rows, cols = floor[:, 0], floor[:, 1]
        xs = (cols - self.map.origin) * self.cell
        ys = (rows - self.map.origin) * self.cell
        if self.target == "unvisited" and self.done:
            keep = np.array([(int(r), int(c)) not in self.done for r, c in floor])
            if keep.any():
                xs, ys = xs[keep], ys[keep]

        x0, y0, lx, ly = self._domain
        u = np.clip((xs - x0) / lx, 0.0, 1.0)
        v = np.clip((ys - y0) / ly, 0.0, 1.0)
        # (modes, modes, points) would be the obvious loop; one matrix product
        # over the points does it instead. Mind the brackets -- ``@`` and ``*``
        # have the same precedence, so the normalisation has to be applied
        # after the contraction, not before it.
        cu = np.cos(np.outer(np.arange(self.modes), np.pi * u))
        cv = np.cos(np.outer(np.arange(self.modes), np.pi * v))
        self._mu = self._norm * (cu @ cv.T) / len(u)

    # ------------------------------------------------------------------
    def telemetry(self) -> dict[str, float]:
        base = super().telemetry()
        base["ergodic"] = self._phi
        base["samples"] = float(len(self._trace))
        return base


@CONTROLLERS.register("smc")
def _make_smc(**kwargs: object) -> SpectralCoverage:
    return SpectralCoverage(**kwargs)  # type: ignore[arg-type]
