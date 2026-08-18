"""Dynamical-systems analysis of a pool cleaner.

Everything else in this library measures *outcomes* -- how much floor, how much
dirt, how far, how long.  This measures *behaviour*: whether the robot repeats
itself, how fast it forgets where it started, whether it is serving the
distribution you meant, and how long a prediction about it is worth anything.

A cleaner is a hybrid system.  Continuous differential-drive flow inside a
bounded domain, punctuated by discrete events -- wall impacts, controller mode
switches, getting stuck -- and coupled to a slowly-varying field that it
consumes.  That last part is what makes it more interesting than a textbook
mobile robot, and each module here takes one classical tool to one part of it:

:mod:`~zimablue.dynamics.returnmap`
    The Poincaré section on the pool wall. Finds periodic orbits -- the robot
    doing the same loop forever, which no coverage percentage will tell you
    about until the cycle is over.

:mod:`~zimablue.dynamics.transfer`
    The Perron-Frobenius operator on a grid of cells. Its leading eigenvector
    is where the robot ends up, its spectral gap is how fast it mixes, and its
    sub-leading eigenvectors find the regions it rarely leaves.

:mod:`~zimablue.dynamics.ergodic`
    The Mathew-Mezić ergodic metric. One number for "spend time in proportion
    to how dirty it is" -- which is this project's whole argument, in the
    formalism that already existed for it.

:mod:`~zimablue.dynamics.lyapunov`
    How fast two runs started a millimetre apart stop agreeing, and therefore
    how far ahead a rollout of this simulator means anything.

:mod:`~zimablue.dynamics.averaging`
    The robot is fast and the dirt is slow. Fit a removal rate on the first
    few minutes of occupancy and predict the rest of the cycle.

:mod:`~zimablue.dynamics.plots`
    A figure for each of the above. Needs ``zimablue[viz]``; nothing else here
    does.

Two pool presets belong to the same argument and live with the other shapes:
``stadium``, whose billiard flow is provably ergodic, and ``mushroom``, whose
phase space is provably divided -- a pool where a robot in the wrong region
never reaches the other, however good its algorithm.

Everything here reads finished recordings, except
:func:`~zimablue.dynamics.lyapunov.divergence`, which has to run the
perturbed twins itself.
"""

from __future__ import annotations

from zimablue.dynamics.averaging import CleaningForecast, forecast_cleaning, occupancy
from zimablue.dynamics.ergodic import ErgodicScore, ergodic_score, target_measure
from zimablue.dynamics.lyapunov import Divergence, divergence
from zimablue.dynamics.returnmap import Orbit, ReturnMap, return_map
from zimablue.dynamics.transfer import TransferOperator, transfer_operator

__all__ = [
    "CleaningForecast",
    "Divergence",
    "ErgodicScore",
    "Orbit",
    "ReturnMap",
    "TransferOperator",
    "divergence",
    "ergodic_score",
    "forecast_cleaning",
    "occupancy",
    "return_map",
    "target_measure",
    "transfer_operator",
]
