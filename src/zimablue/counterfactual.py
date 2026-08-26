"""Deterministically rerun a recording under a different decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zimablue.controllers import Controller
from zimablue.dirt import DirtSpec
from zimablue.pool import Pool
from zimablue.recording import Recording
from zimablue.robot import Cleaner
from zimablue.simulation import Simulation

__all__ = ["CounterfactualResult", "run_counterfactual"]


@dataclass(frozen=True)
class CounterfactualResult:
    """A baseline, its deterministic alternative and their measured change."""

    baseline: Recording
    alternative: Recording
    metric_deltas: dict[str, float]
    divergence_frame: int | None
    divergence_time: float | None
    trajectory_rms: float
    trajectory_final: float

    @property
    def diverged(self) -> bool:
        return self.divergence_frame is not None

    def summary(self) -> str:
        when = (
            "did not diverge"
            if self.divergence_time is None
            else f"diverged at {self.divergence_time:.2f}s"
        )
        changes = sorted(self.metric_deltas.items(), key=lambda item: abs(item[1]), reverse=True)
        headline = ", ".join(f"{name} {value:+.3g}" for name, value in changes[:3])
        suffix = f"; {headline}" if headline else ""
        return f"counterfactual {when}; trajectory RMS {self.trajectory_rms:.3f} m{suffix}"


def run_counterfactual(
    baseline: Recording | str | Path,
    controller: Controller | str,
    *,
    pool: Pool | str | None = None,
    robot: Cleaner | str | None = None,
    dirt: DirtSpec | str | None = None,
    duration: float | None = None,
    divergence_tolerance: float = 1e-6,
) -> CounterfactualResult:
    """Replay baseline conditions with explicit model or controller changes.

    The original seed, timestep, cell size and embedded model configurations
    are reused. The baseline recording itself is never edited.
    """
    source = Recording.load(baseline) if isinstance(baseline, str | Path) else baseline
    if not source.has_ground_truth:
        raise ValueError("counterfactual replay requires a simulated ground-truth recording")
    if source.n_frames == 0:
        raise ValueError("counterfactual replay requires a non-empty recording")
    if not np.isfinite(divergence_tolerance) or divergence_tolerance < 0.0:
        raise ValueError("divergence_tolerance must be finite and non-negative")

    manifest = source.manifest
    required = ("pool_config", "robot_config", "dirt_config", "timestep", "cell")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"recording lacks replay configuration: {', '.join(missing)}")

    run_seconds = source.duration if duration is None else float(duration)
    if not np.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("counterfactual duration must be finite and positive")
    start_pose_data = manifest.get("start_pose")
    start_pose = tuple(float(value) for value in start_pose_data) if start_pose_data else None
    if start_pose is not None and len(start_pose) != 3:
        raise ValueError("recording start_pose must contain x, y and heading")

    simulation = Simulation(
        pool=Pool.from_dict(manifest["pool_config"]) if pool is None else pool,
        robot=Cleaner.from_dict(manifest["robot_config"]) if robot is None else robot,
        dirt=DirtSpec.from_dict(manifest["dirt_config"]) if dirt is None else dirt,
        controller=controller,
        seed=source.seed,
        timestep=float(manifest["timestep"]),
        cell=float(manifest["cell"]),
        backend=str(manifest.get("backend", "fast2d")),
        record=True,
        scenario_name=f"counterfactual:{manifest.get('scenario', {}).get('name', 'adhoc')}",
        start_pose=start_pose,
    )
    alternative = simulation.run(seconds=run_seconds).require_recording()
    alternative.manifest["counterfactual"] = {
        "baseline_seed": source.seed,
        "baseline_controller": manifest.get("scenario", {}).get("controller", "unknown"),
        "controller": getattr(simulation.controller, "name", "custom"),
        "duration": run_seconds,
        "divergence_tolerance": divergence_tolerance,
    }

    deltas = _metric_deltas(source.metrics, alternative.metrics)
    frame, time, rms, final = _trajectory_difference(source, alternative, divergence_tolerance)
    return CounterfactualResult(
        baseline=source,
        alternative=alternative,
        metric_deltas=deltas,
        divergence_frame=frame,
        divergence_time=time,
        trajectory_rms=rms,
        trajectory_final=final,
    )


def _metric_deltas(baseline: dict[str, Any], alternative: dict[str, Any]) -> dict[str, float]:
    deltas = {}
    for name in sorted(baseline.keys() & alternative.keys()):
        before, after = baseline[name], alternative[name]
        if isinstance(before, bool) or isinstance(after, bool):
            continue
        if isinstance(before, int | float) and isinstance(after, int | float):
            delta = float(after) - float(before)
            if np.isfinite(delta):
                deltas[name] = delta
    return deltas


def _trajectory_difference(
    baseline: Recording, alternative: Recording, tolerance: float
) -> tuple[int | None, float | None, float, float]:
    for channel in ("time", "x", "y"):
        if channel not in baseline.frames or channel not in alternative.frames:
            raise ValueError(f"counterfactual comparison requires the {channel!r} channel")
    times = np.asarray(baseline.column("time"), dtype=float)
    alt_times = np.asarray(alternative.column("time"), dtype=float)
    usable = times <= alt_times[-1] + 1e-12
    times = times[usable]
    if not len(times):
        raise ValueError("baseline and counterfactual trajectories do not overlap")
    bx = np.asarray(baseline.column("x"), dtype=float)[usable]
    by = np.asarray(baseline.column("y"), dtype=float)[usable]
    ax = np.interp(times, alt_times, np.asarray(alternative.column("x"), dtype=float))
    ay = np.interp(times, alt_times, np.asarray(alternative.column("y"), dtype=float))
    distance = np.hypot(ax - bx, ay - by)
    indices = np.flatnonzero(distance > tolerance)
    frame = int(np.flatnonzero(usable)[indices[0]]) if len(indices) else None
    time = float(times[indices[0]]) if len(indices) else None
    return frame, time, float(np.sqrt(np.mean(distance**2))), float(distance[-1])
