#!/usr/bin/env python3
"""Watch policy search discover the optimal lap around a pool island.

    python examples/live_training.py
    python examples/live_training.py --generations 12 --population 12
    python examples/live_training.py --gif runs/live-training.gif

Needs ``pip install "zimablue[rl,viz]"``. Saving a GIF also needs the ``image``
extra.

The task is constructed so there is an answer, not merely a rising curve. The
floor is a constant-width ring around a circular island. At maximum speed, the
shortest non-overlapping coverage path is its centreline; differential-drive
kinematics gives the two track speeds for that circle exactly. That reference
policy is evaluated in the same simulation and drawn as the dashed line.

The learner is not given it. Cross-entropy method (CEM) starts with random left
and right track speeds, runs each policy in the Gymnasium environment, retains
the best quarter, and resamples around them. With the default seed it discovers
the stable orbit in generation three and reaches the reference score in four.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

OUTER_RADIUS = 3.0
INNER_RADIUS = 2.1
CENTRE = (OUTER_RADIUS, OUTER_RADIUS)
ORBIT_RADIUS = 0.5 * (OUTER_RADIUS + INNER_RADIUS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--minutes", type=float, default=1.0, help="length of each trial")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.001, help="UI delay per displayed frame")
    parser.add_argument("--gif", type=Path, default=None, help="capture the live dashboard")
    return parser.parse_args()


def island_pool() -> Any:
    """A tiled circular pool whose central island leaves one cleaning loop."""
    from shapely.geometry import Point, Polygon

    import zimablue as zb

    outer = Point(*CENTRE).buffer(OUTER_RADIUS, resolution=16)
    island = Point(*CENTRE).buffer(INNER_RADIUS, resolution=16)
    boundary = Polygon(outer.exterior.coords, [island.exterior.coords])
    return zb.Pool(boundary, depth=1.4, name="island_loop", material="tile")


def combined_reward(previous: dict[str, Any], current: dict[str, Any]) -> float:
    """Equal credit for newly covered floor and newly removed dirt."""
    coverage = current["coverage"] - previous["coverage"]
    cleanliness = current["dirt_removed"] - previous["dirt_removed"]
    return 50.0 * (coverage + cleanliness)


def run_policy(env: Any, action: np.ndarray, dashboard: Dashboard | None = None, **show: Any):
    """Evaluate one constant-action policy and optionally display its trial."""
    observation, info = env.reset(seed=42)
    del observation
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if dashboard is not None:
            stride = 12 if dashboard.capture else 6
            if env.elapsed % stride == 0 or terminated or truncated:
                dashboard.update(env.render(), action=action, score=total, info=info, **show)
        if terminated or truncated:
            return total, info


class Dashboard:
    """A fixed-layout training view that can also capture itself as a GIF."""

    def __init__(
        self,
        initial: np.ndarray,
        *,
        generations: int,
        reference_score: float,
        delay: float,
        capture: bool,
    ) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.delay = delay
        self.capture = capture
        self.frames: list[Any] = []
        self.points_x: list[int] = []
        self.points_y: list[float] = []
        self.best_x: list[int] = []
        self.best_y: list[float] = []

        if not capture:
            plt.ion()
        self.figure, (self.pool_ax, self.curve_ax) = plt.subplots(
            1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": (1.15, 1)}
        )
        self.figure.patch.set_facecolor("#071923")
        self.image = self.pool_ax.imshow(initial)
        self.pool_ax.axis("off")
        self.status = self.pool_ax.text(
            0.03,
            0.97,
            "",
            transform=self.pool_ax.transAxes,
            va="top",
            color="#dff8ff",
            family="monospace",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#071923",
                "alpha": 0.82,
                "edgecolor": "none",
            },
        )

        self.curve_ax.set_facecolor("#0d2733")
        self.curve_ax.set_xlabel("generation", color="#9ab9c4")
        self.curve_ax.set_ylabel("coverage + cleanliness score", color="#9ab9c4")
        self.curve_ax.set_xlim(0.5, generations + 0.5)
        self.curve_ax.set_ylim(0.0, reference_score * 1.14)
        self.curve_ax.tick_params(colors="#9ab9c4")
        for spine in self.curve_ax.spines.values():
            spine.set_color("#315568")
        self.curve_ax.axhline(
            reference_score,
            color="#b8f28b",
            linestyle="--",
            linewidth=1.5,
            label=f"known optimum  {reference_score:.1f}",
        )
        (self.candidates,) = self.curve_ax.plot(
            [], [], "o", color="#37c7e8", alpha=0.5, label="candidate"
        )
        (self.best,) = self.curve_ax.plot(
            [], [], "-o", color="#ffcc66", linewidth=2, label="best found"
        )
        legend = self.curve_ax.legend(loc="lower right", frameon=False)
        for text in legend.get_texts():
            text.set_color("#dff8ff")
        self.figure.tight_layout()

    def update(
        self,
        frame: np.ndarray,
        *,
        generation: int,
        candidate: int | str,
        action: np.ndarray,
        score: float,
        info: dict[str, Any],
    ) -> None:
        self.image.set_data(frame)
        trial = f"candidate {candidate:02d}" if isinstance(candidate, int) else candidate
        self.status.set_text(
            f"generation  {generation:02d}\n"
            f"{trial:<14}\n"
            f"tracks      [{action[0]:+.2f}, {action[1]:+.2f}]\n"
            f"coverage     {info['coverage']:5.1%}\n"
            f"cleanliness  {info['dirt_removed']:5.1%}\n"
            f"score        {score:5.1f}"
        )
        if self.points_x:
            self.candidates.set_data(self.points_x, self.points_y)
            self.best.set_data(self.best_x, self.best_y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        if self.capture:
            from PIL import Image

            self.frames.append(
                Image.fromarray(np.asarray(self.figure.canvas.buffer_rgba())[:, :, :3])
            )
        else:
            self.plt.pause(self.delay)

    def score(self, generation: int, score: float) -> None:
        self.points_x.append(generation)
        self.points_y.append(score)

    def champion(self, generation: int, score: float) -> None:
        self.best_x.append(generation)
        self.best_y.append(score)

    def finish(self, path: Path | None) -> None:
        if path is not None and self.frames:
            path.parent.mkdir(parents=True, exist_ok=True)
            durations = [80] * len(self.frames)
            durations[-1] = 2200
            self.frames[0].save(
                path,
                save_all=True,
                append_images=self.frames[1:],
                duration=durations,
                loop=0,
                optimize=True,
            )
            print(f"wrote {path}")
        elif path is None:
            self.plt.ioff()
            self.plt.show()


def main() -> None:
    args = parse_args()
    if args.generations <= 0 or args.population < 4 or args.minutes <= 0:
        raise SystemExit("generations and minutes must be positive; population must be at least 4")

    from zimablue.rl import PoolCleaningEnv

    pool = island_pool()
    start = (CENTRE[0] + ORBIT_RADIUS, CENTRE[1], np.pi / 2)
    env = PoolCleaningEnv(
        pool=pool,
        dirt="light_sediment",
        minutes=args.minutes,
        reward=combined_reward,
        render_mode="rgb_array",
        timestep=0.05,
        start_pose=start,
    )

    # On the centreline, differential-drive kinematics gives the exact inner
    # to outer track-speed ratio. Either direction has the same score.
    probe = env._build(0)
    half_track = 0.5 * probe.robot.locomotion.track_width
    probe.backend.close()
    ratio = (ORBIT_RADIUS - half_track) / (ORBIT_RADIUS + half_track)
    reference_action = np.array([ratio, 1.0], dtype=np.float32)
    reference_score, reference_info = run_policy(env, reference_action)
    print(
        f"known optimum  {reference_score:5.1f}  "
        f"({reference_info['coverage']:.1%} coverage, "
        f"{reference_info['dirt_removed']:.1%} cleanliness)"
    )

    dashboard = Dashboard(
        env.render(),
        generations=args.generations,
        reference_score=reference_score,
        delay=args.delay,
        capture=args.gif is not None,
    )

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(2)
    deviation = np.ones(2)
    elite_count = max(args.population // 4, 2)
    best_action: np.ndarray | None = None
    best_score = -np.inf

    try:
        for generation in range(1, args.generations + 1):
            population = np.clip(
                rng.normal(mean, deviation, size=(args.population, 2)), -1.0, 1.0
            ).astype(np.float32)
            if best_action is not None:
                population[0] = best_action

            scores: list[float] = []
            for candidate, action in enumerate(population, start=1):
                score, _ = run_policy(
                    env,
                    action,
                    None if dashboard.capture else dashboard,
                    generation=generation,
                    candidate=candidate,
                )
                scores.append(score)
                dashboard.score(generation, score)

            elite = population[np.argsort(scores)[-elite_count:]]
            mean = elite.mean(axis=0)
            deviation = np.maximum(elite.std(axis=0) * 1.4, 0.08)
            winner = int(np.argmax(scores))
            if scores[winner] > best_score:
                best_score = scores[winner]
                best_action = population[winner].copy()
            assert best_action is not None
            dashboard.champion(generation, best_score)
            print(
                f"generation {generation:2d}  best {best_score:5.1f}  "
                f"mean {np.mean(scores):5.1f}  tracks {best_action}"
            )

            run_policy(
                env,
                best_action,
                dashboard,
                generation=generation,
                candidate="best policy",
            )
    finally:
        env.close()

    dashboard.finish(args.gif)


if __name__ == "__main__":
    main()
