#!/usr/bin/env python3
"""Train a controller with PPO, then watch it.

    python examples/train_policy.py
    python examples/train_policy.py --steps 500000 --minutes 4
    python examples/train_policy.py --reward coverage

Needs ``pip install "zimablue[rl]" stable-baselines3``.

Trains, evaluates against the shipped controllers on held-out seeds, wraps the
policy back into a ``Controller``, and writes a ``.zbr`` you can replay. The
evaluation is the part worth keeping: a learning curve tells you the policy
improved, and only the comparison tells you whether it improved to anywhere
worth being.

Expect it to lose. ``baseline_coverage`` is a hand-written boustrophedon sweep
with wall following and stuck recovery, and a few hundred thousand steps of PPO
from scratch is not much. Reporting that honestly is the point of the script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--dirt", default="autumn")
    parser.add_argument("--minutes", type=float, default=4.0, help="episode length")
    parser.add_argument("--steps", type=int, default=300_000, help="total training steps")
    parser.add_argument("--envs", type=int, default=4, help="parallel workers")
    parser.add_argument("--reward", default="dirt", choices=("dirt", "coverage"))
    parser.add_argument("--episodes", type=int, default=5, help="held-out evaluation seeds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--gif", action="store_true", help="also render the episode")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int):
    """A factory, because SubprocVecEnv builds each worker in its own process."""

    def build():
        from zimablue.rl import EstimatedPose, PoolCleaningEnv

        return PoolCleaningEnv(
            pool=args.pool,
            dirt=args.dirt,
            minutes=args.minutes,
            reward=args.reward,
            seed=seed,
            # Hand it the EKF. Learning to localise *and* to plan from raw
            # sensors needs a recurrent policy and far more compute than this.
            extra_observations=EstimatedPose(),
        )

    return build


def evaluate(controller, args: argparse.Namespace, *, truth: bool = False) -> dict[str, float]:
    """Run held-out seeds and average. Seeds the training never saw."""
    import zimablue as zb

    coverage, dirt = [], []
    for episode in range(args.episodes):
        result = zb.Simulation(
            pool=args.pool,
            dirt=args.dirt,
            controller=controller,
            seed=10_000 + episode,
            record=False,
            expose_truth=truth,
        ).run(minutes=args.minutes)
        coverage.append(result.metrics.coverage)
        dirt.append(result.metrics.dirt_removed_fraction)
    return {"coverage": float(np.mean(coverage)), "dirt": float(np.mean(dirt))}


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc}\n\nInstall it with:  pip install stable-baselines3") from exc

    from zimablue.rl import EstimatedPose, PolicyController

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{args.pool} pool, {args.dirt} dirt, {args.minutes:g} min episodes")
    print(f"training {args.steps:,} steps on {args.envs} workers, reward = {args.reward}\n")

    vec = VecMonitor(SubprocVecEnv([make_env(args, args.seed + i) for i in range(args.envs)]))
    # Observations span sonar metres, gyro rad/s and unit fractions. Left raw,
    # the value function spends its first 100k steps learning the scales.
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec, verbose=1, seed=args.seed, n_steps=512, batch_size=512)
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    elapsed = time.perf_counter() - started
    print(f"\ntrained in {elapsed / 60:.1f} min ({args.steps / elapsed:.0f} steps/s)")

    model.save(args.out / "ppo_cleaner")
    vec.save(str(args.out / "ppo_cleaner_vecnormalize.pkl"))
    # The running mean and variance the policy was trained against. Kept as
    # numbers rather than reloaded from the pickle, which wants a live vector
    # env to attach to -- and forgetting them entirely is the classic way to
    # deploy a policy that worked in training and drives into a wall here.
    mean, var = vec.obs_rms.mean.copy(), vec.obs_rms.var.copy()
    clip, epsilon = vec.clip_obs, vec.epsilon
    vec.close()

    # ------------------------------------------------------------------
    # Back onto the ordinary controller interface, and scored like anything
    # else. Training-time reward is not a result; this is.
    # ------------------------------------------------------------------
    def policy(observation: np.ndarray) -> np.ndarray:
        scaled = np.clip((observation - mean) / np.sqrt(var + epsilon), -clip, clip)
        return model.predict(scaled.reshape(1, -1), deterministic=True)[0][0]

    trained = PolicyController(policy, extra_observations=EstimatedPose())

    print(f"\nheld-out seeds ({args.episodes}), {args.minutes:g} min each")
    print(f"{'controller':<20}{'coverage':>10}{'dirt removed':>14}")
    contenders = [
        ("ppo (trained)", trained, False),
        ("random_bounce", "random_bounce", False),
        ("baseline_coverage", "baseline_coverage", False),
        ("systematic", "systematic", False),
        ("dirt_oracle", "dirt_oracle", True),
    ]
    scores = {}
    for label, controller, truth in contenders:
        scores[label] = evaluate(controller, args, truth=truth)
        print(f"{label:<20}{scores[label]['coverage']:>9.1%}{scores[label]['dirt']:>14.1%}")

    ppo, best = (
        scores["ppo (trained)"],
        max((s for k, s in scores.items() if k != "ppo (trained)"), key=lambda s: s[args.reward]),
    )
    verdict = "beats" if ppo[args.reward] >= best[args.reward] else "loses to"
    print(f"\nPPO {verdict} the best shipped controller on {args.reward}.")

    # ------------------------------------------------------------------
    import zimablue as zb

    result = zb.Simulation(pool=args.pool, dirt=args.dirt, controller=trained, seed=10_000).run(
        minutes=args.minutes
    )
    path = result.save(args.out / "ppo_episode.zbr")
    print(f"\nwrote {path}\n  watch it:  zimablue replay {path}")

    if args.gif:
        from zimablue.replay import export_movie, export_summary

        export_summary(result.recording, args.out / "ppo_episode.png")
        export_movie(
            result.recording,
            args.out / "ppo_episode.gif",
            speed=args.minutes * 60 / 12,  # the whole episode in ~12 seconds
            fps=12,
            dpi=58,
        )
        print(f"  or look at {args.out / 'ppo_episode.gif'}")


if __name__ == "__main__":
    main()
