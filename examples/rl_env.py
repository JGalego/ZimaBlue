#!/usr/bin/env python3
"""Train nothing, but show what training would sit on top of.

    python examples/rl_env.py
    python examples/rl_env.py --minutes 5 --reward coverage
    python examples/rl_env.py --save runs/episode.zbr

Needs ``pip install "zimablue[rl]"``.

Three policies are run through the Gymnasium env on the same seed: random,
straight ahead, and the ``baseline_coverage`` controller that ships with the
library, driven through the env so it is scored the same way. The baseline is
what an RL policy has to beat, and it is worth knowing that number before
spending a GPU-day finding out.

The last section prints both reward functions for the same trajectories. They
disagree, and which one you train on decides what you get: ``coverage`` pays
for driving, ``dirt`` pays for cleaning, and the controller that wins one
loses the other.
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--dirt", default="autumn")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--reward", default="dirt", choices=("dirt", "coverage"))
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="write the last episode to a .zbr")
    return parser.parse_args()


def random_policy(rng):
    def act(_observation):
        return rng.uniform(-1.0, 1.0, 2)

    return act


def straight_ahead(_observation):
    return np.array([1.0, 1.0])


def baseline_policy(env):
    """The shipped controller, driven through the env's action space.

    It is a proper controller, so it wants a ControlInput rather than an
    observation vector; the env hands it the same one it built the
    observation from. That is the fair comparison -- same sensors, same rate.
    """
    from zimablue.controllers.baseline import BaselineCoverage

    controller = BaselineCoverage()
    controller.reset(env.sim.robot)
    limit = env.sim.robot.locomotion.max_speed

    def act(_observation):
        command = controller.step(env.controller.last)
        return np.array([command.left / limit, command.right / limit])

    return act


def run(env, policy, *, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        total += reward
        if terminated or truncated:
            return {"return": total, **info}


def main() -> None:
    args = parse_args()
    try:
        from zimablue.rl import PoolCleaningEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc}\n\nInstall it with:  pip install 'zimablue[rl]'") from exc

    env = PoolCleaningEnv(
        pool=args.pool,
        dirt=args.dirt,
        minutes=args.minutes,
        reward=args.reward,
        control_hz=args.control_hz,
        seed=args.seed,
        record=args.save is not None,
    )
    print(
        f"{args.pool} pool, {args.dirt} dirt, {args.minutes:g} min at "
        f"{env.control_hz:g} Hz -> {env.max_steps} decisions per episode"
    )
    print(f"observation: {env.observation_space.shape[0]} channels   action: 2 track speeds")
    print(f"reward: {args.reward}\n")

    rng = np.random.default_rng(args.seed)
    policies = {
        "random": random_policy(rng),
        "straight": straight_ahead,
        "baseline_coverage": None,  # needs the env, built after the first reset
    }

    print(f"{'policy':20s} {'return':>9s} {'coverage':>9s} {'dirt':>8s} {'distance':>9s}")
    for name in policies:
        # The baseline needs a live simulation to attach to, so it is built
        # against the env after a reset rather than up front.
        if name == "baseline_coverage":
            env.reset(seed=args.seed)
            policy = baseline_policy(env)
        else:
            policy = policies[name]
        result = run(env, policy, seed=args.seed)
        print(
            f"{name:20s} {result['return']:9.1f} {result['coverage']:8.1%} "
            f"{result['dirt_removed']:7.1%} {result['distance']:8.1f} m"
        )

    print(
        "\nA policy has to beat the baseline on the number you care about, not on\n"
        "the one that is easiest to improve. Coverage and dirt removed rank\n"
        "controllers differently -- try --reward coverage and compare."
    )

    if args.save:
        path = env.save(args.save)
        print(f"\nwrote {path}\n  zimablue replay {path}")
    else:
        env.close()


if __name__ == "__main__":
    main()
