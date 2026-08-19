"""Reinforcement learning on top of a cleaner.

::

    pip install "zimablue[rl]"

    import gymnasium as gym

    env = gym.make("zimablue.rl:ZimaBlue-v0", dirt="autumn", minutes=10)

The module prefix makes Gymnasium import this package first, so nothing else
has to; a plain ``import zimablue.rl`` followed by ``gym.make("ZimaBlue-v0")``
is the same thing spelt in two lines.

The environment is in :mod:`zimablue.rl.env`. Gymnasium is an optional
dependency, so importing this package without it fails with an instruction
rather than a traceback about a missing module.

What is here is the plumbing: an env, a decimated control rate, an
observation and two reward functions. What is not here is a trained policy or
a training loop -- those belong to whichever algorithm you bring, and
stable-baselines3 or CleanRL will drive this as they would any other Box-to-Box
env. See ``docs/ml.md`` for what a sensible baseline to beat looks like, and
why it is probably not RL.
"""

from __future__ import annotations

RL_HINT = "gymnasium is needed for zimablue.rl. Install it with:  pip install 'zimablue[rl]'"

try:
    import gymnasium as _gym
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
    raise ModuleNotFoundError(RL_HINT) from exc

from zimablue.rl.env import REWARDS, PoolCleaningEnv, channel_names, observe  # noqa: E402
from zimablue.rl.observations import EstimatedPose, ExtraObservations  # noqa: E402
from zimablue.rl.policy import PolicyController  # noqa: E402

__all__ = [
    "REWARDS",
    "RL_HINT",
    "EstimatedPose",
    "ExtraObservations",
    "PolicyController",
    "PoolCleaningEnv",
    "channel_names",
    "observe",
    "register_envs",
]

_REGISTERED = False


def register_envs() -> None:
    """Register ``ZimaBlue-v0`` with Gymnasium. Idempotent.

    Called on import, so ``gym.make("ZimaBlue-v0")`` works after
    ``import zimablue.rl``. Kept public because a subprocess vector env in a
    fresh interpreter has to call it again.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _gym.register(id="ZimaBlue-v0", entry_point="zimablue.rl.env:PoolCleaningEnv")
    _REGISTERED = True


register_envs()
