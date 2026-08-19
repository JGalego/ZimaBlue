"""The Gymnasium environment.

What matters more than the rest here is that an episode is reproducible
from its seed, because an RL result nobody can re-run is not a result -- and
that the reward is exactly the quantity it claims to be:
a reward that is *nearly* grams collected trains a policy to exploit the
difference.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from zimablue.recording import Recording  # noqa: E402
from zimablue.rl import PoolCleaningEnv  # noqa: E402


def rollout(env, steps=25, seed=0):
    """Drive the env with a fixed pseudo-random policy and log everything."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(7)
    observations, rewards = [obs], []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        )
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            break
    return np.asarray(observations), np.asarray(rewards), info


@pytest.fixture
def env():
    made = PoolCleaningEnv(pool="rectangular", dirt="light_sediment", minutes=1.0)
    yield made
    made.close()


# ----------------------------------------------------------------------
def test_it_satisfies_the_gymnasium_api():
    from gymnasium.utils.env_checker import check_env

    made = PoolCleaningEnv(pool="rectangular", minutes=0.5)
    check_env(made, skip_render_check=True)
    made.close()


def test_it_is_registered_under_an_id():
    made = gym.make("ZimaBlue-v0", pool="rectangular", minutes=0.5)
    made.reset(seed=1)
    made.close()


def test_the_same_seed_gives_the_same_episode(env):
    first_obs, first_rewards, _ = rollout(env, seed=11)
    second_obs, second_rewards, _ = rollout(env, seed=11)
    assert np.array_equal(first_obs, second_obs)
    assert np.array_equal(first_rewards, second_rewards)


def test_a_different_seed_gives_a_different_episode(env):
    first_obs, _, _ = rollout(env, seed=11)
    second_obs, _, _ = rollout(env, seed=12)
    assert not np.array_equal(first_obs, second_obs)


# ----------------------------------------------------------------------
def test_one_action_covers_several_physics_ticks(env):
    """The physics runs at 50 Hz and the agent decides at 5, so ten ticks."""
    assert env.repeat == 10
    assert env.control_hz == pytest.approx(5.0)

    env.reset(seed=0)
    before = env.sim.state.time
    env.step(np.zeros(2, dtype=np.float32))
    assert env.sim.state.time - before == pytest.approx(env.repeat * env.timestep)


def test_deciding_faster_than_the_physics_is_refused():
    with pytest.raises(ValueError, match="control_hz"):
        PoolCleaningEnv(control_hz=200.0, timestep=0.02)


def test_an_unknown_reward_is_refused():
    with pytest.raises(ValueError, match="reward must be one of"):
        PoolCleaningEnv(reward="vibes")


def test_the_episode_is_the_requested_length():
    made = PoolCleaningEnv(pool="rectangular", minutes=1.0, control_hz=5.0)
    assert made.max_steps == 300  # 60 s at 5 Hz
    made.reset(seed=0)
    for _ in range(made.max_steps - 1):
        _, _, _terminated, truncated, _ = made.step(np.zeros(2, dtype=np.float32))
        assert not truncated
    _, _, _, truncated, _ = made.step(np.zeros(2, dtype=np.float32))
    assert truncated
    made.close()


# ----------------------------------------------------------------------
def test_the_dirt_reward_is_exactly_the_grams_collected(env):
    _, rewards, info = rollout(env, steps=40)
    assert rewards.sum() == pytest.approx(info["dirt_collected"], rel=1e-9)


def test_the_coverage_reward_is_the_floor_newly_reached():
    """Newly, so the swath the robot is dropped onto is not paid for."""
    made = PoolCleaningEnv(pool="rectangular", minutes=1.0, reward="coverage")
    made.reset(seed=0)
    footprint = made._visited

    _, rewards, info = rollout(made, steps=40)
    navigable = made.sim.pool.navigable_mask(made.sim.world.cell)
    reached = info["coverage"] * int(navigable.sum())
    assert footprint > 0, "the robot occupies some floor before it moves"
    assert rewards.sum() == pytest.approx((reached - footprint) * made.sim.world.cell**2)
    made.close()


def test_the_two_rewards_are_not_the_same_number(env):
    """If they agreed there would be no library. Same seed, same actions."""
    coverage = PoolCleaningEnv(pool="rectangular", minutes=1.0, reward="coverage")
    _, dirt_rewards, _ = rollout(env, steps=40)
    _, coverage_rewards, _ = rollout(coverage, steps=40)
    coverage.close()

    # Same trajectory -- only the payout differs.
    assert dirt_rewards.shape == coverage_rewards.shape
    assert np.corrcoef(dirt_rewards, coverage_rewards)[0, 1] < 0.99


# ----------------------------------------------------------------------
def test_actions_are_clipped_to_the_motor_limit(env):
    env.reset(seed=0)
    env.step(np.array([50.0, -50.0], dtype=np.float32))
    limit = env.sim.robot.locomotion.max_speed
    assert env.controller.command.left == pytest.approx(limit)
    assert env.controller.command.right == pytest.approx(-limit)


def test_observation_channels_are_named_and_ordered(env):
    assert env.channels[:3] == ["battery", "filter_load", "elapsed"]
    sensors = [c.split(".")[0] for c in env.channels[3:]]
    assert sensors == sorted(sensors), "a new sensor must not permute the vector"
    assert len(env.channels) == env.observation_space.shape[0]


def test_stepping_before_reset_says_so():
    made = PoolCleaningEnv(pool="rectangular", minutes=0.5)
    with pytest.raises(RuntimeError, match="reset"):
        made.step(np.zeros(2, dtype=np.float32))


# ----------------------------------------------------------------------
def test_an_episode_can_be_saved_and_replayed(tmp_path):
    """The payoff for recording: watch what the policy actually did."""
    made = PoolCleaningEnv(pool="rectangular", minutes=0.5, record=True)
    rollout(made, steps=20)
    path = made.save(str(tmp_path / "episode.zbr"))

    recording = Recording.load(path)
    assert recording.n_frames > 0
    assert recording.manifest["scenario"]["controller"] == "rl_agent"


def test_saving_without_recording_says_what_to_do(env):
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="record=True"):
        env.save("nope.zbr")


# ----------------------------------------------------------------------
# The round trip: a policy back into the ordinary controller interface
# ----------------------------------------------------------------------
def test_a_policy_runs_as_a_controller():
    import zimablue as zb
    from zimablue.rl import PolicyController

    seen = []

    def policy(observation):
        seen.append(observation)
        return np.array([1.0, 0.6], dtype=np.float32)

    result = zb.Simulation(pool="rectangular", controller=PolicyController(policy), seed=5).run(
        seconds=20
    )

    assert result.metrics.distance_traveled > 0
    assert seen, "the policy should have been asked for an action"
    assert result.metrics.coverage > 0


def test_the_policy_is_asked_at_its_own_rate():
    """Held between decisions, so a 5 Hz policy is not run at 50 Hz."""
    import zimablue as zb
    from zimablue.rl import PolicyController

    times = []

    def policy(_observation):
        return np.zeros(2, dtype=np.float32)

    controller = PolicyController(policy, control_hz=5.0)
    original = controller.step

    def spy(control_input):
        before = controller._next_decision
        command = original(control_input)
        if controller._next_decision != before:
            times.append(control_input.time)
        return command

    controller.step = spy  # type: ignore[method-assign]
    zb.Simulation(pool="rectangular", controller=controller, seed=5).run(seconds=10)

    gaps = np.diff(times)
    assert len(times) == pytest.approx(50, abs=2), "10 s at 5 Hz"
    assert gaps == pytest.approx(0.2, abs=0.02)


def test_a_policy_with_extra_observations_still_matches_the_env():
    """The layout has to survive the round trip, extras included."""
    import zimablue as zb
    from zimablue.rl import EstimatedPose, PolicyController

    made = PoolCleaningEnv(pool="rectangular", minutes=0.5, extra_observations=EstimatedPose())
    obs, _ = made.reset(seed=5)
    made.close()

    seen = []
    zb.Simulation(
        pool="rectangular",
        controller=PolicyController(
            lambda o: (seen.append(o), np.zeros(2))[1], extra_observations=EstimatedPose()
        ),
        seed=5,
    ).run(seconds=1)
    assert seen[0].shape == obs.shape


def test_a_policy_and_the_env_see_the_same_observation():
    """Training and deployment must not disagree about the input layout."""
    import zimablue as zb
    from zimablue.rl import PolicyController, channel_names

    made = PoolCleaningEnv(pool="rectangular", minutes=0.5)
    obs, _ = made.reset(seed=5)
    made.close()

    seen = []
    zb.Simulation(
        pool="rectangular",
        controller=PolicyController(lambda o: (seen.append(o), np.zeros(2))[1]),
        seed=5,
    ).run(seconds=1)

    assert seen[0].shape == obs.shape
    assert len(channel_names(zb.make_robot("tracked"))) == obs.shape[0]


# ----------------------------------------------------------------------
# Extra observations
# ----------------------------------------------------------------------
def test_extra_observations_extend_the_vector_and_the_space():
    from zimablue.rl import EstimatedPose

    plain = PoolCleaningEnv(pool="rectangular", minutes=0.5)
    extended = PoolCleaningEnv(pool="rectangular", minutes=0.5, extra_observations=EstimatedPose())
    obs, _ = extended.reset(seed=2)

    assert extended.channels[: len(plain.channels)] == plain.channels
    assert extended.channels[-7:] == list(EstimatedPose.channels)
    assert obs.shape == extended.observation_space.shape
    assert extended.observation_space.contains(obs)
    plain.close()
    extended.close()


def test_the_estimate_tracks_the_robot_it_cannot_see():
    """A wiring check, not an estimator check.

    Reading the wrong encoder channels or dropping dt would leave the estimate
    somewhere unrelated to the robot. Drive straight and the estimated
    displacement should be within a few percent of the true one -- over a
    short run, before dead reckoning has had time to wander.
    """
    from zimablue.rl import EstimatedPose

    env = PoolCleaningEnv(pool="rectangular", minutes=1.0, extra_observations=EstimatedPose())
    env.reset(seed=4)
    start = (env.sim.state.x, env.sim.state.y)
    for _ in range(60):
        obs, *_ = env.step(np.array([1.0, 1.0], dtype=np.float32))

    estimated = float(np.hypot(obs[-7], obs[-6]))  # est.x, est.y from its own origin
    travelled = float(np.hypot(env.sim.state.x - start[0], env.sim.state.y - start[1]))
    env.close()
    assert estimated == pytest.approx(travelled, rel=0.15), (
        f"estimated {estimated:.2f} m against {travelled:.2f} m actually travelled"
    )


def test_extra_observations_never_get_ground_truth():
    """The rule the controller interface has always had, extended to here."""
    seen = []

    class Peeker:
        channels = ("peek",)
        bounds = ((0.0,), (1.0,))

        def reset(self, robot):
            pass

        def __call__(self, control_input):
            seen.append(control_input.truth)
            return np.zeros(1, dtype=np.float32)

    env = PoolCleaningEnv(pool="rectangular", minutes=0.5, extra_observations=Peeker())
    env.reset(seed=1)
    env.step(np.zeros(2, dtype=np.float32))
    env.close()
    assert seen and all(truth is None for truth in seen)


def test_extra_observations_run_every_tick_not_every_decision():
    """An EKF fed one sample in ten is a different filter."""
    calls = []

    class Counter:
        channels = ("n",)
        bounds = ((0.0,), (np.inf,))

        def reset(self, robot):
            calls.clear()

        def __call__(self, control_input):
            calls.append(control_input.time)
            return np.array([len(calls)], dtype=np.float32)

    env = PoolCleaningEnv(pool="rectangular", minutes=0.5, extra_observations=Counter())
    env.reset(seed=1)
    before = len(calls)
    env.step(np.zeros(2, dtype=np.float32))
    env.close()
    assert len(calls) - before == env.repeat


# ----------------------------------------------------------------------
def test_it_vectorises():
    """The throughput claim in docs/ml.md assumes several of these at once."""
    vector = gym.make_vec("ZimaBlue-v0", num_envs=3, pool="rectangular", minutes=0.5)
    observations, _ = vector.reset(seed=0)
    assert observations.shape[0] == 3

    observations, rewards, terminated, truncated, _ = vector.step(
        np.zeros((3, 2), dtype=np.float32)
    )
    assert rewards.shape == (3,)
    assert terminated.shape == truncated.shape == (3,)
    vector.close()


def test_the_episode_is_over_once_it_is_saved(tmp_path):
    made = PoolCleaningEnv(pool="rectangular", minutes=0.5, record=True)
    rollout(made, steps=5)
    made.save(str(tmp_path / "episode.zbr"))
    with pytest.raises(RuntimeError, match="reset"):
        made.step(np.zeros(2, dtype=np.float32))
    # And a reset gets you a working env back.
    made.reset(seed=1)
    made.step(np.zeros(2, dtype=np.float32))
    made.close()


def test_the_map_fractions_are_fractions_of_the_map():
    """They are divided by a cell count, and that count was once wrong.

    Nothing downstream can tell a fraction that is four times too small from a
    correct one, so it gets checked against the map's own array.
    """
    from zimablue.controllers.systematic import MapCell
    from zimablue.rl import EstimatedPose

    extra = EstimatedPose()
    env = PoolCleaningEnv(pool="rectangular", minutes=0.5, extra_observations=extra)
    env.reset(seed=3)
    for _ in range(40):
        obs, *_ = env.step(np.array([1.0, 0.85], dtype=np.float32))

    known = float((extra.map.grid != MapCell.UNKNOWN).sum()) / extra.map.grid.size
    assert obs[-2] == pytest.approx(known, rel=1e-6)
    assert 0.0 < obs[-2] <= 1.0
    assert 0.0 < obs[-1] <= obs[-2], "covered floor cannot exceed explored floor"
    env.close()


# ----------------------------------------------------------------------
# Episode seeding
# ----------------------------------------------------------------------
def test_reset_without_a_seed_starts_a_new_episode(env):
    """A training loop never passes a seed.

    This env used to replay its construction seed on every unseeded reset, so
    a policy would have been shown one episode for the whole run and taught to
    memorise it. Nothing in the suite noticed, because every test seeded.
    """
    episodes = []
    for _ in range(3):
        obs, info = env.reset()
        for _ in range(10):
            obs, *_ = env.step(np.array([1.0, 0.7], dtype=np.float32))
        episodes.append((info["seed"], obs.copy()))

    seeds = [seed for seed, _ in episodes]
    assert len(set(seeds)) == 3, f"three resets gave the seeds {seeds}"
    assert not np.array_equal(episodes[0][1], episodes[1][1])
    assert not np.array_equal(episodes[0][1], episodes[2][1])


def test_the_sequence_of_episodes_is_reproducible():
    """Unseeded does not mean unrepeatable: it is drawn, not arbitrary."""

    def sequence(seed):
        made = PoolCleaningEnv(pool="rectangular", minutes=0.5, seed=seed)
        seeds = [made.reset()[1]["seed"] for _ in range(4)]
        made.close()
        return seeds

    assert sequence(5) == sequence(5)
    assert sequence(5) != sequence(6)


def test_an_explicit_seed_still_pins_the_episode(env):
    """And re-pins the sequence after it, so an evaluation is repeatable."""
    first = [env.reset(seed=3)[1]["seed"]] + [env.reset()[1]["seed"] for _ in range(2)]
    second = [env.reset(seed=3)[1]["seed"]] + [env.reset()[1]["seed"] for _ in range(2)]
    assert first[0] == 3
    assert first == second
