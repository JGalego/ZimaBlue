"""The interactive player's state machine.

The window itself blocks on user input and is not tested -- mocking an event
loop would test the mock. What *is* tested is everything the window merely
calls: seeking, clamping, speed cycling, wrap-around and the key bindings,
which is where the behaviour a user actually notices lives.
"""

from __future__ import annotations

import pytest

from zimablue.recording import Recording
from zimablue.simulation import Simulation

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from zimablue.replay.player import SPEEDS, ReplayPlayer  # noqa: E402


class FakeKey:
    """Stands in for a matplotlib KeyEvent, which needs a live canvas."""

    def __init__(self, key: str) -> None:
        self.key = key


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Recording:
    result = Simulation(pool="rectangular", dirt="light_sediment", seed=4).run(seconds=40)
    return Recording.load(result.save(tmp_path_factory.mktemp("zbr") / "r.zbr"))


@pytest.fixture
def player(recording) -> ReplayPlayer:
    return ReplayPlayer(recording, speed=1.0, start_paused=True)


def test_starts_at_the_beginning(player):
    assert player.index == 0
    assert player.paused


def test_space_toggles_pause(player):
    player._on_key(FakeKey(" "))
    assert not player.paused
    player._on_key(FakeKey(" "))
    assert player.paused


def test_arrows_step_by_a_second_and_shift_by_ten(player):
    player._on_key(FakeKey("right"))
    assert player.index == pytest.approx(1 / player.dt, abs=1)
    player._on_key(FakeKey("left"))
    assert player.index == 0

    player._on_key(FakeKey("shift+right"))
    assert player.index == pytest.approx(10 / player.dt, abs=1)


def test_seeking_clamps_to_the_recording(player):
    player._seek(-500)
    assert player.index == 0
    player._seek(10_000_000)
    assert player.index == player.recording.n_frames - 1


def test_speed_cycles_through_the_presets_and_stops_at_the_ends(player):
    player.speed = SPEEDS[0]
    player._on_key(FakeKey("down"))
    assert player.speed == SPEEDS[0], "should not go below the slowest speed"

    for _ in range(len(SPEEDS) + 3):
        player._on_key(FakeKey("up"))
    assert player.speed == SPEEDS[-1], "should not exceed the fastest speed"


def test_r_restarts_and_resumes(player):
    player._seek(50)
    player.paused = True
    player._on_key(FakeKey("r"))
    assert player.index == 0
    assert not player.paused


def test_an_unbound_key_is_harmless(player):
    before = (player.index, player.paused, player.speed)
    player._on_key(FakeKey("z"))
    assert (player.index, player.paused, player.speed) == before


def test_tick_advances_only_while_playing(player):
    player.paused = True
    player._tick()
    assert player.index == 0

    player.paused = False
    player._tick()
    assert player.index > 0


def test_tick_wraps_at_the_end_rather_than_stalling(player):
    player.paused = False
    player._seek(player.recording.n_frames - 1)
    player._tick()
    assert player.index == 0


def test_faster_playback_takes_bigger_steps(player):
    player.paused = False

    player.speed = 1.0
    player._seek(0)
    player._tick()
    slow = player.index

    player.speed = 25.0
    player._seek(0)
    player._tick()
    assert player.index > slow


def test_scrubbing_the_slider_pauses_and_moves(player):
    player.paused = False
    player._on_scrub(120.0)
    assert player.index == 120
    assert player.paused, "grabbing the scrubber should pause playback"


def test_seeking_does_not_re_enter_through_the_slider(player):
    """_seek sets the slider, whose callback calls back in. The guard flag
    stops that becoming a loop -- and stops a seek silently pausing playback."""
    player.paused = False
    player._seek(75)
    assert player.index == 75
    assert not player.paused


def test_saving_a_frame_writes_a_png(player, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    player._seek(10)
    player._on_key(FakeKey("s"))
    assert list(tmp_path.glob("zimablue-frame-*.png"))
