"""The .zbr format."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from zimablue.recording import FORMAT, SCHEMA_VERSION, Recorder, Recording
from zimablue.simulation import Simulation


def test_recording_has_the_expected_members(short_run, tmp_path):
    path = short_run.save(tmp_path / "run.zbr")
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert {"manifest.json", "frames.npz", "events.json", "metrics.json", "dirt.npz"} <= names


def test_manifest_is_readable_without_zimablue(short_run, tmp_path):
    """Inspectability is a feature: plain JSON, no custom decoder."""
    path = short_run.save(tmp_path / "run.zbr")
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == FORMAT
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["seed"] == 7
    assert "pool_config" in manifest and "robot_config" in manifest


def test_frames_are_rectangular_and_aligned(short_run):
    frames = short_run.recording.frames
    lengths = {name: values.size for name, values in frames.items()}
    assert len(set(lengths.values())) == 1, f"ragged columns: {lengths}"
    assert frames["time"].size == short_run.recording.n_frames
    assert np.all(np.diff(frames["time"]) > 0), "time must be strictly increasing"


def test_recording_is_self_contained(short_run, tmp_path):
    """A recording must render after its preset changes, so it embeds geometry."""
    from zimablue.replay.renderer import load_scene

    path = short_run.save(tmp_path / "run.zbr")
    scene = load_scene(Recording.load(path))
    assert scene.pool.floor_area > 0
    assert scene.swath > 0


def test_dirt_keyframes_are_sampled_not_dense(short_run):
    rec = short_run.recording
    assert 0 < len(rec.dirt_times) < rec.n_frames
    assert rec.dirt_keyframes.ndim == 4


def test_dirt_lookup_uses_the_preceding_keyframe(short_run):
    rec = short_run.recording
    first = rec.dirt_at(0.0).sum()
    last = rec.dirt_at(rec.duration).sum()
    assert last <= first, "dirt should not increase over a run"


def test_events_carry_time_and_kind(short_run):
    for event in short_run.recording.events:
        assert "time" in event and "kind" in event
        assert 0.0 <= event["time"] <= short_run.recording.duration + 1e-6


def test_loading_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no recording"):
        Recording.load(tmp_path / "nope.zbr")


def test_a_newer_schema_is_refused(short_run, tmp_path):
    """Better to refuse than to misread columns that have moved."""
    recording = short_run.recording
    recording.manifest["schema_version"] = SCHEMA_VERSION + 1
    path = recording.save(tmp_path / "future.zbr")
    with pytest.raises(ValueError, match="schema"):
        Recording.load(path)


def test_a_non_zimablue_zip_is_refused(tmp_path):
    path = tmp_path / "fake.zbr"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="not a ZimaBlue recording"):
        Recording.load(path)


def test_recording_can_be_disabled():
    result = Simulation(pool="rectangular", seed=1, record=False).run(seconds=20)
    assert result.recording is None
    with pytest.raises(RuntimeError, match="not recorded"):
        result.save("nowhere.zbr")


def test_unknown_channel_error_lists_channels(short_run):
    with pytest.raises(KeyError, match="available"):
        short_run.recording.column("nope")


# ----------------------------------------------------------------------
# Reading dirt back between keyframes
# ----------------------------------------------------------------------
def test_the_exact_reading_is_the_keyframe_and_says_so(short_run):
    """Default is the keyframe at or before t, unchanged and un-blended."""
    rec = short_run.recording
    assert len(rec.dirt_times) >= 2
    for i, t in enumerate(rec.dirt_times):
        exact = rec.dirt_at(float(t))
        assert exact == pytest.approx(rec.dirt_keyframes[i].sum(axis=0, dtype=np.float32))
        # On a keyframe there is nothing to blend, so both agree.
        assert rec.dirt_at(float(t), interpolate=True) == pytest.approx(exact)


def test_the_dirt_field_creeps_between_keyframes_instead_of_stepping(short_run):
    """The replay bug: dirt appeared in one frame rather than gradually.

    Keyframes are ten simulated seconds apart, so the nearest-keyframe reading
    holds a cell still for five hundred rendered frames and then jumps it. On
    the cell that gains the most dirt over a run that is the pile at the deep
    end appearing out of nowhere, over and over.
    """
    rec = short_run.recording
    first, last = rec.dirt_at(0.0), rec.dirt_at(rec.duration)
    row, col = np.unravel_index(np.argmax(last - first), first.shape)

    times = np.arange(0.0, rec.duration, 0.02)
    stepped = np.array([rec.dirt_at(t)[row, col] for t in times])
    blended = np.array([rec.dirt_at(t, interpolate=True)[row, col] for t in times])

    assert np.abs(np.diff(blended)).max() < 0.1 * np.abs(np.diff(stepped)).max()
    # And it moves every frame rather than a handful of times.
    assert (np.abs(np.diff(blended)) > 0).mean() > 0.9
    assert (np.abs(np.diff(stepped)) > 0).mean() < 0.1


def test_a_blend_never_leaves_the_pair_it_blends(short_run):
    """Interpolating must not invent dirt -- every value lies between two the
    cell genuinely held. It did not when the blend was computed in the float16
    the keyframes are stored in, which could land the result below both."""
    rec = short_run.recording
    before, after = (
        rec.dirt_keyframes[0].sum(axis=0, dtype=np.float32),
        rec.dirt_keyframes[1].sum(axis=0, dtype=np.float32),
    )
    low, high = np.minimum(before, after), np.maximum(before, after)
    for t in np.linspace(float(rec.dirt_times[0]), float(rec.dirt_times[1]), 12):
        blended = rec.dirt_at(float(t), interpolate=True)
        assert (blended >= low - 1e-6).all()
        assert (blended <= high + 1e-6).all()


def test_dirt_is_read_back_wider_than_it_is_stored(short_run):
    """float16 keeps the file small and is far too coarse to compute in."""
    assert short_run.recording.dirt_keyframes.dtype == np.float16
    assert short_run.recording.dirt_at(0.0).dtype == np.float32


@pytest.fixture(scope="module")
def autumn_run():
    """A run with discrete debris in it, which light sediment has none of."""
    return Simulation(pool="kidney", dirt="autumn", seed=4).run(seconds=120)


def test_debris_glides_between_keyframes_rather_than_teleporting(autumn_run):
    """The robot shoves oversized debris around, so positions really do move.

    Between two keyframes an item can travel metres, and the nearest-keyframe
    reading delivers all of it in one rendered frame.
    """
    rec = autumn_run.recording
    assert rec.debris_keyframes.size
    times = np.arange(0.0, rec.duration, 0.02)
    stepped = np.array([rec.debris_at(t)[:, 0:2] for t in times])
    blended = np.array([rec.debris_at(t, interpolate=True)[:, 0:2] for t in times])
    assert np.abs(np.diff(blended, axis=0)).max() < 0.1 * np.abs(np.diff(stepped, axis=0)).max()


def test_only_the_interpolated_reading_makes_collected_a_fraction(autumn_run):
    """Counting has to stay on the exact reading, which stays 0 or 1."""
    rec = autumn_run.recording
    assert rec.debris_keyframes.size
    for t in np.linspace(0.0, rec.duration, 25):
        exact = rec.debris_at(float(t))[:, 4]
        assert np.isin(exact, (0.0, 1.0)).all()
        fraction = rec.debris_at(float(t), interpolate=True)[:, 4]
        assert ((fraction >= 0.0) & (fraction <= 1.0)).all()


def test_a_keyframe_span_past_the_end_does_not_blend_off_the_array(short_run):
    rec = short_run.recording
    before, after, blend = rec.keyframe_span(rec.duration + 500.0)
    assert before == after == len(rec.dirt_times) - 1
    assert blend == 0.0
    assert rec.dirt_at(rec.duration + 500.0, interpolate=True) == pytest.approx(
        rec.dirt_at(rec.duration + 500.0)
    )


# ----------------------------------------------------------------------
# The parts a full run never exercises, because a full run is never empty,
# never disabled, and never has a channel that started late.


def _recording(**kwargs) -> Recording:
    manifest = {"format": "zbr", "version": 1, "seed": 7, "timestep": 0.05}
    manifest.update(kwargs.pop("manifest", {}))
    frames = kwargs.pop("frames", {"time": np.arange(0.0, 1.0, 0.05, dtype=np.float32)})
    return Recording(manifest=manifest, frames=frames, **kwargs)


def test_the_timestep_falls_back_to_the_manifest_when_there_is_nothing_to_measure():
    """One frame gives no interval; the manifest recorded one anyway."""
    single = _recording(frames={"time": np.zeros(1, dtype=np.float32)})
    assert single.frame_dt == pytest.approx(0.05)
    assert _recording().frame_dt == pytest.approx(0.05)


def test_the_seed_comes_off_the_manifest():
    assert _recording().seed == 7
    assert Recording(manifest={}, frames={"time": np.zeros(2)}).seed == 0


def test_a_time_between_frames_resolves_to_the_frame_before_it():
    recording = _recording()
    assert recording.frame_index_at(0.0) == 0
    assert recording.frame_index_at(0.13) == 2
    # Past either end it clamps rather than raising or wrapping.
    assert recording.frame_index_at(-5.0) == 0
    assert recording.frame_index_at(1e6) == recording.n_frames - 1


def test_a_recording_with_no_keyframes_has_a_span_anyway():
    """Dirt keyframes are optional; the player asks for a span regardless."""
    assert _recording().keyframe_span(3.0) == (0, 0, 0.0)


def test_a_keyframe_span_blends_between_the_two_it_falls_between():
    recording = _recording(dirt_times=np.array([0.0, 10.0], dtype=np.float32))
    assert recording.keyframe_span(0.0) == (0, 1, 0.0)
    before, after, blend = recording.keyframe_span(2.5)
    assert (before, after) == (0, 1)
    assert blend == pytest.approx(0.25)
    # Past the last keyframe both indices are the last one.
    assert recording.keyframe_span(99.0) == (1, 1, 0.0)


def test_events_are_selected_half_open_so_a_window_walk_sees_each_once():
    recording = _recording(
        events=[{"time": t, "kind": "bump"} for t in (0.0, 1.0, 2.0, 3.0)],
    )
    assert [e["time"] for e in recording.events_between(0.0, 2.0)] == [0.0, 1.0]
    assert [e["time"] for e in recording.events_between(2.0, 4.0)] == [2.0, 3.0]
    assert recording.events_between(9.0, 10.0) == []


def test_a_fleet_recording_says_how_many_robots_and_what_drove_them():
    recording = _recording(
        manifest={"fleet": {"count": 3, "controllers": ["bsa", "bsa", "random_bounce"]}}
    )
    described = recording.describe()
    assert "robots" in described
    assert "3" in described
    # Repeated controllers are named once, not three times.
    assert described.count("bsa") == 1


# ----------------------------------------------------------------------
def test_a_disabled_recorder_records_nothing_and_still_finishes():
    """Runs default to record=False; the recorder is built either way."""
    recorder = Recorder({"seed": 1}, enabled=False)
    recorder.add_frame({"time": 0.0, "x": 1.0})
    recorder.add_event({"time": 0.0, "kind": "bump"})
    finished = recorder.finish()
    assert finished.n_frames == 0
    assert finished.events == []


def test_a_channel_that_started_late_is_padded_so_the_table_stays_rectangular():
    """A sensor that comes online mid-run must not shorten every other column.

    Floats are padded with NaN to mark "no data yet"; an integer channel
    cannot hold NaN, so it pads with zero.
    """
    recorder = Recorder({"seed": 1})
    recorder.add_frame({"time": 0.0, "x": 1.0})
    recorder.add_frame({"time": 0.1, "x": 2.0, "contacts": 1, "sonar": 0.5})
    finished = recorder.finish()

    assert {len(c) for c in finished.frames.values()} == {2}
    assert np.isnan(finished.column("sonar")[0]), "a late float channel reads as no data"
    assert finished.column("contacts")[0] == 0, "an int channel cannot hold NaN"
    assert finished.column("contacts").dtype == np.int32


def test_finishing_merges_extra_manifest_keys_over_the_originals():
    recorder = Recorder({"seed": 1, "scenario": "old"})
    recorder.add_frame({"time": 0.0})
    finished = recorder.finish(extra_manifest={"scenario": "new", "pool": "kidney"})
    assert finished.manifest["scenario"] == "new"
    assert finished.manifest["pool"] == "kidney"
    assert finished.manifest["seed"] == 1
    assert finished.manifest["format"] == "zbr"
