"""The .zbr format."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from zimablue.recording import FORMAT, SCHEMA_VERSION, Recording
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
