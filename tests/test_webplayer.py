"""The recording as a web page."""

from __future__ import annotations

import base64
import json

import pytest

import zimablue as zb
from zimablue.replay.webplayer import export_web_player


@pytest.fixture(scope="module")
def recording():
    result = zb.Simulation(pool="rectangular", dirt="autumn", seed=3).run(minutes=1.0)
    return result.require_recording()


def payload_of(path):
    text = path.read_text()
    start = text.index("const DATA = ") + len("const DATA = ")
    end = text.index(";\n", start)
    return json.loads(text[start:end].replace("<\\/", "</"))


def test_a_single_recording_becomes_one_page(tmp_path, recording):
    path = export_web_player(recording, tmp_path / "run.html")
    text = path.read_text()
    assert text.startswith("<!DOCTYPE html>")
    assert "<canvas" not in text  # canvases are created by the script
    assert "getContext" in text

    data = payload_of(path)
    run = data["runs"][0]
    assert run["duration"] == pytest.approx(60.0, abs=1.0)
    assert len(run["x"]) == len(run["y"]) == len(run["time"])
    assert run["pool"]["exterior"], "the pool outline must be embedded"


def test_the_dirt_blob_decodes_to_its_declared_shape(tmp_path, recording):
    data = payload_of(export_web_player(recording, tmp_path / "run.html"))
    dirt = data["runs"][0]["dirt"]
    blob = base64.b64decode(dirt["data"])
    assert len(blob) == len(dirt["times"]) * dirt["rows"] * dirt["cols"]
    assert max(blob) > 0, "an autumn pool quantised to all zeros lost the dirt"


def test_two_recordings_share_one_clock(tmp_path, recording):
    other = zb.Simulation(
        pool="rectangular", dirt="autumn", controller="random_bounce", seed=3
    ).run(minutes=0.5)
    path = export_web_player(
        {"baseline": recording, "bounce": other.require_recording()}, tmp_path / "duel.html"
    )
    data = payload_of(path)
    assert [run["name"] for run in data["runs"]] == ["baseline", "bounce"]
    assert data["duration"] == pytest.approx(max(r["duration"] for r in data["runs"]))


def test_the_page_needs_no_network(tmp_path, recording):
    text = export_web_player(recording, tmp_path / "run.html").read_text()
    for marker in ("http://", "https://", "src=", "@import"):
        assert marker not in text


def test_the_cli_writes_it_without_matplotlib_paths(tmp_path, recording):
    from typer.testing import CliRunner

    from zimablue.cli import app

    zbr = tmp_path / "r.zbr"
    recording.save(zbr)
    out = tmp_path / "r.html"
    result = CliRunner().invoke(app, ["replay", str(zbr), "--html", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
