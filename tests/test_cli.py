"""The command line."""

from __future__ import annotations

import json

import matplotlib
import pytest
from typer.testing import CliRunner

matplotlib.use("Agg", force=True)

from zimablue._version import __version__
from zimablue.cli import app

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("demo", "run", "replay", "batch", "compare", "bench", "inspect", "list"):
        assert command in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_list_shows_presets():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    for name in ("kidney", "tracked", "autumn", "baseline_coverage", "fast2d", "sweep_optimal"):
        assert name in result.stdout


def test_compare_ranks_planners(tmp_path):
    csv = tmp_path / "trials.csv"
    result = runner.invoke(
        app,
        [
            "compare",
            "random_bounce",
            "systematic",
            "--minutes",
            "0.5",
            "--csv",
            str(csv),
        ],
    )
    assert result.exit_code == 0, result.stdout
    lines = csv.read_text().splitlines()
    assert lines[0].startswith("planner,pool,seed,coverage")
    # The table itself may be cropped to the terminal width; the CSV is the
    # full record, one row per trial.
    assert {line.split(",")[0] for line in lines[1:]} == {"random_bounce", "systematic"}


def test_compare_rejects_a_typo_before_running():
    result = runner.invoke(app, ["compare", "boustrophedont@odometry"])
    assert result.exit_code == 1
    assert "unknown planner" in result.stdout
    assert "boustrophedon" in result.stdout


def test_compare_rejects_an_unknown_fleet_entry():
    result = runner.invoke(app, ["compare", "darp+nope", "--fleet", "2"])
    assert result.exit_code == 1
    assert "unknown planner" in result.stdout


def test_run_executes_a_scenario(tmp_path):
    out = tmp_path / "run.zbr"
    result = runner.invoke(
        app,
        ["run", "scenarios/rectangular.yaml", "--minutes", "0.5", "--record", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert "coverage" in result.stdout


def test_run_reports_a_missing_scenario_clearly():
    result = runner.invoke(app, ["run", "scenarios/does_not_exist.yaml"])
    assert result.exit_code == 1
    assert "error" in result.stdout


def test_run_reports_an_invalid_scenario_clearly(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\npolo: {preset: kidney}\n")
    result = runner.invoke(app, ["run", str(bad)])
    assert result.exit_code == 1
    assert "unknown key" in result.stdout


def test_demo_runs_without_a_display(tmp_path):
    result = runner.invoke(
        app,
        ["demo", "--minutes", "0.5", "--pool", "rectangular", "--no-watch", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert list(tmp_path.glob("*.zbr")), "demo should leave a recording behind"
    assert list(tmp_path.glob("*_summary.png"))


def test_demo_rejects_an_unknown_preset():
    result = runner.invoke(app, ["demo", "--pool", "hexagonal", "--no-watch"])
    assert result.exit_code == 1
    assert "unknown pool preset" in result.stdout


def test_inspect_describes_a_recording(tmp_path):
    out = tmp_path / "r.zbr"
    runner.invoke(
        app, ["run", "scenarios/rectangular.yaml", "--minutes", "0.5", "--record", str(out)]
    )
    result = runner.invoke(app, ["inspect", str(out), "--events", "--channels"])
    assert result.exit_code == 0
    assert "zbr v1" in result.stdout
    assert "channels" in result.stdout


def test_inspect_reports_a_missing_recording(tmp_path):
    result = runner.invoke(app, ["inspect", str(tmp_path / "none.zbr")])
    assert result.exit_code == 1
    assert "error" in result.stdout


def test_batch_reports_aggregates(tmp_path):
    result = runner.invoke(
        app,
        [
            "batch",
            "scenarios/rectangular.yaml",
            "--episodes",
            "2",
            "--minutes",
            "0.5",
            "--out",
            str(tmp_path / "b.json"),
            "--csv",
            str(tmp_path / "b.csv"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "mean_coverage" in result.stdout
    assert (tmp_path / "b.json").exists()
    assert (tmp_path / "b.csv").exists()


def test_replay_renders_a_summary(tmp_path):
    out = tmp_path / "r.zbr"
    runner.invoke(
        app, ["run", "scenarios/rectangular.yaml", "--minutes", "0.5", "--record", str(out)]
    )
    png = tmp_path / "s.png"
    result = runner.invoke(app, ["replay", str(out), "--summary", str(png)])
    assert result.exit_code == 0, result.stdout
    assert png.exists()


def test_replay_without_a_display_explains_itself(tmp_path):
    out = tmp_path / "r.zbr"
    runner.invoke(
        app, ["run", "scenarios/rectangular.yaml", "--minutes", "0.5", "--record", str(out)]
    )
    result = runner.invoke(app, ["replay", str(out)])
    assert result.exit_code == 0
    assert "--gif" in result.stdout


def test_viz_hint_survives_rich_markup():
    """Rich reads [viz] as a style tag and silently swallows it unless escaped.

    The hint is useless if it renders as "pip install 'zimablue'", so assert on
    what the user actually sees rather than on the source string.
    """
    import io

    from rich.console import Console

    import zimablue.cli as cli

    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False)
    for message, hint in ((cli._VIZ_MISSING, None), ("x", cli._VIZ_HINT)):
        console.print(message if hint is None else hint)
    rendered = buffer.getvalue()
    assert rendered.count("zimablue[viz]") == 2, rendered


# ----------------------------------------------------------------------
# The rest of the surface: the flags that pick a camera or a file format, and
# the mistakes a user makes on the way to them.
#
# These share one short recording. Building a new one per test is a simulated
# run each time, and the thing under test is the argument handling, not the
# simulator.


@pytest.fixture(scope="module")
def zbr(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "run.zbr"
    result = runner.invoke(
        app, ["run", "scenarios/rectangular.yaml", "--minutes", "0.4", "--record", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    return out


def test_run_stays_quiet_when_asked(tmp_path):
    result = runner.invoke(
        app, ["run", "scenarios/rectangular.yaml", "--minutes", "0.3", "--quiet"]
    )
    assert result.exit_code == 0, result.stdout
    assert "coverage" in result.stdout
    assert "simulating" not in result.stdout, "--quiet should drop the progress bar"


def test_run_takes_a_built_in_scenario_name_and_a_seed_override(tmp_path):
    out = tmp_path / "seeded.zbr"
    result = runner.invoke(
        app,
        ["run", "rectangular", "--minutes", "0.3", "--seed", "99", "--record", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()


def test_run_writes_a_summary_alongside_the_recording(tmp_path):
    png = tmp_path / "s.png"
    result = runner.invoke(
        app,
        [
            "run",
            "scenarios/rectangular.yaml",
            "--minutes",
            "0.3",
            "--record",
            str(tmp_path / "r.zbr"),
            "--summary",
            str(png),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert png.exists()


def test_a_summary_without_a_recording_says_what_is_missing(tmp_path):
    """Nothing is recorded unless --record asked for it, and a summary is
    drawn from the recording."""
    result = runner.invoke(
        app,
        [
            "run",
            "scenarios/rectangular.yaml",
            "--minutes",
            "0.3",
            "--summary",
            str(tmp_path / "s.png"),
        ],
    )
    assert result.exit_code == 1
    assert "--summary needs a recording" in result.stdout
    assert "--record" in result.stdout


# ----------------------------------------------------------------------
def test_replay_writes_still_frames_to_a_directory(tmp_path, zbr):
    frames = tmp_path / "frames"
    result = runner.invoke(app, ["replay", str(zbr), "--frames", str(frames)])
    assert result.exit_code == 0, result.stdout
    assert list(frames.glob("*.png"))
    assert "frames to" in result.stdout


def test_replay_renders_a_gif(tmp_path, zbr):
    gif = tmp_path / "run.gif"
    result = runner.invoke(app, ["replay", str(zbr), "--gif", str(gif)])
    assert result.exit_code == 0, result.stdout
    assert gif.stat().st_size > 0


def test_two_cameras_at_once_is_refused_by_name(zbr):
    """The check is over the set, not pairwise -- adding a third camera to an
    'if a and b' left a hole that this pins shut."""
    result = runner.invoke(app, ["replay", str(zbr), "--3d", "--chase"])
    assert result.exit_code == 1
    assert "--3d" in result.stdout and "--chase" in result.stdout
    assert "different cameras" in result.stdout


def test_all_three_cameras_at_once_names_all_three(zbr):
    result = runner.invoke(app, ["replay", str(zbr), "--3d", "--chase", "--dirtcam"])
    assert result.exit_code == 1
    for flag in ("--3d", "--chase", "--dirtcam"):
        assert flag in result.stdout


@pytest.mark.parametrize("camera", ["--dirtcam", "--chase"])
def test_a_robot_eye_camera_needs_somewhere_to_write(zbr, camera):
    result = runner.invoke(app, ["replay", str(zbr), camera])
    assert result.exit_code == 1
    assert "renders to a file" in result.stdout
    assert camera in result.stdout


@pytest.mark.parametrize("camera", ["--dirtcam", "--chase"])
def test_a_robot_eye_camera_writes_stills(tmp_path, zbr, camera):
    png = tmp_path / f"{camera.strip('-')}.png"
    result = runner.invoke(app, ["replay", str(zbr), camera, "--summary", str(png)])
    assert result.exit_code == 0, result.stdout
    assert png.exists()


def test_the_dirt_cam_can_drop_the_map_panel(tmp_path, zbr):
    gif = tmp_path / "bumper.gif"
    result = runner.invoke(app, ["replay", str(zbr), "--dirtcam", "--no-map", "--gif", str(gif)])
    assert result.exit_code == 0, result.stdout
    assert gif.stat().st_size > 0


def test_3d_replay_renders_to_a_file_not_a_window(zbr):
    result = runner.invoke(app, ["replay", str(zbr), "--3d"])
    assert result.exit_code == 1
    assert "renders to a file" in result.stdout


def test_3d_replay_writes_stills(tmp_path, zbr):
    png = tmp_path / "basin.png"
    result = runner.invoke(app, ["replay", str(zbr), "--3d", "--summary", str(png)])
    assert result.exit_code == 0, result.stdout
    assert png.exists()


# ----------------------------------------------------------------------
def test_inspect_prints_the_recorded_metrics(tmp_path, zbr):
    result = runner.invoke(app, ["inspect", str(zbr)])
    assert result.exit_code == 0, result.stdout
    assert "recorded metrics" in result.stdout


def test_inspect_does_not_invent_coverage_for_a_robot_log(tmp_path, zbr):
    """A recording from real hardware carries a shorter set of metrics.

    Pouring those into Metrics.from_dict zero-fills the missing keys, which
    reads as a robot that drove nowhere rather than as a run nobody measured.
    """
    from zimablue.recording import Recording

    rec = Recording.load(zbr)
    rec.metrics = {"distance": 12.5, "duration": 24.0}
    rec.manifest["ground_truth"] = False
    stripped = tmp_path / "robot.zbr"
    rec.save(stripped)
    assert not Recording.load(stripped).has_ground_truth

    result = runner.invoke(app, ["inspect", str(stripped)])
    assert result.exit_code == 0, result.stdout
    assert "12.5" in result.stdout
    assert "No ground truth" in result.stdout
    # The full metrics table would zero-fill these; the short one must not
    # print them at all.
    for invented in ("dirt removed", "battery left", "stuck events"):
        assert invented not in result.stdout


# ----------------------------------------------------------------------
def test_bench_runs_the_smoke_tier_and_writes_the_leaderboard(tmp_path):
    result = runner.invoke(app, ["bench", "--quick", "--out", str(tmp_path / "bench")])
    assert result.exit_code == 0, result.stdout
    written = list((tmp_path / "bench").iterdir())
    assert {p.suffix for p in written} >= {".json", ".csv", ".md"}, written


def test_compare_draws_the_matrix_when_asked(tmp_path):
    png = tmp_path / "matrix.png"
    result = runner.invoke(
        app,
        ["compare", "random_bounce", "systematic", "--minutes", "0.3", "--matrix", str(png)],
    )
    assert result.exit_code == 0, result.stdout
    assert png.stat().st_size > 0


def test_compare_rejects_an_unknown_localisation():
    result = runner.invoke(app, ["compare", "--localisation", "gps"])
    assert result.exit_code == 1
    assert "unknown localisation" in result.stdout


def test_compare_rejects_an_unknown_localisation_suffix():
    result = runner.invoke(app, ["compare", "sweep_optimal@gps"])
    assert result.exit_code == 1
    assert "use @truth or @odometry" in result.stdout


def test_compare_rejects_an_unknown_controller():
    result = runner.invoke(app, ["compare", "wander_aimlessly"])
    assert result.exit_code == 1
    assert "unknown entry" in result.stdout


def test_compare_rejects_an_unknown_partition_in_a_fleet_entry():
    result = runner.invoke(app, ["compare", "nope+sweep_optimal", "--fleet", "2"])
    assert result.exit_code == 1
    assert "unknown partition" in result.stdout


def test_compare_rejects_an_unknown_fleet_controller():
    result = runner.invoke(app, ["compare", "wander_aimlessly", "--fleet", "2"])
    assert result.exit_code == 1
    assert "unknown entry" in result.stdout


def test_compare_runs_a_fleet(tmp_path):
    csv = tmp_path / "fleet.csv"
    result = runner.invoke(
        app, ["compare", "mstc", "--fleet", "2", "--minutes", "0.3", "--csv", str(csv)]
    )
    assert result.exit_code == 0, result.stdout
    assert "fleet comparison" in result.stdout
    assert csv.exists()


def test_compare_says_how_many_runs_a_median_is_over(tmp_path):
    result = runner.invoke(app, ["compare", "random_bounce", "--minutes", "0.2", "--seeds", "2"])
    assert result.exit_code == 0, result.stdout
    assert "median over" in result.stdout


# ----------------------------------------------------------------------
def test_demo_renders_a_gif_when_asked(tmp_path):
    result = runner.invoke(
        app,
        [
            "demo",
            "--minutes",
            "0.3",
            "--pool",
            "rectangular",
            "--no-watch",
            "--gif",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert list(tmp_path.glob("*.gif"))


def test_watching_without_a_display_explains_itself_rather_than_opening_nothing():
    """conftest forces Agg, which is what a headless machine gets too."""
    import zimablue.cli as cli

    cli._watch(object())  # never reaches the player, so the argument is unused


# ----------------------------------------------------------------------
# Tracing a pool out of a photograph, from the command line.
#
# The tracing itself is checked in test_imaging.py against a scene with a
# known answer. What is checked here is the wrapper around it: the scale
# argument that has no default, the sample pixel's little syntax, and the
# overlay nag -- segmenting a photo is a guess, and the CLI's job is to make
# sure you look at it.


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    """The kidney-pool photograph from test_imaging, written to a file.

    Returns the path, a pixel known to be inside the water, and the pool's
    real width -- which is the scale the command refuses to guess.
    """
    from test_imaging import TRUTH, _scene, _seed_pixel

    image, project = _scene()
    path = tmp_path_factory.mktemp("photo") / "pool.png"
    image.save(path)
    minx, _, maxx, _ = TRUTH.boundary.bounds
    return path, _seed_pixel(project), maxx - minx


def test_trace_reads_a_pool_out_of_a_photograph(tmp_path, photo):
    path, sample, width = photo
    out = tmp_path / "pool.json"
    result = runner.invoke(
        app,
        [
            "trace",
            str(path),
            "--width",
            str(width),
            "--sample",
            f"{sample[0]},{sample[1]}",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["name"] == "traced"
    assert "no --check given" in result.stdout, "a guess you never looked at is not a pool"


def test_trace_writes_the_overlay_that_catches_a_wrong_guess(tmp_path, photo):
    path, sample, width = photo
    check = tmp_path / "overlay.png"
    result = runner.invoke(
        app,
        [
            "trace",
            str(path),
            "--width",
            str(width),
            "--sample",
            f"{sample[0]},{sample[1]}",
            "--check",
            str(check),
            "--name",
            "backyard",
            "--depth",
            "2.0",
            "--out",
            str(tmp_path / "p.json"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert check.exists()
    assert "no --check given" not in result.stdout
    assert json.loads((tmp_path / "p.json").read_text())["name"] == "backyard"


def test_trace_insists_on_exactly_one_scale(photo):
    """A photograph does not carry its own scale, so there is no default."""
    path, _, width = photo
    neither = runner.invoke(app, ["trace", str(path)])
    both = runner.invoke(app, ["trace", str(path), "--width", str(width), "--mpp", "0.01"])
    for result in (neither, both):
        assert result.exit_code == 1
        assert "exactly one of --width or --mpp" in result.stdout


def test_trace_explains_a_malformed_sample(photo):
    path, _, width = photo
    result = runner.invoke(app, ["trace", str(path), "--width", str(width), "--sample", "middle"])
    assert result.exit_code == 1
    assert "--sample 640,410" in result.stdout


def test_trace_reports_a_missing_picture(tmp_path):
    result = runner.invoke(app, ["trace", str(tmp_path / "nope.png"), "--width", "8"])
    assert result.exit_code == 1
    assert "error" in result.stdout


def test_trace_accepts_a_ground_resolution_instead_of_a_width(tmp_path, photo):
    path, sample, width = photo
    from PIL import Image

    pixels = Image.open(path).size[0]
    result = runner.invoke(
        app,
        [
            "trace",
            str(path),
            "--mpp",
            str(width / pixels),
            "--sample",
            f"{sample[0]},{sample[1]}",
            "--out",
            str(tmp_path / "mpp.json"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "mpp.json").exists()
