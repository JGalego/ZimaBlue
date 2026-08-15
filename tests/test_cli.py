"""The command line."""

from __future__ import annotations

import matplotlib
from typer.testing import CliRunner

matplotlib.use("Agg", force=True)

from zimablue._version import __version__
from zimablue.cli import app

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("demo", "run", "replay", "batch", "inspect", "list"):
        assert command in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_list_shows_presets():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    for name in ("kidney", "tracked", "autumn", "baseline_coverage", "fast2d"):
        assert name in result.stdout


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
