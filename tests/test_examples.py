"""The examples have to keep working.

Documentation that no longer runs is worse than none, and examples rot the
moment an API changes underneath them. Each is executed as a subprocess -- the
way a reader will run it.

Every example takes ``--minutes`` so this can ask for a two-minute run instead
of the twenty the defaults use. Without that these tests took longer than the
rest of the suite put together, which is how example tests end up deleted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def run(script: str, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        check=False,
    )


def test_every_example_is_listed_in_the_readme():
    """A new example nobody can find is not documentation."""
    readme = (ROOT / "README.md").read_text()
    for script in sorted(EXAMPLES.glob("*.py")):
        assert script.name in readme, f"{script.name} is not mentioned in the README"


@pytest.mark.parametrize("script", sorted(p.name for p in EXAMPLES.glob("*.py")))
def test_example_has_a_usage_docstring(script: str):
    text = (EXAMPLES / script).read_text()
    assert text.startswith('#!/usr/bin/env python3\n"""'), f"{script} needs a shebang + docstring"
    assert "python examples/" in text, f"{script} should show how to run it"


def test_basic_runs():
    result = run("basic.py", "--minutes", "2")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "coverage" in result.stdout


def test_custom_robot_runs():
    result = run("custom_robot.py", "--minutes", "2")
    assert result.returncode == 0, result.stderr[-2000:]


def test_custom_pool_runs():
    result = run("custom_pool.py", "--minutes", "2")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "lap_pool" in result.stdout
    assert "never visited" in result.stdout, "should report spatial coverage"


def test_custom_controller_runs():
    result = run("custom_controller.py", "--minutes", "2")
    assert result.returncode == 0, result.stderr[-2000:]


def test_pool_from_photo_runs(tmp_path):
    result = run("pool_from_photo.py", "--minutes", "2", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "traced" in result.stdout
    # It synthesises a photo of a known pool, so it can grade itself.
    assert "came within" in result.stdout
    assert (tmp_path / "trace_overlay.png").exists()


def test_rl_env_runs_and_ranks_the_baseline_above_random():
    pytest.importorskip("gymnasium")
    result = run("rl_env.py", "--minutes", "2")
    assert result.returncode == 0, result.stderr[-2000:]

    scores = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("random", "straight", "baseline_coverage"):
            scores[parts[0]] = float(parts[1])
    assert scores["baseline_coverage"] > scores["random"], (
        "the shipped controller should beat a random policy, or the example is "
        "not measuring what it says it is"
    )


def test_tune_controller_improves_on_the_defaults():
    result = run("tune_controller.py", "--minutes", "2", "--episodes", "1", "--iterations", "4")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "(defaults)" in result.stdout
    # A search that cannot beat its own starting point on any objective is
    # either broken or scoring noise.
    assert "<-- kept" in result.stdout


def test_replay_real_trajectory_runs_when_a_log_is_present(tmp_path):
    """Skips without the data, because the data is somebody else's and 3 MB.

    ``tools/fetch_trajectory.py`` downloads it; see docs/hardware.md.
    """
    if not sorted((ROOT / "data" / "trajectories").glob("*.txt")):
        pytest.skip("no trajectories fetched; run tools/fetch_trajectory.py")
    result = run("replay_real_trajectory.py", "--save", str(tmp_path))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "final" in result.stdout
    assert sorted(tmp_path.glob("*.zbr")), "each log should produce a replayable recording"


def test_analyse_dynamics_runs_and_reports_every_section():
    result = run("analyse_dynamics.py", "--minutes", "4", "--seeds", "2")
    assert result.returncode == 0, result.stderr[-2000:]
    for heading in ("contacts", "mixing", "wasted", "lambda", "forecast err"):
        assert heading in result.stdout, f"the {heading} section is missing"


def test_batch_experiment_runs_and_reproduces_its_worst_episode():
    result = run("batch_experiment.py", "--episodes", "3", "--minutes", "3")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "identical: True" in result.stdout, "the determinism check should pass"


# ----------------------------------------------------------------------
# The tour notebook
# ----------------------------------------------------------------------
NOTEBOOK = EXAMPLES / "tour.ipynb"

RUNNER = """
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
cells = json.loads(pathlib.Path(sys.argv[1]).read_text())["cells"]
namespace = {"__name__": "__main__"}
for index, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    # Magics are notebook syntax, not Python. The tour needs %matplotlib
    # inline so its figures become outputs when it is executed for real; here
    # the Agg backend is already set, so dropping the line is exactly right.
    source = "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))
    try:
        exec(compile(source, f"<cell {index}>", "exec"), namespace)
    except Exception:
        print(f"cell {index} failed:\\n{source}", file=sys.stderr)
        raise
"""


def test_the_notebook_is_listed_in_the_readme():
    assert NOTEBOOK.name in (ROOT / "README.md").read_text()


def test_the_notebook_is_valid_and_executed():
    """The tour is committed with its outputs.

    A tour whose plots only appear if you run it is a worse tour, and GitHub
    renders a stored notebook directly.
    """
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) > 10, "the tour should actually cover the library"

    executed = [c for c in code_cells if c["outputs"]]
    assert len(executed) >= len(code_cells) - 2, "commit the notebook with its outputs"
    images = sum(1 for c in code_cells for o in c["outputs"] if "image/png" in o.get("data", {}))
    assert images >= 5, f"only {images} figures stored -- was %matplotlib inline dropped?"


def test_notebook_source_lines_end_in_newlines():
    """Otherwise every renderer runs the markdown together.

    nbformat stores a cell as a list of lines, each keeping its own line
    ending. Without them a heading and the paragraph under it are displayed as
    one sentence, which is how the tour first shipped.
    """
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        lines = cell["source"]
        for line in lines[:-1]:
            assert line.endswith("\n"), f"cell {index} has a line with no newline: {line!r}"


def test_every_notebook_cell_runs(tmp_path):
    """A notebook nobody executes is a notebook that has already broken.

    Run as a script rather than through a kernel: this suite should not need
    jupyter installed to find out that the tour no longer imports.
    """
    script = tmp_path / "run_notebook.py"
    script.write_text(RUNNER)
    result = subprocess.run(
        [sys.executable, str(script), str(NOTEBOOK)],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=ROOT,
        env={**os.environ, "ZIMABLUE_TOUR_MINUTES": "1", "MPLBACKEND": "Agg"},
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
