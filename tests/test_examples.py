"""The examples have to keep working.

Documentation that no longer runs is worse than none, and examples rot the
moment an API changes underneath them. Each is executed as a subprocess -- the
way a reader will run it.

Every example takes ``--minutes`` so this can ask for a two-minute run instead
of the twenty the defaults use. Without that these tests took longer than the
rest of the suite put together, which is how example tests end up deleted.
"""

from __future__ import annotations

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


def test_batch_experiment_runs_and_reproduces_its_worst_episode():
    result = run("batch_experiment.py", "--episodes", "3", "--minutes", "3")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "identical: True" in result.stdout, "the determinism check should pass"
