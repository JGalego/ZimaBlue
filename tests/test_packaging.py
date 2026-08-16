"""What a `pip install zimablue` actually gets.

These are the things that only break for users, never for developers: the repo
has a `scenarios/` directory and a full dev environment, so a wheel that ships
neither still passes every other test in this suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

import zimablue as zb
from zimablue.scenarios import bundled_scenarios, load_scenario, resolve_scenario

ROOT = Path(__file__).resolve().parent.parent


def test_version_has_one_source():
    """pyproject reads the version from the module, so they cannot drift."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in config["project"].get("dynamic", []), (
        "pyproject should take the version from src/zimablue/_version.py"
    )
    assert config["tool"]["hatch"]["version"]["path"] == "src/zimablue/_version.py"


def test_scenarios_are_declared_as_package_data():
    """Without this, every documented `zimablue run` needs a git clone."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert include["scenarios"] == "zimablue/data/scenarios"


def test_repo_scenarios_all_load():
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        assert scenario.seed >= 0
        assert scenario.duration > 0


def test_a_bare_name_resolves_when_scenarios_are_bundled():
    bundled = bundled_scenarios()
    if not bundled:
        pytest.skip("running from a source tree without the built data directory")
    name = next(iter(bundled))
    assert resolve_scenario(name).exists()
    assert load_scenario(name).seed >= 0


def test_a_local_path_wins_over_a_bundled_name(tmp_path):
    local = tmp_path / "kidney.yaml"
    local.write_text("name: local\npool:\n  preset: rectangular\nseed: 7\n")
    assert resolve_scenario(local) == local


def test_unknown_scenario_names_the_alternatives():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_scenario("definitely-not-a-scenario")
    message = str(excinfo.value)
    assert "definitely-not-a-scenario" in message
    assert "built-in names" in message


def test_importing_zimablue_does_not_import_matplotlib():
    """The core install has no matplotlib, so importing it at module scope
    would make `import zimablue` fail for everyone who skipped the extra."""
    import subprocess
    import sys

    code = "import zimablue, sys; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_public_api_is_importable_and_complete():
    missing = [name for name in zb.__all__ if not hasattr(zb, name)]
    assert not missing, f"__all__ lists names that do not exist: {missing}"


def test_viz_hint_names_the_extra():
    from zimablue.replay import VIZ_HINT

    assert "zimablue[viz]" in VIZ_HINT
