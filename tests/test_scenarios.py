"""Scenario loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from zimablue.scenarios import Scenario, load_scenario

SCENARIO_DIR = Path("scenarios")


@pytest.mark.parametrize("path", sorted(SCENARIO_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_scenarios_load(path):
    scenario = load_scenario(path)
    assert scenario.name
    assert scenario.duration > 0
    assert scenario.timestep > 0
    scenario.build_pool()
    scenario.build_robot()
    scenario.build_dirt()
    scenario.build_controller()


def test_scenario_round_trips_through_yaml(tmp_path):
    original = load_scenario(SCENARIO_DIR / "kidney.yaml")
    path = tmp_path / "copy.yaml"
    path.write_text(original.to_yaml())
    restored = load_scenario(path)
    assert restored.pool == original.pool
    assert restored.robot == original.robot
    assert restored.dirt == original.dirt
    assert restored.seed == original.seed
    assert restored.duration == original.duration


def test_a_bare_preset_name_is_accepted():
    scenario = Scenario.from_dict({"name": "t", "pool": "oval", "robot": "compact"})
    assert scenario.pool == "oval"
    assert scenario.robot == "compact"


def test_unknown_top_level_key_is_rejected():
    """Silently ignoring a typo would run a different experiment than intended."""
    with pytest.raises(ValueError, match="unknown key"):
        Scenario.from_dict({"name": "t", "polo": {"preset": "kidney"}})


def test_unknown_simulation_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        Scenario.from_dict({"name": "t", "simulation": {"duratoin": 10}})


def test_unknown_preset_is_rejected_with_alternatives():
    with pytest.raises(ValueError, match="Available"):
        Scenario.from_dict({"name": "t", "pool": {"preset": "hexagonal"}})


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no scenario"):
        load_scenario(tmp_path / "absent.yaml")


def test_invalid_yaml_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_scenario(path)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ValueError, match="mapping"):
        load_scenario(path)


def test_scenario_runs_and_produces_metrics():
    scenario = load_scenario(SCENARIO_DIR / "rectangular.yaml")
    scenario.duration = 45.0
    result = scenario.run()
    assert result.metrics.runtime == pytest.approx(45.0, abs=0.1)
    assert result.recording.manifest["scenario"]["name"] == "rectangular"


def test_seed_override_changes_the_run():
    scenario = load_scenario(SCENARIO_DIR / "rectangular.yaml")
    scenario.duration = 30.0
    assert scenario.run(seed=1).metrics.coverage != scenario.run(seed=2).metrics.coverage
