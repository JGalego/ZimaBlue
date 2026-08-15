"""Scenarios: an experiment as a file.

A scenario names everything a run needs -- pool, robot, dirt, controller,
seed, duration, termination -- so that "reproduce this" is a filename plus a
seed rather than a paragraph of instructions.

```yaml
name: autumn_kidney_pool
seed: 42
pool:
  preset: kidney
robot:
  preset: tracked
dirt:
  preset: autumn
simulation:
  duration: 1800
  timestep: 0.02
controller:
  preset: baseline_coverage
```

Unknown keys are rejected rather than ignored. A typo in an experiment
definition that silently runs the default is worse than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from zimablue.dirt import DIRT_PRESETS, DirtSpec, make_dirt
from zimablue.pool import DEFAULT_CELL, POOL_PRESETS, Pool, make_pool
from zimablue.robot import ROBOT_PRESETS, Cleaner, make_robot
from zimablue.simulation import RunResult, Simulation

__all__ = ["Scenario", "load_scenario"]

_TOP_LEVEL = {
    "name",
    "description",
    "seed",
    "pool",
    "robot",
    "dirt",
    "simulation",
    "controller",
    "termination",
}


def _check_keys(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in scenario section {section!r}. "
            f"Allowed: {sorted(allowed)}"
        )


def _preset_section(
    section: str, data: Any, registry_names: list[str]
) -> tuple[str, dict[str, Any]]:
    """Parse ``{preset: name, ...overrides}`` or a bare preset name."""
    if isinstance(data, str):
        return data, {}
    if not isinstance(data, dict):
        raise ValueError(f"scenario section {section!r} must be a name or a mapping")
    _check_keys(section, data, {"preset", "params"})
    name = data.get("preset")
    if name is None:
        raise ValueError(f"scenario section {section!r} needs a 'preset' key")
    if name not in registry_names:
        raise ValueError(
            f"unknown {section} preset {name!r}. Available: {', '.join(registry_names)}"
        )
    return str(name), dict(data.get("params") or {})


@dataclass
class Scenario:
    """A fully specified experiment."""

    name: str = "scenario"
    description: str = ""
    seed: int = 0

    pool: str = "rectangular"
    pool_params: dict[str, Any] = field(default_factory=dict)
    robot: str = "tracked"
    robot_params: dict[str, Any] = field(default_factory=dict)
    dirt: str = "light_sediment"
    dirt_params: dict[str, Any] = field(default_factory=dict)
    controller: str = "baseline_coverage"
    controller_params: dict[str, Any] = field(default_factory=dict)

    duration: float = 1800.0
    timestep: float = 0.02
    cell: float = DEFAULT_CELL
    backend: str = "fast2d"

    coverage_target: float | None = None
    dirt_target: float | None = None
    stop_on_empty_battery: bool = True

    source: Path | None = None

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> Scenario:
        _check_keys("(top level)", data, _TOP_LEVEL)

        pool, pool_params = _preset_section(
            "pool", data.get("pool", "rectangular"), POOL_PRESETS.names()
        )
        robot, robot_params = _preset_section(
            "robot", data.get("robot", "tracked"), ROBOT_PRESETS.names()
        )
        dirt, dirt_params = _preset_section(
            "dirt", data.get("dirt", "light_sediment"), DIRT_PRESETS.names()
        )

        controller_raw = data.get("controller", "baseline_coverage")
        if isinstance(controller_raw, str):
            controller, controller_params = controller_raw, {}
        else:
            _check_keys("controller", controller_raw, {"preset", "params"})
            controller = str(controller_raw.get("preset", "baseline_coverage"))
            controller_params = dict(controller_raw.get("params") or {})

        sim = dict(data.get("simulation") or {})
        _check_keys("simulation", sim, {"duration", "timestep", "cell", "backend"})

        term = dict(data.get("termination") or {})
        _check_keys(
            "termination", term, {"coverage_target", "dirt_target", "stop_on_empty_battery"}
        )

        return cls(
            name=str(data.get("name", source.stem if source else "scenario")),
            description=str(data.get("description", "")),
            seed=int(data.get("seed", 0)),
            pool=pool,
            pool_params=pool_params,
            robot=robot,
            robot_params=robot_params,
            dirt=dirt,
            dirt_params=dirt_params,
            controller=controller,
            controller_params=controller_params,
            duration=float(sim.get("duration", 1800.0)),
            timestep=float(sim.get("timestep", 0.02)),
            cell=float(sim.get("cell", DEFAULT_CELL)),
            backend=str(sim.get("backend", "fast2d")),
            coverage_target=(
                float(term["coverage_target"]) if term.get("coverage_target") is not None else None
            ),
            dirt_target=(
                float(term["dirt_target"]) if term.get("dirt_target") is not None else None
            ),
            stop_on_empty_battery=bool(term.get("stop_on_empty_battery", True)),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "pool": {"preset": self.pool},
            "robot": {"preset": self.robot},
            "dirt": {"preset": self.dirt},
            "controller": {"preset": self.controller},
            "simulation": {
                "duration": self.duration,
                "timestep": self.timestep,
                "cell": self.cell,
                "backend": self.backend,
            },
        }
        if self.description:
            out["description"] = self.description
        if self.pool_params:
            out["pool"]["params"] = self.pool_params
        if self.robot_params:
            out["robot"]["params"] = self.robot_params
        if self.controller_params:
            out["controller"]["params"] = self.controller_params
        termination = {
            k: v
            for k, v in (
                ("coverage_target", self.coverage_target),
                ("dirt_target", self.dirt_target),
            )
            if v is not None
        }
        if not self.stop_on_empty_battery:
            termination["stop_on_empty_battery"] = False
        if termination:
            out["termination"] = termination
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # ------------------------------------------------------------------
    def build_pool(self) -> Pool:
        return make_pool(self.pool, **self.pool_params)

    def build_robot(self) -> Cleaner:
        return make_robot(self.robot, **self.robot_params)

    def build_dirt(self) -> DirtSpec:
        return make_dirt(self.dirt, **self.dirt_params)

    def build_controller(self) -> Any:
        from zimablue.controllers.base import CONTROLLERS

        params = dict(self.controller_params)
        if self.controller == "random_bounce":
            # Seed the controller from the scenario so a batch of episodes
            # actually varies, rather than every episode bouncing identically.
            params.setdefault("seed", self.seed)
        return CONTROLLERS.create(self.controller, **params)

    def simulation(self, seed: int | None = None, *, record: bool = True) -> Simulation:
        """Construct a :class:`~zimablue.simulation.Simulation` for this scenario."""
        effective = self.seed if seed is None else seed
        scenario = self if seed is None else _with_seed(self, effective)
        return Simulation(
            pool=scenario.build_pool(),
            robot=scenario.build_robot(),
            dirt=scenario.build_dirt(),
            controller=scenario.build_controller(),
            seed=effective,
            timestep=self.timestep,
            cell=self.cell,
            backend=self.backend,
            record=record,
            expose_truth=self.controller == "lawnmower_oracle",
            coverage_target=self.coverage_target,
            dirt_target=self.dirt_target,
            stop_on_empty_battery=self.stop_on_empty_battery,
            scenario_name=self.name,
        )

    def run(self, seed: int | None = None, *, record: bool = True, **kwargs: Any) -> RunResult:
        """Build and run in one call."""
        return self.simulation(seed, record=record).run(seconds=self.duration, **kwargs)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.pool} pool, {self.robot} cleaner, {self.dirt} dirt, "
            f"{self.controller} controller, {self.duration / 60:.0f} min, seed {self.seed}"
        )


def _with_seed(scenario: Scenario, seed: int) -> Scenario:
    from dataclasses import replace

    return replace(scenario, seed=seed)


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no scenario at {path}. Try one of the files in the scenarios/ directory."
        )
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return Scenario.from_dict(data, source=path)
