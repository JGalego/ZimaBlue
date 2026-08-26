# Getting started

## Install

```bash
git clone https://github.com/JGalego/ZimaBlue
cd ZimaBlue
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The core install is NumPy, Shapely, PyYAML, Typer and Rich. It needs no GPU,
ROS, Docker, Omniverse or multi-gigabyte assets. Matplotlib arrives with the
`viz` or `dev` extra and is only needed for replay.

## Your first simulation

```bash
zimablue demo
```

That builds a kidney pool, generates autumn dirt in it, runs a tracked cleaner
for twenty simulated minutes, records everything, prints the metrics, saves a
summary image and opens the replay.

On a headless machine it will tell you so and leave you a recording to render.

## The shortest useful program

```python
import zimablue as zb

sim = zb.Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=42)
result = sim.run(minutes=30)

print(result.metrics.summary())
result.save("runs/first.zbr")
```

```bash
zimablue replay runs/first.zbr
```

## The one idea to take away

```python
print(f"{result.metrics.coverage:.0%} driven over")
print(f"{result.metrics.dirt_removed_fraction:.0%} actually removed")
```

Those are different numbers, and the gap between them is the interesting part.
A robot with a worn brush covers the floor exactly as well and leaves the algae
behind. See [`research.md`](research.md) §1 and §9 for why the literature draws
the same distinction.

## What to look at next

| If you want to… | Go to |
|---|---|
| Understand the design | [`architecture.md`](architecture.md) |
| See the evidence behind it | [`research.md`](research.md) |
| Write experiments | [`scenarios.md`](scenarios.md) |
| Let an experiment choose its next trial | [`autonomous-experiments.md`](autonomous-experiments.md) |
| Optimise commands with motion gradients | [`differentiable-physics.md`](differentiable-physics.md) |
| Read or convert recordings | [`recording.md`](recording.md) |
| Ask what another controller would do | [`counterfactual.md`](counterfactual.md) |
| Calibrate a digital twin | [`calibration.md`](calibration.md) |
| Monitor a live machine | [`shadow.md`](shadow.md) |
| Reconstruct from several phone views | [`phone-reconstruction.md`](phone-reconstruction.md) |
| Watch runs properly | [`replay.md`](replay.md) |
| Know what is missing | [`roadmap.md`](roadmap.md) |

## Common tasks

### List everything available

```bash
zimablue list
```

### Run a scenario and keep the recording

```bash
zimablue run scenarios/kidney.yaml --record runs/kidney.zbr --summary runs/kidney.png
```

### Look inside a recording without replaying it

```bash
zimablue inspect runs/kidney.zbr --channels --events
```

### Run a hundred seeded episodes

```bash
zimablue batch scenarios/kidney.yaml --episodes 100 --out results.json
```

### Build a custom robot

Compose components; nothing in ZimaBlue needs editing.

```python
robot = zb.Cleaner(
    name="my_cleaner",
    chassis=zb.Chassis(length=0.45, width=0.40, mass=10.5),
    cleaning=zb.CleaningSystem(
        brush=zb.Brush(width=0.38, aggressiveness=1.2),
        filter=zb.Filter(capacity=1200.0, mesh=45e-6),
    ),
    sensors=[zb.Encoder(), zb.IMU(), zb.Sonar(beam_angles=(0.0, 0.7, -0.7))],
)
```

See `examples/custom_robot.py`.

### Break a sensor on purpose

```python
robot.sensors.sonar.inject_fault(
    bias=0.15,  # reads 15 cm long
    dropout_probability=0.02,  # loses 2% of pings
    start_time=300.0,  # from five minutes in
)
```

Faults compose, and any of them can start part-way through a run.

### Write a controller

One class, two methods. It sees sensor readings only — never ground-truth pose.

```python
class MyController:
    name = "mine"

    def reset(self, robot): ...

    def step(self, ctl):
        sonar = ctl.reading("sonar")
        if sonar is not None and sonar.valid and sonar[0] < 0.4:
            return zb.DriveCommand(left=-0.2, right=0.2)
        return zb.DriveCommand(left=0.3, right=0.3)


zb.Simulation(pool="kidney", controller=MyController(), seed=1).run(minutes=10)
```

Benchmark it against the shipped `random_bounce` (a floor) and the two
ground-truth oracles: `lawnmower_oracle` drives a perfect path and
`dirt_oracle` heads for whatever is dirtiest. They do not agree about which
run was better, which is the whole argument. See
`examples/custom_controller.py`.

### Add a pool shape

```python
from shapely.geometry import Polygon
from zimablue.pool import POOL_PRESETS, Pool


@POOL_PRESETS.register("my_pool")
def my_pool() -> Pool:
    return Pool(boundary=Polygon([...]), depth=1.8, material="tile")
```

It is now available everywhere, including in scenario YAML.

## Performance

The 2D backend runs at **25–30× real time** on one core — a 30-minute clean in
about a minute. The spread is the pool: a big rectangular one with light
sediment is nearer 30×, a kidney with autumn debris and recording on nearer
25×. For sweeps, turn recording off (`record=False`, which `run_batch` does by
default) and consider a coarser `cell` or a larger `timestep`.

## Determinism

Same version, same platform, same scenario, same seed ⇒ bit-identical
recording. If you need to reproduce a run, the seed and the full configuration
are inside the `.zbr`. Cross-platform bit-identity is not promised; see
[`architecture.md`](architecture.md#determinism-contract) for why.

## Running the tests

```bash
pytest
ruff check . && ruff format --check .
mypy
```
