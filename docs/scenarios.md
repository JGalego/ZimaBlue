# Scenarios and experiments

A scenario is an experiment written down. It names everything a run needs, so
that "reproduce this" is a filename and a seed rather than a paragraph of
instructions.

```bash
zimablue run scenarios/kidney.yaml
zimablue run scenarios/kidney.yaml --seed 7 --record runs/kidney7.zbr
zimablue batch scenarios/autumn_kidney.yaml --episodes 100 --out results.json
```

## The file

```yaml
name: autumn_kidney
description: >
  Leaf fall in a kidney pool. Leaves are discrete and some are too large for
  the intake, so this exercises the debris path rather than just the raster.

seed: 42

pool:
  preset: kidney

robot:
  preset: tracked

dirt:
  preset: autumn

simulation:
  duration: 1800      # simulated seconds
  timestep: 0.02      # 50 Hz
  cell: 0.10          # raster resolution, metres
  backend: fast2d

controller:
  preset: baseline_coverage

termination:
  dirt_target: 0.9            # stop once 90% of the dirt is gone
  coverage_target: 0.95       # ...or 95% of the floor is covered
  stop_on_empty_battery: true
```

Every section except `name` has a default, so the shortest valid scenario is
three lines. A bare preset name works where a mapping would:

```yaml
name: quick
pool: oval
robot: compact
```

## Unknown keys are errors

```yaml
simulation:
  duratoin: 600     # typo
```

```
error unknown key(s) ['duratoin'] in scenario section 'simulation'.
      Allowed: ['backend', 'cell', 'duration', 'timestep']
```

This is deliberate. A scenario that silently ignores a misspelled key runs a
*different experiment* than the one you wrote down, and reports its results as
though nothing were wrong. That is the one failure mode a reproducibility tool
must not have.

## Passing parameters to a preset

Presets take keyword arguments. `params` forwards them:

```yaml
pool:
  preset: rectangular
  params:
    length: 16.0
    width: 7.0
    depth: 2.2

controller:
  preset: random_bounce
  params:
    turn_range: [0.4, 1.8]
```

## Shipped scenarios

| File | What it is for |
|---|---|
| `rectangular.yaml` | The control. A flat 10×5 m box with light sediment — if a controller cannot cover this, the pool shape is not the problem. |
| `kidney.yaml` | Curved boundary with a concave side and a hopper floor. |
| `autumn_kidney.yaml` | Leaf fall: exercises discrete debris, oversized items and filter fill. |
| `neglected.yaml` | A month unattended. Mostly adhered algae and biofilm, so coverage and cleanliness come apart hardest here. |
| `corner_heavy.yaml` | Dirt piled where a lawnmower path arrives last, in a pool with stairs and a ladder foot. |
| `oracle_baseline.yaml` | The upper bound, for calibration. |

## Choosing a duration

A real cleaning cycle is 1.5–3 hours; the presets use 30 minutes so that a
scenario runs in about a minute of wall clock. The fast 2D backend runs at
25–30× real time on one core, so:

| Simulated | Wall clock (approx) |
|---|---|
| 5 min | 10 s |
| 30 min | 1 min |
| 2 hours | 4 min |

`--minutes` overrides the file without editing it, which is what you want while
iterating.

## Batch experiments

```bash
zimablue batch scenarios/kidney.yaml --episodes 100 --out results.json --csv results.csv
```

Seeds default to `scenario.seed + i`, so the whole batch is reproducible from
the file alone. Recording is **off** by default in a batch — 100 recordings is
a few hundred megabytes, and most of the time you want the table. Pass
`--record-dir runs/sweep/` when you want to keep them.

The output looks like:

```
episodes           100
success_rate       100.0 %
mean_coverage      81.2 %  (sd 5.0, min 63.7, max 86.9)
mean_dirt_removed  55.4 %  (sd 11.5, min 25.4, max 78.1)
mean_runtime       30.0 min
mean_energy        33.3 Wh
stuck_rate         0.0 %
mean_collisions    467
worst_episode      seed 130 (63.7 % coverage)
```

Two of those deserve comment.

**`success_rate` counts only runs that finished** — on the duration or on a
target. A run that flattens its battery at 40% coverage did not succeed, and
letting it average in with the rest silently flatters every other number.

**The worst episodes are named, with the command to reproduce them.** An
aggregate mean tells you the middle; the interesting engineering is almost
always in the tail:

```bash
zimablue run scenarios/kidney.yaml --seed 130 --record runs/worst.zbr
zimablue replay runs/worst.zbr
```

## In Python

```python
from zimablue.batch import run_batch
from zimablue.scenarios import load_scenario

scenario = load_scenario("scenarios/kidney.yaml")
scenario.duration = 600.0

result = run_batch(scenario, episodes=50)
print(result.stats("coverage"))  # mean, std, min, max, median
print(result.success_rate)
for episode in result.worst("dirt_removed_fraction", 3):
    print(episode.seed, episode.metrics.dirt_removed_fraction)
```

To sweep a parameter rather than a seed, build the scenarios in a loop — there
is no sweep DSL, and there will not be one until a real use case needs
something a `for` loop cannot express:

```python
from dataclasses import replace

for robot in ("compact", "tracked", "heavy_duty"):
    outcome = run_batch(replace(scenario, robot=robot), episodes=20)
    print(f"{robot:12s} {outcome.stats('dirt_removed_fraction')['mean']:.1%}")
```

## Writing your own

Anything registered is available by name. To use a component that is not a
preset, skip YAML and construct `Simulation` directly — the scenario file is a
convenience for the common case, not the only way in.

```python
import zimablue as zb

sim = zb.Simulation(
    pool=my_pool,  # a Pool you built
    robot=my_robot,  # a Cleaner you composed
    dirt=my_dirt_spec,
    controller=MyController(),
    seed=42,
)
```
