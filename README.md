<div align="center">

<img src="docs/assets/logo-animated.svg" alt="A robotic cleaner tracing a coverage path across a kidney-shaped pool" width="640">

# 🌊 ZimaBlue

### **Simulate, test, and replay robotic pool cleaners.**

*Driving everywhere is not the same as cleaning everything.*<br>
*ZimaBlue measures both — and lets you watch it happen.*

[![License: MIT](https://img.shields.io/badge/license-MIT-0e6cb2?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-0e6cb2?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![No GPU required](https://img.shields.io/badge/no%20GPU-required-3ddcff?style=flat-square)](docs/architecture.md)
[![Tests: 198](https://img.shields.io/badge/tests-198%20passing-3fb950?style=flat-square)](tests)
[![Linted with Ruff](https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Typed: mypy](https://img.shields.io/badge/typing-mypy%20clean-0e6cb2?style=flat-square)](pyproject.toml)

<img src="docs/assets/replay.gif" alt="Replay of a cleaning run: the robot traces the pool while the cleaned swath and remaining dirt update live" width="720">

<sub>25 simulated minutes in a kidney pool, replayed at 260×.<br>
Watch the two left-hand meters diverge — that gap is the point.</sub>

</div>

---

Give ZimaBlue a pool, a cleaner, some dirt and a control algorithm. It
simulates what happens, records the run, replays it, and scores how well the
pool actually got cleaned.

```bash
git clone https://github.com/JGalego/ZimaBlue
cd ZimaBlue
pip install -e ".[dev]"
zimablue demo
```

No GPU. No ROS. No Docker. No Omniverse. No multi-gigabyte assets.

```python
import zimablue as zb

sim = zb.Simulation(pool="kidney", robot="tracked", dirt="autumn", seed=42)
result = sim.run(minutes=30)

print(result.metrics.summary())
result.save("runs/example.zbr")
```

```
  coverage            78.5 %   (walls 70 %)
  dirt removed        57.1 %   (572 g of 1002 g)
  uniformity          71.8 %
  revisits             1.51   extra passes/cell
  distance           318.6 m
  runtime             25.0 min
  energy              27.7 Wh   (battery 77 % left)
  collisions           399
  stuck                  0 events, 0.0 s
```

```bash
zimablue replay runs/example.zbr
```

## Results

Two numbers that a generic simulator will not separate for you:

> **coverage** — where the robot drove<br>
> **cleanliness** — what the robot removed

The clearest demonstration is the oracle. Over 30 minutes in a kidney pool:

| controller | coverage | dirt removed | distance |
|---|---|---|---|
| `lawnmower_oracle` (ground truth) | **88.3%** | 33.3% | 180 m |
| `baseline_coverage` | 78.2% | **44.5%** | 390 m |
| `random_bounce` | 84.7% | 44.8% | 393 m |

The oracle wins on coverage and *loses* on cleaning. It drives a perfect path,
finishes early and stops, while the scrappier controllers keep going over the
same adhered dirt — which is what actually removes it. Report only coverage and
you rank these backwards.

A second result the testbed surfaced, this one uncomfortable: **better
localisation currently makes things worse.** Calibrating the odometry improves
the mapping controller's position estimate fivefold and halves its coverage.

| `encoder_scale` | position error | coverage |
|---|---|---|
| 1.00 (uncalibrated) | 13.7 m | **73.9%** |
| 0.94 (calibrated) | **3.8 m** | 52.6% |

The estimator is not at fault — 3.8 m after 340 m of travel with no absolute
reference is respectable dead reckoning. The planner is. With a poor estimate
the lane plan is effectively randomised and the robot wanders widely, covering
ground the way random bounce does; with a good one it runs short disciplined
lanes and spends its time turning. Coverage is being won by accident, and
fixing that is the top [roadmap](docs/roadmap.md) item.

## In the box

**Pool shapes** — rectangular, sloped, L-shaped, kidney, oval, and one with
stairs and a ladder foot. Shapely polygons plus a pluggable depth model, with
drains, returns, skimmers and obstacles.

**Cleaners**, composed from components rather than subclassed:

```python
robot = zb.Cleaner(
    chassis=zb.Chassis(length=0.45, mass=10.5),
    cleaning=zb.CleaningSystem(
        brush=zb.Brush(width=0.38, aggressiveness=1.2),
        filter=zb.Filter(capacity=1200.0, mesh=45e-6),
    ),
    sensors=[zb.Encoder(), zb.IMU(), zb.Sonar(beam_angles=(0.0, 0.7, -0.7))],
)
```

**Sensors, imperfect by default** — encoders, IMU, pressure/depth, contact and
sonar share one pipeline of sampling rate, noise, bias with random walk,
latency, quantisation, saturation, dropout and stuck values. Encoders report
*wheel* speed, so odometry drifts because of real slip rather than an injected
error.

```python
robot.sensors.sonar.inject_fault(
    bias=0.15,  # reads 15 cm long
    dropout_probability=0.02,  # loses 2% of pings
    start_time=300.0,  # ...starting five minutes in
)
```

**Dirt** with density, particle size, adhesion and settling velocity derived
from the Ferguson–Church equation — 350 µm sand comes out at 47.6 mm/s against
a measured ~45. Continuous rasters for sediment and algae, discrete items for
leaves and twigs, some too big for the intake.

**Cleaning that depends on the brush.** Removal is gated by how much agitation
breaks the bond, so the brush advantage rises with adhesion: ~1.0× for sand,
2.2× for algae, 3.5× for biofilm.

**Controllers** — a boustrophedon baseline, a random-bounce floor, a
map-building `systematic` controller, and a `lawnmower_oracle` upper bound that
reads ground truth and is explicitly *not* deployable. Yours needs one class
with two methods, and sees sensor readings only.

**State estimation.** `systematic` runs an EKF over position, heading and
**gyro bias**, fed by the encoders and IMU. The bias is only observable when
the robot stops — a stationary gyro's reading *is* its bias — so zero-velocity
updates are what keep heading from fanning out over half an hour. Replay draws
the estimate as an amber ghost beside the true pose.

**Recording.** `.zbr` is a ZIP of a JSON manifest, columnar `npz` frames,
sparse events and dirt keyframes. Unzip it and read it with `numpy.load`. Pool
geometry and robot config are embedded, so a recording stays replayable after
the preset it came from changes.

Same version, platform, scenario and seed give a bit-identical recording —
fixed timestep, no wall-clock reads while stepping, and one seeded RNG tree
whose named streams mean adding a sensor never shifts another's noise.
Asserted in `tests/test_determinism.py`.

## Replay

<div align="center">
<img src="docs/assets/summary.png" alt="Four-panel summary: path driven, visit counts, dirt at start, dirt at end" width="760">
</div>

Playback runs at 0.25× to 25×, with pause, scrub, step and speed control. The
cleaned swath is drawn *under* the dirt, so a patch the robot drove over but
failed to clean still reads as dirty — exactly the failure worth seeing. Sonar
beams, wall contacts, battery and filter fill are all on screen.

Headless? `zimablue replay run.zbr --gif out.gif`.

### In three dimensions

<div align="center">
<img src="docs/assets/3d-sloped.gif" alt="A sloped pool rendered as a 3D basin, the camera orbiting as the cleaner works the floor" width="700">

<sub>A sloped pool: 1.0 m at the shallow end, 2.4 m at the deep end.</sub>

<img src="docs/assets/3d-kidney.png" alt="The same kidney run as a 3D basin at four points in time, the floor clearing from brown to blue" width="820">
</div>

```bash
zimablue replay runs/example.zbr --3d --gif out.gif
```

The floor is a surface built from the pool's depth model, the walls are
extruded from its boundary, and the robot sits at the local floor depth — so in
a sloped pool it really is metres lower at the deep end. The camera orbits
slowly for parallax, and vertical scale is exaggerated about 3.6× because a
12 m pool 2 m deep is otherwise a pancake.

**This renders in 3D; it does not simulate in 3D.** The motion still comes from
the 2D backend. A 3D *backend* — buoyancy, contact, wall climbing, cameras — is
designed but not built, and the [roadmap](docs/roadmap.md) says so.

## Why it exists

Commercial pool cleaners are evaluated by driving them around a real pool with
real dirt: slow, expensive, impossible to repeat exactly. Meanwhile Gazebo,
MuJoCo and Isaac Sim each make *their engine* the API, so anything
pool-specific you build cannot outlive the engine.

ZimaBlue takes the opposite stance: **the domain model is the API.** Pools,
dirt, cleaners, scenarios, recordings and metrics are ZimaBlue concepts. What
integrates the equations is a swappable backend behind an interface — today a
deterministic CPU-only 2D backend at ~50× real time.

```
                         ZimaBlue domain API
                                 │
              ┌──────────────────┼──────────────────┐
        World model         Robot model         Controller
     pool · water · dirt   body · sensors      (replaceable)
              └──────────────────┼──────────────────┘
                                 │
                        SimulationBackend
                                 │
                  ┌──────────────┴──────────────┐
             Fast2DBackend                IsaacSimBackend
              (CPU, today)                  (planned)
                                 │
                    Recording · Replay · Metrics
```

A backend owns dynamics and sensing, nothing else. Dirt accounting, metrics and
recording are computed by shared code from the state it returns, so a new
backend inherits them and cannot redefine how they are measured. The acceptance
test for the 3D backend is deliberately strict: **a `.zbr` it produces must
replay in the 2D viewer.**

## Scenarios

```bash
zimablue run   scenarios/autumn_kidney.yaml --record runs/autumn.zbr
zimablue batch scenarios/kidney.yaml --episodes 100 --out results.json
```

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first run, common tasks |
| [Architecture](docs/architecture.md) | Layering, backends, determinism contract |
| [Research](docs/research.md) | Prior art, and which decision each finding drove |
| [Scenarios](docs/scenarios.md) | YAML experiments and batch sweeps |
| [Recording](docs/recording.md) | The `.zbr` format, channel by channel |
| [Replay](docs/replay.md) | Controls, exporters, rendering notes |
| [Roadmap](docs/roadmap.md) | Done, next, and deliberately not planned |

Examples: [`basic.py`](examples/basic.py) ·
[`custom_robot.py`](examples/custom_robot.py) ·
[`custom_controller.py`](examples/custom_controller.py) ·
[`estimation_replay.py`](examples/estimation_replay.py) ·
[`replay.py`](examples/replay.py)

## Contributing

Issues and pull requests welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
The short version: extend through the registries rather than through
conditionals, do not break determinism, and prefer a small real model to a
large fake one.

## License

[MIT](LICENSE).
