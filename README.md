<div align="center">

<img src="docs/assets/logo-animated.svg" alt="A robotic cleaner tracing a coverage path across a kidney-shaped pool" width="640">

# 🌊 ZimaBlue

### **Simulate, test, and replay robotic pool cleaners.**

*Driving everywhere is not the same as cleaning everything.*<br>
*ZimaBlue measures both — and lets you watch it happen.*

[![License: MIT](https://img.shields.io/badge/license-MIT-0e6cb2?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-0e6cb2?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Status: alpha](https://img.shields.io/badge/status-alpha-f59e0b?style=flat-square)](docs/roadmap.md)
[![No GPU required](https://img.shields.io/badge/no%20GPU-required-3ddcff?style=flat-square)](docs/architecture.md)
[![Tests: 159](https://img.shields.io/badge/tests-159%20passing-3fb950?style=flat-square)](tests)
[![Linted with Ruff](https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Typed: mypy](https://img.shields.io/badge/typing-mypy%20clean-0e6cb2?style=flat-square)](pyproject.toml)

<img src="docs/assets/replay.gif" alt="Replay of a cleaning run: the robot traces the pool while the cleaned swath and remaining dirt update live" width="720">

<sub>25 simulated minutes in a kidney pool, replayed at 260×.<br>
Watch the two left-hand meters diverge — that gap is what ZimaBlue exists to measure.</sub>

</div>

---

## What is ZimaBlue?

ZimaBlue is a robotics testbed for **swimming-pool cleaning robots**. Give it a
pool, a cleaner, some dirt and a control algorithm; it simulates what happens,
records the whole run, replays it, and scores how well the pool actually got
cleaned.

It is built around one distinction a generic physics simulator will not make
for you:

> **coverage** — where the robot drove
> **cleanliness** — what the robot removed

Those are different numbers. In the run above the cleaner drove over **78%** of
the floor and removed **57%** of the dirt. Turn its brush off and the first
number barely moves while the second collapses, because suction alone cannot
lift algae off a wall.

## Quick start

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
  termination       duration
```

```bash
zimablue replay runs/example.zbr
```

## The replay

<div align="center">
<img src="docs/assets/summary.png" alt="Four-panel summary: path driven, visit counts, dirt at start, dirt at end" width="760">
</div>

Playback runs at 0.25× to 25×, with pause, scrub, step and speed control. The
cleaned swath is drawn *under* the dirt, so a patch the robot drove over but
failed to clean still reads as dirty — which is exactly the failure worth
seeing. Sonar beams, wall contacts, battery and filter fill are all on screen.

Headless? `zimablue replay run.zbr --gif out.gif` — the player detects it and
tells you.

## Why it exists

Commercial pool cleaners are evaluated by driving them around a real pool with
real dirt: slow, expensive, and impossible to repeat exactly. Meanwhile the
general-purpose simulators (Gazebo, MuJoCo, Isaac Sim) each make *their engine*
the API, so anything pool-specific you build on top cannot outlive the engine.

ZimaBlue takes the opposite stance: **the domain model is the API.** Pools,
dirt, cleaners, scenarios, recordings and metrics are ZimaBlue concepts. The
thing that integrates the equations is a swappable backend behind an interface.
Today that is a fast, deterministic, CPU-only 2D backend running at ~50× real
time. Tomorrow it could be Isaac Sim, without changing a line of your
experiment code.

[`docs/research.md`](docs/research.md) traces every design decision to the
prior art behind it — coverage path planning, underwater localization, IMU
noise models, MCAP, IEC 62929, and the sediment-transport literature.

## What is in the box

**Six pool shapes** — rectangular, sloped, L-shaped, kidney, oval, and one with
stairs and a ladder foot. Geometry is Shapely polygons plus a pluggable depth
model, with drains, returns, skimmers and obstacles.

**Three cleaners**, composed from components rather than subclassed:

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

**Five sensors, imperfect by default** — encoders, IMU, pressure/depth, contact
and sonar, all sharing one pipeline of sampling rate, noise, bias with random
walk, latency, quantisation, saturation, dropout and stuck values. Encoders
report *wheel* speed, so odometry drifts because of real slip rather than an
injected error.

```python
robot.sensors.sonar.inject_fault(
    bias=0.15,  # reads 15 cm long
    dropout_probability=0.02,  # loses 2% of pings
    start_time=300.0,  # ...starting five minutes in
)
```

**Seven dirt types** with density, particle size, adhesion and settling
velocity derived from the Ferguson–Church equation — 350 µm sand comes out at
47.6 mm/s against a measured ~45. Continuous rasters for sediment and algae,
discrete items for leaves and twigs, some of which are too big for the intake.

**Cleaning that depends on the brush.** Removal is gated by how much agitation
breaks the bond, so the brush advantage rises with adhesion: ~1.0× for sand,
2.2× for algae, 3.5× for biofilm.

**Three controllers** — a boustrophedon baseline, a random-bounce floor, and a
ground-truth `lawnmower_oracle` upper bound that is explicitly *not*
deployable. Yours needs one class with two methods, and it sees sensor readings
only, never ground-truth pose.

**Recording and replay.** `.zbr` is a ZIP of a JSON manifest, columnar `npz`
frames, sparse events and dirt keyframes. Unzip it and read it with
`numpy.load`; the pool geometry and robot config are embedded, so a recording
stays replayable after the preset it came from changes.

**Scenarios and batches.**

```bash
zimablue run   scenarios/autumn_kidney.yaml --record runs/autumn.zbr
zimablue batch scenarios/kidney.yaml --episodes 100 --out results.json
```

## Coverage is not cleanliness

The clearest demonstration is the oracle. Over 30 minutes in a kidney pool:

| controller | coverage | dirt removed | distance |
|---|---|---|---|
| `lawnmower_oracle` (ground truth) | **88.3%** | 33.3% | 180 m |
| `baseline_coverage` | 78.2% | **44.5%** | 390 m |
| `random_bounce` | 84.7% | 44.8% | 393 m |

The oracle wins on coverage and *loses* on cleaning. It drives a perfect path,
finishes early, and stops — while the scrappier controllers keep going over the
same adhered dirt, which is what actually removes it. A testbed that reported
only coverage would rank these exactly backwards.

The baseline is also, honestly, beaten by random bounce on coverage. It is a
deliberately simple behaviour stack, not a contribution; better planners are
[on the roadmap](docs/roadmap.md) and are an easy first contribution.

## Determinism

> Same ZimaBlue version + same platform + same scenario + same seed
> ⇒ **bit-identical recording.**

Fixed timestep, no wall-clock reads in the stepping path, and one seeded RNG
tree whose named child streams mean adding a sixth sensor never shifts the
fifth one's noise. Asserted in `tests/test_determinism.py`, including across a
save/load cycle. Cross-platform bit-identity is *not* promised, and
[the docs say why](docs/architecture.md#determinism-contract).

## Architecture

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

A backend owns dynamics and sensing, and nothing else. Dirt accounting, metrics
and recording are computed by shared code from the state it returns, so a new
backend inherits them and cannot redefine how they are measured. The acceptance
test for the 3D backend is deliberately strict: **a `.zbr` it produces must
replay in the 2D viewer.**

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first run, common tasks |
| [Architecture](docs/architecture.md) | Layering, backends, determinism contract |
| [Research](docs/research.md) | Prior art, and which decision each finding drove |
| [Scenarios](docs/scenarios.md) | YAML experiments and batch sweeps |
| [Recording](docs/recording.md) | The `.zbr` format, channel by channel |
| [Replay](docs/replay.md) | Controls, exporters, rendering notes |
| [Roadmap](docs/roadmap.md) | What is done, next, and deliberately not planned |

Examples: [`basic.py`](examples/basic.py) ·
[`custom_robot.py`](examples/custom_robot.py) ·
[`custom_controller.py`](examples/custom_controller.py) ·
[`replay.py`](examples/replay.py)

## Status

Alpha, and honest about it. The fast 2D backend, sensors, dirt, cleaning,
metrics, recording, replay, scenarios, batch and CLI all work and are tested.
The 3D backend is an interface and a design document — nothing more, and the
roadmap says so.

## Contributing

Issues and pull requests welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
The short version: extend through the registries rather than through
conditionals, do not break determinism, and prefer a small real model to a
large fake one.

## License

[MIT](LICENSE).

---

<div align="center">
<sub>

The logo is not an illustration — `tools/make_logo.py` renders it from the real
`kidney` pool preset and a real boustrophedon coverage path.
Change the preset, and the logo changes with it.

</sub>
</div>
