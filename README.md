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
[![Linted with Ruff](https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Typed: mypy](https://img.shields.io/badge/typing-mypy-0e6cb2?style=flat-square)](pyproject.toml)

</div>

---

## What is ZimaBlue?

ZimaBlue is a robotics testbed for **swimming-pool cleaning robots**. Give it a
pool, a cleaner, some dirt and a control algorithm; it simulates what happens,
records the whole run, replays it, and scores how well the pool actually got
cleaned.

It is built around one distinction that a generic physics simulator will not
make for you:

> **coverage** — where the robot drove
> **cleanliness** — what the robot removed

A cleaner can visit 100 % of the floor and leave the algae exactly where it
was. ZimaBlue tracks both as first-class metrics, with a spatial map for each.

## Why it exists

Commercial pool cleaners are evaluated by driving them around a real pool with
real dirt — slow, expensive, and impossible to repeat exactly. Meanwhile the
general-purpose simulators (Gazebo, MuJoCo, Isaac Sim) make *their engine* the
API, so anything pool-specific you build on top cannot outlive the engine.

ZimaBlue takes the opposite stance: **the domain model is the API.** Pools,
dirt, cleaners, scenarios, recordings and metrics are ZimaBlue concepts. The
thing that integrates the equations is a swappable backend behind an interface.
Today that is a fast, deterministic, CPU-only 2D backend. Tomorrow it could be
Isaac Sim — without changing a line of your experiment code.

See [`docs/research.md`](docs/research.md) for the prior art behind these
choices and [`docs/architecture.md`](docs/architecture.md) for how it fits
together.

## Design principles

1. **Domain model first** — pools and dirt are the API, not the engine.
2. **Fast 2D by default** — no GPU, no ROS, no Omniverse, no multi-GB assets.
3. **Determinism is a contract** — same version + seed ⇒ identical run.
4. **Record everything** needed to reproduce an experiment.
5. **Sensors are imperfect** — noise, bias, latency, dropout and faults are the
   default, not an add-on.
6. **Dirt is spatial and physical** — seven types with density, particle size,
   adhesion and Stokes-derived settling.
7. **Your algorithm is replaceable** — the baseline controller is deliberately
   simple so it is easy to beat.

## Status

Early development, built in the open. What exists and works today:

| Area | Status |
|---|---|
| Pool geometry, depth models, features, 6 presets | ✅ implemented |
| Cleaner component model, 3 presets | ✅ implemented |
| Sensors (encoder, IMU, pressure, contact, sonar) + fault injection | ✅ implemented |
| Deterministic seeded RNG tree | ✅ implemented |
| Dirt model and generators | 🚧 in progress |
| Fast 2D physics backend | 🚧 in progress |
| Cleaning interaction and metrics | 🚧 in progress |
| Recording (`.zbr`) and replay viewer | 🚧 in progress |
| CLI (`demo`, `run`, `replay`, `batch`) | 🚧 in progress |
| Isaac Sim / 3D backend | 📐 interface + design only |

Nothing above is claimed as done until it is tested and runnable. See
[`docs/roadmap.md`](docs/roadmap.md).

## Quick start

```bash
git clone https://github.com/JGalego/ZimaBlue
cd ZimaBlue
pip install -e ".[dev]"
```

```python
import zimablue as zb

pool = zb.make_pool("kidney")
robot = zb.make_robot("tracked")

print(pool)   # Pool(name='kidney', floor_area=54.8 m2, max_depth=2.00 m, features=4)
print(robot.describe())
```

Sensors are imperfect by default, and you can make them worse on purpose:

```python
robot.sensors.sonar.inject_fault(
    bias=0.15,                 # reads 15 cm long
    dropout_probability=0.02,  # 2 % of pings lost
    start_time=300.0,          # ...starting five minutes in
)
```

## Architecture at a glance

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

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

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
