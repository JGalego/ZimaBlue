# Roadmap

Honest status. Nothing is marked done until it is implemented, tested and
runnable from the CLI.

## v0.1 — the vertical slice

The goal is one complete path from a pool to a watchable replay, with every
layer real rather than stubbed.

| | Component | State |
|---|---|---|
| ✅ | Package, tooling, CI-ready layout | done |
| ✅ | `RngTree` — named deterministic streams | done |
| ✅ | Pool geometry, depth models, features, 6 presets | done |
| ✅ | Cleaner components, 3 presets | done |
| ✅ | Sensors + noise/fault pipeline (5 models) | done |
| 🚧 | Dirt types, dirt field, debris, generators | in progress |
| 🚧 | `Fast2DBackend` — diff-drive, slip, collisions, battery | in progress |
| 🚧 | Cleaning interaction (brush, suction, filter) | in progress |
| 🚧 | Metrics — coverage and cleanliness, scalar + spatial | in progress |
| 🚧 | Baseline coverage controller | in progress |
| 🚧 | `.zbr` recording format | in progress |
| 🚧 | Replay viewer with playback controls | in progress |
| 🚧 | Scenario YAML, `zimablue run` / `demo` | in progress |
| 🚧 | Batch experiments | in progress |

## v0.2 — depth

- **Wall and waterline cleaning** as a modelled behaviour rather than an
  unrolled-perimeter approximation.
- **State estimation baseline** — an EKF fusing encoders, IMU and pressure, so
  that "the robot's belief" can be plotted against ground truth in replay. The
  sensors already drift correctly; nothing consumes that yet.
- **Better planners** — spanning-tree coverage, and a planner that decomposes
  concave pools into cells instead of treating them as one.
- **Dirt dynamics** — resuspension driven by the robot's own wake, plus
  settling over long runs.
- **Scenario sweeps** — parameter grids and randomised distributions, not just
  seed sweeps.

## v0.3 — the 3D backend

Design is in [`architecture.md`](architecture.md#3d-backend-intended-design).
The acceptance test is deliberately strict: **a `.zbr` produced by the 3D
backend must replay in the 2D viewer**, with extra channels ignored. If that
holds, the abstraction is real; if it does not, the 3D engine has leaked into
the domain model.

- USD stage generation from `Pool` and `Cleaner`
- PhysX rigid bodies with buoyancy and drag
- Cameras and depth sensors reusing the existing imperfection pipeline
- Synthetic dataset export

## Not planned

- **CFD.** Water is a drift field plus resuspension heuristics, and that is
  enough for navigation and cleaning decisions.
- **A ROS 2 dependency.** The recording format is the interchange boundary. A
  bridge is welcome as a separate package that consumes `.zbr`; it will not
  become an import of the core.
- **A framework for frameworks.** Registries are dicts. They stay dicts until
  something concrete needs more.

## Ideas worth an issue

- Compare controllers head-to-head over a fixed scenario suite and publish the
  table (a benchmark is only useful if it is run).
- Energy-optimal coverage, following the genetic-algorithm framing in the
  pool-cleaning literature.
- Filter-aware planning: return to the shallow end before the basket saturates.
- Import a real pool outline from a photo or a DXF.
