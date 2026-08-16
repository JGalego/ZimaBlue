# Roadmap

Honest status. Nothing is marked done until it is implemented, tested and
runnable from the CLI.

## v0.1 — the vertical slice

One complete path from a pool to a watchable replay, with every layer real
rather than stubbed. **Shipped.**

| | Component | State |
|---|---|---|
| ✅ | Package, tooling, CI-ready layout | done |
| ✅ | `RngTree` — named deterministic streams | done |
| ✅ | Pool geometry, depth models, features, 6 presets | done |
| ✅ | Cleaner components, 3 presets | done |
| ✅ | Sensors + noise/fault pipeline (5 models) | done |
| ✅ | Dirt types, dirt field, debris, generators | done |
| ✅ | `Fast2DBackend` — diff-drive, slip, collisions, battery | done |
| ✅ | Cleaning interaction (brush, suction, filter) | done |
| ✅ | Metrics — coverage and cleanliness, scalar + spatial | done |
| ✅ | Baseline coverage controller | done |
| ✅ | `.zbr` recording format | done |
| ✅ | Replay viewer with playback controls | done |
| ✅ | Scenario YAML, `zimablue run` / `demo` | done |
| ✅ | Batch experiments | done |
| ✅ | EKF pose estimator with ZUPT gyro-bias observation | done |
| ✅ | Occupancy mapping + `systematic` coverage controller | done |
| ✅ | Estimate-vs-truth overlay in replay | done |
| ✅ | 3D replay renderer (geometry, not physics) | done |

## v0.2 — depth

- **Wall and waterline cleaning** as a modelled behaviour rather than an
  unrolled-perimeter approximation.
- **A planner that can spend a good estimate.** This is now the top item, and
  it is backed by a measurement rather than a hunch. Calibrating the odometry
  improves the `systematic` controller's position estimate five-fold (13.7 m
  to 2.7 m of error over a 25-minute kidney run) and *halves* its coverage,
  because with a poor estimate the lane plan is effectively randomised and the
  robot wanders widely, while with a good one it executes short disciplined
  lanes and spends its time turning (182 m travelled against 340 m). Coverage
  is currently being won by accident. Candidate fixes: plan over the map
  instead of locally, decompose concave pools into cells, and cost turns
  properly.

- **Loop closure.** The estimator has no absolute reference, so position error
  grows without bound -- the filter reports several metres of uncertainty by
  the end of a run and is, measurably, still overconfident. Matching the
  current sonar returns against the map it has already built is the standard
  answer and would bound the drift.
- **Dirt dynamics** — resuspension driven by the robot's own wake, plus
  settling over long runs.
- **Scenario sweeps** — parameter grids and randomised distributions, not just
  seed sweeps.

## v0.3 — the 3D backend

Not to be confused with the 3D *renderer*, which ships today: that draws a
recorded 2D run as a basin with real depth. This is the other half -- actually
integrating the dynamics in three dimensions.

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
