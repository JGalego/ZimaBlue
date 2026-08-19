# Roadmap

Nothing here is marked done until it is implemented, tested and runnable
from the CLI.

## v0.1 — the vertical slice

One complete path from a pool to a watchable replay, with every layer real
rather than stubbed. **Shipped.**

| | Component | State |
|---|---|---|
| ✅ | Package, tooling, CI-ready layout | done |
| ✅ | `RngTree` — named deterministic streams | done |
| ✅ | Pool geometry, depth models, features, presets | done |
| ✅ | Cleaner components and presets | done |
| ✅ | Sensors and the shared noise/fault pipeline | done |
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

## v0.2 — the pool as something you can see, learn and trace

**Shipped.**

| | Component | State |
|---|---|---|
| ✅ | Debris drawn as leaves and twigs rather than blobs | done |
| ✅ | Dirt cam — the pool from the cleaner's own bumper | done |
| ✅ | `zb.preview` — the pool turnable in a notebook | done |
| ✅ | `trace_pool` — a pool from a photograph, with a pluggable mask | done |
| ✅ | `SamSegmenter` — SAM over onnxruntime, `zimablue[ml]` | done |
| ✅ | `zimablue.rl` — Gymnasium env, `PolicyController`, `zimablue[rl]` | done |
| ✅ | `dirt_oracle` — the cleaning-side counterpart to the lawnmower | done |

## v0.3 — planners, fleets, and what the render was hiding

**Shipped.** Same bar: implemented, tested, runnable.

| | Component | State |
|---|---|---|
| ✅ | `zimablue.hardware` — the control loop with no simulator under it | done |
| ✅ | Estimator scored against a real robot's logged trajectory | done |
| ✅ | Chase cam — the view from behind, with the machine in shot | done |
| ✅ | `CleanerDesign` — silhouettes with real differences, drawing only | done |
| ✅ | `zimablue.dynamics` — sections, transfer operators, ergodic metric | done |
| ✅ | `stadium` and `mushroom` pools, chosen for their billiard dynamics | done |
| ✅ | `sketch_pool` — a pool from a hand drawing | done |
| ✅ | `zimablue.planners` — the offline and online coverage literature | done |
| ✅ | `compare` — every planner on every pool, and the matrix plot | done |
| ✅ | `Fleet` — several cleaners in one pool, colliding and talking | done |
| ✅ | Partitioning (`darp`, `forest`, …) and cooperation (`mstc`, `auction`, …) | done |
| ✅ | Dirt drift that conserves mass, and a drain that collects the pile | done |
| ✅ | The `kidney` as a parameterised arc chain | done |
| ✅ | `dirt_ceiling` — the share of the dirt no intake can lift | done |
| ✅ | Replay continuity: dirt blends between keyframes, debris moves | done |

## Next — depth

- **Wall and waterline cleaning** as a modelled behaviour rather than an
  unrolled-perimeter approximation.
- **A planner that can spend a good estimate.** The `systematic` controller
  sweeps lanes locally and falls back to nearest-frontier search; nothing in it
  plans over the map it went to the trouble of building. This used to be
  backed by a striking measurement -- calibrate the odometry, watch coverage
  halve -- which turned out to be a bug in the map rather than a fact about
  planning, and is written up in
  [`controllers/systematic.py`](../src/zimablue/controllers/systematic.py).
  With that fixed, calibration changes neither the estimate nor the coverage
  much, so this item is back to being an argument rather than a number.
  Candidate fixes: plan over the map instead of locally, decompose concave
  pools into cells, and cost turns properly.

- **Loop closure.** The estimator has no absolute reference, so position error
  grows without bound -- the filter reports several metres of uncertainty by
  the end of a run and is, measurably, still overconfident. Matching the
  current sonar returns against the map it has already built is the standard
  answer and would bound the drift.

  Replaying real trajectories sharpened the case. The gyro bias is only
  observable when the robot stops, because the zero-velocity update is the only
  thing that makes it observable. On a logged run that rarely stops
  (`pioneer_slam2`) the heading ends 39 degrees out, while two runs that pause
  more often stay under five. In simulation this is invisible: every shipped
  controller stops and turns constantly, so the bias is always being observed.
  See [hardware.md](hardware.md).

- **Measured sensor parameters.** Every figure in `SensorConfig` is a
  consumer-MEMS-class guess. An hour of a real gyro sitting still, Allan-
  variance'd, replaces the two that matter most, and a phone is a good enough
  gyro to start with. Until then every noise-dependent result in this
  repository rests on numbers nobody measured -- which is now testable, because
  `zimablue.hardware` can be pointed at a real log.
- **Dirt dynamics** — resuspension driven by the robot's own wake, plus
  settling over long runs.
- **Scenario sweeps** — parameter grids and randomised distributions, not just
  seed sweeps.

## Later — the 3D backend

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
- Import a real pool outline from a DXF. (From a photo is
  [done](imaging.md).)
- **Pool furniture detection.** `SamSegmenter` finds the water; nothing finds
  the ladder, the steps, the skimmer or the drains, so `traced.pool()` comes
  back with no features. A detector would fix that and needs a labelled
  dataset of pool photographs, which as far as we can tell does not exist.
- **A wheel-speed loop tuned on something real.** `WheelSpeedLoop`'s gains are
  a starting point for a small geared drive, not a measurement. A step response
  off any real differential drive would replace them, and the procedure is two
  paragraphs in [hardware.md](hardware.md).
- **An overhead camera rig.** The one thing that would let cleaning be scored
  on real hardware. Coverage needs the true pose and dirt removed needs the
  true dirt field, and a robot has neither -- so a camera above the pool is not
  a nice-to-have, it is the entire difference between measuring cleaning and
  asserting it.
- **A trained controller worth shipping.** The env is there, the baseline to
  beat is measured, and `examples/train_policy.py` will point PPO at it. Half
  an hour of CPU buys a policy that drives forward, turns off walls and loses
  to random bounce; somebody spending real compute is the open item. The
  result would belong in the benchmark table above, not in the library.
