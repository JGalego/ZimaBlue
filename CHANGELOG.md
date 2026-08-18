# Changelog

All notable changes to ZimaBlue are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is 0, the domain API may change between minor
versions. The `.zbr` schema version is tracked separately and is bumped only
on an incompatible layout change; readers refuse anything newer than they
understand.

## [Unreleased]

### Added
- **`zimablue.dynamics`**: analysis of how a controller *behaves*, rather than
  what a run achieved. Reads finished recordings; needs nothing installed.
  - `return_map` — the Poincaré section on the pool wall, and the periodic
    orbits in it. Contacts are debounced, because 69% of raw bump-switch rising
    edges in a real run land under half a second apart and counting those as
    arrivals fills the section with chatter. No attracting orbits were found in
    any pool or controller tried, which is a real answer and not a broken tool;
    the detector is verified against a synthetic periodic section.
  - `transfer_operator` — the Perron-Frobenius operator on a grid. Invariant
    measure, spectral gap, mixing time, and *almost-invariant sets*: it
    separates an L-shaped pool into its two arms and a mushroom pool into cap
    and stem without being told the shape. The mushroom's partition falls below
    the geometric neck, which is right — the robot leaves the top of the stem
    freely and the bottom is what traps it.
  - `ergodic_score` — the Mathew-Mezić metric, with the dirt field as the
    target measure. One number for "spend time in proportion to how dirty it
    is". Unlike coverage and dirt removed it is **not monotone**, so it is the
    only metric here that can see a controller finish early and park: both
    oracles score their best mid-run and get worse from there.
  - `divergence` — how fast two runs a millimetre apart stop agreeing, and the
    horizon past which a rollout of this backend is a sample rather than a
    prediction. `random_bounce` turns out to be the *least* sensitive
    controller and `baseline_coverage` the most, which inverts the obvious
    guess.
  - `forecast_cleaning` — fit a removal rate on the first few minutes of
    occupancy and predict the rest of the cycle. Within 3.2% for
    `random_bounce` and 3.6% for `systematic`; 16.5% for `baseline_coverage`,
    which changes strategy halfway through. Forecast error is a
    strategy-change detector.
  - `zimablue.dynamics.plots` for each of the above, and `docs/dynamics.md`.
- Pool presets `stadium` and `mushroom`, chosen for their billiard dynamics:
  one provably ergodic, the other a provable trap. The mushroom's stem is 21%
  of the floor and takes 58% of the robot's time — 85% on one seed and 37% on
  another, with no change to the controller.
- `Simulation(start_pose=...)`, so a run can be started a millimetre from
  another one. Sensitivity analysis needs it and nothing else does.
- **Chase cam**: `zimablue replay run.zbr --chase`. A metre behind the robot
  and half a metre up — the view that shows the machine *and* what it is
  leaving behind, which neither the top-down view nor the dirt cam does. The
  camera's heading lags the robot's, so a turn reads as the robot swinging
  across frame rather than as the pool rotating.
- **Cleaner designs**: `CleanerDesign`, `Part`, and seven archetypes —
  `tracked`, `compact`, `heavy_duty`, `domed`, `flat_scrubber`, `quad_brush`,
  `suction_disc`. Both the top-down view and the chase cam draw the robot from
  its design, so a domed suction unit no longer looks like a quad-brush
  commercial machine. Coordinates are normalised and scaled by the chassis, so
  any design fits any robot, and the design rides inside the `.zbr`.

  Purely cosmetic, and tested to be: a run scores identically whichever design
  it wears. Physics reads `Chassis` and only `Chassis`.
- `zimablue.replay.floorcam.FloorCamera`, the inverse-perspective renderer both
  underwater cameras share. They differ in where the camera sits and in nothing
  else — two cameras that disagreed about how far away a leaf was would be two
  simulators wearing one name.
- `FloorCamera.project(..., height=)` projects points off the floor, which is
  what lets the chase cam draw a robot with thickness. At `height=0` it reduces
  exactly to the floor-only form it replaces, and there is a test that says so.
- `zimablue list` includes the designs.
- `zimablue.hardware`: the control loop with no simulator underneath it. A
  controller written against `Simulation` runs on a robot unmodified, because
  `ControlInput` and `DriveCommand` were always the entire contract. Standard
  library and NumPy only — no extra to install.
  - `DeviceSource` wraps one polling function and does the bookkeeping the
    simulated sensor pipeline was doing for free: timestamps, hold-last-good,
    staleness.
  - `WheelSpeedLoop`, PI with feedforward, closing metres per second onto duty
    cycle. Anti-windup, a slew limit that ramps up and cuts immediately, and
    symmetric saturation so a saturating turn keeps its radius.
  - `Watchdog` for the failures a simulator is not obliged to have: a sensor
    gone quiet, a loop that missed its deadline, a controller that raised.
  - `HardwareRuntime` writes the same `.zbr` the simulator writes, so a real
    run replays in the ordinary viewer. Its manifest carries
    `pose_source: "estimate"`, and `HardwareRun.metrics()` deliberately omits
    coverage and dirt removed — both need ground truth a robot does not have.
  - `RecordedSource` replays a recording back out as readings with jitter and
    dropouts, so a controller can be tested against conditions the simulator
    cannot produce.
- `zimablue.hardware.logs` and `TrajectorySource`: score the estimator against
  a trajectory a real robot really drove. `tools/fetch_trajectory.py` fetches
  Pioneer 3-DX logs from the TUM RGB-D benchmark and
  `examples/replay_real_trajectory.py` runs the whole stack over them — the
  first estimator numbers in this project measured against motion it did not
  generate. 0.15 m of error after 42.5 m of real driving, with the caveats in
  `docs/hardware.md`.
- `recording.build_frame`, shared by the simulator and the hardware runtime so
  the two cannot disagree about what a `.zbr` column means.
- `docs/hardware.md`.

### Fixed
- `export_summary` printed "coverage 0%  dirt removed 0%" for a run with no
  ground truth, which is the most misleading thing that figure could say — it
  looks exactly like a controller that did nothing. It now reports the metrics
  that exist and states that the others are unmeasured, and labels the three
  panels that need ground truth.
- `zimablue inspect` poured a hardware recording's metrics into
  `Metrics.from_dict`, which fills the keys it does not find with zeros — so a
  real run that drove 41 metres printed "distance 0.0 m" and "coverage 0.0%".
  It prints what the recording actually holds now, and says why the rest is
  absent. `Recording.has_ground_truth` is the flag to check.
- `load_scene` failed with a `TypeError` from inside `Pool.from_dict` when a
  recording carried no pool. It now says what is missing and why a recording
  written on a robot might not have one.
- `SystematicCoverage` crashed on a range reading of NaN, which is what a real
  ultrasonic rangefinder reports when no echo comes back — `int(nan / cell)`
  raises, from inside an occupancy map with no reason to expect one. The
  simulated sonar never produces one, so nothing caught it until a recording
  was replayed through the hardware runtime. Non-finite ranges are now treated
  as "no information" rather than as a measurement.

## [0.2.0] — 2026-08-17

### Added
- `trace_pool(..., segmenter=...)`: the water mask is pluggable. The colour
  rules stay the default and need nothing installed.
- `zimablue.segment.SamSegmenter`, a segmenter backed by a SAM ONNX export
  over onnxruntime — `pip install "zimablue[ml]"`, no torch, no GPU, weights
  not bundled. SAM proposes candidate masks and the colour rule picks between
  them, weighting recall so the shallow end is not clipped off the pool.
- `zimablue.rl`: a Gymnasium environment over `Simulation`, with a decimated
  control rate, an observation built only from what a controller can see, and
  a choice between rewarding dirt removed or coverage —
  `pip install "zimablue[rl]"`.
- `PolicyController` runs a trained policy through the ordinary controller
  interface, so it is scored, recorded, batched and replayed like any other.
- `Simulation.termination_reason()`, public for anything driving `step()`
  itself.
- `PoolCleaningEnv(extra_observations=...)` and `EstimatedPose`: derived
  channels appended to the observation, so a policy can be handed a pose
  estimate and an occupancy map and learn the planner alone. Runs every
  physics tick rather than every decision, and `PolicyController` takes the
  same object so training and deployment cannot disagree about the layout.
- `dirt_oracle`, a ground-truth controller that drives at whatever is
  dirtiest. The cleaning-side counterpart to `lawnmower_oracle`, and greedy
  rather than optimal: it removes 50% of a kidney pool's dirt in ten minutes
  against the baseline's 18%, and is behind the baseline by twenty-five.
- A controller may set `needs_truth = True`; scenarios and the CLI then turn
  `expose_truth` on for it.
- `docs/ml.md`, `examples/rl_env.py`, `examples/tune_controller.py` and
  `examples/train_policy.py` — the last trains a policy with PPO, scores it
  against the shipped controllers on held-out seeds, and replays it.
- `info["seed"]` on every env step, so an episode can be reproduced from what
  it reported.
- EKF pose estimator (`PoseEstimator`) over position, heading and gyro bias,
  with zero-velocity updates to make the bias observable.
- `SystematicCoverage` controller: an occupancy map built from bump switches
  and sonar in the estimated frame, boustrophedon lanes and nearest-frontier
  recovery.
- Controllers may publish `telemetry()`; it is recorded as `ctl.*` channels and
  replay draws the estimated pose against ground truth.
- 3D replay renderer: `zimablue replay run.zbr --3d`. Renders the pool as a
  basin from its depth model. Geometry only -- the motion still comes from the
  2D backend.
- Scenarios ship inside the wheel, so `zimablue run kidney` works from a plain
  `pip install` rather than only from a git clone.
- `Recording.frame_dt`, taken from the timestamps rather than the manifest.

### Changed
- The `kidney` preset is a proper kidney: ellipse booleans resampled as a
  low-order Fourier curve, so the boundary is smooth everywhere instead of
  carrying cusps from the boolean operations.
- The version lives only in `src/zimablue/_version.py`; `pyproject.toml` reads
  it, so a tag can no longer disagree with what gets published.
- Source distributions no longer carry the rendered GIFs: 4.4 MB to 170 KB.

### Fixed
- `PoolCleaningEnv.reset()` with no seed replayed the env's construction seed,
  so every episode was identical — and a training loop never passes a seed, so
  a policy would have been shown one episode for the whole run. It draws the
  next episode from the env's own generator now, which keeps the *sequence*
  reproducible from the construction seed. Every test had passed a seed
  explicitly, so none of them noticed; PPO did, immediately.
- A scenario picked `expose_truth` by matching one hard-coded controller name,
  so the second oracle to be written crashed on its first tick when run from
  YAML. It now asks the controller.
- The tour notebook's *Coverage is not cleanliness* section printed "two
  different controllers -- which is the point" above a table in which one
  controller had won both columns. It races the two oracles now, which invert.
- The documented backend throughput was about 50x real time; measured, it is
  25-30x on one core. Corrected in the README, getting-started and scenarios,
  including the wall-clock table.
- Playback below 1x did not work. Advancing a whole number of frames per tick
  cannot go slower than one frame per tick, so 1x played at 1.2x and both 0.5x
  and 0.25x played at 0.6x. The position is tracked as a float now and rounded
  only when a frame is drawn; `export_movie(..., speed=1.0)` writes a file
  exactly as long as the run.
- Drawing without matplotlib gave a traceback through the rendering internals.
  It now names the extra to install, and the CLI prints one line instead of a
  stack. Rich was also swallowing the `[viz]` in that advice as a style tag,
  so the message read "pip install 'zimablue'" -- what the user had already
  done.
- The zero-velocity update judged stillness from average wheel speed, which is
  zero for a robot spinning on the spot, so the filter charged the whole
  rotation to the gyro bias.
- The frontier search ignored the free/unknown boundary, so an accurate map
  produced an empty search and the robot declared a mostly-unexplored pool
  finished.
- 3D wall panels were decimated by a fixed stride, which collapsed the
  four-segment rectangular pool to a single degenerate panel and rendered it
  with no walls.

## [0.1.0] — 2026-08-15

The first working version: a complete path from a dirty pool to a watchable,
reproducible, scored replay.

### Added

**Domain model**
- `Pool`: Shapely boundary, pluggable depth models (constant, plane slope,
  composite), surface materials, and features split into blocking (obstacles,
  stairs) and hydraulic (drains, returns, skimmers).
- Pool presets: `rectangular`, `sloped`, `l_shaped`, `kidney`, `oval`,
  `stairs`.
- `Cleaner`: a composition of chassis, locomotion, cleaning system and power
  components, so a custom robot needs no changes to ZimaBlue. Presets:
  `tracked`, `compact`, `heavy_duty`.
- Dirt types with density, particle size, adhesion and pickup difficulty;
  a continuous mass raster plus discrete debris items.
- Seven dirt scenarios from `clean` to `neglected_pool`.

**Simulation**
- `SimulationBackend` protocol, and `Fast2DBackend`: differential-drive
  kinematics with slip, exact arc integration, penetration-based contact,
  battery draw, stuck detection and coverage bookkeeping. ~50× real time.
- Cleaning model in which removal of adhered dirt is gated by brush agitation.
- Five sensor models sharing one imperfection pipeline (rate, noise, bias with
  random walk, latency, quantisation, saturation, dropout, stuck), with
  schedulable fault injection.
- Deterministic `RngTree` with named child streams.

**Autonomy**
- `Controller` protocol; controllers see sensor readings only.
- `baseline_coverage` (boustrophedon with perimeter pass and recovery),
  `random_bounce` (a floor), `lawnmower_oracle` (a ground-truth upper bound).

**Results**
- Metrics split into geometric and cleaning-quality families, each with a
  spatial companion array.
- `.zbr` recording format v1: JSON manifest with embedded pool and robot
  configuration, columnar `npz` frames, sparse events, float16 dirt keyframes.
- Replay: interactive player with scrub, pause and 0.25×–25× speed, plus
  headless GIF, MP4, stills and four-panel summary exporters.

**Experiments**
- YAML scenarios with strict key validation, and six shipped scenario files.
- Batch runner with aggregate statistics, worst-episode reporting and CSV/JSON
  export.
- CLI: `demo`, `run`, `replay`, `batch`, `inspect`, `list`.

**Project**
- 198 tests, `mypy` clean across 45 modules, `ruff` lint and format.
- Documentation: research notes, architecture, getting started, scenarios,
  recording, replay, roadmap.
- A logo generated from the real `kidney` preset by `tools/make_logo.py`.

### Notes on physical models

- Settling velocity uses the Ferguson & Church (2004) universal equation rather
  than Stokes' law. Stokes is only valid below ~100 µm, and pool dirt spans
  8 µm to 70 mm: it put 350 µm sand at 124 mm/s against a measured ~45.
- Flocculation, hindered settling and CFD are deliberately not modelled, and
  the code says so where it matters.
- Wall climbing is not simulated; wall coverage is tracked as an unrolled
  perimeter visit record.

[Unreleased]: https://github.com/JGalego/ZimaBlue/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/JGalego/ZimaBlue/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JGalego/ZimaBlue/releases/tag/v0.1.0
