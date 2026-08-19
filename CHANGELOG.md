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
- `Metrics.debris_oversize`, `uncollectable_dirt` and `dirt_ceiling`: how many
  items are wider than the intake, what they weigh, and the fraction of the
  dirt a perfect run could therefore still remove. The summary, the CLI table
  and the replay HUD all show it. An autumn kidney run starts with 60 items,
  19 of which no pump in the model can lift; the cleaner collected all 41 it
  could and still read as leaving a quarter of the leaves behind.

- **Fleets** (`zimablue.fleet`): several cleaners in one pool.
  `zb.Fleet(pool="kidney", robots=3, controllers="auction").run(minutes=20)`.
  Each robot gets its own backend, sensors and controller; all of them are
  reset against one `World`, which is what makes the dirt shared. Nothing in
  the single-robot path changed to allow it.

  Three things are genuinely new. They **collide** -- every backend is told
  where the others are as discs before each tick, the resolver pushes them
  apart, and the sonar sees them, with `Contact.is_robot` separating a
  team-mate from a wall. They **talk badly** -- `Blackboard` is a radio, not a
  god view: a robot publishes its own *estimate* of where it is and what it has
  covered, so a fleet inherits every member's localisation error and has to
  coordinate through it, and `comms_range` limits who hears whom. And the tick
  is **sense all, decide all, move all**, so no robot gets a turn-order
  advantage from being listed first.

- `FleetMetrics`: team coverage as a union, per-robot metrics, plus three
  numbers a single-robot run cannot have -- `speedup` (team coverage over the
  best member's, ceiling = the robot count), `overlap` (floor more than one
  robot did) and `balance` (shortest robot's distance over the longest's,
  which catches one robot working while another parks).

- **Five ways to divide a pool** (`zimablue.planners.partition`): `voronoi`,
  `geodesic`, `strips`, `darp` (Kapoutsis et al., 2017) and `forest` (Zheng et
  al., 2005). Each territory becomes a small `Pool`, so all eight offline
  planners work inside one unchanged:
  `controllers=partitioned("darp", "sweep_optimal")`. On a kidney with three
  robots the fairness -- smallest share over largest -- runs 0.51 for Voronoi,
  0.54 geodesic, 0.82 forest, 0.96 DARP, 0.99 strips.

- **Five ways to cooperate without dividing** (`zimablue.planners.cooperative`):
  `mstc` and `mstc_backtracking` (Hazon & Kaminka, 2005), `auction` (Zlot et
  al., 2002), `binn_swarm` (Luo & Yang, 2008) and `smc_swarm` (multi-agent
  Mathew & Mezic). Plus a sixth that needed no code: every online planner
  becomes cooperative when the fleet hands it a blackboard, and
  `Fleet(..., share=True)` is the default.

- `zimablue.fleetplots`: paths, territory, overlap and progress, and
  `plot_fleet` for all four. The replay window draws fleets too -- a coloured
  ring and trail per robot, with the HUD following robot 0.

- `compare_fleets` and `FLEET_DIMENSIONS` reuse the planner comparison harness
  with team measurements, so `plot_matrix` works on a fleet unchanged.

- `examples/fleet.py` (including `--scaling`, which prints what the second,
  third and fourth robot are actually worth) and
  [docs/multi-robot.md](docs/multi-robot.md). On a kidney pool with `auction`,
  coverage goes 43% -> 64% -> 78% -> 84% as the fleet grows to four, while
  speedup peaks at three robots and then falls: half the pool is already being
  done twice at three, and the fourth buys six points of floor for eight more
  of overlap and half again as many collisions.
- **Coverage path planning** (`zimablue.planners`): eighteen planners, which is
  most of the single-robot 2D literature. Eight offline -- `boustrophedon`,
  `sweep_optimal`, `trapezoidal`, `boustrophedon_cells`, `morse`, `contour`,
  `wavefront`, `spanning_tree` -- each returning a `CoveragePath` that
  `PathFollower` drives. Ten online, registered as ordinary controllers:
  `spiral_stc`, `full_stc`, `bsa`, `ba_star`, `brick_and_mortar`, `binn`,
  `epsilon_star`, `ppcpp`, `frontier` and `smc`.

  The online ten share one substrate. `OnlineCoverage` owns the EKF, the
  occupancy grid, the bump recovery and the driving; each algorithm implements
  one method that returns the next cell. If they each brought their own motion
  layer, a difference in coverage could always be the motion layer's fault.

- `EvidenceMap`, an occupancy grid that can change its mind about a wall. The
  online planners drive on the map they build, and the first version of all ten
  covered about an eighth of a pool before declaring it finished -- not because
  of any of the algorithms, but because `OccupancyMap` never retracts a wall,
  and three minutes of sonar echoes scattered by a drifting pose turned an
  eight-metre pool into 552 wall cells around 108 of floor. A wall now needs
  three sightings, a beam passing through a cell takes a vote away, and the
  robot's own footprint overrides everything.

- `PathFollower(planner, localisation="truth" | "odometry")`. Following a plan
  from the true pose measures the *route*; following it through the same EKF a
  real machine has measures the route plus the localisation. On the rectangular
  pool `sweep_optimal` reaches 70.5% on truth and 49.6% on odometry, and that
  gap is the most informative number in the package.

- `zimablue.planners.compare`: a harness that runs every planner on every pool
  and scores twelve dimensions -- coverage, dirt, evenness, the largest patch
  left untouched, edge coverage, path efficiency, turning per metre, time to
  half the pool, ergodic error, wasted time, energy and collisions.
  Deliberately no scalar ranking.
  `zimablue.planners.plots` draws four views of it: a normalised matrix, the
  trajectories side by side, coverage against time, and any two dimensions
  against each other with the Pareto front. `plot_plans` draws the offline
  routes and their decompositions before anyone has tried to drive them.

- `smc`, spectral multiscale coverage (Mathew & Mezic). It never chooses a
  cell: it carries a Fourier description of where it has spent its time,
  compares it with one of where it should, and drives down the difference. The
  metric it minimises is the one `zimablue.dynamics.ergodic_score` already
  measured, so the analysis module and the controller now close a loop.

- `examples/compare_planners.py` and [docs/planners.md](docs/planners.md).
- **Pools from drawings**: `zb.pool_from_sketch("napkin.jpg", width=9.0)`. A
  `SketchSegmenter` that satisfies the existing `Segmenter` protocol, so
  everything downstream of "which pixels are pool" -- region picking, hole
  filling, tracing, scale, simplify, smooth -- is the same code the photograph
  path uses.

  Three things make a drawing hard and each is handled: a lifted pen leaves
  gaps in the outline (the stroke is dilated to bridge them), sketches have
  arrows and labels inside the outline (filling inward from the page border
  includes them, which is correct), and a phone photo of paper has a shadow
  across it (the threshold is local).

  Filling from the border rather than from a seed also decides *how it fails*.
  A seed fill leaks through an unclosed gap and returns the whole page as one
  enormous pool. A border fill returns almost nothing instead -- and that is
  detectable, so an outline that did not close is an error naming the setting
  to raise rather than a pool the size of the photograph.
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

### Changed
- The `kidney` preset is a chain of four circular arcs -- two lobe radii, a
  belly arc under both, a scoop bitten out between them -- meeting where their
  circles are tangent, and every radius is an argument. It was a union of
  ellipses run through a Fourier smoother: right to look at, but with no
  parameters, so there was no way to ask for a bigger kidney or a deeper
  scoop, and nothing a test could check. Same footprint as before, 12.5 x 6.4 m
  over 54 m2, now a stated 18 x 36 ft rather than a coincidence. Its floor is a
  hopper -- flat at 1.0 m, sloping over the middle 45%, flat at 1.8 m under the
  drain -- rather than one ramp from wall to wall, and the drain, returns and
  skimmer are placed from the geometry, so they follow when the shape changes.
- `resolve()` takes `neighbours`, and `SensorContext` carries them, so robots
  are obstacles to each other in both the physics and the rangefinder. Empty
  for a single-robot run, which pays nothing for the fleet's existence.
- `RngTree.branch(name)` returns a whole sub-tree. Each robot's backend needs
  its own `"slip"` stream; sharing one tree would have them drawing in turn
  from a single sequence, so adding a fourth robot would change the first
  three's runs.
- A fleet recording carries `r0.x`, `r1.x` and so on, *and* robot 0's channels
  flat as `x`. The duplication is deliberate: every existing tool reads the
  flat names, so the replay window, the dirt cam, the dynamics module and the
  planner comparison all open a fleet recording and follow one member.
- `partitioned` cuts the *navigable* mask rather than the workspace. Cutting
  the workspace insets by the robot radius, and the single-robot planners then
  inset again, leaving a two-radius ring along the wall belonging to nobody --
  29 points of coverage on the kidney with three robots.
- `DARP` iterates up to 150 times rather than 60. It stops the moment it is
  within tolerance, so this costs nothing where 60 was enough; on the kidney it
  needs 93 and used to give up at 0.84 fairness instead of reaching 0.96.
- `PathFollower` gained a give-way rule (a lower-numbered robot has right of
  way, which is a total order so two robots cannot both defer) and a
  progress-based stall guard.
- `OccupancyMap.absorb` folds a tick of contact and sonar readings into the
  grid. It was the body of `SystematicCoverage._update_map`; moving it onto the
  map is what let the planners reuse it instead of copying it.
- `sweep_optimal` minimises `length + turn_cost * turning` rather than a
  length-dominated proxy. Lane count is the right criterion on a convex region,
  where it *is* the turn count; on the kidney pool the shortest sweep is 161 m
  and turns 6480 degrees while the least-turning one is 189 m and turns 3600,
  so the weighting is a real choice. The new cost picks neither, at 164 m and
  4320 degrees.

### Fixed
- Dirt appeared in the replay rather than accumulating. It is keyframed every
  ten simulated seconds and every view read the nearest keyframe at or before
  the frame's time, so a cell held still for five hundred rendered frames and
  then stepped: the heap the returns build at the deep end changed on 19 of
  10 000 frames. `dirt_at` and `debris_at` take `interpolate=True`, which the
  views use and measurements do not -- the exact keyframe is still the default,
  because it is the field the simulator really held. The largest single-frame
  change in that heap goes from 0.025 g to 0.00005 g.

  The blend has to be computed in `float32`. Keyframes are stored as `float16`
  to keep the file small, and weighting two of them at that precision loses
  more than the step being smoothed -- enough to land the result below both of
  the values it sits between.

- Debris was drawn where it started, for the whole run. Every view built the
  outlines once from the first frame on a stated assumption that debris settles
  and is only ever removed; `DebrisSet.nudge` moves it, because the robot
  shoves anything too big for the intake out of the way. Of the 20 items left
  in a 25-minute kidney run, the median had moved 1.26 m and 16 had moved
  further than their own length -- all painted at their `t=0` positions.
  `debris_polygons` now splits into `debris_offsets` (an outline about its own
  centre) plus a live centre, and items fade over the interval in which they
  were collected instead of winking out between frames.

- The dirt overlay saturated. Its scale is a percentile of the dirt a run
  *starts* with, and the corrected physics sweeps the floor into heaps more
  than twenty times that, so a heap was one flat patch whose edge was the only
  part that moved. Past the scale the colour now darkens towards silt, over a
  range taken from the heaviest cell the run actually holds.

- `tools/make_assets.py` builds `chase.gif`, which the README has shown since
  the chase cam landed with nothing in the repository that could regenerate it.

- `OccupancyMap` could only ever gain walls. `mark_free` refused to overwrite
  one, and walls are stamped from the *estimated* pose, so a drifting estimate
  painted a few spurious ones in open water every minute; over twenty-five
  minutes they joined up into a cage. The frontier search walks over non-wall
  cells, so it reported nothing reachable while a third of the pool sat
  uncovered on the far side, and `SystematicCoverage` declared a kidney
  finished at 23% coverage with twenty minutes of battery left. The hull now
  clears walls under itself, which is the one piece of evidence strong enough
  to overrule them -- a real wall wrongly cleared is stamped straight back by
  the next bump. `EvidenceMap` had already learned this for the online
  planners; the base map had not.

  This takes a documented result with it. The `systematic` docstring, the
  roadmap and the dynamics notes all carried a measurement showing that
  calibrating the odometry improved the estimate five-fold and *halved*
  coverage. The better the calibration, the sooner the cage closed; that was
  all it was measuring. Over five seeds, calibration now changes neither
  materially, and all three places say so.

- `SystematicCoverage` gave up permanently after 25 failed frontier attempts.
  Failures are now only counted while the robot is covering nothing new --
  squeezing along a wall towards a frontier is blocked on most ticks and still
  cleaning -- and running out of patience backs up and turns rather than
  parking for the rest of the run. Stopping for good is still what it does when
  the map genuinely has no frontier left.
- Dirt drift moved mass with an off-by-one upwind gate: it decided whether a
  cell could give dirt away by looking at the cell downstream, then rescaled
  the whole layer to put the difference back. Advection with no diffusion at
  all did the rest, so wherever a flow converged the field collapsed onto a
  one-cell line. On the kidney that line was a column carrying ten times its
  starting load, which read on screen as floor the robot had missed. Drift now
  sends dirt cell by cell -- nothing leaves that a valid neighbour cannot
  receive, so mass is conserved exactly rather than restored afterwards -- and
  carries a small diffusion term, refused outright above the stability limit.
- The kidney had a return at each end aimed at the other. Two opposed jets have
  a stagnation line in the middle and pile everything onto it; both returns now
  sit at the shallow end and sweep towards the drain, which is what every other
  preset already did.
- The kidney's main drain sat three metres short of the deep end, so the floor
  slope and the returns swept dirt straight past it into the far wall.
- `PathFollower`'s stall guard measured elapsed time rather than progress, so
  it confiscated a waypoint every twelve seconds while the robot was driving a
  nine-metre lane exactly as instructed -- skipping three quarters of a plan.
  It now resets whenever the robot gets meaningfully closer, and waiting for a
  team-mate does not count as stalling.
- `PathFollower` deadlocked against a wall. It advanced its waypoint index only
  on arrival -- within 18 cm -- while aiming a lookahead distance further
  along. A plan's first waypoint sits in a corner the hull cannot quite reach;
  the robot closed to 40 cm, started aiming at the far end of the lane,
  reversed out, found itself more than a lookahead from the corner again, and
  turned back. It paced a 15 cm stretch of tile for a whole run at 4% coverage.
  Pure pursuit now consumes every waypoint inside the lookahead circle.
- `binn` integrated the shunting equation forward and rang between its own
  bounds: with excitation 100 against decay 8, any step large enough to
  propagate activity overshoots, and on an even iteration count the field
  reported its lowest value at the cells that should be brightest. The robot
  paced between two cells and covered 0.4% of the pool. Iterating the
  equilibrium instead takes it to the top of the table.
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
