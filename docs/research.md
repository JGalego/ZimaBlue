# Research notes

These are the notes that shaped ZimaBlue's architecture. They are deliberately
opinionated: the goal was not a literature survey but a set of decisions with
reasons attached. Every section ends with **→ what ZimaBlue does**, so the
design can be traced back to the evidence.

Last reviewed: 2026-08.

---

## 1. What a robotic pool cleaner actually is

A modern robotic pool cleaner is a small, tethered or battery-powered,
near-neutrally-buoyant crawler. It is not a free-swimming AUV and it is not a
floor vacuum, and both differences matter for simulation.

The recurring mechanical anatomy across commercial units
([Maytronics overview](https://www.maytronics.com/en-us/what-is-a-robotic-pool-cleaner.html),
[Madimack teardown-style guide](https://madimack.com/us/blog/how-robotic-pool-cleaners-work-a-complete-guide),
[US 11,619,060 "Robotic pool cleaner with extended brush assembly"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11619060)):

- **Locomotion**: two independently driven tracks or wheel pairs — i.e. a
  differential drive — with enough traction to climb walls.
- **Agitation**: one or more rotating brushes that dislodge algae, biofilm and
  adhered grime. Brush material is matched to the surface (rubber for
  plaster/concrete, softer foam for vinyl/fiberglass).
- **Suction**: an impeller pump that pulls water and loosened debris through an
  intake into a filter basket or cartridge.
- **Filtration**: finite capacity. A full filter degrades suction, which is a
  real failure mode worth simulating.
- **Navigation**: older units use timers and bump-and-turn; newer ones use
  gyroscopes and accelerometers for heading hold and systematic scanning, with
  premium units adding cameras or LiDAR
  ([Dolphin Sigma / SmartNav](https://poolrobots.com/products/dolphin-sigma-robotic-pool-cleaner),
  [navigation overview](https://www.techloy.com/how-robotic-pool-cleaners-navigate-underwater/)).

The important insight, stated plainly in the vendor literature: **navigation and
cleaning are separable subsystems**. Navigation gets the robot over the debris;
brushes, intake flow and filtration determine whether the debris is actually
removed once it gets there. A robot can cover 100% of the floor and still leave
the pool dirty.

> **→ what ZimaBlue does.** The `Cleaner` is a composition of independent
> components (`chassis`, `drive`, `cleaning`, `power`, `sensors`) rather than a
> monolithic robot class, and the metrics subsystem reports *coverage* and
> *dirt removed* as two separate first-class numbers. Filter capacity is
> modelled and saturating it degrades pickup.

## 2. Coverage path planning

The canonical reference is Galceran & Carreras, *A survey on coverage path
planning for robotics*, Robotics and Autonomous Systems 61(12), 2013
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S092188901300167X)).
Coverage path planning (CPP) is the problem of finding a path that passes over
every point of a target area while avoiding obstacles. The survey's taxonomy —
heuristic / cellular-decomposition / grid-based, complete vs. partial, online
vs. offline — is the vocabulary the rest of the field uses.

The classical exact method is **boustrophedon ("ox-turning") cellular
decomposition** (Choset & Pignon; see
[Springer](https://link.springer.com/chapter/10.1007/978-1-4471-1273-0_32)),
which improves on trapezoidal decomposition by merging cells that do not need
splitting, then covers each cell with simple back-and-forth motions and solves
the remaining problem as an exhaustive walk over the cell-adjacency graph.
Spanning-tree coverage (STC) and its descendants are the other major family, and
recent work still builds on both
([full-coverage planning in semi-structured environments](https://www.sciencedirect.com/science/article/abs/pii/S0921889025001368)).

Pool-specific CPP work exists but is thin. Two useful anchors:

- *Optimising Robotic Pool-Cleaning with a Genetic Algorithm*, Journal of
  Intelligent & Robotic Systems, 2019
  ([DOI](https://dl.acm.org/doi/abs/10.1007/s10846-018-0953-y)) — evolves
  near-optimal cleaning sequences minimising energy for a full-pool clean. The
  framing of *energy* as the objective, not time, is notable.
- *Autonomous Pool Cleaning: Self Localization and Autonomous Navigation for
  Cleaning*
  ([ResearchGate](https://www.researchgate.net/publication/235435232_Autonomous_Pool_Cleaning_Self_Localization_and_Autonomous_Navigation_for_Cleaning))
  — Kalman filtering over sonar returns plus a robot dynamic model to estimate
  pose inside a pool.

Evaluation metrics that recur in this literature: coverage completeness (area
cleaned above a threshold ÷ total area), **path overlap / revisit rate**, energy
consumption, and cleaning uniformity.

> **→ what ZimaBlue does.** The baseline controller is a boustrophedon
> lawnmower with a perimeter (wall-following) pass and a stuck-recovery
> behaviour — deliberately simple, so it is a *baseline* rather than a
> contribution. It is swappable behind a small `Controller` protocol so that
> STC, GA, or learned planners can be dropped in without touching the
> simulator. `revisits`, `energy_consumed` and `coverage` are all in the metrics
> table because the literature evaluates on them.

## 3. Localization underwater

Pool cleaners cannot use GNSS and rarely carry a DVL. The underwater
localization literature is about drift management: dead reckoning from IMU alone
is accurate in theory but degrades with real hardware, so IMU is fused with a
velocity reference (DVL), a depth reference (pressure), and an exteroceptive
bound (sonar or camera)
([AQUALOC dataset](https://arxiv.org/pdf/1910.14532),
[SVIn2](https://rlab.cs.dartmouth.edu/publications/rahman2022svin2.pdf),
[TURTLMap](https://arxiv.org/html/2408.01569v2)).
The consistent message: **dead-reckoning error is unbounded and grows with
distance travelled**, and pressure depth is the one cheap, drift-free channel.

A pool cleaner's practical sensor set is therefore: wheel/track encoders (a poor
man's DVL, corrupted by slip), a MEMS IMU for heading, a pressure sensor for
depth (which, in a pool of known bathymetry, is also a weak *position* prior),
contact/bump switches, and possibly a short-range sonar.

> **→ what ZimaBlue does.** Exactly that sensor set ships in v0.1: encoder, IMU,
> pressure/depth, contact, sonar. Encoders report *commanded wheel motion
> corrupted by the slip the physics actually applied*, so odometry drifts for
> the right reason rather than by an arbitrary injected error. The simulator
> exposes ground-truth pose separately from sensor observations, so a user can
> build and grade a state estimator against truth.

## 4. Sensor realism

Two sources set the bar for what "realistic sensor" means without becoming a
research project of its own.

**IMU noise models.** The standard characterisation is Allan variance, which
decomposes inertial sensor error into quantization noise, angle/velocity random
walk, bias instability, rate random walk and rate ramp
([Analysis and Modeling of Inertial Sensors Using Allan Variance](https://www.researchgate.net/publication/3094132_Analysis_and_Modeling_of_Inertial_Sensors_Using_Allan_Variance),
[Kalibr IMU noise model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)).
The practically dominant terms are white noise plus a slowly varying bias — a
first-order Gauss–Markov process is the usual tractable model.

**Why bother.** The sim-to-real literature is blunt about the cost of perfect
sensors: policies trained on idealised sensor data learn to exploit information
that is unavailable or corrupted on real hardware, and improving sensor
simulation directly improves transfer. Domain randomisation
([overview](https://www.emergentmind.com/topics/domain-randomization-dr))
generalises this into deliberately randomising generative parameters so models
must learn invariant features.

> **→ what ZimaBlue does.** Every sensor derives from one `Sensor` base with a
> shared, composable imperfection pipeline: sample rate, Gaussian white noise,
> constant + random-walk bias, latency (a delay line), dropout, saturation and
> stuck-value faults. That is the white-noise + Gauss–Markov-bias core of the
> Allan model, not the full five-term decomposition — the honest simplification.
> `inject_fault()` mutates those parameters at runtime so faults can start
> mid-run, which is what fault-injection testing actually needs. Every noise
> draw comes from a per-sensor child RNG so adding a sensor does not perturb
> another sensor's stream.

## 5. What to learn from the big simulators (and what not to copy)

| | Strength | Cost |
|---|---|---|
| **Gazebo** | Deep ROS 2 integration; sensors publish as topics; mature plugin model | Heavy; ROS 2 in the critical path |
| **MuJoCo** | Convex-optimisation contact solver, deterministic stepping, hundreds of × real-time on one core | Contact-rich manipulation focus; not a domain model |
| **Isaac Sim** | Photorealistic synthetic data (Replicator), thousands of GPU-parallel envs (Isaac Lab), open-sourced in 5.0 | GPU, Omniverse, large assets ([arXiv:2606.03551](https://arxiv.org/pdf/2606.03551)) |

Sources: [MuJoCo/Isaac/Gazebo comparison](https://www.trossenrobotics.com/post/robot-arm-simulation-mujoco-isaac-sim-gazebo),
[Black Coffee Robotics 2026 perspective](https://www.blackcoffeerobotics.com/blog/which-robot-simulation-software-to-use).

The scene-description layer converged on **OpenUSD**: Isaac Sim ingests CAD /
URDF / captures and converts to USD, then physics and materials are authored on
the USD prims; "SimReady" assets are USD models with semantic labels and
`USDPhysics` properties attached
([NVIDIA on OpenUSD for modular robotic simulation](https://developer.nvidia.com/blog/using-openusd-for-modular-and-scalable-robotic-simulation-and-development/),
[Isaac Sim OpenUSD fundamentals](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/open_usd.html),
[Learn OpenUSD: Robotics Best Practices, SIGGRAPH 2025](https://dl.acm.org/doi/full/10.1145/3721251.3736528)).

The lesson taken from all three is a *negative* one. Each of these systems makes
its own engine the public API: you write against Gazebo's plugins, or MuJoCo's
`mjModel`, or USD prims. Anything domain-specific you build sits on top and
cannot outlive the engine. For a pool-cleaner testbed the durable asset is the
domain model — pool, dirt, cleaner, scenario, run — not the integrator.

> **→ what ZimaBlue does.** The domain model is the API; a backend is a
> replaceable strategy behind `SimulationBackend`. `Fast2DBackend` is the
> reference implementation. A future `IsaacSimBackend` is expected to *consume*
> `Pool`/`Cleaner` and emit USD, not to replace them — see
> [`architecture.md`](architecture.md#3d-backend-intended-design). Nothing in
> the core imports NumPy-adjacent heavy machinery beyond NumPy itself, and
> matplotlib is an optional extra.

## 6. Determinism and reproducibility

Isaac Lab's reproducibility page is the most candid public statement of the
problem: given the same hardware and version, results are identical; **across
different hardware they are not**, because GPU work scheduling changes operation
ordering and floating-point non-associativity turns that into least-significant-
bit divergence that compounds over thousands of steps
([Isaac Lab: Reproducibility and Determinism](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html),
[NVIDIA on deterministic replay](https://perspectives.nvidia.com/isaac-sim/task/faq/deterministic-replay-mechanisms-reproducible-benchmarks-1/)).
Their prescription: lock physics parameters, pin assets, control stepping
manually, seed every RNG. ROS 2-side work makes the same argument for
node-graph-level replay
([RSLCPP, arXiv:2601.07052](https://arxiv.org/pdf/2601.07052)).

The other half of reproducibility is *provenance*: a seed is useless if you
cannot recover the configuration it was applied to.

> **→ what ZimaBlue does.** Fixed timestep, no wall-clock anywhere in the
> stepping path, and a single seeded `numpy.random.Generator` tree — the root
> seed spawns named, stable child streams (`dirt`, `sensor:imu`, `slip`, …) via
> `SeedSequence`, so adding a consumer never shifts another's draws. The CPU
> float64 path avoids the GPU-ordering hazard entirely at v0.1's scale. Every
> `.zbr` recording embeds the full resolved scenario, the seed and the producing
> ZimaBlue version, and `tests/test_determinism.py` asserts that same
> scenario + same seed ⇒ bit-identical frame arrays. Determinism is promised
> *per platform and version*, which is the only promise the evidence supports.

## 7. Recording and replay formats

**MCAP** is the reference point: a self-describing, append-only, row-oriented
binary container for heterogeneous timestamped channels, with embedded schemas,
indexed seeking and LZ4/Zstd compression; it is the default `rosbag2` storage
plugin from ROS 2 Iron onward
([spec](https://mcap.dev/specification/index.html),
[Foxglove announcement](https://foxglove.dev/blog/mcap-as-the-ros2-default-bag-format),
[format evaluation, Hurliman 2021](https://mcap.dev/files/evaluation.pdf)).
The design principles worth stealing: self-describing (schema travels with the
data), append-friendly write path, indexed seek, and cheap compression.

MCAP itself is the wrong fit for v0.1: it is optimised for many heterogeneous
pub/sub channels arriving asynchronously, whereas a ZimaBlue run is a small,
*fixed* set of dense, uniformly-sampled numeric columns plus a handful of sparse
event streams. That shape is a columnar array file, not a message log — and the
scientific-Python ecosystem already has a portable one.

> **→ what ZimaBlue does.** `.zbr` is a ZIP container:
> `manifest.json` (human-readable metadata: format, schema version, ZimaBlue
> version, seed, resolved scenario, channel descriptors, metrics),
> `frames.npz` (compressed columnar `float32`/`int32` arrays — pose, velocity,
> commands, battery, per-sensor observations), `events.json` (sparse events:
> collisions, stuck, faults, filter-full), and `dirt/` keyframes (the dirt
> raster snapshotted every N steps rather than every step, since it changes
> slowly and storing it densely would dominate the file). Unzip it and the
> metadata is readable in any text editor; `np.load` reads the arrays without
> ZimaBlue installed. Schema version is explicit and checked on load.

## 8. Dirt: what is actually in a pool

Pool contamination is not one substance, and the differences are exactly what
makes a cleaning-quality metric non-trivial:

- **Fine sediment / silt** — settles slowly, resuspends easily, spreads thin.
- **Sand** — dense, settles fast, collects in low points and dead zones.
- **Leaves and twigs** — discrete, large, initially floating then waterlogged;
  they can clog an intake rather than pass through it.
- **Floating debris** — pollen, insects, films; interacts with skimmers, not the
  floor robot.
- **Algae bloom and biofilm** — *adhered*, not deposited. Requires mechanical
  agitation; suction alone does nothing.

The physics of the settled fraction is Stokes' law: settling velocity scales
with the square of particle diameter and with the density difference between
particle and fluid
([Stokes settling primer](https://www.geological-digressions.com/fluid-flow-stokes-law-and-particle-settling/)).
Two complications from the sediment-transport literature are worth knowing about
even when not modelled: biofilm growth on particles measurably changes their
drag and settling velocity — increases of up to ~130% for non-buoyant
microplastics
([Nature Comms Earth & Environment](https://www.nature.com/articles/s43247-023-00690-z),
[Biofilm effects on settling velocity](https://www.sciencedirect.com/science/article/abs/pii/S1001627914600603)) —
and fine particles flocculate, while dense suspensions settle *slower* than
Stokes predicts through hindered settling
([arXiv:1812.01365](https://arxiv.org/pdf/1812.01365)).

> **→ what ZimaBlue does.** Dirt is a first-class entity with two
> representations that coexist: a **continuous raster field** (grams per cell,
> one layer per dirt type) for sediment/sand/algae, and **discrete `Debris`
> items** for leaves and twigs, which have a position, a mass and a
> waterlogging state. Each `DirtType` carries `density`, `particle_size`,
> `buoyancy`, `settling_velocity`, `adhesion` and `pickup_difficulty`. Settling
> velocity is *derived from Stokes' law* at construction time from particle size
> and density rather than hand-tuned per preset, so the presets stay physically
> ordered. Flocculation and hindered settling are explicitly **not** modelled
> and are noted as such in the code — the aim is useful simulation, not fake
> precision.

## 9. Benchmark metrics

IEC 62929:2014, *Cleaning robots for household use – Dry cleaning: methods of
measuring performance*, is the closest thing to a standard
([IEC webstore](https://webstore.iec.ch/en/publication/7477),
[summary](https://standards.globalspec.com/std/9859111/IEC%2062929)). It
specifies dust-removal ability, autonomous navigation, a coverage test, and
average speed. The research literature adds coverage completeness against a
threshold, path overlap rate, energy efficiency and cleaning uniformity
([SHIFT Planner, arXiv:2412.10706](https://arxiv.org/pdf/2412.10706)).

The distinction IEC draws between a **coverage test** and a **dust-removal
test** is the same distinction that motivates ZimaBlue.

> **→ what ZimaBlue does.** The metrics module reports both families and keeps
> them apart: geometric (`coverage`, `floor_coverage`, `wall_coverage`,
> `revisits`, `distance_traveled`) and cleaning-quality (`dirt_removed`,
> `dirt_removed_fraction`, `remaining_dirt`, per-dirt-type breakdown,
> `cleaning_uniformity`). Each scalar has a spatial companion array — the
> visit-count grid and the remaining-dirt grid — so the replay can *show* the
> difference between "drove everywhere" and "cleaned everything".

---

## Summary of architectural consequences

| Finding | Consequence in ZimaBlue |
|---|---|
| Navigation and cleaning are separable | `coverage` ≠ `dirt_removed`; both are first-class metrics |
| Cleaner = brushes + suction + filter | Component-composed `Cleaner`; filter fill degrades pickup |
| Differential drive, wall climbing | Diff-drive kinematics with slip; wall segments are cleanable surfaces |
| Dead reckoning drifts, depth does not | Encoders corrupted by real slip; pressure is the clean channel |
| Idealised sensors break transfer | Shared noise/bias/latency/dropout/fault pipeline on every sensor |
| Engines make themselves the API | Domain model is the API; `SimulationBackend` is a strategy |
| GPU ordering breaks determinism | CPU float64, fixed dt, seeded `SeedSequence` child streams |
| MCAP: self-describing + indexed | `.zbr` = readable JSON manifest + columnar `npz` + sparse events |
| Stokes' law orders settling behaviour | `settling_velocity` derived from size and density, not tuned |
| IEC 62929 separates coverage from removal | Two metric families, each with a spatial companion |

## Sources

- Galceran & Carreras, [A survey on coverage path planning for robotics](https://www.sciencedirect.com/science/article/abs/pii/S092188901300167X), RAS 2013
- Choset & Pignon, [Coverage Path Planning: The Boustrophedon Cellular Decomposition](https://link.springer.com/chapter/10.1007/978-1-4471-1273-0_32)
- [Optimising Robotic Pool-Cleaning with a Genetic Algorithm](https://dl.acm.org/doi/abs/10.1007/s10846-018-0953-y), JINT
- [Autonomous Pool Cleaning: Self Localization and Autonomous Navigation](https://www.researchgate.net/publication/235435232_Autonomous_Pool_Cleaning_Self_Localization_and_Autonomous_Navigation_for_Cleaning)
- [AQUALOC: An Underwater Dataset for Visual-Inertial-Pressure Localization](https://arxiv.org/pdf/1910.14532)
- [SVIn2: A multi-sensor fusion-based underwater SLAM system](https://rlab.cs.dartmouth.edu/publications/rahman2022svin2.pdf)
- [TURTLMap](https://arxiv.org/html/2408.01569v2)
- [Kalibr IMU Noise Model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)
- [Analysis and Modeling of Inertial Sensors Using Allan Variance](https://www.researchgate.net/publication/3094132_Analysis_and_Modeling_of_Inertial_Sensors_Using_Allan_Variance)
- [Domain randomization overview](https://www.emergentmind.com/topics/domain-randomization-dr)
- [NVIDIA Isaac Sim: Enabling Scalable, GPU-Accelerated Simulation for Robotics](https://arxiv.org/pdf/2606.03551)
- [Isaac Lab: Reproducibility and Determinism](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html)
- [Using OpenUSD for Modular and Scalable Robotic Simulation](https://developer.nvidia.com/blog/using-openusd-for-modular-and-scalable-robotic-simulation-and-development/)
- [Isaac Sim OpenUSD Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/open_usd.html)
- [Learn OpenUSD: Robotics Best Practices](https://dl.acm.org/doi/full/10.1145/3721251.3736528), SIGGRAPH 2025
- [MCAP specification](https://mcap.dev/specification/index.html) and [format evaluation](https://mcap.dev/files/evaluation.pdf)
- [MCAP as the ROS 2 default bag format](https://foxglove.dev/blog/mcap-as-the-ros2-default-bag-format)
- [RSLCPP — Deterministic Simulations Using ROS 2](https://arxiv.org/pdf/2601.07052)
- [IEC 62929:2014](https://webstore.iec.ch/en/publication/7477)
- [SHIFT Planner](https://arxiv.org/pdf/2412.10706)
- [Stokes' law and particle settling](https://www.geological-digressions.com/fluid-flow-stokes-law-and-particle-settling/)
- [Non-buoyant microplastic settling velocity varies with biofilm growth](https://www.nature.com/articles/s43247-023-00690-z)
- [Biofilm effects on settling velocity of sediment particles](https://www.sciencedirect.com/science/article/abs/pii/S1001627914600603)
- [Estimating the settling velocity of fine sediment at high concentrations](https://arxiv.org/pdf/1812.01365)
- [Maytronics: What is a robotic pool cleaner](https://www.maytronics.com/en-us/what-is-a-robotic-pool-cleaner.html)
- [Madimack: How robotic pool cleaners work](https://madimack.com/us/blog/how-robotic-pool-cleaners-work-a-complete-guide)
- [US 11,619,060 — Robotic pool cleaner with extended brush assembly](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11619060)
- [US 11,274,462 — Controlling a movement of a pool cleaning robot](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11274462)
