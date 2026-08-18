# Architecture

ZimaBlue is organised around one rule:

> **The domain model is the public API. Everything that computes is a backend.**

A pool, a cleaner, dirt and a scenario are ZimaBlue concepts and stay stable. How
the next state is computed — a fast analytical 2D step today, PhysX/USD
tomorrow — is an implementation detail behind an interface.

## Layers

```
                            ┌──────────────────────────┐
   user-facing              │   Scenario  ·  CLI       │
                            └────────────┬─────────────┘
                                         │
                            ┌────────────▼─────────────┐
   orchestration            │       Simulation         │
                            │  fixed-dt stepping loop  │
                            └───┬──────────┬───────┬───┘
                                │          │       │
        ┌───────────────────────┘          │       └──────────────┐
        │                                  │                      │
┌───────▼────────┐              ┌──────────▼─────────┐   ┌────────▼────────┐
│  World model   │              │    Robot model     │   │   Controller    │
│  Pool          │              │  Cleaner           │   │  (replaceable)  │
│   geometry     │              │   chassis          │   └─────────────────┘
│   depth field  │              │   drive            │
│   features     │              │   cleaning head    │
│  Water         │              │   power / battery  │
│  DirtField     │              │   sensors ─────────┼──► observations
│  Debris        │              └────────────────────┘
└────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       │
           ┌───────────▼────────────┐
           │   SimulationBackend    │   ← interface
           ├────────────────────────┤
           │  Fast2DBackend  (v0.1) │
           │  IsaacSimBackend (tbd) │
           └───────────┬────────────┘
                       │
      ┌────────────────┼─────────────────┐
      │                │                 │
┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
│  Metrics  │   │  Recording  │   │   Events    │
└───────────┘   └──────┬──────┘   └─────────────┘
                       │
                 ┌─────▼─────┐
                 │  .zbr on  │
                 │   disk    │
                 └─────┬─────┘
                       │
                 ┌─────▼─────┐
                 │  Replay   │
                 └───────────┘
```

## Package map

| Module | Responsibility |
|---|---|
| `zimablue.pool` | Pool geometry, depth field, surface features, presets |
| `zimablue.robot` | Cleaner and its components; presets |
| `zimablue.sensors` | Sensor base + noise/fault pipeline; the five sensor models |
| `zimablue.dirt` | `DirtType`, `DirtField`, `Debris`, deterministic generators |
| `zimablue.physics` | Diff-drive kinematics, collision resolution, cleaning interaction |
| `zimablue.backends` | `SimulationBackend` protocol, `Fast2DBackend`, registry |
| `zimablue.simulation` | `Simulation`, `SimState`, `StepResult`, `RunResult` |
| `zimablue.controllers` | `Controller` protocol, the shipped controllers and oracles, registry |
| `zimablue.estimation` | EKF over position, heading and gyro bias; ZUPT |
| `zimablue.metrics` | Scalar metrics + spatial companions |
| `zimablue.recording` | `.zbr` writer/reader, schema versioning |
| `zimablue.replay` | Renderer, interactive player, headless exporters |
| `zimablue.notebook` | `preview()` — the pool as a turnable page in a notebook |
| `zimablue.scenarios` | YAML schema, loading, resolution, batch runner |
| `zimablue.geometry` | Rasters, rings, angle helpers |
| `zimablue.rng` | Named child-stream RNG tree |
| `zimablue.cli` | Typer application |

Dependencies point *down* this table, never up. `pool` knows nothing about
`simulation`; `simulation` knows nothing about `cli`.

Three modules sit outside it, each behind an optional extra, and each importing
*into* the core rather than being imported by it.

| Module | Extra | Responsibility |
|---|---|---|
| `zimablue.imaging` | `[image]` | Trace a pool out of a photograph |
| `zimablue.segment` | `[ml]` | SAM over onnxruntime as an alternative water mask |
| `zimablue.rl` | `[rl]` | Gymnasium env, extra observations, policy-as-controller |

The extra is a *dependency* boundary, not an import one, and only `segment` and
`rl` are both. `imaging` is imported at package import because `pool_from_image`
lives in the top-level namespace, so what it defers is Pillow — the import sits
inside the functions that read a file, and a test asserts that `import zimablue`
leaves `PIL` out of `sys.modules`. The same test covers onnxruntime and
gymnasium, which have the easier job of being in modules nothing imports.

## The stepping loop

One tick of `Simulation.step()`:

1. **Sense** — each sensor is polled. A sensor only produces a new sample when
   its own sampling period has elapsed; otherwise the last sample is held.
   Observations are built from ground truth plus the sensor's imperfection
   pipeline.
2. **Decide** — the controller receives the observation bundle (plus elapsed
   time and battery) and returns a `DriveCommand` (left/right wheel speed,
   brush on/off, pump duty). Controllers never see ground truth unless they ask
   for the debug channel, which is off by default.
3. **Actuate** — motor models clamp the command to torque/acceleration limits.
4. **Integrate** — the backend advances pose with slip and traction, resolves
   wall/obstacle contact, and updates battery from drive + brush + pump load.
5. **Clean** — the cleaning model computes dirt removal over the footprint swept
   during this tick and updates the dirt field, filter load and debris list.
6. **Record** — a frame is appended to the recorder; events are emitted for
   collisions, stuck detection, faults and filter saturation.
7. **Score** — the visit grid and running metric accumulators are updated.

Steps 1–7 contain no wall-clock reads, no unseeded randomness and no dict
iteration whose order can vary. That is what makes replay exact.

## Determinism contract

ZimaBlue promises: **same ZimaBlue version + same platform + same scenario +
same seed ⇒ bit-identical recording.**

It does *not* promise cross-platform bit-identity; floating-point library
differences make that unenforceable, and the
[Isaac Lab reproducibility notes](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html)
document the same limitation for a much better-resourced simulator.

The mechanism is `zimablue.rng.RngTree`. A root seed is expanded with
`numpy.random.SeedSequence`, and every consumer asks for a *named* stream:

```python
rng = RngTree(42)
dirt_rng = rng.stream("dirt")
imu_rng = rng.stream("sensor:imu")
```

Names are hashed into the spawn key, so a stream's values depend only on the
root seed and its own name. Adding a sixth sensor does not shift the fifth
sensor's noise.

## Backend interface

```python
class SimulationBackend(Protocol):
    name: str

    def reset(self, world: World, robot: Cleaner, rng: RngTree) -> SimState: ...
    def step(self, state: SimState, command: DriveCommand, dt: float) -> StepResult: ...
    def sense(self, state: SimState) -> dict[str, Observation]: ...
    def close(self) -> None: ...
```

A backend owns *dynamics and sensing*. It does not own the pool, the dirt, the
metrics or the recording — those are computed by shared code against the state
the backend returns, so any backend gets them for free.

### Fast2DBackend

The v0.1 reference implementation. Planar rigid body, differential-drive
kinematics with longitudinal slip and lateral traction limits, penetration-based
wall and obstacle resolution against Shapely geometry, and analytical sensor
models. Pure NumPy float64 on the CPU. Wall climbing is modelled as an
unrolled-perimeter 1D excursion rather than true 3D motion — an explicit,
documented approximation.

### 3D backend: intended design

Not implemented in v0.1. The interface above is what it must satisfy. The
intended shape, informed by
[NVIDIA's OpenUSD robotics guidance](https://developer.nvidia.com/blog/using-openusd-for-modular-and-scalable-robotic-simulation-and-development/):

- **Asset generation.** `Pool` → USD stage. The pool footprint is already a
  Shapely polygon and the depth field is already a raster, so the floor is an
  extruded/warped mesh and walls are swept from the boundary. `Cleaner`
  components carry mass, dimensions and mounting transforms, which is enough to
  emit a `UsdPhysics`-annotated articulation.
- **Physics.** PhysX rigid bodies with a buoyancy and drag force applied per
  step from the water properties already on `World`.
- **Sensors.** The same `Sensor` imperfection pipeline wraps rendered/raycast
  ground truth, so noise behaviour is shared between 2D and 3D and only the
  ground-truth source differs. Cameras and depth sensors are new sensor classes,
  not a new sensor framework.
- **Recording.** Unchanged. A 3D run writes the same `.zbr` columns plus optional
  image channels stored as separate members in the container.
- **Isolation.** Everything Omniverse-specific lives in
  `zimablue/backends/isaac/` behind an optional extra. Importing `zimablue` must
  never import `omni.*`.

The test that the boundary holds: a `.zbr` produced by the 3D backend must be
replayable by the 2D replay viewer, with the extra channels ignored.

### No backend at all

`zimablue.hardware` is the same argument taken one step further. A backend owns
dynamics and sensing; on a robot those belong to physics and to a driver, and
what is left is the part `Simulation.step` was doing anyway — build a
`ControlInput`, ask the controller, write the answer somewhere, record the
frame.

So there is no `HardwareBackend`. A backend has to return a `SimState` it
integrated and a `World` it owns, and a robot has neither: it cannot report its
own true pose, and there is no dirt field to account against. Pretending
otherwise would make `metrics.coverage` computable from the pose estimate,
which is a number that looks exactly like the simulated one and means something
else. `HardwareRuntime` is a sibling of `Simulation` rather than a backend
under it, and it produces a deliberately shorter set of metrics.

What *is* shared is the part that must not diverge: `recording.build_frame`
turns a state, a command and a set of readings into `.zbr` columns, and both
callers use it. The same acceptance test applies — a recording written on a
robot has to replay in the ordinary viewer.

## Extension points

| To add… | Implement | Register with |
|---|---|---|
| a pool shape | a function returning `Pool` | `@pool_preset("name")` |
| a robot | compose `Cleaner(...)` | `@robot_preset("name")` |
| a sensor | subclass `Sensor` | pass into `Cleaner(sensors=[...])` |
| a dirt scenario | a function returning `DirtSpec` | `@dirt_preset("name")` |
| a controller | satisfy `Controller` | `@controller_preset("name")` |
| a backend | satisfy `SimulationBackend` | `@backend("name")` |
| a robot's sensors | satisfy `ReadingSource` | pass to `HardwareRuntime` |

Registries are plain dicts with decorator sugar and a lookup that raises a
`KeyError` listing the valid names. No plugin framework, no entry points, no
metaclasses — those can be added when there is a second consumer that needs
them.

## Deliberate non-goals for v0.1

- Computational fluid dynamics. Water motion is a static drift field plus
  resuspension heuristics.
- True 3D wall traversal in the 2D backend.
- A GUI beyond the matplotlib replay player.
- ROS 2 bridge. The recording format is the interchange boundary; a bridge is a
  consumer of it, not a dependency of it.
