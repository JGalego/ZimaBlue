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

[Unreleased]: https://github.com/JGalego/ZimaBlue/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JGalego/ZimaBlue/releases/tag/v0.1.0
