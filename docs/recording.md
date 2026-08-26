# Recording: the `.zbr` format

A recording contains everything needed to replay and audit one run. Backward
compatibility comes first: **a `.zbr` from six months ago must still open,
replay and identify exactly what produced it.**

## Layout

A `.zbr` is a ZIP archive. Rename it to `.zip` and look inside:

```
run.zbr
├── manifest.json    metadata, and the full resolved configuration
├── frames.npz       dense columnar time series, one array per channel
├── events.json      sparse events with payloads
├── dirt.npz         dirt-field and debris keyframes
├── metrics.json     the scalar metrics computed at the end
└── spatial.npz      the visit grid and dirt rasters behind those metrics
```

Two properties matter more than the layout:

**Inspectable.** The metadata is plain JSON. The arrays are `npz`, so
`numpy.load` reads them with no ZimaBlue installed and no custom decoder:

```python
import json, zipfile, numpy as np, io

with zipfile.ZipFile("runs/kidney.zbr") as z:
    manifest = json.loads(z.read("manifest.json"))
    frames = np.load(io.BytesIO(z.read("frames.npz")))
    print(manifest["seed"], frames["x"].shape)
```

**Self-contained.** The manifest embeds the *resolved* pool geometry (as WKT),
the robot configuration and the dirt spec — not the preset names they came
from. A recording made against last month's `kidney` preset still renders
correctly after the preset is reshaped, because the recording carries its own
copy.

## Why not MCAP?

[MCAP](https://mcap.dev/specification/index.html) is the right answer for
robotics logging in general and is the default `rosbag2` storage format. Its
principles are worth stealing — self-describing, schemas travel with the data,
indexed seek, cheap compression — and `.zbr` takes all of them.

MCAP itself is the wrong *shape* here. It is optimised for many heterogeneous
pub/sub channels arriving asynchronously. A ZimaBlue run is a small, fixed set
of dense, uniformly-sampled numeric columns plus a few sparse side channels.
That is a columnar array file, and the scientific-Python ecosystem already has
a portable one that every consumer of this data already has installed.

Converting `.zbr` to MCAP takes a small script over `frames.npz`. That script
belongs in a bridge package outside the core.

## Schema versioning

`manifest.json` carries `format: "zbr"` and an integer `schema_version`.

A reader **refuses** a recording whose schema version is newer than it
understands, rather than reading it optimistically. Misinterpreting a column
that has moved is worse than an error message.

```python
ValueError: runs/x.zbr uses .zbr schema v2, but this ZimaBlue (0.1.0)
understands up to v1. Upgrade ZimaBlue.
```

## Channels

Every channel is one `float32` (or `int32`) array with one entry per simulation
step. At the default 50 Hz, a 30-minute run is 90 000 frames per channel.

| Channel | Meaning |
|---|---|
| `time`, `step` | simulation clock and tick index |
| `x`, `y`, `heading` | ground-truth pose |
| `v`, `omega` | body-frame ground velocity |
| `wheel_left`, `wheel_right` | **track-surface** speeds — differ from ground speed under slip |
| `slip_left`, `slip_right` | fraction of commanded speed lost |
| `depth` | water depth at the robot |
| `battery`, `power` | state of charge, instantaneous draw |
| `filter_load` | grams retained |
| `distance`, `dirt_collected` | odometers |
| `contacts` | 4-bit field: front, left, right, rear |
| `collided`, `stuck` | flags |
| `cmd_left`, `cmd_right`, `cmd_brush`, `cmd_pump` | what the controller asked for |
| `<sensor>.<channel>` | every sensor output, already noisy |
| `<sensor>.valid` | 0 when that sample was dropped |

Recording both the *command* and the *achieved* motion is deliberate: the gap
between them is where slip, saturation and contact live.

A sensor channel is `NaN` before its first sample arrives — a 10 Hz sensor with
20 ms of latency has nothing to report on frame 0, and inventing a value would
be worse than admitting it.

## Events

Dense columns are the wrong home for things that happen rarely and carry a
payload, so events live separately:

```json
{"time": 41.6, "kind": "collision",
 "detail": {"x": 9.71, "y": 4.72, "penetration": 0.004, "obstacle": false}}
```

Kinds: `collision`, `stuck`, `unstuck`, `filter_full`, `battery_low`,
`battery_empty`, `debris_collected`, `debris_blocked`.

Collisions and blockages are **edge-triggered**. A robot pressed against a wall
for two seconds is one collision, not a hundred; sustained contact is already
visible in the per-frame `contacts` column. Before this was fixed, a two-minute
run logged 3 937 events, of which 3 820 were the same leaf being pushed along.

## Dirt keyframes

Dirt is snapshotted every 10 simulated seconds rather than every frame.

The arithmetic: a 30-minute run at 50 Hz over a 65×125 raster with four dirt
layers would be about 90 000 × 4 × 8 125 × 4 bytes — roughly 12 GB — for a
field that changes by a fraction of a gram per tick. Keyframes reproduce it to
the eye and to within a gram, at 0.1% of the size.

Keyframes are `float16`. They are a visualisation and analysis artefact; the
metrics that matter are computed from the live field and stored separately in
`metrics.json`. Across the range a cell actually covers — a fifth of a gram of
ordinary sediment up to the eight or so grams a heap at the drain reaches —
`float16` resolves to a few milligrams at worst.

It is a *storage* format and nothing more: reading it back widens to `float32`
first, because summing the layers or weighting two keyframes at this precision
loses far more than the quantisation does.

Lookup takes the nearest keyframe at or *before* the requested time. That is
the exact field the simulator held at a moment it really recorded, which is
what a measurement wants, so it is the default — `ergodic_score` and
`forecast_cleaning` both read it that way.

`dirt_at(t, interpolate=True)` blends linearly towards the next keyframe
instead, and the replay views use it. Ten seconds is five hundred rendered
frames, so without the blend a cell holds still and then jumps, and dirt reads
as appearing rather than accumulating. The blend remains opt-in because it is
an estimate: a cell cleaned early in an interval fades across the whole of it
rather than dropping when it was really cleaned. It cannot invent dirt —
every value it returns lies between two the cell genuinely held — and it is
computed in `float32`, because doing the weighting in the stored `float16`
loses more than the step it is smoothing.

Discrete debris is keyframed alongside the field, as an `(n, 6)` table of
`x, y, mass, size, collected, type`. Debris *moves*: the robot shoves anything
too big for the intake out of the way. `debris_at(t, interpolate=True)` glides
each item between keyframes and turns `collected` into the fraction of the way
through the interval in which it was picked up, so a view can fade it out;
anything counting items stays on the exact reading, where the column is 0 or 1.
Row order is the item's identity in both. `type` indexes
`manifest["dirt_types"]["debris"]`, and it is there because a leaf and a twig
are not the same object: without it a replay can only draw both as the same
anonymous blob. Recordings written before the column existed are padded rather
than rejected, so every item in them reads as the first type.

## Size

A 20-minute kidney run with the standard sensor suite is about 2–3 MB.
Roughly two thirds is the frame table and one third the dirt keyframes.

To make one smaller:

```python
zb.Simulation(..., dirt_keyframe_interval=30.0)  # fewer dirt snapshots
zb.Simulation(..., timestep=0.05)  # 20 Hz instead of 50
zb.Simulation(..., record=False)  # metrics only
```

Widening the keyframe interval costs replay smoothness, not correctness: the
views blend between keyframes, and the further apart they are the more of the
timing that blend is guessing at.

`record=False` is the right default for batch sweeps, and `run_batch` uses it
unless you pass `--record-dir`.

## Reading one back

```python
from zimablue.recording import Recording

rec = Recording.load("runs/kidney.zbr")
print(rec.describe())

rec.n_frames, rec.duration, rec.seed
rec.column("x")  # a channel, as an array
rec.frame_index_at(120.0)  # index of the frame at t = 2 min
rec.dirt_at(120.0)  # summed dirt raster at that moment
rec.events_between(100.0, 200.0)
rec.metrics["coverage"]
```

Or from the shell:

```bash
zimablue inspect runs/kidney.zbr --channels --events
```

## Determinism

Same ZimaBlue version, same platform, same scenario, same seed ⇒ **bit-identical
frame arrays.** `tests/test_determinism.py` asserts exactly that, including
across a save/load cycle.

Cross-platform bit-identity is *not* promised. Floating-point library
differences make it unenforceable, and better-resourced simulators document the
same limitation — see
[Isaac Lab on reproducibility](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html).
The seed and the full configuration are embedded precisely so that a run can be
*regenerated* anywhere, even where the bits differ.
