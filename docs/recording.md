# Recording — the `.zbr` format

A recording is everything needed to replay and audit one run. The design goal
is not compactness first: it is that **a `.zbr` from six months ago still opens,
still replays, and still tells you exactly what produced it.**

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

If you want MCAP, converting `.zbr` to it is a small script over `frames.npz` —
and that script belongs in a bridge package, not in the core.

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
`metrics.json`. At typical per-cell masses (~0.2 g), `float16` still resolves
well under a milligram.

Lookup takes the nearest keyframe at or *before* the requested time, with no
interpolation — averaging two mass fields would invent dirt that never existed.

## Size

A 20-minute kidney run with the standard sensor suite is about 2–3 MB.
Roughly two thirds is the frame table and one third the dirt keyframes.

To make one smaller:

```python
zb.Simulation(..., dirt_keyframe_interval=30.0)  # fewer dirt snapshots
zb.Simulation(..., timestep=0.05)  # 20 Hz instead of 50
zb.Simulation(..., record=False)  # metrics only
```

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
