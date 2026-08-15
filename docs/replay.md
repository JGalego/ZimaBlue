# Replay

> The brief for this part of ZimaBlue was: *make it obvious what the robot is
> doing.* The failure mode to avoid is a debugging spreadsheet.

```bash
zimablue replay runs/kidney.zbr
```

## What you are looking at

```
┌────────────────────────────────────────────────────────────┐
│ kidney · tracked · autumn · seed 42                        │
│ 13:20    168.0 m travelled                                 │
│                                                            │
│        ╭──────────────────────────────────╮                │
│        │  water, shaded by depth          │                │
│        │  ░ pale wash = already cleaned   │                │
│        │  ▓ brown     = dirt still there  │                │
│        │  ● orange    = leaves and twigs  │                │
│        │  ━ cyan      = the robot's trail │                │
│        │  ⋯ dotted    = sonar beams       │                │
│        ╰──────────────────────────────────╯                │
├────────────────────────────────────────────────────────────┤
│ coverage 58%  dirt removed 34%  battery 88%  filter 49%    │
│ where it drove   what it cleaned                           │
└────────────────────────────────────────────────────────────┘
```

The two left-hand meters are the point of the whole project, which is why they
sit side by side with those subtitles. Watching **58% driven / 34% cleaned**
diverge over a run makes the argument better than any amount of prose.

Layer order is deliberate: the cleaned wash is drawn *under* the dirt, so a
patch the robot has driven over but failed to clean still shows as dirty. If
coverage were painted on top it would hide exactly the failure worth seeing.

## Controls

| Key | Action |
|---|---|
| `space` | pause / resume |
| `←` `→` | step one second (with `shift`, ten) |
| `↑` `↓` | faster / slower |
| `r` | restart |
| `s` | save a PNG of the current frame |
| `q` | quit |

Speeds: **0.25×, 0.5×, 1×, 2×, 5×, 10×, 25×**. Drag the slider to scrub;
scrubbing pauses so the frame stays where you put it.

A 30-minute run at 1× is thirty minutes of watching, so the player starts at
8×. 25× exists because sometimes you only want the shape of the path.

Speed changes the number of recorded frames consumed per rendered frame, not
the render rate — 10× is genuinely ten times faster rather than ten times
choppier.

## Headless

No display, or you want an artefact for an issue:

```bash
zimablue replay runs/kidney.zbr --gif  runs/kidney.gif
zimablue replay runs/kidney.zbr --summary runs/summary.png
zimablue replay runs/kidney.zbr --frames runs/stills/
```

The player detects a headless matplotlib backend and points you at `--gif`
rather than failing with a traceback.

`--summary` writes the four-panel post-run view: path driven, visit counts,
dirt at the start, dirt at the end. The bottom two share a colour scale, so
"before and after" is a fair comparison rather than a rescaled one.

## From Python

```python
from zimablue.recording import Recording
from zimablue.replay import ReplayPlayer, export_movie, export_summary

rec = Recording.load("runs/kidney.zbr")

ReplayPlayer(rec, speed=4.0).show()          # interactive
export_movie(rec, "out.gif", speed=90.0)     # GIF (or .mp4 with ffmpeg)
export_summary(rec, "out.png")               # four-panel summary
```

`ReplayRenderer` draws single frames if you want to build your own viewer:

```python
from zimablue.replay import ReplayRenderer

renderer = ReplayRenderer(rec, show_sensors=True, trail_seconds=120)
renderer.draw(rec.frame_index_at(300.0))     # the frame at t = 5 min
renderer.fig.savefig("t300.png")
```

## Design notes

**Everything comes from the recording.** The renderer rebuilds the pool and
robot from the embedded manifest, never from the live presets, so a recording
made against an older preset still renders correctly.

**Artists are created once.** `draw()` updates data rather than rebuilding the
figure, which is what makes scrubbing feel immediate.

**Coverage is exact at any frame, in any order.** A one-pass index of *when
each cell was first covered* is built when the renderer starts; coverage at
frame *n* is then a comparison against that array. Accumulating forward would
be cheaper but would show stale coverage after scrubbing backwards — a rewind
would display area the robot has not reached yet.

**Rasters are clipped to the pool outline** and their alpha is blurred. Without
either, the 10 cm cells read as a staircase fringe against the smooth coping,
which looks like a rendering bug rather than like resolution.

**Dirt keyframes are not interpolated.** The nearest keyframe at or before the
current time is used; averaging two mass fields would invent dirt that never
existed.

## matplotlib is optional

Replay is the only part of ZimaBlue that needs it, and it is imported lazily
inside the functions that draw:

```bash
pip install 'zimablue[viz]'
```

A headless batch of a thousand episodes never imports a plotting library, and
`import zimablue` never pulls in a GUI stack.

## Not yet

- Playing two runs side by side (comparing controllers is currently two windows)
- Plotting a state estimate against ground truth — the sensors already drift
  correctly, but nothing consumes them yet; see [`roadmap.md`](roadmap.md)
- A web viewer
