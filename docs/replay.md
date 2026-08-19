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
│        │  ◌ amber     = where it thinks   │                │
│        │              it is (if estimated)│                │
│        ╰──────────────────────────────────╯                │
├────────────────────────────────────────────────────────────┤
│ coverage 58%  dirt removed 34%▏ battery 88%  filter 49%    │
│ where it drove   what it cleaned                           │
│              debris 27/59, 20 too big   controller ...     │
└────────────────────────────────────────────────────────────┘
```

The two left-hand meters are the point of the whole project, which is why they
sit side by side with those subtitles. Watching **58% driven / 34% cleaned**
diverge over a run makes the argument better than any amount of prose.

A close-up camera is limited by `speed / fps` — the simulated time between one
*displayed* frame and the next — rather than by playback speed on its own. The
robot covers its own length in about a second, so past two or three seconds a
frame the floor arrives already swept and dirt reads as popping into existence
instead of lifting. The dirt cam and chase cam therefore show a window of a
long run (`seconds=`) rather than the whole of it wound up to 440x. The
top-down view has no such limit and plays the entire clean at 260x: it shows
the whole pool at once, and nothing in it moves a body length between frames.

The amber tick on the dirt bar is where that bar physically cannot go past.
Some of the pool's debris is wider than the intake, so a run that lifted
everything it could still stops short of 100%, and without the tick it reads
as a run that gave up. The footer says how many items are in that category.

Layer order is deliberate: the cleaned wash is drawn *under* the dirt, so a
patch the robot has driven over but failed to clean still shows as dirty. If
coverage were painted on top it would hide exactly the failure worth seeing.

Dirt darkens towards near-black silt past the top of the scale. The scale is
a percentile of what the pool *started* with, and the returns sweep the floor
into heaps twenty times that, so without a second regime a heap is one flat
brown patch whose edge is the only part that appears to grow.

The dirt itself is keyframed every ten simulated seconds and blended between
keyframes for drawing. Without the blend a cell holds still for five hundred
rendered frames and then jumps, which reads as dirt appearing out of nowhere
rather than piling up -- most obvious where the returns sweep it into a heap at
the deep end. `Recording.dirt_at` still hands you the exact keyframe by
default; only the views ask for the blend, because it is an estimate of a
moment that was never recorded.

### The dots

Several unrelated things are drawn as dots, which is worth spelling out because
at 260x playback they all read as flicker.

**Rust-coloured flecks** are debris. Each is drawn as its own silhouette --
an ovate leaf with a stem, or a twig with a side shoot -- at true scale, in a
colour and orientation fixed to that item, so a drift of leaves does not read
as one flat brown smear. At the whole-pool zoom a 9 cm leaf is a few pixels
across, and that is the honest size of a 9 cm leaf in a 12 m pool; zoom in, or
look at the dirt cam, and it is recognisably a leaf.

They fade out as they are collected, and they drift while they are down:
the robot shoves anything too big for the intake out of the way, and over a
long run the typical survivor travels further than its own length. Debris is
keyframed every ten simulated seconds rather than per frame, so both the
positions and the fade are blended between keyframes -- otherwise a leaf
teleports a couple of metres and winks out between one frame and the next. A
typical autumn kidney run starts with 59 and finishes with 20 still down --
all twenty of them wider than the intake, which the HUD says in as many words.

**A red star at the robot** is a bump: it appears on any frame where a contact
switch is closed, and in a tight corner it can strobe for several seconds.
That strobe is real -- the collision counter is climbing at the same rate.

**Faint pale dots that never move** are the pool's hydraulic features: drains,
returns and skimmers, drawn at 40% alpha so they stay legible under the dirt
without competing with it.

### The amber ghost

If the controller publishes a pose estimate -- `systematic` does -- replay
draws it as an amber ring with a dashed line back to the true pose, and the HUD
prints the error in metres. Watching the two drift apart is the clearest
possible statement of what dead reckoning does over half an hour.

The estimate lives in the controller's own frame, anchored at the start pose,
so it is rotated into world coordinates for display. Any controller can join in
by growing a `telemetry()` method returning `est_x`, `est_y` and `est_heading`;
those become `ctl.*` channels in the recording.

## Chase cam

```bash
zimablue replay run.zbr --chase --gif out.gif      # animation
zimablue replay run.zbr --chase --summary out.png  # contact sheet
```

```python
from zimablue.replay import ChaseCam, ChaseCamConfig, export_chasecam, render_chasecam
```

A metre behind the robot and half a metre up. The top-down view shows the path
and a machine the size of a postage stamp; the dirt cam shows what the machine
sees and never the machine. This is the one that shows both — close enough to
see the brushes turning, far enough to see the clean stripe opening up behind.

Because the robot is now in front of the camera it has to be drawn, and what
gets drawn is the cleaner's own
[design](../src/zimablue/robot/design.py): the hull extruded to give it
thickness, the parts at their own heights, and a contact shadow so it sits on
the floor instead of floating over it. Two polygon fills and a brightness
multiply — there is no lighting model, and calling it one would be generous.

`ChaseCamConfig` adds `distance`, `lag` and `shadow` to the shared settings.
`lag` is the one that matters: bolt the camera rigidly to the robot and a turn
reads as the *pool* rotating, which is disorienting and wrong. Letting the
camera's heading chase the robot's — a fifth of a second by default — makes a
turn read as the robot swinging across frame with the camera easing after it.

The lag filter runs over the whole recording at construction rather than per
frame. It is a recurrence, so evaluating frame 900 alone would need the 899
before it anyway, and doing that per frame turns a scrub into an O(n²) wait.
The heading is unwrapped first, or every pass through ±π sends the camera the
long way round.

## Dirt cam

```bash
zimablue replay run.zbr --dirtcam --gif out.gif      # animation, map alongside
zimablue replay run.zbr --dirtcam --gif out.gif --no-map
zimablue replay run.zbr --dirtcam --summary out.png  # contact sheet
```

```python
from zimablue.replay import DirtCam, DirtCamConfig, export_dirtcam, render_dirtcam
```

The camera sits 18 cm off the floor behind the brush and looks forward. From
there the pool is not the calm blue sheet the top-down view shows; it is a silt
plain with leaves in it, and that is a fairer impression of what the machine is
driving through. Both views read the same dirt raster, so when they disagree
the disagreement is real: from above you see *where the robot went*, from the
bumper you see *what it left behind*.

The technique is inverse perspective mapping. Each output pixel is a ray cast
through the floor plane; where it lands, the dirt raster is sampled. The whole
frame is one vectorised NumPy expression over a precomputed grid of rays --
no mesh, no z-buffer, no engine.

Both underwater cameras are the same renderer,
[`FloorCamera`](../src/zimablue/replay/floorcam.py), differing in exactly one
thing: where the camera sits. That is not tidiness for its own sake -- two
cameras that disagreed about how far away a leaf was would be two different
simulators wearing one name.

Tunable through `DirtCamConfig`: frame size, `camera_height`, `pitch`, `fov`,
`far` and the grout `tile` pitch. The floor texture is locked to world
coordinates rather than to the screen, because without something sliding past
underneath, a colour field only changes shade and the view reads as a gradient
instead of as motion.

The projection cuts corners in ways worth knowing. It treats the floor as flat
even where the depth model slopes, which costs a little foreshortening accuracy
at the far edge of frame and saves a ray–surface intersection per pixel. And
anything off the navigable floor is shaded as wall rather than raycast against
real wall geometry, so the pool edge is a darkened region at floor resolution,
not a modelled surface.

## The 3D view

```bash
zimablue replay run.zbr --3d --gif out.gif      # orbiting animation
zimablue replay run.zbr --3d --summary out.png  # contact sheet
```

```python
from zimablue.replay import export_3d_movie, export_3d_frames, render_3d
```

The floor is a surface sampled from the pool's depth model, the walls are
extruded from its boundary, and the robot box sits at the local floor depth --
so a sloped pool renders as a real basin and the cleaner is metres lower at the
deep end. Floor colour is remaining dirt, as in the 2D view. The camera orbits
about 50 degrees across the run, because parallax is what makes a rendered
scene read as solid.

Vertical scale is exaggerated roughly 3.6x. A 12 m pool 2 m deep is a pancake
at true scale, and depth is the entire reason for this view -- so do not read a
gradient off the picture.

**It renders in 3D; it does not simulate in 3D.** Motion comes from
`Fast2DBackend` either way. A 3D *backend* is designed and not built; see
[`architecture.md`](architecture.md#3d-backend-intended-design). The geometry
comes from the recording's embedded pool config, so the 3D view works on any
`.zbr`, including ones written before the renderer existed.

Interactive 3D is not wired up -- the view renders to a file. Rebuilding the
floor surface every frame is far too slow for scrubbing, and solving that means
caching the mesh and updating only its colours, which is worth doing when
somebody wants it.

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
8×. Press `↓` to get down to 1× and the run plays at the speed it happened.
25× exists because sometimes you only want the shape of the path.

Speed changes how many recorded frames are consumed per rendered frame, so 10×
is ten times faster and not ten times choppier. The position is tracked as a
float and rounded only when a frame is drawn, which is what lets 0.25× be four
times slower than real time instead of bottoming out at one frame per tick.

The clock is the display's, not the wall's: if a frame takes longer than 33 ms
to draw, playback slips behind. Exports do not have that problem, so
`export_movie(rec, "out.gif", speed=1.0)` gives you a file exactly as long as
the run — and, for half an hour of pool cleaning, exactly as long to watch.

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

## In a browser

```bash
zimablue replay runs/kidney.zbr --html runs/kidney.html
```

writes one self-contained HTML file: scrubber, playback speed, layer toggles
for dirt, trail and debris, and a live coverage readout. It needs no
matplotlib, no server and no network — the file opens from disk, which makes
it the replay you attach to an issue or mail to someone whose machine has
nothing installed.

Several runs go side by side on a shared clock from Python:

```python
from zimablue.replay import export_web_player

export_web_player({"bsa": one_recording, "binn": another}, "duel.html")
```

The dirt layer blends between keyframes exactly as the GIF cameras do, and it
is quantised onto a coarser grid to keep the file small — measure on the
recording, not on the page.

## From Python

```python
from zimablue.recording import Recording
from zimablue.replay import ReplayPlayer, export_movie, export_summary

rec = Recording.load("runs/kidney.zbr")

ReplayPlayer(rec, speed=4.0).show()  # interactive
export_movie(rec, "out.gif", speed=90.0)  # GIF (or .mp4 with ffmpeg)
export_summary(rec, "out.png")  # four-panel summary
```

`ReplayRenderer` draws single frames if you want to build your own viewer:

```python
from zimablue.replay import ReplayRenderer

renderer = ReplayRenderer(rec, show_sensors=True, trail_seconds=120)
renderer.draw(rec.frame_index_at(300.0))  # the frame at t = 5 min
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

**Rasters are clipped to the pool outline**, so the 10 cm cells do not read as
a staircase fringe against the smooth coping.

**The coverage overlay is deliberately not smoothed.** It used to be: a 3x3 box
blur on the alpha channel plus bilinear interpolation, both there to soften the
cell edges. Measured against the visit grid, that combination rendered 28% more
area as covered than the robot had driven over -- 9.5 m2 of coverage that never
happened in a 59 m2 pool, and every cleaned lane drawn at 1.4x its true width.
A viewer would have concluded the dirt was vanishing from places the machine
never reached, which is precisely the error this project exists to catch. Both
overlays now use nearest-neighbour sampling and no blur. The picture is
blockier and it is true.

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

## Turning a pool over in a notebook

```python
import zimablue as zb

zb.preview("kidney")  # drag to rotate, scroll to zoom
zb.preview(result)  # ...tinted with the dirt left behind, path drawn on it
```

This one does not go through matplotlib at all. The pool's geometry is built in
Python, shipped to the page as JSON, and projected by a small canvas renderer,
so dragging costs a matrix multiply over a few thousand vertices rather than a
round trip to the kernel. There is no `ipywidgets`, no `ipympl` and no widget
state, which means it works in JupyterLab, classic Notebook, VS Code and Colab
alike -- and keeps working in an exported HTML file after the kernel is gone.
`PoolPreview.save("pool.html")` writes that file directly.

The renderer is a painter's algorithm with flat shading and no z-buffer. For a
basin that is enough; faces that pass through each other would be a bug in the
mesh rather than a limit of the sort. Vertical scale is exaggerated 2.6x, and
the page says so.

## Not yet

- Playing two runs side by side (comparing controllers is currently two windows)
- The dirt cam inside the interactive player, as a live side panel
