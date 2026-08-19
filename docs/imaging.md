# A pool from a photograph

```python
import zimablue as zb

pool = zb.pool_from_image("backyard.jpg", sample=(640, 410), width=8.4, depth=1.6)
```

Point a phone at a pool, get something you can run a cleaner in. Finding the
water, tracing its edge and turning pixels into metres can each go wrong in a
way the next step will not notice, so check the result before you trust it.

```python
traced = zb.trace_pool("backyard.jpg", sample=(640, 410), width=8.4)
print(traced.summary())
traced.overlay("check.png")  # look at this
pool = traced.pool(depth=1.6)
```

`overlay` draws the outline and the mask over your photo and labels the regions
it rejected. It matters as much as the polygon does.

## Scale is not optional

A single photograph does not contain its own scale. Nothing in an image
separates a 3 m plunge pool from a 30 m lap pool shot from ten times further
away — the projection is identical. So one of these is required and there is
no default:

| | |
|---|---|
| `metres_per_pixel=0.05` | You know the ground resolution. A satellite tile. |
| `width=8.4` | The pool's real width, across the widest part traced. |
| `reference=((x1, y1), (x2, y2), 3.0)` | Two pixels and the metres between them: a diving board, a paving slab, a door. |
| `corners=(((x1,y1), …, (x4,y4)), (6.0, 4.0))` | Four pixels forming a real rectangle, and its size. |

The first three assume the camera looked straight down.

## Photos taken from the poolside

They don't. A phone at head height foreshortens the far end of the pool, and
scaling such a photo as though it were a plan is wrong by a lot. Measured
against a known pool in the test suite:

| | traced area | error |
|---|---|---|
| `width=` alone, oblique photo | 39.7 m² | **−26.7%** |
| `corners=` on the same photo | 54.5 m² | **+0.6%** |
| truth | 54.1 m² | |

`corners` takes four image points that form a rectangle you can measure in the
real world — a patio slab, a towel laid square, the pool's own coping if it is
rectangular — plus that rectangle's size. From those it solves a homography and
undoes the perspective.

```python
traced = zb.trace_pool(
    "poolside.jpg",
    sample=(640, 410),
    corners=(((214, 588), (735, 588), (690, 402), (299, 402)), (6.0, 4.0)),
)
```

Give the four points in the order they trace the rectangle **as it appears in
the photo**, starting at the corner that should become the world origin, so the
second point is `width` metres away. If the result comes out mirrored, reverse
the order.

Only the traced outline is transformed, not the image — nothing is resampled,
so nothing is blurred by the correction.

## Which blue is the pool

Backgrounds contain sky, blue parasols, blue tiling and blue cars. With nothing
to go on, the largest blue-ish region wins. That is usually the pool and is
sometimes the sky: in a poolside photo the pool is foreshortened and the sky is
not, so the sky can genuinely be bigger.

**For a photograph, pass `sample=(x, y)`** — any pixel inside the water. Three
things change:

- The water's colour is read off the water instead of assumed, which matters
  because pools go green and tiles go navy.
- The region **containing that pixel** is taken rather than the biggest one.
- The region is then **grown outward until it meets an edge**.

The second point is what settles the sky, and no colour rule could: a pool and
a summer sky are close enough in hue to be indistinguishable. They are not
joined to each other, though, so connectivity decides it.

Every candidate is listed on the trace, and `region=1` picks by size instead if
you would rather.

### Why hue and not colour

Depth changes a pool's colour enormously and its hue barely at all — it is the
same water over the same plaster, with more or less of it in between. Measured
on one drone photo:

| | hue | saturation |
|---|---|---|
| deep end | 183° | 0.83 |
| outermost step | 181° | 0.23 |
| coping | 202° | 0.14 |
| deck | 208° | 0.10 |

In RGB the deep end and the top step are 100 apart out of 255, so any distance
wide enough to hold both swallows the coping on the way — and the outermost
step and the coping differ by **twelve**, which is less than two patches of
open water differ from each other. Hue separates them cleanly and saturation
confirms it, since wet plaster is vivid and dry stone is not. Brightness is
ignored, which is what keeps sunlit and shaded water in one region.

### Why grow at all

A pool does not end where a threshold ends. The last stretch before the coping
is very shallow water grading continuously out of the water beside it — no
step in it anywhere — while the coping arrives as a 36° jump. So the mask
creeps outward a bounded number of pixels, taking each one only if it looks
like the neighbours that already belong rather than like the distant sample,
and the hard edge stops it. Missing that rim cost about 20 cm all the way
round, which on a 25 m pool is 9% of the floor. `grow=0` turns it off.

### Or let a model find it

```python
from zimablue.segment import SamSegmenter

seg = SamSegmenter.load("mobile_sam_image_encoder.onnx", "sam_mask_decoder_multi.onnx")
traced = zb.trace_pool("pool.jpg", sample=(700, 800), width=25.0, segmenter=seg)
```

`segmenter` swaps out the colour rules and leaves everything else alone. Worth
it for a pool the hue rule cannot lock onto — black-bottomed, green, half in
shade — and for edges cluttered with things that happen to be pool-coloured.
It needs `pip install "zimablue[ml]"` and a checkpoint you fetch yourself; the
same `sample=(x, y)` becomes the prompt. See [machine learning](ml.md).

## Sun on the water

A specular highlight is white, so no rule that looks for blue will find it.
In the middle of the pool it leaves a hole; against the edge — where the sun
actually puts it — it takes a bite out of the outline. An early version lost
15% of the traced area to one.

Holes are filled. Bites are repaired by a separate rule: a blob brighter than
the water, not running off the frame, and whose own border is mostly water, is
water. That last condition is what stops a white sun lounger at the pool edge
going in too.

The brightness threshold is read off the water rather than fixed. A fixed one
cuts through the middle of the highlight's gradient, and the dim outer ring
then belongs to neither the water nor the blown-out core — it insulates one
from the other and nothing is ever absorbed. That took a while to see.

Colour matching discounts brightness for the same reason. Sunlight adds white,
sliding a pixel along the grey axis without changing its hue, so plain RGB
distance rejects the lit half of a pool. Splitting the difference into its grey
part and the rest, and thresholding only the rest, keeps sunlit and shaded
water in one region.

## Tidying the outline

A traced edge carries two kinds of roughness and they want different
treatments.

```python
traced = zb.trace_pool("pool.jpg", sample=(700, 800), width=25.0, smooth_edges=0.20)
```

`smooth_edges` is a radius in metres. It first collapses any wobble smaller
than that radius, so an edge that should be one straight run stops being thirty
segments a pixel out of line; then it rolls a ball of that radius along the
outline in both directions, which rounds the convex corners and the concave
ones by exactly that much. **Straight edges, curved vertices** — which is what
a real pool is, since its corners are struck with a radius rather than mitred
to a point. On the drone photo it took a 65-vertex outline to 18 and moved the
area by 0.6%.

`smooth` is the other one: a count of Fourier harmonics. It has no scale, so it
cannot spend its smoothing locally — on a pool with straight sides, sixteen
harmonics turned the sides into curves to pay for the corners. Keep it for a
pool that genuinely is a curve and came out ragged. Both are off by default.

## What it cannot do

**Depth.** A photograph of the surface says nothing about the floor beneath it.
`depth` defaults to a flat 1.5 m and is a guess you are making, not one the
library makes for you. If you know the ends, pass a `PlaneSlopeDepth`.

**Features.** Drains, returns and steps are not detected. Add them to the pool
afterwards if you need them.

**Anything cut off by the frame.** If the water runs off the edge of the photo,
the trace is of the part you photographed. It says so in `warnings`.

## From the shell

```bash
zimablue trace backyard.jpg --width 8.4 --sample 640,410 --check overlay.png --out pool.json
```

`--check` writes the overlay. Without it the command says so, because
segmenting a photo is a guess and that picture is how you catch a wrong one.
The CLI covers `--width` and `--mpp`; for `reference` and `corners` use the
Python API.

## Reading the trace

```python
traced.area  # m², after scaling
traced.boundary  # the Shapely polygon, metres, origin at (0, 0)
traced.regions  # every candidate found, largest first
traced.warnings  # cut off by the frame, suspiciously small, near-tie
traced.scale  # how pixels became metres, in words
traced.mask  # the boolean water mask, working resolution
traced.outline_px  # the outline in working-image pixels
```

Working resolution is capped at 720 px on the long side — a pool outline has no
detail that survives past it. Scale arguments are still given in the original
image's pixels.

## Accuracy

Against synthetic photographs of a known pool, including sky, blue decoys, sun
glare and noise, the traced area lands within a few percent. That measures the
geometry, not the segmentation: a real photo can be lit oddly, have a pool
that is partly shaded or partly out of frame, or contain something the rules
here have never seen. Which is the argument for looking at the overlay.
