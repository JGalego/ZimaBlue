# Machine learning

Two places a model earns its keep here, and neither is in the core install.

| | |
|---|---|
| `pip install "zimablue[ml]"` | Segment a pool out of a photograph with SAM. onnxruntime, no torch. |
| `pip install "zimablue[rl]"` | Train a controller. A Gymnasium env, and whichever algorithm you like. |

---

# Segmenting a pool with SAM

```python
from zimablue.segment import SamSegmenter

seg = SamSegmenter.load(
    "models/mobile_sam_image_encoder.onnx", "models/sam_mask_decoder_multi.onnx"
)
traced = zb.trace_pool("pool.jpg", sample=(700, 800), width=25.0, segmenter=seg)
```

`segmenter` replaces the colour rules and nothing else. Region selection, hole
filling, notch closing, edge smoothing, scaling and the overlay are all
unchanged, so a SAM trace and a colour trace are the same object and can be
compared directly.

## Why both, rather than one

The colour rules know what water looks like and not where it stops. They match
a hue, and then have to be walked outward pixel by pixel to reach the coping,
because the last stretch of a pool is very shallow water grading continuously
into the water beside it.

SAM is the other way round. Prompted with a point it is excellent at the edge
of a thing and has no idea which thing you meant. On the drone photo below it
returned the whole sunlit terrace and rated it 0.997.

So SAM proposes and the colour rule disposes: every candidate mask is scored
against the pixels the colour rule believes are water, and the best agreement
wins. Predicted IoU does not enter into it, because it measures how cleanly a
mask cuts out *a* thing — and the terrace is a perfectly clean thing.

### Recall counts double

The judge is biased and the bias has a direction. The colour rule under-detects
shallow water, so it cannot see steps. Score the candidates by F1 and the
precision of stopping short is worth more than the recall of going on, and
because SAM's candidates are nested the mistake is always the same one: the
shallow end gets clipped off the pool.

Weighting recall twice (F2) says what is meant. A candidate that misses water
the colour rule is sure about is worse than one that includes water the colour
rule was never going to find. On the drone photo that is the difference between
a trace that stops at the steps and one that includes them.

If the frame has nothing blue in it at all — a black-bottomed pool, which is
one of the reasons to be here — the colour rule cannot referee, and the ranking
falls back to predicted IoU among the candidates that contain the prompt.
`segmenter.ranked_by` says which rule was used and `segmenter.candidates` shows
the scores.

## Does it help

On a 2000 × 1500 drone photo of a hotel pool, calibrated the same way:

| | traced area | vertices | notes |
|---|---|---|---|
| colour rules | 156.2 m² | 17 | needs `close_gaps=0.6` for the underwater lamps |
| SAM | 162.0 m² | 16 | no gap closing needed |

Two methods that share no code land 3.7% apart. The difference comes from the
shallow stepped end, which SAM includes more of.

Each lamp pokes a slot into the
colour mask, and the fix was a morphological closing at a stated physical
scale. SAM does not need it: an underwater light is part of the pool, and it
knows that in a way no threshold does.

SAM helps with:

- Pools the hue rule cannot lock onto — black-bottomed, green, deep in shade.
- Photos where half the water is in sun and half is not.
- Anything with clutter at the water's edge that happens to be pool-coloured.

Where it changes nothing: **scale**. A neural mask has no metres in it either.
`width`, `metres_per_pixel`, `reference` or `corners` is still required, and an
oblique photo still needs `corners` to undo the perspective. See
[imaging](imaging.md).

## Weights

None are bundled and nothing is downloaded at import. MobileSAM is ~45 MB
across two files and runs in a second or two on a CPU:

```bash
pip install "zimablue[ml]"
huggingface-cli download Acly/MobileSAM \
    mobile_sam_image_encoder.onnx sam_mask_decoder_multi.onnx --local-dir models/
```

Any export following the reference SAM ONNX scripts works, MobileSAM and
SAM-ViT alike; the decoder must declare `image_embeddings`, `point_coords`,
`point_labels`, `mask_input`, `has_mask_input` and `orig_im_size`, and you are
told plainly if it does not. Prefer the *multi*-mask decoder — a single-mask
one gives the chooser one option, which is no choice at all.

Two encoder conventions are in the wild: one takes `HxWx3` and normalises and
pads inside the graph, the other takes a finished `1x3x1024x1024` tensor. Both
are handled. Neither *resizes*, which is worth knowing if you ever write this
yourself — the padding is a plain pad, so a 2000 px photo handed straight to
the first kind is silently cropped to its top-left 1024², and what comes back
is a confident segmentation of the wrong corner of the picture.

From the shell, via the example:

```bash
python examples/pool_from_photo.py --photo pool.jpg --width 25 --sample 700,800 \
    --sam models/mobile_sam_image_encoder.onnx,models/sam_mask_decoder_multi.onnx
```

## What is not here

Detection. SAM segments what you point at; it does not know a ladder from a
skimmer, so `traced.pool()` still comes back with no features. A detector
trained on pool furniture would be the thing that changes that, and it would
need a dataset that does not currently exist.

---

# Training a controller

```python
import gymnasium as gym

env = gym.make("zimablue.rl:ZimaBlue-v0", pool="kidney", dirt="autumn", minutes=10)
```

The module prefix makes Gymnasium import `zimablue.rl` on the way, which is
where the id gets registered -- no import line of your own needed.

or directly:

```python
from zimablue.rl import PoolCleaningEnv

env = PoolCleaningEnv(pool="kidney", dirt="autumn", minutes=10, reward="dirt")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Action**: two track speeds as fractions of the motor limit, in `[-1, 1]`. The
brush and pump stay on — switching the brush off is a way to score zero that an
agent finds in ten minutes and learns nothing from.

**Observation**: what a controller sees. Every sensor channel plus a freshness
flag, the battery, the filter load, and how much of the episode is gone. No
pose, no map, no dirt field. The bounds come from each sensor's own saturation
limits rather than an invented normalisation, so wrapping in
`RescaleObservation` gives you honest numbers.

**Info**: coverage, dirt removed, grams collected, distance, battery.

### Splitting estimation off from planning

A policy fed raw sensors has to solve both at once, in a pool with no absolute
reference, which means a recurrent policy and a long run. Handing it a pose
estimate instead leaves only the planner to learn:

```python
from zimablue.rl import EstimatedPose, PoolCleaningEnv

env = PoolCleaningEnv(extra_observations=EstimatedPose())
```

That adds seven channels — position and heading in the estimator's own drifting
frame, the heading split into sine and cosine so it is continuous across the
wrap, the filter's own uncertainty, and how much of the map is explored and
swept. They come from the same EKF and occupancy map the `systematic`
controller uses, and none of them reads ground truth.

It is also the shape of a fix for the localisation result in the README, where
a *better* position estimate halves coverage. The estimator is fine; the lane
planner is brittle. Replacing the planner is the interesting move.

Anything with `channels`, `bounds`, `reset` and `__call__` works — see
`zimablue.rl.observations.ExtraObservations`. It runs every physics tick rather
than every decision, because a filter fed one sample in ten is a different
filter. Pass the same object to `PolicyController` when you deploy, or the
policy gets an input it never trained on.

## Deciding slower than the physics

The simulation integrates at 50 Hz. Asking a policy for a fresh command fifty
times a second means 90,000 decisions in half an hour, nearly all identical to
the one before, over a credit-assignment horizon no algorithm will thank you
for. Real cleaner firmware does not do that either.

`control_hz` decimates. At the default 5 Hz one action is held for ten physics
ticks, which makes the episode ten times shorter without touching the dynamics.
It does not make it ten times faster — the physics still runs at 50 Hz — it
makes the problem ten times smaller, which is the part that matters.

## What to reward

This is the experiment, not a default to skip past.

| | |
|---|---|
| `reward="dirt"` | Grams collected during the step. The default. |
| `reward="coverage"` | Square metres of floor newly reached. |

They disagree, and the whole library exists because they disagree. Train on
coverage and you will get the oracle's failure mode: a policy that drives a
beautiful path over dirt it never picks up. The
[README's controller table](../README.md#measure-it) is the same result
arrived at by hand — best coverage is worst cleaning.

Both are paid per decision on what changed during it, so the episode return is
exactly the final number: summed dirt reward equals grams collected to the last
significant figure, and summed coverage reward equals the floor reached minus
the swath the robot was dropped onto.

## Rewards beyond the two built-ins

`reward="dirt"` and `reward="coverage"` are the two ends of the project's
argument. Anything in between is a callable taking the `info` dict from before
and after the decision:

```python
env = PoolCleaningEnv(
    reward=lambda prev, now: (
        (now["dirt_collected"] - prev["dirt_collected"]) - 40.0 * (prev["battery"] - now["battery"])
    )  # grams, net of the energy bill
)
```

`info` carries `coverage`, `dirt_removed`, `dirt_collected`, `distance`,
`battery` and `time`, so shaping terms stay one lambda rather than a subclass.

## Watching it train

`render_mode="rgb_array"` draws the pool from the simulation grids with numpy
alone -- dirt darkening the water, visited floor lifted a shade, the robot on
top -- which is what `RecordVideo` needs:

```python
env = gym.make("zimablue.rl:ZimaBlue-v0", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda e: e % 50 == 0)
```

For anything closer to the real thing, construct with `record=True` and call
`env.save("episode.zbr")` -- the replay cameras beat a training video.

## Throughput

About **24× real time** and **120 agent decisions per second** on one core, for
a ten-minute kidney-pool episode. That is roughly 430k steps an hour per
worker, so eight workers under `SubprocVecEnv` put a 1–10M step PPO run in the
1–10 hour range on a laptop. No GPU is involved at any point, which is the
argument for doing this here rather than in Isaac.

Vector envs need `zimablue.rl` imported in each subprocess; `register_envs()`
is public and idempotent for exactly that.

## Getting the policy back out

A policy that has only ever been evaluated in the env it was trained in is how
one comes to look better than it is. `PolicyController` puts it back on the
ordinary interface:

```python
from stable_baselines3 import PPO
from zimablue.rl import PolicyController

model = PPO.load("cleaner")
controller = PolicyController(lambda obs: model.predict(obs, deterministic=True)[0])

result = zb.Simulation(pool="kidney", controller=controller, seed=7).run(minutes=30)
```

Same metrics, same recordings, same `zimablue batch` across held-out seeds,
same replay to watch it fail. `control_hz` must match training: get it wrong
and the policy still runs, it just acts ten times more often than it learned to
and drives like it.

Recording an episode from the env works too, which is worth the memory —
a coverage number tells you a policy is bad, thirty seconds of the replay tells
you why:

```python
env = PoolCleaningEnv(record=True)
...
env.save("runs/episode.zbr")  # zimablue replay runs/episode.zbr
```

## Try the cheap thing first

```bash
python examples/rl_env.py --minutes 10
```

runs random, straight-ahead and `baseline_coverage` through the env on one
seed and prints what each returns. On a ten-minute kidney pool the baseline
collects several times what a random policy does, and that is the number to
beat — worth knowing before spending a GPU-day discovering it.

What is likely to beat a from-scratch policy, for a thousandth of the
compute:

**Tune the baseline.**

```bash
python examples/tune_controller.py --minutes 10
```

Five of `BaselineCoverage`'s parameters, searched with a (1+1) evolution
strategy over a batch of seeds. On a four-minute two-seed budget — a couple of
CPU-minutes — it took dirt removed from 13.5% to 15.6% in ten iterations.
Search for coverage instead and it finds a different setting, which is the same
disagreement the reward section is about.

**Imitate an oracle** — but not `lawnmower_oracle`. It is a good driver and a
poor cleaner, so imitating it teaches the wrong lesson. `dirt_oracle` reads
the dirt field and drives at whatever is dirtiest, which is the behaviour
worth copying.

Up to a point, and the point is interesting. Kidney pool, autumn dirt, seed 42,
dirt removed:

| minutes | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| `dirt_oracle` | **37.0%** | **49.7%** | **55.9%** | **58.5%** | **59.0%** |
| `baseline_coverage` | 16.1% | 24.0% | 28.0% | 38.6% | 39.2% |
| `dirt_oracle` coverage | 16.5% | — | — | — | 23.9% |

Greedy is more than twice as good at ten minutes and still twenty points ahead
at thirty, and the third row is why that is not a recommendation: it removes
three fifths of the dirt having driven over a quarter of the pool. It works
the richest patch until the easy mass is gone and the returns flatten, then
crosses to the next one, and what it leaves behind is clean in blotches. So
`dirt_oracle` is an upper bound on nothing — it is the best *myopic* policy,
which makes it a good teacher for what to head towards and a bad one to copy
wholesale. Both oracles need `Simulation(expose_truth=True)` and neither is
deployable.

That two oracles built from the same ground truth disagree this completely —
one covers 88% and lifts a third of the dirt, the other covers 24% and lifts
three fifths — is the coverage-versus-cleanliness result one level down: a
reward, an oracle and an episode length have to be chosen together or the
answer means nothing.

The genuinely interesting RL problem here is not driving. It is that the pool
has no absolute reference, so a policy from raw sensors has to be recurrent.
Feeding it the `systematic` controller's EKF estimate and occupancy map instead
splits the job — classical estimation, learned planning — and that is also the
shape of the fix for the localisation result in the README, where a *better*
position estimate currently halves coverage because the lane planner is
brittle.

## What this cannot tell you

A 2D kinematic backend with no buoyancy, no cable drag and no wall climbing
will not transfer to hardware. This is a fair testbed for comparing algorithms
and a bad one for shipping a policy. The [roadmap](roadmap.md) says the same
thing about the 3D backend that would change it.

Note which half of that is about the *interface* and which about the *physics*,
because only one of them is a problem. `PolicyController` already runs a trained
policy through `zimablue.hardware` on a real robot with no changes — the
plumbing is done, and [hardware.md](hardware.md) is the whole of it. What will
not transfer is the policy, because it learned a drivetrain whose slip is a
constant somebody guessed.
