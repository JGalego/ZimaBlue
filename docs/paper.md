# Coverage is not cleanliness

**Measuring what pool-cleaning planners actually remove.**
João Galego · [ZimaBlue](https://github.com/JGalego/ZimaBlue) · 2026

> Coverage path planning optimises the fraction of a floor a robot passes
> over, and the cleaning literature reports it as though it were the point of
> cleaning. It is not. On a simulated pool with a physical dirt model, the
> ranking of planners by floor covered and their ranking by dirt removed
> disagree — reliably, and by enough to change which machine you would buy.
> We make the claim measurable in a deliberately small, deterministic
> simulator; bound what any policy could have collected in the time, so every
> score becomes a regret; and show that a deployable controller reading a
> turbidity probe collects more dirt while covering less floor than the
> random baseline that out-covers most of the published planners.

## 1. The claim

A cleaning robot is bought to remove dirt and evaluated, almost everywhere,
on the area it drives over. The substitution is convenient — area is easy to
measure and dirt is not — and it is harmless exactly when dirt is uniform,
loose, and removed in one pass. Real dirt is none of that: it concentrates
where the water carries it, it adheres, and the adhered part needs repeated
passes. So a route that maximises fresh floor per minute and a route that
maximises grams per minute are different routes, and a benchmark that only
scores the first will steer the field toward the wrong optimum.

This writeup does not argue the point; it measures it. Everything below is
reproducible from the repository at the commit this document ships in:

```bash
pip install -e ".[dev]"
zimablue bench --jobs 4          # the frozen suite
zimablue compare --pool rectangular --pool kidney --pool l_shaped --minutes 15 --jobs 4
```

## 2. The instrument

ZimaBlue simulates a differential-drive pool cleaner in a 2D pool with a
physical dirt model, and records every run into a replayable, bit-identical
format. The parts that matter for the claim:

- **Dirt is mass, not paint.** A per-type raster in grams per cell (sediment,
  sand, algae, biofilm) plus discrete debris items with a size the intake may
  or may not admit. Removal is exponential in time under the head, scaled by
  suction, adhesion, and brush agitation — so adhered dirt genuinely needs
  passes, and one drive-by genuinely does not finish a cell.
- **The water works too.** Fine dirt rides the return-jet circulation,
  buoyant debris drifts to the skimmer, and the robot's own wake resuspends
  what it passes over. The pool is an environment, not a scoreboard.
- **Sensors lie by construction.** Encoders read wheel motion, not ground
  motion; the gyro has a turn-on bias; everything passes one
  noise/latency/dropout pipeline. Controllers see readings, never state.
  Localisation error is therefore a property of the run, not an ablation.
- **Determinism.** Same version, same platform, same scenario, same seed —
  bit-identical recording. Every number below is a lookup, not a hope.

None of this claims hydrodynamic fidelity. The claim is weaker and more
useful: the dirt model has the *structure* that makes coverage and
cleanliness distinct quantities, and the instrument measures both.

## 3. Scores need a denominator: the collectable bound

"Removed 31% of the dirt" says little on its own — 31% of what was reachable,
or of what was there? The two differ whenever the run is shorter than the
job, which is always.

We bound what any policy could have collected. In `T` seconds a head of
swath `w` moving at speed `v` passes over at most `v·T·w` of floor, and only
mass under the head is ever collectable. Grant a fictitious cleaner
everything a real one must earn — it teleports between the richest cells,
never turns, never revisits, lifts a cell's whole mass in one pass, and
swallows every item its intake can physically admit — and its haul is the sum
of the heaviest cells that fit in the swept-area budget, capped by the
collectable total. Every real run collects less, whatever the planner. This
is a relaxation of the initial field, not an optimal policy: it is loose the
safe way, and the myopic `dirt_oracle` in the package deliberately does not
play this role, because greedy bounds nothing.

Dividing by the bound gives the **"of possible"** column: the share of the
physically reachable dirt a planner actually got. The distance from 100% is
regret, and it decomposes into exactly the things a planner exists to manage
— travel between the grams, revisits, and not knowing where the dirt is.

## 4. Results

Fifteen simulated minutes per run, three pools (rectangular, kidney,
L-shaped), autumn dirt on a tracked cleaner, medians across pools; offline
planners followed on dead reckoning with wall-touch relocalisation. The full
table lives in [planners.md](planners.md) and regenerates with one command;
the findings that carry the thesis:

The full numbers are in the repository ([planners.md](planners.md)); the
findings:

**The two rankings disagree at the top.** The coverage winner, `binn`
(68.9%), sits mid-table on dirt removed (24.1%). The dirt winner, `ppcpp`
(32.7%), covers a point and a half less floor than `binn`. A buyer ranking
by the published metric and a buyer ranking by the job's actual objective
walk out with different machines.

**A random walk out-covers most of the field.** `random_bounce` — drive
straight, turn randomly on contact — covers 65.0% of the floor, more than
every planner in the table except the field methods that carry an EKF, an
evidence map, and a decision rule to beat it. Any method below that line is
not paying for its own machinery, completeness proofs included.

**Sensing dirt beats counting floor, with less machinery than either.**
`dirt_seeker` reads the turbidity probe and layers three habits: spiral over
a reading that spikes above the running ambient, remember the find in the
estimated frame, wander when the trail goes cold. It removes 28.3% of the
dirt against the bounce's 26.0% while covering three and a half points
*less* floor — the inversion the thesis predicts, produced by a controller
every claim of which is deployable. Where dirt hugs the walls its spirals
fight the geometry and it gives a couple of points back; edge dirt wants an
edge follower, and it does not have one.

**Nobody reaches 35% of what was physically possible.** Against the
collectable bound, the best planner in the table (`ppcpp`, 34%) leaves two
thirds of the reachable dirt in the pool, and the worst
(`brick_and_mortar`, 13%) leaves almost nine tenths. The regret is not
noise: it decomposes into travel between the grams, revisits adhered dirt
demands, and ignorance of where the dirt is — the first two are the price
of physics, the third is the one sensing can buy back.

**Localisation is worth more than planning, and touches are localisation.**
Followed on dead reckoning, offline plans drift by tens of metres of
estimate error over a fifteen-minute run. Treating each wall touch as a
measurement — the hull sits at wall distance along the believed wall's
normal — cuts mean estimate error from 79.9 m to 3.4 m on the rectangle and
39.7 m to 1.9 m on the oval, with one gate doing the safety work: the bumper
that fired has to agree with where the believed wall lies, because in an
inside corner correcting against the wrong wall turns a good estimate into a
confident bad one.

**Efficiency finally has a unit.** Grams captured per watt-hour separates
planners whose coverage numbers sit together: `frontier` captures 22.9 g/Wh
and `brick_and_mortar` 7.2 g/Wh from energy budgets that differ by four
percent. "Efficient", for a cleaner, was never area per joule.

## 5. What this does not show

Simulation only, one robot class, planar dynamics, one seed per pool with
medians across pools. The dirt parameters are literature-shaped guesses, not
measurements — the repository's own roadmap calls this out, and the hardware
module exists precisely so real logs can start replacing them. The bound is
computed on the initial field, so mass drifting into the swept set during a
run can in principle exceed it; at the model's drift rates the effect is far
inside the bound's slack. And the strongest single-number result — the
inversion between the coverage ranking and the dirt ranking — is robust to
seeds in our runs, but the mid-table orderings inside a few points are not,
and we do not read them.

## 6. Reproducing

Every number above comes from `zimablue compare` or `zimablue bench` on the
repository at this commit. `zb-bench-v1` freezes entries, pools, seeds and
duration; its JSON header records the package, Python, and numpy versions.
Two differing results on the same platform mean a real change, not noise —
that is the point of shipping the benchmark next to the claim.

```bibtex
@techreport{galego2026coverage,
  author = {Galego, Jo{\~a}o},
  title  = {Coverage is not cleanliness: measuring what pool-cleaning
            planners actually remove},
  year   = {2026},
  url    = {https://github.com/JGalego/ZimaBlue/blob/main/docs/paper.md}
}
```
