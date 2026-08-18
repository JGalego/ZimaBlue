# Fleets

A second robot is not a second run. The two share a dirt field, so whoever
arrives later finds the patch already clean; they cannot occupy the same water;
and anything they know about each other, one of them had to say out loud.

```python
import zimablue as zb

fleet = zb.Fleet(pool="kidney", robots=3, controllers="auction")
result = fleet.run(minutes=20)
print(result.summary())
```

```
  robots            3
  team coverage       84.4 %
  dirt removed        47.2 %
  overlap             55.0 %   (floor two or more robots both did)
  speedup             1.70 x   (against the best single member)
  balance             0.98     (shortest run / longest)
  distance           252.3 m  (all robots)
  encounters            64     (robot-on-robot)
  per robot         r0 50%  r1 49%  r2 48%
```

Coverage is the least interesting number there. **Speedup** — team coverage
over the best single member's — has a ceiling equal to the robot count, and how
far short it falls is the cost of sharing a pool. **Overlap** is the floor more
than one robot did. **Balance** catches the failure coverage hides completely:
one robot doing the work while another finishes early and parks.

## How it is put together

`Fleet` composes the single-robot machinery rather than replacing it. Each
robot gets its own backend, its own sensors and its own controller; all of them
are reset against **one** `World`, which is what makes the dirt shared. Nothing
in the single-robot path changed, and `Simulation` is still the right thing for
one robot.

Three things are new.

**They collide.** Every backend is told where the others are before each tick,
as discs. The collision resolver pushes them apart and the sonar sees them, so
`Contact.is_robot` distinguishes bumping a team-mate from bumping the wall — a
fleet logging a hundred of the first is telling you something a lumped
collision count cannot.

**They talk, badly.** The `Blackboard` is a radio, not a god view. A robot
publishes *its own estimate* of where it is and what it has covered, so a fleet
inherits every member's localisation error and then has to coordinate through
it. `comms_range` limits who hears whom; the default is unlimited only because
that is what almost every published algorithm assumes.

**Sense all, decide all, then move all.** Not sense-decide-move per robot in
turn, which would let robot 1 react to robot 0's *new* position inside the same
tick — a turn-order advantage that grows with the fleet and makes the result
depend on the order the robots happen to be listed in.

### The recording

A fleet writes one `.zbr` with every robot's channels prefixed — `r0.x`,
`r1.heading` — and robot 0's channels *also* written flat as `x`, `heading`.
The duplication is deliberate. Every tool in the package reads the flat names,
and aliasing them to the first robot means the replay window, the dirt cam, the
dynamics module and the planner comparison all open a fleet recording and
follow one member, instead of refusing to open it.

## Dividing the pool

Most multi-robot coverage in the literature is one idea: cut the area into as
many pieces as there are robots, then run a single-robot planner in each. Five
ways to cut, each failing differently:

| | | fairness on a kidney, 3 robots |
|---|---|---|
| `voronoi` | nearest robot wins, in a straight line | 0.49 |
| `geodesic` | nearest robot wins, through the water | 0.55 |
| `strips` | equal-area bands across the long axis | 0.98 |
| `darp` | iterate until the shares are equal | 0.95 |
| `forest` | split a spanning tree into balanced subtrees | 0.93 |

Fairness is the smallest share over the largest. `voronoi` hands the robot in
the waist of a kidney a third of what the robot in a lobe gets, and it will
happily assign a cell on the far side of a wall because the crow flies across
the concavity. `geodesic` fixes the wall and not the fairness. `darp`
(Kapoutsis et al., 2017) iterates a multiplier per robot until both are right,
and reports whether it converged. `forest` cannot produce a disconnected share
because it cuts a tree, not a map.

Every territory becomes a small `Pool`, so all eight offline planners work
inside one unchanged:

```python
from zimablue.planners import partitioned

zb.Fleet(pool="kidney", robots=3,
         controllers=partitioned("darp", "sweep_optimal")).run(minutes=20)
```

### A partition is only as good as the localisation that drives it

The same DARP partition, the same plans, the same pool, eight minutes, three
robots — followed from the true pose and from dead reckoning:

| | team coverage | overlap | speedup |
|---|---|---|---|
| `localisation="truth"` | 62.9% | 0.3% | **2.87x** |
| `localisation="odometry"` | 73.5% | 43.4% | 1.88x |

On truth the division of labour is nearly perfect: 2.87 of a possible 3.0, and
three tenths of one per cent of the floor done twice. On dead reckoning the
partition survives in the plan and has evaporated in the execution — the robots
drift into each other's territories and 43% of the pool gets done twice.

And coverage goes **up**, because the drifting robots wander into the margin
along the wall that the plans never covered. The same effect the `systematic`
controller shows on one robot, at fleet scale: better localisation does not
help until the planner can spend it.

## Coordinating without dividing

| | |
|---|---|
| `mstc` | one spanning-tree circuit, cut into arcs by robot position |
| `mstc_backtracking` | the same, and a finished robot takes over a tail |
| `auction` | bid for the next cell; the cheapest robot gets it |
| `binn_swarm` | the neural field, with team-mates as inhibition |
| `smc_swarm` | one ergodic time-average, shared across the fleet |

There is a sixth that needed no code. Every online planner becomes cooperative
when the fleet hands it a blackboard: it publishes what it has covered and
skips what its team-mates say they have done. `Fleet(..., share=True)` is that,
it is the default, and it is the baseline the five above have to beat.

### What the measurements said

Three robots, kidney pool, eight minutes, one seed. Eleven dimensions, the same
harness as the single-robot comparison with the three that only mean something
for a team -- speedup, overlap, balance -- in place of the ones that do not.

<div align="center">
<img src="assets/fleet-matrix.png" alt="Seventeen multi-robot methods scored on eleven dimensions" width="900">
</div>

```
                             coverage       dirt    speedup    overlap    balance  worst gap efficiency    turning    to half      bumps     energy
---------------------------------------------------------------------------------------------------------------------------------------------------
bsa                             66.3%      29.6%     *2.09x        35%       0.91      7.3m2       0.53      219.9      never    5.3/min     26.2Wh
frontier                        75.2%      38.9%      1.77x        50%       0.84      4.5m2       0.41       86.1      never    7.8/min     26.6Wh
binn                            80.8%      39.9%      1.68x        55%       0.91      2.4m2       0.41       66.6      never    9.1/min     26.6Wh
epsilon_star                    80.5%      37.9%      1.63x        54%       0.95      3.2m2       0.42       68.6      never   11.1/min     26.6Wh
ppcpp                           62.1%      47.4%      1.67x        48%       0.87     15.2m2       0.36       99.0      never   15.0/min     26.5Wh
smc                             79.8%      48.7%      1.58x        59%       0.85      1.4m2       0.40       66.4       480s   16.9/min     26.7Wh
auction                        *84.4%      47.2%      1.70x        55%      *0.98     *1.1m2       0.43       66.4      never    8.0/min     26.6Wh
binn_swarm                      77.6%      44.0%      1.62x        61%       0.88      5.2m2       0.38       57.2      *472s   14.3/min     26.7Wh
smc_swarm                       72.0%      34.9%      1.83x        37%       0.97      1.8m2       0.48      160.1      never    5.9/min     26.5Wh
mstc                            68.8%      44.4%      1.74x        37%       0.51      2.9m2       0.46       74.1      never   19.0/min     22.6Wh
mstc_nobt                       43.0%      31.5%      1.18x        *8%       0.08     15.6m2      *0.67      111.3      never   66.3/min    *11.6Wh
voronoi+sweep_optimal           78.2%     *52.6%      2.09x        26%       0.84      2.0m2       0.37       38.4      never   13.8/min     27.0Wh
geodesic+sweep_optimal          72.9%      43.2%      1.72x        30%       0.61      2.5m2       0.44       39.7      never    9.0/min     27.0Wh
strips+sweep_optimal            77.3%      52.1%      1.87x        37%       0.89      1.5m2       0.36       32.7      never   25.6/min     27.0Wh
darp+sweep_optimal              73.5%      41.1%      1.88x        43%       0.72      2.9m2       0.37       38.8      never   12.3/min     27.0Wh
forest+sweep_optimal            70.3%      45.0%      1.55x        51%       0.59      2.6m2       0.35      *30.4      never   64.8/min     27.0Wh
darp+boustrophedon_cells        79.1%      42.8%      1.89x        39%       0.71      2.0m2       0.40       40.1      never   *4.4/min     27.0Wh
```

Six things are worth saying out loud.

**Partitioning does what it claims: it halves the overlap.** The five
partition-and-sweep rows sit at 26–51% overlap; the cooperative rows sit at
35–61%. `voronoi+sweep_optimal` at 26% is doing a third less duplicate work
than `binn_swarm` at 61%, and it removes more dirt while doing it.

**No method gets near 3x.** The best speedup here is 2.09, from two entirely
different methods -- a Voronoi partition and three `bsa` robots sharing a map.
Everything else is between 1.5 and 1.9. Three robots in a domestic pool spend a
real fraction of their time being three robots in a domestic pool.

**MSTC demonstrates its own paper's point.** Where the robots happen to sit on
the circuit decides how long an arc each gets, and nothing balances that: plain
MSTC (`mstc_nobt`) scores a balance of 0.08 -- one robot did almost nothing
while another did almost everything -- and 43% coverage. Turning on the
backtracking variant, where a finished robot takes the back half of the busiest
team-mate's remaining stretch and announces it, takes coverage to 68.8% and
balance to 0.51.

**`mstc_nobt` wins two columns by failing.** It has the lowest overlap (8%) and
the highest path efficiency (0.67) in the table, because a fleet that barely
moves does not repeat itself. This is the strongest argument in the package for
not ranking on one number.

**The swarm variants trade coverage for coordination, explicitly.**
`smc_swarm` covers eight points less than plain `smc` and does it with 37%
overlap instead of 59%, balance 0.97 instead of 0.85, and a third of the
collisions. That is what sharing one ergodic time-average *does*: the objective
is to match a distribution, and three robots serving one distribution spread
out. `binn_swarm` is the negative result -- peer inhibition makes the neural
field worse on coverage *and* on overlap, because it pushes robots off work
they were in the middle of.

**Sharing the covered map buys less than it should.** Three `frontier` robots
with the blackboard against three without, three seeds: overlap 57.5% to 50.0%,
and team coverage does not move. Knowing where your colleague has been is worth
about seven points of overlap and nothing else, because by the time you hear
about a cell you were usually not going there anyway.

## Pictures

```python
from zimablue.fleetplots import plot_fleet
plot_fleet(result).savefig("fleet.png")
```

<div align="center">
<img src="assets/fleet-views.png" alt="Paths, territory, overlap and progress for a three-robot fleet" width="900">
</div>

Four views, because none of them is enough on its own:

- **paths** — every robot's trajectory in its own colour. A partition is
  obvious at a glance here, and a partition that went wrong is more obvious
  still.
- **territory** — which robot actually covered each cell. Not the plan: the
  outcome. Put it next to the partition the partitioner drew, and the
  difference is the follower's error.
- **overlap** — how many different robots went over each cell. The waste,
  drawn. A good partition leaves a thin seam along its internal borders; a
  cooperative fleet with no partition covers the pool in plaid.
- **progress** — team coverage over time with each robot's own curve under it,
  so a robot that finished early and parked shows up as a line that goes flat
  while the others climb.

The replay window handles fleets too: each robot gets a coloured ring and its
own trail, and the HUD follows robot 0.

## Comparing fleets

```bash
python examples/fleet.py --compare --robots 3 --jobs 4
python examples/fleet.py --scaling --robots 4     # what each robot is worth
```

`compare_fleets` uses the same harness as the single-robot comparison with a
different set of dimensions — speedup, overlap and balance replace the ones
that have no team meaning — so `plot_matrix` and the rest work unchanged.
