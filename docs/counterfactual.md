# Counterfactual replay

A replay says what happened. A counterfactual reruns the recorded initial
conditions and stochastic streams while changing one explicit choice.

```python
import zimablue as zb

baseline = zb.Recording.load("runs/kidney.zbr")
comparison = zb.run_counterfactual(baseline, "random_bounce")

print(comparison.summary())
print(comparison.metric_deltas["coverage"])
comparison.alternative.save("runs/kidney-random-bounce.zbr")
```

The pool, robot, dirt, seed, timestep, cell size and initial pose come from the
recording's embedded configuration. The alternative controller receives fresh
sensor readings from a new deterministic simulation; recorded sensor values
are not fed into it. This distinction matters: after the first different
command, the robot occupies a different state and must sense that state.

`divergence_time` is the first baseline timestamp whose planar position differs
by more than `divergence_tolerance`. `trajectory_rms` measures separation over
the overlapping interval. `metric_deltas` are alternative minus baseline, so a
positive coverage delta means the changed run covered more floor.

The complete alternative recording carries a `counterfactual` manifest entry
with the baseline seed, original and changed controller names, duration and
divergence tolerance.

## Change one model

Whole model objects can be replaced while other conditions remain fixed:

```python
comparison = zb.run_counterfactual(
    baseline,
    "baseline_coverage",
    robot=zb.make_robot("compact"),
)
```

Pass `pool=`, `robot=` or `dirt=` for controlled model substitutions. This is
a model comparison, not evidence that the physical machine would have followed
the simulated branch.

Counterfactual replay rejects hardware recordings whose pose is an estimate.
Their embedded measurements describe one real trajectory, and calling a new
simulation branch ground truth would overstate what the data can establish.