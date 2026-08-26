# Autonomous experiments

`AutonomousExperiment` chooses the next bounded parameter proposals from the
best trials so far. It is for simulation design questions such as controller
gains, brush settings or operating speed—not for fitting a recorded trajectory,
which is the job of `TwinCalibrator`.

```python
import zimablue as zb

def evaluate(parameters, seed):
    controller = MyController(turn_gain=parameters["turn_gain"])
    result = zb.Simulation(
        pool="kidney",
        dirt="autumn",
        controller=controller,
        seed=seed,
        record=False,
    ).run(minutes=10)
    return result.metrics.dirt_removed_fraction

experiment = zb.AutonomousExperiment(
    evaluate,
    [zb.Parameter("turn_gain", 0.2, 2.0, initial=0.8)],
    zb.ExperimentObjective("dirt removed", maximize=True, unit="fraction"),
    seed=42,
)
result = experiment.run(generations=8, population=10, replicates=4)
print(result.parameters, result.value, result.confidence_interval)
```

Every proposal in every generation receives the same replicate seeds. This is
common-random-number experimental design: a windy dirt field is compared with
the same windy field, rather than letting one candidate win because it drew an
easier world. The optimiser's `seed` independently fixes proposal generation.

The first generation explores the full bounds and includes the declared
initial point. Later generations sample around the elite proposals while
retaining the incumbent. Sampling contracts but keeps a one-percent floor on
each parameter range, avoiding a numerically frozen search.

`ExperimentResult.history` retains every candidate, every replicated value,
the incumbent and cumulative evaluation count. The reported 95% interval is a
normal approximation over the incumbent's replicate outcomes; with few seeds
it is a compact uncertainty indicator, not a substitute for a larger final
confirmation batch.

The evaluator must be deterministic for a given parameter dictionary and seed,
and must return one finite scalar. Run the winning parameters against a fresh,
held-out seed set before treating them as an improvement.