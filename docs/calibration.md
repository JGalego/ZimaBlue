# Calibration

A digital twin is only useful while its parameters match the machine. ZimaBlue
can identify bounded parameters from an observed `.zbr` trajectory without
assuming which robot, controller or backend produced it.

`TwinCalibrator` takes three things:

1. a reference recording;
2. named parameter bounds;
3. a function that runs the twin with one proposed parameter mapping.

```python
from dataclasses import replace

import zimablue as zb

reference = zb.Recording.load("runs/measured.zbr")
base = zb.make_robot("tracked")

def run_twin(values):
    motor = replace(base.locomotion.left.motor, max_accel=values["max_accel"])
    left = replace(base.locomotion.left, motor=motor)
    right = replace(base.locomotion.right, motor=motor)
    robot = replace(base, locomotion=replace(base.locomotion, left=left, right=right))
    return zb.Simulation(pool="kidney", robot=robot, seed=42).run(minutes=2).require_recording()

calibrator = zb.TwinCalibrator(
    reference,
    run_twin,
    [zb.Parameter("max_accel", 0.4, 1.4, initial=0.9)],
)
fitted = calibrator.fit(seed=7, generations=30)
print(fitted.parameters, fitted.loss)
fitted.annotate().save("runs/fitted.zbr")
```

The callback owns model construction on purpose. Parameter paths would tie the
calibrator to today's dataclass layout and exclude plugins. A closure can change
robot components, controller tuning, sensor faults, backend coefficients or a
user model without changes to the calibration engine.

## Loss

The default `trajectory_loss` linearly resamples the candidate onto the
reference clock and averages squared `x`, `y` and `heading` residuals. Heading
uses circular distance, so $-\pi$ and $+\pi$ are neighbours rather than a full
turn apart. Pass a custom loss when sensor distributions or task metrics are
the observable quantity.

## Reproducibility

The optimizer is bounded differential evolution implemented with NumPy. Its
seed controls proposals and crossover. The simulation callback must also hold
its own seed fixed during fitting; otherwise optimizer noise and model error
become indistinguishable.

`CalibrationResult.annotate()` writes the fitted parameters, loss, seed,
evaluation count and convergence history under `manifest["calibration"]`.
That metadata survives a normal `.zbr` save/load cycle without changing the
recording schema.