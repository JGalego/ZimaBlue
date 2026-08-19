# Running on a robot

Everything in this library talks to a controller through two dataclasses.
`ControlInput` is a clock, the latest reading per sensor, battery, filter load
and the robot's own specification. `DriveCommand` is two track speeds, a brush
and a pump. Nothing in either of them is simulator-specific, which was the
point of writing them that way, and it means the port is narrow:

```python
from zimablue.hardware import DeviceSource, HardwareRuntime, WheelSpeedLoop
from zimablue.controllers.systematic import SystematicCoverage
import zimablue as zb

robot = zb.make_robot("tracked")

runtime = HardwareRuntime(
    controller=SystematicCoverage(),
    robot=robot,
    source=DeviceSource(
        channels={"encoder": ("left", "right"), "imu": ("ax", "ay", "gz")},
        poll=read_my_bus,
    ),
    actuate=write_my_motors,
    speed_loop=WheelSpeedLoop.for_robot(robot),
    pool=my_pool,
)
run = runtime.run(minutes=20)
run.save("runs/tuesday.zbr")
```

That replays in the ordinary viewer. `read_my_bus` and `write_my_motors` are
yours, and they are the only two functions the port actually requires.

Nothing here needs an extra installed — standard library and NumPy, because a
robot is the last place you want a dependency tree.

## What each piece is for

| | |
|---|---|
| [`DeviceSource`](../src/zimablue/hardware/sources.py) | Timestamps, holds the last good value, tracks staleness. Your `poll()` returns raw numbers or `None`. |
| [`WheelSpeedLoop`](../src/zimablue/hardware/motors.py) | PI plus feedforward, closing metres-per-second onto duty cycle. |
| [`Watchdog`](../src/zimablue/hardware/safety.py) | Stops the motors when a sensor goes quiet, the loop misses its deadline, or the controller raises. |
| [`HardwareRuntime`](../src/zimablue/hardware/runtime.py) | Sense, decide, actuate, record. Paces itself and counts overruns. |
| [`RecordedSource`](../src/zimablue/hardware/sources.py) | Replays a `.zbr` back out as readings, with jitter and dropouts. Test fixture. |
| [`TrajectorySource`](../src/zimablue/hardware/sources.py) | Drives the sensor models from a trajectory a real robot really drove. |

## The two numbers you have to measure

**Sensor noise.** The figures in `SensorConfig` — `noise_std`, `bias_walk`,
`latency`, `quantization`, `dropout_probability` — are consumer-MEMS-class
guesses, and the code has always said so. On real hardware you measure them:
leave the robot still for an hour and run an Allan variance on the gyro, drive
a known ten metres and read the encoders, log arrival times to find the real
latency. That is a week with a tape measure, and it is the highest-value week
in this whole project — it turns a plausible simulator into a calibrated one.
Every number in the README has to be re-run afterwards.

**Speed-loop gains.** `kp` and `ki` default to a starting point for a small
geared drive. Measure the step response: command a fixed effort, log the
encoder, fit a first-order lag, and set the gains from the time constant. Then
check what is easy to get wrong and painful to debug on a
robot — that a saturating turn keeps its radius, and that a jammed track does
not store up a lurch. Both are tested in
[`tests/test_hardware.py`](../tests/test_hardware.py); neither is tested by
anything the simulator can do.

## What a real run cannot tell you

`HardwareRun.metrics()` gives runtime, distance, collisions, watchdog trips and
loop overruns. It does not give coverage or dirt removed, and that is not an
oversight.

Both are computed from the true pose against the true dirt field. A robot has
neither. Computing coverage from the pose estimate instead produces a number
that looks exactly like the simulator's and means something else — a controller
whose estimate has drifted three metres will report near-perfect coverage of a
pool it never crossed, and the worse its localisation the better it will score.

To score cleaning on real hardware you need an external measurement: an
overhead camera for the pose, and a before-and-after of the floor for the dirt.
That is presumably why the industry advertises coverage rather than
cleanliness, and it is the same argument this library exists to make, arriving
from the other direction.

Every recording written by the runtime carries `pose_source: "estimate"` and
`ground_truth: false` in its manifest. Check it before comparing a real run
against a simulated one.

## Testing it, without a robot

Three tiers, in increasing cost and increasing truth.

**Replay a recording through the runtime.** `RecordedSource` turns any `.zbr`
into a reading stream, with `jitter` and `dropout` on top. This tests the
plumbing and the controller's timing assumptions, and nothing about physics.
It runs in CI, costs nothing, and found two bugs the first time it was pointed
at a real controller: a sonar reporting NaN — which is what a real rangefinder
does when no echo comes back — crashed the occupancy map on the first tick,
and a watchdog written to trip on a single invalid sample stopped the robot
every time the bus dropped a packet.

**Replay a real robot's trajectory.**

```bash
python tools/fetch_trajectory.py --all
python examples/replay_real_trajectory.py
```

A Pioneer 3-DX driving a real building, tracked at 300 Hz by a real motion
capture rig, from the [TUM RGB-D benchmark][tum]. A differential drive of
roughly a pool cleaner's mass moving at roughly its speed over roughly a pool's
worth of floor. `TrajectorySource` drives the shipped sensor models from that
motion and the estimator is scored against where the robot actually was:

| log | drove | final error | mean | worst |
|---|---|---|---|---|
| `pioneer_360` | 17.0 m | 0.18 m | 0.22 m | 0.44 m |
| `pioneer_slam` | 42.5 m | 0.15 m | 0.34 m | 1.08 m |
| `pioneer_slam2` | 23.3 m | 2.16 m | 0.86 m | 2.17 m |

Read those with the caveat attached, because it is a large one. The *motion* is
real — real accelerations, real stop-start, real turning, real tracker
dropouts including one of a full second. The *sensors* are not: readings are
synthesised through the shipped noise models, so the noise is still ours. And
there is **no slip**, because encoder readings are derived from the true body
velocity by inverse kinematics. A real drivetrain's encoders run long, and that
bias is the largest single term in dead-reckoning drift. The estimator is being
flattered here, in a known direction.

It told us something anyway. The estimator handles real motion without
falling over, which means the 13.7 m of drift it shows over a 25-minute
simulated run comes from the slip model rather than from real trajectories
being hard to integrate. And `pioneer_slam2` ends 39° out on heading while the
other two stay under 5°: the gyro bias is only observable when the robot stops,
the zero-velocity update is the only thing that makes it observable, and a
robot that rarely stops accumulates heading error the filter cannot see. That
is the case for the loop closure already on the [roadmap](roadmap.md).

**Log a real sensor.** The only way to close the gap, and it is cheap: tape a
phone to something that moves, log the gyro and accelerometer, and run an Allan
variance. Real MEMS noise, real bias walk, real timestamp jitter, real
dropouts. An afternoon's work, and the single highest-value contribution
anybody could make to this repository.

## What still will not transfer

A geometric planner ports and gets retuned. A learned policy does not — no
buoyancy, no cable drag, no wall climbing, and a slip model that is a constant
rather than a measurement. [`docs/ml.md`](ml.md) says the same thing from the
other end, and the [roadmap](roadmap.md)'s 3D backend is what would change it.

[tum]: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
