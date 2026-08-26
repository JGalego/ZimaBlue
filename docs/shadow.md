# Live shadow twin

`ShadowTwin` runs the same model beside a hardware control loop. It receives
the command that the real controller already chose and the readings that led
to that decision. It predicts the same sensor channels, advances under the
same command and reports their difference.

It cannot actuate. There is no actuator callback in its API.

```python
import zimablue as zb

shadow = zb.ShadowTwin(
    pool=survey.pool,
    robot=robot,
    seed=42,
    thresholds={
        "encoder.left": 0.08,
        "encoder.right": 0.08,
        "imu.gz": 0.15,
    },
)

tick = runtime.tick()  # the real controller owns this command
health = shadow.observe(tick.command, tick.readings, dt=tick.loop_period)

if not health.healthy:
    logger.warning(health.summary())
```

Residuals are `predicted - observed` and are kept in a bounded rolling window.
Each scalar channel reports sample count, mean, RMS and maximum absolute error.
An anomaly starts only after `minimum_samples` and when rolling RMS exceeds the
channel's configured threshold. `health.score` is the largest RMS-to-threshold
ratio, so values above one mean at least one threshold was crossed.

Pass the measured hardware loop interval as `dt`. The model uses that interval
for its next physics step, preventing clock drift from looking like a failing
motor when the loop jitters. If `dt` is omitted, the configured timestep is
used. Invalid or dropped readings are skipped rather than treated as zeros.

Use the same calibrated robot and sensor configuration in the shadow that the
machine carries. Persistent encoder residuals then point to changing slip,
obstruction or drivetrain wear; persistent IMU residuals point to a turn-rate
or bias mismatch. Thresholds retain physical channel units and remain a field
engineering decision rather than a universal constant hidden in the library.