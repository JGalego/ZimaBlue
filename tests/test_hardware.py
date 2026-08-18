"""The parts of a port that a simulator cannot exercise.

Every test here is about a way the world differs from ``Simulation.step``:
readings that arrive late or not at all, a loop that misses its deadline, a
controller that raises, a motor driver that takes duty cycle rather than metres
per second.  None of it is reachable from the ordinary test suite, because the
ordinary test suite runs against a backend that cannot fail.
"""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.controllers.base import ControlInput
from zimablue.controllers.baseline import BaselineCoverage
from zimablue.hardware import (
    DeviceSource,
    HardwareRuntime,
    MotorEffort,
    RecordedSource,
    SafetyLimits,
    Trajectory,
    TrajectorySource,
    Watchdog,
    WheelSpeedLoop,
)
from zimablue.robot import DriveCommand
from zimablue.sensors import Reading


@pytest.fixture(scope="module")
def recording():
    return (
        zb.Simulation(
            pool="kidney", robot="tracked", dirt="autumn", controller="baseline_coverage", seed=3
        )
        .run(seconds=30)
        .recording
    )


class FakeClock:
    """A clock the test advances by hand, so no test waits on wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


def runtime_for(source, controller=None, clock=None, **kwargs):
    robot = zb.make_robot("tracked")
    clock = clock or FakeClock()
    return (
        HardwareRuntime(
            controller=controller or BaselineCoverage(),
            robot=robot,
            source=source,
            actuate=kwargs.pop("actuate", lambda command: None),
            clock=clock,
            sleep=clock.sleep,
            **kwargs,
        ),
        clock,
    )


# -- DeviceSource ---------------------------------------------------------


def test_a_sensor_that_has_never_reported_is_not_valid():
    source = DeviceSource({"encoder": ("left", "right")}, poll=lambda: {})
    reading = source.read(0.0)["encoder"]
    assert not reading.valid
    assert not reading.fresh
    assert reading.values.tolist() == [0.0, 0.0]


def test_a_missing_sample_holds_the_last_good_values():
    samples = [{"encoder": (0.3, 0.31)}, {}, {}]
    source = DeviceSource({"encoder": ("left", "right")}, poll=lambda: samples.pop(0))

    first = source.read(0.0)["encoder"]
    assert first.fresh and first.valid

    held = source.read(0.02)["encoder"]
    assert not held.fresh, "a held sample must not claim to be a new measurement"
    assert held.valid, "20 ms old is perfectly usable"
    assert held.values.tolist() == first.values.tolist()


def test_a_sensor_goes_invalid_once_it_is_properly_stale():
    samples = [{"encoder": (0.3, 0.31)}]
    source = DeviceSource(
        {"encoder": ("left", "right")},
        poll=lambda: samples.pop(0) if samples else {},
        stale_after=0.5,
    )
    source.read(0.0)
    assert source.read(0.4)["encoder"].valid
    assert not source.read(0.6)["encoder"].valid
    assert source.ages["encoder"] == pytest.approx(0.6)


def test_the_wrong_number_of_values_is_a_loud_error():
    """Silently accepting it would steer the robot on whatever landed in slot 0."""
    source = DeviceSource({"imu": ("ax", "ay", "gz")}, poll=lambda: {"imu": (0.1, 0.2)})
    with pytest.raises(ValueError, match="3 channels"):
        source.read(0.0)


def test_an_undeclared_sensor_is_a_loud_error():
    source = DeviceSource({"encoder": ("left", "right")}, poll=lambda: {"sonar": (1.0,)})
    with pytest.raises(KeyError, match="not declared"):
        source.read(0.0)


# -- RecordedSource -------------------------------------------------------


def test_a_recording_replays_as_readings(recording):
    source = RecordedSource(recording)
    assert set(source.channels) == {"encoder", "imu", "pressure", "contact", "sonar"}
    assert source.channels["imu"] == ("ax", "ay", "gz")

    readings = source.read(5.0)
    assert readings["encoder"].values.size == 2
    assert readings["sonar"].values.size == 3


def test_nan_in_a_recording_means_not_reported_yet(recording):
    """The recorder back-fills a late-arriving sensor's early frames with NaN.

    Handing those to a controller as though they were measurements is how a
    replayed run used to crash on the first tick -- ``int(nan / cell)`` raises,
    from inside an occupancy map that had no reason to expect one.
    """
    source = RecordedSource(recording)
    first = source.read(0.0)
    assert np.isnan(first["sonar"].values).any(), "expected the fixture to have a late sensor"
    assert not first["sonar"].valid, "a NaN channel must not be presented as valid"


def test_dropouts_hold_the_previous_reading(recording):
    source = RecordedSource(recording, dropout=1.0, seed=0)
    source._held["encoder"] = Reading("encoder", 0.0, np.array([0.5, 0.5]))
    reading = source.read(1.0)["encoder"]
    assert not reading.valid
    assert reading.values.tolist() == [0.5, 0.5]


# -- the controllers, under conditions they were not written for -----------


def test_a_sonar_with_no_echo_does_not_crash_the_planner():
    """A real rangefinder reports NaN when nothing comes back.

    The simulated one never does, so this went unnoticed until a recording was
    replayed through the runtime and the occupancy map raised on tick one.
    """
    from zimablue.controllers.systematic import SystematicCoverage

    robot = zb.make_robot("tracked")
    controller = SystematicCoverage()
    controller.reset(robot)
    readings = {
        "encoder": Reading("encoder", 0.0, np.array([0.2, 0.2])),
        "imu": Reading("imu", 0.0, np.array([0.0, 0.0, 0.0])),
        "sonar": Reading("sonar", 0.0, np.array([np.nan, np.nan, np.nan])),
        "contact": Reading("contact", 0.0, np.zeros(4)),
    }
    command = controller.step(
        ControlInput(
            time=0.0, dt=0.02, readings=readings, battery=1.0, filter_load=0.0, robot=robot
        )
    )
    assert np.isfinite(command.left) and np.isfinite(command.right)


def test_an_infinite_range_is_treated_as_clear_not_as_a_wall():
    from zimablue.controllers.systematic import OccupancyMap

    grid = OccupancyMap(extent=4.0, cell=0.1)
    before = grid.grid.copy()
    grid.observe_ray(0.0, 0.0, 0.0, float("inf"), hit=False)
    grid.observe_ray(0.0, 0.0, 0.0, float("nan"), hit=True)
    assert np.array_equal(grid.grid, before), "a non-measurement must not carve or mark anything"


# -- WheelSpeedLoop -------------------------------------------------------


def first_order_plant(target_history, tau=0.15, dt=0.02, max_speed=0.32):
    """A lag between commanded effort and achieved speed. Enough to tune against."""
    speed = np.zeros(2)
    for effort in target_history:
        speed += (np.asarray(effort) * max_speed - speed) * (dt / tau)
        yield speed.copy()


def test_the_speed_loop_converges_on_the_commanded_speed():
    loop = WheelSpeedLoop(max_speed=0.32)
    command = DriveCommand(left=0.25, right=0.25)
    speed = np.zeros(2)
    dt, tau = 0.02, 0.15
    for _ in range(300):
        effort = loop(command, (speed[0], speed[1]), dt)
        speed += (np.array(effort.as_tuple()) * 0.32 - speed) * (dt / tau)
    assert speed == pytest.approx([0.25, 0.25], abs=0.01)


def test_a_saturating_turn_keeps_its_radius():
    """Clipping the two sides independently changes the commanded turn.

    A classic differential-drive bug and an ugly one, because the symptom is a
    planner that seems to mis-measure its own turns.
    """
    loop = WheelSpeedLoop(max_speed=0.32, slew=0.0)
    command = DriveCommand(left=-0.32, right=0.32)
    effort = loop(command, (0.0, 0.0), 0.02)
    assert effort.saturated
    assert effort.left == pytest.approx(-effort.right), "the turn was made asymmetric by clipping"


def test_a_stalled_track_does_not_wind_up_the_integrator():
    """The lurch when a stuck robot comes free is stored-up integral."""
    loop = WheelSpeedLoop(max_speed=0.32)
    command = DriveCommand(left=0.3, right=0.3)
    for _ in range(500):
        loop(command, (0.0, 0.0), 0.02)  # jammed: never moves
    assert abs(loop.integral[0]) <= loop.integral_limit + 1e-9


def test_a_commanded_stop_clears_the_integral():
    loop = WheelSpeedLoop(max_speed=0.32)
    for _ in range(50):
        loop(DriveCommand(left=0.3, right=0.3), (0.0, 0.0), 0.02)
    effort = loop(DriveCommand.stop(), (0.0, 0.0), 0.02)
    assert loop.integral == (0.0, 0.0)
    assert effort.as_tuple() == pytest.approx((0.0, 0.0)), (
        "a commanded stop must take effect this tick, not be ramped down over "
        "several -- the slew limit is there for the way up"
    )


def test_the_slew_limit_bounds_how_fast_effort_can_change():
    loop = WheelSpeedLoop(max_speed=0.32, slew=2.0)
    effort = loop(DriveCommand(left=0.32, right=0.32), (0.0, 0.0), 0.02)
    assert abs(effort.left) <= 2.0 * 0.02 + 1e-9


def test_a_zero_timestep_is_rejected():
    loop = WheelSpeedLoop(max_speed=0.32)
    with pytest.raises(ValueError, match="dt must be positive"):
        loop(DriveCommand(), (0.0, 0.0), 0.0)


# -- Watchdog -------------------------------------------------------------


def live(name="encoder", t=0.0):
    return {name: Reading(name, t, np.zeros(2), valid=True, fresh=True)}


def test_one_dropped_sample_is_not_a_lost_sensor():
    """Every real bus drops packets; a watchdog that stops on one stops always."""
    dog = Watchdog(SafetyLimits(required=("encoder",), startup_grace=0.0, max_reading_age=0.5))
    dog.check(0.0, live(t=0.0))
    dropped = {"encoder": Reading("encoder", 0.1, np.zeros(2), valid=False, fresh=False)}
    assert dog.check(0.1, dropped) == []
    assert not dog.tripped


def test_a_sensor_that_stays_quiet_trips_the_watchdog():
    dog = Watchdog(SafetyLimits(required=("encoder",), startup_grace=0.0, max_reading_age=0.5))
    dog.check(0.0, live(t=0.0))
    dead = {"encoder": Reading("encoder", 0.0, np.zeros(2), valid=False, fresh=False)}
    assert dog.check(0.4, dead) == []
    reasons = dog.check(0.7, dead)
    assert reasons and "no usable reading" in reasons[0]
    assert dog.tripped


def test_a_sensor_that_never_reports_trips_once_the_grace_period_ends():
    dog = Watchdog(SafetyLimits(required=("encoder",), startup_grace=1.0, max_reading_age=0.5))
    never = {"encoder": Reading("encoder", 0.0, np.zeros(2), valid=False, fresh=False)}
    assert dog.check(0.0, never) == []
    assert dog.check(0.5, never) == [], "still inside the grace period"
    assert dog.check(1.6, never), "an unplugged sensor must not be tolerated forever"


def test_the_watchdog_latches():
    """A sensor that flickers back for one tick has not recovered."""
    dog = Watchdog(SafetyLimits(required=("encoder",), startup_grace=0.0, max_reading_age=0.1))
    dead = {"encoder": Reading("encoder", 0.0, np.zeros(2), valid=False, fresh=False)}
    dog.check(0.0, dead)
    dog.check(0.5, dead)
    assert dog.tripped
    dog.check(0.6, live(t=0.6))
    assert dog.tripped, "recovery is not the watchdog's decision"
    dog.clear()
    assert not dog.tripped


def test_a_slow_loop_and_a_slow_controller_are_different_faults():
    dog = Watchdog(SafetyLimits(max_loop_period=0.1, max_decide_time=0.05))
    assert "loop period" in dog.check(0.0, {}, loop_period=0.3)[0]
    dog.clear()
    assert "controller took" in dog.check(0.0, {}, decide_time=0.2)[0]


def test_the_safe_command_stops_everything():
    command = Watchdog.safe_command()
    assert (command.left, command.right, command.brush, command.pump) == (0.0, 0.0, False, 0.0)


# -- HardwareRuntime ------------------------------------------------------


def test_a_run_replays_in_the_ordinary_viewer(recording, tmp_path):
    """The acceptance test for the whole port.

    The roadmap sets the same bar for the 3D backend: whatever writes a .zbr,
    the 2D replay has to be able to open it. If that holds the abstraction is
    real, and if it does not then the recording format has quietly become two
    formats that share a file extension.
    """
    runtime, _ = runtime_for(RecordedSource(recording), pool=zb.make_pool("kidney"))
    run = runtime.run(seconds=5)
    path = run.save(tmp_path / "real.zbr")

    reloaded = zb.Recording.load(path)
    assert reloaded.n_frames == run.ticks
    assert {"x", "y", "heading", "cmd_left", "encoder.left"} <= set(reloaded.channels)
    assert reloaded.manifest["backend"] == "hardware"


def test_a_hardware_recording_says_its_pose_is_an_estimate(recording):
    """Without this key nothing downstream can tell the two apart, and a real
    run's numbers get compared against a simulated one's as though they were
    the same measurement."""
    runtime, _ = runtime_for(RecordedSource(recording))
    run = runtime.run(seconds=2)
    assert run.recording.manifest["pose_source"] == "estimate"
    assert run.recording.manifest["ground_truth"] is False


def test_a_controller_that_raises_stops_the_robot(recording):
    class Broken:
        name = "broken"

        def reset(self, robot):
            pass

        def step(self, control_input):
            raise RuntimeError("planner exploded")

    sent = []
    runtime, _ = runtime_for(RecordedSource(recording), controller=Broken(), actuate=sent.append)
    run = runtime.run(max_ticks=5)

    assert run.watchdog_reasons and "planner exploded" in run.watchdog_reasons[0]
    assert all(command.left == 0.0 and command.right == 0.0 for command in sent)
    assert any(event.kind == "fault" for event in run.events)


def test_finishing_stops_the_motors_even_when_nothing_else_happened(recording):
    sent = []
    runtime, _ = runtime_for(RecordedSource(recording), actuate=sent.append)
    runtime.run(max_ticks=3)
    assert sent[-1].left == 0.0 and sent[-1].right == 0.0 and sent[-1].pump == 0.0


def test_a_finished_runtime_refuses_to_finish_twice(recording):
    runtime, _ = runtime_for(RecordedSource(recording))
    runtime.run(max_ticks=2)
    with pytest.raises(RuntimeError, match="already been finished"):
        runtime.finish()


def test_a_run_needs_a_stopping_condition(recording):
    runtime, _ = runtime_for(RecordedSource(recording))
    with pytest.raises(ValueError, match="stopping condition"):
        runtime.run()


def test_overruns_are_counted_rather_than_absorbed_silently(recording):
    """A 50 Hz loop quietly becoming a 30 Hz one is how an estimator starts
    drifting for no visible reason."""
    clock = FakeClock()
    slow = {"n": 0}

    class Slow:
        name = "slow"

        def reset(self, robot):
            pass

        def step(self, control_input):
            slow["n"] += 1
            if slow["n"] % 3 == 0:
                clock.now += 0.09  # blocked on something
            return DriveCommand(left=0.1, right=0.1)

    runtime, _ = runtime_for(RecordedSource(recording), controller=Slow(), clock=clock)
    run = runtime.run(max_ticks=30)
    assert run.overruns > 0
    assert run.metrics()["overruns"] == run.overruns


def test_the_speed_loop_is_used_when_one_is_given(recording):
    sent = []
    runtime, _ = runtime_for(
        RecordedSource(recording),
        actuate=sent.append,
        speed_loop=WheelSpeedLoop(max_speed=0.32),
    )
    runtime.run(max_ticks=5)
    assert all(isinstance(item, MotorEffort) for item in sent)
    assert all(-1.0 <= item.left <= 1.0 for item in sent)


def test_a_controller_never_sees_ground_truth_on_hardware(recording):
    """There is no argument to turn this on, and there should not be: an
    oracle cannot run on a robot, and the option would invite somebody to
    wire the pose estimate into it."""
    seen = []

    class Nosy:
        name = "nosy"

        def reset(self, robot):
            pass

        def step(self, control_input):
            seen.append(control_input.truth)
            return DriveCommand()

    runtime, _ = runtime_for(RecordedSource(recording), controller=Nosy())
    runtime.run(max_ticks=5)
    assert seen and all(truth is None for truth in seen)


def test_a_summary_of_a_hardware_run_does_not_claim_zero_coverage(recording, tmp_path):
    """Printing "coverage 0%" for an unmeasured run is worse than printing
    nothing: it looks exactly like a controller that did nothing."""
    pytest.importorskip("matplotlib")
    from zimablue.replay import export_summary

    runtime, _ = runtime_for(RecordedSource(recording), pool=zb.make_pool("kidney"))
    run = runtime.run(seconds=3)
    export_summary(run.recording, tmp_path / "summary.png")
    assert (tmp_path / "summary.png").exists()


def test_replaying_a_recording_with_no_pool_says_what_is_missing(recording):
    pytest.importorskip("matplotlib")
    from zimablue.replay.renderer import load_scene

    runtime, _ = runtime_for(RecordedSource(recording))  # no pool= given
    run = runtime.run(max_ticks=3)
    with pytest.raises(ValueError, match="no pool geometry"):
        load_scene(run.recording)


def test_inspect_does_not_report_zeros_for_a_hardware_recording(recording, tmp_path):
    """``Metrics.from_dict`` fills missing keys with zeros, so a real run that
    drove 41 m came out of ``zimablue inspect`` reading "distance 0.0 m"."""
    from typer.testing import CliRunner

    from zimablue.cli import app

    runtime, _ = runtime_for(RecordedSource(recording))
    run = runtime.run(seconds=5)
    path = run.save(tmp_path / "hw.zbr")

    result = CliRunner().invoke(app, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "no ground truth" in result.output.lower()
    # Rows only the full Metrics table emits. Their presence means the short
    # hardware metrics were padded out with zeros again.
    assert "wall coverage" not in result.output.lower()
    assert "uniformity" not in result.output.lower()
    assert f"{run.distance:.4g}"[:4] in result.output, "the real distance should be reported"


def test_a_recording_knows_whether_its_pose_is_true(recording):
    simulated = recording
    runtime, _ = runtime_for(RecordedSource(recording))
    real = runtime.run(max_ticks=3).recording
    assert simulated.has_ground_truth
    assert not real.has_ground_truth


def test_metrics_omit_the_ones_a_robot_cannot_measure(recording):
    runtime, _ = runtime_for(RecordedSource(recording))
    metrics = runtime.run(max_ticks=10).metrics()
    assert "coverage" not in metrics and "dirt_removed" not in metrics
    assert {"distance", "runtime", "collisions"} <= set(metrics)


def test_the_ticks_arrive_at_the_requested_rate(recording):
    clock = FakeClock()
    runtime, _ = runtime_for(RecordedSource(recording), clock=clock, rate_hz=25.0)
    run = runtime.run(seconds=4.0)
    assert run.ticks == pytest.approx(100, abs=3)


def test_jitter_and_dropouts_do_not_break_a_real_controller(recording):
    """The cheapest possible sim-to-real check, and it found two bugs."""
    from zimablue.controllers.systematic import SystematicCoverage

    runtime, _ = runtime_for(
        RecordedSource(recording, jitter=0.005, dropout=0.1, seed=7),
        controller=SystematicCoverage(),
        watchdog=Watchdog(SafetyLimits(required=("encoder", "imu"))),
    )
    run = runtime.run(seconds=20)
    assert run.watchdog_reasons == []
    assert run.ticks > 900


# -- Trajectory -----------------------------------------------------------


def straight_line(n=600, rate=100.0, speed=0.4):
    t = np.arange(n) / rate
    return Trajectory(time=t, x=speed * t, y=np.zeros(n), heading=np.zeros(n), source="synthetic")


def test_a_trajectory_differentiates_to_the_speed_it_was_built_with():
    v, omega = straight_line().body_velocities()
    assert v[len(v) // 2] == pytest.approx(0.4, abs=0.01)
    assert abs(omega).max() < 1e-6


def test_reversing_reads_as_negative_speed():
    t = np.arange(400) / 100.0
    x = np.concatenate([np.linspace(0, 1, 200), np.linspace(1, 0, 200)])
    v, _ = Trajectory(time=t, x=x, y=np.zeros(400), heading=np.zeros(400)).body_velocities()
    assert v[100] > 0 and v[300] < 0


def test_a_trajectory_needs_more_than_one_sample():
    with pytest.raises(ValueError, match="at least two"):
        Trajectory(time=np.zeros(1), x=np.zeros(1), y=np.zeros(1), heading=np.zeros(1))


def test_recentring_puts_the_start_at_the_origin():
    t = np.arange(100) / 50.0
    trajectory = Trajectory(time=t, x=t + 5.0, y=np.full(100, -3.0), heading=np.full(100, 1.2))
    recentred = trajectory.recentre()
    assert (recentred.x[0], recentred.y[0], recentred.heading[0]) == pytest.approx((0, 0, 0))
    assert recentred.path_length == pytest.approx(trajectory.path_length)


def test_a_trajectory_drives_the_real_sensor_models():
    robot = zb.make_robot("tracked")
    source = TrajectorySource(straight_line(), robot, seed=0)
    assert set(source.channels) == {"encoder", "imu"}

    readings = source.read(3.0)
    left, right = readings["encoder"].values
    assert left == pytest.approx(0.4, abs=0.05), "driving straight, both tracks at the body speed"
    assert right == pytest.approx(0.4, abs=0.05)
    assert source.truth_at(3.0)[0] == pytest.approx(1.2, abs=0.01)


def test_asking_a_trajectory_for_a_sensor_the_robot_lacks_is_a_loud_error():
    with pytest.raises(KeyError, match="no sensors named"):
        TrajectorySource(straight_line(), zb.make_robot("tracked"), sensors=("lidar",))
