"""The fast 2D backend -- ZimaBlue's reference implementation.

Planar rigid body, differential-drive kinematics with slip, penetration-based
contact against Shapely-derived segments, an exponential cleaning model, and
analytical sensors.  Pure NumPy float64 on the CPU, no GPU, no external engine.

Known approximations, stated rather than hidden:

* **Walls are 1D.** The robot moves on the floor plane. Wall climbing is not
  simulated; wall coverage is tracked as an unrolled perimeter visit record.
* **The model is kinematic, not dynamic.** Motors are rate- and acceleration-limited and
  slip is modelled, but there is no explicit force balance or momentum. At
  cleaner speeds (< 0.4 m/s) and masses (< 20 kg) the difference does not
  change navigation outcomes.
* **Water is a static drift field.** Dirt advects along it; the robot does not
  disturb it beyond local resuspension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zimablue.backends.base import BACKENDS, Event, SimState, StepResult
from zimablue.physics.cleaning import apply_cleaning
from zimablue.physics.collision import resolve
from zimablue.physics.kinematics import exact_arc_step, slip_factors
from zimablue.sensors import SensorContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.rng import RngTree
    from zimablue.robot import Cleaner, DriveCommand
    from zimablue.sensors import Reading
    from zimablue.world import World

__all__ = ["Fast2DBackend"]

STUCK_SPEED = 0.02
"""Ground speed below which the robot counts as not translating, m/s."""

STUCK_YAW_RATE = 0.08
"""Yaw rate below which the robot counts as not rotating, rad/s.

Rotation has to be part of the test: a robot deliberately spinning in place
translates at zero speed, and a translation-only check flags that as stuck --
which then makes every escape manoeuvre look like the failure it is trying to
resolve, and traps the robot in a permanent turn.
"""

STUCK_DURATION = 1.5
"""How long it must be commanded to move without moving, s."""

DRIFT_INTERVAL = 1.0
"""Simulated seconds between dirt advection updates. The flow field is slow and
smooth; recomputing it every 20 ms would cost time and change nothing."""


class Fast2DBackend:
    """Deterministic planar simulation of a cleaner in a pool."""

    name = "fast2d"

    def __init__(self, *, enable_dirt_drift: bool = True) -> None:
        self.enable_dirt_drift = enable_dirt_drift
        self.world: World | None = None
        self.neighbours: tuple[tuple[float, float, float], ...] = ()
        """Other robots this tick, as ``(x, y, radius)`` discs.

        Set by :class:`~zimablue.fleet.Fleet` before each tick and empty
        otherwise, so a single-robot run pays nothing for the fleet's
        existence. Feeding it through the backend rather than the world is
        deliberate: it changes every tick, and the world does not."""
        self.robot: Cleaner | None = None
        self._rng: RngTree | None = None
        self._slip_rng: np.random.Generator | None = None
        self._last_drift = 0.0
        self._visits: np.ndarray | None = None
        self._last_visit_step: np.ndarray | None = None
        self._wall_visits: np.ndarray | None = None
        self._prev_v = 0.0
        self._prev_omega = 0.0
        self._warned_filter = False
        self._warned_battery = False
        self._blocking = False

    # ------------------------------------------------------------------
    def reset(self, world: World, robot: Cleaner, rng: RngTree) -> SimState:
        self.world = world
        self.robot = robot
        self._rng = rng
        self._slip_rng = rng.stream("slip")
        self._last_drift = 0.0
        self._prev_v = 0.0
        self._prev_omega = 0.0
        self._warned_filter = False
        self._warned_battery = False
        self._blocking = False

        grid = world.pool.grid(world.cell)
        self._visits = np.zeros(grid.shape, dtype=np.int32)
        # -1 rather than 0: step 0 must count as a first visit, not a repeat.
        self._last_visit_step = np.full(grid.shape, -2, dtype=np.int64)
        # Wall coverage as an unrolled perimeter: 10 cm bins around the outline.
        self._wall_visits = np.zeros(
            max(8, int(np.ceil(world.pool.perimeter_length / 0.1))), dtype=np.int32
        )

        robot.sensors.attach(rng)

        x, y, heading = world.pool.start_pose(clearance=robot.radius + 0.05)
        capacity = robot.power.battery.capacity_wh
        state = SimState(
            x=x,
            y=y,
            heading=heading,
            depth=float(world.pool.depth_at(x, y)),
            battery_wh=capacity * robot.power.battery.initial_charge,
            _battery_capacity=capacity,
            _battery_fraction=robot.power.battery.initial_charge,
        )
        self._mark_visit(state)
        return state

    # ------------------------------------------------------------------
    def step(self, state: SimState, command: DriveCommand, dt: float) -> StepResult:
        if self.world is None or self.robot is None or self._slip_rng is None:
            raise RuntimeError("Fast2DBackend.step() called before reset()")
        world, robot = self.world, self.robot
        pool = world.pool
        events: list[Event] = []

        command = command.clamped(robot.locomotion)

        # 1. Motors: slew the tracks toward the commanded speeds.
        left = robot.locomotion.left.motor.apply_limits(state.wheel_left, command.left, dt)
        right = robot.locomotion.right.motor.apply_limits(state.wheel_right, command.right, dt)

        # 2. Slip: what the tracks do vs what the ground sees.
        noise = self._slip_rng.normal(0.0, 0.01, size=2)
        slip_l, slip_r = slip_factors(
            robot.locomotion,
            pool.material.friction,
            left,
            right,
            noise_left=float(noise[0]),
            noise_right=float(noise[1]),
        )
        ground_left = left * (1.0 - slip_l)
        ground_right = right * (1.0 - slip_r)
        v, omega = robot.locomotion.to_body_velocity(ground_left, ground_right)

        # 3. Integrate.
        nx, ny, heading = exact_arc_step(state.x, state.y, state.heading, v, omega, dt)

        # 4. Contact.
        contact = resolve(pool, nx, ny, heading, robot.radius, self.neighbours)
        if contact.touching:
            nx, ny = contact.x, contact.y
            # Edge-triggered: a robot pressed against a wall for two seconds is
            # one collision, not a hundred. Sustained contact is visible in the
            # per-frame contact flags instead.
            if not state.collided:
                events.append(
                    Event(
                        state.time + dt,
                        "collision",
                        {
                            "x": nx,
                            "y": ny,
                            "penetration": contact.penetration,
                            "obstacle": contact.is_obstacle,
                            "robot": contact.is_robot,
                        },
                    )
                )

        # Actual achieved motion, which is what the IMU should see.
        moved = float(np.hypot(nx - state.x, ny - state.y))
        achieved_v = float(np.copysign(moved / dt, v) if dt > 0 else 0.0)

        new = state.copy()
        new.time = state.time + dt
        new.step = state.step + 1
        new.x, new.y, new.heading = nx, ny, heading
        new.v = achieved_v
        new.omega = omega
        new.accel_forward = (achieved_v - self._prev_v) / dt if dt > 0 else 0.0
        # Centripetal term: what a body-mounted lateral accelerometer feels.
        new.accel_lateral = achieved_v * omega
        new.wheel_left, new.wheel_right = left, right
        new.slip_left, new.slip_right = slip_l, slip_r
        new.depth = float(pool.depth_at(nx, ny))
        new.contacts = contact.flags
        new.collided = contact.touching
        new.distance = state.distance + moved
        self._prev_v, self._prev_omega = achieved_v, omega

        # 5. Stuck detection: asked to move somehow, achieving neither
        # translation nor rotation.
        v_cmd, omega_cmd = robot.locomotion.to_body_velocity(command.left, command.right)
        wants_motion = abs(v_cmd) > 0.03 or abs(omega_cmd) > 0.15
        has_motion = abs(achieved_v) >= STUCK_SPEED or abs(omega) >= STUCK_YAW_RATE
        if wants_motion and not has_motion:
            new.stuck_time = state.stuck_time + dt
            if new.stuck_time >= STUCK_DURATION and not state.stuck:
                new.stuck = True
                events.append(Event(new.time, "stuck", {"x": nx, "y": ny}))
        else:
            if state.stuck:
                events.append(Event(new.time, "unstuck", {"x": nx, "y": ny}))
            new.stuck = False
            new.stuck_time = 0.0

        # 6. Cleaning.
        outcome = apply_cleaning(
            pool,
            world.dirt,
            robot,
            x=nx,
            y=ny,
            heading=heading,
            speed=abs(achieved_v),
            brush_on=command.brush,
            pump_duty=command.pump,
            filter_load=state.filter_load,
            dt=dt,
            cell=world.cell,
        )
        new.filter_load = outcome.filter_load
        new.dirt_collected = state.dirt_collected + outcome.captured
        new.dirt_removed = state.dirt_removed + outcome.total_removed - outcome.passed_through
        if outcome.debris_collected:
            events.append(Event(new.time, "debris_collected", {"count": outcome.debris_collected}))
        if outcome.debris_blocked and not self._blocking:
            # Also edge-triggered: shoving the same leaf along for a second
            # should read as one blockage, not fifty.
            events.append(Event(new.time, "debris_blocked", {"count": outcome.debris_blocked}))
        self._blocking = bool(outcome.debris_blocked)
        if outcome.filter_full and not self._warned_filter:
            self._warned_filter = True
            events.append(Event(new.time, "filter_full", {"load": outcome.filter_load}))

        # 7. Power.
        load = robot.chassis.submerged_weight * pool.material.friction * 0.5
        power = robot.power_draw(
            left,
            right,
            load,
            load,
            brush_on=command.brush,
            pump_duty=command.pump,
        )
        used = power * dt / 3600.0
        new.power_w = power
        new.battery_wh = max(0.0, state.battery_wh - used)
        new.energy_used_wh = state.energy_used_wh + used
        new._battery_fraction = new.battery_wh / max(new._battery_capacity, 1e-9)
        cutoff = robot.power.battery.cutoff
        if new._battery_fraction <= cutoff and not self._warned_battery:
            self._warned_battery = True
            events.append(Event(new.time, "battery_empty", {"fraction": new._battery_fraction}))
        elif new._battery_fraction <= cutoff + 0.10 and state._battery_fraction > cutoff + 0.10:
            events.append(Event(new.time, "battery_low", {"fraction": new._battery_fraction}))

        # 8. Environment: let the water move fine dirt around, occasionally.
        if self.enable_dirt_drift and new.time - self._last_drift >= DRIFT_INTERVAL:
            elapsed = new.time - self._last_drift
            vx, vy = pool.flow_grid(world.cell)
            world.dirt.field.drift(vx, vy, elapsed)
            # The robot's own wake stirs whatever it is currently sitting on.
            # Batched here rather than per tick: over 20 ms the swath barely
            # moves, so diffusing that often costs a third of the step budget
            # and changes nothing.
            wake = pool.grid(world.cell).window(nx, ny, 0.5 * robot.swath_width)
            if wake is not None:
                world.dirt.field.disturb_window(wake, strength=0.35)
            self._last_drift = new.time

        self._mark_visit(new)
        return StepResult(state=new, events=events)

    # ------------------------------------------------------------------
    def sense(self, state: SimState) -> dict[str, Reading]:
        if self.world is None or self.robot is None:
            raise RuntimeError("Fast2DBackend.sense() called before reset()")
        ctx = SensorContext(
            time=state.time,
            x=state.x,
            y=state.y,
            heading=state.heading,
            vx=state.v * float(np.cos(state.heading)),
            vy=state.v * float(np.sin(state.heading)),
            speed=state.v,
            yaw_rate=state.omega,
            accel_forward=state.accel_forward,
            accel_lateral=state.accel_lateral,
            wheel_speed_left=state.wheel_left,
            wheel_speed_right=state.wheel_right,
            depth=state.depth,
            contacts=state.contacts,
            pool=self.world.pool,
            water=self.world.pool.water,
            neighbours=self.neighbours,
            dirt_density=self.world.dirt.field.density_at(
                state.x, state.y, 0.5 * self.robot.swath_width
            ),
        )
        return self.robot.sensors.update(ctx)

    def close(self) -> None:
        """Nothing to release; present so the protocol is satisfied honestly."""

    # ------------------------------------------------------------------
    # Coverage bookkeeping -- shared by metrics and replay.
    # ------------------------------------------------------------------
    def _mark_visit(self, state: SimState) -> None:
        assert self.world is not None and self.robot is not None
        assert self._visits is not None and self._wall_visits is not None
        assert self._last_visit_step is not None
        grid = self.world.pool.grid(self.world.cell)
        window = grid.window(state.x, state.y, 0.5 * self.robot.swath_width)
        if window is not None:
            # Count *passes*, not ticks. Incrementing every tick would make a
            # robot that pauses for two seconds look like it covered the cell
            # a hundred times, and would put the revisit metric in the
            # hundreds regardless of the path. A cell counts again only once
            # the head has left it and come back.
            seen = window.view(self._last_visit_step)
            fresh = window.mask & (seen < state.step - 1)
            window.view(self._visits)[fresh] += 1
            seen[window.mask] = state.step

        # Near the wall, also credit the corresponding unrolled perimeter bin.
        distance, _, _, is_obstacle = self.world.pool.nearest_wall(state.x, state.y)
        if not is_obstacle and distance <= self.robot.radius + 0.15:
            arc = self.world.pool.project_to_perimeter(state.x, state.y)
            index = int(arc / 0.1) % len(self._wall_visits)
            self._wall_visits[index] += 1

    @property
    def visit_grid(self) -> np.ndarray:
        """Per-cell visit counts. Zero where the robot never reached."""
        if self._visits is None:
            raise RuntimeError("no visit grid before reset()")
        return self._visits

    @property
    def wall_visits(self) -> np.ndarray:
        """Per-bin visit counts along the unrolled pool perimeter."""
        if self._wall_visits is None:
            raise RuntimeError("no wall visits before reset()")
        return self._wall_visits


@BACKENDS.register("fast2d")
def _make_fast2d(**kwargs: object) -> Fast2DBackend:
    return Fast2DBackend(**kwargs)  # type: ignore[arg-type]
