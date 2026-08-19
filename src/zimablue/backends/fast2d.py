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

from typing import TYPE_CHECKING, Any

import numpy as np

from zimablue.backends.base import BACKENDS, Event, SimState, StepResult
from zimablue.physics.cleaning import apply_cleaning
from zimablue.physics.collision import resolve
from zimablue.physics.kinematics import exact_arc_step, slip_factors
from zimablue.pool.features import Skimmer
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

# The wall, for robots that can hold onto it. Pressing the wall for this long
# starts a climb; the excursion is kinematic -- up, a hold at the waterline,
# back down -- because modelling adhesion hydrodynamics would be a backend of
# its own. Bands split the wall strip by height: the cove a floor robot
# brushes anyway, the mid wall, and the waterline where the scum lives.
CLIMB_GRAB_TIME = 1.0
CLIMB_HOLD = 2.5
CLIMB_SPEED_FACTOR = 0.4
COVE_BAND = 0.30
WATERLINE_BAND = 0.15
"""Simulated seconds between dirt advection updates. The flow field is slow and
smooth; recomputing it every 20 ms would cost time and change nothing."""


class Fast2DBackend:
    """Deterministic planar simulation of a cleaner in a pool."""

    name = "fast2d"

    def __init__(
        self,
        *,
        enable_dirt_drift: bool = True,
        wake_strength: float = 0.35,
        drift_spread: float = 0.12,
    ) -> None:
        self.enable_dirt_drift = enable_dirt_drift
        self.wake_strength = wake_strength
        """How hard the hull's wake stirs whatever it drives over, 0-1."""
        self.drift_spread = drift_spread
        """Diffusion per drift step; see :meth:`DirtField.drift`."""
        self.world: World | None = None
        self.neighbours: tuple[tuple[float, float, float], ...] = ()
        """Other robots this tick, as ``(x, y, radius)`` discs.

        Set by :class:`~zimablue.fleet.Fleet` before each tick and empty
        otherwise, so a single-robot run pays nothing for the fleet's
        existence. Feeding it through the backend rather than the world is
        deliberate: it changes every tick, and the world does not."""
        self.robot: Cleaner | None = None
        self._stir_cells: np.ndarray | None = None
        self._rng: RngTree | None = None
        self._slip_rng: np.random.Generator | None = None
        self._last_drift = 0.0
        self._visits: np.ndarray | None = None
        self._last_visit_step: np.ndarray | None = None
        self._wall_visits: np.ndarray | None = None
        self._wall_bands: np.ndarray | None = None
        self._climb: dict[str, Any] | None = None
        self._pressing_since: float | None = None
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
        self._stir_cells = None
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
        # The same bins, split by height: cove, mid wall, waterline.
        self._wall_bands = np.zeros((len(self._wall_visits), 3), dtype=np.int32)
        self._climb = None
        self._pressing_since = None

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
        if self._climb is not None:
            return self._climb_step(state, command, dt)
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

        # 5b. The wall. A grip-capable robot pressing forward against it for
        # long enough grabs on and climbs; everything else just bumps.
        if robot.locomotion.wall_grip:
            pressing = (
                contact.touching
                and bool(contact.flags[0])
                and not contact.is_obstacle
                and not contact.is_robot
                and v_cmd > 0.05
            )
            if not pressing:
                self._pressing_since = None
            elif self._pressing_since is None:
                self._pressing_since = state.time
            elif state.time - self._pressing_since >= CLIMB_GRAB_TIME:
                assert self._wall_visits is not None
                arc = pool.project_to_perimeter(nx, ny)
                self._climb = {
                    "phase": "up",
                    "height": 0.0,
                    "x": nx,
                    "y": ny,
                    "floor_depth": float(pool.depth_at(nx, ny)),
                    "bin": int(arc / 0.1) % len(self._wall_visits),
                    "hold_left": CLIMB_HOLD,
                }
                self._pressing_since = None
                events.append(
                    Event(new.time, "climb_started", {"x": nx, "y": ny, "arc": float(arc)})
                )

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

        # 8. Environment: everything the pool does to itself, on one cadence.
        if self.enable_dirt_drift and new.time - self._last_drift >= DRIFT_INTERVAL:
            elapsed = new.time - self._last_drift
            world.dirt.field.deposit(elapsed)
            self._stir(new.time, elapsed)
            vx, vy = pool.flow_grid(world.cell)
            world.dirt.field.drift(vx, vy, elapsed, spread=self.drift_spread)
            # The robot's own wake stirs whatever it is currently sitting on.
            # Batched here rather than per tick: over 20 ms the swath barely
            # moves, so diffusing that often costs a third of the step budget
            # and changes nothing.
            wake = pool.grid(world.cell).window(nx, ny, 0.5 * robot.swath_width)
            if wake is not None:
                world.dirt.field.disturb_window(wake, strength=self.wake_strength)
            events.extend(self._surface(new.time, elapsed))
            self._last_drift = new.time

        self._mark_visit(new)
        return StepResult(state=new, events=events)

    # ------------------------------------------------------------------
    def _stir(self, now: float, elapsed: float) -> None:
        """A swimmer pushes off somewhere, and the loose dirt there lifts.

        Timed on the dirt spec's interval, placed by a seeded stream, so a
        pool party is as reproducible as everything else.
        """
        assert self.world is not None and self._rng is not None
        interval = self.world.dirt.stir_interval
        if interval <= 0:
            return
        due = int(now / interval) - int((now - elapsed) / interval)
        if due <= 0:
            return
        pool = self.world.pool
        grid = pool.grid(self.world.cell)
        if self._stir_cells is None:
            navigable = pool.navigable_mask(self.world.cell)
            self._stir_cells = np.argwhere(navigable)
        if not len(self._stir_cells):
            return
        stream = self._rng.stream("environment:stir")
        for _ in range(due):
            row, col = self._stir_cells[int(stream.integers(len(self._stir_cells)))]
            x = grid.minx + (col + 0.5) * grid.cell
            y = grid.miny + (row + 0.5) * grid.cell
            window = grid.window(float(x), float(y), 0.75)
            if window is not None:
                self.world.dirt.field.disturb_window(window, strength=self.world.dirt.stir_strength)

    def _surface(self, now: float, elapsed: float) -> list[Event]:
        """What happens on the water while the robot works the floor.

        Whatever floats rides the return-jet circulation, and a skimmer takes
        what drifts into its reach. Both are the pool acting on its own: the
        skimmed items are marked apart from the robot's catch, so the metrics
        can say who did the work.
        """
        assert self.world is not None
        pool = self.world.pool
        debris = self.world.dirt.debris
        events: list[Event] = []
        if not len(debris):
            return events

        moving = debris.active & debris.buoyant
        if moving.any():
            vx, vy = pool.flow_grid(self.world.cell)
            grid = pool.grid(self.world.cell)
            rows = np.clip(((debris.y - grid.miny) / grid.cell).astype(int), 0, grid.nrows - 1)
            cols = np.clip(((debris.x - grid.minx) / grid.cell).astype(int), 0, grid.ncols - 1)
            navigable = pool.navigable_mask(self.world.cell)

            def inside(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
                r = np.clip(((ys - grid.miny) / grid.cell).astype(int), 0, grid.nrows - 1)
                c = np.clip(((xs - grid.minx) / grid.cell).astype(int), 0, grid.ncols - 1)
                return navigable[r, c]

            debris.advect(vx[rows, cols] * elapsed, vy[rows, cols] * elapsed, inside)

        for feature in pool.features:
            if not isinstance(feature, Skimmer):
                continue
            sx, sy = feature.position
            caught = debris.near(sx, sy, feature.capture_radius) & debris.buoyant
            mass, count = debris.skim(caught)
            if count:
                events.append(
                    Event(now, "skimmed", {"count": count, "grams": mass, "skimmer": feature.name})
                )
        return events

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
        # A floor robot's brush reaches the cove -- the bottom band of the
        # wall strip -- and nothing above it.
        distance, _, _, is_obstacle = self.world.pool.nearest_wall(state.x, state.y)
        if not is_obstacle and distance <= self.robot.radius + 0.15:
            arc = self.world.pool.project_to_perimeter(state.x, state.y)
            index = int(arc / 0.1) % len(self._wall_visits)
            self._wall_visits[index] += 1
            assert self._wall_bands is not None
            self._wall_bands[index, 0] += 1

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

    @property
    def wall_band_visits(self) -> np.ndarray:
        """The wall strip: perimeter bins by height band (cove, mid, waterline)."""
        if self._wall_bands is None:
            raise RuntimeError("no wall strip before reset()")
        return self._wall_bands

    def _climb_step(self, state: SimState, command: DriveCommand, dt: float) -> StepResult:
        """One tick of the wall excursion: up, hold at the waterline, down.

        Kinematic on purpose. The dynamics of staying stuck to a wall are a
        hydro problem this backend does not claim; what it does claim is the
        time, the energy, and the wall area credited -- the quantities the
        metrics read. The robot is committed: drive commands are ignored
        until it is back on the floor, the way the real machines transition.
        The encoders keep reading track motion during the climb, so dead
        reckoning pays for the trip -- a wall is exactly where odometry goes
        to get worse, and hiding that would flatter every wall-cleaning run.
        """
        assert self.world is not None and self.robot is not None
        assert self._climb is not None and self._wall_bands is not None
        assert self._wall_visits is not None
        robot = self.robot
        climb = self._climb
        events: list[Event] = []

        v_climb = CLIMB_SPEED_FACTOR * robot.locomotion.max_speed
        moved = v_climb * dt
        depth = climb["floor_depth"]

        if climb["phase"] == "up":
            climb["height"] = min(climb["height"] + moved, depth)
            if climb["height"] >= depth - 1e-9:
                climb["phase"] = "hold"
                events.append(
                    Event(state.time + dt, "climb_topped", {"x": climb["x"], "y": climb["y"]})
                )
        elif climb["phase"] == "hold":
            moved = 0.0
            climb["hold_left"] -= dt
            if climb["hold_left"] <= 0.0:
                climb["phase"] = "down"
        else:
            climb["height"] = max(climb["height"] - moved, 0.0)

        new = state.copy()
        new.time = state.time + dt
        new.step = state.step + 1
        new.x, new.y = climb["x"], climb["y"]
        new.v = v_climb if climb["phase"] != "hold" else 0.0
        new.omega = 0.0
        new.accel_forward = 0.0
        new.accel_lateral = 0.0
        wheel = v_climb if climb["phase"] != "hold" else 0.0
        new.wheel_left = new.wheel_right = wheel
        new.slip_left = new.slip_right = 0.0
        new.depth = max(depth - climb["height"], 0.0)
        new.contacts = (True, False, False, False)
        new.collided = True
        new.distance = state.distance + moved
        new.stuck = False
        new.stuck_time = 0.0
        self._prev_v, self._prev_omega = new.v, 0.0

        # Credit the band the hull is passing over. Tick-counted: coverage
        # reads it as visited-or-not, and a slow pass *is* more cleaning.
        height = climb["height"]
        if height >= depth - WATERLINE_BAND:
            band = 2
        elif height <= COVE_BAND:
            band = 0
        else:
            band = 1
        self._wall_bands[climb["bin"], band] += 1
        self._wall_visits[climb["bin"]] += 1

        # Power: same draw model, with the climb's own load -- the full
        # submerged weight hangs on the tracks, not half of it.
        load = robot.chassis.submerged_weight * 0.8
        power = robot.power_draw(
            wheel, wheel, load, load, brush_on=command.brush, pump_duty=command.pump
        )
        used = power * dt / 3600.0
        new.power_w = power
        new.battery_wh = max(0.0, state.battery_wh - used)
        new.energy_used_wh = state.energy_used_wh + used
        new._battery_fraction = new.battery_wh / max(new._battery_capacity, 1e-9)
        if new._battery_fraction <= robot.power.battery.cutoff and not self._warned_battery:
            self._warned_battery = True
            events.append(Event(new.time, "battery_empty", {"fraction": new._battery_fraction}))

        if climb["phase"] == "down" and climb["height"] <= 0.0:
            self._climb = None
            events.append(Event(new.time, "climb_ended", {"x": new.x, "y": new.y}))
        return StepResult(state=new, events=events)


@BACKENDS.register("fast2d")
def _make_fast2d(**kwargs: object) -> Fast2DBackend:
    return Fast2DBackend(**kwargs)  # type: ignore[arg-type]
