"""Several cleaners in one pool.

A fleet is not N runs added together. The robots share a dirt field, so the
second one to reach a patch finds it already clean; they get in each other's
way, because two hulls cannot occupy the same water; and whatever they know
about each other they had to tell each other, over a radio, using positions
they estimated themselves.

::

    fleet = zb.Fleet(pool="kidney", robots=3, controllers="bsa")
    result = fleet.run(minutes=20)
    print(result.summary())

Design
------

:class:`Fleet` composes the single-robot machinery rather than replacing it.
Each robot gets its own backend instance, its own sensors and its own
controller; all of them are reset against **one** :class:`~zimablue.world.World`,
which is what makes the dirt shared. Nothing in the single-robot path changed
to make this work, and a :class:`~zimablue.simulation.Simulation` is still the
right thing to reach for when there is one robot.

What is genuinely new is everything between the robots.

**They collide.** Before each tick every backend is told where the others are,
as discs. The collision resolver pushes them apart and the sonar sees them.
A fleet that drove through itself would make every coordination algorithm look
better than it is.

**They talk, badly.** :class:`Blackboard` is a radio, not a god view. A robot
publishes *its own estimate* of where it is and what it has covered, so a fleet
inherits every member's localisation error and then has to coordinate through
it. ``comms_range`` limits who hears whom, and the default is unlimited only
because that is the assumption almost every published algorithm makes.

**They are scored as a team and as individuals.** Team coverage is the union;
overlap is the part two robots both did; balance is how evenly the work fell.
A fleet with excellent coverage and 40% overlap has three robots doing the work
of two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point as ShapelyPoint

from zimablue._version import __version__
from zimablue.backends.base import Event, SimState
from zimablue.backends.fast2d import Fast2DBackend
from zimablue.controllers.base import ControlInput, Controller
from zimablue.dirt import DirtSpec, make_dirt
from zimablue.metrics import Metrics, SpatialMetrics, compute_metrics
from zimablue.pool import DEFAULT_CELL, Pool, make_pool
from zimablue.recording import Recorder, Recording, build_frame
from zimablue.rng import RngTree
from zimablue.robot import Cleaner, DriveCommand, make_robot
from zimablue.world import World

__all__ = ["Blackboard", "Fleet", "FleetMetrics", "FleetResult", "Peer", "spread_poses"]


# ----------------------------------------------------------------------
@dataclass
class Peer:
    """What one robot has told the others about itself."""

    index: int
    x: float
    y: float
    heading: float
    """Where it *believes* it is. Not ground truth."""

    covered: set[tuple[int, int]] = field(default_factory=set)
    """Grid cells it says it has done, in the shared frame."""

    time: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


class Blackboard:
    """The fleet's radio.

    Every robot publishes what it knows; every robot reads what the others
    published, subject to range. Nothing here is ground truth -- the poses are
    the robots' own estimates and the covered sets are in each robot's own
    reckoning of the shared frame, so two robots that have drifted apart will
    disagree about which cell is which. That is not a bug to be fixed in the
    blackboard; it is the thing a real fleet has to survive.

    The shared frame exists because each robot's estimator is started at its
    own place in it. A cleaner dropped at a surveyed point, or one that finds
    itself once at the start, has exactly this much. What it does *not* have is
    a way to stop drifting afterwards.
    """

    def __init__(self, count: int, *, comms_range: float = math.inf, cell: float = 0.3) -> None:
        self.count = count
        self.comms_range = float(comms_range)
        self.cell = cell
        self.posts: dict[int, Peer] = {}
        self.claims: dict[int, tuple[int, int]] = {}
        """Announcements addressed to one robot by another: ``{loser: (taker,
        where)}``. Used by MSTC's handover, where a robot that has finished
        takes the tail of a team-mate's arc and says so rather than asking."""

        self.messages = 0
        self.dropped = 0

    def publish(
        self,
        index: int,
        x: float,
        y: float,
        heading: float,
        *,
        covered: set[tuple[int, int]] | None = None,
        extras: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> None:
        self.posts[index] = Peer(
            index=index,
            x=float(x),
            y=float(y),
            heading=float(heading),
            covered=covered if covered is not None else set(),
            time=time,
            extras=extras or {},
        )
        self.messages += 1

    def peers(self, index: int) -> list[Peer]:
        """Everyone else this robot can hear, from where it thinks it is."""
        mine = self.posts.get(index)
        # A robot that has not published yet has nowhere to measure range from,
        # so it hears everyone; the alternative is going deaf on its first tick.
        here = (mine.x, mine.y) if mine is not None and math.isfinite(self.comms_range) else None
        out = []
        for other, post in self.posts.items():
            if other == index:
                continue
            if here is not None and math.hypot(post.x - here[0], post.y - here[1]) > (
                self.comms_range
            ):
                self.dropped += 1
                continue
            out.append(post)
        return out

    def covered_by_others(self, index: int) -> set[tuple[int, int]]:
        """The union of what everyone in range says they have done."""
        out: set[tuple[int, int]] = set()
        for peer in self.peers(index):
            out |= peer.covered
        return out

    def reset(self) -> None:
        self.posts.clear()
        self.claims.clear()
        self.messages = 0
        self.dropped = 0


# ----------------------------------------------------------------------
@dataclass
class FleetMetrics:
    """How the team did, and how each member did.

    The team numbers are computed over the *union* of what the robots covered
    and the sum of what they spent, so they are directly comparable with a
    single-robot :class:`~zimablue.metrics.Metrics` from the same pool -- which
    is the comparison that decides whether a second robot was worth buying.
    """

    team: Metrics
    robots: list[Metrics]

    overlap: float = 0.0
    """Fraction of the covered floor that more than one robot went over.

    The number a single-robot metric cannot express and the one a fleet lives
    or dies by. Three robots at 90% team coverage with 45% overlap have done
    the work of about two."""

    balance: float = 1.0
    """Shortest robot's distance over the longest robot's. 1 is a fleet
    sharing the work; 0.3 is one robot doing most of it while another circles
    a corner it was assigned."""

    encounters: int = 0
    """Robot-on-robot collisions. Wall collisions are counted separately, in
    each robot's own metrics."""

    messages: int = 0
    dropped_messages: int = 0
    speedup: float = 1.0
    """Team coverage divided by the best single member's. The honest ceiling
    is the robot count; anything near 1 means the fleet is one robot with
    company."""

    def summary(self) -> str:
        lines = [
            f"  robots            {len(self.robots)}",
            f"  team coverage     {self.team.coverage * 100:6.1f} %",
            f"  dirt removed      {self.team.dirt_removed_fraction * 100:6.1f} %",
            f"  overlap           {self.overlap * 100:6.1f} %   "
            f"(floor two or more robots both did)",
            f"  speedup           {self.speedup:6.2f} x   (against the best single member)",
            f"  balance           {self.balance:6.2f}     (shortest run / longest)",
            f"  distance          {self.team.distance_traveled:6.1f} m  (all robots)",
            f"  encounters        {self.encounters:6d}     (robot-on-robot)",
        ]
        if self.dropped_messages:
            lines.append(
                f"  radio             {self.messages} sent, {self.dropped_messages} out of range"
            )
        per = "  ".join(f"r{i} {m.coverage:.0%}" for i, m in enumerate(self.robots))
        lines.append(f"  per robot         {per}")
        return "\n".join(lines)


@dataclass
class FleetResult:
    """Everything a finished fleet run produced."""

    metrics: FleetMetrics
    spatial: SpatialMetrics
    visits_by_robot: list[np.ndarray]
    recording: Recording | None
    world: World
    states: list[SimState]
    events: list[Event] = field(default_factory=list)

    def summary(self) -> str:
        return self.metrics.summary()

    def require_recording(self) -> Recording:
        """The recording, or a clear error saying how to get one."""
        if self.recording is None:
            raise RuntimeError(
                "this fleet run was not recorded; construct it with Fleet(..., record=True)"
            )
        return self.recording

    def save(self, path: str | Path) -> Path:
        return self.require_recording().save(path)

    @property
    def territory(self) -> np.ndarray:
        """Which robot covered each cell, ``-1`` where nobody did.

        Ties go to whoever spent longest there, which for a partitioned fleet
        draws the partition and for a cooperative one draws what actually
        happened rather than what was planned.
        """
        stack = np.stack(self.visits_by_robot)
        best = stack.argmax(axis=0)
        return np.where(stack.max(axis=0) > 0, best, -1)

    @property
    def times_covered(self) -> np.ndarray:
        """How many different robots went over each cell."""
        return np.stack([(visits > 0).astype(np.int32) for visits in self.visits_by_robot]).sum(
            axis=0
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"FleetResult(robots={len(self.states)}, "
            f"coverage={self.metrics.team.coverage:.1%}, "
            f"overlap={self.metrics.overlap:.1%})"
        )


# ----------------------------------------------------------------------
def spread_poses(
    pool: Pool, robot: Cleaner, count: int, *, heading: str = "inward"
) -> list[tuple[float, float, float]]:
    """``count`` well-separated start poses inside the pool.

    Farthest-point sampling over the navigable raster, seeded from the pool's
    own start pose. Deterministic, shape-aware, and it spreads a fleet round a
    kidney rather than lining it up across the bounding box -- which matters
    more than it sounds, because DARP and every other partition-by-position
    method takes these as its input and will happily hand a robot a region on
    the far side of the pool if they are placed badly.
    """
    if count < 1:
        raise ValueError(f"a fleet needs at least one robot, got {count}")
    workspace = pool.navigable.buffer(-robot.radius * 1.5)
    if workspace.is_empty:
        workspace = pool.navigable

    cell = max(pool.navigable.area / 400.0, 0.04) ** 0.5
    grid = pool.grid(cell)
    xs, ys = grid.cell_centers()
    from shapely import contains_xy

    inside = np.asarray(contains_xy(workspace, xs, ys))
    candidates = np.column_stack([np.asarray(xs)[inside], np.asarray(ys)[inside]])
    if len(candidates) < count:
        raise ValueError(
            f"{pool.name} is too small for {count} robots of this size: "
            f"only {len(candidates)} placements fit"
        )

    first = np.array(pool.start_pose(clearance=robot.radius)[:2])
    chosen = [candidates[int(np.argmin(np.hypot(*(candidates - first).T)))]]
    gaps = np.hypot(*(candidates - chosen[0]).T)
    while len(chosen) < count:
        pick = int(np.argmax(gaps))
        chosen.append(candidates[pick])
        gaps = np.minimum(gaps, np.hypot(*(candidates - candidates[pick]).T))

    centre = np.array([workspace.centroid.x, workspace.centroid.y])
    poses = []
    for point in chosen:
        if heading == "inward":
            angle = float(np.arctan2(centre[1] - point[1], centre[0] - point[0]))
        else:
            angle = 0.0
        poses.append((float(point[0]), float(point[1]), angle))
    return poses


# ----------------------------------------------------------------------
class Fleet:
    """Runs several robots, in one pool, sharing one dirt field."""

    def __init__(
        self,
        pool: Pool | str = "rectangular",
        robots: int | Cleaner | str | list[Cleaner | str] = 2,
        *,
        dirt: DirtSpec | str = "light_sediment",
        controllers: Any = "bsa",
        seed: int = 0,
        timestep: float = 0.02,
        backend: Any = "fast2d",
        cell: float = DEFAULT_CELL,
        record: bool = True,
        dirt_keyframe_interval: float = 10.0,
        expose_truth: bool = False,
        comms_range: float = math.inf,
        share: bool = True,
        start_poses: list[tuple[float, float, float]] | None = None,
        coverage_target: float | None = None,
        dirt_target: float | None = None,
        stop_on_empty_battery: bool = True,
        scenario_name: str = "adhoc",
    ) -> None:
        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        self.timestep = timestep
        self.seed = seed
        self.rng = RngTree(seed)
        self.expose_truth = expose_truth
        self.coverage_target = coverage_target
        self.dirt_target = dirt_target
        self.stop_on_empty_battery = stop_on_empty_battery
        self.scenario_name = scenario_name
        self.share = share

        self.pool = make_pool(pool) if isinstance(pool, str) else pool
        self.robots = _make_robots(robots)
        self.count = len(self.robots)
        self.dirt_spec = make_dirt(dirt) if isinstance(dirt, str) else dirt

        self.world = World.build(self.pool, self.dirt_spec, self.rng.stream("dirt"), cell)
        self._initial_dirt = self.world.dirt.field.total_grid().copy()

        self.start_poses = start_poses or spread_poses(self.pool, self.robots[0], self.count)
        if len(self.start_poses) != self.count:
            raise ValueError(f"{len(self.start_poses)} start poses for {self.count} robots")
        for x, y, _ in self.start_poses:
            if not self.pool.navigable.contains(ShapelyPoint(x, y)):
                raise ValueError(f"start pose ({x:.2f}, {y:.2f}) is outside the navigable pool")

        self.blackboard = Blackboard(self.count, comms_range=comms_range)
        self.controllers = _make_controllers(
            controllers, self.count, self.pool, self.robots, self.start_poses
        )

        self.backends = []
        self.states: list[SimState] = []
        for index, robot in enumerate(self.robots):
            engine = _make_backend(backend)
            # One RNG stream per robot, named by index: adding a fourth robot
            # must not shift the noise the first three see.
            state = engine.reset(self.world, robot, self.rng.branch(f"robot:{index}"))
            x, y, heading = self.start_poses[index]
            state.x, state.y, state.heading = float(x), float(y), float(heading)
            state.depth = float(self.pool.depth_at(x, y))
            self.backends.append(engine)
            self.states.append(state)

        # Same courtesy the scenario runner does: a controller that says it
        # needs ground truth gets it, rather than raising on its first tick.
        if any(getattr(c, "needs_truth", False) for c in self.controllers):
            self.expose_truth = True

        for index, controller in enumerate(self.controllers):
            attach = getattr(controller, "attach_fleet", None)
            if attach is not None:
                attach(
                    index=index,
                    blackboard=self.blackboard,
                    origin=self.start_poses[index],
                    fleet_size=self.count,
                    share=self.share,
                )
            controller.reset(self.robots[index])

        self.events: list[Event] = []
        self._encounters = 0
        self._finished = False
        self._termination = "running"
        self.time = 0.0

        self.recorder = Recorder(
            self._manifest(),
            dirt_keyframe_interval=dirt_keyframe_interval,
            enabled=record,
        )
        self.recorder.maybe_keyframe(0.0, self.world.dirt, force=True)

    # ------------------------------------------------------------------
    def _manifest(self) -> dict[str, Any]:
        return {
            "zimablue_version": __version__,
            "seed": self.seed,
            "timestep": self.timestep,
            "cell": self.world.cell,
            "backend": getattr(self.backends[0], "name", "unknown") if self.backends else "none",
            "fleet": {
                "count": self.count,
                "controllers": [getattr(c, "name", "custom") for c in self.controllers],
                "start_poses": [list(p) for p in self.start_poses],
                "comms_range": (
                    None if math.isinf(self.blackboard.comms_range) else self.blackboard.comms_range
                ),
                "share": self.share,
            },
            "scenario": {
                "name": self.scenario_name,
                "pool": self.pool.name,
                "robot": self.robots[0].name,
                "dirt": self.dirt_spec.name,
                "controller": getattr(self.controllers[0], "name", "custom"),
            },
            "pool_config": self.pool.to_dict(),
            "robot_config": self.robots[0].to_dict(),
            "dirt_config": self.dirt_spec.to_dict(),
            "dirt_types": {
                "layers": self.world.dirt.field.layer_names(),
                "debris": self.world.dirt.debris.type_names(),
            },
        }

    # ------------------------------------------------------------------
    def step(self) -> list[SimState]:
        """Advance every robot one timestep.

        Sense all, decide all, then move all. Not sense-decide-move per robot
        in turn: that would let robot 1 react to robot 0's *new* position
        within the same tick, which is a subtle turn-order advantage that gets
        larger as the fleet does, and it would make the result depend on the
        order the robots happen to be listed in.
        """
        dt = self.timestep
        radii = [robot.radius for robot in self.robots]

        for index, engine in enumerate(self.backends):
            engine.neighbours = tuple(
                (self.states[other].x, self.states[other].y, radii[other])
                for other in range(self.count)
                if other != index
            )

        observations = [
            engine.sense(self.states[index]) for index, engine in enumerate(self.backends)
        ]

        commands: list[DriveCommand] = []
        for index, controller in enumerate(self.controllers):
            state = self.states[index]
            control_input = ControlInput(
                time=state.time,
                dt=dt,
                readings=observations[index],
                battery=state.battery_fraction,
                filter_load=state.filter_load,
                robot=self.robots[index],
                truth=self._truth_view(index) if self.expose_truth else None,
                extras={
                    "stuck": 1.0 if state.stuck else 0.0,
                    "collided": 1.0 if state.collided else 0.0,
                    "robot_index": float(index),
                    "fleet_size": float(self.count),
                },
            )
            commands.append(controller.step(control_input))

        results = [
            engine.step(self.states[index], commands[index], dt)
            for index, engine in enumerate(self.backends)
        ]
        for index, result in enumerate(results):
            self.states[index] = result.state
            for event in result.events:
                if event.kind == "collision" and event.detail.get("robot"):
                    self._encounters += 1
                tagged = Event(event.time, event.kind, {**event.detail, "robot": index})
                self.events.append(tagged)
                self.recorder.add_event(tagged)

        self.time = self.states[0].time
        self._record(commands, observations)
        self.recorder.maybe_keyframe(self.time, self.world.dirt)
        return self.states

    def _truth_view(self, index: int) -> SimState:
        state = self.states[index]
        state.pool = self.pool
        state.world = self.world
        return state

    def _record(self, commands: list[DriveCommand], observations: list[dict[str, Any]]) -> None:
        """One frame, with every robot's channels in it.

        Robot 0's channels appear twice: once prefixed as ``r0.x`` and once
        flat as ``x``. The duplication is deliberate and cheap. Every tool in
        the package -- the replay window, the dirt cam, the dynamics module,
        the planner comparison -- reads the flat names, and aliasing them to
        the first robot means all of it works on a fleet recording, following
        one member, instead of refusing to open it.
        """
        if not self.recorder.enabled:
            return
        frame: dict[str, float] = {}
        for index, controller in enumerate(self.controllers):
            telemetry = getattr(controller, "telemetry", None)
            one = build_frame(
                self.states[index],
                commands[index],
                observations[index],
                {name: sensor.channels for name, sensor in self.robots[index].sensors.items()},
                telemetry() if telemetry is not None else None,
            )
            if index == 0:
                frame.update(one)
            frame.update({f"r{index}.{key}": value for key, value in one.items()})
        self.recorder.add_frame(frame)

    # ------------------------------------------------------------------
    def run(
        self,
        minutes: float | None = None,
        *,
        seconds: float | None = None,
        max_steps: int | None = None,
        progress: Any = None,
    ) -> FleetResult:
        if minutes is None and seconds is None:
            seconds = 1800.0
        duration = float(seconds if seconds is not None else (minutes or 0) * 60.0)
        if duration <= 0:
            raise ValueError("run duration must be positive")

        for controller in self.controllers:
            if hasattr(controller, "run_duration"):
                controller.run_duration = duration

        steps = int(np.ceil(duration / self.timestep))
        if max_steps is not None:
            steps = min(steps, max_steps)

        self._termination = "duration"
        for i in range(steps):
            self.step()
            if progress is not None and i % 250 == 0:
                progress(self.time, duration)
            reason = self._check_termination()
            if reason is not None:
                self._termination = reason
                break
        return self.finish()

    def _check_termination(self) -> str | None:
        if self.stop_on_empty_battery and all(
            state.battery_fraction <= robot.power.battery.cutoff
            for state, robot in zip(self.states, self.robots, strict=True)
        ):
            # Every robot flat, not the first: a fleet keeps working while any
            # member has charge, which is half the reason to have one.
            return "battery_empty"
        if self.dirt_target is not None and self.world.dirt.removed_fraction >= self.dirt_target:
            return "target_reached"
        if self.coverage_target is not None:
            navigable = self.pool.navigable_mask(self.world.cell)
            total = int(navigable.sum())
            covered = navigable & (self._union_visits() > 0)
            if total and float(covered.sum()) / total >= self.coverage_target:
                return "target_reached"
        return None

    def _union_visits(self) -> np.ndarray:
        return sum(engine.visit_grid for engine in self.backends)

    # ------------------------------------------------------------------
    def finish(self) -> FleetResult:
        if self._finished:
            raise RuntimeError("this fleet run has already been finished")
        self._finished = True
        self.recorder.maybe_keyframe(self.time, self.world.dirt, force=True)

        visits_by_robot = [np.asarray(engine.visit_grid).copy() for engine in self.backends]
        # np.stack rather than sum(): a bare sum() starts at the integer 0, so
        # the result is only an array by luck of there being something to add.
        union = np.stack(visits_by_robot).sum(axis=0).astype(np.int32)
        wall_visits = (
            np.stack([np.asarray(engine.wall_visits) for engine in self.backends])
            .sum(axis=0)
            .astype(np.int32)
        )

        # The team's state: robot 0's clock and consumables, with the odometers
        # summed. compute_metrics reads a single state for distance, energy and
        # runtime, and for a fleet those are fleet totals -- assembling the
        # struct here is more honest than teaching the metrics layer to take a
        # list and quietly picking one member's clock anyway.
        team_state = self.states[0].copy()
        team_state.distance = float(sum(s.distance for s in self.states))
        team_state.energy_used_wh = float(sum(s.energy_used_wh for s in self.states))
        team_state.dirt_collected = float(sum(s.dirt_collected for s in self.states))
        team_state.dirt_removed = float(sum(s.dirt_removed for s in self.states))
        team_state.stuck_time = float(sum(s.stuck_time for s in self.states))

        team, spatial = compute_metrics(
            self.world,
            team_state,
            self.events,
            union,
            wall_visits,
            self._initial_dirt,
            termination=self._termination,
            robot=self.robots[0],
        )

        navigable = spatial.navigable
        per_robot: list[Metrics] = []
        for index, visits in enumerate(visits_by_robot):
            one, _ = compute_metrics(
                self.world,
                self.states[index],
                [e for e in self.events if e.detail.get("robot") == index],
                visits,
                np.asarray(self.backends[index].wall_visits),
                self._initial_dirt,
                termination=self._termination,
                robot=self.robots[index],
            )
            per_robot.append(one)

        touched = sum((visits > 0).astype(np.int16) for visits in visits_by_robot)
        covered_cells = int((navigable & (union > 0)).sum())
        shared = int((navigable & (touched > 1)).sum())
        distances = [s.distance for s in self.states]
        best_single = max((m.coverage for m in per_robot), default=0.0)

        metrics = FleetMetrics(
            team=team,
            robots=per_robot,
            overlap=shared / covered_cells if covered_cells else 0.0,
            balance=(min(distances) / max(distances)) if max(distances) > 0 else 1.0,
            encounters=self._encounters,
            messages=self.blackboard.messages,
            dropped_messages=self.blackboard.dropped,
            speedup=(team.coverage / best_single) if best_single > 0 else 1.0,
        )

        recording = None
        if self.recorder.enabled:
            recording = self.recorder.finish(
                metrics=team.to_dict(),
                spatial={
                    "visits": spatial.visits,
                    "remaining_dirt": spatial.remaining_dirt.astype(np.float32),
                    "initial_dirt": spatial.initial_dirt.astype(np.float32),
                    "wall_visits": spatial.wall_visits,
                    "navigable": spatial.navigable,
                    "territory": np.where(
                        np.stack(visits_by_robot).max(axis=0) > 0,
                        np.stack(visits_by_robot).argmax(axis=0),
                        -1,
                    ).astype(np.int16),
                },
            )
        for engine in self.backends:
            engine.close()

        return FleetResult(
            metrics=metrics,
            spatial=spatial,
            visits_by_robot=visits_by_robot,
            recording=recording,
            world=self.world,
            states=list(self.states),
            events=list(self.events),
        )


# ----------------------------------------------------------------------
def _make_robots(robots: Any) -> list[Cleaner]:
    if isinstance(robots, int):
        if robots < 1:
            raise ValueError(f"a fleet needs at least one robot, got {robots}")
        # Separate instances even when identical: a Cleaner is configuration
        # and sensors carry per-instance filter state.
        return [make_robot("tracked") for _ in range(robots)]
    if isinstance(robots, str | Cleaner):
        return [make_robot(robots) if isinstance(robots, str) else robots]
    return [make_robot(r) if isinstance(r, str) else r for r in robots]


def _make_controllers(
    controllers: Any,
    count: int,
    pool: Pool,
    robots: list[Cleaner],
    poses: list[tuple[float, float, float]],
) -> list[Controller]:
    """One controller per robot, and never the same object twice.

    Passing a name builds ``count`` independent instances. Passing a list uses
    it as given -- a heterogeneous fleet is a legitimate experiment, and
    sometimes the interesting one. Passing a single already-built controller
    for a fleet of several is refused: a controller holds a map, an estimator
    and a plan, and sharing one between robots produces a very confident
    machine that is wrong about everything.

    Passing a *callable* hands it the pool, the robots and the start poses and
    lets it build the team. That is how the partitioners arrive: DARP and its
    relatives divide the pool by where the robots are standing, so they cannot
    be constructed until the fleet has placed them.
    """
    from zimablue.controllers.base import CONTROLLERS

    if callable(controllers) and not hasattr(controllers, "step"):
        built = list(controllers(pool, robots, poses))
        if len(built) != count:
            raise ValueError(
                f"{getattr(controllers, 'name', 'the controller factory')} returned "
                f"{len(built)} controllers for {count} robots"
            )
        return built
    if isinstance(controllers, str):
        return [CONTROLLERS.create(controllers) for _ in range(count)]
    if isinstance(controllers, list | tuple):
        if len(controllers) != count:
            raise ValueError(f"{len(controllers)} controllers for {count} robots")
        built = [CONTROLLERS.create(c) if isinstance(c, str) else c for c in controllers]
        if len({id(c) for c in built}) != count:
            raise ValueError("each robot needs its own controller instance, not a shared one")
        return built
    if count == 1:
        return [controllers]
    raise ValueError(
        "pass a controller name, or a list of one per robot. A single controller "
        "object cannot drive a fleet: it holds one map, one estimator and one plan."
    )


def _make_backend(backend: Any) -> Any:
    if isinstance(backend, str):
        if backend == "fast2d":
            return Fast2DBackend()
        from zimablue.backends.base import BACKENDS

        return BACKENDS.create(backend)
    if isinstance(backend, type):
        return backend()
    return backend
