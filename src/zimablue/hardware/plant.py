"""The simulator standing in for the robot.

Bring-up order matters on hardware: the day the real machine is in the water
is the wrong day to discover the loop rate, the watchdog thresholds, or the
survey frame were wrong. :class:`SimulatedPlant` puts the simulator behind the
same two interfaces a driver would fill -- a
:class:`~zimablue.hardware.sources.ReadingSource` and an actuator -- so the
*entire* hardware stack (runtime, speed loop, watchdog, recording, and the
controller itself) runs exactly as it will on the robot, against physics you
trust::

    plant = SimulatedPlant(pool="kidney", dirt="autumn", seed=3)
    runtime = HardwareRuntime(
        controller=PathFollower("sweep_optimal"),
        robot=plant.robot,
        source=plant,
        actuate=plant.actuate,
        platform=plant.platform,
        survey=plant.survey(),
    )

Nothing in the runtime knows it is being fooled: the readings arrive noisy and
delayed through the ordinary sensor models, the pose stays unavailable, and
the recording says ``ground_truth: False`` like any other hardware run. The
one honest extra is :meth:`truth_pose`, so a rehearsal can *measure* how far
the estimate drifted -- which is the number the rehearsal exists to produce.
"""

from __future__ import annotations

from typing import Any

from zimablue.controllers.base import ControlInput
from zimablue.hardware.motors import MotorEffort
from zimablue.hardware.runtime import Survey
from zimablue.robot import Cleaner, DriveCommand
from zimablue.sensors import Reading
from zimablue.simulation import Simulation

__all__ = ["SimulatedPlant"]


class _Puppet:
    """The controller that does as the actuator said."""

    name = "plant_puppet"

    def __init__(self) -> None:
        self.command = DriveCommand.stop()
        self.readings: dict[str, Reading] = {}

    def reset(self, robot: Cleaner) -> None:
        self.command = DriveCommand.stop()
        self.readings = {}

    def step(self, control_input: ControlInput) -> DriveCommand:
        self.readings = control_input.readings
        return self.command


class SimulatedPlant:
    """A :class:`~zimablue.simulation.Simulation` behind the hardware port."""

    def __init__(
        self,
        *,
        pool: Any = "rectangular",
        robot: Any = "tracked",
        dirt: Any = "light_sediment",
        seed: int = 0,
        timestep: float = 0.02,
        **simulation_kwargs: Any,
    ) -> None:
        self._puppet = _Puppet()
        self.simulation = Simulation(
            pool=pool,
            robot=robot,
            dirt=dirt,
            controller=self._puppet,
            seed=seed,
            timestep=timestep,
            record=False,
            scenario_name="plant",
            **simulation_kwargs,
        )
        self.robot = self.simulation.robot
        self.timestep = float(timestep)
        self._consumed = 0.0
        # One tick so the first read() has readings to hand out, mirroring
        # what a real bus does: the sensors were already running when the
        # loop started.
        self.simulation.step()
        state = self.simulation.state
        self._start = (float(state.x), float(state.y), float(state.heading))

    # -- the ReadingSource half -------------------------------------------
    @property
    def channels(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(self.robot.sensors[name].channels) for name in self.robot.sensors}

    def read(self, now: float) -> dict[str, Reading]:
        """Advance the physics to ``now`` under the last actuated command."""
        while self._consumed + self.timestep <= now + 1e-9:
            self.simulation.step()
            self._consumed += self.timestep
        return dict(self._puppet.readings)

    def close(self) -> None:
        self.simulation.backend.close()

    # -- the actuator half -------------------------------------------------
    def actuate(self, command: Any) -> None:
        """Accepts what the runtime sends: a drive command, or motor duty."""
        if isinstance(command, MotorEffort):
            limit = self.robot.locomotion.max_speed
            command = DriveCommand(
                left=command.left * limit,
                right=command.right * limit,
                brush=command.brush,
                pump=command.pump,
            )
        self._puppet.command = command

    # -- the robot's own bus -----------------------------------------------
    def platform(self) -> dict[str, float]:
        state = self.simulation.state
        return {
            "battery": float(state.battery_fraction),
            "filter_load": float(state.filter_load),
            "power_w": float(state.power_w),
            "depth": float(state.depth),
        }

    # -- what a person would measure ----------------------------------------
    def survey(self) -> Survey:
        """The survey a careful person would have made: true shape, true start."""
        return Survey(pool=self.simulation.pool, start=self._start)

    def truth_pose(self) -> tuple[float, float, float]:
        """The real pose, for scoring a rehearsal. The runtime never sees it."""
        state = self.simulation.state
        return (float(state.x), float(state.y), float(state.heading))
