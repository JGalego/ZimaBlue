"""A named collection of sensors.

Exists so that user code reads the way the domain does::

    robot.sensors.sonar.inject_fault(bias=0.15, dropout_probability=0.02)

rather than ``robot.sensors["sonar"]``.  Both work; the attribute form is the
one that appears in examples.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from zimablue.sensors.base import Reading, Sensor, SensorConfig, SensorContext
from zimablue.sensors.models import IMU, ContactSensor, Encoder, PressureSensor, Sonar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.rng import RngTree

__all__ = ["SENSOR_CLASSES", "SensorSuite", "sensor_from_spec"]

SENSOR_CLASSES: dict[str, type[Sensor]] = {
    "Encoder": Encoder,
    "IMU": IMU,
    "PressureSensor": PressureSensor,
    "ContactSensor": ContactSensor,
    "Sonar": Sonar,
}


def sensor_from_spec(spec: dict[str, Any]) -> Sensor:
    """Rebuild a sensor from the dict produced by :meth:`Sensor.spec`.

    Used when replaying: a recording carries the sensor configuration so the run
    can be reconstructed without the original script.
    """
    class_name = spec["class"]
    try:
        cls = SENSOR_CLASSES[class_name]
    except KeyError:
        raise KeyError(
            f"unknown sensor class {class_name!r}; "
            f"register it in SENSOR_CLASSES (known: {sorted(SENSOR_CLASSES)})"
        ) from None

    params = dict(spec.get("params", {}))
    if "beam_angles" in params:
        params["beam_angles"] = tuple(params["beam_angles"])
    sensor = cls(
        name=spec["name"],
        config=SensorConfig.from_dict(spec["config"]),
        **params,
    )
    for fault in spec.get("faults", []):
        sensor.inject_fault(**fault)
    return sensor


class SensorSuite(Mapping[str, Sensor]):
    """Immutable mapping of sensor name to sensor, with attribute access."""

    def __init__(self, sensors: list[Sensor] | tuple[Sensor, ...] = ()) -> None:
        self._sensors: dict[str, Sensor] = {}
        for sensor in sensors:
            if sensor.name in self._sensors:
                raise ValueError(
                    f"duplicate sensor name {sensor.name!r}; "
                    "give each sensor a unique name so faults can target one of them"
                )
            self._sensors[sensor.name] = sensor

    # -- Mapping protocol -----------------------------------------------
    def __getitem__(self, key: str) -> Sensor:
        return self._sensors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sensors)

    def __len__(self) -> int:
        return len(self._sensors)

    def __getattr__(self, name: str) -> Sensor:
        # Only reached when normal attribute lookup fails, so it cannot shadow
        # real attributes.
        try:
            return self.__dict__["_sensors"][name]
        except KeyError:
            raise AttributeError(
                f"no sensor named {name!r}; this robot has: "
                f"{', '.join(self.__dict__['_sensors']) or '(none)'}"
            ) from None

    # -- lifecycle --------------------------------------------------------
    def attach(self, rng: RngTree) -> None:
        for sensor in self._sensors.values():
            sensor.attach(rng)

    def reset(self) -> None:
        for sensor in self._sensors.values():
            sensor.reset()

    def update(self, ctx: SensorContext) -> dict[str, Reading]:
        """Poll every sensor; omit ones that have not produced a first reading."""
        out: dict[str, Reading] = {}
        for name, sensor in self._sensors.items():
            reading = sensor.update(ctx)
            if reading is not None:
                out[name] = reading
        return out

    def clear_faults(self) -> None:
        for sensor in self._sensors.values():
            sensor.clear_faults()

    def specs(self) -> list[dict[str, Any]]:
        return [sensor.spec() for sensor in self._sensors.values()]

    @classmethod
    def from_specs(cls, specs: list[dict[str, Any]]) -> SensorSuite:
        return cls([sensor_from_spec(s) for s in specs])

    def channel_names(self) -> list[str]:
        """Flat ``sensor.channel`` names, in the order readings are recorded."""
        return [f"{name}.{ch}" for name, s in self._sensors.items() for ch in s.channels]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SensorSuite({list(self._sensors)})"
