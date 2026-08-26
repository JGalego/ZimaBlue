"""The ``.zbr`` recording format.

A recording is a ZIP container.  The design follows MCAP's principles --
self-describing, metadata travels with the data, cheap compression -- without
adopting MCAP itself, because a ZimaBlue run is not a stream of heterogeneous
pub/sub messages.  It is a small fixed set of dense uniformly-sampled numeric
columns plus a few sparse side channels, which is a columnar array file
(``docs/research.md`` section 7).

::

    run.zbr
    |-- manifest.json      format, schema version, ZimaBlue version, seed,
    |                      the full resolved scenario, channel descriptors
    |-- frames.npz         columnar float32/int32 arrays, one per channel
    |-- events.json        sparse events with payloads
    |-- dirt.npz           dirt-field keyframes and debris snapshots
    +-- metrics.json       scalar metrics and spatial grids

Two properties matter more than the layout:

* **Inspectable.** Unzip it and the metadata is plain JSON; ``np.load`` reads
  the arrays without ZimaBlue installed.
* **Self-contained.** The pool geometry and robot configuration are embedded,
  not referenced by preset name, so a recording stays replayable after the
  preset it came from is changed or deleted.

Dirt is keyframed rather than stored per frame: at 50 Hz a 30-minute run is
90 000 frames, and a full dirt raster each time would be gigabytes for a field
that changes slowly. Keyframes every few seconds reproduce it to the eye and to
within a gram.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue._version import __version__

__all__ = ["FORMAT", "SCHEMA_VERSION", "Recorder", "Recording", "build_frame"]

FORMAT = "zbr"
SCHEMA_VERSION = 1
"""Bump when the layout changes incompatibly. Readers check it and refuse
anything newer rather than silently misinterpreting columns."""

FloatArray = NDArray[np.float64]


def _padded_debris(snapshot: NDArray) -> NDArray:
    """A debris snapshot widened to six columns, for files written before six."""
    if snapshot.shape[-1] >= 6:
        return snapshot
    pad = np.zeros((snapshot.shape[0], 6 - snapshot.shape[-1]), dtype=snapshot.dtype)
    return np.concatenate([snapshot, pad], axis=1)


@dataclass
class Recording:
    """A complete, replayable record of one run."""

    manifest: dict[str, Any]
    frames: dict[str, NDArray]
    events: list[dict[str, Any]] = field(default_factory=list)
    dirt_times: NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dirt_keyframes: NDArray = field(default_factory=lambda: np.zeros((0, 0, 0, 0), np.float32))
    debris_keyframes: NDArray = field(default_factory=lambda: np.zeros((0, 0, 6), np.float32))
    metrics: dict[str, Any] = field(default_factory=dict)
    spatial: dict[str, NDArray] = field(default_factory=dict)

    # -- convenience -----------------------------------------------------
    @property
    def n_frames(self) -> int:
        times = self.frames.get("time")
        return 0 if times is None else int(times.size)

    @property
    def duration(self) -> float:
        times = self.frames.get("time")
        return 0.0 if times is None or times.size == 0 else float(times[-1])

    @property
    def frame_dt(self) -> float:
        """Seconds between recorded frames.

        Not the simulation timestep: the recorder may keep every Nth tick.
        Taken from the timestamps rather than the manifest so it stays right
        for a recording written at a decimated rate.
        """
        times = self.frames.get("time")
        if times is None or times.size < 2:
            return float(self.manifest.get("timestep", 0.02))
        return float(times[1] - times[0])

    @property
    def seed(self) -> int:
        return int(self.manifest.get("seed", 0))

    @property
    def channels(self) -> list[str]:
        return sorted(self.frames)

    def column(self, name: str) -> NDArray:
        try:
            return self.frames[name]
        except KeyError:
            raise KeyError(
                f"no channel {name!r} in this recording; available: {self.channels}"
            ) from None

    def frame_index_at(self, t: float) -> int:
        """Index of the last frame at or before time ``t``."""
        times = self.frames["time"]
        if times.size == 0:
            raise ValueError("recording has no frames")
        return int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, times.size - 1))

    def keyframe_span(self, t: float) -> tuple[int, int, float]:
        """The keyframes bracketing ``t``, and how far between them it falls.

        Returns ``(before, after, blend)`` with ``blend`` in ``[0, 1]``.  Past
        the last keyframe both indices are the last one and ``blend`` is 0.
        """
        times = self.dirt_times
        last = len(times) - 1
        if last < 0:
            return (0, 0, 0.0)
        before = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, last))
        after = min(before + 1, last)
        span = float(times[after] - times[before])
        if after == before or span <= 0.0:
            return (before, after, 0.0)
        return (before, after, float(np.clip((t - times[before]) / span, 0.0, 1.0)))

    def dirt_at(self, t: float, *, interpolate: bool = False) -> NDArray:
        """The dirt raster (summed over layers) at time ``t``.

        By default this is the nearest keyframe at or before ``t``: the exact
        field the simulator held at a moment it actually recorded, which is
        what a measurement wants.

        ``interpolate=True`` blends linearly towards the next keyframe instead.
        Keyframes are ten simulated seconds apart, so the exact answer holds
        still and then steps -- one cell can move by half the whole field's
        peak in a single rendered frame, which reads as dirt appearing out of
        nowhere. The blend is an estimate and says so by being opt-in: a cell
        cleaned early in an interval fades over the whole of it rather than
        dropping when it was really cleaned. Every value it returns lies
        between two the cell genuinely held, so it smooths the timing without
        inventing dirt.
        """
        if self.dirt_keyframes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        before, after, blend = self.keyframe_span(t)
        # Keyframes are stored as float16 to keep the file small, and float16
        # is far too coarse to compute in: summing the layers or weighting two
        # of them in that precision loses more than the step being smoothed,
        # and can land the blend below both of the values it sits between.
        field = self.dirt_keyframes[before].sum(axis=0, dtype=np.float32)
        if not interpolate or blend <= 0.0:
            return field
        ahead = self.dirt_keyframes[after].sum(axis=0, dtype=np.float32)
        return (1.0 - blend) * field + blend * ahead

    def debris_at(self, t: float, *, interpolate: bool = False) -> NDArray:
        """Debris snapshot ``(n, 6)`` -- ``x, y, mass, size, collected, type``.

        ``type`` indexes :meth:`debris_type_names`. Recordings written before
        the column existed are padded with zeros rather than rejected, which
        makes every item read as the first type -- wrong in the drawing, but
        the alternative is refusing to open the file at all.

        ``interpolate=True`` glides each item between its bracketing keyframes
        rather than teleporting it, and turns ``collected`` into the *fraction*
        of the way through the interval in which it was collected, so a caller
        can fade an item out over that interval instead of winking it away.
        Anything counting items must therefore stay on the exact reading, and
        the row order is the item's identity in both.
        """
        if self.debris_keyframes.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        before, after, blend = self.keyframe_span(t)
        snapshot = _padded_debris(self.debris_keyframes[before])
        if not interpolate or blend <= 0.0:
            return snapshot
        ahead = _padded_debris(self.debris_keyframes[after])
        if ahead.shape != snapshot.shape:  # pragma: no cover - defensive
            return snapshot
        blended = snapshot.astype(np.float32, copy=True)
        # Position and the collected flag move; mass, size and type do not.
        for column in (0, 1, 4):
            blended[:, column] = (1.0 - blend) * snapshot[:, column].astype(np.float32) + (
                blend * ahead[:, column].astype(np.float32)
            )
        return blended

    def debris_type_names(self) -> list[str]:
        """Names for the ``type`` column, in index order."""
        dirt = self.manifest.get("dirt_types", {})
        names = dirt.get("debris") if isinstance(dirt, dict) else None
        return list(names) if names else ["leaves"]

    def events_between(self, t0: float, t1: float) -> list[dict[str, Any]]:
        return [e for e in self.events if t0 <= e["time"] < t1]

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write this recording to a ``.zbr`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("manifest.json", json.dumps(self.manifest, indent=2, sort_keys=True))
            zf.writestr("events.json", json.dumps(self.events, indent=1))
            zf.writestr("metrics.json", json.dumps(self.metrics, indent=2, sort_keys=True))
            zf.writestr("frames.npz", _npz_bytes(self.frames))
            zf.writestr(
                "dirt.npz",
                _npz_bytes(
                    {
                        "times": self.dirt_times,
                        "field": self.dirt_keyframes,
                        "debris": self.debris_keyframes,
                    }
                ),
            )
            if self.spatial:
                zf.writestr("spatial.npz", _npz_bytes(self.spatial))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Read a ``.zbr`` file, checking the schema version."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no recording at {path}")
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            _check_schema(manifest, path)
            frames = _npz_load(zf.read("frames.npz"))
            events = json.loads(zf.read("events.json"))
            metrics = json.loads(zf.read("metrics.json")) if "metrics.json" in zf.namelist() else {}
            dirt = _npz_load(zf.read("dirt.npz")) if "dirt.npz" in zf.namelist() else {}
            spatial = _npz_load(zf.read("spatial.npz")) if "spatial.npz" in zf.namelist() else {}
        return cls(
            manifest=manifest,
            frames=frames,
            events=events,
            metrics=metrics,
            dirt_times=dirt.get("times", np.zeros(0, dtype=np.float32)),
            dirt_keyframes=dirt.get("field", np.zeros((0, 0, 0, 0), dtype=np.float32)),
            debris_keyframes=dirt.get("debris", np.zeros((0, 0, 6), dtype=np.float32)),
            spatial=spatial,
        )

    def describe(self) -> str:
        """Human-readable summary, used by ``zimablue inspect``."""
        m = self.manifest
        lines = [
            f"format          {m.get('format')} v{m.get('schema_version')}",
            f"zimablue        {m.get('zimablue_version')}",
            f"scenario        {m.get('scenario', {}).get('name', '(unnamed)')}",
            f"seed            {m.get('seed')}",
            f"pool            {m.get('scenario', {}).get('pool', '?')}",
            f"robot           {m.get('scenario', {}).get('robot', '?')}",
            f"dirt            {m.get('scenario', {}).get('dirt', '?')}",
            f"controller      {m.get('scenario', {}).get('controller', '?')}",
        ]
        fleet = m.get("fleet")
        if fleet:
            names = ", ".join(dict.fromkeys(fleet.get("controllers", [])))
            lines.append(f"robots          {fleet.get('count')} ({names})")
            reach = fleet.get("comms_range")
            lines.append(
                f"radio           {'unlimited' if reach is None else f'{reach:g} m'}"
                f"{'' if fleet.get('share', True) else ', not sharing coverage'}"
            )
        lines += [
            f"timestep        {m.get('timestep')} s",
            f"frames          {self.n_frames} ({self.duration / 60:.1f} min)",
            f"channels        {len(self.frames)}",
            f"events          {len(self.events)}",
            f"dirt keyframes  {len(self.dirt_times)}",
        ]
        if not self.has_ground_truth:
            lines.append(f"pose            {m.get('pose_source', 'estimate')} -- no ground truth")
        return "\n".join(lines)

    @property
    def has_ground_truth(self) -> bool:
        """Whether ``x``, ``y`` and ``heading`` are the true pose.

        False for anything recorded off a robot, where they are the
        controller's estimate. Readers that compute coverage or compare against
        a simulated run have to check this: the arithmetic works either way and
        means something entirely different.
        """
        return bool(self.manifest.get("ground_truth", True))


class Recorder:
    """Accumulates frames, events and dirt keyframes during a run.

    Columns are collected as Python lists and converted once at the end: for a
    90 000-frame run that is far cheaper than growing NumPy arrays per tick,
    and it keeps the recorder out of the step budget.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        dirt_keyframe_interval: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.manifest = manifest
        self.dirt_keyframe_interval = dirt_keyframe_interval
        self.enabled = enabled
        self._columns: dict[str, list[float]] = {}
        self._events: list[dict[str, Any]] = []
        self._dirt_times: list[float] = []
        self._dirt_fields: list[NDArray] = []
        self._debris: list[NDArray] = []
        self._last_keyframe = -np.inf
        self._length = 0

    # -- capture -----------------------------------------------------------
    def add_frame(self, values: dict[str, float]) -> None:
        if not self.enabled:
            return
        # Track the frame count explicitly rather than reading it off the
        # "time" column: that column is appended to inside this same loop, so
        # using its length back-filled every later channel by one row and left
        # the whole table off by one against "time".
        index = self._length
        for key, value in values.items():
            column = self._columns.get(key)
            if column is None:
                # A channel appearing late (a sensor's first reading) is
                # back-filled so every column stays the same length.
                fill = 0.0 if key in _INT_CHANNELS else float("nan")
                column = [fill] * index
                self._columns[key] = column
            column.append(float(value))
        self._length += 1

    def add_event(self, event: Any) -> None:
        if self.enabled:
            self._events.append(event.to_dict() if hasattr(event, "to_dict") else dict(event))

    def maybe_keyframe(self, time: float, dirt: Any, *, force: bool = False) -> None:
        """Snapshot the dirt state if enough time has passed."""
        if not self.enabled:
            return
        if not force and time - self._last_keyframe < self.dirt_keyframe_interval:
            return
        self._last_keyframe = time
        self._dirt_times.append(time)
        self._dirt_fields.append(dirt.field.snapshot())
        self._debris.append(dirt.debris.snapshot())

    # -- finish ------------------------------------------------------------
    def finish(
        self,
        metrics: dict[str, Any] | None = None,
        spatial: dict[str, NDArray] | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> Recording:
        frames = {}
        length = self._length
        for key, column in self._columns.items():
            is_int = key in _INT_CHANNELS
            # Pad any column that started late, so the table stays rectangular.
            # Integer channels pad with 0 rather than NaN, which has no int32
            # representation; float channels keep NaN to mark "no data yet".
            if len(column) < length:
                column = column + [0 if is_int else float("nan")] * (length - len(column))
            frames[key] = np.asarray(column, dtype=np.int32 if is_int else np.float32)

        manifest = dict(self.manifest)
        if extra_manifest:
            manifest.update(extra_manifest)
        manifest.setdefault("format", FORMAT)
        manifest.setdefault("schema_version", SCHEMA_VERSION)
        manifest.setdefault("zimablue_version", __version__)
        manifest["channels"] = sorted(frames)
        manifest["n_frames"] = length

        return Recording(
            manifest=manifest,
            frames=frames,
            events=list(self._events),
            dirt_times=np.asarray(self._dirt_times, dtype=np.float32),
            dirt_keyframes=(
                np.stack(self._dirt_fields)
                if self._dirt_fields
                else np.zeros((0, 0, 0, 0), dtype=np.float32)
            ),
            debris_keyframes=(
                np.stack(self._debris)
                if self._debris and self._debris[0].size
                else np.zeros((0, 0, 6), dtype=np.float32)
            ),
            metrics=metrics or {},
            spatial=spatial or {},
        )


_INT_CHANNELS = frozenset({"step", "contacts", "collided", "stuck", "cmd_brush"})
"""Channels that are counts or bitfields, stored as int32 rather than float32."""


def build_frame(
    state: Any,
    command: Any,
    observations: Mapping[str, Any] | None = None,
    channels: Mapping[str, Sequence[str]] | None = None,
    telemetry: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Flatten one tick into the columns a ``.zbr`` stores.

    Shared by the simulator and by :mod:`zimablue.hardware`, which is the whole
    point: a recording written on a robot has to have the same columns as one
    written by the backend, or the replay viewer and every metric downstream of
    it quietly mean something different depending on where the file came from.

    ``state`` only has to have the attributes it is asked for, so a hardware
    runtime can hand over a :class:`~zimablue.backends.base.SimState` filled in
    from telemetry, with NaN in the fields it cannot know.
    """
    contacts = sum(1 << i for i, flag in enumerate(state.contacts) if flag)
    frame: dict[str, float] = {
        "time": state.time,
        "step": state.step,
        "x": state.x,
        "y": state.y,
        "heading": state.heading,
        "v": state.v,
        "omega": state.omega,
        "wheel_left": state.wheel_left,
        "wheel_right": state.wheel_right,
        "slip_left": state.slip_left,
        "slip_right": state.slip_right,
        "depth": state.depth,
        "battery": state.battery_fraction,
        "power": state.power_w,
        "filter_load": state.filter_load,
        "distance": state.distance,
        "dirt_collected": state.dirt_collected,
        "contacts": contacts,
        "collided": 1 if state.collided else 0,
        "stuck": 1 if state.stuck else 0,
        "cmd_left": command.left,
        "cmd_right": command.right,
        "cmd_brush": 1 if command.brush else 0,
        "cmd_pump": command.pump,
    }
    # A controller may publish its own channels -- an estimated pose, a planner
    # phase. Recording them next to ground truth is what lets replay show
    # estimation error rather than merely assert it. On hardware there is no
    # ground truth to show it against, and these are all you have.
    if telemetry:
        for key, value in telemetry.items():
            frame[f"ctl.{key}"] = float(value)

    for name, reading in (observations or {}).items():
        names = (channels or {}).get(name, ())
        for channel, value in zip(names, reading.values, strict=False):
            frame[f"{name}.{channel}"] = float(value)
        frame[f"{name}.valid"] = 1.0 if reading.valid else 0.0
    return frame


def _npz_bytes(arrays: dict[str, NDArray]) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return buffer.getvalue()


def _npz_load(data: bytes) -> dict[str, NDArray]:
    import io

    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return {k: loaded[k] for k in loaded.files}


def _check_schema(manifest: dict[str, Any], path: Path) -> None:
    fmt = manifest.get("format")
    if fmt != FORMAT:
        raise ValueError(f"{path} is not a ZimaBlue recording (format={fmt!r})")
    version = int(manifest.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses .zbr schema v{version}, but this ZimaBlue "
            f"({__version__}) understands up to v{SCHEMA_VERSION}. Upgrade ZimaBlue."
        )
