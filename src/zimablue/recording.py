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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue._version import __version__

__all__ = ["FORMAT", "SCHEMA_VERSION", "Recorder", "Recording"]

FORMAT = "zbr"
SCHEMA_VERSION = 1
"""Bump when the layout changes incompatibly. Readers check it and refuse
anything newer rather than silently misinterpreting columns."""

FloatArray = NDArray[np.float64]


@dataclass
class Recording:
    """A complete, replayable record of one run."""

    manifest: dict[str, Any]
    frames: dict[str, NDArray]
    events: list[dict[str, Any]] = field(default_factory=list)
    dirt_times: NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dirt_keyframes: NDArray = field(default_factory=lambda: np.zeros((0, 0, 0, 0), np.float32))
    debris_keyframes: NDArray = field(default_factory=lambda: np.zeros((0, 0, 5), np.float32))
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
        return int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, times.size - 1))

    def dirt_at(self, t: float) -> NDArray:
        """The dirt raster (summed over layers) at time ``t``.

        Nearest keyframe at or before ``t`` -- no interpolation, because
        interpolating between mass fields would invent dirt that never existed.
        """
        if self.dirt_keyframes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        idx = int(
            np.clip(
                np.searchsorted(self.dirt_times, t, side="right") - 1, 0, len(self.dirt_times) - 1
            )
        )
        return self.dirt_keyframes[idx].sum(axis=0)

    def debris_at(self, t: float) -> NDArray:
        """Debris snapshot ``(n, 5)`` -- ``x, y, mass, size, collected``."""
        if self.debris_keyframes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        idx = int(
            np.clip(
                np.searchsorted(self.dirt_times, t, side="right") - 1, 0, len(self.dirt_times) - 1
            )
        )
        return self.debris_keyframes[idx]

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
            debris_keyframes=dirt.get("debris", np.zeros((0, 0, 5), dtype=np.float32)),
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
            f"timestep        {m.get('timestep')} s",
            f"frames          {self.n_frames} ({self.duration / 60:.1f} min)",
            f"channels        {len(self.frames)}",
            f"events          {len(self.events)}",
            f"dirt keyframes  {len(self.dirt_times)}",
        ]
        return "\n".join(lines)


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
                else np.zeros((0, 0, 5), dtype=np.float32)
            ),
            metrics=metrics or {},
            spatial=spatial or {},
        )


_INT_CHANNELS = frozenset({"step", "contacts", "collided", "stuck", "cmd_brush"})
"""Channels that are counts or bitfields, stored as int32 rather than float32."""


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
