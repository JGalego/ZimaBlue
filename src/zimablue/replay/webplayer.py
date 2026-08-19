"""A recording as a web page.

``export_web_player`` writes one self-contained HTML file: the trajectory,
the dirt keyframes, the debris, the pool it all happened in, and a small
canvas player over them -- scrubber, speed, layer toggles, a shared clock
across runs. No server, no dependencies, no matplotlib; the file opens from
disk and works offline, which makes it the version of a replay you can attach
to an issue or mail to someone.

::

    from zimablue.replay import export_web_player

    export_web_player(recording, "run.html")
    export_web_player({"bsa": one, "binn": another}, "duel.html")  # side by side

Size is kept honest the same way the GIF exporters keep theirs: the
trajectory is decimated to what a scrubber can show, and the dirt field is
quantised to a byte per cell on a coarsened grid. Between keyframes the player
blends, exactly as the matplotlib cameras do -- every value shown lies between
two the cell genuinely held.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from zimablue.recording import Recording
from zimablue.replay.renderer import load_scene

__all__ = ["export_web_player"]

MAX_TRAIL_POINTS = 2400
MAX_DIRT_CELLS = 72
MAX_EVENTS = 400


def export_web_player(
    recordings: Recording | dict[str, Recording],
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write ``recordings`` (one, or ``{label: recording}``) as an HTML player."""
    if isinstance(recordings, Recording):
        label = recordings.manifest.get("scenario_name") or "recording"
        recordings = {str(label): recordings}
    if not recordings:
        raise ValueError("nothing to export: pass a Recording or a non-empty dict of them")

    runs = [_run_payload(name, recording) for name, recording in recordings.items()]
    payload = {
        "title": title or (runs[0]["name"] if len(runs) == 1 else "side by side"),
        "duration": max(run["duration"] for run in runs),
        "runs": runs,
    }
    # A "</script>" inside a string would end the tag early; JSON lets us
    # escape the slash without changing the value.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.replace("__ZB_DATA__", data))
    return path


def _run_payload(name: str, recording: Recording) -> dict[str, Any]:
    scene = load_scene(recording)
    pool = scene.pool

    time = np.asarray(recording.column("time"), dtype=float)
    stride = max(int(np.ceil(len(time) / MAX_TRAIL_POINTS)), 1)
    time = time[::stride]

    def track(channel: str) -> list[float]:
        return _round(np.asarray(recording.column(channel), dtype=float)[::stride])

    minx, miny, maxx, maxy = pool.bounds

    payload: dict[str, Any] = {
        "name": name,
        "duration": float(recording.duration),
        "bounds": [minx, miny, maxx, maxy],
        "pool": {
            "exterior": _ring(pool.boundary.exterior),
            "holes": [_ring(ring) for ring in pool.boundary.interiors],
            "obstacles": [_ring(obstacle.exterior) for obstacle in _obstacle_polygons(pool)],
        },
        "robot": {"radius": float(max(scene.robot_width, scene.robot_length) / 2)},
        "time": _round(time),
        "x": track("x"),
        "y": track("y"),
        "heading": _round(np.asarray(recording.column("heading"), dtype=float)[::stride], 3),
        "events": [
            {"t": round(float(event.get("time", 0.0)), 2), "kind": str(event.get("kind", ""))}
            for event in recording.events[:MAX_EVENTS]
        ],
        "metrics": {
            key: round(float(value), 4)
            for key, value in recording.metrics.items()
            if isinstance(value, int | float) and np.isfinite(float(value))
        },
        "dirt": _dirt_payload(recording, scene),
        "debris": _debris_payload(recording),
        "curve": _curve_payload(recording, scene),
    }
    return payload


def _ring(ring: Any) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in ring.coords]


def _obstacle_polygons(pool: Any) -> list[Any]:
    blockers = []
    for feature in getattr(pool, "features", ()):  # islands, steps -- whatever blocks
        footprint = getattr(feature, "footprint", None)
        if footprint is not None and getattr(feature, "blocking", True):
            blockers.append(footprint)
    return blockers


def _round(values: np.ndarray, digits: int = 2) -> list[float]:
    return [round(float(v), digits) for v in values]


def _dirt_payload(recording: Recording, scene: Any) -> dict[str, Any] | None:
    if recording.dirt_keyframes.size == 0:
        return None
    times = np.asarray(recording.dirt_times, dtype=float)
    rows, cols = recording.dirt_keyframes.shape[-2:]
    factor = max(int(np.ceil(max(rows, cols) / MAX_DIRT_CELLS)), 1)
    trim_r, trim_c = rows - rows % factor, cols - cols % factor

    initial = recording.dirt_at(0.0)
    positive = initial[initial > 0]
    vmax = float(np.percentile(positive, 92.0)) if positive.size else 1.0
    vmax = max(vmax, 1e-9)

    frames = []
    for t in times:
        grid = recording.dirt_at(float(t))[:trim_r, :trim_c]
        coarse = grid.reshape(trim_r // factor, factor, trim_c // factor, factor).mean(axis=(1, 3))
        frames.append(np.clip(coarse / vmax, 0.0, 1.0))
    stack = (np.asarray(frames) * 255).astype(np.uint8)

    grid = scene.grid
    return {
        "times": _round(times, 1),
        "rows": int(stack.shape[1]),
        "cols": int(stack.shape[2]),
        "cell": float(grid.cell * factor),
        "origin": [float(grid.minx), float(grid.miny)],
        "data": base64.b64encode(stack.tobytes()).decode("ascii"),
    }


def _debris_payload(recording: Recording) -> dict[str, Any] | None:
    if recording.debris_keyframes.size == 0:
        return None
    times = np.asarray(recording.dirt_times, dtype=float)
    frames = []
    for t in times:
        snapshot = recording.debris_at(float(t))
        frames.append(
            [
                [round(float(x), 2), round(float(y), 2), round(float(collected), 2)]
                for x, y, collected in zip(
                    snapshot[:, 0], snapshot[:, 1], snapshot[:, 4], strict=True
                )
            ]
        )
    sizes = [round(float(s), 3) for s in recording.debris_at(0.0)[:, 3]]
    return {"times": _round(times, 1), "frames": frames, "sizes": sizes}


def _curve_payload(recording: Recording, scene: Any) -> dict[str, Any] | None:
    from zimablue.planners.compare import coverage_curve

    try:
        times, fractions = coverage_curve(recording, scene.pool, swath=scene.swath)
    except Exception:  # pragma: no cover - a run too short to sample
        return None
    return {"times": _round(times, 1), "coverage": _round(fractions, 4)}


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZimaBlue replay</title>
<style>
  :root {
    --panel: #08111b; --ink: #dbe7f0; --dim: #6d8296; --water: #0e6cb2;
    --deep: #053a68; --accent: #3ddcff; --trail: #7fe9ff; --coping: #e9eff4;
  }
  body { margin: 0; background: var(--panel); color: var(--ink);
         font: 14px/1.4 system-ui, sans-serif; }
  header { padding: 10px 16px; display: flex; align-items: baseline; gap: 12px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header span { color: var(--dim); }
  #panels { display: flex; flex-wrap: wrap; gap: 12px; padding: 0 16px; }
  .panel { flex: 1 1 320px; max-width: 640px; }
  .panel .label { display: flex; justify-content: space-between;
                  padding: 2px 4px; color: var(--dim); }
  .panel .label b { color: var(--ink); font-weight: 600; }
  canvas { width: 100%; display: block; border-radius: 6px; background: var(--deep); }
  #controls { display: flex; align-items: center; gap: 10px; padding: 12px 16px;
              flex-wrap: wrap; }
  button { background: var(--water); color: var(--ink); border: 0; border-radius: 4px;
           padding: 6px 14px; font-size: 14px; cursor: pointer; }
  input[type=range] { flex: 1 1 200px; accent-color: var(--accent); }
  select { background: #10202f; color: var(--ink); border: 1px solid #24425d;
           border-radius: 4px; padding: 4px; }
  label.toggle { color: var(--dim); user-select: none; cursor: pointer; }
  #clock { font-variant-numeric: tabular-nums; color: var(--ink); min-width: 96px; }
</style>
</head>
<body>
<header><h1 id="title"></h1><span>ZimaBlue replay</span></header>
<div id="controls">
  <button id="play">pause</button>
  <span id="clock"></span>
  <input id="scrub" type="range" min="0" max="1000" value="0">
  <select id="speed">
    <option>1</option><option>2</option><option>4</option><option selected>8</option>
    <option>16</option><option>32</option><option>64</option>
  </select><span style="color:var(--dim)">x</span>
  <label class="toggle"><input id="show-dirt" type="checkbox" checked> dirt</label>
  <label class="toggle"><input id="show-trail" type="checkbox" checked> trail</label>
  <label class="toggle"><input id="show-debris" type="checkbox" checked> debris</label>
</div>
<div id="panels"></div>
<script>
const DATA = __ZB_DATA__;

const SILT = [51, 39, 15];
function decodeDirt(dirt) {
  if (!dirt) return null;
  const raw = atob(dirt.data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}
function span(times, t) {
  let before = 0;
  while (before + 1 < times.length && times[before + 1] <= t) before++;
  const after = Math.min(before + 1, times.length - 1);
  const gap = times[after] - times[before];
  const blend = gap > 0 ? Math.min(Math.max((t - times[before]) / gap, 0), 1) : 0;
  return [before, after, blend];
}
function frameAt(times, t) {
  let low = 0, high = times.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (times[mid] <= t) low = mid; else high = mid - 1;
  }
  return low;
}
function mmss(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

class Panel {
  constructor(run, host) {
    this.run = run;
    this.dirtBytes = decodeDirt(run.dirt);
    const wrap = document.createElement("div");
    wrap.className = "panel";
    const label = document.createElement("div");
    label.className = "label";
    label.innerHTML = `<b>${run.name}</b><span class="cov"></span>`;
    this.covEl = label.querySelector(".cov");
    this.canvas = document.createElement("canvas");
    const [minx, miny, maxx, maxy] = run.bounds;
    const aspect = (maxy - miny) / (maxx - minx);
    this.canvas.width = 640;
    this.canvas.height = Math.round(640 * aspect);
    wrap.append(label, this.canvas);
    host.append(wrap);
    this.ctx = this.canvas.getContext("2d");
    const pad = 12;
    this.scale = Math.min((this.canvas.width - 2 * pad) / (maxx - minx),
                          (this.canvas.height - 2 * pad) / (maxy - miny));
    this.ox = pad - minx * this.scale;
    this.oy = this.canvas.height - pad + miny * this.scale;
    if (run.dirt) {
      this.dirtCanvas = document.createElement("canvas");
      this.dirtCanvas.width = run.dirt.cols;
      this.dirtCanvas.height = run.dirt.rows;
      this.dirtCtx = this.dirtCanvas.getContext("2d");
      this.dirtImage = this.dirtCtx.createImageData(run.dirt.cols, run.dirt.rows);
    }
  }
  X(x) { return this.ox + x * this.scale; }
  Y(y) { return this.oy - y * this.scale; }
  path(ring) {
    const ctx = this.ctx;
    ctx.moveTo(this.X(ring[0][0]), this.Y(ring[0][1]));
    for (let i = 1; i < ring.length; i++) ctx.lineTo(this.X(ring[i][0]), this.Y(ring[i][1]));
    ctx.closePath();
  }
  poolPath() {
    this.ctx.beginPath();
    this.path(this.run.pool.exterior);
    for (const hole of this.run.pool.holes) this.path(hole);
    for (const obstacle of this.run.pool.obstacles) this.path(obstacle);
  }
  draw(t, layers) {
    const run = this.run, ctx = this.ctx;
    const tt = Math.min(t, run.duration);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.poolPath();
    ctx.fillStyle = "#0e6cb2";
    ctx.fill("evenodd");

    if (layers.dirt && this.dirtBytes) this.drawDirt(tt);

    this.poolPath();
    ctx.strokeStyle = "#e9eff4";
    ctx.lineWidth = 2;
    ctx.stroke();

    if (layers.debris && run.debris) this.drawDebris(tt);

    const idx = frameAt(run.time, tt);
    if (layers.trail && idx > 1) {
      ctx.beginPath();
      ctx.moveTo(this.X(run.x[0]), this.Y(run.y[0]));
      for (let i = 1; i <= idx; i++) ctx.lineTo(this.X(run.x[i]), this.Y(run.y[i]));
      ctx.strokeStyle = "rgba(127, 233, 255, 0.55)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    const r = Math.max(run.robot.radius * this.scale, 3);
    const x = this.X(run.x[idx]), y = this.Y(run.y[idx]), h = run.heading[idx];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = "#16212e";
    ctx.fill();
    ctx.strokeStyle = "#3ddcff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 1.6 * r * Math.cos(h), y - 1.6 * r * Math.sin(h));
    ctx.stroke();

    if (run.curve) {
      const c = run.curve.coverage[frameAt(run.curve.times, tt)];
      this.covEl.textContent = `${(c * 100).toFixed(0)}% covered`;
    }
  }
  drawDirt(t) {
    const dirt = this.run.dirt;
    const [before, after, blend] = span(dirt.times, t);
    const cells = dirt.rows * dirt.cols;
    const a = before * cells, b = after * cells;
    const px = this.dirtImage.data;
    for (let i = 0; i < cells; i++) {
      const v = (1 - blend) * this.dirtBytes[a + i] + blend * this.dirtBytes[b + i];
      const j = i * 4;
      px[j] = SILT[0]; px[j + 1] = SILT[1]; px[j + 2] = SILT[2];
      px[j + 3] = Math.min(v * 0.85, 217);
    }
    this.dirtCtx.putImageData(this.dirtImage, 0, 0);
    const ctx = this.ctx;
    ctx.save();
    this.poolPath();
    ctx.clip("evenodd");
    ctx.imageSmoothingEnabled = true;
    const w = dirt.cols * dirt.cell * this.scale;
    const hgt = dirt.rows * dirt.cell * this.scale;
    const x0 = this.X(dirt.origin[0]);
    const y0 = this.Y(dirt.origin[1] + dirt.rows * dirt.cell);
    ctx.save();
    ctx.translate(x0, y0 + hgt);
    ctx.scale(1, -1);
    ctx.drawImage(this.dirtCanvas, 0, 0, w, hgt);
    ctx.restore();
    ctx.restore();
  }
  drawDebris(t) {
    const debris = this.run.debris, ctx = this.ctx;
    const [before, after, blend] = span(debris.times, t);
    const f0 = debris.frames[before], f1 = debris.frames[after];
    for (let i = 0; i < f0.length; i++) {
      const x = (1 - blend) * f0[i][0] + blend * f1[i][0];
      const y = (1 - blend) * f0[i][1] + blend * f1[i][1];
      const gone = (1 - blend) * f0[i][2] + blend * f1[i][2];
      if (gone >= 1) continue;
      ctx.beginPath();
      ctx.arc(this.X(x), this.Y(y),
              Math.max(debris.sizes[i] * this.scale * 0.5, 1.5), 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(107, 87, 53, ${(1 - gone).toFixed(2)})`;
      ctx.fill();
    }
  }
}

document.getElementById("title").textContent = DATA.title;
const panels = DATA.runs.map(run => new Panel(run, document.getElementById("panels")));

const play = document.getElementById("play");
const scrub = document.getElementById("scrub");
const speed = document.getElementById("speed");
const clock = document.getElementById("clock");
const layers = {
  dirt: document.getElementById("show-dirt"),
  trail: document.getElementById("show-trail"),
  debris: document.getElementById("show-debris"),
};
let t = 0, playing = true, last = null;

function render() {
  const active = { dirt: layers.dirt.checked, trail: layers.trail.checked,
                   debris: layers.debris.checked };
  for (const panel of panels) panel.draw(t, active);
  clock.textContent = `${mmss(t)} / ${mmss(DATA.duration)}`;
  scrub.value = Math.round(1000 * t / DATA.duration);
}
function tick(now) {
  if (last !== null && playing) {
    t += (now - last) / 1000 * Number(speed.value);
    if (t >= DATA.duration) { t = DATA.duration; playing = false; play.textContent = "replay"; }
  }
  last = now;
  render();
  requestAnimationFrame(tick);
}
play.addEventListener("click", () => {
  if (!playing && t >= DATA.duration) t = 0;
  playing = !playing;
  play.textContent = playing ? "pause" : "play";
});
scrub.addEventListener("input", () => { t = scrub.value / 1000 * DATA.duration; });
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
