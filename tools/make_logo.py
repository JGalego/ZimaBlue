#!/usr/bin/env python3
"""Generate the ZimaBlue logo (static and animated SVG).

The logo is not a drawing of a pool -- it *is* the ``kidney`` pool preset, and
the robot's route is a real boustrophedon coverage path clipped to that
polygon.  Change the preset and the logo follows.  Regenerate with::

    python tools/make_logo.py

Outputs ``docs/assets/logo.svg`` and ``docs/assets/logo-animated.svg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString

from zimablue.pool import make_pool

# --- palette ---------------------------------------------------------------
DEEP = "#053a68"
MID = "#0e6cb2"
SHALLOW = "#54c8ea"
FOAM = "#c8f1fd"
COPING = "#eef3f7"
COPING_EDGE = "#aebecb"
HULL = "#141f2b"
HULL_LIGHT = "#2c4056"
ACCENT = "#3ddcff"

# --- layout ----------------------------------------------------------------
SCALE = 52.0
"""Pixels per metre for the pool."""

ROBOT_SCALE = SCALE * 2.15
"""The robot is drawn oversized. At true scale a 42 cm cleaner in a 12 m pool
is a 22 px speck; the logo needs it to read as the subject."""

PAD = 26.0
LANE_SPACING = 0.95
TRAIL_WIDTH_M = 0.66
PROGRESS = 0.60
"""How far along the route the static robot sits."""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
def coverage_lanes(polygon, spacing: float, inset: float) -> list[tuple[float, float]]:
    """Boustrophedon lane endpoints clipped to ``polygon``, in visiting order."""
    region = polygon.buffer(-inset)
    if region.is_empty:
        region = polygon
    minx, miny, maxx, maxy = region.bounds

    points: list[tuple[float, float]] = []
    for i, y in enumerate(np.arange(miny + spacing * 0.5, maxy, spacing)):
        cut = region.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
        if cut.is_empty:
            continue
        parts = list(cut.geoms) if isinstance(cut, MultiLineString) else [cut]
        parts = [p for p in parts if p.length > spacing * 0.6]
        if not parts:
            continue
        # One lane per row: the longest span. A real planner visits every span;
        # the logo wants a single legible sweep.
        lane = max(parts, key=lambda p: p.length)
        (x0, y0), (x1, y1) = lane.coords[0], lane.coords[-1]
        if x0 > x1:
            x0, x1 = x1, x0
        if i % 2:
            x0, x1 = x1, x0
        points += [(x0, y0), (x1, y1)]
    return points


class Route:
    """A rounded boustrophedon path in screen space.

    Holds both the SVG ``d`` string and a dense arc-length sample of the *same*
    curve, so the static robot can be placed exactly at the end of the drawn
    trail instead of approximately.
    """

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.commands: list[str] = []
        samples: list[np.ndarray] = []
        if not points:
            self.d = ""
            self.samples = np.zeros((0, 2))
            self.lengths = np.zeros(0)
            return

        self.commands.append(f"M {points[0][0]:.1f} {points[0][1]:.1f}")
        samples.append(np.array([points[0]]))
        i = 1
        while i < len(points):
            if i % 2 == 1 and i + 1 < len(points):
                # Lane end, then a rounded U-turn into the next lane.
                end = np.array(points[i])
                nxt = np.array(points[i + 1])
                prev = np.array(points[i - 1])
                ctrl = np.array([end[0] + (end[0] - prev[0]) * 0.09, (end[1] + nxt[1]) / 2.0])
                self.commands.append(f"L {end[0]:.1f} {end[1]:.1f}")
                self.commands.append(f"Q {ctrl[0]:.1f} {ctrl[1]:.1f} {nxt[0]:.1f} {nxt[1]:.1f}")
                samples.append(_line_samples(samples[-1][-1], end))
                samples.append(_quad_samples(end, ctrl, nxt))
                i += 2
            else:
                end = np.array(points[i])
                self.commands.append(f"L {end[0]:.1f} {end[1]:.1f}")
                samples.append(_line_samples(samples[-1][-1], end))
                i += 1

        self.d = " ".join(self.commands)
        self.samples = np.vstack(samples)
        steps = np.linalg.norm(np.diff(self.samples, axis=0), axis=1)
        self.lengths = np.concatenate([[0.0], np.cumsum(steps)])

    def at(self, fraction: float) -> tuple[float, float, float]:
        """Point and heading (degrees) at a fraction of total arc length."""
        target = float(np.clip(fraction, 0.0, 1.0)) * self.lengths[-1]
        idx = int(np.searchsorted(self.lengths, target))
        idx = min(max(idx, 1), len(self.samples) - 1)
        p0, p1 = self.samples[idx - 1], self.samples[idx]
        angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        return (float(p1[0]), float(p1[1]), angle)


def _line_samples(start, end, step: float = 3.0) -> np.ndarray:
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(2, int(np.linalg.norm(end - start) / step))
    t = np.linspace(0, 1, n)[1:, None]
    return start + t * (end - start)


def _quad_samples(p0, p1, p2, n: int = 24) -> np.ndarray:
    t = np.linspace(0, 1, n)[1:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def robot_markup(*, animated: bool) -> str:
    """The cleaner, centred on the origin and pointing along +x."""
    scale = ROBOT_SCALE
    length, width = 0.42 * scale, 0.38 * scale
    half_l, half_w = length / 2, width / 2
    track_h = width * 0.2
    spin = (
        '<animateTransform attributeName="transform" type="rotate" from="0" to="360" '
        'dur="0.6s" repeatCount="indefinite"/>'
        if animated
        else ""
    )
    pulse = (
        '<animate attributeName="opacity" values="1;0.25;1" dur="1.5s" repeatCount="indefinite"/>'
        if animated
        else ""
    )
    bubbles = ""
    if animated:
        bubbles = "".join(
            f'<circle cx="{-half_l - 3 - i * 2:.1f}" cy="0" r="{2.6 - i * 0.5:.1f}" '
            f'fill="{FOAM}" opacity="0">'
            f'<animate attributeName="cy" values="0;{-14 - i * 5}" dur="{1.5 + i * 0.35:.2f}s" '
            f'repeatCount="indefinite" begin="{i * 0.45:.2f}s"/>'
            f'<animate attributeName="opacity" values="0;0.75;0" dur="{1.5 + i * 0.35:.2f}s" '
            f'repeatCount="indefinite" begin="{i * 0.45:.2f}s"/></circle>'
            for i in range(3)
        )

    # A few tread ticks read as tracks; a full ladder of them reads as stripes.
    tread = "".join(
        f'<rect x="{-half_l * 0.74 + i * (length * 0.5 / 3):.1f}" y="{-half_w + 0.6:.1f}" '
        f'width="1.8" height="{track_h - 1.2:.1f}" fill="#7f97ad" opacity="0.35"/>'
        f'<rect x="{-half_l * 0.74 + i * (length * 0.5 / 3):.1f}" '
        f'y="{half_w - track_h + 0.6:.1f}" width="1.8" height="{track_h - 1.2:.1f}" '
        f'fill="#7f97ad" opacity="0.35"/>'
        for i in range(4)
    )

    return f"""
    {bubbles}
    <g filter="url(#robotShadow)">
      <rect x="{-half_l:.1f}" y="{-half_w:.1f}" width="{length:.1f}" height="{width:.1f}"
            rx="{width * 0.26:.1f}" fill="url(#hull)" stroke="#080e15" stroke-width="1.6"/>
      <rect x="{-half_l * 0.88:.1f}" y="{-half_w:.1f}" width="{length * 0.8:.1f}"
            height="{track_h:.1f}" rx="{track_h / 2:.1f}" fill="#0a141d"/>
      <rect x="{-half_l * 0.88:.1f}" y="{half_w - track_h:.1f}" width="{length * 0.8:.1f}"
            height="{track_h:.1f}" rx="{track_h / 2:.1f}" fill="#0a141d"/>
      {tread}
      <rect x="{-half_l * 0.62:.1f}" y="{-half_w * 0.44:.1f}" width="{length * 0.56:.1f}"
            height="{width * 0.44:.1f}" rx="{width * 0.11:.1f}" fill="{HULL_LIGHT}"/>
      <rect x="{-half_l * 0.44:.1f}" y="{-half_w * 0.14:.1f}" width="{length * 0.34:.1f}"
            height="{width * 0.14:.1f}" rx="{width * 0.07:.1f}" fill="#93a9bd" opacity="0.55"/>
      <rect x="{-half_l * 0.52:.1f}" y="{-half_w * 0.36:.1f}" width="{length * 0.4:.1f}"
            height="{width * 0.07:.1f}" rx="{width * 0.035:.1f}" fill="#ffffff" opacity="0.22"/>
      <g transform="translate({half_l * 0.74:.1f} 0)">
        <g>{spin}
          <circle r="{width * 0.23:.1f}" fill="none" stroke="{ACCENT}" stroke-width="3.4"
                  stroke-dasharray="4.4 4.6"/>
        </g>
        <circle r="{width * 0.085:.1f}" fill="{ACCENT}"/>
      </g>
      <circle cx="{-half_l * 0.76:.1f}" cy="0" r="{width * 0.085:.1f}"
              fill="{ACCENT}">{pulse}</circle>
    </g>"""


def caustics(rng: np.random.Generator, w: float, h: float, animated: bool) -> str:
    """Soft light ribbons on the pool floor."""
    parts = []
    for i in range(8):
        x, y = rng.uniform(0.06, 0.94) * w, rng.uniform(0.08, 0.92) * h
        rx, ry = rng.uniform(28, 70), rng.uniform(5, 12)
        rot, op = rng.uniform(-26, 26), rng.uniform(0.09, 0.18)
        drift = (
            f'<animate attributeName="opacity" values="{op:.2f};{op * 2.1:.2f};{op:.2f}" '
            f'dur="{rng.uniform(3.5, 6.5):.1f}s" repeatCount="indefinite" begin="{i * 0.4:.1f}s"/>'
            if animated
            else ""
        )
        parts.append(
            f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'transform="rotate({rot:.1f} {x:.1f} {y:.1f})" fill="#ffffff" '
            f'opacity="{op:.2f}">{drift}</ellipse>'
        )
    return f'<g filter="url(#causticBlur)">{"".join(parts)}</g>'


def defs(pool_path: str) -> str:
    return f"""
  <defs>
    <linearGradient id="water" x1="0" y1="0" x2="1" y2="0.4">
      <stop offset="0%" stop-color="{SHALLOW}"/>
      <stop offset="48%" stop-color="{MID}"/>
      <stop offset="100%" stop-color="{DEEP}"/>
    </linearGradient>
    <linearGradient id="hull" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{HULL_LIGHT}"/>
      <stop offset="100%" stop-color="{HULL}"/>
    </linearGradient>
    <radialGradient id="sheen" cx="0.28" cy="0.18" r="0.95">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.26"/>
      <stop offset="55%" stop-color="#ffffff" stop-opacity="0.04"/>
      <stop offset="100%" stop-color="#001427" stop-opacity="0.26"/>
    </radialGradient>
    <filter id="robotShadow" x="-90%" y="-90%" width="280%" height="280%">
      <feDropShadow dx="0" dy="3" stdDeviation="3.6" flood-color="#01162a" flood-opacity="0.6"/>
    </filter>
    <filter id="poolShadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#0b2740" flood-opacity="0.22"/>
    </filter>
    <filter id="soften" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2.8"/>
    </filter>
    <filter id="causticBlur" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4"/>
    </filter>
    <clipPath id="poolClip"><path d="{pool_path}"/></clipPath>
  </defs>"""


def build(animated: bool, duration: float = 15.0) -> str:
    pool = make_pool("kidney")
    minx, miny, maxx, maxy = pool.bounds
    width = (maxx - minx) * SCALE + 2 * PAD
    height = (maxy - miny) * SCALE + 2 * PAD

    def tf(x: float, y: float) -> tuple[float, float]:
        return (PAD + (x - minx) * SCALE, height - PAD - (y - miny) * SCALE)

    pool_path = (
        "M "
        + " L ".join(
            f"{px:.2f} {py:.2f}" for px, py in (tf(x, y) for x, y in pool.boundary.exterior.coords)
        )
        + " Z"
    )
    route = Route([tf(x, y) for x, y in coverage_lanes(pool.boundary, LANE_SPACING, inset=0.6)])
    trail_w = TRAIL_WIDTH_M * SCALE
    rng = np.random.default_rng(1712)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="A robotic cleaner tracing a coverage path across a kidney-shaped pool">',
        defs(pool_path),
        f'<g filter="url(#poolShadow)">'
        f'<path d="{pool_path}" fill="none" stroke="{COPING_EDGE}" stroke-width="17" '
        f'stroke-linejoin="round" opacity="0.5"/>'
        f'<path d="{pool_path}" fill="none" stroke="{COPING}" stroke-width="14" '
        f'stroke-linejoin="round"/></g>',
        f'<path d="{pool_path}" fill="url(#water)"/>',
        '<g clip-path="url(#poolClip)">',
        caustics(rng, width, height, animated),
    ]

    if animated:
        out.append(
            f'<path d="{route.d}" fill="none" stroke="{FOAM}" stroke-opacity="0.32" '
            f'stroke-width="{trail_w:.1f}" stroke-linecap="round" stroke-linejoin="round" '
            f'filter="url(#soften)" pathLength="1000" stroke-dasharray="1000" '
            f'stroke-dashoffset="1000">'
            f'<animate attributeName="stroke-dashoffset" values="1000;0" dur="{duration}s" '
            f'repeatCount="indefinite"/>'
            f'<animate attributeName="stroke-opacity" values="0.32;0.32;0" keyTimes="0;0.93;1" '
            f'dur="{duration}s" repeatCount="indefinite"/></path>'
        )
        out.append(
            f'<path d="{route.d}" fill="none" stroke="{ACCENT}" stroke-opacity="0.5" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-dasharray="4 8">'
            f'<animate attributeName="stroke-dashoffset" values="0;-240" dur="4s" '
            f'repeatCount="indefinite"/></path>'
        )
    else:
        drawn = round(PROGRESS * 1000)
        out.append(
            f'<path d="{route.d}" fill="none" stroke="{FOAM}" stroke-opacity="0.32" '
            f'stroke-width="{trail_w:.1f}" stroke-linecap="round" stroke-linejoin="round" '
            f'filter="url(#soften)" pathLength="1000" stroke-dasharray="{drawn} 1000"/>'
        )
        out.append(
            f'<path d="{route.d}" fill="none" stroke="{ACCENT}" stroke-opacity="0.45" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-dasharray="4 8"/>'
        )

    out.append(f'<path d="{pool_path}" fill="url(#sheen)"/>')
    out.append(
        f'<path d="{pool_path}" fill="none" stroke="#ffffff" '
        f'stroke-opacity="0.5" stroke-width="2.5"/>'
    )

    if animated:
        out.append(
            f"<g>{robot_markup(animated=True)}"
            f'<animateMotion dur="{duration}s" repeatCount="indefinite" rotate="auto" '
            f'path="{route.d}"/></g>'
        )
    else:
        rx, ry, angle = route.at(PROGRESS)
        out.append(
            f'<g transform="translate({rx:.1f} {ry:.1f}) rotate({angle:.1f})">'
            f"{robot_markup(animated=False)}</g>"
        )

    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logo.svg").write_text(build(animated=False), encoding="utf-8")
    (args.out / "logo-animated.svg").write_text(build(animated=True), encoding="utf-8")
    print(f"wrote {args.out / 'logo.svg'} and {args.out / 'logo-animated.svg'}")


if __name__ == "__main__":
    main()
