#!/usr/bin/env python3
"""Build a pool that ZimaBlue has never seen, and clean it.

    python examples/custom_pool.py
    python examples/custom_pool.py --minutes 5

Pools are not a fixed menu. A pool is a boundary polygon, a depth model, a
surface material and a list of features -- supply those and the
simulator treats yours exactly like a built-in preset.

This builds a lap pool with a beach entry, a shallow ledge, an island planter
and a set of steps, registers it under a name, runs a clean, and then looks at
the *spatial* metrics: not just how much was covered, but which parts were not.
"""

from __future__ import annotations

import argparse

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

import zimablue as zb


def build_pool() -> zb.Pool:
    """A 14 x 7 m lap pool with a rounded west end."""
    # Any Shapely polygon works. This one is a rectangle with one end rounded
    # off, built by unioning a box with a half-disc rather than by typing out
    # vertices.
    body = shapely_box(1.9, 0.0, 14.0, 7.0)
    angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 48)
    cap = Polygon(np.column_stack([2.0 + 2.0 * np.cos(angles), 3.5 + 3.5 * np.sin(angles)]))
    # Overlap the two shapes rather than letting them merely touch: pieces that
    # share only an edge union into a MultiPolygon, which a Pool will reject.
    merged = body.union(cap).buffer(0)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    boundary = Polygon(merged.exterior)

    # Depth: a shallow beach entry at the west end, deepening eastward, with a
    # flat ledge along the north wall. CompositeDepth layers regions over a
    # base model -- the ledge does not need to know about the slope beneath it.
    depth = zb.CompositeDepth(
        base=zb.PlaneSlopeDepth(
            shallow=0.4, deep=2.2, origin=(0.0, 0.0), direction=(1.0, 0.0), length=14.0
        ),
        regions=((shapely_box(6.0, 6.0, 14.0, 7.0), zb.ConstantDepth(0.9)),),
    )

    return zb.Pool(
        boundary=boundary,
        depth=depth,
        name="lap_pool",
        material="tile",  # smooth: less grip, so more wheel slip
        features=(
            # Blocking features come out of the navigable area, so coverage is
            # scored against the floor the robot can actually reach.
            zb.Obstacle("island_planter", polygon=shapely_box(7.5, 3.0, 8.7, 4.2), height=0.9),
            zb.Stairs(
                "corner_steps",
                polygon=shapely_box(12.4, 0.0, 14.0, 1.8),
                steps=3,
                top_depth=0.3,
                bottom_depth=1.4,
            ),
            # Hydraulic features do not block; they push dirt around.
            zb.Drain("main_drain", position=(11.0, 3.5), radius=0.3, flow_rate=0.2),
            zb.Return("return_west", position=(0.6, 3.5), direction=(1.0, 0.0)),
            zb.Skimmer("skimmer", position=(13.2, 6.6)),
        ),
        water=zb.Water(temperature_c=24.0, turbidity=0.12),
    )


# Registering makes it available everywhere a preset name is accepted --
# including `zimablue run` and scenario YAML -- without touching the library.
zb.POOL_PRESETS.add("lap_pool", build_pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=25.0)
    args = parser.parse_args()

    pool = build_pool()
    print(pool)
    print(f"  boundary area  {pool.boundary.area:6.1f} m2")
    print(f"  navigable      {pool.floor_area:6.1f} m2   (the planter and steps are excluded)")
    print(f"  wetted walls   {pool.wall_area:6.1f} m2")
    print(
        f"  depth          {pool.depth_at(3.0, 3.5)[()]:.2f} m at the beach end, "
        f"{pool.depth_at(13.0, 3.5)[()]:.2f} m at the deep end"
    )

    # A preset name works anywhere a Pool does, and vice versa.
    result = zb.Simulation(
        pool="lap_pool",
        robot=zb.make_robot("heavy_duty"),  # wide brush, fine filter
        dirt="autumn",
        seed=7,
        record=False,
    ).run(minutes=args.minutes)

    print()
    print(result.metrics.summary())

    # ------------------------------------------------------------------
    # Spatial metrics: where the scalar numbers came from.
    # ------------------------------------------------------------------
    spatial = result.spatial
    navigable = spatial.navigable
    missed = navigable & (spatial.visits == 0)
    cell_area = pool.grid(0.1).cell_area

    print("\nwhere the coverage went")
    print(f"  never visited      {missed.sum() * cell_area:6.1f} m2")
    print(f"  visited once       {(navigable & (spatial.visits == 1)).sum() * cell_area:6.1f} m2")
    print(f"  visited 5+ times   {(navigable & (spatial.visits >= 5)).sum() * cell_area:6.1f} m2")

    # The distinction the whole project is about: driving over a cell is not
    # the same as cleaning it.
    driven_over = navigable & (spatial.visits > 0)
    still_dirty = driven_over & (spatial.remaining_dirt > 0.2 * spatial.initial_dirt.max())
    print(f"\n  driven over but still dirty: {still_dirty.sum() * cell_area:.1f} m2")
    print("  (suction alone does not lift adhered dirt -- it takes brush passes)")

    if missed.any():
        grid = pool.grid(0.1)
        xs, ys = grid.cell_centers()
        print(
            f"\n  the worst gap is around "
            f"({xs[missed].mean():.1f}, {ys[missed].mean():.1f}) m -- "
            "usually the corner behind an obstacle"
        )


if __name__ == "__main__":
    main()
