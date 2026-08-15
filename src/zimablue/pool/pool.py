"""The :class:`Pool` -- ZimaBlue's world geometry.

A pool is a planar boundary, a depth model, a surface material and a list of
features.  Everything the simulator needs at run time (navigable mask, depth
raster, collision segments, perimeter parameterisation) is *derived* from those
four things and cached, so presets stay declarative and no preset needs to know
how the simulator consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkt import dumps as wkt_dumps
from shapely.wkt import loads as wkt_loads

from zimablue.geometry import Grid, closest_point_on_segments, polygon_segments, raycast
from zimablue.pool.depth import (
    CompositeDepth,
    ConstantDepth,
    DepthModel,
    PlaneSlopeDepth,
)
from zimablue.pool.features import (
    Drain,
    Obstacle,
    PoolFeature,
    Return,
    Skimmer,
    Stairs,
    flow_field,
)
from zimablue.pool.materials import SurfaceMaterial, get_material

__all__ = ["Pool", "Water"]

FloatArray = NDArray[np.float64]

DEFAULT_CELL = 0.10
"""Default raster resolution in metres. 10 cm balances fidelity and speed."""


@dataclass(frozen=True)
class Water:
    """Bulk water properties.

    Only the handful of quantities the dirt and sensor models actually consume.
    """

    temperature_c: float = 26.0
    density: float = 997.0
    """kg/m^3."""

    viscosity: float = 8.9e-4
    """Dynamic viscosity, Pa*s (fresh water near 25 C)."""

    turbidity: float = 0.05
    """0 (crystal) to 1 (soup); attenuates optical/acoustic range sensing."""

    circulation: float = 1.0
    """Scales the drain/return flow field. 0 disables circulation."""

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature_c": self.temperature_c,
            "density": self.density,
            "viscosity": self.viscosity,
            "turbidity": self.turbidity,
            "circulation": self.circulation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Water:
        return cls(**{k: float(v) for k, v in data.items()})


class Pool:
    """A swimming pool: geometry, floor, walls, features and water."""

    def __init__(
        self,
        boundary: Polygon,
        depth: DepthModel | float,
        *,
        name: str = "pool",
        material: str | SurfaceMaterial = "plaster",
        features: tuple[PoolFeature, ...] | list[PoolFeature] = (),
        water: Water | None = None,
    ) -> None:
        if not isinstance(boundary, Polygon):
            raise TypeError(f"boundary must be a shapely Polygon, got {type(boundary).__name__}")
        if boundary.is_empty or not boundary.is_valid:
            raise ValueError(f"pool {name!r} has an empty or invalid boundary")
        self.name = name
        self.boundary = boundary
        self.depth_model: DepthModel = (
            ConstantDepth(float(depth)) if isinstance(depth, (int, float)) else depth
        )
        self.material = get_material(material)
        self.features: tuple[PoolFeature, ...] = tuple(features)
        self.water = water if water is not None else Water()
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(self.boundary.bounds)  # type: ignore[return-value]

    @property
    def max_depth(self) -> float:
        return self.depth_model.max_depth

    @property
    def navigable(self) -> BaseGeometry:
        """Boundary minus every blocking feature footprint."""
        cached = self._cache.get("navigable")
        if cached is None:
            blockers = [
                f.footprint for f in self.features if f.blocking and f.footprint is not None
            ]
            cached = self.boundary.difference(unary_union(blockers)) if blockers else self.boundary
            self._cache["navigable"] = cached
        return cached

    @property
    def floor_area(self) -> float:
        """Navigable floor area in m^2 (blocking features excluded)."""
        return float(self.navigable.area)

    @property
    def perimeter_length(self) -> float:
        """Length of the pool's outer wall line in metres."""
        return float(self.boundary.exterior.length)

    @property
    def wall_area(self) -> float:
        """Approximate wetted wall area: perimeter integrated over local depth."""
        cached = self._cache.get("wall_area")
        if cached is None:
            ring = self.boundary.exterior
            n = max(64, int(ring.length / 0.25))
            ss = np.linspace(0.0, ring.length, n, endpoint=False)
            pts = [ring.interpolate(float(s)) for s in ss]
            xs = np.array([p.x for p in pts])
            ys = np.array([p.y for p in pts])
            depths = self.depth_at(xs, ys)
            cached = float(np.sum(depths) * (ring.length / n))
            self._cache["wall_area"] = cached
        return cached

    def depth_at(self, x: FloatArray | float, y: FloatArray | float) -> FloatArray:
        """Water depth in metres at ``(x, y)``."""
        return np.asarray(self.depth_model.depth_at(x, y), dtype=float)

    def contains(self, x: FloatArray | float, y: FloatArray | float) -> NDArray[np.bool_]:
        """Whether each point is inside the navigable region."""
        return np.asarray(contains_xy(self.navigable, x, y))

    # ------------------------------------------------------------------
    # Collision geometry
    # ------------------------------------------------------------------
    @property
    def wall_segments(self) -> FloatArray:
        """``(n, 4)`` segments of the pool's outer wall."""
        cached = self._cache.get("wall_segments")
        if cached is None:
            cached = polygon_segments(Polygon(self.boundary.exterior))
            self._cache["wall_segments"] = cached
        return cached

    @property
    def obstacle_segments(self) -> FloatArray:
        """``(n, 4)`` segments of every blocking feature."""
        cached = self._cache.get("obstacle_segments")
        if cached is None:
            rows: list[FloatArray] = [
                polygon_segments(f.footprint)
                for f in self.features
                if f.blocking and f.footprint is not None and not f.footprint.is_empty
            ]
            # Holes punched into the boundary itself are obstacles too.
            rows.extend(polygon_segments(Polygon(ring)) for ring in self.boundary.interiors)
            cached = np.vstack(rows) if rows else np.zeros((0, 4), dtype=float)
            self._cache["obstacle_segments"] = cached
        return cached

    @property
    def collision_segments(self) -> FloatArray:
        """Every segment the robot can hit: walls plus obstacles."""
        cached = self._cache.get("collision_segments")
        if cached is None:
            cached = np.vstack([self.wall_segments, self.obstacle_segments])
            self._cache["collision_segments"] = cached
        return cached

    def nearest_wall(self, x: float, y: float) -> tuple[float, float, float, bool]:
        """Distance and point of the closest surface, plus whether it is an obstacle.

        Returns ``(distance, nx, ny, is_obstacle)``.
        """
        wall_d, wx, wy, _ = closest_point_on_segments(self.wall_segments, x, y)
        obs_d, ox, oy, _ = closest_point_on_segments(self.obstacle_segments, x, y)
        if obs_d < wall_d:
            return (obs_d, ox, oy, True)
        return (wall_d, wx, wy, False)

    def raycast(
        self, origin: tuple[float, float], angles: FloatArray, max_range: float
    ) -> FloatArray:
        """Distances from ``origin`` to the first surface along each angle."""
        return raycast(self.collision_segments, origin, angles, max_range)

    def project_to_perimeter(self, x: float, y: float) -> float:
        """Arc length along the outer wall of the point nearest ``(x, y)``."""
        return float(self.boundary.exterior.project(Point(x, y)))

    # ------------------------------------------------------------------
    # Rasters
    # ------------------------------------------------------------------
    def grid(self, cell: float = DEFAULT_CELL) -> Grid:
        """Raster covering the pool at resolution ``cell`` (cached per size)."""
        key = f"grid:{cell:.6f}"
        cached = self._cache.get(key)
        if cached is None:
            cached = Grid.covering(self.bounds, cell)
            self._cache[key] = cached
        return cached

    def navigable_mask(self, cell: float = DEFAULT_CELL) -> NDArray[np.bool_]:
        """Boolean raster: cells whose centre is navigable floor."""
        key = f"nav_mask:{cell:.6f}"
        cached = self._cache.get(key)
        if cached is None:
            grid = self.grid(cell)
            xs, ys = grid.cell_centers()
            cached = np.asarray(contains_xy(self.navigable, xs, ys))
            self._cache[key] = cached
        return cached

    def depth_grid(self, cell: float = DEFAULT_CELL) -> FloatArray:
        """Depth raster, zero outside the navigable region."""
        key = f"depth_grid:{cell:.6f}"
        cached = self._cache.get(key)
        if cached is None:
            grid = self.grid(cell)
            xs, ys = grid.cell_centers()
            depths = self.depth_at(xs, ys)
            cached = np.where(self.navigable_mask(cell), depths, 0.0)
            self._cache[key] = cached
        return cached

    def flow_grid(self, cell: float = DEFAULT_CELL) -> tuple[FloatArray, FloatArray]:
        """Water velocity raster from drains and returns."""
        key = f"flow_grid:{cell:.6f}"
        cached = self._cache.get(key)
        if cached is None:
            grid = self.grid(cell)
            xs, ys = grid.cell_centers()
            vx, vy = flow_field(self.features, xs, ys, self.water.circulation)
            mask = self.navigable_mask(cell)
            cached = (np.where(mask, vx, 0.0), np.where(mask, vy, 0.0))
            self._cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------
    def start_pose(self, clearance: float = 0.35) -> tuple[float, float, float]:
        """A deterministic drop point: deepest navigable cell with clearance.

        Real cleaners are dropped in by hand, so a fixed, sensible start beats a
        random one -- and a deterministic default keeps seeded runs comparable
        across pools.
        """
        cell = DEFAULT_CELL
        grid = self.grid(cell)
        mask = self.navigable_mask(cell)
        depths = self.depth_grid(cell)
        xs, ys = grid.cell_centers()

        if clearance > 0:
            # Shrinking the navigable polygon is both faster and more exact than
            # testing every cell against every wall segment.
            inner = self.navigable.buffer(-clearance)
            if not inner.is_empty:
                mask = np.asarray(contains_xy(inner, xs, ys))

        if not mask.any():
            centroid = self.navigable.representative_point()
            return (float(centroid.x), float(centroid.y), 0.0)

        # Deepest water wins; on a flat floor every cell ties, so break toward
        # the centroid rather than letting argmax pick an arbitrary corner.
        centroid = self.navigable.centroid
        to_centre = np.hypot(xs - centroid.x, ys - centroid.y)
        scored = np.where(mask, depths - 1e-3 * to_centre, -np.inf)
        row, col = np.unravel_index(int(np.argmax(scored)), scored.shape)
        return (float(xs[row, col]), float(ys[row, col]), 0.0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Full geometric description, portable and JSON-safe.

        Recordings embed this rather than a preset name, so a ``.zbr`` remains
        replayable even if the preset is later changed or removed.
        """
        return {
            "name": self.name,
            "boundary_wkt": wkt_dumps(self.boundary, rounding_precision=-1),
            "depth": _depth_to_dict(self.depth_model),
            "material": self.material.name,
            "features": [_feature_to_dict(f) for f in self.features],
            "water": self.water.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pool:
        return cls(
            boundary=wkt_loads(data["boundary_wkt"]),
            depth=_depth_from_dict(data["depth"]),
            name=data.get("name", "pool"),
            material=data.get("material", "plaster"),
            features=tuple(_feature_from_dict(f) for f in data.get("features", [])),
            water=Water.from_dict(data.get("water", {})),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Pool(name={self.name!r}, floor_area={self.floor_area:.1f} m2, "
            f"max_depth={self.max_depth:.2f} m, features={len(self.features)})"
        )


# ----------------------------------------------------------------------
# Serialisation helpers -- tagged dicts keep the format readable and
# forward-compatible without pulling in a schema library.
# ----------------------------------------------------------------------
def _depth_to_dict(model: DepthModel) -> dict[str, Any]:
    if isinstance(model, ConstantDepth):
        return {"type": "constant", "depth": model.depth}
    if isinstance(model, PlaneSlopeDepth):
        return {
            "type": "plane_slope",
            "shallow": model.shallow,
            "deep": model.deep,
            "origin": list(model.origin),
            "direction": list(model.direction),
            "length": model.length,
        }
    if isinstance(model, CompositeDepth):
        return {
            "type": "composite",
            "base": _depth_to_dict(model.base),
            "regions": [
                {"polygon_wkt": wkt_dumps(p, rounding_precision=-1), "depth": _depth_to_dict(m)}
                for p, m in model.regions
            ],
        }
    raise TypeError(
        f"cannot serialise depth model {type(model).__name__}; "
        "add a case to _depth_to_dict or supply a serialisable model"
    )


def _depth_from_dict(data: dict[str, Any]) -> DepthModel:
    kind = data["type"]
    if kind == "constant":
        return ConstantDepth(float(data["depth"]))
    if kind == "plane_slope":
        return PlaneSlopeDepth(
            shallow=float(data["shallow"]),
            deep=float(data["deep"]),
            origin=tuple(data["origin"]),  # type: ignore[arg-type]
            direction=tuple(data["direction"]),  # type: ignore[arg-type]
            length=float(data["length"]),
        )
    if kind == "composite":
        return CompositeDepth(
            base=_depth_from_dict(data["base"]),
            regions=tuple(
                (wkt_loads(r["polygon_wkt"]), _depth_from_dict(r["depth"]))
                for r in data.get("regions", [])
            ),
        )
    raise ValueError(f"unknown depth model type {kind!r}")


def _feature_to_dict(feature: PoolFeature) -> dict[str, Any]:
    base = {"kind": type(feature).__name__.lower(), "name": feature.name}
    if isinstance(feature, Obstacle):
        return {
            **base,
            "polygon_wkt": wkt_dumps(feature.polygon, rounding_precision=-1),
            "height": feature.height,
        }
    if isinstance(feature, Stairs):
        return {
            **base,
            "polygon_wkt": wkt_dumps(feature.polygon, rounding_precision=-1),
            "steps": feature.steps,
            "top_depth": feature.top_depth,
            "bottom_depth": feature.bottom_depth,
            "climbable": feature.climbable,
        }
    if isinstance(feature, Drain):
        return {
            **base,
            "position": list(feature.position),
            "radius": feature.radius,
            "flow_rate": feature.flow_rate,
        }
    if isinstance(feature, Return):
        return {
            **base,
            "position": list(feature.position),
            "direction": list(feature.direction),
            "flow_rate": feature.flow_rate,
            "reach": feature.reach,
        }
    if isinstance(feature, Skimmer):
        return {
            **base,
            "position": list(feature.position),
            "width": feature.width,
            "capture_radius": feature.capture_radius,
        }
    raise TypeError(f"cannot serialise pool feature {type(feature).__name__}")


def _feature_from_dict(data: dict[str, Any]) -> PoolFeature:
    kind = data["kind"]
    name = data.get("name", kind)
    if kind == "obstacle":
        return Obstacle(name=name, polygon=wkt_loads(data["polygon_wkt"]), height=data["height"])
    if kind == "stairs":
        return Stairs(
            name=name,
            polygon=wkt_loads(data["polygon_wkt"]),
            steps=int(data["steps"]),
            top_depth=float(data["top_depth"]),
            bottom_depth=float(data["bottom_depth"]),
            climbable=bool(data["climbable"]),
        )
    if kind == "drain":
        return Drain(
            name=name,
            position=tuple(data["position"]),  # type: ignore[arg-type]
            radius=float(data["radius"]),
            flow_rate=float(data["flow_rate"]),
        )
    if kind == "return":
        return Return(
            name=name,
            position=tuple(data["position"]),  # type: ignore[arg-type]
            direction=tuple(data["direction"]),  # type: ignore[arg-type]
            flow_rate=float(data["flow_rate"]),
            reach=float(data["reach"]),
        )
    if kind == "skimmer":
        return Skimmer(
            name=name,
            position=tuple(data["position"]),  # type: ignore[arg-type]
            width=float(data["width"]),
            capture_radius=float(data["capture_radius"]),
        )
    raise ValueError(f"unknown pool feature kind {kind!r}")
