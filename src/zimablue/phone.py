"""Reconstruct a pool from several perspective-corrected phone photographs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

from zimablue.imaging import PoolTrace, Segmenter, trace_pool
from zimablue.pool import ConstantDepth, DepthModel, PlaneSlopeDepth, Pool

__all__ = [
    "DepthObservation",
    "PhoneReconstruction",
    "PhoneView",
    "fit_phone_depth",
    "fuse_phone_traces",
    "pool_from_phones",
]

Point = tuple[float, float]


@dataclass(frozen=True)
class PhoneView:
    """One phone image and its planar calibration target.

    ``corners`` are the image pixels of the same physical rectangle in every
    view, ordered top-left, top-right, bottom-right, bottom-left in the shared
    survey frame. ``rectangle`` is its width and height in metres.
    """

    image: Any
    corners: tuple[Point, Point, Point, Point]
    rectangle: tuple[float, float]
    sample: tuple[int, int] | None = None


@dataclass(frozen=True)
class DepthObservation:
    """A measured water depth at a point in the rectified survey frame."""

    x: float
    y: float
    depth: float


@dataclass(frozen=True)
class PhoneReconstruction:
    """Consensus geometry and per-view checks from a phone survey."""

    boundary: Polygon
    traces: tuple[PoolTrace, ...]
    agreement: tuple[float, ...]
    area_variation: float
    quorum: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> float:
        """Mean intersection-over-union between each view and consensus."""
        return float(np.mean(self.agreement))

    def summary(self) -> str:
        return (
            f"{len(self.traces)} views, {self.boundary.area:.1f} m2, "
            f"agreement {self.confidence:.1%}, area CV {self.area_variation:.1%}"
        )

    def pool(
        self,
        depth: DepthModel | float = 1.5,
        *,
        name: str = "phone_reconstructed",
        material: str = "plaster",
    ) -> Pool:
        model = ConstantDepth(float(depth)) if isinstance(depth, int | float) else depth
        return Pool(boundary=self.boundary, depth=model, name=name, material=material)


def fuse_phone_traces(
    traces: Sequence[PoolTrace], *, quorum: int | None = None
) -> PhoneReconstruction:
    """Fuse metrically rectified traces by exact polygon majority vote."""
    if len(traces) < 2:
        raise ValueError("phone reconstruction requires at least two views")
    required = len(traces) // 2 + 1 if quorum is None else quorum
    if not 1 <= required <= len(traces):
        raise ValueError(f"quorum must be between 1 and {len(traces)}")

    polygons = [trace.boundary for trace in traces]
    faces = polygonize(unary_union([polygon.boundary for polygon in polygons]))
    accepted = [
        face
        for face in faces
        if sum(polygon.covers(face.representative_point()) for polygon in polygons) >= required
    ]
    if not accepted:
        raise ValueError(
            "the rectified views have no quorum overlap; check corner order and trace overlays"
        )
    consensus = unary_union(accepted)
    warnings = [warning for trace in traces for warning in trace.warnings]
    if consensus.geom_type == "MultiPolygon":
        pieces = sorted(consensus.geoms, key=lambda item: item.area, reverse=True)
        discarded = sum(piece.area for piece in pieces[1:])
        warnings.append(f"discarded {discarded:.2f} m2 of disconnected quorum geometry")
        consensus = pieces[0]
    consensus = Polygon(consensus.exterior)

    agreements = []
    for polygon in polygons:
        union_area = polygon.union(consensus).area
        agreements.append(float(polygon.intersection(consensus).area / union_area))
    areas = np.asarray([polygon.area for polygon in polygons], dtype=float)
    variation = float(areas.std() / areas.mean())
    if min(agreements) < 0.75:
        warnings.append("one or more views agree with less than 75% of the consensus")

    return PhoneReconstruction(
        boundary=consensus,
        traces=tuple(traces),
        agreement=tuple(agreements),
        area_variation=variation,
        quorum=required,
        warnings=tuple(warnings),
    )


def pool_from_phones(
    views: Sequence[PhoneView],
    *,
    depth: DepthModel | float = 1.5,
    name: str = "phone_reconstructed",
    material: str = "plaster",
    segmenter: Segmenter | None = None,
    quorum: int | None = None,
    overlay_directory: str | Path | None = None,
    **trace_options: Any,
) -> Pool:
    """Trace calibrated phone views, fuse them and return a simulation pool."""
    if len(views) < 2:
        raise ValueError("phone reconstruction requires at least two views")
    traces = [
        trace_pool(
            view.image,
            sample=view.sample,
            corners=(view.corners, view.rectangle),
            segmenter=segmenter,
            **trace_options,
        )
        for view in views
    ]
    result = fuse_phone_traces(traces, quorum=quorum)
    if overlay_directory is not None:
        directory = Path(overlay_directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, trace in enumerate(traces):
            trace.overlay(directory / f"view-{index + 1}.png")
    return result.pool(depth, name=name, material=material)


def fit_phone_depth(boundary: Polygon, observations: Sequence[DepthObservation]) -> DepthModel:
    """Fit a flat or linearly sloped depth model to manual depth readings."""
    if not observations:
        raise ValueError("at least one depth observation is required")
    values = np.asarray([(item.x, item.y, item.depth) for item in observations], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 2] <= 0.0):
        raise ValueError("depth observations must be finite and positive")
    if len(values) < 3:
        if np.ptp(values[:, 2]) > 1e-9:
            raise ValueError("at least three observations are needed to fit a slope")
        return ConstantDepth(float(values[:, 2].mean()))

    design = np.column_stack([values[:, 0], values[:, 1], np.ones(len(values))])
    if np.linalg.matrix_rank(design) < 3 and np.ptp(values[:, 2]) > 1e-9:
        raise ValueError("depth observations must include three non-collinear locations")
    coefficients, *_ = np.linalg.lstsq(design, values[:, 2], rcond=None)
    gradient = coefficients[:2]
    magnitude = float(np.linalg.norm(gradient))
    if magnitude < 1e-9:
        return ConstantDepth(float(coefficients[2]))

    direction = gradient / magnitude
    vertices = np.asarray(boundary.exterior.coords, dtype=float)
    projection = vertices @ direction
    start, end = float(projection.min()), float(projection.max())
    centroid = np.asarray(boundary.centroid.coords[0], dtype=float)
    origin = centroid + direction * (start - float(centroid @ direction))
    shallow = float(coefficients[2] + origin @ gradient)
    deep = shallow + magnitude * (end - start)
    if shallow <= 0.0 or deep <= 0.0:
        raise ValueError("fitted depth plane is non-positive inside the pool")
    return PlaneSlopeDepth(
        shallow=shallow,
        deep=deep,
        origin=(float(origin[0]), float(origin[1])),
        direction=(float(direction[0]), float(direction[1])),
        length=end - start,
    )
