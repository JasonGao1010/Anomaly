#!/usr/bin/env python3
"""Deterministic world-level counterfactual rendering for AJAE.

The renderer preserves organized file slots for I/O.  A slot becomes a
canonical LiDAR ray only after the explicit RayGrid audit.  Inserted objects
live in world coordinates and compete with native returns by nearest distance.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
from scipy.optimize import brentq, differential_evolution
from scipy.spatial import ConvexHull, QhullError, cKDTree

try:
    from .scene import PointLabels, SourceFrame, make_source_frame
except ImportError:  # Direct module execution and small isolated checks.
    from scene import PointLabels, SourceFrame, make_source_frame


LASER_BEAMS = 128
GROUND_SEMANTIC_IDS = (40, 44, 48, 49, 60)
WORLD_FORMAT = "ajae-world-v2"
CALIBRATION_FORMAT = "ajae-sensor-calibration-v4"
DEVELOPMENT_FORMAT = "ajae-development-worlds-v2"
DEVELOPMENT_PROTOCOL_SCHEMA = 30
GATE1_EVIDENCE_KEYS = (
    "ray_slot_audit",
    "range_image_round_trip",
    "render_source_leakage",
    "beam_range_intensity",
)
DEVELOPMENT_VALIDATION_KEYS = (
    "physical_placement",
    "sequence_visibility",
    "difficulty_coverage",
    "normal_control_and_proxy_composition",
    "held_out_mechanism_isolation",
)
SYNTHETIC_INSTANCE_BASE = 60_000
MAX_OBJECT_ID = np.iinfo(np.uint16).max - SYNTHETIC_INSTANCE_BASE
OBJECT_LABELS = ("normal-control", "anomaly-proxy")
ObjectLabel: TypeAlias = Literal["normal-control", "anomaly-proxy"]
WORLD_TYPES = ("pure_normal", "control_only", "mixed", "anomaly_only")
WorldType: TypeAlias = Literal["pure_normal", "control_only", "mixed", "anomaly_only"]
NORMAL_TEMPLATE_SEMANTICS = frozenset((10, 11, 15, 18, 20, 30, 31, 32))
NORMAL_SEMANTIC_TARGET = {10: 0, 11: 1, 15: 2, 18: 3, 20: 4, 30: 5, 31: 6, 32: 7}
VEHICLE_TEMPLATE_SEMANTICS = frozenset((10, 11, 15, 18, 20))
PERSON_RIDER_TEMPLATE_SEMANTICS = frozenset((30, 31, 32))
VEHICLE_SUPPORT_SEMANTICS = frozenset((40, 44))
PERSON_RIDER_SUPPORT_SEMANTICS = frozenset((40, 48, 60))
DEFAULT_RANGE_EDGES_M = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 120.0)
DEFAULT_INCIDENCE_EDGES_RAD = (
    0.0,
    math.pi / 6.0,
    math.pi / 3.0,
    math.pi / 2.0,
)
EPSILON = 1.0e-9


class RenderError(ValueError):
    """Report an invalid physical world, calibration, or rendered frame."""


class PlacementError(RenderError):
    """Report that no physically admissible placement was found."""


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RenderError(f"{name} must be finite")
    return result


def _integer(name: str, value: int, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RenderError(f"{name} must be an integer >= {minimum}")
    return value


def _tuple_values(name: str, value: Sequence[float], length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise RenderError(f"{name} must contain {length} values")
    return tuple(
        _finite_scalar(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _nested_values(
    name: str, value: Sequence[Sequence[float]], length: int
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise RenderError(f"{name} must be a sequence")
    return tuple(
        _tuple_values(f"{name}[{index}]", item, length)
        for index, item in enumerate(value)
    )


def _freeze(array: np.ndarray, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _rotation_tuple(
    name: str, value: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    rows = _nested_values(name, value, 3)
    if len(rows) != 3:
        raise RenderError(f"{name} must have shape (3, 3)")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-6, rtol=1.0e-6):
        raise RenderError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-6):
        raise RenderError(f"{name} determinant must be +1")
    return rows


def _pose(frame: SourceFrame) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(frame.lidar_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise RenderError("SourceFrame.lidar_pose must be finite float64[4,4]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-3, rtol=1.0e-3):
        raise RenderError("SourceFrame.lidar_pose rotation is not orthonormal")
    return rotation, pose[:3, 3]


def _component_count(mask: np.ndarray, *, stop_after: int = 2) -> int:
    remaining = np.asarray(mask, dtype=np.bool_).copy()
    components = 0
    shape = remaining.shape
    while bool(remaining.any()):
        components += 1
        if components >= stop_after:
            return components
        start = tuple(int(item) for item in np.argwhere(remaining)[0])
        remaining[start] = False
        queue: deque[tuple[int, int, int]] = deque((start,))
        while queue:
            x, y, z = queue.popleft()
            for nx, ny, nz in (
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ):
                if (
                    0 <= nx < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nz < shape[2]
                    and remaining[nx, ny, nz]
                ):
                    remaining[nx, ny, nz] = False
                    queue.append((nx, ny, nz))
    return components


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    """A closed CSG composition of deformed superquadric primitives."""

    primitive_scales_m: tuple[tuple[float, float, float], ...]
    primitive_offsets_m: tuple[tuple[float, float, float], ...]
    primitive_exponents: tuple[tuple[float, float], ...]
    primitive_yaws_rad: tuple[float, ...]
    operations: tuple[str, ...]
    twist_rad_per_m: float = 0.0
    bend_per_m: tuple[float, float] = (0.0, 0.0)
    taper_per_m: tuple[float, float] = (0.0, 0.0)
    surface_amplitude_m: float = 0.0
    surface_frequency_per_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    surface_phase_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        scales = _nested_values("primitive_scales_m", self.primitive_scales_m, 3)
        offsets = _nested_values("primitive_offsets_m", self.primitive_offsets_m, 3)
        exponents = _nested_values("primitive_exponents", self.primitive_exponents, 2)
        yaws = tuple(
            _finite_scalar(f"primitive_yaws_rad[{index}]", value)
            for index, value in enumerate(self.primitive_yaws_rad)
        )
        operations = tuple(str(value) for value in self.operations)
        count = len(scales)
        if not 1 <= count <= 5:
            raise RenderError("a shape must contain one through five primitives")
        if not (
            len(offsets) == len(exponents) == len(yaws) == len(operations) == count
        ):
            raise RenderError("all primitive fields must have the same length")
        if operations[0] != "union" or any(
            value not in {"union", "difference", "intersection"} for value in operations
        ):
            raise RenderError(
                "the first operation must be union and all operations must be valid CSG"
            )
        if any(any(axis <= 0.02 or axis > 5.0 for axis in scale) for scale in scales):
            raise RenderError("primitive half-scales must lie in (0.02, 5] metres")
        if any(
            any(not 0.2 <= exponent <= 2.5 for exponent in pair) for pair in exponents
        ):
            raise RenderError("superquadric exponents must lie in [0.2, 2.5]")
        bend = _tuple_values("bend_per_m", self.bend_per_m, 2)
        taper = _tuple_values("taper_per_m", self.taper_per_m, 2)
        frequency = _tuple_values(
            "surface_frequency_per_m", self.surface_frequency_per_m, 3
        )
        phase = _tuple_values("surface_phase_rad", self.surface_phase_rad, 3)
        twist = _finite_scalar("twist_rad_per_m", self.twist_rad_per_m)
        amplitude = _finite_scalar("surface_amplitude_m", self.surface_amplitude_m)
        if abs(twist) > 4.0 or any(abs(value) > 0.75 for value in bend):
            raise RenderError("twist or bend exceeds the stable deformation range")
        if any(abs(value) > 0.45 for value in taper):
            raise RenderError("taper exceeds the stable deformation range")
        if amplitude < 0.0 or amplitude > 0.25 * min(min(item) for item in scales):
            raise RenderError("surface displacement is negative or too large")
        if any(value <= 0.0 or value > 20.0 for value in frequency):
            raise RenderError("surface frequencies must lie in (0, 20]")
        for name, value in (
            ("primitive_scales_m", scales),
            ("primitive_offsets_m", offsets),
            ("primitive_exponents", exponents),
            ("primitive_yaws_rad", yaws),
            ("operations", operations),
            ("bend_per_m", bend),
            ("taper_per_m", taper),
            ("surface_frequency_per_m", frequency),
            ("surface_phase_rad", phase),
            ("twist_rad_per_m", twist),
            ("surface_amplitude_m", amplitude),
        ):
            object.__setattr__(self, name, value)
        self.geometry_report(resolution=25)

    @property
    def primitive_count(self) -> int:
        return len(self.primitive_scales_m)

    @property
    def bound_radius_m(self) -> float:
        primitive = max(
            float(np.linalg.norm(offset)) + float(np.linalg.norm(scale))
            for offset, scale in zip(
                self.primitive_offsets_m, self.primitive_scales_m, strict=True
            )
        )
        deformation = (
            self.surface_amplitude_m
            + max(map(abs, self.bend_per_m)) * primitive * primitive
            + max(map(abs, self.taper_per_m)) * primitive
        )
        return 1.15 * (primitive + deformation + 1.0e-3)

    def _single_primitive_surface_point(
        self, latitude: float, longitude: float
    ) -> np.ndarray:
        """Solve the continuous outer surface along one undeformed direction."""

        if self.primitive_count != 1:
            raise RenderError("continuous primitive bounds require exactly one primitive")
        if self.operations != ("union",) or self.primitive_offsets_m != ((0.0, 0.0, 0.0),):
            raise RenderError("continuous primitive bounds require one centered union primitive")
        cosine = math.cos(latitude)
        direction = np.asarray(
            (
                cosine * math.cos(longitude),
                cosine * math.sin(longitude),
                math.sin(latitude),
            ),
            dtype=np.float64,
        )
        scale = self.primitive_scales_m[0]
        exponent = self.primitive_exponents[0]
        yaw = self.primitive_yaws_rad[0]
        minimum_scale = min(scale)
        unit_value = float(
            self._primitive_distance(direction[None], scale, (0.0, 0.0, 0.0), exponent, yaw)[0]
            / minimum_scale
            + 1.0
        )
        if not np.isfinite(unit_value) or unit_value <= 0.0:
            raise RenderError("single-primitive radial function is not finite and positive")
        upper = (1.0 + self.surface_amplitude_m / minimum_scale) / unit_value

        def implicit(radius: float) -> float:
            point = radius * direction
            base = minimum_scale * (radius * unit_value - 1.0)
            displacement = self.surface_amplitude_m * float(
                np.mean(
                    np.sin(
                        point * np.asarray(self.surface_frequency_per_m)
                        + np.asarray(self.surface_phase_rad)
                    )
                )
            )
            return base - displacement

        samples = np.linspace(0.0, upper * (1.0 + 1.0e-12), 65)
        values = np.asarray([implicit(float(value)) for value in samples])
        crossings = np.flatnonzero((values[:-1] <= 0.0) & (values[1:] >= 0.0))
        if crossings.size == 0:
            raise RenderError("continuous surface root was not bracketed")
        index = int(crossings[-1])
        radius = brentq(
            implicit,
            float(samples[index]),
            float(samples[index + 1]),
            xtol=1.0e-13,
            rtol=1.0e-13,
        )
        undeformed = radius * direction
        z = float(undeformed[2])
        angle = self.twist_rad_per_m * z
        rotation_cosine = math.cos(angle)
        rotation_sine = math.sin(angle)
        rotated_x = rotation_cosine * undeformed[0] - rotation_sine * undeformed[1]
        rotated_y = rotation_sine * undeformed[0] + rotation_cosine * undeformed[1]
        scale_z = self.primitive_scales_m[0][2]
        factor_x = float(np.clip(1.0 + self.taper_per_m[0] * z / scale_z, 0.25, 4.0))
        factor_y = float(np.clip(1.0 + self.taper_per_m[1] * z / scale_z, 0.25, 4.0))
        return np.asarray(
            (
                factor_x * rotated_x + self.bend_per_m[0] * z * z,
                factor_y * rotated_y + self.bend_per_m[1] * z * z,
                z,
            ),
            dtype=np.float64,
        )

    def continuous_bounds(
        self,
        *,
        maximum_iterations: int = 160,
        population_size: int = 15,
        safety_margin_m: float = 1.0e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate continuous single-primitive AABB extrema without a mesh grid."""

        iterations = _integer("maximum_iterations", maximum_iterations, minimum=20)
        population = _integer("population_size", population_size, minimum=5)
        margin = _finite_scalar("safety_margin_m", safety_margin_m)
        if margin < 0.0 or margin > 1.0e-3:
            raise RenderError("continuous-bound safety margin must lie in [0,1e-3]")
        lower = np.empty(3, dtype=np.float64)
        upper = np.empty(3, dtype=np.float64)
        angular_bounds = ((-0.5 * math.pi, 0.5 * math.pi), (-math.pi, math.pi))
        for axis in range(3):
            for sign in (-1.0, 1.0):
                result = differential_evolution(
                    lambda value: -sign
                    * self._single_primitive_surface_point(value[0], value[1])[axis],
                    angular_bounds,
                    seed=1009 + 17 * axis + int(sign > 0.0),
                    maxiter=iterations,
                    popsize=population,
                    tol=1.0e-10,
                    atol=1.0e-11,
                    polish=True,
                    updating="immediate",
                    workers=1,
                )
                if not result.success or not np.isfinite(result.fun):
                    raise RenderError("continuous-bound optimization did not converge")
                value = -float(result.fun) * sign
                if sign < 0.0:
                    lower[axis] = value - margin
                else:
                    upper[axis] = value + margin
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise RenderError("continuous primitive bounds are invalid")
        return _freeze(lower), _freeze(upper)

    def _undeform(self, points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64).copy()
        scale_z = max(item[2] for item in self.primitive_scales_m)
        z = result[..., 2]
        result[..., 0] -= self.bend_per_m[0] * np.square(z)
        result[..., 1] -= self.bend_per_m[1] * np.square(z)
        normalized_z = z / scale_z
        factor_x = np.clip(1.0 + self.taper_per_m[0] * normalized_z, 0.25, 4.0)
        factor_y = np.clip(1.0 + self.taper_per_m[1] * normalized_z, 0.25, 4.0)
        result[..., 0] /= factor_x
        result[..., 1] /= factor_y
        angle = -self.twist_rad_per_m * z
        cosine = np.cos(angle)
        sine = np.sin(angle)
        x = result[..., 0].copy()
        y = result[..., 1].copy()
        result[..., 0] = cosine * x - sine * y
        result[..., 1] = sine * x + cosine * y
        return result

    @staticmethod
    def _primitive_distance(
        points: np.ndarray,
        scale: tuple[float, ...],
        offset: tuple[float, ...],
        exponent: tuple[float, ...],
        yaw: float,
    ) -> np.ndarray:
        local = points - np.asarray(offset, dtype=np.float64)
        cosine = math.cos(-yaw)
        sine = math.sin(-yaw)
        x = cosine * local[..., 0] - sine * local[..., 1]
        y = sine * local[..., 0] + cosine * local[..., 1]
        z = local[..., 2]
        a, b, c = scale
        vertical, horizontal = exponent
        with np.errstate(over="ignore", invalid="ignore"):
            xy = (
                np.power(np.abs(x / a), 2.0 / horizontal)
                + np.power(np.abs(y / b), 2.0 / horizontal)
            ) ** (horizontal / vertical)
            implicit = (xy + np.power(np.abs(z / c), 2.0 / vertical)) ** (
                vertical / 2.0
            ) - 1.0
        return implicit * min(scale)

    def signed_distance(self, points_local: np.ndarray) -> np.ndarray:
        """Return a finite implicit signed-distance approximation in metres."""

        points = np.asarray(points_local, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise RenderError("points_local must be finite [...,3]")
        undeformed = self._undeform(points)
        result: np.ndarray | None = None
        for scale, offset, exponent, yaw, operation in zip(
            self.primitive_scales_m,
            self.primitive_offsets_m,
            self.primitive_exponents,
            self.primitive_yaws_rad,
            self.operations,
            strict=True,
        ):
            primitive = self._primitive_distance(
                undeformed, scale, offset, exponent, yaw
            )
            if result is None or operation == "union":
                result = primitive if result is None else np.minimum(result, primitive)
            elif operation == "difference":
                result = np.maximum(result, -primitive)
            else:
                result = np.maximum(result, primitive)
        assert result is not None
        phase = np.asarray(self.surface_phase_rad, dtype=np.float64)
        frequency = np.asarray(self.surface_frequency_per_m, dtype=np.float64)
        displacement = self.surface_amplitude_m * np.mean(
            np.sin(undeformed * frequency + phase), axis=-1
        )
        result = result - displacement
        if not np.isfinite(result).all():
            raise RenderError("shape evaluation produced NaN or Inf")
        return result

    def geometry_report(self, *, resolution: int = 31) -> dict[str, float | int | bool]:
        """Numerically reject empty, open, or disconnected CSG results."""

        if type(resolution) is not int or resolution < 17 or resolution % 2 == 0:
            raise RenderError("geometry resolution must be an odd integer >= 17")
        radius = self.bound_radius_m
        axis = np.linspace(-radius, radius, resolution, dtype=np.float64)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = self.signed_distance(np.stack((x, y, z), axis=-1))
        inside = values <= 0.0
        inside_count = int(np.count_nonzero(inside))
        if inside_count < 8 or inside_count == inside.size:
            raise RenderError("CSG result has no effective enclosed volume")
        boundary = np.zeros_like(inside)
        boundary[[0, -1], :, :] = True
        boundary[:, [0, -1], :] = True
        boundary[:, :, [0, -1]] = True
        if bool(np.any(inside & boundary)):
            raise RenderError(
                "shape touches its conservative bound and is not verified closed"
            )
        components = _component_count(inside)
        if components != 1:
            raise RenderError("CSG result is split into disconnected components")
        surface = inside.copy()
        core = inside[1:-1, 1:-1, 1:-1]
        surrounded = (
            inside[:-2, 1:-1, 1:-1]
            & inside[2:, 1:-1, 1:-1]
            & inside[1:-1, :-2, 1:-1]
            & inside[1:-1, 2:, 1:-1]
            & inside[1:-1, 1:-1, :-2]
            & inside[1:-1, 1:-1, 2:]
        )
        surface[1:-1, 1:-1, 1:-1] = core & ~surrounded
        surface_count = int(np.count_nonzero(surface))
        if surface_count < 6:
            raise RenderError("shape has no effective surface")
        occupied = np.argwhere(inside)
        step = 2.0 * radius / (resolution - 1)
        lower = axis[occupied.min(axis=0)] - step
        upper = axis[occupied.max(axis=0)] + step
        return {
            "bounded": True,
            "closed": True,
            "components": components,
            "inside_voxels": inside_count,
            "surface_voxels": surface_count,
            "minimum_x_m": float(lower[0]),
            "minimum_y_m": float(lower[1]),
            "minimum_z_m": float(lower[2]),
            "maximum_x_m": float(upper[0]),
            "maximum_y_m": float(upper[1]),
            "maximum_z_m": float(upper[2]),
        }

    def local_bounds(self, *, resolution: int = 41) -> tuple[np.ndarray, np.ndarray]:
        report = self.geometry_report(resolution=resolution)
        lower = np.asarray(
            [report[f"minimum_{axis}_m"] for axis in "xyz"], dtype=np.float64
        )
        upper = np.asarray(
            [report[f"maximum_{axis}_m"] for axis in "xyz"], dtype=np.float64
        )
        return lower, upper

    def minimum_z_m(self, *, xy_resolution: int = 33, z_steps: int = 129) -> float:
        """Numerically locate the true lower support surface in local coordinates."""

        if xy_resolution < 17 or xy_resolution % 2 == 0 or z_steps < 65:
            raise RenderError(
                "support search requires odd xy_resolution >=17 and z_steps >=65"
            )
        radius = self.bound_radius_m

        def roots(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            z_axis = np.linspace(-radius, radius, z_steps, dtype=np.float64)
            points = np.empty((xy.shape[0], z_steps, 3), dtype=np.float64)
            points[..., :2] = xy[:, None, :]
            points[..., 2] = z_axis
            inside = self.signed_distance(points) <= 0.0
            found = inside.any(axis=1)
            first = np.argmax(inside, axis=1)
            found &= first > 0
            output = np.full(xy.shape[0], np.inf, dtype=np.float64)
            ids = np.flatnonzero(found)
            if ids.size:
                lo = z_axis[first[ids] - 1].copy()
                hi = z_axis[first[ids]].copy()
                for _ in range(20):
                    middle = 0.5 * (lo + hi)
                    query = np.column_stack((xy[ids], middle))
                    middle_inside = self.signed_distance(query) <= 0.0
                    hi = np.where(middle_inside, middle, hi)
                    lo = np.where(middle_inside, lo, middle)
                output[ids] = 0.5 * (lo + hi)
            return output, found

        axis = np.linspace(-radius, radius, xy_resolution, dtype=np.float64)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        xy = np.column_stack((x.ravel(), y.ravel()))
        values, found = roots(xy)
        if not bool(found.any()):
            raise RenderError("shape has no lower support surface")
        best = int(np.argmin(values))
        center = xy[best]
        step = 2.0 * radius / (xy_resolution - 1)
        best_value = float(values[best])
        # Refine horizontal position because the minimum need not lie on the grid.
        for _ in range(5):
            offsets = np.linspace(-step, step, 5, dtype=np.float64)
            dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
            candidates = center + np.column_stack((dx.ravel(), dy.ravel()))
            candidate_values, candidate_found = roots(candidates)
            if bool(candidate_found.any()):
                index = int(np.argmin(candidate_values))
                if candidate_values[index] < best_value:
                    center = candidates[index]
                    best_value = float(candidate_values[index])
            step *= 0.25
        return best_value

    def intersect(
        self,
        origin_local: np.ndarray,
        directions_local: np.ndarray,
        *,
        steps: int = 96,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find each ray's first outside-to-inside surface crossing."""

        origin = np.asarray(origin_local, dtype=np.float64)
        directions = np.asarray(directions_local, dtype=np.float64)
        if origin.shape == (3,):
            origin = np.broadcast_to(origin, directions.shape)
        if (
            origin.shape != directions.shape
            or directions.ndim != 2
            or directions.shape[1] != 3
        ):
            raise RenderError("ray origins and directions must be [N,3]")
        if not np.isfinite(origin).all() or not np.isfinite(directions).all():
            raise RenderError("rays must be finite")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= EPSILON):
            raise RenderError("ray directions must be nonzero")
        directions = directions / norms[:, None]
        if type(steps) is not int or steps < 32:
            raise RenderError("ray sampling steps must be an integer >= 32")

        radius = self.bound_radius_m
        projection = np.sum(origin * directions, axis=1)
        discriminant = np.square(projection) - (
            np.sum(np.square(origin), axis=1) - radius * radius
        )
        near = -projection - np.sqrt(np.maximum(discriminant, 0.0))
        far = -projection + np.sqrt(np.maximum(discriminant, 0.0))
        near = np.maximum(near, 1.0e-5)
        candidate = (discriminant >= 0.0) & (far > near)
        distance = np.full(directions.shape[0], np.inf, dtype=np.float64)
        normal = np.zeros_like(directions)
        candidate_ids = np.flatnonzero(candidate)
        fractions = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        for start in range(0, candidate_ids.size, 2048):
            ids = candidate_ids[start : start + 2048]
            ray_t = near[ids, None] + (far[ids] - near[ids])[:, None] * fractions
            samples = origin[ids, None, :] + ray_t[..., None] * directions[ids, None, :]
            values = self.signed_distance(samples)
            inside = values <= 0.0
            starts_inside = inside[:, 0]
            entry = (~inside[:, :-1]) & inside[:, 1:]
            exit_surface = inside[:, :-1] & (~inside[:, 1:])
            has_entry = entry.any(axis=1)
            has_exit = exit_surface.any(axis=1)
            has_hit = np.where(starts_inside, has_exit, has_entry)
            if not bool(has_hit.any()):
                continue
            hit_ids = ids[has_hit]
            hit_t = ray_t[has_hit]
            hit_starts_inside = starts_inside[has_hit]
            entry_index = np.argmax(entry[has_hit], axis=1) + 1
            exit_index = np.argmax(exit_surface[has_hit], axis=1) + 1
            first = np.where(hit_starts_inside, exit_index, entry_index)
            row = np.arange(hit_ids.size)
            lo = hit_t[row, first - 1]
            hi = hit_t[row, first]
            for _ in range(18):
                middle = 0.5 * (lo + hi)
                points = origin[hit_ids] + middle[:, None] * directions[hit_ids]
                middle_inside = self.signed_distance(points) <= 0.0
                entry_ray = ~hit_starts_inside
                hi = np.where(entry_ray & middle_inside, middle, hi)
                lo = np.where(entry_ray & ~middle_inside, middle, lo)
                lo = np.where(hit_starts_inside & middle_inside, middle, lo)
                hi = np.where(hit_starts_inside & ~middle_inside, middle, hi)
            distance[hit_ids] = 0.5 * (lo + hi)

        valid = np.isfinite(distance)
        if bool(valid.any()):
            points = origin[valid] + distance[valid, None] * directions[valid]
            delta = max(1.0e-5, radius * 1.0e-5)
            gradient = np.empty_like(points)
            for axis in range(3):
                offset = np.zeros(3, dtype=np.float64)
                offset[axis] = delta
                gradient[:, axis] = (
                    self.signed_distance(points + offset)
                    - self.signed_distance(points - offset)
                ) / (2.0 * delta)
            length = np.linalg.norm(gradient, axis=1)
            finite_normal = np.isfinite(length) & (length > EPSILON)
            valid_ids = np.flatnonzero(valid)
            rejected = valid_ids[~finite_normal]
            distance[rejected] = np.inf
            valid[rejected] = False
            kept = valid_ids[finite_normal]
            normal[kept] = gradient[finite_normal] / length[finite_normal, None]
        return _freeze(distance), _freeze(normal), _freeze(valid)

    @classmethod
    def sample(
        cls,
        seed: int,
        *,
        primitive_count: int | None = None,
        size_m_range: tuple[float, float] = (0.2, 3.0),
    ) -> "ShapeSpec":
        """Sample a reproducible connected shape; invalid CSG draws are rejected."""

        _integer("seed", seed)
        minimum, maximum = _tuple_values("size_m_range", size_m_range, 2)
        if not 0.2 <= minimum <= maximum <= 3.0:
            raise RenderError("anomaly size_m_range must lie in [0.2,3.0]")
        if (
            primitive_count is not None
            and not 1 <= _integer("primitive_count", primitive_count, minimum=1) <= 5
        ):
            raise RenderError("primitive_count must lie in [1,5]")
        rng = np.random.default_rng(seed)
        for _ in range(64):
            count = (
                int(rng.integers(1, 6)) if primitive_count is None else primitive_count
            )
            half = float(rng.uniform(minimum / 2.0, maximum / 2.0))
            base = np.clip(half * rng.uniform(0.65, 1.25, size=3), 0.055, maximum / 2.0)
            scales = [tuple(map(float, base))]
            offsets = [(0.0, 0.0, 0.0)]
            exponents = [tuple(map(float, rng.uniform(0.55, 1.65, size=2)))]
            yaws = [float(rng.uniform(-math.pi, math.pi))]
            operations = ["union"]
            for _primitive in range(1, count):
                operation = str(
                    rng.choice(
                        ("union", "difference", "intersection"), p=(0.65, 0.2, 0.15)
                    )
                )
                scale = base * rng.uniform(0.32, 0.78, size=3)
                if operation == "difference":
                    offset = base * rng.uniform(-0.35, 0.35, size=3)
                else:
                    offset = base * rng.uniform(-0.55, 0.55, size=3)
                scales.append(tuple(map(float, scale)))
                offsets.append(tuple(map(float, offset)))
                exponents.append(tuple(map(float, rng.uniform(0.5, 1.8, size=2))))
                yaws.append(float(rng.uniform(-math.pi, math.pi)))
                operations.append(operation)
            amplitude = float(rng.uniform(0.0, 0.08 * float(base.min())))
            try:
                result = cls(
                    primitive_scales_m=tuple(scales),
                    primitive_offsets_m=tuple(offsets),
                    primitive_exponents=tuple(exponents),
                    primitive_yaws_rad=tuple(yaws),
                    operations=tuple(operations),
                    twist_rad_per_m=float(rng.uniform(-0.65, 0.65)),
                    bend_per_m=tuple(map(float, rng.uniform(-0.12, 0.12, size=2))),
                    taper_per_m=tuple(map(float, rng.uniform(-0.18, 0.18, size=2))),
                    surface_amplitude_m=amplitude,
                    surface_frequency_per_m=tuple(
                        map(float, rng.uniform(0.6, 2.2, size=3))
                    ),
                    surface_phase_rad=tuple(
                        map(float, rng.uniform(-math.pi, math.pi, size=3))
                    ),
                )
                # Require connectivity at both audit and placement resolutions.
                result.geometry_report(resolution=31)
                lower, upper = result.local_bounds(resolution=41)
                diameter = float(np.max(upper - lower))
                if minimum <= diameter <= maximum:
                    return result
            except RenderError:
                continue
        raise RenderError(
            "could not sample a connected shape within 64 deterministic attempts"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "procedural-csg",
            "primitive_scales_m": [list(item) for item in self.primitive_scales_m],
            "primitive_offsets_m": [list(item) for item in self.primitive_offsets_m],
            "primitive_exponents": [list(item) for item in self.primitive_exponents],
            "primitive_yaws_rad": list(self.primitive_yaws_rad),
            "operations": list(self.operations),
            "twist_rad_per_m": self.twist_rad_per_m,
            "bend_per_m": list(self.bend_per_m),
            "taper_per_m": list(self.taper_per_m),
            "surface_amplitude_m": self.surface_amplitude_m,
            "surface_frequency_per_m": list(self.surface_frequency_per_m),
            "surface_phase_rad": list(self.surface_phase_rad),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShapeSpec":
        if not isinstance(value, Mapping):
            raise RenderError("ShapeSpec JSON must be an object")
        plain = dict(value)
        if plain.pop("kind", None) != "procedural-csg":
            raise RenderError("ShapeSpec JSON has the wrong geometry kind")
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid ShapeSpec JSON: {error}") from error


def _sampled_sdf_intersection(
    signed_distance: Callable[[np.ndarray], np.ndarray],
    bound_radius_m: float,
    origin_local: np.ndarray,
    directions_local: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the first surface crossing of one bounded implicit solid."""

    origin = np.asarray(origin_local, dtype=np.float64)
    directions = np.asarray(directions_local, dtype=np.float64)
    if origin.shape == (3,):
        origin = np.broadcast_to(origin, directions.shape)
    if (
        origin.shape != directions.shape
        or directions.ndim != 2
        or directions.shape[1] != 3
    ):
        raise RenderError("ray origins and directions must be [N,3]")
    if not np.isfinite(origin).all() or not np.isfinite(directions).all():
        raise RenderError("rays must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= EPSILON):
        raise RenderError("ray directions must be nonzero")
    directions = directions / norms[:, None]
    if type(steps) is not int or steps < 32:
        raise RenderError("ray sampling steps must be an integer >= 32")

    radius = _finite_scalar("bound_radius_m", bound_radius_m)
    projection = np.sum(origin * directions, axis=1)
    discriminant = np.square(projection) - (
        np.sum(np.square(origin), axis=1) - radius * radius
    )
    near = -projection - np.sqrt(np.maximum(discriminant, 0.0))
    far = -projection + np.sqrt(np.maximum(discriminant, 0.0))
    near = np.maximum(near, 1.0e-5)
    candidate = (discriminant >= 0.0) & (far > near)
    distance = np.full(directions.shape[0], np.inf, dtype=np.float64)
    normal = np.zeros_like(directions)
    candidate_ids = np.flatnonzero(candidate)
    fractions = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    for start in range(0, candidate_ids.size, 2048):
        ids = candidate_ids[start : start + 2048]
        ray_t = near[ids, None] + (far[ids] - near[ids])[:, None] * fractions
        samples = origin[ids, None, :] + ray_t[..., None] * directions[ids, None, :]
        values = signed_distance(samples)
        inside = values <= 0.0
        starts_inside = inside[:, 0]
        entry = (~inside[:, :-1]) & inside[:, 1:]
        exit_surface = inside[:, :-1] & ~inside[:, 1:]
        has_hit = np.where(starts_inside, exit_surface.any(axis=1), entry.any(axis=1))
        if not bool(has_hit.any()):
            continue
        hit_ids = ids[has_hit]
        hit_t = ray_t[has_hit]
        hit_inside = starts_inside[has_hit]
        first = np.where(
            hit_inside,
            np.argmax(exit_surface[has_hit], axis=1) + 1,
            np.argmax(entry[has_hit], axis=1) + 1,
        )
        row = np.arange(hit_ids.size)
        lo = hit_t[row, first - 1]
        hi = hit_t[row, first]
        for _ in range(18):
            middle = 0.5 * (lo + hi)
            points = origin[hit_ids] + middle[:, None] * directions[hit_ids]
            middle_inside = signed_distance(points) <= 0.0
            entering = ~hit_inside
            hi = np.where(entering & middle_inside, middle, hi)
            lo = np.where(entering & ~middle_inside, middle, lo)
            lo = np.where(hit_inside & middle_inside, middle, lo)
            hi = np.where(hit_inside & ~middle_inside, middle, hi)
        distance[hit_ids] = 0.5 * (lo + hi)

    valid = np.isfinite(distance)
    if bool(valid.any()):
        points = origin[valid] + distance[valid, None] * directions[valid]
        delta = max(1.0e-5, radius * 1.0e-5)
        gradient = np.empty_like(points)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = delta
            gradient[:, axis] = (
                signed_distance(points + offset) - signed_distance(points - offset)
            ) / (2.0 * delta)
        lengths = np.linalg.norm(gradient, axis=1)
        stable = np.isfinite(lengths) & (lengths > EPSILON)
        valid_ids = np.flatnonzero(valid)
        distance[valid_ids[~stable]] = np.inf
        valid[valid_ids[~stable]] = False
        kept = valid_ids[stable]
        normal[kept] = gradient[stable] / lengths[stable, None]
    return _freeze(distance), _freeze(normal), _freeze(valid)


@dataclass(frozen=True, slots=True)
class NormalTemplateShape:
    """Closed convex-hull approximation of one labelled normal instance from 206."""

    vertices_m: np.ndarray
    faces: np.ndarray
    source_sequence_id: int
    source_frame_id: int
    raw_semantic_id: int
    source_instance_id: int
    source_center_sensor_m: tuple[float, float, float]
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    plane_normals: np.ndarray = field(init=False, repr=False, compare=False)
    plane_offsets: np.ndarray = field(init=False, repr=False, compare=False)
    hull_volume_m3: float = field(init=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 4:
            raise RenderError("normal template needs at least four 3D hull vertices")
        if not np.isfinite(vertices).all():
            raise RenderError("normal template vertices must be finite")
        if _integer("source_sequence_id", self.source_sequence_id) != 206:
            raise RenderError("normal templates must be extracted only from train/206")
        _integer("source_frame_id", self.source_frame_id)
        semantic = _integer("raw_semantic_id", self.raw_semantic_id)
        if semantic not in NORMAL_TEMPLATE_SEMANTICS:
            raise RenderError(
                "normal template semantic is not an allowed instance class"
            )
        _integer("source_instance_id", self.source_instance_id, minimum=1)
        center = _tuple_values("source_center_sensor_m", self.source_center_sensor_m, 3)
        scale = _tuple_values("scale_xyz", self.scale_xyz, 3)
        if any(not 0.9 <= value <= 1.1 for value in scale):
            raise RenderError("normal-template scale factors must lie in [0.9,1.1]")
        try:
            hull = ConvexHull(vertices)
        except QhullError as error:
            raise RenderError(
                "normal instance does not form a finite 3D convex hull"
            ) from error
        if not math.isfinite(float(hull.volume)) or hull.volume <= 1.0e-6:
            raise RenderError("normal template convex hull has negligible volume")
        hull_vertices = vertices[np.asarray(hull.vertices, dtype=np.int64)]
        hull = ConvexHull(hull_vertices)
        equations = np.asarray(hull.equations, dtype=np.float64)
        object.__setattr__(self, "vertices_m", _freeze(hull_vertices))
        object.__setattr__(
            self, "faces", _freeze(np.asarray(hull.simplices, dtype=np.int32))
        )
        object.__setattr__(self, "source_center_sensor_m", center)
        object.__setattr__(self, "scale_xyz", scale)
        object.__setattr__(self, "plane_normals", _freeze(equations[:, :3]))
        object.__setattr__(self, "plane_offsets", _freeze(equations[:, 3]))
        object.__setattr__(self, "hull_volume_m3", float(hull.volume))

    @classmethod
    def from_source_frame(
        cls,
        frame: SourceFrame,
        *,
        raw_semantic_id: int,
        instance_id: int,
        source_sequence_id: int = 206,
        scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
        maximum_points: int = 4096,
    ) -> "NormalTemplateShape":
        """Extract one labelled single-frame instance and close it by a convex hull."""

        if frame.labels is None:
            raise RenderError(
                "normal-template extraction requires packed instance labels"
            )
        if (
            frame.partition != "train"
            or frame.sequence_id != 206
            or source_sequence_id != 206
        ):
            raise RenderError(
                "normal-template extraction requires an identified train/206 frame"
            )
        if type(maximum_points) is not int or maximum_points < 16:
            raise RenderError("maximum_points must be an integer >=16")
        semantic = _integer("raw_semantic_id", raw_semantic_id)
        instance = _integer("instance_id", instance_id, minimum=1)
        selected = (
            (frame.labels.semantic == np.uint16(semantic))
            & (frame.labels.instance == np.uint16(instance))
            & ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        )
        points = np.asarray(frame.xyzi[selected, :3], dtype=np.float64)
        if points.shape[0] < 4:
            raise RenderError("normal instance has fewer than four visible points")
        if points.shape[0] > maximum_points:
            keep = np.linspace(0, points.shape[0] - 1, maximum_points, dtype=np.int64)
            points = points[keep]
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        factors = np.asarray(_tuple_values("scale_xyz", scale_xyz, 3), dtype=np.float64)
        local = (points - center) * factors
        try:
            hull = ConvexHull(local)
        except QhullError as error:
            raise RenderError("normal instance points are not volumetric") from error
        return cls(
            vertices_m=local[np.asarray(hull.vertices, dtype=np.int64)],
            faces=np.asarray(hull.simplices, dtype=np.int32),
            source_sequence_id=source_sequence_id,
            source_frame_id=int(frame.frame_id),
            raw_semantic_id=semantic,
            source_instance_id=instance,
            source_center_sensor_m=tuple(map(float, center)),
            scale_xyz=tuple(map(float, factors)),
        )

    @property
    def bound_radius_m(self) -> float:
        return float(np.max(np.linalg.norm(self.vertices_m, axis=1))) + 1.0e-6

    def local_bounds(self, *, resolution: int = 0) -> tuple[np.ndarray, np.ndarray]:
        del resolution
        return self.vertices_m.min(axis=0), self.vertices_m.max(axis=0)

    def minimum_z_m(self, *, xy_resolution: int = 0, z_steps: int = 0) -> float:
        del xy_resolution, z_steps
        return float(np.min(self.vertices_m[:, 2]))

    def signed_distance(self, points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise RenderError("points_local must be finite [...,3]")
        return np.max(points @ self.plane_normals.T + self.plane_offsets, axis=-1)

    def intersect(
        self,
        origin_local: np.ndarray,
        directions_local: np.ndarray,
        *,
        steps: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del steps
        directions = np.asarray(directions_local, dtype=np.float64)
        origin = np.asarray(origin_local, dtype=np.float64)
        if origin.shape == (3,):
            origin = np.broadcast_to(origin, directions.shape)
        if (
            origin.shape != directions.shape
            or directions.ndim != 2
            or directions.shape[1] != 3
        ):
            raise RenderError("ray origins and directions must be [N,3]")
        lengths = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origin).all() or np.any(lengths <= EPSILON):
            raise RenderError("convex-hull rays must be finite and nonzero")
        directions = directions / lengths[:, None]
        count = directions.shape[0]
        distance = np.full(count, np.inf, dtype=np.float64)
        normals = np.zeros((count, 3), dtype=np.float64)
        for start in range(0, count, 4096):
            stop = min(start + 4096, count)
            o = origin[start:stop]
            d = directions[start:stop]
            numerator = -(o @ self.plane_normals.T + self.plane_offsets)
            denominator = d @ self.plane_normals.T
            parallel_outside = np.any(
                (np.abs(denominator) <= EPSILON) & (numerator < 0.0), axis=1
            )
            lower_terms = np.full_like(denominator, -np.inf)
            upper_terms = np.full_like(denominator, np.inf)
            np.divide(
                numerator,
                denominator,
                out=lower_terms,
                where=denominator < -EPSILON,
            )
            np.divide(
                numerator,
                denominator,
                out=upper_terms,
                where=denominator > EPSILON,
            )
            enter_plane = np.argmax(lower_terms, axis=1)
            exit_plane = np.argmin(upper_terms, axis=1)
            enter = np.max(lower_terms, axis=1)
            exit_distance = np.min(upper_terms, axis=1)
            inside = np.all(numerator >= -1.0e-8, axis=1)
            candidate = np.where(inside, exit_distance, enter)
            valid = (
                ~parallel_outside
                & (exit_distance >= np.maximum(enter, 0.0))
                & (candidate > 1.0e-5)
            )
            local_ids = np.flatnonzero(valid)
            if local_ids.size:
                distance[start + local_ids] = candidate[local_ids]
                plane = np.where(inside, exit_plane, enter_plane)[local_ids]
                normals[start + local_ids] = self.plane_normals[plane]
        return _freeze(distance), _freeze(normals), _freeze(np.isfinite(distance))

    def geometry_report(self, *, resolution: int = 0) -> dict[str, float | int | bool]:
        del resolution
        lower, upper = self.local_bounds()
        return {
            "bounded": True,
            "closed": True,
            "components": 1,
            "vertices": int(self.vertices_m.shape[0]),
            "faces": int(self.faces.shape[0]),
            "volume_m3": self.hull_volume_m3,
            **{
                f"minimum_{axis}_m": float(lower[index])
                for index, axis in enumerate("xyz")
            },
            **{
                f"maximum_{axis}_m": float(upper[index])
                for index, axis in enumerate("xyz")
            },
        }

    def rescaled(self, scale_xyz: Sequence[float]) -> "NormalTemplateShape":
        factors = np.asarray(_tuple_values("scale_xyz", scale_xyz, 3), dtype=np.float64)
        combined = factors * np.asarray(self.scale_xyz, dtype=np.float64)
        if np.any((combined < 0.9) | (combined > 1.1)):
            raise RenderError("combined normal-template scale lies outside [0.9,1.1]")
        return NormalTemplateShape(
            vertices_m=self.vertices_m * factors,
            faces=self.faces,
            source_sequence_id=self.source_sequence_id,
            source_frame_id=self.source_frame_id,
            raw_semantic_id=self.raw_semantic_id,
            source_instance_id=self.source_instance_id,
            source_center_sensor_m=self.source_center_sensor_m,
            scale_xyz=tuple(map(float, combined)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "normal-template-convex-hull",
            "vertices_m": self.vertices_m.tolist(),
            "faces": self.faces.tolist(),
            "source_sequence_id": self.source_sequence_id,
            "source_frame_id": self.source_frame_id,
            "raw_semantic_id": self.raw_semantic_id,
            "source_instance_id": self.source_instance_id,
            "source_center_sensor_m": list(self.source_center_sensor_m),
            "scale_xyz": list(self.scale_xyz),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "NormalTemplateShape":
        plain = dict(value)
        if plain.pop("kind", None) != "normal-template-convex-hull":
            raise RenderError("normal-template JSON has the wrong geometry kind")
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid normal-template JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class HeldOutTorusShape:
    """Independent closed torus SDF reserved for generator-held-out diagnosis."""

    major_radius_m: float
    tube_radius_m: float

    def __post_init__(self) -> None:
        major = _finite_scalar("major_radius_m", self.major_radius_m)
        tube = _finite_scalar("tube_radius_m", self.tube_radius_m)
        if not 0.15 <= major <= 2.0 or not 0.04 <= tube < major:
            raise RenderError("held-out torus radii are outside their stable range")
        object.__setattr__(self, "major_radius_m", major)
        object.__setattr__(self, "tube_radius_m", tube)

    @classmethod
    def sample(
        cls,
        seed: int,
        *,
        size_m_range: tuple[float, float] = (0.4, 3.0),
    ) -> "HeldOutTorusShape":
        rng = np.random.default_rng(_integer("seed", seed))
        minimum, maximum = _tuple_values("size_m_range", size_m_range, 2)
        if not 0.4 <= minimum <= maximum <= 3.0:
            raise RenderError("held-out torus size range must lie in [0.4,3.0]")
        outer_radius = 0.5 * float(rng.uniform(minimum, maximum))
        tube = min(
            max(0.04, outer_radius * float(rng.uniform(0.18, 0.35))),
            outer_radius - 0.15,
        )
        return cls(outer_radius - tube, tube)

    @property
    def bound_radius_m(self) -> float:
        return self.major_radius_m + self.tube_radius_m + 1.0e-6

    def signed_distance(self, points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise RenderError("points_local must be finite [...,3]")
        radial = np.linalg.norm(points[..., :2], axis=-1) - self.major_radius_m
        return (
            np.sqrt(np.square(radial) + np.square(points[..., 2])) - self.tube_radius_m
        )

    def intersect(
        self,
        origin_local: np.ndarray,
        directions_local: np.ndarray,
        *,
        steps: int = 160,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _sampled_sdf_intersection(
            self.signed_distance,
            self.bound_radius_m,
            origin_local,
            directions_local,
            steps=steps,
        )

    def local_bounds(self, *, resolution: int = 0) -> tuple[np.ndarray, np.ndarray]:
        del resolution
        radius = self.major_radius_m + self.tube_radius_m
        return (
            np.asarray((-radius, -radius, -self.tube_radius_m), dtype=np.float64),
            np.asarray((radius, radius, self.tube_radius_m), dtype=np.float64),
        )

    def minimum_z_m(self, *, xy_resolution: int = 0, z_steps: int = 0) -> float:
        del xy_resolution, z_steps
        return -self.tube_radius_m

    def geometry_report(self, *, resolution: int = 0) -> dict[str, float | int | bool]:
        del resolution
        return {
            "bounded": True,
            "closed": True,
            "components": 1,
            "major_radius_m": self.major_radius_m,
            "tube_radius_m": self.tube_radius_m,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "held-out-torus-sdf",
            "major_radius_m": self.major_radius_m,
            "tube_radius_m": self.tube_radius_m,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "HeldOutTorusShape":
        plain = dict(value)
        if plain.pop("kind", None) != "held-out-torus-sdf":
            raise RenderError("held-out torus JSON has the wrong geometry kind")
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid held-out torus JSON: {error}") from error


InsertShape: TypeAlias = ShapeSpec | NormalTemplateShape | HeldOutTorusShape


def shape_from_dict(value: Mapping[str, object]) -> InsertShape:
    kind = value.get("kind") if isinstance(value, Mapping) else None
    if kind == "procedural-csg":
        return ShapeSpec.from_dict(value)
    if kind == "normal-template-convex-hull":
        return NormalTemplateShape.from_dict(value)
    if kind == "held-out-torus-sdf":
        return HeldOutTorusShape.from_dict(value)
    raise RenderError(f"unsupported geometry kind: {kind!r}")


def extract_normal_template_library(
    frames_206: Iterable[SourceFrame],
    *,
    minimum_points: int = 12,
    maximum_templates_per_class: int = 64,
) -> tuple[NormalTemplateShape, ...]:
    """Extract deterministic single-frame convex templates from labelled normal 206."""

    if type(minimum_points) is not int or minimum_points < 4:
        raise RenderError("minimum_points must be an integer >=4")
    if type(maximum_templates_per_class) is not int or maximum_templates_per_class < 1:
        raise RenderError("maximum_templates_per_class must be positive")
    result: list[NormalTemplateShape] = []
    counts = {semantic: 0 for semantic in NORMAL_TEMPLATE_SEMANTICS}
    for frame in frames_206:
        if frame.partition != "train" or frame.sequence_id != 206:
            raise RenderError("normal-template library input is not train/206")
        if frame.labels is None:
            raise RenderError("normal-template extraction requires train/206 labels")
        semantic = frame.labels.semantic
        instance = frame.labels.instance
        for packed in np.unique(
            semantic.astype(np.uint32) | (instance.astype(np.uint32) << np.uint32(16))
        ):
            raw = int(packed & np.uint32(0xFFFF))
            identifier = int(packed >> np.uint32(16))
            if (
                raw not in counts
                or identifier == 0
                or counts[raw] >= maximum_templates_per_class
            ):
                continue
            selected = (semantic == raw) & (instance == identifier)
            if int(np.count_nonzero(selected)) < minimum_points:
                continue
            try:
                template = NormalTemplateShape.from_source_frame(
                    frame,
                    raw_semantic_id=raw,
                    instance_id=identifier,
                )
            except RenderError:
                continue
            result.append(template)
            counts[raw] += 1
        if all(count >= maximum_templates_per_class for count in counts.values()):
            break
    if not result:
        raise RenderError("normal train/206 produced no valid convex instance template")
    return tuple(result)


def sample_training_anomaly_shape(
    seed: int,
    *,
    size_m_range: tuple[float, float] = (0.2, 3.0),
) -> ShapeSpec:
    """The training sampler is intentionally unable to emit held-out torus geometry."""

    return ShapeSpec.sample(seed, size_m_range=size_m_range)


def sample_held_out_anomaly_shape(
    seed: int,
    *,
    size_m_range: tuple[float, float] = (0.4, 3.0),
) -> HeldOutTorusShape:
    return HeldOutTorusShape.sample(seed, size_m_range=size_m_range)


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """Class-independent surface response within the calibrated intensity law."""

    intensity_quantile: float
    roughness: float
    return_bias: float = 0.0

    def __post_init__(self) -> None:
        quantile = _finite_scalar("intensity_quantile", self.intensity_quantile)
        roughness = _finite_scalar("roughness", self.roughness)
        return_bias = _finite_scalar("return_bias", self.return_bias)
        if not 0.0 <= quantile <= 1.0 or not 0.0 <= roughness <= 1.0:
            raise RenderError(
                "material intensity_quantile and roughness must lie in [0,1]"
            )
        if not -0.35 <= return_bias <= 0.35:
            raise RenderError("material return_bias must lie in [-0.35,0.35]")
        object.__setattr__(self, "intensity_quantile", quantile)
        object.__setattr__(self, "roughness", roughness)
        object.__setattr__(self, "return_bias", return_bias)

    @classmethod
    def sample(cls, seed: int) -> "MaterialSpec":
        rng = np.random.default_rng(_integer("seed", seed))
        return cls(
            float(rng.uniform(0.05, 0.95)),
            float(rng.uniform(0.05, 0.35)),
            float(rng.uniform(-0.18, 0.18)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "intensity_quantile": self.intensity_quantile,
            "roughness": self.roughness,
            "return_bias": self.return_bias,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MaterialSpec":
        if not isinstance(value, Mapping):
            raise RenderError("MaterialSpec JSON must be an object")
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid MaterialSpec JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """One labelled generator entity in the sequence world coordinate system."""

    object_id: int
    label: ObjectLabel
    shape: InsertShape
    material: MaterialSpec
    translation_world_m: tuple[float, float, float]
    rotation_world_from_local: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        identifier = _integer("object_id", self.object_id, minimum=1)
        if identifier > MAX_OBJECT_ID:
            raise RenderError(f"object_id must not exceed {MAX_OBJECT_ID}")
        label = str(self.label)
        if label not in OBJECT_LABELS:
            raise RenderError(f"object label must be one of {OBJECT_LABELS}")
        if not isinstance(
            self.shape, (ShapeSpec, NormalTemplateShape, HeldOutTorusShape)
        ) or not isinstance(self.material, MaterialSpec):
            raise TypeError("shape and material have unsupported types")
        if label == "normal-control" and not isinstance(
            self.shape, NormalTemplateShape
        ):
            raise RenderError(
                "normal-control objects require a 206 convex-hull template"
            )
        if label == "anomaly-proxy" and isinstance(self.shape, NormalTemplateShape):
            raise RenderError(
                "anomaly-proxy objects cannot use normal instance templates"
            )
        translation = _tuple_values("translation_world_m", self.translation_world_m, 3)
        rotation = _rotation_tuple(
            "rotation_world_from_local", self.rotation_world_from_local
        )
        object.__setattr__(self, "object_id", identifier)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "translation_world_m", translation)
        object.__setattr__(self, "rotation_world_from_local", rotation)

    @property
    def bounding_radius_m(self) -> float:
        return self.shape.bound_radius_m

    @property
    def normal_semantic_id(self) -> int | None:
        if isinstance(self.shape, NormalTemplateShape):
            return self.shape.raw_semantic_id
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "shape": self.shape.to_dict(),
            "material": self.material.to_dict(),
            "translation_world_m": list(self.translation_world_m),
            "rotation_world_from_local": [
                list(row) for row in self.rotation_world_from_local
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ObjectSpec":
        if not isinstance(value, Mapping):
            raise RenderError("ObjectSpec JSON must be an object")
        expected = {
            "object_id",
            "label",
            "shape",
            "material",
            "translation_world_m",
            "rotation_world_from_local",
        }
        if set(value) != expected:
            raise RenderError("ObjectSpec JSON has missing or unexpected fields")
        return cls(
            object_id=value["object_id"],  # type: ignore[arg-type]
            label=value["label"],  # type: ignore[arg-type]
            shape=shape_from_dict(value["shape"]),  # type: ignore[arg-type]
            material=MaterialSpec.from_dict(value["material"]),  # type: ignore[arg-type]
            translation_world_m=value["translation_world_m"],  # type: ignore[arg-type]
            rotation_world_from_local=value["rotation_world_from_local"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """The complete immutable procedural world shared by every frame/window."""

    seed: int
    source_sequence_id: int
    objects: tuple[ObjectSpec, ...] = ()
    tie_tolerance_m: float = 1.0e-6

    def __post_init__(self) -> None:
        seed = _integer("seed", self.seed)
        sequence = _integer("source_sequence_id", self.source_sequence_id)
        objects = tuple(self.objects)
        if any(not isinstance(item, ObjectSpec) for item in objects):
            raise RenderError("world objects must all be ObjectSpec values")
        normal_count = sum(item.label == "normal-control" for item in objects)
        anomaly_count = sum(item.label == "anomaly-proxy" for item in objects)
        if normal_count > 9 or anomaly_count > 9:
            raise RenderError(
                "normal-control and anomaly-proxy counts must each lie in [0,9]"
            )
        identifiers = [item.object_id for item in objects]
        if len(set(identifiers)) != len(identifiers):
            raise RenderError("world object IDs must be unique")
        tolerance = _finite_scalar("tie_tolerance_m", self.tie_tolerance_m)
        if not 0.0 < tolerance <= 1.0e-3:
            raise RenderError("tie_tolerance_m must lie in (0, 1e-3]")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "source_sequence_id", sequence)
        object.__setattr__(
            self, "objects", tuple(sorted(objects, key=lambda item: item.object_id))
        )
        object.__setattr__(self, "tie_tolerance_m", tolerance)

    @property
    def normal_control_count(self) -> int:
        return sum(item.label == "normal-control" for item in self.objects)

    @property
    def anomaly_proxy_count(self) -> int:
        return sum(item.label == "anomaly-proxy" for item in self.objects)

    @property
    def world_type(self) -> str:
        if not self.objects:
            return "pure_normal"
        if self.normal_control_count and self.anomaly_proxy_count:
            return "mixed"
        if self.normal_control_count:
            return "control_only"
        return "anomaly_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "format": WORLD_FORMAT,
            "seed": self.seed,
            "source_sequence_id": self.source_sequence_id,
            "objects": [item.to_dict() for item in self.objects],
            "tie_tolerance_m": self.tie_tolerance_m,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorldSpec":
        if not isinstance(value, Mapping) or value.get("format") != WORLD_FORMAT:
            raise RenderError("WorldSpec JSON has an unsupported format")
        if set(value) != {
            "format",
            "seed",
            "source_sequence_id",
            "objects",
            "tie_tolerance_m",
        }:
            raise RenderError("WorldSpec JSON has missing or unexpected fields")
        objects = value["objects"]
        if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
            raise RenderError("WorldSpec objects must be an array")
        return cls(
            seed=value["seed"],  # type: ignore[arg-type]
            source_sequence_id=value["source_sequence_id"],  # type: ignore[arg-type]
            objects=tuple(ObjectSpec.from_dict(item) for item in objects),  # type: ignore[arg-type]
            tie_tolerance_m=value["tie_tolerance_m"],  # type: ignore[arg-type]
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class RayGrid:
    """Calibrated Ouster-like rays kept in released STU file-slot order."""

    directions_sensor: np.ndarray
    beam_elevation_rad: np.ndarray
    azimuth_rad: np.ndarray
    beam_count: int = LASER_BEAMS
    calibration_frame_ids: tuple[int, ...] = ()
    beam_azimuth_offset_rad: np.ndarray | None = None
    origins_sensor: np.ndarray | None = None
    canonical_ray_by_slot: np.ndarray | None = None
    official_range_offset_m: float = 0.0

    def __post_init__(self) -> None:
        beams = _integer("beam_count", self.beam_count, minimum=1)
        directions = np.asarray(self.directions_sensor, dtype=np.float64)
        elevation = np.asarray(self.beam_elevation_rad, dtype=np.float64)
        azimuth = np.asarray(self.azimuth_rad, dtype=np.float64)
        if (
            directions.ndim != 2
            or directions.shape[1] != 3
            or directions.shape[0] % beams
        ):
            raise RenderError("directions_sensor must be [beam_count*columns,3]")
        columns = directions.shape[0] // beams
        if elevation.shape != (beams,) or azimuth.shape != (columns,):
            raise RenderError("beam elevation or azimuth lattice has the wrong shape")
        offset = (
            np.zeros(beams, dtype=np.float64)
            if self.beam_azimuth_offset_rad is None
            else np.asarray(self.beam_azimuth_offset_rad, dtype=np.float64)
        )
        if offset.shape != (beams,):
            raise RenderError("beam_azimuth_offset_rad must be [beam_count]")
        origins = (
            np.zeros_like(directions)
            if self.origins_sensor is None
            else np.asarray(self.origins_sensor, dtype=np.float64)
        )
        mapping = (
            np.arange(directions.shape[0], dtype=np.int32)
            if self.canonical_ray_by_slot is None
            else np.asarray(self.canonical_ray_by_slot)
        )
        if origins.shape != directions.shape or not np.isfinite(origins).all():
            raise RenderError("origins_sensor must be finite [slot,3]")
        if mapping.dtype != np.int32 or mapping.shape != (directions.shape[0],):
            raise TypeError("canonical_ray_by_slot must be int32[slot]")
        if np.unique(mapping).size != mapping.size or np.any(
            (mapping < 0) | (mapping >= directions.shape[0])
        ):
            raise RenderError("canonical_ray_by_slot must be a complete permutation")
        range_offset = _finite_scalar(
            "official_range_offset_m", self.official_range_offset_m
        )
        if range_offset < 0.0:
            raise RenderError("official_range_offset_m must be non-negative")
        if (
            not np.isfinite(directions).all()
            or not np.isfinite(elevation).all()
            or not np.isfinite(azimuth).all()
        ):
            raise RenderError("ray-grid values must be finite")
        if not np.isfinite(offset).all():
            raise RenderError("beam azimuth offsets must be finite")
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-7, rtol=1.0e-7):
            raise RenderError("every ray-grid direction must be a unit vector")
        frame_ids = tuple(
            _integer("calibration frame", item) for item in self.calibration_frame_ids
        )
        object.__setattr__(self, "beam_count", beams)
        object.__setattr__(self, "directions_sensor", _freeze(directions))
        object.__setattr__(self, "beam_elevation_rad", _freeze(elevation))
        object.__setattr__(self, "azimuth_rad", _freeze(azimuth))
        object.__setattr__(self, "calibration_frame_ids", frame_ids)
        object.__setattr__(self, "beam_azimuth_offset_rad", _freeze(offset))
        object.__setattr__(self, "origins_sensor", _freeze(origins))
        object.__setattr__(self, "canonical_ray_by_slot", _freeze(mapping))
        object.__setattr__(self, "official_range_offset_m", range_offset)

    @property
    def slot_count(self) -> int:
        return int(self.directions_sensor.shape[0])

    @property
    def columns(self) -> int:
        return self.slot_count // self.beam_count

    @property
    def slot_ids(self) -> np.ndarray:
        return _freeze(np.arange(self.slot_count, dtype=np.int32))

    @property
    def beam_ids(self) -> np.ndarray:
        return _freeze(np.arange(self.slot_count, dtype=np.int32) // self.columns)

    @property
    def column_ids(self) -> np.ndarray:
        return _freeze(np.arange(self.slot_count, dtype=np.int32) % self.columns)

    def slot_from_ray(self, beam_id: int, column_id: int) -> int:
        beam = _integer("beam_id", beam_id)
        column = _integer("column_id", column_id)
        if beam >= self.beam_count or column >= self.columns:
            raise IndexError((beam, column))
        return beam * self.columns + column

    def ray_from_slot(self, slot_id: int) -> tuple[int, int]:
        slot = _integer("slot_id", slot_id)
        if slot >= self.slot_count:
            raise IndexError(slot)
        return divmod(slot, self.columns)

    def directions_for(
        self,
        frame: SourceFrame,
        *,
        observed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the fixed calibrated directions aligned to released file slots."""

        if int(frame.xyzi.shape[0]) != self.slot_count:
            raise RenderError("frame and ray grid have different slot counts")
        if observed_mask is not None:
            supplied = np.asarray(observed_mask)
            if supplied.dtype != np.bool_ or supplied.shape != (self.slot_count,):
                raise RenderError("observed_mask must be bool[slot]")
        return self.directions_sensor

    def origins_for(self, frame: SourceFrame) -> np.ndarray:
        """Return calibrated beam origins aligned to released file slots."""

        if int(frame.xyzi.shape[0]) != self.slot_count:
            raise RenderError("frame and ray grid have different slot counts")
        assert self.origins_sensor is not None
        return self.origins_sensor

    def empty_slot_cross_validation(
        self,
        frame: SourceFrame,
        *,
        stride: int = 17,
        block_width: int = 1,
    ) -> dict[str, float | int]:
        """Mask native returns and measure the empty-slot interpolation error."""

        if type(stride) is not int or stride < 3:
            raise RenderError("cross-validation stride must be an integer >= 3")
        if type(block_width) is not int or not 1 <= block_width < stride - 1:
            raise RenderError("block_width must be an integer in [1, stride-2]")
        native = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        hidden = native & ((self.column_ids + 3 * self.beam_ids) % stride < block_width)
        if not bool(hidden.any()):
            raise RenderError("cross-validation mask selected no native return")
        observed = native & ~hidden
        predicted = self.directions_for(frame, observed_mask=observed)
        xyz = np.asarray(frame.xyzi[:, :3], dtype=np.float64)
        origins = self.origins_for(frame)
        vectors = xyz[hidden] - origins[hidden]
        truth = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        angle = np.arccos(np.clip(np.sum(predicted[hidden] * truth, axis=1), -1.0, 1.0))
        ranges = np.linalg.norm(xyz[hidden], axis=1)
        evaluation = (ranges >= 2.5) & (ranges <= 50.0)
        evaluation_angle = angle[evaluation]
        return {
            "hidden_returns": int(np.count_nonzero(hidden)),
            "median_direction_error_rad": float(np.median(angle)),
            "q99_direction_error_rad": float(np.quantile(angle, 0.99)),
            "maximum_direction_error_rad": float(np.max(angle, initial=0.0)),
            "evaluation_hidden_returns": int(np.count_nonzero(evaluation)),
            "evaluation_q99_direction_error_rad": float(
                np.quantile(evaluation_angle, 0.99) if evaluation_angle.size else 0.0
            ),
            "evaluation_maximum_direction_error_rad": float(
                np.max(evaluation_angle, initial=0.0)
            ),
        }

    def ranges(self, frame: SourceFrame) -> np.ndarray:
        if int(frame.xyzi.shape[0]) != self.slot_count:
            raise RenderError("frame and ray grid have different slot counts")
        xyz = np.asarray(frame.xyzi[:, :3], dtype=np.float64)
        assert self.origins_sensor is not None
        result = np.sum((xyz - self.origins_sensor) * self.directions_sensor, axis=1)
        result[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = 0.0
        if np.any(result < 0.0):
            raise RenderError("a published return lies behind its calibrated beam origin")
        return _freeze(result)

    def official_ranges(self, frame: SourceFrame) -> np.ndarray:
        """Return Ouster-form ranges, including the frozen beam-origin offset."""

        result = np.asarray(self.ranges(frame)).copy()
        valid = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        result[valid] += self.official_range_offset_m
        return _freeze(result)

    def range_image(self, frame: SourceFrame) -> np.ndarray:
        return _freeze(self.ranges(frame).reshape(self.beam_count, self.columns))

    def points_from_ranges(self, ranges: np.ndarray, frame: SourceFrame) -> np.ndarray:
        """Back-project along the fixed calibrated beam lines."""

        array = np.asarray(ranges, dtype=np.float64)
        if array.shape == (self.beam_count, self.columns):
            flat = array.reshape(-1)
            output_shape = (self.beam_count, self.columns, 3)
        elif array.shape == (self.slot_count,):
            flat = array
            output_shape = (self.slot_count, 3)
        else:
            raise RenderError("ranges must be [slot] or [beam,column]")
        if not np.isfinite(flat).all() or np.any(flat < 0.0):
            raise RenderError("ranges must be finite and non-negative")
        assert self.origins_sensor is not None
        points = self.origins_sensor + flat[:, None] * self.directions_sensor
        points[flat == 0.0] = 0.0
        return _freeze(points.reshape(output_shape))

    def round_trip(self, frame: SourceFrame) -> dict[str, float | int]:
        ranges = self.ranges(frame)
        directions = self.directions_for(frame)
        recovered = self.points_from_ranges(ranges, frame)
        xyz = np.asarray(frame.xyzi[:, :3], dtype=np.float64)
        valid = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        error = np.linalg.norm(recovered - xyz, axis=1)
        assert self.origins_sensor is not None
        recovered_range = np.sum(
            (recovered[valid] - self.origins_sensor[valid])
            * self.directions_sensor[valid],
            axis=1,
        )
        raw_vector = xyz[valid] - self.origins_sensor[valid]
        raw_unit = raw_vector / np.linalg.norm(raw_vector, axis=1, keepdims=True)
        angle = np.arccos(
            np.clip(np.sum(raw_unit * directions[valid], axis=1), -1.0, 1.0)
        )
        return {
            "slots": self.slot_count,
            "real_returns": int(np.count_nonzero(valid)),
            "recovered_real_returns": int(
                np.count_nonzero(np.linalg.norm(recovered, axis=1) > 0.0)
            ),
            "maximum_range_error_m": float(
                np.max(np.abs(recovered_range - ranges[valid]), initial=0.0)
            ),
            "maximum_direction_error_rad": float(np.max(angle, initial=0.0)),
            "maximum_point_error_m": float(np.max(error, initial=0.0)),
        }

    def assert_round_trip(
        self,
        frame: SourceFrame,
        *,
        point_atol_m: float = 1.0e-5,
        direction_atol_rad: float = 1.0e-7,
    ) -> None:
        report = self.round_trip(frame)
        if (
            report["real_returns"] != report["recovered_real_returns"]
            or float(report["maximum_point_error_m"]) > point_atol_m
            or float(report["maximum_direction_error_rad"]) > direction_atol_rad
        ):
            raise RenderError(f"range-image round trip failed: {report}")

    def audit(
        self,
        frames: Iterable[SourceFrame],
        *,
        sample_stride: int = 17,
    ) -> dict[str, object]:
        """Measure every recoverable ray/slot property without inventing a pass threshold."""

        if type(sample_stride) is not int or sample_stride < 1:
            raise RenderError("sample_stride must be a positive integer")
        slot_counts: list[int] = []
        empty_counts: list[int] = []
        empty_patterns: set[bytes] = set()
        row_elevation: list[list[float]] = [[] for _ in range(self.beam_count)]
        row_spread: list[float] = []
        azimuth_deviation: list[np.ndarray] = []
        azimuth_closure_error: list[np.ndarray] = []
        aligned_nominal_error: list[np.ndarray] = []
        aligned_direction_samples: list[np.ndarray] = []
        aligned_observed_samples: list[np.ndarray] = []
        duplicate_observed_directions: list[int] = []
        round_trips: list[dict[str, float | int]] = []
        expected_step = 2.0 * math.pi / self.columns
        nominal_azimuth = np.arctan2(
            self.directions_sensor[:, 1], self.directions_sensor[:, 0]
        ).reshape(self.beam_count, self.columns)
        frame_ids: list[int] = []
        for frame in frames:
            if int(frame.xyzi.shape[0]) != self.slot_count:
                raise RenderError("ray audit found a frame with a different slot count")
            frame_ids.append(int(frame.frame_id))
            slot_counts.append(int(frame.xyzi.shape[0]))
            empty = np.asarray(frame.zero_slot_mask, dtype=np.bool_).reshape(
                self.beam_count, self.columns
            )
            empty_counts.append(int(np.count_nonzero(empty)))
            empty_patterns.add(np.packbits(empty.reshape(-1)).tobytes())
            xyz = np.asarray(frame.xyzi[:, :3], dtype=np.float64).reshape(
                self.beam_count, self.columns, 3
            )
            ranges = np.linalg.norm(xyz, axis=2)
            observed = ranges > EPSILON
            elevation = np.arctan2(xyz[..., 2], np.linalg.norm(xyz[..., :2], axis=2))
            for beam in range(self.beam_count):
                selected = observed[beam]
                if bool(selected.any()):
                    median = float(np.median(elevation[beam, selected]))
                    row_elevation[beam].append(median)
                    row_spread.extend(
                        np.abs(elevation[beam, selected] - median).tolist()
                    )
            directions = self.directions_for(frame).reshape(
                self.beam_count, self.columns, 3
            )
            azimuth = np.unwrap(
                np.arctan2(directions[..., 1], directions[..., 0]), axis=1
            )
            steps = np.abs(np.diff(azimuth, axis=1))
            azimuth_deviation.append(
                np.abs(steps[:, ::sample_stride].reshape(-1) - expected_step)
            )
            closure = np.abs(np.angle(np.exp(1j * (azimuth[:, 0] - azimuth[:, -1]))))
            azimuth_closure_error.append(np.abs(closure - expected_step))
            raw_azimuth = np.arctan2(directions[..., 1], directions[..., 0])
            raw_elevation = np.arctan2(
                directions[..., 2], np.linalg.norm(directions[..., :2], axis=2)
            )
            phase = float(
                np.angle(
                    np.mean(np.exp(1j * (raw_azimuth - nominal_azimuth))[observed])
                )
            )
            aligned_azimuth = raw_azimuth - phase
            cosine = np.cos(raw_elevation)
            aligned_current = np.stack(
                (
                    cosine * np.cos(aligned_azimuth),
                    cosine * np.sin(aligned_azimuth),
                    np.sin(raw_elevation),
                ),
                axis=2,
            ).reshape(-1, 3)
            nominal = self.directions_sensor
            sampled = np.arange(0, self.slot_count, sample_stride)
            aligned_direction_samples.append(aligned_current[sampled])
            sampled_observed = observed.reshape(-1)[sampled]
            aligned_observed_samples.append(sampled_observed)
            aligned_nominal_error.append(
                np.arccos(
                    np.clip(
                        np.sum(
                            aligned_current[sampled][sampled_observed]
                            * nominal[sampled][sampled_observed],
                            axis=1,
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            observed_unit = directions.reshape(-1, 3)[observed.reshape(-1)]
            duplicate_observed_directions.append(
                int(
                    observed_unit.shape[0]
                    - np.unique(np.round(observed_unit, decimals=12), axis=0).shape[0]
                )
            )
            round_trips.append(self.round_trip(frame))
        if not frame_ids:
            raise RenderError("ray audit requires at least one frame")

        def summary(
            values: Sequence[float] | np.ndarray,
        ) -> dict[str, float | int | None]:
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if not array.size:
                return {"count": 0, "median": None, "q99": None, "maximum": None}
            return {
                "count": int(array.size),
                "median": float(np.median(array)),
                "q99": float(np.quantile(array, 0.99)),
                "maximum": float(np.max(array, initial=0.0)),
            }

        elevation_median = [
            float(np.median(values)) if values else None for values in row_elevation
        ]
        round_trip_mismatches = sum(
            self.ray_from_slot(self.slot_from_ray(beam, column)) != (beam, column)
            for beam in range(self.beam_count)
            for column in range(self.columns)
        )
        nominal_rounded = np.round(self.directions_sensor, decimals=12)
        duplicate_nominal = self.slot_count - int(
            np.unique(nominal_rounded, axis=0).shape[0]
        )
        aligned_reference = aligned_direction_samples[0]
        reference_observed = aligned_observed_samples[0]
        aligned_cross_frame_error = [
            np.arccos(
                np.clip(
                    np.sum(
                        current[observed & reference_observed]
                        * aligned_reference[observed & reference_observed],
                        axis=1,
                    ),
                    -1.0,
                    1.0,
                )
            )
            for current, observed in zip(
                aligned_direction_samples[1:], aligned_observed_samples[1:], strict=True
            )
            if bool((observed & reference_observed).any())
        ]
        return {
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "slot_layout": {
                "beam_count": self.beam_count,
                "columns": self.columns,
                "slot_count_by_frame": slot_counts,
                "bijection": "slot=beam*columns+column",
                "forward_reverse_mismatches": round_trip_mismatches,
            },
            "empty_slots": {
                "count_by_frame": empty_counts,
                "unique_patterns": len(empty_patterns),
            },
            "elevation_rows": {
                "row_count": self.beam_count,
                "nominal_row_rad": self.beam_elevation_rad.tolist(),
                "row_median_rad": elevation_median,
                "within_row_absolute_deviation_rad": summary(row_spread),
                "rows_with_observations": sum(
                    value is not None for value in elevation_median
                ),
            },
            "beam_periodicity": {
                "period_slots": self.columns,
                "rows": self.beam_count,
                "expected_azimuth_step_rad": expected_step,
            },
            "azimuth_continuity_absolute_error_rad": summary(
                np.concatenate(azimuth_deviation)
            ),
            "azimuth_period_closure_absolute_error_rad": summary(
                np.concatenate(azimuth_closure_error)
            ),
            "phase_aligned_cross_frame_direction_error_rad": summary(
                np.concatenate(aligned_cross_frame_error)
                if aligned_cross_frame_error
                else np.empty(0)
            ),
            "phase_aligned_nominal_direction_error_rad": summary(
                np.concatenate(aligned_nominal_error)
            ),
            "duplicate_and_multi_return": {
                "duplicate_nominal_directions": duplicate_nominal,
                "duplicate_observed_directions_by_frame": duplicate_observed_directions,
                "file_values_per_slot": 1,
                "multi_return_reordering_observable": False,
                "limitation": "released first-return slots cannot reveal discarded echo order",
            },
            "round_trip": {
                "by_frame": round_trips,
                "maximum_point_error_m": max(
                    float(item["maximum_point_error_m"]) for item in round_trips
                ),
                "maximum_direction_error_rad": max(
                    float(item["maximum_direction_error_rad"]) for item in round_trips
                ),
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "directions_sensor": self.directions_sensor.tolist(),
            "origins_sensor": self.origins_sensor.tolist(),
            "canonical_ray_by_slot": self.canonical_ray_by_slot.tolist(),
            "official_range_offset_m": self.official_range_offset_m,
            "beam_elevation_rad": self.beam_elevation_rad.tolist(),
            "azimuth_rad": self.azimuth_rad.tolist(),
            "beam_count": self.beam_count,
            "calibration_frame_ids": list(self.calibration_frame_ids),
            "beam_azimuth_offset_rad": self.beam_azimuth_offset_rad.tolist(),
            "layout": "beam_major",
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "RayGrid":
        if not isinstance(value, Mapping):
            raise RenderError("ray-grid payload must be an object")
        plain = dict(value)
        if plain.pop("layout", None) != "beam_major":
            raise RenderError("ray-grid payload does not use STU's beam-major layout")
        if "canonical_ray_by_slot" in plain:
            plain["canonical_ray_by_slot"] = np.asarray(
                plain["canonical_ray_by_slot"], dtype=np.int32
            )
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid ray-grid payload: {error}") from error


def calibrated_ray_grid_from_e11(path: Path | str) -> RayGrid:
    """Build the single authoritative ray grid from the frozen E11-D4b artifact."""

    artifact = np.load(Path(path).expanduser().resolve(strict=True), allow_pickle=False)
    required = {"even_params", "even_local", "integer_shift", "passed"}
    if not required.issubset(artifact.files) or not bool(artifact["passed"]):
        raise RenderError("E11-D4b artifact is incomplete or did not pass")
    parameters = np.asarray(artifact["even_params"], dtype=np.float64)
    local = np.asarray(artifact["even_local"], dtype=np.float64)
    shifts = np.asarray(artifact["integer_shift"])
    if parameters.shape != (3,) or local.shape != (LASER_BEAMS, 3):
        raise RenderError("E11-D4b artifact has invalid calibrated parameters")
    if shifts.shape != (LASER_BEAMS,) or not np.issubdtype(shifts.dtype, np.integer):
        raise RenderError("E11-D4b artifact has invalid row shifts")
    gamma, origin_x, origin_z = map(float, parameters)
    columns = 1024
    raw_column = np.arange(columns, dtype=np.float64)[None, :]
    shift = shifts.astype(np.float64)[:, None]
    # The D4b artifact stores gamma in Ouster's encoder gauge; Cartesian bearing
    # has the fixed pi offset used by the formal D4b/D4c/v3 evaluations.
    eta = math.pi + gamma - 2.0 * math.pi * raw_column / columns + shift * (
        2.0 * math.pi / columns
    )
    cosine = np.cos(eta)
    sine = np.sin(eta)
    directions = np.empty((LASER_BEAMS, columns, 3), dtype=np.float64)
    directions[..., 0] = cosine * local[:, None, 0] - sine * local[:, None, 1]
    directions[..., 1] = sine * local[:, None, 0] + cosine * local[:, None, 1]
    directions[..., 2] = local[:, None, 2]
    origins = np.stack(
        (origin_x * cosine, origin_x * sine, np.full_like(cosine, origin_z)), axis=2
    )
    raw_to_column = (
        np.arange(columns, dtype=np.int32)[None, :] - shifts.astype(np.int32)[:, None]
    ) % columns
    mapping = (
        np.arange(LASER_BEAMS, dtype=np.int32)[:, None] * columns + raw_to_column
    ).reshape(-1)
    elevation = np.arcsin(np.clip(local[:, 2], -1.0, 1.0))
    beam_offset = np.arctan2(local[:, 1], local[:, 0])
    azimuth = (
        math.pi
        + gamma
        - 2.0 * math.pi * np.arange(columns, dtype=np.float64) / columns
    )
    return RayGrid(
        directions.reshape(-1, 3),
        elevation,
        azimuth,
        beam_count=LASER_BEAMS,
        calibration_frame_ids=tuple(range(449)),
        beam_azimuth_offset_rad=beam_offset,
        origins_sensor=origins.reshape(-1, 3),
        canonical_ray_by_slot=mapping.astype(np.int32),
        official_range_offset_m=math.hypot(origin_x, origin_z),
    )


@dataclass(frozen=True, slots=True)
class SensorCalibration:
    """Beam/range/incidence return and intensity laws estimated from normal 206."""

    range_edges_m: np.ndarray
    incidence_edges_rad: np.ndarray
    quantile_levels: np.ndarray
    return_probability: np.ndarray
    intensity_quantiles: np.ndarray
    opportunity_counts: np.ndarray
    return_counts: np.ndarray
    fallback_mask: np.ndarray
    intensity_min: float
    intensity_max: float
    source_sequence_id: int = 206
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ranges = np.asarray(self.range_edges_m, dtype=np.float64)
        incidence = np.asarray(self.incidence_edges_rad, dtype=np.float64)
        levels = np.asarray(self.quantile_levels, dtype=np.float64)
        probability = np.asarray(self.return_probability, dtype=np.float64)
        quantiles = np.asarray(self.intensity_quantiles, dtype=np.float32)
        opportunities = np.asarray(self.opportunity_counts, dtype=np.int64)
        returns = np.asarray(self.return_counts, dtype=np.int64)
        fallback = np.asarray(self.fallback_mask, dtype=np.bool_)
        cells = (LASER_BEAMS, ranges.size - 1, incidence.size - 1)
        if (
            ranges.ndim != 1
            or incidence.ndim != 1
            or levels.ndim != 1
            or min(ranges.size, incidence.size, levels.size) < 2
            or not np.isfinite(ranges).all()
            or not np.isfinite(incidence).all()
            or not np.isfinite(levels).all()
            or np.any(np.diff(ranges) <= 0.0)
            or np.any(np.diff(incidence) <= 0.0)
            or np.any(np.diff(levels) <= 0.0)
        ):
            raise RenderError("sensor calibration axes must be finite and increasing")
        if (
            probability.shape != cells
            or opportunities.shape != cells
            or returns.shape != cells
            or fallback.shape != cells
            or quantiles.shape != (*cells, levels.size)
        ):
            raise RenderError("sensor calibration arrays have inconsistent shapes")
        if (
            not np.isfinite(probability).all()
            or np.any((probability < 0.0) | (probability > 1.0))
            or np.any(opportunities < 0)
            or np.any(returns < 0)
            or np.any(returns > opportunities)
            or not np.isfinite(quantiles).all()
            or np.any(np.diff(quantiles, axis=3) < -1.0e-6)
            or not math.isclose(float(levels[0]), 0.0, abs_tol=1.0e-8)
            or not math.isclose(float(levels[-1]), 1.0, abs_tol=1.0e-8)
        ):
            raise RenderError("sensor probabilities, counts, or quantiles are invalid")
        minimum = _finite_scalar("intensity_min", self.intensity_min)
        maximum = _finite_scalar("intensity_max", self.intensity_max)
        if minimum < 0.0 or maximum < minimum:
            raise RenderError("intensity support is invalid")
        if _integer("source_sequence_id", self.source_sequence_id) != 206:
            raise RenderError("sensor calibration must come from normal train/206")
        provenance = tuple((str(key), str(value)) for key, value in self.provenance)
        if len({key for key, _ in provenance}) != len(provenance):
            raise RenderError("sensor calibration provenance keys must be unique")
        for name, value in (
            ("range_edges_m", ranges),
            ("incidence_edges_rad", incidence),
            ("quantile_levels", levels),
            ("return_probability", probability),
            ("intensity_quantiles", quantiles),
            ("opportunity_counts", opportunities),
            ("return_counts", returns),
            ("fallback_mask", fallback),
        ):
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "intensity_min", minimum)
        object.__setattr__(self, "intensity_max", maximum)
        object.__setattr__(self, "provenance", tuple(sorted(provenance)))

    @classmethod
    def constant(
        cls,
        intensity: float,
        *,
        return_probability: float = 1.0,
    ) -> "SensorCalibration":
        """Create an explicit beam-conditioned fixture for geometry tests."""

        value = _finite_scalar("intensity", intensity)
        chance = _finite_scalar("return_probability", return_probability)
        if value < 0.0 or not 0.0 <= chance <= 1.0:
            raise RenderError("constant sensor fixture is outside its domain")
        ranges = np.asarray(DEFAULT_RANGE_EDGES_M, dtype=np.float64)
        incidence = np.asarray(DEFAULT_INCIDENCE_EDGES_RAD, dtype=np.float64)
        levels = np.linspace(0.0, 1.0, 17, dtype=np.float64)
        cells = (LASER_BEAMS, ranges.size - 1, incidence.size - 1)
        return cls(
            ranges,
            incidence,
            levels,
            np.full(cells, chance, dtype=np.float64),
            np.full((*cells, levels.size), value, dtype=np.float32),
            np.ones(cells, dtype=np.int64),
            np.ones(cells, dtype=np.int64)
            if chance > 0.0
            else np.zeros(cells, dtype=np.int64),
            np.zeros(cells, dtype=np.bool_),
            value,
            value,
            provenance=(("fixture", "constant"),),
        )

    def _bins(
        self,
        beam_ids: np.ndarray,
        ranges_m: np.ndarray,
        incidence_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beam = np.asarray(beam_ids, dtype=np.int64)
        ranges = np.asarray(ranges_m, dtype=np.float64)
        incidence = np.asarray(incidence_rad, dtype=np.float64)
        if beam.shape != ranges.shape or beam.shape != incidence.shape:
            raise RenderError("sensor lookup inputs must have the same shape")
        if (
            np.any((beam < 0) | (beam >= LASER_BEAMS))
            or not np.isfinite(ranges).all()
            or not np.isfinite(incidence).all()
            or np.any(ranges < 0.0)
            or np.any(incidence < 0.0)
        ):
            raise RenderError("sensor lookup inputs are outside their domains")
        range_bin = np.clip(
            np.searchsorted(self.range_edges_m, ranges, side="right") - 1,
            0,
            self.range_edges_m.size - 2,
        )
        incidence_bin = np.clip(
            np.searchsorted(self.incidence_edges_rad, incidence, side="right") - 1,
            0,
            self.incidence_edges_rad.size - 2,
        )
        return beam, range_bin, incidence_bin

    def return_chance(
        self,
        beam_ids: np.ndarray,
        ranges_m: np.ndarray,
        incidence_rad: np.ndarray,
        material_return_bias: float,
    ) -> np.ndarray:
        beam, range_bin, incidence_bin = self._bins(beam_ids, ranges_m, incidence_rad)
        base = self.return_probability[beam, range_bin, incidence_bin]
        bias = _finite_scalar("material_return_bias", material_return_bias)
        clipped = np.clip(base, 1.0e-5, 1.0 - 1.0e-5)
        modulated = 1.0 / (
            1.0 + np.exp(-(np.log(clipped / (1.0 - clipped)) + 2.0 * bias))
        )
        return _freeze(modulated)

    def sample_intensity(
        self,
        beam_ids: np.ndarray,
        ranges_m: np.ndarray,
        incidence_rad: np.ndarray,
        uniform: np.ndarray,
        material: MaterialSpec,
    ) -> np.ndarray:
        beam, range_bin, incidence_bin = self._bins(beam_ids, ranges_m, incidence_rad)
        random = np.asarray(uniform, dtype=np.float64)
        if (
            random.shape != beam.shape
            or not np.isfinite(random).all()
            or np.any((random < 0.0) | (random > 1.0))
        ):
            raise RenderError("intensity uniforms must be finite values in [0,1]")
        quantile = np.clip(
            material.intensity_quantile + material.roughness * (random - 0.5),
            0.0,
            1.0,
        )
        output = np.empty(beam.shape, dtype=np.float32)
        for key in set(
            zip(
                beam.reshape(-1),
                range_bin.reshape(-1),
                incidence_bin.reshape(-1),
                strict=True,
            )
        ):
            b, r_bin, i_bin = map(int, key)
            selected = (beam == b) & (range_bin == r_bin) & (incidence_bin == i_bin)
            output[selected] = np.interp(
                quantile[selected],
                self.quantile_levels,
                self.intensity_quantiles[b, r_bin, i_bin],
            ).astype(np.float32)
        np.clip(output, self.intensity_min, self.intensity_max, out=output)
        return _freeze(output)

    def to_payload(self) -> dict[str, object]:
        return {
            "range_edges_m": self.range_edges_m.tolist(),
            "incidence_edges_rad": self.incidence_edges_rad.tolist(),
            "quantile_levels": self.quantile_levels.tolist(),
            "return_probability": self.return_probability.tolist(),
            "intensity_quantiles": self.intensity_quantiles.tolist(),
            "opportunity_counts": self.opportunity_counts.tolist(),
            "return_counts": self.return_counts.tolist(),
            "fallback_mask": self.fallback_mask.tolist(),
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "source_sequence_id": self.source_sequence_id,
            "provenance": [list(item) for item in self.provenance],
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "SensorCalibration":
        if not isinstance(value, Mapping):
            raise RenderError("sensor calibration payload must be an object")
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid sensor calibration payload: {error}") from error


def _surface_measurements(
    frame: SourceFrame, ray_grid: RayGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate incidence and explicit smooth-surface opportunities on one range image."""

    if int(frame.xyzi.shape[0]) != ray_grid.slot_count:
        raise RenderError("frame and ray grid have different slot counts")
    points = np.asarray(frame.xyzi[:, :3], dtype=np.float64).reshape(
        ray_grid.beam_count, ray_grid.columns, 3
    )
    ranges = np.linalg.norm(points, axis=2)
    valid = ranges > EPSILON
    horizontal = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    vertical = np.zeros_like(points)
    vertical[1:-1] = points[2:] - points[:-2]
    vertical[0] = points[1] - points[0]
    vertical[-1] = points[-1] - points[-2]
    neighbour_valid = (
        valid
        & np.roll(valid, -1, axis=1)
        & np.roll(valid, 1, axis=1)
        & np.vstack((valid[1], valid[:-1]))
        & np.vstack((valid[1:], valid[-2]))
    )
    normal = np.cross(horizontal, vertical)
    length = np.linalg.norm(normal, axis=2)
    usable = neighbour_valid & (length > 1.0e-6)
    normal[usable] /= length[usable, None]
    rays = ray_grid.directions_for(frame).reshape(
        ray_grid.beam_count, ray_grid.columns, 3
    )
    incidence = np.full(valid.shape, np.nan, dtype=np.float64)
    incidence[usable] = np.arccos(
        np.clip(np.abs(np.sum(normal[usable] * -rays[usable], axis=1)), 0.0, 1.0)
    )
    left_range = np.roll(ranges, 1, axis=1)
    right_range = np.roll(ranges, -1, axis=1)
    continuity = np.abs(left_range - right_range) <= np.maximum(
        0.5, 0.05 * 0.5 * (left_range + right_range)
    )
    # The immediate neighbours define the missing surface range. Their normal
    # stencils touch the empty slot, so incidence comes from the next samples out.
    left_incidence = np.roll(incidence, 2, axis=1)
    right_incidence = np.roll(incidence, -2, axis=1)
    potential = (
        ~valid
        & np.roll(valid, 1, axis=1)
        & np.roll(valid, -1, axis=1)
        & continuity
        & np.isfinite(left_incidence)
        & np.isfinite(right_incidence)
    )
    potential_range = 0.5 * (left_range + right_range)
    potential_incidence = 0.5 * (left_incidence + right_incidence)
    intensity = np.asarray(frame.xyzi[:, 3], dtype=np.float32).reshape(valid.shape)
    return (
        ranges,
        incidence,
        intensity,
        potential_range,
        np.where(potential, potential_incidence, np.nan),
    )


def calibrate_sensor(
    frames: Iterable[SourceFrame],
    ray_grid: RayGrid,
    *,
    source_sequence_id: int = 206,
    provenance: Mapping[str, object] | None = None,
    range_edges_m: Sequence[float] = DEFAULT_RANGE_EDGES_M,
    incidence_edges_rad: Sequence[float] = DEFAULT_INCIDENCE_EDGES_RAD,
    quantile_count: int = 257,
    maximum_samples_per_cell: int = 4096,
    minimum_cell_count: int = 64,
    seed: int = 0,
) -> SensorCalibration:
    """Fit beam-conditioned return and intensity laws from normal train/206."""

    if not isinstance(ray_grid, RayGrid):
        raise TypeError("ray_grid must be RayGrid")
    if ray_grid.beam_count != LASER_BEAMS:
        raise RenderError("formal sensor calibration requires the OS1-128 beam grid")
    if _integer("source_sequence_id", source_sequence_id) != 206:
        raise RenderError("sensor calibration must come from normal train/206")
    if type(quantile_count) is not int or quantile_count < 17:
        raise RenderError("quantile_count must be an integer >=17")
    if type(maximum_samples_per_cell) is not int or maximum_samples_per_cell < 64:
        raise RenderError("maximum_samples_per_cell must be an integer >=64")
    if type(minimum_cell_count) is not int or minimum_cell_count < 1:
        raise RenderError("minimum_cell_count must be positive")
    range_edges = np.asarray(range_edges_m, dtype=np.float64)
    incidence_edges = np.asarray(incidence_edges_rad, dtype=np.float64)
    if (
        range_edges.ndim != 1
        or incidence_edges.ndim != 1
        or range_edges.size < 2
        or incidence_edges.size < 2
        or not np.isfinite(range_edges).all()
        or not np.isfinite(incidence_edges).all()
        or np.any(np.diff(range_edges) <= 0.0)
        or np.any(np.diff(incidence_edges) <= 0.0)
    ):
        raise RenderError("sensor calibration edges must be finite and increasing")
    cells = (ray_grid.beam_count, range_edges.size - 1, incidence_edges.size - 1)
    opportunities = np.zeros(cells, dtype=np.int64)
    returns = np.zeros(cells, dtype=np.int64)
    values: dict[tuple[int, int, int], np.ndarray] = {}
    priorities: dict[tuple[int, int, int], np.ndarray] = {}
    rng = np.random.default_rng(_integer("seed", seed))
    frame_ids: list[int] = []
    support_min = math.inf
    support_max = -math.inf

    def bin_samples(
        beam: np.ndarray,
        ranges: np.ndarray,
        incidence: np.ndarray,
        *,
        returned: bool,
        intensity: np.ndarray | None = None,
    ) -> None:
        nonlocal support_min, support_max
        valid = (
            np.isfinite(ranges)
            & np.isfinite(incidence)
            & (ranges >= range_edges[0])
            & (ranges <= range_edges[-1])
            & (incidence >= incidence_edges[0])
            & (incidence <= incidence_edges[-1])
        )
        beam = beam[valid]
        ranges = ranges[valid]
        incidence = incidence[valid]
        observed_intensity = None if intensity is None else intensity[valid]
        range_bin = np.clip(
            np.searchsorted(range_edges, ranges, side="right") - 1,
            0,
            range_edges.size - 2,
        )
        incidence_bin = np.clip(
            np.searchsorted(incidence_edges, incidence, side="right") - 1,
            0,
            incidence_edges.size - 2,
        )
        for key_values in set(zip(beam, range_bin, incidence_bin, strict=True)):
            key = tuple(map(int, key_values))
            selected = (
                (beam == key[0]) & (range_bin == key[1]) & (incidence_bin == key[2])
            )
            count = int(np.count_nonzero(selected))
            opportunities[key] += count
            if not returned:
                continue
            returns[key] += count
            assert observed_intensity is not None
            new_values = observed_intensity[selected].astype(np.float32, copy=False)
            support_min = min(support_min, float(np.min(new_values)))
            support_max = max(support_max, float(np.max(new_values)))
            new_priorities = rng.random(new_values.size)
            combined_values = np.concatenate(
                (values.get(key, np.empty(0, dtype=np.float32)), new_values)
            )
            combined_priorities = np.concatenate(
                (priorities.get(key, np.empty(0, dtype=np.float64)), new_priorities)
            )
            if combined_values.size > maximum_samples_per_cell:
                keep = np.argpartition(combined_priorities, -maximum_samples_per_cell)[
                    -maximum_samples_per_cell:
                ]
                combined_values = combined_values[keep]
                combined_priorities = combined_priorities[keep]
            values[key] = combined_values
            priorities[key] = combined_priorities

    beam_grid = np.broadcast_to(
        np.arange(ray_grid.beam_count, dtype=np.int64)[:, None],
        (ray_grid.beam_count, ray_grid.columns),
    )
    for frame in frames:
        if frame.partition != "train" or frame.sequence_id != 206:
            raise RenderError("sensor calibration received a frame outside train/206")
        frame_ids.append(int(frame.frame_id))
        ranges, incidence, intensity, potential_range, potential_incidence = (
            _surface_measurements(frame, ray_grid)
        )
        returned = np.isfinite(incidence)
        bin_samples(
            beam_grid[returned],
            ranges[returned],
            incidence[returned],
            returned=True,
            intensity=intensity[returned],
        )
        potential = np.isfinite(potential_incidence)
        bin_samples(
            beam_grid[potential],
            potential_range[potential],
            potential_incidence[potential],
            returned=False,
        )
    if not frame_ids or not values or not math.isfinite(support_min):
        raise RenderError(
            "normal train/206 produced no usable sensor calibration samples"
        )

    pooled_opportunities = opportunities.sum(axis=0)
    pooled_returns = returns.sum(axis=0)
    probability = np.empty(cells, dtype=np.float64)
    fallback = opportunities < minimum_cell_count
    for beam in range(cells[0]):
        local = ~fallback[beam]
        probability[beam, local] = (returns[beam, local] + 0.5) / (
            opportunities[beam, local] + 1.0
        )
        probability[beam, ~local] = (pooled_returns[~local] + 0.5) / (
            pooled_opportunities[~local] + 1.0
        )
    levels = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    table = np.empty((*cells, quantile_count), dtype=np.float32)
    populated = tuple(values)
    for beam in range(cells[0]):
        for range_bin in range(cells[1]):
            for incidence_bin in range(cells[2]):
                key = (beam, range_bin, incidence_bin)
                reference = values.get(key)
                if reference is None or returns[key] < minimum_cell_count:
                    peers = [
                        sample
                        for (other_beam, r_bin, i_bin), sample in values.items()
                        if r_bin == range_bin and i_bin == incidence_bin
                    ]
                    reference = np.concatenate(peers) if peers else None
                if reference is None or not reference.size:
                    nearest = min(
                        abs(range_bin - key_item[1]) + abs(incidence_bin - key_item[2])
                        for key_item in populated
                    )
                    reference = np.concatenate(
                        [
                            sample
                            for key_item, sample in values.items()
                            if abs(range_bin - key_item[1])
                            + abs(incidence_bin - key_item[2])
                            == nearest
                        ]
                    )
                table[beam, range_bin, incidence_bin] = np.quantile(
                    reference, levels
                ).astype(np.float32)
    details = {
        "partition": "train",
        "sequence": "206",
        "frames": str(len(frame_ids)),
        "first_frame": str(min(frame_ids)),
        "last_frame": str(max(frame_ids)),
        "empty_ray_opportunity": (
            "same_beam_two_sided_valid_neighbours_with_range_consistency_then_range_interpolation"
        ),
        "empty_ray_incidence": "mean_of_finite_next_outward_same_beam_incidence_samples",
        "empty_ray_range_consistency": "absolute_difference_le_max_0.5m_or_5pct_mean_range",
        "return_estimator": "jeffreys_beta_smoothed_binomial_rate",
        "return_low_count_fallback": (
            f"cross_beam_same_range_incidence_below_{minimum_cell_count}_opportunities"
        ),
        "intensity_low_count_fallback": (
            f"cross_beam_same_range_incidence_below_{minimum_cell_count}_returns"
        ),
        "empty_intensity_cell_fallback": "nearest_populated_range_incidence_bin_cross_beam",
        "sampling": f"priority_reservoir_{maximum_samples_per_cell}",
    }
    if provenance is not None:
        details.update(
            {
                str(key): json.dumps(value, ensure_ascii=False, sort_keys=True)
                for key, value in provenance.items()
            }
        )
    return SensorCalibration(
        range_edges,
        incidence_edges,
        levels,
        probability,
        table,
        opportunities,
        returns,
        fallback,
        support_min,
        support_max,
        provenance=tuple(details.items()),
    )


def save_sensor_calibration(
    path: Path | str,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
) -> None:
    """Atomically write the reconstructible runtime calibration payload."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production environment has torch.
        raise RuntimeError("saving calibration requires PyTorch") from error
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format": CALIBRATION_FORMAT,
        "ray_grid": ray_grid.to_payload(),
        "sensor": sensor.to_payload(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_sensor_calibration(
    path: Path | str,
) -> tuple[RayGrid, SensorCalibration]:
    """Load a plain-data calibration without accepting executable pickle types."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production environment has torch.
        raise RuntimeError("loading calibration requires PyTorch") from error
    payload = torch.load(
        Path(path).expanduser().resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or payload.get("format") != CALIBRATION_FORMAT:
        raise RenderError("sensor calibration has an unsupported format")
    return (
        RayGrid.from_payload(payload["ray_grid"]),  # type: ignore[arg-type]
        SensorCalibration.from_payload(payload["sensor"]),  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class _ObjectCompetition:
    distance_m: np.ndarray
    normal_world: np.ndarray
    object_id: np.ndarray
    geometric_hits: Mapping[int, int]
    accepted_hits: Mapping[int, int]


def _accepted_object_hits(
    origin_world: np.ndarray,
    directions_world: np.ndarray,
    world: WorldSpec,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    frame_id: int,
) -> _ObjectCompetition:
    """Accept each object's returns independently before nearest-return competition."""

    count = directions_world.shape[0]
    origins = np.asarray(origin_world, dtype=np.float64)
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, directions_world.shape)
    if origins.shape != directions_world.shape or not np.isfinite(origins).all():
        raise RenderError("ray origins and directions must be aligned finite [slot,3]")
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_normal = np.zeros((count, 3), dtype=np.float64)
    best_object = np.full(count, -1, dtype=np.int32)
    geometric_hits: dict[int, int] = {}
    accepted_hits: dict[int, int] = {}
    slots = np.arange(count, dtype=np.int32)
    beam_ids = ray_grid.beam_ids
    for item in world.objects:
        translation = np.asarray(item.translation_world_m, dtype=np.float64)
        rotation = np.asarray(item.rotation_world_from_local, dtype=np.float64)
        local_origin = (origins - translation) @ rotation
        local_direction = directions_world @ rotation
        distance, local_normal, valid = item.shape.intersect(
            local_origin, local_direction
        )
        world_normal = local_normal @ rotation.T
        incidence = np.zeros(count, dtype=np.float64)
        incidence[valid] = np.arccos(
            np.clip(
                np.abs(np.sum(world_normal[valid] * -directions_world[valid], axis=1)),
                0.0,
                1.0,
            )
        )
        chance = np.zeros(count, dtype=np.float64)
        chance[valid] = sensor.return_chance(
            beam_ids[valid],
            distance[valid],
            incidence[valid],
            item.material.return_bias,
        )
        uniform = _slot_uniform(
            world,
            frame_id,
            slots,
            np.full(count, item.object_id, dtype=np.int32),
            channel=0,
        )
        accepted = valid & (uniform < chance)
        geometric_hits[item.object_id] = int(np.count_nonzero(valid))
        accepted_hits[item.object_id] = int(np.count_nonzero(accepted))
        comparable = accepted & np.isfinite(best_distance)
        tied = np.zeros(count, dtype=np.bool_)
        tied[comparable] = (
            np.abs(distance[comparable] - best_distance[comparable])
            <= world.tie_tolerance_m
        )
        closer = accepted & (
            (distance < best_distance - world.tie_tolerance_m)
            | tied & ((best_object < 0) | (item.object_id < best_object))
        )
        if bool(closer.any()):
            best_distance[closer] = distance[closer]
            best_normal[closer] = world_normal[closer]
            best_object[closer] = item.object_id
    return _ObjectCompetition(
        _freeze(best_distance),
        _freeze(best_normal),
        _freeze(best_object),
        geometric_hits,
        accepted_hits,
    )


def _slot_uniform(
    world: WorldSpec,
    frame_id: int,
    slots: np.ndarray,
    object_ids: np.ndarray,
    *,
    channel: int,
) -> np.ndarray:
    """Map stable identities to reproducible U[0,1) values without global RNG state."""

    mask = (1 << 64) - 1
    stream = _integer("channel", channel)
    base = (
        world.seed
        ^ (world.source_sequence_id << 24)
        ^ (frame_id << 40)
        ^ (stream * 0xA24BAED4963EE407)
    ) & mask
    with np.errstate(over="ignore"):
        value = np.asarray(slots, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        value ^= np.asarray(object_ids, dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03)
        value ^= np.uint64(base)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return (value.astype(np.float64) + 0.5) / float(2**64)


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    """One counterfactual scan with masks aligned to every original file slot."""

    source: SourceFrame
    packed_labels: np.ndarray
    normal_control_mask: np.ndarray
    anomaly_proxy_mask: np.ndarray
    inserted_mask: np.ndarray
    occluded_original_mask: np.ndarray
    unchanged_normal_mask: np.ndarray
    object_id_internal: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.source.xyzi.shape[0])
        packed = np.asarray(self.packed_labels)
        normal_control = np.asarray(self.normal_control_mask)
        anomaly_proxy = np.asarray(self.anomaly_proxy_mask)
        inserted = np.asarray(self.inserted_mask)
        occluded = np.asarray(self.occluded_original_mask)
        unchanged = np.asarray(self.unchanged_normal_mask)
        object_id = np.asarray(self.object_id_internal)
        if packed.dtype != np.uint32 or packed.shape != (count,):
            raise TypeError("packed_labels must be uint32[slot]")
        for name, value in (
            ("normal_control_mask", normal_control),
            ("anomaly_proxy_mask", anomaly_proxy),
            ("inserted_mask", inserted),
            ("occluded_original_mask", occluded),
            ("unchanged_normal_mask", unchanged),
        ):
            if value.dtype != np.bool_ or value.shape != (count,):
                raise TypeError(f"{name} must be bool[slot]")
        if object_id.dtype != np.int32 or object_id.shape != (count,):
            raise TypeError("object_id_internal must be int32[slot]")
        if np.any(normal_control & anomaly_proxy):
            raise RenderError("normal-control and anomaly-proxy masks overlap")
        if not np.array_equal(inserted, normal_control | anomaly_proxy):
            raise RenderError("inserted mask must be the union of both entity labels")
        if np.any(occluded & ~inserted) or np.any(inserted & unchanged):
            raise RenderError("render masks have contradictory slot semantics")
        if np.any((object_id >= 0) != inserted):
            raise RenderError(
                "internal object IDs must identify exactly inserted slots"
            )
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16)
        if not np.all(semantic[anomaly_proxy] == np.uint16(2)):
            raise RenderError("anomaly-proxy returns must carry raw semantic 2")
        if not np.all(
            np.isin(semantic[normal_control], tuple(NORMAL_TEMPLATE_SEMANTICS))
        ):
            raise RenderError(
                "normal-control returns must preserve a legal normal class"
            )
        if self.source.labels is not None and not np.array_equal(
            self.source.labels.packed, packed
        ):
            raise RenderError("SourceFrame labels and packed_labels differ")
        for name, value in (
            ("packed_labels", packed),
            ("normal_control_mask", normal_control),
            ("anomaly_proxy_mask", anomaly_proxy),
            ("inserted_mask", inserted),
            ("occluded_original_mask", occluded),
            ("unchanged_normal_mask", unchanged),
            ("object_id_internal", object_id),
        ):
            object.__setattr__(self, name, _freeze(value))

    @property
    def changed_mask(self) -> np.ndarray:
        return self.inserted_mask

    @property
    def frame_id(self) -> int:
        return int(self.source.frame_id)

    @property
    def xyzi(self) -> np.ndarray:
        return self.source.xyzi

    @property
    def slot_ids(self) -> np.ndarray:
        """Return file-slot indices for I/O alignment, not canonical point IDs."""

        return _freeze(np.arange(self.xyzi.shape[0], dtype=np.int32))

    @property
    def visible_slots(self) -> np.ndarray:
        """Return visible file slots for I/O alignment only."""

        return self.source.real_slots


def render_frame(
    source: SourceFrame,
    world: WorldSpec,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
) -> RenderedFrame:
    """Deterministically render one frame of a fixed complete virtual world."""

    if not isinstance(world, WorldSpec) or not isinstance(ray_grid, RayGrid):
        raise TypeError("world and ray_grid must be WorldSpec and RayGrid")
    if not isinstance(sensor, SensorCalibration):
        raise TypeError("sensor must be SensorCalibration from normal train/206")
    if int(source.sequence_id) != world.source_sequence_id:
        raise RenderError(
            "source frame and counterfactual world use different sequences"
        )
    if world.source_sequence_id in {201, 206} and source.partition != "train":
        raise RenderError(
            "formal counterfactual worlds require identified train frames"
        )
    if int(source.xyzi.shape[0]) != ray_grid.slot_count:
        raise RenderError("source frame and ray grid have different slot counts")
    rotation, lidar_origin_world = _pose(source)
    directions_sensor = ray_grid.directions_for(source)
    directions_world = directions_sensor @ rotation.T
    origins_sensor = ray_grid.origins_for(source)
    origins_world = origins_sensor @ rotation.T + lidar_origin_world
    competition = _accepted_object_hits(
        origins_world,
        directions_world,
        world,
        ray_grid,
        sensor,
        int(source.frame_id),
    )
    normal_range = np.asarray(ray_grid.ranges(source)).copy()
    normal_range[np.asarray(source.zero_slot_mask, dtype=np.bool_)] = np.inf
    inserted = np.isfinite(competition.distance_m) & (
        competition.distance_m < normal_range - world.tie_tolerance_m
    )
    slots = np.flatnonzero(inserted).astype(np.int32)
    object_by_id = {item.object_id: item for item in world.objects}
    normal_control = np.zeros(source.xyzi.shape[0], dtype=np.bool_)
    anomaly_proxy = np.zeros(source.xyzi.shape[0], dtype=np.bool_)
    for item in world.objects:
        won = inserted & (competition.object_id == item.object_id)
        if item.label == "normal-control":
            normal_control |= won
        else:
            anomaly_proxy |= won
    xyzi = np.asarray(source.xyzi, dtype=np.float32).copy()
    original_real = ~np.asarray(source.zero_slot_mask, dtype=np.bool_)
    xyzi[original_real, :3] = (
        origins_sensor[original_real]
        + normal_range[original_real, None] * directions_sensor[original_real]
    ).astype(np.float32)
    if slots.size:
        xyzi[slots, :3] = (
            origins_sensor[slots]
            + competition.distance_m[slots, None] * directions_sensor[slots]
        ).astype(np.float32)
        incidence = np.arccos(
            np.clip(
                np.abs(
                    np.sum(
                        competition.normal_world[slots] * -directions_world[slots],
                        axis=1,
                    )
                ),
                0.0,
                1.0,
            )
        )
        for item in world.objects:
            selected = competition.object_id[slots] == item.object_id
            if not bool(selected.any()):
                continue
            object_slots = slots[selected]
            random_quantile = _slot_uniform(
                world,
                int(source.frame_id),
                object_slots,
                np.full(object_slots.size, item.object_id, dtype=np.int32),
                channel=1,
            )
            xyzi[object_slots, 3] = sensor.sample_intensity(
                ray_grid.beam_ids[object_slots],
                competition.distance_m[object_slots],
                incidence[selected],
                random_quantile,
                item.material,
            )

    labels = source.labels
    if labels is None:
        packed = np.zeros(source.xyzi.shape[0], dtype=np.uint32)
        semantic = None
        rendered_labels = None
    else:
        packed = labels.packed.copy()
        semantic = labels.semantic.copy()
        instance = labels.instance.copy()
        semantic_target = (
            None if labels.semantic_target is None else labels.semantic_target.copy()
        )
        instance[slots] = (
            SYNTHETIC_INSTANCE_BASE + competition.object_id[slots]
        ).astype(np.uint16)
        for slot in slots:
            item = object_by_id[int(competition.object_id[slot])]
            if item.label == "anomaly-proxy":
                semantic[slot] = np.uint16(2)
                if semantic_target is not None:
                    semantic_target[slot] = np.uint8(255)
            else:
                normal_semantic = item.normal_semantic_id
                assert normal_semantic is not None
                semantic[slot] = np.uint16(normal_semantic)
                if semantic_target is not None:
                    semantic_target[slot] = np.uint8(
                        NORMAL_SEMANTIC_TARGET[normal_semantic]
                    )
        packed = semantic.astype(np.uint32) | (
            instance.astype(np.uint32) << np.uint32(16)
        )
        rendered_labels = PointLabels(
            packed=_freeze(packed),
            semantic=_freeze(semantic),
            instance=_freeze(instance),
            semantic_target=(
                None if semantic_target is None else _freeze(semantic_target)
            ),
        )
    if labels is None:
        for slot in slots:
            item = object_by_id[int(competition.object_id[slot])]
            semantic_id = (
                2 if item.label == "anomaly-proxy" else item.normal_semantic_id
            )
            assert semantic_id is not None
            packed[slot] = np.uint32(semantic_id) | np.uint32(
                SYNTHETIC_INSTANCE_BASE + item.object_id
            ) << np.uint32(16)
    occluded = inserted & original_real
    unchanged = original_real & ~inserted
    if semantic is not None:
        original_semantic = labels.semantic
        unchanged &= (original_semantic != np.uint16(0)) & (
            original_semantic != np.uint16(2)
        )
    object_id = np.full(source.xyzi.shape[0], -1, dtype=np.int32)
    object_id[slots] = competition.object_id[slots]
    rendered_source = make_source_frame(
        int(source.frame_id),
        _freeze(xyzi),
        source.lidar_pose,
        rendered_labels,
        partition=source.partition,
        sequence_id=source.sequence_id,
    )
    return RenderedFrame(
        source=rendered_source,
        packed_labels=_freeze(packed),
        normal_control_mask=_freeze(normal_control),
        anomaly_proxy_mask=_freeze(anomaly_proxy),
        inserted_mask=_freeze(inserted),
        occluded_original_mask=_freeze(occluded),
        unchanged_normal_mask=_freeze(unchanged),
        object_id_internal=_freeze(object_id),
    )


def render_frames(
    frames: Iterable[SourceFrame],
    world: WorldSpec,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
) -> Iterable[RenderedFrame]:
    """Stream deterministic frames without materializing a complete sequence."""

    for frame in frames:
        yield render_frame(frame, world, ray_grid, sensor)


@dataclass(frozen=True, slots=True)
class SupportPoints:
    ground_world: np.ndarray
    ground_semantic: np.ndarray
    obstacle_world: np.ndarray

    def __post_init__(self) -> None:
        ground = np.asarray(self.ground_world, dtype=np.float64)
        semantic = np.asarray(self.ground_semantic, dtype=np.uint16)
        obstacle = np.asarray(self.obstacle_world, dtype=np.float64)
        if (
            ground.ndim != 2
            or ground.shape[1] != 3
            or semantic.shape != (ground.shape[0],)
        ):
            raise RenderError(
                "ground support coordinates and semantics are not aligned"
            )
        if obstacle.ndim != 2 or obstacle.shape[1] != 3:
            raise RenderError("obstacle support coordinates must be [N,3]")
        if not np.isfinite(ground).all() or not np.isfinite(obstacle).all():
            raise RenderError("support coordinates must be finite")
        object.__setattr__(self, "ground_world", _freeze(ground))
        object.__setattr__(self, "ground_semantic", _freeze(semantic))
        object.__setattr__(self, "obstacle_world", _freeze(obstacle))


def collect_support_context(
    frames: Iterable[SourceFrame],
    *,
    ground_semantic_ids: Sequence[int] = GROUND_SEMANTIC_IDS,
    maximum_points_per_class: int = 500_000,
) -> SupportPoints:
    """Collect bounded world ground semantics and observed obstacle samples."""

    if type(maximum_points_per_class) is not int or maximum_points_per_class < 1:
        raise RenderError("maximum_points_per_class must be positive")
    ground_ids = np.asarray(
        tuple(int(item) for item in ground_semantic_ids), dtype=np.uint16
    )
    if ground_ids.size == 0:
        raise RenderError("ground_semantic_ids must not be empty")
    ground_chunks: list[np.ndarray] = []
    semantic_chunks: list[np.ndarray] = []
    obstacle_chunks: list[np.ndarray] = []

    def compact_ground() -> None:
        count = sum(item.shape[0] for item in ground_chunks)
        if count <= 2 * maximum_points_per_class:
            return
        joined = np.concatenate(ground_chunks, axis=0)
        semantics = np.concatenate(semantic_chunks, axis=0)
        keep = np.linspace(
            0, joined.shape[0] - 1, maximum_points_per_class, dtype=np.int64
        )
        ground_chunks[:] = [joined[keep]]
        semantic_chunks[:] = [semantics[keep]]

    def compact_obstacles() -> None:
        chunks = obstacle_chunks
        count = sum(item.shape[0] for item in chunks)
        if count <= 2 * maximum_points_per_class:
            return
        joined = np.concatenate(chunks, axis=0)
        keep = np.linspace(
            0, joined.shape[0] - 1, maximum_points_per_class, dtype=np.int64
        )
        chunks[:] = [joined[keep]]

    for frame in frames:
        if frame.labels is None:
            raise PlacementError("ground placement requires normal semantic labels")
        rotation, translation = _pose(frame)
        real = np.asarray(frame.real_slots, dtype=np.int32)
        sensor = np.asarray(frame.xyzi[real, :3], dtype=np.float64)
        world = sensor @ rotation.T + translation
        semantic = frame.labels.semantic[real]
        ground = np.isin(semantic, ground_ids)
        obstacle = (semantic != np.uint16(0)) & ~ground
        ground_chunks.append(world[ground])
        semantic_chunks.append(semantic[ground])
        obstacle_chunks.append(world[obstacle])
        compact_ground()
        compact_obstacles()
    if not ground_chunks or sum(item.shape[0] for item in ground_chunks) < 3:
        raise PlacementError(
            "normal frames contain fewer than three stable ground points"
        )

    def finish(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return _freeze(np.empty((0, 3), dtype=np.float64))
        joined = np.concatenate(chunks, axis=0)
        if joined.shape[0] > maximum_points_per_class:
            keep = np.linspace(
                0, joined.shape[0] - 1, maximum_points_per_class, dtype=np.int64
            )
            joined = joined[keep]
        return _freeze(joined, np.float64)

    ground = finish(ground_chunks)
    semantics = np.concatenate(semantic_chunks, axis=0)
    if semantics.size > maximum_points_per_class:
        keep = np.linspace(
            0, semantics.size - 1, maximum_points_per_class, dtype=np.int64
        )
        semantics = semantics[keep]
    return SupportPoints(
        ground, _freeze(semantics.astype(np.uint16)), finish(obstacle_chunks)
    )


def collect_support_points(
    frames: Iterable[SourceFrame],
    *,
    ground_semantic_ids: Sequence[int] = GROUND_SEMANTIC_IDS,
    maximum_points_per_class: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the legacy coordinate pair while retaining one authoritative collector."""

    context = collect_support_context(
        frames,
        ground_semantic_ids=ground_semantic_ids,
        maximum_points_per_class=maximum_points_per_class,
    )
    return context.ground_world, context.obstacle_world


def normal_control_support_semantics(raw_semantic_id: int) -> frozenset[int]:
    semantic = _integer("raw_semantic_id", raw_semantic_id)
    if semantic in VEHICLE_TEMPLATE_SEMANTICS:
        return VEHICLE_SUPPORT_SEMANTICS
    if semantic in PERSON_RIDER_TEMPLATE_SEMANTICS:
        return PERSON_RIDER_SUPPORT_SEMANTICS
    raise RenderError("normal-control semantic has no support-surface policy")


def fit_support_plane(
    ground_points_world: np.ndarray,
    anchor_world: np.ndarray,
    *,
    radius_m: float = 2.0,
    maximum_rms_m: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a robust local support plane and orient its normal upward."""

    points = np.asarray(ground_points_world, dtype=np.float64)
    anchor = np.asarray(anchor_world, dtype=np.float64)
    radius = _finite_scalar("radius_m", radius_m)
    maximum_rms = _finite_scalar("maximum_rms_m", maximum_rms_m)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PlacementError("ground_points_world must be finite [N,3]")
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise PlacementError("anchor_world must be finite [3]")
    if radius <= 0.0 or maximum_rms <= 0.0:
        raise PlacementError("support radius and residual bound must be positive")
    selected = np.linalg.norm(points[:, :2] - anchor[:2], axis=1) <= radius
    local = points[selected]
    if local.shape[0] < 12:
        raise PlacementError("candidate has fewer than 12 nearby ground samples")
    center = np.median(local, axis=0)
    _, _, vectors = np.linalg.svd(local - center, full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0.0:
        normal = -normal
    residual = np.abs((local - center) @ normal)
    median = float(np.median(residual))
    threshold = max(0.02, 3.0 * 1.4826 * float(np.median(np.abs(residual - median))))
    inlier = residual <= threshold
    if np.count_nonzero(inlier) >= 12:
        local = local[inlier]
        center = local.mean(axis=0)
        _, _, vectors = np.linalg.svd(local - center, full_matrices=False)
        normal = vectors[-1]
        if normal[2] < 0.0:
            normal = -normal
    rms = float(np.sqrt(np.mean(np.square((local - center) @ normal))))
    if normal[2] < 0.7 or rms > maximum_rms:
        raise PlacementError("candidate support is too steep or non-planar")
    # Use the fitted plane at the candidate xy, not the median point's height.
    contact = anchor.copy()
    contact[2] = (
        center[2]
        - (normal[0] * (contact[0] - center[0]) + normal[1] * (contact[1] - center[1]))
        / normal[2]
    )
    return _freeze(contact), _freeze(normal), rms


def _ground_rotation(normal: np.ndarray, yaw: float) -> np.ndarray:
    up = np.array(normal, dtype=np.float64, copy=True)
    up /= np.linalg.norm(up)
    heading = np.asarray((math.cos(yaw), math.sin(yaw), 0.0), dtype=np.float64)
    tangent_x = heading - up * float(np.dot(heading, up))
    if np.linalg.norm(tangent_x) <= EPSILON:
        heading = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        tangent_x = heading - up * float(np.dot(heading, up))
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(up, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    return np.column_stack((tangent_x, tangent_y, up))


def place_object(
    shape: InsertShape,
    material: MaterialSpec,
    ground_points_world: np.ndarray,
    non_ground_points_world: np.ndarray,
    *,
    object_id: int,
    label: ObjectLabel,
    seed: int,
    ground_semantic_ids: np.ndarray | None = None,
    allowed_support_semantics: Sequence[int] | None = None,
    existing_objects: Sequence[ObjectSpec] = (),
    support_radius_m: float = 2.0,
    penetration_tolerance_m: float = 0.03,
    collision_margin_m: float = 0.05,
    maximum_candidates: int = 128,
) -> ObjectSpec:
    """Place an object's numerical bottom on legal ground without intersections."""

    if not isinstance(
        shape, (ShapeSpec, NormalTemplateShape, HeldOutTorusShape)
    ) or not isinstance(material, MaterialSpec):
        raise TypeError("shape and material have unsupported types")
    label_value = str(label)
    if label_value not in OBJECT_LABELS:
        raise PlacementError(f"label must be one of {OBJECT_LABELS}")
    ground = np.asarray(ground_points_world, dtype=np.float64)
    obstacles = np.asarray(non_ground_points_world, dtype=np.float64)
    if ground.ndim != 2 or ground.shape[1] != 3 or ground.shape[0] < 3:
        raise PlacementError("ground_points_world must be [N,3] with N>=3")
    if obstacles.ndim != 2 or obstacles.shape[1] != 3:
        raise PlacementError("non_ground_points_world must be [M,3]")
    if not np.isfinite(ground).all() or not np.isfinite(obstacles).all():
        raise PlacementError("placement points must be finite")
    ground_semantic = None
    if ground_semantic_ids is not None:
        ground_semantic = np.asarray(ground_semantic_ids, dtype=np.uint16)
        if ground_semantic.shape != (ground.shape[0],):
            raise PlacementError("ground semantic IDs must align with ground points")
    allowed = None
    if label_value == "normal-control":
        if not isinstance(shape, NormalTemplateShape):
            raise PlacementError("normal-control placement requires a normal template")
        allowed = set(
            normal_control_support_semantics(shape.raw_semantic_id)
            if allowed_support_semantics is None
            else map(int, allowed_support_semantics)
        )
        if ground_semantic is None:
            raise PlacementError(
                "normal-control placement requires support semantic IDs"
            )
    elif allowed_support_semantics is not None:
        allowed = set(map(int, allowed_support_semantics))
        if ground_semantic is None:
            raise PlacementError("support policy requires aligned ground semantic IDs")
    if type(maximum_candidates) is not int or maximum_candidates < 1:
        raise PlacementError("maximum_candidates must be positive")
    penetration = _finite_scalar("penetration_tolerance_m", penetration_tolerance_m)
    margin = _finite_scalar("collision_margin_m", collision_margin_m)
    if penetration < 0.0 or margin < 0.0:
        raise PlacementError(
            "penetration tolerance and collision margin must be non-negative"
        )
    rng = np.random.default_rng(_integer("seed", seed))
    eligible = np.arange(ground.shape[0])
    if allowed is not None:
        assert ground_semantic is not None
        eligible = eligible[np.isin(ground_semantic, tuple(sorted(allowed)))]
    if not eligible.size:
        raise PlacementError("no ground point satisfies the entity support policy")
    order = rng.permutation(eligible)[:maximum_candidates]
    lower_z = shape.minimum_z_m()
    for candidate in order:
        try:
            contact, normal, _ = fit_support_plane(
                ground, ground[candidate], radius_m=support_radius_m
            )
        except PlacementError:
            continue
        rotation = _ground_rotation(normal, float(rng.uniform(-math.pi, math.pi)))
        translation = contact - normal * lower_z
        proposed = ObjectSpec(
            object_id=object_id,
            label=label_value,  # type: ignore[arg-type]
            shape=shape,
            material=material,
            translation_world_m=tuple(map(float, translation)),
            rotation_world_from_local=tuple(tuple(map(float, row)) for row in rotation),
        )
        collision = False
        for other in existing_objects:
            separation = np.linalg.norm(
                translation - np.asarray(other.translation_world_m, dtype=np.float64)
            )
            if (
                separation
                < proposed.bounding_radius_m + other.bounding_radius_m + margin
            ):
                collision = True
                break
        if collision:
            continue
        if obstacles.size:
            nearby = np.linalg.norm(obstacles - translation, axis=1) <= (
                proposed.bounding_radius_m + margin
            )
            if bool(nearby.any()):
                local = (obstacles[nearby] - translation) @ rotation
                if bool(np.any(shape.signed_distance(local) < -penetration)):
                    continue
        return proposed
    raise PlacementError(
        "no candidate passed support, penetration, and collision checks"
    )


def validate_world_visibility(
    world: WorldSpec,
    frames: Iterable[SourceFrame],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    minimum_returns_per_object: int = 1,
) -> dict[int, int]:
    """Require each entity to win nearest-return competition at least once."""

    if type(minimum_returns_per_object) is not int or minimum_returns_per_object < 1:
        raise RenderError("minimum_returns_per_object must be positive")
    counts = {item.object_id: 0 for item in world.objects}
    frame_count = 0
    for frame in frames:
        if frame.sequence_id != world.source_sequence_id:
            raise RenderError("visibility frame and world use different sequences")
        if int(frame.xyzi.shape[0]) != ray_grid.slot_count:
            raise RenderError(
                "visibility frame and ray grid have different slot counts"
            )
        rotation, lidar_origin = _pose(frame)
        directions_world = ray_grid.directions_for(frame) @ rotation.T
        origins_world = ray_grid.origins_for(frame) @ rotation.T + lidar_origin
        competition = _accepted_object_hits(
            origins_world,
            directions_world,
            world,
            ray_grid,
            sensor,
            int(frame.frame_id),
        )
        normal_range = np.asarray(ray_grid.ranges(frame)).copy()
        normal_range[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = np.inf
        visible = np.isfinite(competition.distance_m) & (
            competition.distance_m < normal_range - world.tie_tolerance_m
        )
        for identifier in counts:
            counts[identifier] += int(
                np.count_nonzero(visible & (competition.object_id == identifier))
            )
        frame_count += 1
    if frame_count == 0:
        raise RenderError("world visibility validation requires at least one frame")
    missing = {
        identifier: count
        for identifier, count in counts.items()
        if count < minimum_returns_per_object
    }
    if missing:
        raise PlacementError(
            f"world objects have insufficient visible returns: {missing}"
        )
    return counts


def _training_entity_counts(world_type: WorldType, seed: int) -> tuple[int, int]:
    """Freeze label counts with 90% of nonzero draws in the one-to-three range."""

    if world_type not in WORLD_TYPES:
        raise RenderError(f"world_type must be one of {WORLD_TYPES}")
    if world_type == "pure_normal":
        return 0, 0
    rng = np.random.default_rng(_integer("seed", seed))
    values = np.arange(1, 10)
    probability = np.asarray((0.36, 0.32, 0.22, 0.03, 0.02, 0.015, 0.01, 0.01, 0.015))

    def draw() -> int:
        return int(rng.choice(values, p=probability))

    normal_count = draw() if world_type in {"control_only", "mixed"} else 0
    anomaly_count = draw() if world_type in {"mixed", "anomaly_only"} else 0
    return normal_count, anomaly_count


def sample_training_world(
    support_frames_206: Sequence[SourceFrame],
    normal_template_library: Sequence[NormalTemplateShape],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    world_type: WorldType,
    seed: int,
    *,
    maximum_attempts: int = 48,
    minimum_returns_per_object: int = 1,
) -> WorldSpec:
    """Sample one deterministic train/206 world, pairing mixed labels before residuals."""

    frames = tuple(support_frames_206)
    templates = tuple(normal_template_library)
    world_seed = _integer("seed", seed)
    if not isinstance(ray_grid, RayGrid) or not isinstance(sensor, SensorCalibration):
        raise TypeError("ray_grid and sensor must be calibrated renderer values")
    if not frames:
        raise RenderError("training world sampling requires support frames")
    if any(
        frame.partition != "train"
        or frame.sequence_id != 206
        or int(frame.xyzi.shape[0]) != ray_grid.slot_count
        for frame in frames
    ):
        raise RenderError("training world support must be aligned normal train/206")
    if any(
        right.frame_id != left.frame_id + 1 for left, right in zip(frames, frames[1:])
    ):
        raise RenderError("training world support frames must be consecutive")
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise RenderError("maximum_attempts must be positive")
    if type(minimum_returns_per_object) is not int or minimum_returns_per_object < 1:
        raise RenderError("minimum_returns_per_object must be positive")
    normal_count, anomaly_count = _training_entity_counts(world_type, world_seed)
    if (normal_count or anomaly_count) and (
        not templates
        or any(not isinstance(item, NormalTemplateShape) for item in templates)
        or any(item.source_sequence_id != 206 for item in templates)
    ):
        raise RenderError("training entities require a 206 normal-template library")
    if normal_count == anomaly_count == 0:
        return WorldSpec(world_seed, 206)

    context = collect_support_context(frames)
    for attempt in range(maximum_attempts):
        attempt_seed = world_seed + 1_000_003 * attempt
        rng = np.random.default_rng(attempt_seed)
        anchor = context.ground_world[
            int(rng.integers(0, context.ground_world.shape[0]))
        ]
        local_ground = (
            np.linalg.norm(context.ground_world[:, :2] - anchor[:2], axis=1) <= 20.0
        )
        if int(np.count_nonzero(local_ground)) < 64:
            local_ground = np.ones(context.ground_world.shape[0], dtype=np.bool_)
        ground = context.ground_world[local_ground]
        ground_semantic = context.ground_semantic[local_ground]
        objects: list[ObjectSpec] = []
        try:
            _, reference_origin = _pose(frames[len(frames) // 2])
            range_edges = np.asarray(sensor.range_edges_m, dtype=np.float64)
            ground_distance_tier = np.clip(
                np.searchsorted(
                    range_edges,
                    np.linalg.norm(ground - reference_origin, axis=1),
                    side="right",
                )
                - 1,
                0,
                range_edges.size - 2,
            )

            def distance_tier(point: np.ndarray) -> int:
                distance = float(np.linalg.norm(point - reference_origin))
                return int(
                    np.clip(
                        np.searchsorted(range_edges, distance, side="right") - 1,
                        0,
                        range_edges.size - 2,
                    )
                )

            def normal_shape(local_rng: np.random.Generator) -> NormalTemplateShape:
                template = templates[int(local_rng.integers(0, len(templates)))]
                target_scale = local_rng.uniform(0.9, 1.1, size=3)
                return template.rescaled(
                    tuple(target_scale / np.asarray(template.scale_xyz))
                )

            def maximum_extent(shape: InsertShape) -> float:
                lower, upper = shape.local_bounds()
                return float(np.max(upper - lower))

            def matched_proxy(
                reference: NormalTemplateShape, shape_seed: int
            ) -> ShapeSpec:
                target = float(np.clip(maximum_extent(reference), 0.2, 3.0))
                size_range = (max(0.2, 0.85 * target), min(3.0, 1.15 * target))
                for retry in range(16):
                    proxy = sample_training_anomaly_shape(
                        shape_seed + 7_919 * retry,
                        size_m_range=size_range,
                    )
                    extent = maximum_extent(proxy)
                    if size_range[0] <= extent <= size_range[1]:
                        return proxy
                raise RenderError("proxy extent left its paired size range")

            def support_region(
                reference: NormalTemplateShape,
                local_rng: np.random.Generator,
            ) -> tuple[np.ndarray, np.ndarray, int, int]:
                allowed = normal_control_support_semantics(reference.raw_semantic_id)
                eligible = np.flatnonzero(
                    np.isin(ground_semantic, tuple(sorted(allowed)))
                )
                if not eligible.size:
                    raise PlacementError(
                        "local world has no category-compatible support"
                    )
                for candidate in local_rng.permutation(eligible)[:128]:
                    support_semantic = int(ground_semantic[candidate])
                    tier = int(ground_distance_tier[candidate])
                    same_surface = ground_semantic == np.uint16(support_semantic)
                    same_tier = ground_distance_tier == tier
                    nearby = (
                        np.linalg.norm(
                            ground[:, :2] - ground[int(candidate), :2], axis=1
                        )
                        <= 8.0
                    )
                    selected = same_surface & same_tier & nearby
                    if int(np.count_nonzero(selected)) >= 24:
                        return (
                            ground[selected],
                            ground_semantic[selected],
                            support_semantic,
                            tier,
                        )
                raise PlacementError(
                    "no local single-surface distance-tier support region"
                )

            # Consecutive internal IDs form pairs; which label is placed first is random.
            pair_count = min(normal_count, anomaly_count)
            for pair_index in range(pair_count):
                pair_seed = attempt_seed + 100_003 * (pair_index + 1)
                pair_rng = np.random.default_rng(pair_seed)
                control_shape = normal_shape(pair_rng)
                proxy_shape = matched_proxy(control_shape, pair_seed + 3)
                pair_ground, pair_semantic, support_semantic, tier = support_region(
                    control_shape, pair_rng
                )
                label_order: list[ObjectLabel] = [
                    "normal-control",
                    "anomaly-proxy",
                ]
                if int(pair_rng.integers(0, 2)):
                    label_order.reverse()
                materials = (
                    MaterialSpec.sample(pair_seed + 11),
                    MaterialSpec.sample(pair_seed + 12),
                )
                placed: dict[str, ObjectSpec] = {}
                first_position: np.ndarray | None = None
                for position, label in enumerate(label_order):
                    candidate_ground = pair_ground
                    candidate_semantic = pair_semantic
                    if first_position is not None:
                        near_first = (
                            np.linalg.norm(
                                pair_ground[:, :2] - first_position[:2], axis=1
                            )
                            <= 6.0
                        )
                        if int(np.count_nonzero(near_first)) < 12:
                            raise PlacementError(
                                "paired support has too few nearby candidates"
                            )
                        candidate_ground = pair_ground[near_first]
                        candidate_semantic = pair_semantic[near_first]
                    shape: InsertShape = (
                        control_shape if label == "normal-control" else proxy_shape
                    )
                    item = place_object(
                        shape,
                        materials[position],
                        candidate_ground,
                        context.obstacle_world,
                        object_id=len(objects) + 1,
                        label=label,
                        seed=pair_seed + 31 + position,
                        ground_semantic_ids=candidate_semantic,
                        allowed_support_semantics=(support_semantic,),
                        existing_objects=objects,
                    )
                    objects.append(item)
                    placed[label] = item
                    if first_position is None:
                        first_position = np.asarray(
                            item.translation_world_m, dtype=np.float64
                        )
                control = placed["normal-control"]
                proxy = placed["anomaly-proxy"]
                control_position = np.asarray(
                    control.translation_world_m, dtype=np.float64
                )
                proxy_position = np.asarray(proxy.translation_world_m, dtype=np.float64)
                if (
                    np.linalg.norm(control_position[:2] - proxy_position[:2]) > 6.25
                    or distance_tier(control_position) != tier
                    or distance_tier(proxy_position) != tier
                ):
                    raise PlacementError(
                        "paired entities left their shared local distance tier"
                    )

            residual_labels: list[ObjectLabel] = ["normal-control"] * (
                normal_count - pair_count
            ) + ["anomaly-proxy"] * (anomaly_count - pair_count)
            rng.shuffle(residual_labels)
            for residual_index, label in enumerate(residual_labels):
                entity_seed = attempt_seed + 10_007 * (residual_index + 1)
                entity_rng = np.random.default_rng(entity_seed)
                # A shadow normal template keeps unmatched proxies on normal laws.
                reference_shape = normal_shape(entity_rng)
                shape = (
                    reference_shape
                    if label == "normal-control"
                    else matched_proxy(reference_shape, entity_seed + 3)
                )
                entity_ground, entity_semantic, support_semantic, tier = support_region(
                    reference_shape, entity_rng
                )
                item = place_object(
                    shape,
                    MaterialSpec.sample(entity_seed + 11),
                    entity_ground,
                    context.obstacle_world,
                    object_id=len(objects) + 1,
                    label=label,
                    seed=entity_seed + 31,
                    ground_semantic_ids=entity_semantic,
                    allowed_support_semantics=(support_semantic,),
                    existing_objects=objects,
                )
                if (
                    distance_tier(
                        np.asarray(item.translation_world_m, dtype=np.float64)
                    )
                    != tier
                ):
                    raise PlacementError(
                        "unpaired entity left its sampled distance tier"
                    )
                objects.append(item)
            world = WorldSpec(world_seed, 206, tuple(objects))
            if world.world_type != world_type:
                raise AssertionError("training sampler produced the wrong world type")
            validate_world_visibility(
                world,
                frames,
                ray_grid,
                sensor,
                minimum_returns_per_object=minimum_returns_per_object,
            )
        except (RenderError, PlacementError):
            continue
        return world
    raise PlacementError(
        f"training {world_type} world failed {maximum_attempts} deterministic attempts"
    )


def five_frame_world_diagnostics(
    world: WorldSpec,
    frames: Sequence[SourceFrame],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
) -> dict[str, object]:
    """Compute explicit Nvis, O, d, and V definitions for one five-frame window."""

    window = tuple(frames)
    if len(window) != 5 or any(
        right.frame_id != left.frame_id + 1 for left, right in zip(window, window[1:])
    ):
        raise RenderError("world diagnostics require five consecutive source frames")
    if any(frame.sequence_id != world.source_sequence_id for frame in window):
        raise RenderError("world diagnostics frame identity is inconsistent")
    visible_ranges: dict[int, list[np.ndarray]] = {
        item.object_id: [] for item in world.objects
    }
    visible_counts = {item.object_id: [] for item in world.objects}
    visible_frames = {item.object_id: 0 for item in world.objects}
    accepted = {item.object_id: 0 for item in world.objects}
    geometric = {item.object_id: 0 for item in world.objects}
    for frame in window:
        rotation, lidar_origin = _pose(frame)
        directions = ray_grid.directions_for(frame) @ rotation.T
        origins = ray_grid.origins_for(frame) @ rotation.T + lidar_origin
        competition = _accepted_object_hits(
            origins, directions, world, ray_grid, sensor, int(frame.frame_id)
        )
        native = np.asarray(ray_grid.ranges(frame)).copy()
        native[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = np.inf
        won = np.isfinite(competition.distance_m) & (
            competition.distance_m < native - world.tie_tolerance_m
        )
        for item in world.objects:
            identifier = item.object_id
            geometric[identifier] += int(competition.geometric_hits[identifier])
            accepted[identifier] += int(competition.accepted_hits[identifier])
            selected = won & (competition.object_id == identifier)
            frame_visible = int(np.count_nonzero(selected))
            visible_counts[identifier].append(frame_visible)
            if frame_visible:
                visible_frames[identifier] += 1
                visible_ranges[identifier].append(competition.distance_m[selected])
    objects: list[dict[str, object]] = []
    for item in world.objects:
        identifier = item.object_id
        ranges = (
            np.concatenate(visible_ranges[identifier])
            if visible_ranges[identifier]
            else np.empty(0, dtype=np.float64)
        )
        positive_counts = [value for value in visible_counts[identifier] if value > 0]
        nvis = float(np.median(positive_counts)) if positive_counts else 0.0
        visible_total = int(ranges.size)
        accepted_count = accepted[identifier]
        objects.append(
            {
                "object_id": identifier,
                "label": item.label,
                "Nvis": nvis,
                "O": (
                    float(1.0 - visible_total / accepted_count)
                    if accepted_count
                    else 1.0
                ),
                "d": float(np.median(ranges)) if ranges.size else None,
                "V": visible_frames[identifier],
                "visible_returns_total": visible_total,
                "geometric_intersections": geometric[identifier],
                "accepted_object_returns_before_occlusion": accepted_count,
            }
        )
    return {
        "center_frame": int(window[2].frame_id),
        "world_type": world.world_type,
        "definitions": {
            "Nvis": "median visible returns over frames in which the entity is visible",
            "O": "one minus total visible returns divided by accepted returns before occlusion",
            "d": "median sensor range of visible inserted returns in metres",
            "V": "number of the five frames with at least one visible return",
        },
        "objects": objects,
    }


def low_level_return_features(
    frame: SourceFrame,
    ray_grid: RayGrid,
    mask: np.ndarray,
    *,
    density_neighbors: int = 8,
) -> np.ndarray:
    """Build x/y/z, intensity, beam, range, and local-density audit features."""

    selected = np.asarray(mask).copy()
    if selected.dtype != np.bool_ or selected.shape != (ray_grid.slot_count,):
        raise RenderError("audit feature mask must be bool[slot]")
    selected &= ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
    slots = np.flatnonzero(selected)
    if slots.size < 2:
        raise RenderError("low-level audit requires at least two selected returns")
    if type(density_neighbors) is not int or density_neighbors < 1:
        raise RenderError("density_neighbors must be positive")
    valid = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
    all_points = np.asarray(frame.xyzi[valid, :3], dtype=np.float64)
    points = np.asarray(frame.xyzi[slots, :3], dtype=np.float64)
    ranges = np.linalg.norm(points, axis=1)
    neighbours = min(density_neighbors + 1, all_points.shape[0])
    distance, _ = cKDTree(all_points).query(points, k=neighbours, workers=1)
    radius = distance[:, -1] if distance.ndim == 2 else np.asarray(distance)
    density = max(neighbours - 1, 1) / (
        (4.0 / 3.0) * math.pi * np.maximum(radius, 1.0e-3) ** 3
    )
    return _freeze(
        np.column_stack(
            (
                points,
                frame.xyzi[slots, 3],
                ray_grid.beam_ids[slots],
                ranges,
                density,
            )
        ),
        np.float64,
    )


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positive = labels == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = labels.size - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def linear_classification_audit(
    class_zero_features: np.ndarray,
    class_one_features: np.ndarray,
    *,
    class_zero_groups: np.ndarray | None = None,
    class_one_groups: np.ndarray | None = None,
    seed: int = 0,
    maximum_per_class: int = 100_000,
    iterations: int = 300,
) -> dict[str, object]:
    """Fit a deterministic capacity-limited linear source classifier."""

    zero = np.asarray(class_zero_features, dtype=np.float64)
    one = np.asarray(class_one_features, dtype=np.float64)
    if zero.ndim != 2 or one.ndim != 2 or zero.shape[1] != one.shape[1]:
        raise RenderError("linear audit classes must be aligned feature matrices")
    if (
        min(zero.shape[0], one.shape[0]) < 10
        or not np.isfinite(zero).all()
        or not np.isfinite(one).all()
    ):
        raise RenderError("linear audit needs at least ten finite samples per class")
    if type(maximum_per_class) is not int or maximum_per_class < 10:
        raise RenderError("maximum_per_class must be an integer >=10")
    if type(iterations) is not int or iterations < 1:
        raise RenderError("iterations must be positive")
    rng = np.random.default_rng(_integer("seed", seed))

    if (class_zero_groups is None) != (class_one_groups is None):
        raise RenderError("both audit classes must provide groups or neither may")
    train_groups: np.ndarray | None = None
    test_groups: np.ndarray | None = None
    if class_zero_groups is None:

        def split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ids = rng.permutation(values.shape[0])[:maximum_per_class]
            stop = max(1, int(0.8 * ids.size))
            return values[ids[:stop]], values[ids[stop:]]

        zero_train, zero_test = split(zero)
        one_train, one_test = split(one)
        split_unit = "point"
    else:
        zero_groups = np.asarray(class_zero_groups)
        one_groups = np.asarray(class_one_groups)
        if (
            zero_groups.shape != (zero.shape[0],)
            or one_groups.shape != (one.shape[0],)
            or not np.issubdtype(zero_groups.dtype, np.integer)
            or not np.issubdtype(one_groups.dtype, np.integer)
        ):
            raise RenderError("audit groups must be aligned integer vectors")
        all_groups = np.unique(np.concatenate((zero_groups, one_groups))).astype(
            np.int64
        )
        if all_groups.size < 2:
            raise RenderError(
                "grouped audit requires at least two frame or world groups"
            )
        test_count = min(
            all_groups.size - 1, max(1, int(math.ceil(0.2 * all_groups.size)))
        )
        for _ in range(64):
            ordered = rng.permutation(all_groups)
            candidate_test = np.sort(ordered[:test_count])
            candidate_train = np.sort(ordered[test_count:])
            masks = (
                np.isin(zero_groups, candidate_train),
                np.isin(zero_groups, candidate_test),
                np.isin(one_groups, candidate_train),
                np.isin(one_groups, candidate_test),
            )
            if all(bool(mask.any()) for mask in masks):
                train_groups = candidate_train
                test_groups = candidate_test
                break
        if train_groups is None or test_groups is None:
            raise RenderError("no grouped split contains both classes on both sides")

        def grouped_split(
            values: np.ndarray, groups: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            train_ids = np.flatnonzero(np.isin(groups, train_groups))
            test_ids = np.flatnonzero(np.isin(groups, test_groups))
            total = train_ids.size + test_ids.size
            if total > maximum_per_class:
                train_limit = min(
                    train_ids.size,
                    max(1, int(round(maximum_per_class * train_ids.size / total))),
                )
                test_limit = min(test_ids.size, maximum_per_class - train_limit)
                if test_limit < 1:
                    test_limit = 1
                    train_limit = maximum_per_class - 1
                remaining = maximum_per_class - train_limit - test_limit
                add_train = min(remaining, train_ids.size - train_limit)
                train_limit += add_train
                test_limit += min(remaining - add_train, test_ids.size - test_limit)
                train_ids = rng.permutation(train_ids)[:train_limit]
                test_ids = rng.permutation(test_ids)[:test_limit]
            return values[train_ids], values[test_ids]

        zero_train, zero_test = grouped_split(zero, zero_groups)
        one_train, one_test = grouped_split(one, one_groups)
        split_unit = "frame_or_world_group"
    if min(zero_test.shape[0], one_test.shape[0]) < 1:
        raise RenderError("linear audit split left no held-out sample")
    train = np.vstack((zero_train, one_train))
    train_labels = np.concatenate(
        (np.zeros(zero_train.shape[0]), np.ones(one_train.shape[0]))
    )
    test = np.vstack((zero_test, one_test))
    test_labels = np.concatenate(
        (
            np.zeros(zero_test.shape[0], dtype=np.int8),
            np.ones(one_test.shape[0], dtype=np.int8),
        )
    )
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    train = (train - mean) / scale
    test = (test - mean) / scale
    weights = np.zeros(train.shape[1], dtype=np.float64)
    bias = 0.0
    for iteration in range(iterations):
        logits = np.clip(train @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - train_labels
        learning_rate = 0.2 / math.sqrt(iteration + 1.0)
        weights -= learning_rate * (
            (train.T @ error) / train.shape[0] + 1.0e-3 * weights
        )
        bias -= learning_rate * float(np.mean(error))
    scores = 1.0 / (1.0 + np.exp(-np.clip(test @ weights + bias, -30.0, 30.0)))
    predicted = scores >= 0.5
    class_accuracy = [
        float(np.mean(predicted[test_labels == label] == bool(label)))
        for label in (0, 1)
    ]
    return {
        "model": "standardized_logistic_regression",
        "model_capacity": {
            "linear_parameters": int(train.shape[1] + 1),
            "optimization_iterations": iterations,
            "l2_weight_penalty": 1.0e-3,
        },
        "feature_count": int(train.shape[1]),
        "train_samples": int(train.shape[0]),
        "test_samples": int(test.shape[0]),
        "split_unit": split_unit,
        "train_groups": (
            None if train_groups is None else train_groups.astype(int).tolist()
        ),
        "test_groups": (
            None if test_groups is None else test_groups.astype(int).tolist()
        ),
        "accuracy": float(np.mean(predicted == test_labels)),
        "balanced_accuracy": float(np.mean(class_accuracy)),
        "auroc": _rank_auc(test_labels, scores),
        "standardized_coefficients": weights.tolist(),
        "intercept": bias,
        "threshold_conclusion": None,
    }


def rendering_source_leakage_audit(
    real_normal_features: np.ndarray,
    rendered_normal_control_features: np.ndarray,
    *,
    real_frame_groups: np.ndarray,
    normal_control_frame_groups: np.ndarray,
    **kwargs: object,
) -> dict[str, object]:
    result = linear_classification_audit(
        real_normal_features,
        rendered_normal_control_features,
        class_zero_groups=real_frame_groups,
        class_one_groups=normal_control_frame_groups,
        **kwargs,  # type: ignore[arg-type]
    )
    return {"class_zero": "real-normal", "class_one": "normal-control", **result}


def anomaly_proxy_difficulty_audit(
    rendered_normal_control_features: np.ndarray,
    anomaly_proxy_features: np.ndarray,
    *,
    normal_control_groups: np.ndarray,
    anomaly_proxy_groups: np.ndarray,
    **kwargs: object,
) -> dict[str, object]:
    result = linear_classification_audit(
        rendered_normal_control_features,
        anomaly_proxy_features,
        class_zero_groups=normal_control_groups,
        class_one_groups=anomaly_proxy_groups,
        **kwargs,  # type: ignore[arg-type]
    )
    return {"class_zero": "normal-control", "class_one": "anomaly-proxy", **result}


def sensor_distribution_statistics(
    original_frames: Iterable[SourceFrame],
    rendered_frames_input: Iterable[RenderedFrame],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    object_ids: Sequence[int] = (),
) -> dict[str, object]:
    """Summarize sensor consistency without assigning a pass/fail threshold."""

    beam_slots = np.zeros(ray_grid.beam_count, dtype=np.int64)
    native_beam_returns = np.zeros(ray_grid.beam_count, dtype=np.int64)
    rendered_beam_returns = np.zeros(ray_grid.beam_count, dtype=np.int64)
    shape = (ray_grid.beam_count, sensor.range_edges_m.size - 1)
    native_intensity_sum = np.zeros(shape, dtype=np.float64)
    rendered_intensity_sum = np.zeros(shape, dtype=np.float64)
    native_counts = np.zeros(shape, dtype=np.int64)
    rendered_counts = np.zeros(shape, dtype=np.int64)
    native_surface_opportunities = np.zeros(shape, dtype=np.int64)
    rendered_surface_opportunities = np.zeros(shape, dtype=np.int64)
    native_surface_returns = np.zeros(shape, dtype=np.int64)
    rendered_surface_returns = np.zeros(shape, dtype=np.int64)
    new_returns = 0
    original_empty = 0
    occluded = 0
    original_returns = 0
    expected_ids = tuple(
        _integer("object_id", int(item), minimum=1) for item in object_ids
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise RenderError("sensor statistics object_ids must be unique")
    visible_by_object: dict[int, list[int]] = {item: [] for item in expected_ids}
    empty_to_valid_by_frame: list[float] = []
    occlusion_by_frame: list[float] = []
    normal_control_returns = 0
    anomaly_proxy_returns = 0
    frame_count = 0
    for original, rendered in zip(original_frames, rendered_frames_input, strict=True):
        if (
            original.frame_id != rendered.frame_id
            or original.partition != rendered.source.partition
            or original.sequence_id != rendered.source.sequence_id
        ):
            raise RenderError("sensor statistics frames are not aligned")
        if original.partition != "train" or original.sequence_id != 206:
            raise RenderError(
                "formal sensor distribution audit must use normal train/206"
            )
        beam_slots += ray_grid.columns
        original_real = ~np.asarray(original.zero_slot_mask, dtype=np.bool_)
        rendered_real = ~np.asarray(rendered.source.zero_slot_mask, dtype=np.bool_)
        native_beam_returns += np.bincount(
            ray_grid.beam_ids[original_real], minlength=ray_grid.beam_count
        )
        rendered_beam_returns += np.bincount(
            ray_grid.beam_ids[rendered_real], minlength=ray_grid.beam_count
        )
        for frame, real, sums, counts in (
            (original, original_real, native_intensity_sum, native_counts),
            (rendered.source, rendered_real, rendered_intensity_sum, rendered_counts),
        ):
            ranges = np.linalg.norm(
                np.asarray(frame.xyzi[:, :3], dtype=np.float64), axis=1
            )
            range_bin = np.clip(
                np.searchsorted(sensor.range_edges_m, ranges[real], side="right") - 1,
                0,
                sensor.range_edges_m.size - 2,
            )
            beams = ray_grid.beam_ids[real]
            intensities = np.asarray(frame.xyzi[real, 3], dtype=np.float64)
            np.add.at(counts, (beams, range_bin), 1)
            np.add.at(sums, (beams, range_bin), intensities)
        for frame, opportunities, returns in (
            (original, native_surface_opportunities, native_surface_returns),
            (
                rendered.source,
                rendered_surface_opportunities,
                rendered_surface_returns,
            ),
        ):
            ranges, incidence, _, potential_range, potential_incidence = (
                _surface_measurements(frame, ray_grid)
            )
            returned = np.isfinite(incidence)
            potential = np.isfinite(potential_incidence)
            returned_bin = np.clip(
                np.searchsorted(sensor.range_edges_m, ranges[returned], side="right")
                - 1,
                0,
                sensor.range_edges_m.size - 2,
            )
            potential_bin = np.clip(
                np.searchsorted(
                    sensor.range_edges_m, potential_range[potential], side="right"
                )
                - 1,
                0,
                sensor.range_edges_m.size - 2,
            )
            beam_grid = np.broadcast_to(
                np.arange(ray_grid.beam_count)[:, None],
                (ray_grid.beam_count, ray_grid.columns),
            )
            np.add.at(
                returns,
                (beam_grid[returned], returned_bin),
                1,
            )
            np.add.at(
                opportunities,
                (beam_grid[returned], returned_bin),
                1,
            )
            np.add.at(
                opportunities,
                (beam_grid[potential], potential_bin),
                1,
            )
        frame_empty = int(np.count_nonzero(~original_real))
        frame_new = int(np.count_nonzero(rendered.inserted_mask & ~original_real))
        frame_original = int(np.count_nonzero(original_real))
        frame_occluded = int(np.count_nonzero(rendered.occluded_original_mask))
        original_empty += frame_empty
        new_returns += frame_new
        original_returns += frame_original
        occluded += frame_occluded
        empty_to_valid_by_frame.append(frame_new / frame_empty if frame_empty else 0.0)
        occlusion_by_frame.append(
            frame_occluded / frame_original if frame_original else 0.0
        )
        normal_control_returns += int(np.count_nonzero(rendered.normal_control_mask))
        anomaly_proxy_returns += int(np.count_nonzero(rendered.anomaly_proxy_mask))
        visible_ids = rendered.object_id_internal[rendered.inserted_mask]
        for identifier in np.unique(visible_ids):
            visible_by_object.setdefault(int(identifier), [0] * frame_count)
        for identifier in visible_by_object:
            visible_by_object[identifier].append(
                int(np.count_nonzero(visible_ids == identifier))
            )
        frame_count += 1
    if frame_count == 0:
        raise RenderError("sensor statistics require at least one aligned frame")

    def means(sums: np.ndarray, counts: np.ndarray) -> list[list[float | None]]:
        output = np.full(sums.shape, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=output, where=counts > 0)
        return [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in output
        ]

    def rates(
        returns: np.ndarray, opportunities: np.ndarray
    ) -> list[list[float | None]]:
        output = np.full(returns.shape, np.nan, dtype=np.float64)
        np.divide(returns, opportunities, out=output, where=opportunities > 0)
        return [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in output
        ]

    def distance_rates(
        returns: np.ndarray, opportunities: np.ndarray
    ) -> list[float | None]:
        numerator = returns.sum(axis=0)
        denominator = opportunities.sum(axis=0)
        return [
            float(value / count) if count else None
            for value, count in zip(numerator, denominator, strict=True)
        ]

    change: dict[str, list[float]] = {}
    for identifier, values in visible_by_object.items():
        array = np.asarray(values, dtype=np.float64)
        change[str(identifier)] = (
            np.abs(np.diff(array)) / np.maximum(array[:-1], 1.0)
        ).tolist()
    return {
        "frames": frame_count,
        "beam_return_rate": {
            "real_normal": (native_beam_returns / beam_slots).tolist(),
            "rendered": (rendered_beam_returns / beam_slots).tolist(),
        },
        "beam_range_return_counts": {
            "real_normal": native_counts.tolist(),
            "rendered": rendered_counts.tolist(),
        },
        "beam_range_surface_return_rate": {
            "real_normal": rates(native_surface_returns, native_surface_opportunities),
            "rendered": rates(rendered_surface_returns, rendered_surface_opportunities),
            "opportunity_rule": dict(sensor.provenance).get(
                "empty_ray_opportunity", "not_recorded"
            ),
        },
        "range_surface_return_rate": {
            "real_normal": distance_rates(
                native_surface_returns, native_surface_opportunities
            ),
            "rendered": distance_rates(
                rendered_surface_returns, rendered_surface_opportunities
            ),
        },
        "beam_range_intensity_mean": {
            "real_normal": means(native_intensity_sum, native_counts),
            "rendered": means(rendered_intensity_sum, rendered_counts),
        },
        "empty_to_valid_rate": new_returns / original_empty if original_empty else 0.0,
        "empty_to_valid_rate_by_frame": empty_to_valid_by_frame,
        "normal_control_returns": normal_control_returns,
        "anomaly_proxy_returns": anomaly_proxy_returns,
        "visible_returns_by_object_and_frame": {
            str(key): value for key, value in visible_by_object.items()
        },
        "visible_count_change_rate_by_object": change,
        "occlusion_rate_of_original_returns": occluded / original_returns
        if original_returns
        else 0.0,
        "occlusion_rate_by_frame": occlusion_by_frame,
        "threshold_conclusion": None,
    }


@dataclass(frozen=True, slots=True)
class DevelopmentWorldDefinition:
    world_id: int
    center_frame: int
    selection_eligible: bool
    world: WorldSpec
    diagnostics: Mapping[str, object]


def _json_plain(value: object, name: str) -> object:
    """Convert an audit value to finite JSON data without changing its meaning."""

    if isinstance(value, np.ndarray):
        return _json_plain(value.tolist(), name)
    if isinstance(value, np.generic):
        return _json_plain(value.item(), name)
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RenderError(f"{name} mapping keys must be strings")
            output[key] = _json_plain(item, f"{name}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_plain(item, f"{name}[{index}]") for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RenderError(f"{name} contains NaN or Inf")
        return value
    raise RenderError(
        f"{name} contains a non-JSON value of type {type(value).__name__}"
    )


def _has_numeric_statistic(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_has_numeric_statistic(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_numeric_statistic(item) for item in value)
    return False


def development_worlds_payload(
    definitions: Sequence[DevelopmentWorldDefinition],
    *,
    gate1_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize fixed worlds while leaving every scientific verdict pending."""

    items = tuple(definitions)
    if len(items) != 30 or any(
        not isinstance(item, DevelopmentWorldDefinition) for item in items
    ):
        raise RenderError("development serialization requires 30 world definitions")
    identifiers = tuple(item.world_id for item in items)
    if len(set(identifiers)) != 30 or set(identifiers) != set(range(30)):
        raise RenderError("development world IDs must be exactly 0 through 29")
    items = tuple(sorted(items, key=lambda item: item.world_id))

    evidence_input = {} if gate1_evidence is None else dict(gate1_evidence)
    unknown_evidence = set(evidence_input) - set(GATE1_EVIDENCE_KEYS)
    if unknown_evidence:
        raise RenderError(f"unknown gate-1 evidence: {sorted(unknown_evidence)}")
    evidence: dict[str, object] = {}
    for key in GATE1_EVIDENCE_KEYS:
        value = evidence_input.get(key)
        if value is None:
            evidence[key] = None
            continue
        if not isinstance(value, Mapping) or not value:
            raise RenderError(f"gate-1 evidence {key} must be a non-empty mapping")
        plain = _json_plain(value, f"gate1.evidence.{key}")
        if not _has_numeric_statistic(plain):
            raise RenderError(f"gate-1 evidence {key} has no finite numeric statistic")
        evidence[key] = plain

    groups: tuple[list[dict[str, object]], list[dict[str, object]]] = ([], [])
    for expected_id, item in enumerate(items):
        if item.world_id != expected_id:
            raise RenderError("development world ordering is not canonical")
        if type(item.selection_eligible) is not bool or item.selection_eligible != (
            expected_id < 24
        ):
            raise RenderError(
                "only development worlds 0 through 23 may affect selection"
            )
        center_frame = _integer("development center_frame", item.center_frame)
        if not 6 <= center_frame <= 679:
            raise RenderError(
                "development center must define a legal 201 symmetric window"
            )
        world = item.world
        if not isinstance(world, WorldSpec) or world.source_sequence_id != 201:
            raise RenderError("development worlds must use sequence 201")
        if world.world_type != "mixed":
            raise RenderError("every fixed development world must contain both labels")
        diagnostics = item.diagnostics
        if (
            not isinstance(diagnostics, Mapping)
            or diagnostics.get("center_frame") != item.center_frame
        ):
            raise RenderError("development diagnostics do not match their center frame")
        raw_difficulty = diagnostics.get("objects")
        if not isinstance(raw_difficulty, Sequence) or isinstance(
            raw_difficulty, (str, bytes)
        ):
            raise RenderError("development diagnostics need per-object difficulty")
        difficulty: list[dict[str, object]] = []
        for index, raw in enumerate(raw_difficulty):
            if not isinstance(raw, Mapping):
                raise RenderError("development difficulty entries must be mappings")
            identifier = _integer(
                f"difficulty[{index}].object_id",
                raw.get("object_id"),  # type: ignore[arg-type]
            )
            nvis = _finite_scalar(f"difficulty[{index}].Nvis", raw.get("Nvis"))  # type: ignore[arg-type]
            occlusion = _finite_scalar(f"difficulty[{index}].O", raw.get("O"))  # type: ignore[arg-type]
            distance = _finite_scalar(f"difficulty[{index}].d", raw.get("d"))  # type: ignore[arg-type]
            visibility = _integer(
                f"difficulty[{index}].V",
                raw.get("V"),
                minimum=1,  # type: ignore[arg-type]
            )
            if nvis <= 0.0 or not 0.0 <= occlusion <= 1.0 or distance <= 0.0:
                raise RenderError(
                    "development difficulty lies outside its physical domain"
                )
            if visibility > 5:
                raise RenderError("development visibility V must lie in [1,5]")
            difficulty.append(
                {
                    "object_id": identifier,
                    "Nvis": nvis,
                    "O": occlusion,
                    "d": distance,
                    "V": visibility,
                }
            )
        difficulty.sort(key=lambda value: int(value["object_id"]))
        if tuple(int(value["object_id"]) for value in difficulty) != tuple(
            object_.object_id for object_ in world.objects
        ):
            raise RenderError("difficulty records do not identify every world object")

        held_out = expected_id >= 24
        anomaly_shapes = [
            object_.shape
            for object_ in world.objects
            if object_.label == "anomaly-proxy"
        ]
        if held_out and any(
            not isinstance(shape, HeldOutTorusShape) for shape in anomaly_shapes
        ):
            raise RenderError(
                "held-out development worlds must use only torus SDF proxies"
            )
        if not held_out and any(
            not isinstance(shape, ShapeSpec) for shape in anomaly_shapes
        ):
            raise RenderError(
                "in-generator development worlds must use training geometry"
            )
        mechanism = "torus_SDF" if held_out else "in_generator"
        groups[int(held_out)].append(
            {
                "world_id": item.world_id,
                "seed": world.seed,
                "center_frame": center_frame,
                "world": world.to_dict(),
                "difficulty": difficulty,
                "mechanism": mechanism,
            }
        )
    return {
        "format": DEVELOPMENT_FORMAT,
        "protocol_schema": DEVELOPMENT_PROTOCOL_SCHEMA,
        "sequence_id": 201,
        "status": "definitions_only_unvalidated",
        "validation": {key: False for key in DEVELOPMENT_VALIDATION_KEYS},
        "gate1": {
            "status": "pending_scientific_verdict",
            "evidence": evidence,
        },
        "in_generator": groups[0],
        "generator_held_out": groups[1],
    }


def save_development_worlds(
    path: Path | str,
    definitions: Sequence[DevelopmentWorldDefinition],
    *,
    gate1_evidence: Mapping[str, object] | None = None,
) -> None:
    """Atomically save fixed definitions without changing scientific status."""

    payload = development_worlds_payload(definitions, gate1_evidence=gate1_evidence)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_fixed_development_worlds(
    frames_201: Sequence[SourceFrame],
    normal_template_library: Sequence[NormalTemplateShape],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    in_generator_seeds: Sequence[int],
    held_out_seeds: Sequence[int],
    *,
    maximum_attempts: int = 32,
) -> tuple[DevelopmentWorldDefinition, ...]:
    """Create 24 selectable mixed worlds and six torus-only diagnostic mechanisms."""

    frames = frames_201
    templates = tuple(normal_template_library)
    in_seeds = tuple(
        _integer(f"in_generator_seeds[{index}]", value)
        for index, value in enumerate(in_generator_seeds)
    )
    held_seeds = tuple(
        _integer(f"held_out_seeds[{index}]", value)
        for index, value in enumerate(held_out_seeds)
    )
    if len(frames) != 678:
        raise RenderError(
            "development generation requires eligible train/201 frames 4 through 681"
        )
    first_frame = frames[0]
    last_frame = frames[len(frames) - 1]
    if (
        first_frame.partition != "train"
        or first_frame.sequence_id != 201
        or first_frame.frame_id != 4
        or last_frame.partition != "train"
        or last_frame.sequence_id != 201
        or last_frame.frame_id != 681
        or first_frame.xyzi.shape[0] != ray_grid.slot_count
        or last_frame.xyzi.shape[0] != ray_grid.slot_count
    ):
        raise RenderError("development backgrounds must be identified normal train/201")
    if (
        len(in_seeds) != 24
        or len(held_seeds) != 6
        or len(set(in_seeds + held_seeds)) != 30
    ):
        raise RenderError("development generation requires 24+6 unique seeds")
    if not templates or any(
        not isinstance(item, NormalTemplateShape) for item in templates
    ):
        raise RenderError(
            "development generation requires a non-empty 206 template library"
        )
    if any(item.source_sequence_id != 206 for item in templates):
        raise RenderError("development normal controls must originate from train/206")
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise RenderError("maximum_attempts must be positive")
    output: list[DevelopmentWorldDefinition] = []
    for world_id, (seed, held_out) in enumerate(
        [(value, False) for value in in_seeds] + [(value, True) for value in held_seeds]
    ):
        built: DevelopmentWorldDefinition | None = None
        for attempt in range(maximum_attempts):
            rng = np.random.default_rng(seed + 1_000_003 * attempt)
            center_index = 2 + int(rng.integers(0, len(frames) - 4))
            window = tuple(
                frames[index] for index in range(center_index - 2, center_index + 3)
            )
            if any(
                frame.partition != "train"
                or frame.sequence_id != 201
                or frame.frame_id != 4 + center_index - 2 + offset
                or frame.xyzi.shape[0] != ray_grid.slot_count
                for offset, frame in enumerate(window)
            ):
                raise RenderError(
                    "selected development window is not canonical train/201"
                )
            context = collect_support_context(window, maximum_points_per_class=100_000)
            template = templates[int(rng.integers(0, len(templates)))]
            try:
                target_scale = rng.uniform(0.9, 1.1, size=3)
                scaled_template = template.rescaled(
                    tuple(target_scale / np.asarray(template.scale_xyz))
                )
                control = place_object(
                    scaled_template,
                    MaterialSpec.sample(seed + attempt * 11 + 1),
                    context.ground_world,
                    context.obstacle_world,
                    object_id=1,
                    label="normal-control",
                    seed=seed + attempt * 11 + 2,
                    ground_semantic_ids=context.ground_semantic,
                )
                lower, upper = scaled_template.local_bounds()
                control_size = float(np.max(upper - lower))
                minimum_proxy_size = 0.4 if held_out else 0.2
                target_size = float(np.clip(control_size, minimum_proxy_size, 3.0))
                proxy_size_range = (
                    max(minimum_proxy_size, 0.75 * target_size),
                    min(3.0, 1.25 * target_size),
                )
                proxy_shape: InsertShape = (
                    sample_held_out_anomaly_shape(
                        seed + attempt * 11 + 3,
                        size_m_range=proxy_size_range,
                    )
                    if held_out
                    else sample_training_anomaly_shape(
                        seed + attempt * 11 + 3,
                        size_m_range=proxy_size_range,
                    )
                )
                control_position = np.asarray(
                    control.translation_world_m, dtype=np.float64
                )
                nearby = (
                    np.linalg.norm(context.ground_world - control_position, axis=1)
                    <= 4.0
                )
                if int(np.count_nonzero(nearby)) < 12:
                    nearby = np.ones(context.ground_world.shape[0], dtype=np.bool_)
                proxy = place_object(
                    proxy_shape,
                    MaterialSpec.sample(seed + attempt * 11 + 4),
                    context.ground_world[nearby],
                    context.obstacle_world,
                    object_id=2,
                    label="anomaly-proxy",
                    seed=seed + attempt * 11 + 5,
                    ground_semantic_ids=context.ground_semantic[nearby],
                    existing_objects=(control,),
                )
                world = WorldSpec(seed, 201, (control, proxy))
                diagnostics = five_frame_world_diagnostics(
                    world, window, ray_grid, sensor
                )
                if any(int(item["Nvis"]) < 1 for item in diagnostics["objects"]):  # type: ignore[index]
                    continue
            except (RenderError, PlacementError):
                continue
            built = DevelopmentWorldDefinition(
                world_id,
                int(window[2].frame_id),
                not held_out,
                world,
                diagnostics,
            )
            break
        if built is None:
            raise PlacementError(
                f"development world {world_id} failed {maximum_attempts} deterministic attempts"
            )
        if held_out and any(
            item.label == "anomaly-proxy"
            and not isinstance(item.shape, HeldOutTorusShape)
            for item in built.world.objects
        ):
            raise AssertionError(
                "held-out development world used a training geometry mechanism"
            )
        output.append(built)
    return tuple(output)
