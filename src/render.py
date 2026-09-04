#!/usr/bin/env python3
"""Deterministic window-level counterfactual rendering for AJAE.

The renderer preserves organized file slots for I/O.  A slot becomes a
canonical LiDAR ray only after the explicit RayGrid audit.  Inserted objects
live in world coordinates and compete with native returns by nearest distance.
"""

# ruff: noqa: E402 -- numerical thread limits must precede NumPy/SciPy imports.

from __future__ import annotations

import json
import hashlib
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq, differential_evolution
from scipy.spatial import cKDTree
from scipy.stats import qmc

try:
    from .scene import (
        PointLabels,
        SourceFrame,
        make_source_frame,
    )
except ImportError:  # Direct module execution and small isolated checks.
    from scene import PointLabels, SourceFrame, make_source_frame


LASER_BEAMS = 128
GROUND_SEMANTIC_IDS = (40, 44, 48, 49, 60)
WORLD_FORMAT = "ajae-world-v3"
WORLD_REPORT_FORMAT = "ajae-world-generation-report-v3"
SUPPORT_POOL_FORMAT = "ajae-qualified-support-pool-v1"
SUPPORT_POOL_SHA256_BY_SEQUENCE = {
    206: "a09ce4701c78ef72d0ea6de2ff5fb98f74e37ffd34f0eb9474f46994494bbb0a",
    201: "47bed8f59f4d9c21c5deaeec6892dd1820a559737c6d07f2ef991c7733422119",
}
SUPPORT_POOL_SEMANTICS = (40, 48, 49)
CALIBRATION_FORMAT = "ajae-sensor-calibration-v4"
PROCEDURAL_GENERATOR_SCHEMA = 7
SHAPE_FAMILIES = ("general", "blocky", "flat", "elongated")
AXIS_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
SCHEMA7_FAMILY_STREAM = 2001
SCHEMA7_RATIO_STREAM = 2002
SCHEMA7_AXIS_STREAM = 2003
SCHEMA7_PARENT_TAU_STREAM = 3001
SCHEMA7_CHILD_TAU_STREAM = 3002
SYNTHETIC_INSTANCE_BASE = 60_000
MAX_OBJECT_ID = np.iinfo(np.uint16).max - SYNTHETIC_INSTANCE_BASE
OBJECT_LABELS = ("anomaly-proxy",)
ObjectLabel: TypeAlias = Literal["anomaly-proxy"]
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


class PlacementExhaustion(PlacementError):
    """Carry the exact evaluated proposal trace for a finite exhausted stream."""

    def __init__(
        self,
        proposal_pool_indices: Sequence[int],
        rejection_reasons: Sequence[str],
        minimum_obstacle_sdf_m: Sequence[float],
    ) -> None:
        super().__init__(
            "no qualified support passed the frozen checks within 128 proposals"
        )
        self.proposal_pool_indices = tuple(map(int, proposal_pool_indices))
        self.rejection_reasons = tuple(map(str, rejection_reasons))
        self.minimum_obstacle_sdf_m = tuple(map(float, minimum_obstacle_sdf_m))


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


def _interval_outward(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _interval_add(
    a_lower: np.ndarray,
    a_upper: np.ndarray,
    b_lower: np.ndarray,
    b_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _interval_outward(a_lower + b_lower, a_upper + b_upper)


def _interval_multiply(
    a_lower: np.ndarray,
    a_upper: np.ndarray,
    b_lower: np.ndarray,
    b_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(
        (
            a_lower * b_lower,
            a_lower * b_upper,
            a_upper * b_lower,
            a_upper * b_upper,
        )
    )
    return _interval_outward(np.min(values, axis=0), np.max(values, axis=0))


def _interval_scale(
    lower: np.ndarray, upper: np.ndarray, value: float
) -> tuple[np.ndarray, np.ndarray]:
    if value >= 0.0:
        return _interval_outward(value * lower, value * upper)
    return _interval_outward(value * upper, value * lower)


def _interval_square(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.where(
        (lower <= 0.0) & (upper >= 0.0),
        0.0,
        np.minimum(lower * lower, upper * upper),
    )
    return _interval_outward(minimum, np.maximum(lower * lower, upper * upper))


def _interval_absolute(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.where(
        (lower <= 0.0) & (upper >= 0.0),
        0.0,
        np.minimum(np.abs(lower), np.abs(upper)),
    )
    return _interval_outward(minimum, np.maximum(np.abs(lower), np.abs(upper)))


def _interval_power(
    lower: np.ndarray, upper: np.ndarray, power: float
) -> tuple[np.ndarray, np.ndarray]:
    return _interval_outward(
        np.power(np.maximum(lower, 0.0), power),
        np.power(np.maximum(upper, 0.0), power),
    )


def _interval_trigonometric(
    lower: np.ndarray, upper: np.ndarray, *, cosine: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    function = np.cos if cosine else np.sin
    result_lower = np.minimum(function(lower), function(upper))
    result_upper = np.maximum(function(lower), function(upper))
    wide = upper - lower >= 2.0 * math.pi
    maximum_phase = 0.0 if cosine else 0.5 * math.pi
    minimum_phase = math.pi if cosine else -0.5 * math.pi
    contains_maximum = np.ceil((lower - maximum_phase) / (2.0 * math.pi)) <= np.floor(
        (upper - maximum_phase) / (2.0 * math.pi)
    )
    contains_minimum = np.ceil((lower - minimum_phase) / (2.0 * math.pi)) <= np.floor(
        (upper - minimum_phase) / (2.0 * math.pi)
    )
    result_upper = np.where(wide | contains_maximum, 1.0, result_upper)
    result_lower = np.where(wide | contains_minimum, -1.0, result_lower)
    return _interval_outward(result_lower, result_upper)


@dataclass(frozen=True, slots=True)
class ShapeGenerationReport:
    """Record deterministic proposal efficiency without changing shape identity."""

    generator_schema: int
    proposal_count: int
    lower_certificate_rejections: int
    upper_certificate_rejections: int
    connectivity_disconnected_rejections: int
    connectivity_unresolved_rejections: int
    other_rejections: int
    accepted_size_lower_m: float
    accepted_size_upper_m: float
    outer_lower_m: tuple[float, float, float]
    outer_upper_m: tuple[float, float, float]
    size_definition: str
    shape_family: str
    child_parent_indices: tuple[int, ...]
    shared_witnesses_undeformed_m: tuple[tuple[float, float, float], ...]
    witness_parent_margins_m: tuple[float, ...]
    witness_child_margins_m: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "generator_schema": self.generator_schema,
            "proposal_count": self.proposal_count,
            "lower_certificate_rejections": self.lower_certificate_rejections,
            "upper_certificate_rejections": self.upper_certificate_rejections,
            "connectivity_disconnected_rejections": self.connectivity_disconnected_rejections,
            "connectivity_unresolved_rejections": self.connectivity_unresolved_rejections,
            "other_rejections": self.other_rejections,
            "accepted_size_lower_m": self.accepted_size_lower_m,
            "accepted_size_upper_m": self.accepted_size_upper_m,
            "outer_lower_m": list(self.outer_lower_m),
            "outer_upper_m": list(self.outer_upper_m),
            "size_definition": self.size_definition,
            "shape_family": self.shape_family,
            "child_parent_indices": list(self.child_parent_indices),
            "shared_witnesses_undeformed_m": [
                list(item) for item in self.shared_witnesses_undeformed_m
            ],
            "witness_parent_margins_m": list(self.witness_parent_margins_m),
            "witness_child_margins_m": list(self.witness_child_margins_m),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShapeGenerationReport":
        if not isinstance(value, Mapping):
            raise RenderError("ShapeGenerationReport JSON must be an object")
        plain = dict(value)
        for name in (
            "outer_lower_m",
            "outer_upper_m",
            "child_parent_indices",
            "shared_witnesses_undeformed_m",
            "witness_parent_margins_m",
            "witness_child_margins_m",
        ):
            if name in plain:
                plain[name] = tuple(
                    tuple(item) if isinstance(item, list) else item
                    for item in plain[name]  # type: ignore[union-attr]
                )
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid ShapeGenerationReport JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class ShapeSizeCertificate:
    """Conservative continuous size interval for one final CSG geometry."""

    outer_lower_m: tuple[float, float, float]
    outer_upper_m: tuple[float, float, float]
    lower_size_m: float
    upper_size_m: float
    witness_start_m: tuple[float, float, float]
    witness_end_m: tuple[float, float, float]
    sobol_probes: int
    interior_lines: int
    maximum_surface_residual_m: float


@dataclass(frozen=True, slots=True)
class ShapeConnectivityCertificate:
    """Conservative continuous connected/disconnected/unresolved evidence."""

    state: Literal["connected", "disconnected", "unresolved"]
    source: str
    standard_stats: tuple[int, int, int]
    strict_stats: tuple[int, int, int]


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
    _connectivity: ShapeConnectivityCertificate = field(
        init=False, repr=False, compare=False
    )

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
        connectivity = self.continuous_connectivity_certificate()
        object.__setattr__(self, "_connectivity", connectivity)
        if connectivity.state == "disconnected":
            raise RenderError("continuous CSG is certified disconnected")
        if connectivity.state == "unresolved":
            raise RenderError("continuous CSG connectivity is unresolved")
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
            raise RenderError(
                "continuous primitive bounds require exactly one primitive"
            )
        if self.operations != ("union",) or self.primitive_offsets_m != (
            (0.0, 0.0, 0.0),
        ):
            raise RenderError(
                "continuous primitive bounds require one centered union primitive"
            )
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
            self._primitive_distance(
                direction[None], scale, (0.0, 0.0, 0.0), exponent, yaw
            )[0]
            / minimum_scale
            + 1.0
        )
        if not np.isfinite(unit_value) or unit_value <= 0.0:
            raise RenderError(
                "single-primitive radial function is not finite and positive"
            )
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
        probe_count = 2048
        probe_id = np.arange(probe_count, dtype=np.float64)
        probe_z = 1.0 - 2.0 * (probe_id + 0.5) / probe_count
        probe_longitude = (math.pi * (3.0 - math.sqrt(5.0)) * probe_id + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        probe_angles = np.column_stack((np.arcsin(probe_z), probe_longitude))
        probe_points = np.asarray(
            [self._single_primitive_surface_point(a, b) for a, b in probe_angles]
        )
        for axis in range(3):
            for sign in (-1.0, 1.0):
                candidate_count = 2 * population
                elite_count = candidate_count // 2
                elite = np.argsort(sign * probe_points[:, axis])[-elite_count:]
                coverage = np.linspace(
                    0, probe_count - 1, candidate_count - elite_count, dtype=np.int64
                )
                initial_population = np.concatenate(
                    (probe_angles[elite], probe_angles[coverage]), axis=0
                )
                result = differential_evolution(
                    lambda value: (
                        -sign
                        * self._single_primitive_surface_point(value[0], value[1])[axis]
                    ),
                    angular_bounds,
                    seed=1009 + 17 * axis + int(sign > 0.0),
                    maxiter=iterations,
                    popsize=population,
                    tol=1.0e-10,
                    atol=1.0e-11,
                    polish=True,
                    init=initial_population,
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
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower >= upper)
        ):
            raise RenderError("continuous primitive bounds are invalid")
        return _freeze(lower), _freeze(upper)

    def _continuous_outer_bounds(
        self, *, safety_margin_m: float = 1.0e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a conservative continuous AABB through CSG and deformation."""

        margin = _finite_scalar("safety_margin_m", safety_margin_m)
        if margin < 0.0 or margin > 1.0e-3:
            raise RenderError("continuous-bound safety margin must lie in [0,1e-3]")
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        for scale, offset, yaw, operation in zip(
            self.primitive_scales_m,
            self.primitive_offsets_m,
            self.primitive_yaws_rad,
            self.operations,
            strict=True,
        ):
            expansion = 1.0 + self.surface_amplitude_m / min(scale)
            a, b, c = expansion * np.asarray(scale, dtype=np.float64)
            cosine = abs(math.cos(yaw))
            sine = abs(math.sin(yaw))
            half = np.asarray(
                (cosine * a + sine * b, sine * a + cosine * b, c),
                dtype=np.float64,
            )
            center = np.asarray(offset, dtype=np.float64)
            primitive_lower = center - half
            primitive_upper = center + half
            if lower is None:
                lower = primitive_lower
                upper = primitive_upper
            elif operation == "union":
                lower = np.minimum(lower, primitive_lower)
                upper = np.maximum(upper, primitive_upper)
            elif operation == "intersection":
                lower = np.maximum(lower, primitive_lower)
                upper = np.minimum(upper, primitive_upper)
            # Difference cannot enlarge the accumulated left-hand geometry.
        assert lower is not None and upper is not None
        if np.any(lower >= upper):
            raise RenderError("continuous CSG outer bounds are empty")

        z_lower = float(lower[2])
        z_upper = float(upper[2])
        z_abs = max(abs(z_lower), abs(z_upper))
        if abs(self.twist_rad_per_m) > EPSILON:
            radial = math.hypot(
                max(abs(float(lower[0])), abs(float(upper[0]))),
                max(abs(float(lower[1])), abs(float(upper[1]))),
            )
            x_interval = (-radial, radial)
            y_interval = (-radial, radial)
        else:
            x_interval = (float(lower[0]), float(upper[0]))
            y_interval = (float(lower[1]), float(upper[1]))

        scale_z = max(item[2] for item in self.primitive_scales_m)

        def deformed_interval(
            interval: tuple[float, float], taper: float, bend: float
        ) -> tuple[float, float]:
            factors = np.clip(
                1.0 + taper * np.asarray((z_lower, z_upper)) / scale_z,
                0.25,
                4.0,
            )
            products = np.asarray(
                [value * factor for value in interval for factor in factors],
                dtype=np.float64,
            )
            z_square_min = (
                0.0
                if z_lower <= 0.0 <= z_upper
                else min(z_lower * z_lower, z_upper * z_upper)
            )
            z_square_max = z_abs * z_abs
            bend_values = bend * np.asarray((z_square_min, z_square_max))
            return (
                float(np.min(products) + np.min(bend_values)),
                float(np.max(products) + np.max(bend_values)),
            )

        x_lower, x_upper = deformed_interval(
            x_interval, self.taper_per_m[0], self.bend_per_m[0]
        )
        y_lower, y_upper = deformed_interval(
            y_interval, self.taper_per_m[1], self.bend_per_m[1]
        )
        result_lower = np.asarray(
            (x_lower - margin, y_lower - margin, z_lower - margin), dtype=np.float64
        )
        result_upper = np.asarray(
            (x_upper + margin, y_upper + margin, z_upper + margin), dtype=np.float64
        )
        if (
            not np.isfinite(result_lower).all()
            or not np.isfinite(result_upper).all()
            or np.any(result_lower >= result_upper)
        ):
            raise RenderError("continuous CSG outer bounds are invalid")
        return _freeze(result_lower), _freeze(result_upper)

    def tight_continuous_outer_bounds(
        self,
        *,
        z_slabs: int = 256,
        safety_margin_m: float = 1.0e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound the deformed CSG with fixed-cost z-sliced interval propagation."""

        slabs = _integer("z_slabs", z_slabs, minimum=16)
        margin = _finite_scalar("safety_margin_m", safety_margin_m)
        if margin < 0.0 or margin > 1.0e-3:
            raise RenderError("continuous-bound safety margin must lie in [0,1e-3]")
        old_lower, old_upper = self._continuous_outer_bounds(safety_margin_m=margin)
        edges = np.linspace(old_lower[2], old_upper[2], slabs + 1)
        z_lower = edges[:-1]
        z_upper = edges[1:]
        valid = np.zeros(slabs, dtype=np.bool_)
        x_lower = np.zeros(slabs, dtype=np.float64)
        x_upper = np.zeros(slabs, dtype=np.float64)
        y_lower = np.zeros(slabs, dtype=np.float64)
        y_upper = np.zeros(slabs, dtype=np.float64)

        for index, (scale, offset, exponent, yaw, operation) in enumerate(
            zip(
                self.primitive_scales_m,
                self.primitive_offsets_m,
                self.primitive_exponents,
                self.primitive_yaws_rad,
                self.operations,
                strict=True,
            )
        ):
            a, b, c = scale
            vertical, horizontal = exponent
            level = 1.0 + self.surface_amplitude_m / min(scale)
            primitive_z_lower = offset[2] - level * c
            primitive_z_upper = offset[2] + level * c
            active = (z_upper >= primitive_z_lower) & (z_lower <= primitive_z_upper)
            nearest_z = np.where(
                (z_lower <= offset[2]) & (z_upper >= offset[2]),
                0.0,
                np.minimum(
                    np.abs(z_lower - offset[2]),
                    np.abs(z_upper - offset[2]),
                ),
            )
            radial_term = np.maximum(
                0.0,
                level ** (2.0 / vertical) - (nearest_z / c) ** (2.0 / vertical),
            )
            cross_scale = radial_term ** (vertical / 2.0)
            cosine = abs(math.cos(yaw))
            sine = abs(math.sin(yaw))
            planar_power = 2.0 / horizontal
            if planar_power > 1.0 + 1.0e-12:
                dual = planar_power / (planar_power - 1.0)
                half_x = cross_scale * ((a * cosine) ** dual + (b * sine) ** dual) ** (
                    1.0 / dual
                )
                half_y = cross_scale * ((a * sine) ** dual + (b * cosine) ** dual) ** (
                    1.0 / dual
                )
            else:
                # The rectangle support remains conservative for non-convex exponents.
                half_x = cross_scale * (a * cosine + b * sine)
                half_y = cross_scale * (a * sine + b * cosine)
            primitive_x_lower = offset[0] - half_x
            primitive_x_upper = offset[0] + half_x
            primitive_y_lower = offset[1] - half_y
            primitive_y_upper = offset[1] + half_y

            if index == 0:
                valid = active.copy()
                x_lower = primitive_x_lower
                x_upper = primitive_x_upper
                y_lower = primitive_y_lower
                y_upper = primitive_y_upper
            elif operation == "union":
                both = valid & active
                x_lower = np.where(
                    both,
                    np.minimum(x_lower, primitive_x_lower),
                    np.where(active, primitive_x_lower, x_lower),
                )
                x_upper = np.where(
                    both,
                    np.maximum(x_upper, primitive_x_upper),
                    np.where(active, primitive_x_upper, x_upper),
                )
                y_lower = np.where(
                    both,
                    np.minimum(y_lower, primitive_y_lower),
                    np.where(active, primitive_y_lower, y_lower),
                )
                y_upper = np.where(
                    both,
                    np.maximum(y_upper, primitive_y_upper),
                    np.where(active, primitive_y_upper, y_upper),
                )
                valid |= active
            elif operation == "intersection":
                valid &= active
                x_lower = np.maximum(x_lower, primitive_x_lower)
                x_upper = np.minimum(x_upper, primitive_x_upper)
                y_lower = np.maximum(y_lower, primitive_y_lower)
                y_upper = np.minimum(y_upper, primitive_y_upper)
                valid &= (x_lower <= x_upper) & (y_lower <= y_upper)
            # Difference can only remove points from the accumulated left set.

        if not bool(valid.any()):
            raise RenderError("z-sliced continuous CSG outer bounds are empty")
        z_lower = z_lower[valid]
        z_upper = z_upper[valid]
        x_lower = x_lower[valid]
        x_upper = x_upper[valid]
        y_lower = y_lower[valid]
        y_upper = y_upper[valid]

        def interval_product(
            first_lower: np.ndarray,
            first_upper: np.ndarray,
            second_lower: np.ndarray,
            second_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            values = np.stack(
                (
                    first_lower * second_lower,
                    first_lower * second_upper,
                    first_upper * second_lower,
                    first_upper * second_upper,
                )
            )
            return np.min(values, axis=0), np.max(values, axis=0)

        def trig_interval(
            angle_lower: np.ndarray,
            angle_upper: np.ndarray,
            *,
            cosine: bool,
        ) -> tuple[np.ndarray, np.ndarray]:
            function = np.cos if cosine else np.sin
            result_lower = np.minimum(function(angle_lower), function(angle_upper))
            result_upper = np.maximum(function(angle_lower), function(angle_upper))
            maximum_phase = 0.0 if cosine else 0.5 * math.pi
            minimum_phase = math.pi if cosine else -0.5 * math.pi
            period = 2.0 * math.pi
            contains_maximum = np.ceil(
                (angle_lower - maximum_phase) / period
            ) <= np.floor((angle_upper - maximum_phase) / period)
            contains_minimum = np.ceil(
                (angle_lower - minimum_phase) / period
            ) <= np.floor((angle_upper - minimum_phase) / period)
            return (
                np.where(contains_minimum, -1.0, result_lower),
                np.where(contains_maximum, 1.0, result_upper),
            )

        angle_a = self.twist_rad_per_m * z_lower
        angle_b = self.twist_rad_per_m * z_upper
        angle_lower = np.minimum(angle_a, angle_b)
        angle_upper = np.maximum(angle_a, angle_b)
        cosine_lower, cosine_upper = trig_interval(
            angle_lower, angle_upper, cosine=True
        )
        sine_lower, sine_upper = trig_interval(angle_lower, angle_upper, cosine=False)
        cosine_x_lower, cosine_x_upper = interval_product(
            cosine_lower, cosine_upper, x_lower, x_upper
        )
        sine_y_lower, sine_y_upper = interval_product(
            sine_lower, sine_upper, y_lower, y_upper
        )
        rotated_x_lower = cosine_x_lower - sine_y_upper
        rotated_x_upper = cosine_x_upper - sine_y_lower
        sine_x_lower, sine_x_upper = interval_product(
            sine_lower, sine_upper, x_lower, x_upper
        )
        cosine_y_lower, cosine_y_upper = interval_product(
            cosine_lower, cosine_upper, y_lower, y_upper
        )
        rotated_y_lower = sine_x_lower + cosine_y_lower
        rotated_y_upper = sine_x_upper + cosine_y_upper

        scale_z = max(item[2] for item in self.primitive_scales_m)

        def deformation_interval(
            coordinate_lower: np.ndarray,
            coordinate_upper: np.ndarray,
            taper: float,
            bend: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            factor_a = np.clip(1.0 + taper * z_lower / scale_z, 0.25, 4.0)
            factor_b = np.clip(1.0 + taper * z_upper / scale_z, 0.25, 4.0)
            factor_lower = np.minimum(factor_a, factor_b)
            factor_upper = np.maximum(factor_a, factor_b)
            scaled_lower, scaled_upper = interval_product(
                coordinate_lower,
                coordinate_upper,
                factor_lower,
                factor_upper,
            )
            z_square_lower = np.where(
                (z_lower <= 0.0) & (z_upper >= 0.0),
                0.0,
                np.minimum(np.square(z_lower), np.square(z_upper)),
            )
            z_square_upper = np.maximum(np.square(z_lower), np.square(z_upper))
            bend_lower = np.minimum(bend * z_square_lower, bend * z_square_upper)
            bend_upper = np.maximum(bend * z_square_lower, bend * z_square_upper)
            return scaled_lower + bend_lower, scaled_upper + bend_upper

        deformed_x_lower, deformed_x_upper = deformation_interval(
            rotated_x_lower,
            rotated_x_upper,
            self.taper_per_m[0],
            self.bend_per_m[0],
        )
        deformed_y_lower, deformed_y_upper = deformation_interval(
            rotated_y_lower,
            rotated_y_upper,
            self.taper_per_m[1],
            self.bend_per_m[1],
        )
        sliced_lower = np.asarray(
            (
                float(np.min(deformed_x_lower)) - margin,
                float(np.min(deformed_y_lower)) - margin,
                float(np.min(z_lower)) - margin,
            )
        )
        sliced_upper = np.asarray(
            (
                float(np.max(deformed_x_upper)) + margin,
                float(np.max(deformed_y_upper)) + margin,
                float(np.max(z_upper)) + margin,
            )
        )
        result_lower = np.maximum(old_lower, sliced_lower)
        result_upper = np.minimum(old_upper, sliced_upper)
        if (
            not np.isfinite(result_lower).all()
            or not np.isfinite(result_upper).all()
            or np.any(result_lower >= result_upper)
        ):
            raise RenderError("tight continuous CSG outer bounds are invalid")
        return _freeze(result_lower), _freeze(result_upper)

    def continuous_size_certificate(
        self,
        *,
        sobol_probes: int = 4096,
        maximum_interior_lines: int = 64,
        safety_margin_m: float = 1.0e-6,
    ) -> ShapeSizeCertificate:
        """Certify a mesh-free lower/upper interval for maximum-axis size."""

        probes = _integer("sobol_probes", sobol_probes, minimum=256)
        if probes & (probes - 1):
            raise RenderError("sobol_probes must be a power of two")
        line_limit = _integer(
            "maximum_interior_lines", maximum_interior_lines, minimum=8
        )
        lower, upper = self._continuous_outer_bounds(safety_margin_m=safety_margin_m)
        exponent = int(math.log2(probes))
        unit = qmc.Sobol(d=3, scramble=False).random_base2(exponent)
        points = lower + unit * (upper - lower)
        values = self.signed_distance(points)
        interior = np.flatnonzero(values < -1.0e-10)[:line_limit]
        if interior.size == 0:
            raise RenderError("continuous size certificate found no interior witness")

        best_span = -math.inf
        best_start: np.ndarray | None = None
        best_end: np.ndarray | None = None
        maximum_residual = 0.0
        for point in points[interior]:
            for axis in range(3):
                left = point.copy()
                right = point.copy()
                left[axis] = lower[axis]
                right[axis] = upper[axis]
                left_value = float(self.signed_distance(left[None])[0])
                right_value = float(self.signed_distance(right[None])[0])
                point_value = float(self.signed_distance(point[None])[0])
                if left_value <= 0.0 or right_value <= 0.0 or point_value >= 0.0:
                    raise RenderError(
                        "continuous outer bound did not bracket the geometry"
                    )

                def along(value: float) -> float:
                    query = point.copy()
                    query[axis] = value
                    return float(self.signed_distance(query[None])[0])

                left_root = brentq(
                    along,
                    float(lower[axis]),
                    float(point[axis]),
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
                right_root = brentq(
                    along,
                    float(point[axis]),
                    float(upper[axis]),
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
                start = point.copy()
                end = point.copy()
                start[axis] = left_root
                end[axis] = right_root
                residual = float(
                    np.max(np.abs(self.signed_distance(np.stack((start, end)))))
                )
                maximum_residual = max(maximum_residual, residual)
                span = right_root - left_root
                if span > best_span:
                    best_span = span
                    best_start = start
                    best_end = end
        assert best_start is not None and best_end is not None
        outer_size = float(np.max(upper - lower))
        if not 0.0 < best_span <= outer_size:
            raise RenderError("continuous size certificate interval is invalid")
        return ShapeSizeCertificate(
            outer_lower_m=tuple(map(float, lower)),
            outer_upper_m=tuple(map(float, upper)),
            lower_size_m=float(best_span),
            upper_size_m=outer_size,
            witness_start_m=tuple(map(float, best_start)),
            witness_end_m=tuple(map(float, best_end)),
            sobol_probes=probes,
            interior_lines=int(interior.size),
            maximum_surface_residual_m=maximum_residual,
        )

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

    def _primitive_perturbed_value(
        self, index: int, points_undeformed: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(points_undeformed, dtype=np.float64)
        primitive = self._primitive_distance(
            points,
            self.primitive_scales_m[index],
            self.primitive_offsets_m[index],
            self.primitive_exponents[index],
            self.primitive_yaws_rad[index],
        )
        displacement = self.surface_amplitude_m * np.mean(
            np.sin(
                points * np.asarray(self.surface_frequency_per_m)
                + np.asarray(self.surface_phase_rad)
            ),
            axis=-1,
        )
        return primitive - displacement

    def _primitive_star_certificate(self, index: int) -> bool:
        scale = self.primitive_scales_m[index]
        lower = min(scale) / (math.sqrt(3.0) * max(scale)) - (
            self.surface_amplitude_m
            * float(np.linalg.norm(self.surface_frequency_per_m))
            / 3.0
        )
        center = np.asarray(self.primitive_offsets_m[index])[None, :]
        center_value = float(self._primitive_perturbed_value(index, center)[0])
        return center_value < 0.0 and lower > 0.0

    def _analytic_connectivity_source(self) -> str | None:
        certified = [
            self._primitive_star_certificate(index)
            for index in range(self.primitive_count)
        ]
        if self.primitive_count == 1 and certified[0]:
            return "strict_radial_star_shaped"
        if all(operation == "union" for operation in self.operations) and all(
            certified
        ):
            adjacency = np.eye(self.primitive_count, dtype=np.bool_)
            weights = np.linspace(0.0, 1.0, 257)
            for left in range(self.primitive_count):
                for right in range(left + 1, self.primitive_count):
                    start = np.asarray(self.primitive_offsets_m[left])
                    end = np.asarray(self.primitive_offsets_m[right])
                    points = start[None, :] + weights[:, None] * (end - start)[None, :]
                    overlap = bool(
                        np.any(
                            (self._primitive_perturbed_value(left, points) < 0.0)
                            & (self._primitive_perturbed_value(right, points) < 0.0)
                        )
                    )
                    adjacency[left, right] = adjacency[right, left] = overlap
            reached = {0}
            while True:
                expanded = reached | {
                    target
                    for source in reached
                    for target in range(self.primitive_count)
                    if adjacency[source, target]
                }
                if expanded == reached:
                    break
                reached = expanded
            if len(reached) == self.primitive_count:
                return "connected_union_graph"
        if (
            self.surface_amplitude_m == 0.0
            and all(
                operation in {"union", "intersection"} for operation in self.operations
            )
            and all(operation == "intersection" for operation in self.operations[1:])
            and all(
                vertical <= 2.0 and horizontal <= 2.0
                for vertical, horizontal in self.primitive_exponents
            )
        ):
            candidates = np.asarray(
                (
                    *self.primitive_offsets_m,
                    tuple(np.mean(np.asarray(self.primitive_offsets_m), axis=0)),
                )
            )
            if any(
                all(
                    self._primitive_perturbed_value(index, point[None, :])[0] < 0.0
                    for index in range(self.primitive_count)
                )
                for point in candidates
            ):
                return "nonempty_convex_intersection"
        if (
            self.surface_amplitude_m == 0.0
            and self.primitive_count == 2
            and self.operations == ("union", "difference")
            and all(exponent == (1.0, 1.0) for exponent in self.primitive_exponents)
            and all(
                np.all(np.asarray(scale) == scale[0])
                for scale in self.primitive_scales_m
            )
        ):
            outer_radius = self.primitive_scales_m[0][0]
            inner_radius = self.primitive_scales_m[1][0]
            separation = float(
                np.linalg.norm(
                    np.asarray(self.primitive_offsets_m[0])
                    - np.asarray(self.primitive_offsets_m[1])
                )
            )
            if separation + inner_radius < outer_radius:
                return "strictly_contained_spherical_cavity"
        return None

    def _implicit_interval(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_lower, y_lower, z_lower = lower[:, 0], lower[:, 1], lower[:, 2]
        x_upper, y_upper, z_upper = upper[:, 0], upper[:, 1], upper[:, 2]
        z2_lower, z2_upper = _interval_square(z_lower, z_upper)
        bend_x_lower, bend_x_upper = _interval_scale(
            z2_lower, z2_upper, -self.bend_per_m[0]
        )
        bend_y_lower, bend_y_upper = _interval_scale(
            z2_lower, z2_upper, -self.bend_per_m[1]
        )
        x_lower, x_upper = _interval_add(x_lower, x_upper, bend_x_lower, bend_x_upper)
        y_lower, y_upper = _interval_add(y_lower, y_upper, bend_y_lower, bend_y_upper)
        scale_z = max(item[2] for item in self.primitive_scales_m)
        factor_x_lower, factor_x_upper = _interval_add(
            np.ones_like(z_lower),
            np.ones_like(z_upper),
            *_interval_scale(z_lower, z_upper, self.taper_per_m[0] / scale_z),
        )
        factor_y_lower, factor_y_upper = _interval_add(
            np.ones_like(z_lower),
            np.ones_like(z_upper),
            *_interval_scale(z_lower, z_upper, self.taper_per_m[1] / scale_z),
        )
        factor_x_lower = np.clip(factor_x_lower, 0.25, 4.0)
        factor_x_upper = np.clip(factor_x_upper, 0.25, 4.0)
        factor_y_lower = np.clip(factor_y_lower, 0.25, 4.0)
        factor_y_upper = np.clip(factor_y_upper, 0.25, 4.0)
        x_lower, x_upper = _interval_multiply(
            x_lower, x_upper, 1.0 / factor_x_upper, 1.0 / factor_x_lower
        )
        y_lower, y_upper = _interval_multiply(
            y_lower, y_upper, 1.0 / factor_y_upper, 1.0 / factor_y_lower
        )
        angle_lower, angle_upper = _interval_scale(
            z_lower, z_upper, -self.twist_rad_per_m
        )
        cosine_lower, cosine_upper = _interval_trigonometric(
            angle_lower, angle_upper, cosine=True
        )
        sine_lower, sine_upper = _interval_trigonometric(angle_lower, angle_upper)
        cosine_x_lower, cosine_x_upper = _interval_multiply(
            cosine_lower, cosine_upper, x_lower, x_upper
        )
        sine_y_lower, sine_y_upper = _interval_multiply(
            sine_lower, sine_upper, y_lower, y_upper
        )
        sine_x_lower, sine_x_upper = _interval_multiply(
            sine_lower, sine_upper, x_lower, x_upper
        )
        cosine_y_lower, cosine_y_upper = _interval_multiply(
            cosine_lower, cosine_upper, y_lower, y_upper
        )
        ux_lower, ux_upper = _interval_add(
            cosine_x_lower, cosine_x_upper, -sine_y_upper, -sine_y_lower
        )
        uy_lower, uy_upper = _interval_add(
            sine_x_lower, sine_x_upper, cosine_y_lower, cosine_y_upper
        )

        result_lower: np.ndarray | None = None
        result_upper: np.ndarray | None = None
        for scale, offset, exponent, yaw, operation in zip(
            self.primitive_scales_m,
            self.primitive_offsets_m,
            self.primitive_exponents,
            self.primitive_yaws_rad,
            self.operations,
            strict=True,
        ):
            local_x_lower = ux_lower - offset[0]
            local_x_upper = ux_upper - offset[0]
            local_y_lower = uy_lower - offset[1]
            local_y_upper = uy_upper - offset[1]
            local_z_lower = z_lower - offset[2]
            local_z_upper = z_upper - offset[2]
            cosine, sine = math.cos(-yaw), math.sin(-yaw)
            rotated_x_lower, rotated_x_upper = _interval_add(
                *_interval_scale(local_x_lower, local_x_upper, cosine),
                *_interval_scale(local_y_lower, local_y_upper, -sine),
            )
            rotated_y_lower, rotated_y_upper = _interval_add(
                *_interval_scale(local_x_lower, local_x_upper, sine),
                *_interval_scale(local_y_lower, local_y_upper, cosine),
            )
            axis_x_lower, axis_x_upper = _interval_absolute(
                rotated_x_lower / scale[0], rotated_x_upper / scale[0]
            )
            axis_y_lower, axis_y_upper = _interval_absolute(
                rotated_y_lower / scale[1], rotated_y_upper / scale[1]
            )
            axis_z_lower, axis_z_upper = _interval_absolute(
                local_z_lower / scale[2], local_z_upper / scale[2]
            )
            vertical, horizontal = exponent
            axis_x_lower, axis_x_upper = _interval_power(
                axis_x_lower, axis_x_upper, 2.0 / horizontal
            )
            axis_y_lower, axis_y_upper = _interval_power(
                axis_y_lower, axis_y_upper, 2.0 / horizontal
            )
            xy_lower, xy_upper = _interval_add(
                axis_x_lower, axis_x_upper, axis_y_lower, axis_y_upper
            )
            xy_lower, xy_upper = _interval_power(
                xy_lower, xy_upper, horizontal / vertical
            )
            axis_z_lower, axis_z_upper = _interval_power(
                axis_z_lower, axis_z_upper, 2.0 / vertical
            )
            total_lower, total_upper = _interval_add(
                xy_lower, xy_upper, axis_z_lower, axis_z_upper
            )
            primitive_lower, primitive_upper = _interval_power(
                total_lower, total_upper, vertical / 2.0
            )
            primitive_lower, primitive_upper = _interval_outward(
                (primitive_lower - 1.0) * min(scale),
                (primitive_upper - 1.0) * min(scale),
            )
            if result_lower is None:
                result_lower, result_upper = primitive_lower, primitive_upper
            elif operation == "union":
                result_lower, result_upper = _interval_outward(
                    np.minimum(result_lower, primitive_lower),
                    np.minimum(result_upper, primitive_upper),
                )
            elif operation == "difference":
                result_lower, result_upper = _interval_outward(
                    np.maximum(result_lower, -primitive_upper),
                    np.maximum(result_upper, -primitive_lower),
                )
            else:
                result_lower, result_upper = _interval_outward(
                    np.maximum(result_lower, primitive_lower),
                    np.maximum(result_upper, primitive_upper),
                )
        assert result_lower is not None and result_upper is not None
        displacement_lower = np.zeros_like(result_lower)
        displacement_upper = np.zeros_like(result_upper)
        for coordinate_lower, coordinate_upper, frequency, phase in zip(
            (ux_lower, uy_lower, z_lower),
            (ux_upper, uy_upper, z_upper),
            self.surface_frequency_per_m,
            self.surface_phase_rad,
            strict=True,
        ):
            phase_lower, phase_upper = _interval_outward(
                coordinate_lower * frequency + phase,
                coordinate_upper * frequency + phase,
            )
            sine_lower, sine_upper = _interval_trigonometric(phase_lower, phase_upper)
            displacement_lower += sine_lower
            displacement_upper += sine_upper
        displacement_lower *= self.surface_amplitude_m / 3.0
        displacement_upper *= self.surface_amplitude_m / 3.0
        return _interval_outward(
            result_lower - displacement_upper,
            result_upper - displacement_lower,
        )

    def _interval_connectivity_stats(self, cells: int) -> tuple[int, int, int]:
        lower, upper = self._continuous_outer_bounds(safety_margin_m=1.0e-6)
        edges = [np.linspace(lower[axis], upper[axis], cells + 1) for axis in range(3)]
        state = np.empty((cells, cells, cells), dtype=np.int8)
        total = cells**3
        batch = 131_072
        for start in range(0, total, batch):
            flat = np.arange(start, min(start + batch, total), dtype=np.int64)
            x = flat // (cells * cells)
            y = (flat // cells) % cells
            z = flat % cells
            box_lower = np.column_stack((edges[0][x], edges[1][y], edges[2][z]))
            box_upper = np.column_stack(
                (edges[0][x + 1], edges[1][y + 1], edges[2][z + 1])
            )
            value_lower, value_upper = self._implicit_interval(box_lower, box_upper)
            current = np.zeros(len(flat), dtype=np.int8)
            current[value_lower > 0.0] = -1
            current[value_upper < 0.0] = 1
            state.reshape(-1)[start : start + len(flat)] = current
        structure = ndimage.generate_binary_structure(3, 1)
        possible_labels, possible_count = ndimage.label(
            state != -1, structure=structure
        )
        _, definite_count = ndimage.label(state == 1, structure=structure)
        witnessed = np.unique(possible_labels[state == 1])
        witnessed = witnessed[witnessed > 0]
        return (
            int(len(witnessed)),
            int(definite_count),
            int(possible_count - len(witnessed)),
        )

    def continuous_connectivity_certificate(
        self,
    ) -> ShapeConnectivityCertificate:
        """Return only sufficient continuous topology evidence."""

        source = self._analytic_connectivity_source()
        if source is not None:
            return ShapeConnectivityCertificate(
                "connected", source, (1, 1, 0), (1, 1, 0)
            )
        standard = self._interval_connectivity_stats(64)
        strict = self._interval_connectivity_stats(128)
        if strict[0] >= 2 and strict[0] >= standard[0]:
            return ShapeConnectivityCertificate(
                "disconnected", "strict_interval_separation", standard, strict
            )
        return ShapeConnectivityCertificate(
            "unresolved", "insufficient_continuous_evidence", standard, strict
        )

    @property
    def connectivity_certificate(self) -> ShapeConnectivityCertificate:
        return self._connectivity

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
        # Continuous certificates establish nonempty interior.  The voxel grid
        # is retained only to report finite sampled support and closed bounds.
        if inside_count < 1 or inside_count == inside.size:
            raise RenderError("CSG result has no effective enclosed volume")
        boundary = np.zeros_like(inside)
        boundary[[0, -1], :, :] = True
        boundary[:, [0, -1], :] = True
        boundary[:, :, [0, -1]] = True
        if bool(np.any(inside & boundary)):
            raise RenderError(
                "shape touches its conservative bound and is not verified closed"
            )
        if self._connectivity.state != "connected":
            raise RenderError("shape lacks a continuous connectedness certificate")
        components = 1
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
        if surface_count < 1:
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
            bracket_lo = np.full(ids.size, np.inf, dtype=np.float64)
            bracket_hi = np.full(ids.size, np.inf, dtype=np.float64)
            if bool(has_hit.any()):
                hit_rows = np.flatnonzero(has_hit)
                entry_index = np.argmax(entry[hit_rows], axis=1) + 1
                exit_index = np.argmax(exit_surface[hit_rows], axis=1) + 1
                first = np.where(starts_inside[hit_rows], exit_index, entry_index)
                bracket_lo[hit_rows] = ray_t[hit_rows, first - 1]
                bracket_hi[hit_rows] = ray_t[hit_rows, first]

            # A thin outside-inside-outside segment can lie between two coarse
            # samples. Refine only nearby positive intervals; a hit still
            # requires an explicit sign change and is never inferred from
            # proximity alone.
            adaptive_rows = np.flatnonzero(~starts_inside)
            if adaptive_rows.size:
                interval_ray = np.repeat(adaptive_rows, steps - 1)
                interval_lo = ray_t[adaptive_rows, :-1].reshape(-1)
                interval_hi = ray_t[adaptive_rows, 1:].reshape(-1)
                value_lo = values[adaptive_rows, :-1].reshape(-1)
                value_hi = values[adaptive_rows, 1:].reshape(-1)
                width = interval_hi - interval_lo
                keep = (
                    (value_lo > 0.0)
                    & (value_hi > 0.0)
                    & (np.minimum(value_lo, value_hi) <= 4.0 * width)
                    & (interval_lo < bracket_lo[interval_ray])
                )
                interval_ray = interval_ray[keep]
                interval_lo = interval_lo[keep]
                interval_hi = interval_hi[keep]
                value_lo = value_lo[keep]
                value_hi = value_hi[keep]
                for _depth in range(8):
                    if not interval_ray.size:
                        break
                    middle = 0.5 * (interval_lo + interval_hi)
                    middle_points = (
                        origin[ids[interval_ray]]
                        + middle[:, None] * directions[ids[interval_ray]]
                    )
                    value_middle = self.signed_distance(middle_points)
                    crossed = value_middle <= 0.0
                    for candidate in np.flatnonzero(crossed):
                        ray = int(interval_ray[candidate])
                        if interval_lo[candidate] < bracket_lo[ray]:
                            bracket_lo[ray] = interval_lo[candidate]
                            bracket_hi[ray] = middle[candidate]

                    outside = ~crossed
                    child_ray = np.concatenate(
                        (interval_ray[outside], interval_ray[outside])
                    )
                    child_lo = np.concatenate((interval_lo[outside], middle[outside]))
                    child_hi = np.concatenate((middle[outside], interval_hi[outside]))
                    child_value_lo = np.concatenate(
                        (value_lo[outside], value_middle[outside])
                    )
                    child_value_hi = np.concatenate(
                        (value_middle[outside], value_hi[outside])
                    )
                    child_width = child_hi - child_lo
                    keep = (
                        np.minimum(child_value_lo, child_value_hi) <= 4.0 * child_width
                    ) & (child_lo < bracket_lo[child_ray])
                    interval_ray = child_ray[keep]
                    interval_lo = child_lo[keep]
                    interval_hi = child_hi[keep]
                    value_lo = child_value_lo[keep]
                    value_hi = child_value_hi[keep]

            has_hit = np.isfinite(bracket_lo)
            if not bool(has_hit.any()):
                continue
            hit_ids = ids[has_hit]
            hit_starts_inside = starts_inside[has_hit]
            lo = bracket_lo[has_hit]
            hi = bracket_hi[has_hit]
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

    @staticmethod
    def _schema7_rng(seed: int, stream: int, *coordinates: int) -> np.random.Generator:
        """Keep qualified schema-7 factors on structurally separate streams."""
        return np.random.default_rng(
            np.random.SeedSequence((seed, stream, *coordinates))
        )

    @classmethod
    def _schema7_base_scale(
        cls, seed: int, half: float
    ) -> tuple[tuple[float, float, float], str]:
        family_value = float(cls._schema7_rng(seed, SCHEMA7_FAMILY_STREAM).random())
        if family_value < 0.4:
            family = 0
        elif family_value < 0.6:
            family = 1
        elif family_value < 0.8:
            family = 2
        else:
            family = 3
        rng = cls._schema7_rng(seed, SCHEMA7_RATIO_STREAM)
        if family == 0:
            factors = np.sort(rng.uniform(0.65, 1.25, 3))[::-1]
            r21, r31 = float(factors[1] / factors[0]), float(factors[2] / factors[0])
        elif family == 1:
            r31 = float(rng.uniform(0.75, 1.0))
            r21 = float(rng.uniform(r31, 1.0))
        elif family == 2:
            r21, r31 = float(rng.uniform(0.75, 1.0)), float(rng.uniform(0.2, 0.4))
        else:
            r21 = float(rng.uniform(0.3, 0.5))
            r31 = float(rng.uniform(0.15, min(0.4, r21)))
        permutation = AXIS_PERMUTATIONS[
            int(cls._schema7_rng(seed, SCHEMA7_AXIS_STREAM).integers(0, 6))
        ]
        ordered = np.asarray((half, half * r21, half * r31))
        return tuple(
            float(value) for value in ordered[list(permutation)]
        ), SHAPE_FAMILIES[family]

    @classmethod
    def _perturbed_primitive_value(
        cls,
        scale: tuple[float, float, float],
        center: np.ndarray,
        exponent: tuple[float, float],
        yaw: float,
        point: np.ndarray,
        amplitude: float,
        frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> float:
        base = float(
            cls._primitive_distance(point[None], scale, tuple(center), exponent, yaw)[0]
        )
        displacement = amplitude * float(
            np.mean(np.sin(point * np.asarray(frequency) + np.asarray(phase)))
        )
        return base - displacement

    @classmethod
    def _primitive_radial_radius(
        cls,
        scale: tuple[float, float, float],
        center: np.ndarray,
        exponent: tuple[float, float],
        yaw: float,
        direction: np.ndarray,
        amplitude: float,
        frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> float:
        def implicit(distance: float) -> float:
            return cls._perturbed_primitive_value(
                scale,
                center,
                exponent,
                yaw,
                center + distance * direction,
                amplitude,
                frequency,
                phase,
            )

        if implicit(0.0) >= 0.0:
            raise RenderError("schema-7 primitive center is not strictly interior")
        upper = 2.0 * float(np.linalg.norm(scale))
        while implicit(upper) <= 0.0 and upper < 64.0:
            upper *= 2.0
        if implicit(upper) <= 0.0:
            raise RenderError("schema-7 radial boundary was not bracketed")
        return float(brentq(implicit, 0.0, upper, xtol=1e-13, rtol=1e-13))

    @classmethod
    def _shared_witness_placement(
        cls,
        parent_scale: tuple[float, float, float],
        parent_center: np.ndarray,
        parent_exponent: tuple[float, float],
        parent_yaw: float,
        child_scale: tuple[float, float, float],
        child_exponent: tuple[float, float],
        child_yaw: float,
        direction: np.ndarray,
        tau_parent: float,
        tau_child: float,
        amplitude: float,
        frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Construct one authoritative witness before global deformation."""
        parent_radius = cls._primitive_radial_radius(
            parent_scale,
            parent_center,
            parent_exponent,
            parent_yaw,
            direction,
            amplitude,
            frequency,
            phase,
        )
        witness = parent_center + tau_parent * parent_radius * direction

        # Translation changes the global-coordinate surface phase, so solve
        # placement and the opposite-direction child boundary together.
        def child_boundary(offset_distance: float) -> float:
            child_center = witness + offset_distance * direction
            boundary = witness - offset_distance * (1.0 / tau_child - 1.0) * direction
            return cls._perturbed_primitive_value(
                child_scale,
                child_center,
                child_exponent,
                child_yaw,
                boundary,
                amplitude,
                frequency,
                phase,
            )

        upper = 2.0 * float(np.linalg.norm(child_scale))
        while child_boundary(upper) <= 0.0 and upper < 64.0:
            upper *= 2.0
        if child_boundary(0.0) >= 0.0 or child_boundary(upper) <= 0.0:
            raise RenderError("schema-7 child boundary was not bracketed")
        offset_distance = float(
            brentq(child_boundary, 0.0, upper, xtol=1e-13, rtol=1e-13)
        )
        child_center = witness + offset_distance * direction
        child_radius = cls._primitive_radial_radius(
            child_scale,
            child_center,
            child_exponent,
            child_yaw,
            -direction,
            amplitude,
            frequency,
            phase,
        )
        if abs(offset_distance - tau_child * child_radius) > 1e-10:
            raise RenderError("schema-7 shared-witness formula is inconsistent")
        parent_margin = -cls._perturbed_primitive_value(
            parent_scale,
            parent_center,
            parent_exponent,
            parent_yaw,
            witness,
            amplitude,
            frequency,
            phase,
        )
        child_margin = -cls._perturbed_primitive_value(
            child_scale,
            child_center,
            child_exponent,
            child_yaw,
            witness,
            amplitude,
            frequency,
            phase,
        )
        if parent_margin <= 0.0 or child_margin <= 0.0:
            raise RenderError("schema-7 shared witness is not strictly interior")
        return child_center, witness, parent_margin, child_margin

    @classmethod
    def sample_with_report(
        cls,
        seed: int,
        *,
        primitive_count: int | None = None,
        size_m_range: tuple[float, float] = (0.2, 3.0),
    ) -> tuple["ShapeSpec", ShapeGenerationReport]:
        """Sample one shape and expose deterministic acceptance diagnostics."""

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
        lower_rejections = 0
        upper_rejections = 0
        disconnected_rejections = 0
        unresolved_rejections = 0
        other = 0
        for proposal_count in range(1, 65):
            count = (
                int(rng.integers(1, 6)) if primitive_count is None else primitive_count
            )
            half = float(rng.uniform(minimum / 2.0, maximum / 2.0))
            rng.uniform(0.65, 1.25, size=3)  # Retired schema-6 axis draw.
            base, shape_family = cls._schema7_base_scale(seed, half)
            base_array = np.asarray(base)
            scales = [base]
            exponents = [tuple(map(float, rng.uniform(0.55, 1.65, size=2)))]
            yaws = [float(rng.uniform(-math.pi, math.pi))]
            child_events: list[tuple[int, int, float]] = []
            for child_index in range(1, count):
                parent = int(rng.integers(0, child_index))
                axis = int(rng.integers(0, 3))
                sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
                rng.uniform(0.10, 0.50)  # Retired embedded-center draw.
                scale = base_array * rng.uniform(0.32, 0.78, size=3)
                scales.append(tuple(map(float, scale)))
                exponents.append(tuple(map(float, rng.uniform(0.5, 1.8, size=2))))
                yaws.append(float(rng.uniform(-math.pi, math.pi)))
                child_events.append((parent, axis, sign))
            amplitude = float(rng.uniform(0.0, 0.08 * min(base)))
            twist = float(rng.uniform(-0.65, 0.65))
            bend = tuple(map(float, rng.uniform(-0.12, 0.12, size=2)))
            taper = tuple(map(float, rng.uniform(-0.18, 0.18, size=2)))
            frequency = tuple(map(float, rng.uniform(0.6, 2.2, size=3)))
            phase = tuple(map(float, rng.uniform(-math.pi, math.pi, size=3)))
            try:
                offsets = [np.zeros(3)]
                child_parents: list[int] = []
                shared_witnesses: list[tuple[float, float, float]] = []
                parent_margins: list[float] = []
                child_margins: list[float] = []
                for child_index, (parent, axis, sign) in enumerate(child_events, 1):
                    direction = np.zeros(3)
                    parent_yaw = yaws[parent]
                    if axis == 0:
                        direction[:2] = (math.cos(parent_yaw), math.sin(parent_yaw))
                    elif axis == 1:
                        direction[:2] = (-math.sin(parent_yaw), math.cos(parent_yaw))
                    else:
                        direction[2] = 1.0
                    direction *= sign
                    tau_parent = float(
                        cls._schema7_rng(
                            seed,
                            SCHEMA7_PARENT_TAU_STREAM,
                            proposal_count,
                            child_index,
                        ).uniform(0.65, 0.85)
                    )
                    tau_child = float(
                        cls._schema7_rng(
                            seed,
                            SCHEMA7_CHILD_TAU_STREAM,
                            proposal_count,
                            child_index,
                        ).uniform(0.55, 0.80)
                    )
                    offset, witness, parent_margin, child_margin = (
                        cls._shared_witness_placement(
                            scales[parent],
                            offsets[parent],
                            exponents[parent],
                            yaws[parent],
                            scales[child_index],
                            exponents[child_index],
                            yaws[child_index],
                            direction,
                            tau_parent,
                            tau_child,
                            amplitude,
                            frequency,
                            phase,
                        )
                    )
                    offsets.append(offset)
                    child_parents.append(parent)
                    shared_witnesses.append(tuple(map(float, witness)))
                    parent_margins.append(parent_margin)
                    child_margins.append(child_margin)
                result = cls(
                    primitive_scales_m=tuple(scales),
                    primitive_offsets_m=tuple(
                        tuple(map(float, item)) for item in offsets
                    ),
                    primitive_exponents=tuple(exponents),
                    primitive_yaws_rad=tuple(yaws),
                    operations=("union",) * count,
                    twist_rad_per_m=twist,
                    bend_per_m=bend,
                    taper_per_m=taper,
                    surface_amplitude_m=amplitude,
                    surface_frequency_per_m=frequency,
                    surface_phase_rad=phase,
                )
                # Require connectivity at both audit and placement resolutions.
                result.geometry_report(resolution=31)
                result.geometry_report(resolution=41)
                if count == 1:
                    # Preserve the E16-v3 qualified single-primitive path.
                    lower, upper = result.continuous_bounds(
                        maximum_iterations=80,
                        population_size=10,
                        safety_margin_m=1.0e-6,
                    )
                    size_lower = float(np.max(upper - lower))
                    size_upper = size_lower
                    size_definition = "continuous-deformed-surface-aabb"
                else:
                    certificate = result.continuous_size_certificate(
                        sobol_probes=4096,
                        maximum_interior_lines=64,
                        safety_margin_m=1.0e-6,
                    )
                    lower, upper = result.tight_continuous_outer_bounds(
                        z_slabs=256,
                        safety_margin_m=1.0e-6,
                    )
                    size_lower = certificate.lower_size_m
                    size_upper = float(np.max(upper - lower))
                    size_definition = "continuous-union-tight-certified-interval"
                if size_upper > maximum:
                    upper_rejections += 1
                    continue
                if size_lower < minimum:
                    lower_rejections += 1
                    continue
                return result, ShapeGenerationReport(
                    generator_schema=PROCEDURAL_GENERATOR_SCHEMA,
                    proposal_count=proposal_count,
                    lower_certificate_rejections=lower_rejections,
                    upper_certificate_rejections=upper_rejections,
                    connectivity_disconnected_rejections=disconnected_rejections,
                    connectivity_unresolved_rejections=unresolved_rejections,
                    other_rejections=other,
                    accepted_size_lower_m=size_lower,
                    accepted_size_upper_m=size_upper,
                    outer_lower_m=tuple(map(float, lower)),
                    outer_upper_m=tuple(map(float, upper)),
                    size_definition=size_definition,
                    shape_family=shape_family,
                    child_parent_indices=tuple(child_parents),
                    shared_witnesses_undeformed_m=tuple(shared_witnesses),
                    witness_parent_margins_m=tuple(parent_margins),
                    witness_child_margins_m=tuple(child_margins),
                )
            except RenderError as error:
                if str(error) == "continuous CSG is certified disconnected":
                    disconnected_rejections += 1
                elif str(error) == "continuous CSG connectivity is unresolved":
                    unresolved_rejections += 1
                else:
                    other += 1
                continue
        raise RenderError(
            "could not sample a connected shape within 64 deterministic attempts"
        )

    @classmethod
    def sample(
        cls,
        seed: int,
        *,
        primitive_count: int | None = None,
        size_m_range: tuple[float, float] = (0.2, 3.0),
    ) -> "ShapeSpec":
        """Sample a reproducible constructively connected schema-6 shape."""

        shape, _ = cls.sample_with_report(
            seed,
            primitive_count=primitive_count,
            size_m_range=size_m_range,
        )
        return shape

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


InsertShape: TypeAlias = ShapeSpec


def shape_from_dict(value: Mapping[str, object]) -> InsertShape:
    kind = value.get("kind") if isinstance(value, Mapping) else None
    if kind == "procedural-csg":
        return ShapeSpec.from_dict(value)
    raise RenderError(f"unsupported geometry kind: {kind!r}")


def sample_training_anomaly_shape(
    seed: int,
    *,
    size_m_range: tuple[float, float] = (0.2, 3.0),
) -> ShapeSpec:
    """Sample the sole procedural anomaly geometry used by schema 33."""

    return ShapeSpec.sample(seed, size_m_range=size_m_range)


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
    shape_generation_report: ShapeGenerationReport | None = None

    def __post_init__(self) -> None:
        identifier = _integer("object_id", self.object_id, minimum=1)
        if identifier > MAX_OBJECT_ID:
            raise RenderError(f"object_id must not exceed {MAX_OBJECT_ID}")
        label = str(self.label)
        if label not in OBJECT_LABELS:
            raise RenderError(f"object label must be one of {OBJECT_LABELS}")
        if not isinstance(self.shape, ShapeSpec) or not isinstance(
            self.material, MaterialSpec
        ):
            raise TypeError("shape and material have unsupported types")
        report = self.shape_generation_report
        if report is not None and not isinstance(report, ShapeGenerationReport):
            raise TypeError("shape_generation_report has an unsupported type")
        if report is not None and (
            not isinstance(self.shape, ShapeSpec)
            or report.generator_schema != PROCEDURAL_GENERATOR_SCHEMA
        ):
            raise RenderError(
                "only current schema-7 objects may carry a generation report"
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
            "shape_generation_report": None
            if self.shape_generation_report is None
            else self.shape_generation_report.to_dict(),
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
            "shape_generation_report",
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
            shape_generation_report=None
            if value["shape_generation_report"] is None
            else ShapeGenerationReport.from_dict(value["shape_generation_report"]),  # type: ignore[arg-type]
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
        if len(objects) > 9:
            raise RenderError("anomaly-proxy count must lie in [0,9]")
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
    def anomaly_proxy_count(self) -> int:
        return len(self.objects)

    @property
    def world_type(self) -> str:
        if not self.objects:
            return "pure_normal"
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

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlacementRecord:
    """Reproduce one accepted entity and every rejected support proposal."""

    object_id: int
    label: ObjectLabel
    shape_seed: int
    material_seed: int
    yaw_seed: int
    accepted_proposal: int
    support_pool_index: int
    support_frame: int
    support_slot: int
    support_semantic: int
    proposal_pool_indices: tuple[int, ...]
    rejection_reasons: tuple[str, ...]
    proposal_minimum_obstacle_sdf_m: tuple[float, ...]
    minimum_obstacle_sdf_m: float
    accepted_shape_proposal: int = 0
    shape_proposal_seeds: tuple[int, ...] = ()
    grounding_rejection_seeds: tuple[int, ...] = ()
    grounding_standard_lower_support_m: float = math.nan
    grounding_strict_lower_support_m: float = math.nan
    grounding_buried_fraction: float = math.nan

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "shape_seed": self.shape_seed,
            "material_seed": self.material_seed,
            "yaw_seed": self.yaw_seed,
            "accepted_proposal": self.accepted_proposal,
            "support_pool_index": self.support_pool_index,
            "support_frame": self.support_frame,
            "support_slot": self.support_slot,
            "support_semantic": self.support_semantic,
            "proposal_pool_indices": list(self.proposal_pool_indices),
            "rejection_reasons": list(self.rejection_reasons),
            "proposal_minimum_obstacle_sdf_m": list(
                self.proposal_minimum_obstacle_sdf_m
            ),
            "minimum_obstacle_sdf_m": self.minimum_obstacle_sdf_m,
            "accepted_shape_proposal": self.accepted_shape_proposal,
            "shape_proposal_seeds": list(self.shape_proposal_seeds),
            "grounding_rejection_seeds": list(self.grounding_rejection_seeds),
            "grounding_standard_lower_support_m": (
                self.grounding_standard_lower_support_m
            ),
            "grounding_strict_lower_support_m": self.grounding_strict_lower_support_m,
            "grounding_buried_fraction": self.grounding_buried_fraction,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PlacementRecord":
        plain = dict(value)
        for name in (
            "proposal_pool_indices",
            "rejection_reasons",
            "proposal_minimum_obstacle_sdf_m",
            "shape_proposal_seeds",
            "grounding_rejection_seeds",
        ):
            if name in plain:
                plain[name] = tuple(plain[name])  # type: ignore[arg-type]
        try:
            return cls(**plain)  # type: ignore[arg-type]
        except TypeError as error:
            raise RenderError(f"invalid PlacementRecord JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class WorldGenerationReport:
    """Separate immutable record of all random streams and placement decisions."""

    world_seed: int
    source_sequence_id: int
    world_type: str
    world_attempt: int
    anomaly_count: int
    count_seed: int
    placement_attempt_seed: int
    placements: tuple[PlacementRecord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "format": WORLD_REPORT_FORMAT,
            "world_seed": self.world_seed,
            "source_sequence_id": self.source_sequence_id,
            "world_type": self.world_type,
            "world_attempt": self.world_attempt,
            "anomaly_count": self.anomaly_count,
            "count_seed": self.count_seed,
            "placement_attempt_seed": self.placement_attempt_seed,
            "placements": [item.to_dict() for item in self.placements],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorldGenerationReport":
        if not isinstance(value, Mapping) or value.get("format") != WORLD_REPORT_FORMAT:
            raise RenderError("WorldGenerationReport JSON has an unsupported format")
        if set(value) != {
            "format",
            "world_seed",
            "source_sequence_id",
            "world_type",
            "world_attempt",
            "anomaly_count",
            "count_seed",
            "placement_attempt_seed",
            "placements",
        }:
            raise RenderError("WorldGenerationReport JSON fields are invalid")
        placements = value["placements"]
        if not isinstance(placements, Sequence) or isinstance(placements, (str, bytes)):
            raise RenderError("WorldGenerationReport placements must be an array")
        return cls(
            world_seed=value["world_seed"],  # type: ignore[arg-type]
            source_sequence_id=value["source_sequence_id"],  # type: ignore[arg-type]
            world_type=str(value["world_type"]),
            world_attempt=value["world_attempt"],  # type: ignore[arg-type]
            anomaly_count=value["anomaly_count"],  # type: ignore[arg-type]
            count_seed=value["count_seed"],  # type: ignore[arg-type]
            placement_attempt_seed=value["placement_attempt_seed"],  # type: ignore[arg-type]
            placements=tuple(PlacementRecord.from_dict(item) for item in placements),  # type: ignore[arg-type]
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
            raise RenderError(
                "a published return lies behind its calibrated beam origin"
            )
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
    eta = (
        math.pi
        + gamma
        - 2.0 * math.pi * raw_column / columns
        + shift * (2.0 * math.pi / columns)
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
        math.pi + gamma - 2.0 * math.pi * np.arange(columns, dtype=np.float64) / columns
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
    *,
    slot_ids: np.ndarray | None = None,
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
    if slot_ids is None:
        if count != ray_grid.slot_count:
            raise RenderError("compact object competition requires original slot IDs")
        slots = np.arange(count, dtype=np.int32)
    else:
        slots = np.asarray(slot_ids, dtype=np.int32)
        if (
            slots.shape != (count,)
            or np.any((slots < 0) | (slots >= ray_grid.slot_count))
            or np.unique(slots).size != count
        ):
            raise RenderError("object competition slot IDs are invalid")
    beam_ids = ray_grid.beam_ids[slots]
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
    anomaly_proxy_mask: np.ndarray
    inserted_mask: np.ndarray
    occluded_original_mask: np.ndarray
    unchanged_normal_mask: np.ndarray
    object_id_internal: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.source.xyzi.shape[0])
        packed = np.asarray(self.packed_labels)
        anomaly_proxy = np.asarray(self.anomaly_proxy_mask)
        inserted = np.asarray(self.inserted_mask)
        occluded = np.asarray(self.occluded_original_mask)
        unchanged = np.asarray(self.unchanged_normal_mask)
        object_id = np.asarray(self.object_id_internal)
        if packed.dtype != np.uint32 or packed.shape != (count,):
            raise TypeError("packed_labels must be uint32[slot]")
        for name, value in (
            ("anomaly_proxy_mask", anomaly_proxy),
            ("inserted_mask", inserted),
            ("occluded_original_mask", occluded),
            ("unchanged_normal_mask", unchanged),
        ):
            if value.dtype != np.bool_ or value.shape != (count,):
                raise TypeError(f"{name} must be bool[slot]")
        if object_id.dtype != np.int32 or object_id.shape != (count,):
            raise TypeError("object_id_internal must be int32[slot]")
        if not np.array_equal(inserted, anomaly_proxy):
            raise RenderError("inserted mask must equal the anomaly-proxy mask")
        if np.any(occluded & ~inserted) or np.any(inserted & unchanged):
            raise RenderError("render masks have contradictory slot semantics")
        if np.any((object_id >= 0) != inserted):
            raise RenderError(
                "internal object IDs must identify exactly inserted slots"
            )
        semantic = (packed & np.uint32(0xFFFF)).astype(np.uint16)
        if not np.all(semantic[anomaly_proxy] == np.uint16(2)):
            raise RenderError("anomaly-proxy returns must carry raw semantic 2")
        if self.source.labels is not None and not np.array_equal(
            self.source.labels.packed, packed
        ):
            raise RenderError("SourceFrame labels and packed_labels differ")
        for name, value in (
            ("packed_labels", packed),
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


def _frame_trace_context(
    source: SourceFrame,
    ray_grid: RayGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the immutable ray/native-return state shared by one frame render."""

    rotation, lidar_origin_world = _pose(source)
    directions_sensor = ray_grid.directions_for(source)
    directions_world = directions_sensor @ rotation.T
    origins_sensor = ray_grid.origins_for(source)
    origins_world = origins_sensor @ rotation.T + lidar_origin_world
    native_range = np.asarray(ray_grid.ranges(source)).copy()
    native_range[np.asarray(source.zero_slot_mask, dtype=np.bool_)] = np.inf
    return (
        directions_sensor,
        directions_world,
        origins_sensor,
        origins_world,
        native_range,
    )


def render_frame(
    source: SourceFrame,
    world: WorldSpec,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    _trace_context: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
    _competition: _ObjectCompetition | None = None,
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
    (
        directions_sensor,
        directions_world,
        origins_sensor,
        origins_world,
        normal_range,
    ) = (
        _frame_trace_context(source, ray_grid)
        if _trace_context is None
        else _trace_context
    )
    competition = (
        _accepted_object_hits(
            origins_world,
            directions_world,
            world,
            ray_grid,
            sensor,
            int(source.frame_id),
        )
        if _competition is None
        else _competition
    )
    inserted = np.isfinite(competition.distance_m) & (
        competition.distance_m < normal_range - world.tie_tolerance_m
    )
    slots = np.flatnonzero(inserted).astype(np.int32)
    object_by_id = {item.object_id: item for item in world.objects}
    anomaly_proxy = inserted.copy()
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
            semantic[slot] = np.uint16(2)
            if semantic_target is not None:
                semantic_target[slot] = np.uint8(255)
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
            packed[slot] = np.uint32(2) | np.uint32(
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


def source_observation_identity(source: SourceFrame) -> str:
    """Bind one rendered scan identity to its actual geometry, pose, and labels."""

    if not isinstance(source, SourceFrame):
        raise TypeError("source observation identity requires a SourceFrame")
    digest = hashlib.sha256(b"AJAE-schema33-rendered-source-observation\0")
    digest.update(
        json.dumps(
            {
                "partition": source.partition,
                "sequence_id": source.sequence_id,
                "frame_id": source.frame_id,
                "labels_available": source.labels is not None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    arrays: list[tuple[str, np.ndarray]] = [
        ("xyzi", source.xyzi),
        ("lidar_pose", source.lidar_pose),
        ("real_slots", source.real_slots),
    ]
    if source.labels is not None:
        arrays.append(("packed_labels", source.labels.packed))
    for name, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RenderedWindow:
    """Five consecutive views cut from one already-rendered physical sequence."""

    window_start: int
    frame_ids: tuple[int, int, int, int, int]
    rendered_frames: tuple[
        RenderedFrame, RenderedFrame, RenderedFrame, RenderedFrame, RenderedFrame
    ]

    def __post_init__(self) -> None:
        start = _integer("window_start", self.window_start)
        frame_ids = tuple(_integer("frame_id", value) for value in self.frame_ids)
        rendered = tuple(self.rendered_frames)
        if frame_ids != tuple(range(start, start + 5)):
            raise RenderError("RenderedWindow requires five consecutive frame IDs")
        if len(rendered) != 5 or tuple(item.frame_id for item in rendered) != frame_ids:
            raise RenderError("RenderedWindow frames do not match its identity")
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "rendered_frames", rendered)

    @property
    def source_observation_identities(self) -> tuple[str, str, str, str, str]:
        identities = tuple(
            source_observation_identity(item.source) for item in self.rendered_frames
        )
        return identities  # type: ignore[return-value]

    @property
    def identity(self) -> str:
        payload = {
            "format": "ajae-rendered-window-v1",
            "window_start": self.window_start,
            "frame_ids": self.frame_ids,
            "source_observation_identities": self.source_observation_identities,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentClipWorld:
    """One fixed anomaly world rendered once over a contiguous source segment."""

    clip_start: int
    frame_ids: tuple[int, ...]
    world: WorldSpec
    report: WorldGenerationReport
    renderer_identity: str
    windows: tuple[RenderedWindow, ...]

    def __post_init__(self) -> None:
        start = _integer("clip_start", self.clip_start)
        frame_ids = tuple(_integer("frame_id", value) for value in self.frame_ids)
        if len(frame_ids) < 9 or frame_ids != tuple(
            range(start, start + len(frame_ids))
        ):
            raise RenderError(
                "a development clip requires at least nine consecutive source scans"
            )
        windows = tuple(self.windows)
        if not windows:
            raise RenderError("a development clip cannot omit its sliding windows")
        if (
            self.world.source_sequence_id != 201
            or self.report.source_sequence_id != 201
            or self.report.world_seed != self.world.seed
        ):
            raise RenderError("development world and report must identify train/201")
        expected_starts = tuple(range(start, frame_ids[-1] - 3))
        if tuple(item.window_start for item in windows) != expected_starts:
            raise RenderError("development clip must contain every sliding window")
        self.source_observation_identities
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "windows", windows)

    @property
    def source_observation_identities(self) -> tuple[str, ...]:
        """Prove overlapping windows reuse identical rendered physical frames."""

        by_frame: dict[int, str] = {}
        for window in self.windows:
            for rendered in window.rendered_frames:
                frame_id = int(rendered.frame_id)
                identity = source_observation_identity(rendered.source)
                previous = by_frame.setdefault(frame_id, identity)
                if previous != identity:
                    raise RenderError(
                        "one source frame changed across overlapping windows"
                    )
        if set(by_frame) != set(self.frame_ids):
            raise RenderError("development windows do not cover every clip frame")
        return tuple(by_frame[frame_id] for frame_id in self.frame_ids)

    @property
    def identity(self) -> str:
        payload = {
            "format": "ajae-development-clip-world-v2",
            "world_identity": self.world.identity,
            "clip_start": self.clip_start,
            "frame_ids": self.frame_ids,
            "renderer_identity": self.renderer_identity,
            "source_observation_identities": self.source_observation_identities,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_manifest(self) -> dict[str, object]:
        return {
            "format": "ajae-development-clip-world-v2",
            "identity": self.identity,
            "world_identity": self.world.identity,
            "clip_start": self.clip_start,
            "frame_ids": list(self.frame_ids),
            "renderer_identity": self.renderer_identity,
            "source_observation_identities": list(self.source_observation_identities),
            "world": self.world.to_dict(),
            "report": self.report.to_dict(),
            "windows": [
                {
                    "identity": item.identity,
                    "window_start": item.window_start,
                    "frame_ids": list(item.frame_ids),
                    "source_observation_identities": list(
                        item.source_observation_identities
                    ),
                }
                for item in self.windows
            ],
        }


def render_development_clip_world(
    world: WorldSpec,
    report: WorldGenerationReport,
    sources: Sequence[SourceFrame],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    renderer_identity: str,
) -> DevelopmentClipWorld:
    """Render a fixed world once, then cut every causal five-scan window."""

    frames = tuple(sorted(tuple(sources), key=lambda item: item.frame_id))
    frame_ids = tuple(item.frame_id for item in frames)
    if len(frames) < 9 or frame_ids != tuple(
        range(frame_ids[0], frame_ids[0] + len(frames))
    ):
        raise RenderError(
            "development clip sources must be at least nine consecutive scans"
        )
    if world.source_sequence_id != 201 or any(
        item.partition != "train" or item.sequence_id != 201 for item in frames
    ):
        raise RenderError("development clip sources must be identified train/201 scans")
    if world.world_type != "anomaly_only":
        raise RenderError("F3 development requires one anomaly-only world")
    frozen_world_identity = world.identity
    rendered = tuple(render_frames(frames, world, ray_grid, sensor))
    if world.identity != frozen_world_identity:
        raise RenderError("WorldSpec changed while the sequence was rendered")
    windows = tuple(
        RenderedWindow(
            window_start=frames[offset].frame_id,
            frame_ids=tuple(item.frame_id for item in frames[offset : offset + 5]),  # type: ignore[arg-type]
            rendered_frames=rendered[offset : offset + 5],  # type: ignore[arg-type]
        )
        for offset in range(len(frames) - 4)
    )
    return DevelopmentClipWorld(
        clip_start=frame_ids[0],
        frame_ids=frame_ids,
        world=world,
        report=report,
        renderer_identity=renderer_identity,
        windows=windows,
    )


def sample_development_clip_world(
    support_pool: QualifiedSupportPool,
    obstacles: ObservedObstacleIndex,
    sources: Sequence[SourceFrame],
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    seed: int,
    *,
    renderer_identity: str,
    maximum_attempts: int = 48,
) -> DevelopmentClipWorld:
    """Construct one fixed world; window visibility never changes its root seed."""

    frames = tuple(sorted(tuple(sources), key=lambda item: item.frame_id))
    if len(frames) < 9:
        raise RenderError("development generation requires at least nine scans")
    root_seed = _integer("seed", seed)
    world, report = sample_anomaly_world(
        support_pool,
        obstacles,
        root_seed,
        source_sequence_id=201,
        support_frame_ids=tuple(item.frame_id for item in frames),
        maximum_attempts=maximum_attempts,
    )
    return render_development_clip_world(
        world,
        report,
        frames,
        ray_grid,
        sensor,
        renderer_identity=renderer_identity,
    )


@dataclass(frozen=True, slots=True)
class SupportPatch:
    """One E21-v4-qualified support plane with its stable source identity."""

    pool_index: int
    semantic: int
    frame_id: int
    slot: int
    range_m: float
    selection_hash: int
    anchor_world_m: tuple[float, float, float]
    normal_world: tuple[float, float, float]
    offset: float


@dataclass(frozen=True, slots=True)
class QualifiedSupportPool:
    """Array-backed E21-v4 pool; failed candidates never enter placement."""

    pool_indices: np.ndarray
    semantics: np.ndarray
    frames: np.ndarray
    slots: np.ndarray
    ranges_m: np.ndarray
    selection_hashes: np.ndarray
    anchors_world_m: np.ndarray
    normals_world: np.ndarray
    offsets: np.ndarray
    source_sequence_id: int = 206

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value)
            for value in (
                self.pool_indices,
                self.semantics,
                self.frames,
                self.slots,
                self.ranges_m,
                self.selection_hashes,
                self.anchors_world_m,
                self.normals_world,
                self.offsets,
            )
        )
        count = arrays[0].shape[0]
        if count < 1 or any(item.shape[0] != count for item in arrays):
            raise PlacementError("qualified support-pool arrays are not aligned")
        if arrays[6].shape != (count, 3) or arrays[7].shape != (count, 3):
            raise PlacementError("qualified support coordinates must be [N,3]")
        if not np.isfinite(arrays[4]).all() or any(
            not np.isfinite(item).all() for item in arrays[6:]
        ):
            raise PlacementError("qualified support-pool geometry must be finite")
        lengths = np.linalg.norm(arrays[7], axis=1)
        if not np.allclose(lengths, 1.0, atol=1.0e-7, rtol=1.0e-7):
            raise PlacementError("qualified support normals must be unit vectors")
        if not np.isin(arrays[1], SUPPORT_POOL_SEMANTICS).all():
            raise PlacementError("qualified support-pool semantic is unsupported")
        sequence_id = _integer("source_sequence_id", self.source_sequence_id)
        if sequence_id not in SUPPORT_POOL_SHA256_BY_SEQUENCE:
            raise PlacementError("qualified support pool has an unsupported source")
        names = (
            "pool_indices",
            "semantics",
            "frames",
            "slots",
            "ranges_m",
            "selection_hashes",
            "anchors_world_m",
            "normals_world",
            "offsets",
        )
        for name, value in zip(names, arrays, strict=True):
            object.__setattr__(self, name, _freeze(value))

    def patch(self, row: int) -> SupportPatch:
        index = _integer("support row", int(row))
        if index >= self.pool_indices.shape[0]:
            raise PlacementError("support row lies outside the qualified pool")
        return SupportPatch(
            int(self.pool_indices[index]),
            int(self.semantics[index]),
            int(self.frames[index]),
            int(self.slots[index]),
            float(self.ranges_m[index]),
            int(self.selection_hashes[index]),
            tuple(map(float, self.anchors_world_m[index])),
            tuple(map(float, self.normals_world[index])),
            float(self.offsets[index]),
        )


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _scientific_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_qualified_support_pool(
    path: Path | str, *, source_sequence_id: int = 206
) -> QualifiedSupportPool:
    """Load one sequence-specific qualified pool after verifying its identity."""

    source = Path(path).expanduser().resolve(strict=True)
    sequence_id = _integer("source_sequence_id", source_sequence_id)
    expected_digest = SUPPORT_POOL_SHA256_BY_SEQUENCE.get(sequence_id)
    if expected_digest is None or _sha256_path(source) != expected_digest:
        raise PlacementError(
            "support-pool artifact does not match its frozen source sequence"
        )
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "semantic",
            "frame",
            "slot",
            "range_m",
            "selection_hash",
            "anchor_world",
            "normal",
            "offset",
            "metadata_json",
        }
        if set(payload.files) != required:
            raise PlacementError("support-pool artifact has unexpected arrays")
        arrays = {
            name: np.asarray(payload[name])
            for name in required - {"metadata_json"}
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
        expected = {
            201: (
                "schema33-development-support-pool",
                [4, 553],
                [6, 551],
                546,
            ),
            206: (
                "schema33-training-support-pool",
                [0, 448],
                [2, 446],
                445,
            ),
        }[sequence_id]
        if (
            metadata.get("experiment") != expected[0]
            or metadata.get("source_sequence") != f"train/{sequence_id}"
            or metadata.get("source_frames") != expected[1]
            or metadata.get("center_frames") != expected[2]
            or metadata.get("covered_center_frames") != expected[3]
            or metadata.get("passed") is not True
            or metadata.get("pool_size") != int(arrays["frame"].shape[0])
            or metadata.get("scientific_array_hash")
            != _scientific_array_hash(arrays)
            or int(np.min(arrays["frame"])) != expected[2][0]
            or int(np.max(arrays["frame"])) != expected[2][1]
        ):
            raise PlacementError("support-pool metadata is not schema-33 qualified")
        return QualifiedSupportPool(
            np.arange(arrays["frame"].shape[0], dtype=np.int64),
            np.asarray(arrays["semantic"], dtype=np.uint16),
            np.asarray(arrays["frame"], dtype=np.int32),
            np.asarray(arrays["slot"], dtype=np.int32),
            np.asarray(arrays["range_m"], dtype=np.float64),
            np.asarray(arrays["selection_hash"], dtype=np.uint64),
            np.asarray(arrays["anchor_world"], dtype=np.float64),
            np.asarray(arrays["normal"], dtype=np.float64),
            np.asarray(arrays["offset"], dtype=np.float64),
            source_sequence_id=sequence_id,
        )


@dataclass(frozen=True, slots=True)
class ObservedObstacleIndex:
    """Observed non-ground returns from one labelled normal train sequence."""

    points_world_m: np.ndarray
    identities: np.ndarray
    tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world_m, dtype=np.float64)
        identities = np.asarray(self.identities, dtype=np.uint64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or identities.shape != (points.shape[0],)
        ):
            raise PlacementError(
                "observed obstacle coordinates and identities are invalid"
            )
        if points.shape[0] < 1 or not np.isfinite(points).all():
            raise PlacementError("observed obstacle index is empty or non-finite")
        points = _freeze(points)
        identities = _freeze(identities)
        object.__setattr__(self, "points_world_m", points)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "tree", cKDTree(points, compact_nodes=True))

    def within_aabb(
        self, lower_world_m: np.ndarray, upper_world_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(lower_world_m, dtype=np.float64)
        upper = np.asarray(upper_world_m, dtype=np.float64)
        center = 0.5 * (lower + upper)
        radius = 0.5 * float(np.linalg.norm(upper - lower))
        candidates = np.asarray(
            self.tree.query_ball_point(center, radius), dtype=np.int64
        )
        if candidates.size == 0:
            return self.points_world_m[:0], self.identities[:0]
        points = self.points_world_m[candidates]
        keep = np.all((points >= lower) & (points <= upper), axis=1)
        return points[keep], self.identities[candidates[keep]]


def collect_observed_obstacle_index(
    frames: Iterable[SourceFrame], *, source_sequence_id: int = 206
) -> ObservedObstacleIndex:
    """Index every nonzero, non-ground return from one identified normal sequence."""

    expected_sequence = _integer("source_sequence_id", source_sequence_id)
    point_chunks: list[np.ndarray] = []
    identity_chunks: list[np.ndarray] = []
    for frame in frames:
        if (
            frame.partition != "train"
            or frame.sequence_id != expected_sequence
            or frame.labels is None
        ):
            raise PlacementError(
                "observed obstacles must come from one labelled normal train sequence"
            )
        real = np.asarray(frame.real_slots, dtype=np.int64)
        semantic = np.asarray(frame.labels.semantic[real], dtype=np.uint16)
        selected = (semantic != 0) & ~np.isin(semantic, GROUND_SEMANTIC_IDS)
        slots = real[selected]
        rotation, translation = _pose(frame)
        sensor = np.asarray(frame.xyzi[slots, :3], dtype=np.float64)
        point_chunks.append(sensor @ rotation.T + translation)
        identity_chunks.append(
            (np.uint64(frame.frame_id) << np.uint64(32)) | slots.astype(np.uint64)
        )
    if not point_chunks:
        raise PlacementError(
            "normal train sequence contains no observed obstacle returns"
        )
    return ObservedObstacleIndex(
        np.concatenate(point_chunks, axis=0), np.concatenate(identity_chunks, axis=0)
    )


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


def _identity_order(
    pool: QualifiedSupportPool,
    allowed_semantics: Sequence[int],
    namespace: str,
    stream_id: int,
) -> np.ndarray:
    """Hash stable support identities; no geometric result enters proposal order."""

    allowed = np.asarray(tuple(sorted(map(int, allowed_semantics))), dtype=np.uint16)
    rows = np.flatnonzero(np.isin(pool.semantics, allowed))
    if rows.size == 0:
        raise PlacementError("qualified pool has no semantically legal support")
    prefix = namespace.encode("ascii") + int(stream_id).to_bytes(8, "little")
    keys = np.empty(rows.size, dtype="S32")
    for index, row in enumerate(rows):
        identity = int(pool.frames[row]).to_bytes(4, "little", signed=False) + int(
            pool.slots[row]
        ).to_bytes(4, "little", signed=False)
        keys[index] = hashlib.sha256(prefix + identity).digest()
    return rows[np.argsort(keys, kind="stable")]


def _shape_outer_bounds(shape: InsertShape) -> tuple[np.ndarray, np.ndarray]:
    return shape.tight_continuous_outer_bounds(z_slabs=256, safety_margin_m=1.0e-6)


def _world_aabb(
    shape: InsertShape,
    rotation: np.ndarray,
    translation: np.ndarray,
    margin_m: float,
    *,
    local_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _shape_outer_bounds(shape) if local_bounds is None else local_bounds
    corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    world = corners @ rotation.T + translation
    return world.min(axis=0) - margin_m, world.max(axis=0) + margin_m


def _grounded_object(
    shape: InsertShape,
    material: MaterialSpec,
    patch: SupportPatch,
    *,
    object_id: int,
    label: ObjectLabel,
    yaw_rad: float,
    shape_generation_report: ShapeGenerationReport | None,
    lower_support_m: float | None = None,
) -> ObjectSpec:
    normal = np.asarray(patch.normal_world, dtype=np.float64)
    anchor = np.asarray(patch.anchor_world_m, dtype=np.float64)
    contact = anchor.copy()
    contact[2] = (
        -(normal[0] * contact[0] + normal[1] * contact[1] + patch.offset) / normal[2]
    )
    rotation = _ground_rotation(normal, yaw_rad)
    lower_support = (
        shape.minimum_z_m(xy_resolution=33, z_steps=129)
        if lower_support_m is None
        else _finite_scalar("lower_support_m", lower_support_m)
    )
    translation = contact - normal * lower_support
    return ObjectSpec(
        object_id=object_id,
        label=label,
        shape=shape,
        material=material,
        translation_world_m=tuple(map(float, translation)),
        rotation_world_from_local=tuple(tuple(map(float, row)) for row in rotation),
        shape_generation_report=shape_generation_report,
    )


def observed_normal_collision(
    proposed: ObjectSpec,
    obstacles: ObservedObstacleIndex,
    *,
    penetration_m: float = 0.05,
    local_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[bool, float, np.ndarray]:
    """Reject iff an actually observed return lies more than 5 cm inside."""

    threshold = _finite_scalar("penetration_m", penetration_m)
    rotation = np.asarray(proposed.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(proposed.translation_world_m, dtype=np.float64)
    lower, upper = _world_aabb(
        proposed.shape,
        rotation,
        translation,
        threshold,
        local_bounds=local_bounds,
    )
    points, identities = obstacles.within_aabb(lower, upper)
    if points.size == 0:
        return False, math.inf, identities
    local = (points - translation) @ rotation
    distance = proposed.shape.signed_distance(local)
    minimum = float(np.min(distance))
    deep = distance < -threshold
    return bool(np.any(deep)), minimum, identities


def _fibonacci_surface_points(shape: InsertShape, count: int = 8192) -> np.ndarray:
    identifiers = np.arange(_integer("surface point count", count, minimum=32))
    z = 1.0 - 2.0 * (identifiers + 0.5) / count
    angle = math.pi * (3.0 - math.sqrt(5.0)) * identifiers
    radial = np.sqrt(np.maximum(0.0, 1.0 - np.square(z)))
    direction = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))
    center = np.zeros(3, dtype=np.float64)
    radius = shape.bound_radius_m
    origins = center + 1.05 * radius * direction
    distance, _, valid = shape.intersect(origins, -direction)
    if not bool(np.all(valid)):
        raise PlacementError("deterministic surface ray missed the inserted geometry")
    points = origins - np.asarray(distance)[:, None] * direction
    residual = np.abs(shape.signed_distance(points))
    if not np.isfinite(points).all() or float(np.max(residual)) > 1.0e-5:
        raise PlacementError("deterministic surface points are not on the geometry")
    return points


def _forward_deform(shape: ShapeSpec, points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    z = result[:, 2]
    angle = shape.twist_rad_per_m * z
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = result[:, 0].copy(), result[:, 1].copy()
    rotated_x = cosine * x - sine * y
    rotated_y = sine * x + cosine * y
    scale_z = max(item[2] for item in shape.primitive_scales_m)
    factor_x = np.clip(1.0 + shape.taper_per_m[0] * z / scale_z, 0.25, 4.0)
    factor_y = np.clip(1.0 + shape.taper_per_m[1] * z / scale_z, 0.25, 4.0)
    result[:, 0] = factor_x * rotated_x + shape.bend_per_m[0] * np.square(z)
    result[:, 1] = factor_y * rotated_y + shape.bend_per_m[1] * np.square(z)
    return result


def _pair_local_witnesses(item: ObjectSpec) -> np.ndarray:
    shape = item.shape
    chunks = [_fibonacci_surface_points(shape, 8192)]
    lower, upper = _shape_outer_bounds(shape)
    unit = qmc.Sobol(d=3, scramble=False).random_base2(13)
    probes = lower + unit * (upper - lower)
    interior = probes[shape.signed_distance(probes) < 0.0]
    if interior.size:
        chunks.append(interior)
    undeformed = list(shape.primitive_offsets_m)
    report = item.shape_generation_report
    if report is not None:
        undeformed.extend(report.shared_witnesses_undeformed_m)
    chunks.append(_forward_deform(shape, np.asarray(undeformed, dtype=np.float64)))
    return np.concatenate(chunks, axis=0)


def _pair_witnesses(
    item: ObjectSpec, local_cache: dict[int, np.ndarray] | None = None
) -> np.ndarray:
    key = id(item.shape)
    if local_cache is None or key not in local_cache:
        local = _pair_local_witnesses(item)
        if local_cache is not None:
            local_cache[key] = local
    else:
        local = local_cache[key]
    rotation = np.asarray(item.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(item.translation_world_m, dtype=np.float64)
    return local @ rotation.T + translation


def obvious_pair_penetration(
    left: ObjectSpec,
    right: ObjectSpec,
    *,
    penetration_m: float = 0.05,
    witness_cache: dict[int, np.ndarray] | None = None,
) -> tuple[bool, float]:
    """Use AABB only as a broad phase, then test bidirectional real witnesses."""

    threshold = _finite_scalar("penetration_m", penetration_m)
    left_rotation = np.asarray(left.rotation_world_from_local, dtype=np.float64)
    right_rotation = np.asarray(right.rotation_world_from_local, dtype=np.float64)
    left_translation = np.asarray(left.translation_world_m, dtype=np.float64)
    right_translation = np.asarray(right.translation_world_m, dtype=np.float64)
    left_lower, left_upper = _world_aabb(
        left.shape, left_rotation, left_translation, threshold
    )
    right_lower, right_upper = _world_aabb(
        right.shape, right_rotation, right_translation, threshold
    )
    if np.any(left_upper < right_lower) or np.any(right_upper < left_lower):
        return False, math.inf
    minimum = math.inf
    for source, target, rotation, translation in (
        (left, right, right_rotation, right_translation),
        (right, left, left_rotation, left_translation),
    ):
        points = _pair_witnesses(source, witness_cache)
        local = (points - translation) @ rotation
        distance = target.shape.signed_distance(local)
        minimum = min(minimum, float(np.min(distance)))
        if bool(np.any(distance < -threshold)):
            return True, minimum
    return False, minimum


@dataclass(frozen=True, slots=True)
class GroundingEligibility:
    """Cache the support-invariant E22-v2 grounding qualification for one shape."""

    shape: InsertShape
    standard_lower_support_m: float
    strict_lower_support_m: float
    buried_fraction: float
    surface_points_local_m: np.ndarray = field(repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return (
            abs(self.strict_lower_support_m - self.standard_lower_support_m) <= 0.01
            and self.buried_fraction <= 0.02
        )


def qualify_grounding(shape: InsertShape) -> GroundingEligibility:
    """Evaluate the frozen E22-v2 conditions before support placement."""

    strict = float(shape.minimum_z_m(xy_resolution=65, z_steps=257))
    standard = float(shape.minimum_z_m(xy_resolution=33, z_steps=129))
    surface = _fibonacci_surface_points(shape, 16384)
    buried_fraction = float(np.mean(surface[:, 2] - standard < -0.02))
    return GroundingEligibility(
        shape, standard, strict, buried_fraction, _freeze(surface)
    )


def _grounding_qualified_shape(
    first_seed: int,
    *,
    stride: int,
    maximum_proposals: int = 64,
) -> tuple[
    ShapeSpec,
    ShapeGenerationReport,
    GroundingEligibility,
    tuple[int, ...],
    tuple[int, ...],
]:
    """Take the first E22-qualified shape from one deterministic seed stream."""

    start = _integer("first_seed", first_seed)
    step = _integer("stride", stride, minimum=1)
    limit = _integer("maximum_proposals", maximum_proposals, minimum=1)
    if limit > 64:
        raise PlacementError("shape proposal limit must not exceed 64")
    proposed: list[int] = []
    rejected: list[int] = []
    for proposal in range(limit):
        shape_seed = start + step * proposal
        shape, report = ShapeSpec.sample_with_report(shape_seed)
        grounding = qualify_grounding(shape)
        proposed.append(shape_seed)
        if grounding.passed:
            return shape, report, grounding, tuple(proposed), tuple(rejected)
        rejected.append(shape_seed)
    raise PlacementError("no E22-qualified shape in 64 deterministic proposals")


def place_object(
    shape: InsertShape,
    material: MaterialSpec,
    support_pool: QualifiedSupportPool,
    obstacles: ObservedObstacleIndex,
    *,
    object_id: int,
    label: ObjectLabel,
    proposal_namespace: str,
    proposal_stream: int,
    yaw_rad: float,
    material_seed: int,
    yaw_seed: int,
    shape_seed: int,
    shape_generation_report: ShapeGenerationReport | None = None,
    existing_objects: Sequence[ObjectSpec] = (),
    allowed_support_semantics: Sequence[int] | None = None,
    proposal_rows: Sequence[int] | None = None,
    maximum_candidates: int = 128,
    grounding_eligibility: GroundingEligibility | None = None,
    yaw_for_support: Callable[[SupportPatch], float] | None = None,
    post_placement_rejection: (
        Callable[[ObjectSpec, SupportPatch], str | None] | None
    ) = None,
) -> tuple[ObjectSpec, PlacementRecord]:
    """The sole support-pool-only E22/E23/E24/E25 placement pipeline."""

    if (
        not isinstance(material, MaterialSpec)
        or not isinstance(support_pool, QualifiedSupportPool)
        or not isinstance(obstacles, ObservedObstacleIndex)
    ):
        raise TypeError("placement inputs have unsupported types")
    label_value = str(label)
    if label_value != "anomaly-proxy":
        raise PlacementError("schema 33 placement only supports anomaly-proxy")
    allowed = frozenset(SUPPORT_POOL_SEMANTICS)
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 128:
        raise PlacementError("maximum_candidates must lie in [1,128]")
    if allowed_support_semantics is not None:
        requested = frozenset(map(int, allowed_support_semantics))
        if not requested or not requested.issubset(allowed):
            raise PlacementError("requested supports violate the label semantic policy")
        allowed = requested
    if proposal_rows is None:
        order = _identity_order(
            support_pool, allowed, proposal_namespace, proposal_stream
        )
    else:
        order = np.asarray(tuple(map(int, proposal_rows)), dtype=np.int64)
        if (
            order.ndim != 1
            or order.size < 1
            or np.any((order < 0) | (order >= support_pool.pool_indices.shape[0]))
        ):
            raise PlacementError("proposal_rows contains an invalid support row")
        if not np.isin(support_pool.semantics[order], tuple(allowed)).all():
            raise PlacementError("proposal_rows violates the support semantic policy")
    grounding = (
        qualify_grounding(shape)
        if grounding_eligibility is None
        else grounding_eligibility
    )
    if not isinstance(grounding, GroundingEligibility) or grounding.shape is not shape:
        raise PlacementError("grounding eligibility belongs to a different shape")
    if not grounding.passed:
        raise PlacementError("shape fails E22 grounding eligibility")
    rejections: list[str] = []
    proposal_pool_indices: list[int] = []
    proposal_minimum_sdf: list[float] = []
    pair_witness_cache: dict[int, np.ndarray] = {}
    local_bounds = _shape_outer_bounds(shape)
    for proposal, row in enumerate(order[:maximum_candidates]):
        patch = support_pool.patch(int(row))
        proposal_pool_indices.append(patch.pool_index)
        proposal_yaw = yaw_rad if yaw_for_support is None else yaw_for_support(patch)
        proposal_yaw = _finite_scalar("proposal_yaw", proposal_yaw)
        proposed = _grounded_object(
            shape,
            material,
            patch,
            object_id=object_id,
            label=label_value,  # type: ignore[arg-type]
            yaw_rad=proposal_yaw,
            shape_generation_report=shape_generation_report,
            lower_support_m=grounding.standard_lower_support_m,
        )
        collision, minimum_sdf, _ = observed_normal_collision(
            proposed, obstacles, local_bounds=local_bounds
        )
        proposal_minimum_sdf.append(minimum_sdf)
        if collision:
            rejections.append("observed_normal_deep_penetration")
            continue
        pair_collision = False
        for other in existing_objects:
            if obvious_pair_penetration(
                proposed, other, witness_cache=pair_witness_cache
            )[0]:
                pair_collision = True
                break
        if pair_collision:
            rejections.append("obvious_pair_penetration")
            continue
        if post_placement_rejection is not None:
            rejection = post_placement_rejection(proposed, patch)
            if rejection is not None:
                if not isinstance(rejection, str) or not rejection:
                    raise PlacementError(
                        "post-placement rejection must be a nonempty string"
                    )
                rejections.append(rejection)
                continue
        record = PlacementRecord(
            object_id,
            label_value,
            _integer("shape_seed", shape_seed),
            _integer("material_seed", material_seed),
            _integer("yaw_seed", yaw_seed),
            proposal,
            patch.pool_index,
            patch.frame_id,
            patch.slot,
            patch.semantic,
            tuple(proposal_pool_indices),
            tuple(rejections),
            tuple(proposal_minimum_sdf),
            minimum_sdf,
            0,
            (shape_seed,),
            (),
        )
        return proposed, replace(
            record,
            grounding_standard_lower_support_m=grounding.standard_lower_support_m,
            grounding_strict_lower_support_m=grounding.strict_lower_support_m,
            grounding_buried_fraction=grounding.buried_fraction,
        )
    raise PlacementExhaustion(proposal_pool_indices, rejections, proposal_minimum_sdf)


def _anomaly_entity_count(seed: int) -> int:
    """Draw one to nine proxies, with 90% probability on one to three."""

    rng = np.random.default_rng(_integer("seed", seed))
    values = np.arange(1, 10)
    probability = np.asarray((0.36, 0.32, 0.22, 0.03, 0.02, 0.015, 0.01, 0.01, 0.015))
    return int(rng.choice(values, p=probability))


def sample_anomaly_world(
    support_pool: QualifiedSupportPool,
    obstacles: ObservedObstacleIndex,
    seed: int,
    *,
    source_sequence_id: int,
    support_frame_ids: Sequence[int] | None = None,
    maximum_attempts: int = 48,
) -> tuple[WorldSpec, WorldGenerationReport]:
    """Build one immutable anomaly-only world before rendering any scan."""

    world_seed = _integer("seed", seed)
    if not isinstance(support_pool, QualifiedSupportPool) or not isinstance(
        obstacles, ObservedObstacleIndex
    ):
        raise TypeError("anomaly world requires the qualified pool and obstacle index")
    sequence_id = _integer("source_sequence_id", source_sequence_id)
    if support_pool.source_sequence_id != sequence_id:
        raise PlacementError(
            "support-pool source sequence differs from the sampled world"
        )
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise RenderError("maximum_attempts must be positive")
    allowed_frames: frozenset[int] | None = None
    if support_frame_ids is not None:
        members = tuple(_integer("support frame", value) for value in support_frame_ids)
        if not members:
            raise RenderError("support_frame_ids cannot be empty")
        allowed_frames = frozenset(members)
    anomaly_count = _anomaly_entity_count(world_seed)

    for attempt in range(maximum_attempts):
        # This is the only retry stream authorized by the schema-33 protocol.
        attempt_seed = world_seed + 1_000_003 * attempt
        objects: list[ObjectSpec] = []
        records: list[PlacementRecord] = []
        try:
            for entity_index in range(anomaly_count):
                entity_seed = attempt_seed + 10_007 * (entity_index + 1)
                (
                    shape,
                    report,
                    grounding,
                    shape_proposals,
                    grounding_rejections,
                ) = _grounding_qualified_shape(
                    entity_seed + 3, stride=3072, maximum_proposals=64
                )
                eligible_rows = np.arange(
                    support_pool.pool_indices.shape[0], dtype=np.int64
                )
                if allowed_frames is not None:
                    eligible_rows = eligible_rows[
                        np.isin(
                            support_pool.frames[eligible_rows], tuple(allowed_frames)
                        )
                    ]
                if eligible_rows.size == 0:
                    raise PlacementError(
                        "the selected sequence has no legal support patch"
                    )
                material_seed = entity_seed + 11
                yaw_seed = entity_seed + 31
                yaw = float(np.random.default_rng(yaw_seed).uniform(-math.pi, math.pi))
                item, record = place_object(
                    shape,
                    MaterialSpec.sample(material_seed),
                    support_pool,
                    obstacles,
                    object_id=entity_index + 1,
                    label="anomaly-proxy",
                    proposal_namespace="schema33-sequence-world-v2",
                    proposal_stream=entity_seed,
                    yaw_rad=yaw,
                    material_seed=material_seed,
                    yaw_seed=yaw_seed,
                    shape_seed=shape_proposals[-1],
                    shape_generation_report=report,
                    existing_objects=objects,
                    grounding_eligibility=grounding,
                    proposal_rows=eligible_rows,
                )
                records.append(
                    replace(
                        record,
                        accepted_shape_proposal=len(shape_proposals) - 1,
                        shape_proposal_seeds=shape_proposals,
                        grounding_rejection_seeds=grounding_rejections,
                    )
                )
                objects.append(item)
            world = WorldSpec(world_seed, sequence_id, tuple(objects))
        except PlacementError:
            continue
        return world, WorldGenerationReport(
            world_seed,
            sequence_id,
            "anomaly_only",
            attempt,
            anomaly_count,
            world_seed,
            attempt_seed,
            tuple(records),
        )
    raise PlacementError(
        f"anomaly world failed {maximum_attempts} deterministic placement attempts"
    )
