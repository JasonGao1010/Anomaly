#!/usr/bin/env python3
"""Deterministic world-level counterfactual rendering for AJAE.

The renderer preserves organized file slots for I/O.  A slot becomes a
canonical LiDAR ray only after the explicit RayGrid audit.  Inserted objects
live in world coordinates and compete with native returns by nearest distance.
"""

from __future__ import annotations

import json
import hashlib
import argparse
import math
import os
import time
import multiprocessing as mp
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq, differential_evolution
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import qmc

try:
    from .scene import PointLabels, SourceFrame, make_source_frame
except ImportError:  # Direct module execution and small isolated checks.
    from scene import PointLabels, SourceFrame, make_source_frame


LASER_BEAMS = 128
GROUND_SEMANTIC_IDS = (40, 44, 48, 49, 60)
WORLD_FORMAT = "ajae-world-v3"
WORLD_REPORT_FORMAT = "ajae-world-generation-report-v2"
SUPPORT_POOL_FORMAT = "ajae-qualified-support-pool-v1"
SUPPORT_POOL_SHA256 = (
    "0e6e7299157f5e9ced0716f6dd14881c66ba1bca0cc9c372550e56f426ea844d"
)
SUPPORT_POOL_SEMANTICS = (40, 48, 49)
CALIBRATION_FORMAT = "ajae-sensor-calibration-v4"
DEVELOPMENT_FORMAT = "ajae-development-worlds-v2"
DEVELOPMENT_PROTOCOL_SCHEMA = 30
PROCEDURAL_GENERATOR_SCHEMA = 7
SHAPE_FAMILIES = ("general", "blocky", "flat", "elongated")
AXIS_PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
)
SCHEMA7_FAMILY_STREAM = 2001
SCHEMA7_RATIO_STREAM = 2002
SCHEMA7_AXIS_STREAM = 2003
SCHEMA7_PARENT_TAU_STREAM = 3001
SCHEMA7_CHILD_TAU_STREAM = 3002
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
VEHICLE_SUPPORT_SEMANTICS = frozenset((40,))
PERSON_RIDER_SUPPORT_SEMANTICS = frozenset((40, 48))
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
    return _interval_outward(
        minimum, np.maximum(lower * lower, upper * upper)
    )


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
    contains_maximum = np.ceil(
        (lower - maximum_phase) / (2.0 * math.pi)
    ) <= np.floor((upper - maximum_phase) / (2.0 * math.pi))
    contains_minimum = np.ceil(
        (lower - minimum_phase) / (2.0 * math.pi)
    ) <= np.floor((upper - minimum_phase) / (2.0 * math.pi))
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
        probe_count = 2048
        probe_id = np.arange(probe_count, dtype=np.float64)
        probe_z = 1.0 - 2.0 * (probe_id + 0.5) / probe_count
        probe_longitude = (
            math.pi * (3.0 - math.sqrt(5.0)) * probe_id + math.pi
        ) % (2.0 * math.pi) - math.pi
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
                    lambda value: -sign
                    * self._single_primitive_surface_point(value[0], value[1])[axis],
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
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
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
        old_lower, old_upper = self._continuous_outer_bounds(
            safety_margin_m=margin
        )
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
            active = (z_upper >= primitive_z_lower) & (
                z_lower <= primitive_z_upper
            )
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
                level ** (2.0 / vertical)
                - (nearest_z / c) ** (2.0 / vertical),
            )
            cross_scale = radial_term ** (vertical / 2.0)
            cosine = abs(math.cos(yaw))
            sine = abs(math.sin(yaw))
            planar_power = 2.0 / horizontal
            if planar_power > 1.0 + 1.0e-12:
                dual = planar_power / (planar_power - 1.0)
                half_x = cross_scale * (
                    (a * cosine) ** dual + (b * sine) ** dual
                ) ** (1.0 / dual)
                half_y = cross_scale * (
                    (a * sine) ** dual + (b * cosine) ** dual
                ) ** (1.0 / dual)
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
        sine_lower, sine_upper = trig_interval(
            angle_lower, angle_upper, cosine=False
        )
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
        lower, upper = self._continuous_outer_bounds(
            safety_margin_m=safety_margin_m
        )
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
                    raise RenderError("continuous outer bound did not bracket the geometry")

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
                    points = start[None, :] + weights[:, None] * (
                        end - start
                    )[None, :]
                    overlap = bool(
                        np.any(
                            (self._primitive_perturbed_value(left, points) < 0.0)
                            & (
                                self._primitive_perturbed_value(right, points)
                                < 0.0
                            )
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
                operation in {"union", "intersection"}
                for operation in self.operations
            )
            and all(
                operation == "intersection" for operation in self.operations[1:]
            )
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
                    self._primitive_perturbed_value(index, point[None, :])[0]
                    < 0.0
                    for index in range(self.primitive_count)
                )
                for point in candidates
            ):
                return "nonempty_convex_intersection"
        if (
            self.surface_amplitude_m == 0.0
            and self.primitive_count == 2
            and self.operations == ("union", "difference")
            and all(
                exponent == (1.0, 1.0)
                for exponent in self.primitive_exponents
            )
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
        x_lower, x_upper = _interval_add(
            x_lower, x_upper, bend_x_lower, bend_x_upper
        )
        y_lower, y_upper = _interval_add(
            y_lower, y_upper, bend_y_lower, bend_y_upper
        )
        scale_z = max(item[2] for item in self.primitive_scales_m)
        factor_x_lower, factor_x_upper = _interval_add(
            np.ones_like(z_lower),
            np.ones_like(z_upper),
            *_interval_scale(
                z_lower, z_upper, self.taper_per_m[0] / scale_z
            ),
        )
        factor_y_lower, factor_y_upper = _interval_add(
            np.ones_like(z_lower),
            np.ones_like(z_upper),
            *_interval_scale(
                z_lower, z_upper, self.taper_per_m[1] / scale_z
            ),
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
        sine_lower, sine_upper = _interval_trigonometric(
            angle_lower, angle_upper
        )
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
            sine_lower, sine_upper = _interval_trigonometric(
                phase_lower, phase_upper
            )
            displacement_lower += sine_lower
            displacement_upper += sine_upper
        displacement_lower *= self.surface_amplitude_m / 3.0
        displacement_upper *= self.surface_amplitude_m / 3.0
        return _interval_outward(
            result_lower - displacement_upper,
            result_upper - displacement_lower,
        )

    def _interval_connectivity_stats(
        self, cells: int
    ) -> tuple[int, int, int]:
        lower, upper = self._continuous_outer_bounds(safety_margin_m=1.0e-6)
        edges = [
            np.linspace(lower[axis], upper[axis], cells + 1)
            for axis in range(3)
        ]
        state = np.empty((cells, cells, cells), dtype=np.int8)
        total = cells**3
        batch = 131_072
        for start in range(0, total, batch):
            flat = np.arange(start, min(start + batch, total), dtype=np.int64)
            x = flat // (cells * cells)
            y = (flat // cells) % cells
            z = flat % cells
            box_lower = np.column_stack(
                (edges[0][x], edges[1][y], edges[2][z])
            )
            box_upper = np.column_stack(
                (edges[0][x + 1], edges[1][y + 1], edges[2][z + 1])
            )
            value_lower, value_upper = self._implicit_interval(
                box_lower, box_upper
            )
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
        return int(len(witnessed)), int(definite_count), int(
            possible_count - len(witnessed)
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
                    child_lo = np.concatenate(
                        (interval_lo[outside], middle[outside])
                    )
                    child_hi = np.concatenate(
                        (middle[outside], interval_hi[outside])
                    )
                    child_value_lo = np.concatenate(
                        (value_lo[outside], value_middle[outside])
                    )
                    child_value_hi = np.concatenate(
                        (value_middle[outside], value_hi[outside])
                    )
                    child_width = child_hi - child_lo
                    keep = (
                        (np.minimum(child_value_lo, child_value_hi) <= 4.0 * child_width)
                        & (child_lo < bracket_lo[child_ray])
                    )
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
    def _schema7_rng(
        seed: int, stream: int, *coordinates: int
    ) -> np.random.Generator:
        """Keep qualified schema-7 factors on structurally separate streams."""
        return np.random.default_rng(
            np.random.SeedSequence((seed, stream, *coordinates))
        )

    @classmethod
    def _schema7_base_scale(
        cls, seed: int, half: float
    ) -> tuple[tuple[float, float, float], str]:
        family_value = float(
            cls._schema7_rng(seed, SCHEMA7_FAMILY_STREAM).random()
        )
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
        return tuple(float(value) for value in ordered[list(permutation)]), SHAPE_FAMILIES[family]

    @classmethod
    def _perturbed_primitive_value(
        cls,
        scale: tuple[float, float, float], center: np.ndarray,
        exponent: tuple[float, float], yaw: float, point: np.ndarray,
        amplitude: float, frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> float:
        base = float(cls._primitive_distance(
            point[None], scale, tuple(center), exponent, yaw
        )[0])
        displacement = amplitude * float(np.mean(
            np.sin(point * np.asarray(frequency) + np.asarray(phase))
        ))
        return base - displacement

    @classmethod
    def _primitive_radial_radius(
        cls,
        scale: tuple[float, float, float], center: np.ndarray,
        exponent: tuple[float, float], yaw: float, direction: np.ndarray,
        amplitude: float, frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> float:
        def implicit(distance: float) -> float:
            return cls._perturbed_primitive_value(
                scale, center, exponent, yaw, center + distance * direction,
                amplitude, frequency, phase,
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
        parent_scale: tuple[float, float, float], parent_center: np.ndarray,
        parent_exponent: tuple[float, float], parent_yaw: float,
        child_scale: tuple[float, float, float], child_exponent: tuple[float, float],
        child_yaw: float, direction: np.ndarray, tau_parent: float, tau_child: float,
        amplitude: float, frequency: tuple[float, float, float],
        phase: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Construct one authoritative witness before global deformation."""
        parent_radius = cls._primitive_radial_radius(
            parent_scale, parent_center, parent_exponent, parent_yaw, direction,
            amplitude, frequency, phase,
        )
        witness = parent_center + tau_parent * parent_radius * direction

        # Translation changes the global-coordinate surface phase, so solve
        # placement and the opposite-direction child boundary together.
        def child_boundary(offset_distance: float) -> float:
            child_center = witness + offset_distance * direction
            boundary = witness - offset_distance * (1.0 / tau_child - 1.0) * direction
            return cls._perturbed_primitive_value(
                child_scale, child_center, child_exponent, child_yaw, boundary,
                amplitude, frequency, phase,
            )

        upper = 2.0 * float(np.linalg.norm(child_scale))
        while child_boundary(upper) <= 0.0 and upper < 64.0:
            upper *= 2.0
        if child_boundary(0.0) >= 0.0 or child_boundary(upper) <= 0.0:
            raise RenderError("schema-7 child boundary was not bracketed")
        offset_distance = float(brentq(
            child_boundary, 0.0, upper, xtol=1e-13, rtol=1e-13
        ))
        child_center = witness + offset_distance * direction
        child_radius = cls._primitive_radial_radius(
            child_scale, child_center, child_exponent, child_yaw, -direction,
            amplitude, frequency, phase,
        )
        if abs(offset_distance - tau_child * child_radius) > 1e-10:
            raise RenderError("schema-7 shared-witness formula is inconsistent")
        parent_margin = -cls._perturbed_primitive_value(
            parent_scale, parent_center, parent_exponent, parent_yaw, witness,
            amplitude, frequency, phase,
        )
        child_margin = -cls._perturbed_primitive_value(
            child_scale, child_center, child_exponent, child_yaw, witness,
            amplitude, frequency, phase,
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
                    tau_parent = float(cls._schema7_rng(
                        seed, SCHEMA7_PARENT_TAU_STREAM, proposal_count, child_index,
                    ).uniform(0.65, 0.85))
                    tau_child = float(cls._schema7_rng(
                        seed, SCHEMA7_CHILD_TAU_STREAM, proposal_count, child_index,
                    ).uniform(0.55, 0.80))
                    offset, witness, parent_margin, child_margin = (
                        cls._shared_witness_placement(
                            scales[parent], offsets[parent], exponents[parent], yaws[parent],
                            scales[child_index], exponents[child_index], yaws[child_index],
                            direction, tau_parent, tau_child, amplitude, frequency, phase,
                        )
                    )
                    offsets.append(offset)
                    child_parents.append(parent)
                    shared_witnesses.append(tuple(map(float, witness)))
                    parent_margins.append(parent_margin)
                    child_margins.append(child_margin)
                result = cls(
                    primitive_scales_m=tuple(scales),
                    primitive_offsets_m=tuple(tuple(map(float, item)) for item in offsets),
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
    minimum_points: int = 32,
    maximum_templates_per_class: int = 64,
) -> tuple[NormalTemplateShape, ...]:
    """Extract E25-observable templates, then select by stable instance identity."""

    if type(minimum_points) is not int or minimum_points < 4:
        raise RenderError("minimum_points must be an integer >=4")
    if type(maximum_templates_per_class) is not int or maximum_templates_per_class < 1:
        raise RenderError("maximum_templates_per_class must be positive")
    candidates: dict[int, list[tuple[bytes, NormalTemplateShape]]] = {
        semantic: [] for semantic in NORMAL_TEMPLATE_SEMANTICS
    }
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
                raw not in candidates
                or identifier == 0
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
            identity = f"206:{frame.frame_id}:{raw}:{identifier}".encode("ascii")
            candidates[raw].append((hashlib.sha256(identity).digest(), template))
    result: list[NormalTemplateShape] = []
    for semantic in sorted(candidates):
        ordered = sorted(candidates[semantic], key=lambda item: item[0])
        if len(ordered) >= 4:
            result.extend(item[1] for item in ordered[:maximum_templates_per_class])
    if len(result) < 32:
        raise PlacementError(
            "train/206 has fewer than 32 templates across E25 active classes"
        )
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
    shape_generation_report: ShapeGenerationReport | None = None

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

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlacementRecord:
    """Reproduce one accepted entity and every rejected support proposal."""

    object_id: int
    label: ObjectLabel
    shape_seed: int | None
    template_identity: str | None
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
    template_seed: int | None = None
    scale_seed: int | None = None
    pose_perturbation_rad: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "shape_seed": self.shape_seed,
            "template_identity": self.template_identity,
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
            "template_seed": self.template_seed,
            "scale_seed": self.scale_seed,
            "pose_perturbation_rad": self.pose_perturbation_rad,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PlacementRecord":
        plain = dict(value)
        for name in (
            "proposal_pool_indices", "rejection_reasons",
            "proposal_minimum_obstacle_sdf_m", "shape_proposal_seeds",
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
    normal_count: int
    anomaly_count: int
    count_seed: int
    label_order_seed: int
    placements: tuple[PlacementRecord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "format": WORLD_REPORT_FORMAT,
            "world_seed": self.world_seed,
            "source_sequence_id": self.source_sequence_id,
            "world_type": self.world_type,
            "world_attempt": self.world_attempt,
            "normal_count": self.normal_count,
            "anomaly_count": self.anomaly_count,
            "count_seed": self.count_seed,
            "label_order_seed": self.label_order_seed,
            "placements": [item.to_dict() for item in self.placements],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorldGenerationReport":
        if not isinstance(value, Mapping) or value.get("format") != WORLD_REPORT_FORMAT:
            raise RenderError("WorldGenerationReport JSON has an unsupported format")
        if set(value) != {
            "format", "world_seed", "source_sequence_id", "world_type",
            "world_attempt", "normal_count", "anomaly_count", "count_seed",
            "label_order_seed", "placements",
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
            normal_count=value["normal_count"],  # type: ignore[arg-type]
            anomaly_count=value["anomaly_count"],  # type: ignore[arg-type]
            count_seed=value["count_seed"],  # type: ignore[arg-type]
            label_order_seed=value["label_order_seed"],  # type: ignore[arg-type]
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

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.pool_indices, self.semantics, self.frames, self.slots,
            self.ranges_m, self.selection_hashes, self.anchors_world_m,
            self.normals_world, self.offsets,
        ))
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
        names = (
            "pool_indices", "semantics", "frames", "slots", "ranges_m",
            "selection_hashes", "anchors_world_m", "normals_world", "offsets",
        )
        for name, value in zip(names, arrays, strict=True):
            object.__setattr__(self, name, _freeze(value))

    def patch(self, row: int) -> SupportPatch:
        index = _integer("support row", int(row))
        if index >= self.pool_indices.shape[0]:
            raise PlacementError("support row lies outside the qualified pool")
        return SupportPatch(
            int(self.pool_indices[index]), int(self.semantics[index]),
            int(self.frames[index]), int(self.slots[index]),
            float(self.ranges_m[index]), int(self.selection_hashes[index]),
            tuple(map(float, self.anchors_world_m[index])),
            tuple(map(float, self.normals_world[index])), float(self.offsets[index]),
        )


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_qualified_support_pool(path: Path | str) -> QualifiedSupportPool:
    """Load only E21-v4 qualified rows after verifying the frozen artifact."""

    source = Path(path).expanduser().resolve(strict=True)
    if _sha256_path(source) != SUPPORT_POOL_SHA256:
        raise PlacementError("support-pool artifact does not match frozen E21-v4")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "semantic_index", "frame", "slot", "range_m", "selection_hash",
            "anchor_world", "qualified", "normals", "offsets", "metadata_json",
        }
        if not required.issubset(payload.files):
            raise PlacementError("support-pool artifact is missing required arrays")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("experiment") != "E21-v4" or not metadata.get(
            "elementwise_reproduced"
        ):
            raise PlacementError("support-pool metadata is not qualified E21-v4")
        qualified = np.asarray(payload["qualified"], dtype=np.bool_)
        rows = np.flatnonzero(qualified)
        semantic_index = np.asarray(payload["semantic_index"], dtype=np.int64)[rows]
        if np.any((semantic_index < 0) | (semantic_index >= len(SUPPORT_POOL_SEMANTICS))):
            raise PlacementError("support-pool semantic index is invalid")
        return QualifiedSupportPool(
            rows.astype(np.int64),
            np.asarray(SUPPORT_POOL_SEMANTICS, dtype=np.uint16)[semantic_index],
            np.asarray(payload["frame"], dtype=np.int32)[rows],
            np.asarray(payload["slot"], dtype=np.int32)[rows],
            np.asarray(payload["range_m"], dtype=np.float64)[rows],
            np.asarray(payload["selection_hash"], dtype=np.uint64)[rows],
            np.asarray(payload["anchor_world"], dtype=np.float64)[rows],
            np.asarray(payload["normals"], dtype=np.float64)[rows, 1],
            np.asarray(payload["offsets"], dtype=np.float64)[rows, 1],
        )


@dataclass(frozen=True, slots=True)
class ObservedObstacleIndex:
    """All actually observed non-ground train/206 returns in world coordinates."""

    points_world_m: np.ndarray
    identities: np.ndarray
    tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world_m, dtype=np.float64)
        identities = np.asarray(self.identities, dtype=np.uint64)
        if points.ndim != 2 or points.shape[1] != 3 or identities.shape != (points.shape[0],):
            raise PlacementError("observed obstacle coordinates and identities are invalid")
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
        candidates = np.asarray(self.tree.query_ball_point(center, radius), dtype=np.int64)
        if candidates.size == 0:
            return self.points_world_m[:0], self.identities[:0]
        points = self.points_world_m[candidates]
        keep = np.all((points >= lower) & (points <= upper), axis=1)
        return points[keep], self.identities[candidates[keep]]


def collect_observed_obstacle_index(frames: Iterable[SourceFrame]) -> ObservedObstacleIndex:
    """Index every nonzero, non-ground real return without spatial subsampling."""

    point_chunks: list[np.ndarray] = []
    identity_chunks: list[np.ndarray] = []
    for frame in frames:
        if frame.partition != "train" or frame.sequence_id != 206 or frame.labels is None:
            raise PlacementError("observed obstacles must come from labelled train/206")
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
        raise PlacementError("train/206 contains no observed obstacle returns")
    return ObservedObstacleIndex(
        np.concatenate(point_chunks, axis=0), np.concatenate(identity_chunks, axis=0)
    )


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


@dataclass(frozen=True, slots=True)
class SupportPlaneEstimate:
    """One deterministic trimmed-SVD plane at a fixed neighborhood radius."""

    radius_m: float
    support_count: int
    normal: np.ndarray
    offset: float
    anchor_height_m: float
    median_residual_m: float
    q95_residual_m: float
    rms_residual_m: float


@dataclass(frozen=True, slots=True)
class SupportPlaneQualification:
    """Three-scale support decision used by placement and E21."""

    estimates: tuple[SupportPlaneEstimate, ...]
    rejection_reason: str | None
    normal_angle_deg: float
    anchor_height_difference_m: float

    @property
    def qualified(self) -> bool:
        return self.rejection_reason is None


def _trimmed_support_plane(
    local_points_world: np.ndarray,
    anchor_world: np.ndarray,
    radius_m: float,
) -> SupportPlaneEstimate:
    """Fit one plane after removing the fixed largest-residual ten percent."""

    local = np.asarray(local_points_world, dtype=np.float64)
    anchor = np.asarray(anchor_world, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3 or local.shape[0] < 32:
        raise PlacementError("insufficient_support")
    try:
        center = local.mean(axis=0)
        covariance = (local - center).T @ (local - center) / float(local.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as error:
        raise PlacementError("degenerate_covariance") from error
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[-2]) <= 1.0e-12:
        raise PlacementError("degenerate_covariance")
    first_normal = eigenvectors[:, 0]
    first_residual = np.abs((local - center) @ first_normal)
    retain_count = int(math.ceil(0.90 * local.shape[0]))
    retained = local[np.argsort(first_residual, kind="stable")[:retain_count]]
    retained_center = retained.mean(axis=0)
    try:
        _, singular_values, vectors = np.linalg.svd(
            retained - retained_center, full_matrices=False
        )
    except np.linalg.LinAlgError as error:
        raise PlacementError("degenerate_covariance") from error
    if (
        singular_values.size < 2
        or not np.isfinite(singular_values).all()
        or float(np.square(singular_values[-2]) / retained.shape[0]) <= 1.0e-12
    ):
        raise PlacementError("degenerate_covariance")
    normal = vectors[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, retained_center))
    if (
        not np.isfinite(normal).all()
        or not math.isfinite(offset)
        or normal[2] <= np.finfo(np.float64).eps
    ):
        raise PlacementError("nonfinite_or_unsolved_plane")
    height = -(
        normal[0] * anchor[0] + normal[1] * anchor[1] + offset
    ) / normal[2]
    residual = np.abs(local @ normal + offset)
    if not math.isfinite(float(height)) or not np.isfinite(residual).all():
        raise PlacementError("nonfinite_or_unsolved_plane")
    return SupportPlaneEstimate(
        radius_m=float(radius_m),
        support_count=int(local.shape[0]),
        normal=_freeze(normal.astype(np.float64, copy=False)),
        offset=offset,
        anchor_height_m=float(height),
        median_residual_m=float(np.median(residual)),
        q95_residual_m=float(np.quantile(residual, 0.95)),
        rms_residual_m=float(np.sqrt(np.mean(np.square(residual)))),
    )


def qualify_support_plane(
    ground_points_world: np.ndarray,
    anchor_world: np.ndarray,
    *,
    radius_m: float = 1.0,
) -> SupportPlaneQualification:
    """Identify a low-residual support patch that is stable across three scales."""

    points = np.asarray(ground_points_world, dtype=np.float64)
    anchor = np.asarray(anchor_world, dtype=np.float64)
    radius = _finite_scalar("radius_m", radius_m)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PlacementError("ground_points_world must be finite [N,3]")
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise PlacementError("anchor_world must be finite [3]")
    if radius <= 0.0:
        raise PlacementError("support radius must be positive")
    estimates: list[SupportPlaneEstimate] = []
    for scale in (0.75, 1.0, 1.25):
        current_radius = scale * radius
        selected = np.linalg.norm(points[:, :2] - anchor[:2], axis=1) <= current_radius
        try:
            estimates.append(_trimmed_support_plane(points[selected], anchor, current_radius))
        except PlacementError as error:
            reason = str(error)
            if reason not in {
                "insufficient_support",
                "degenerate_covariance",
                "nonfinite_or_unsolved_plane",
            }:
                raise
            return SupportPlaneQualification(tuple(estimates), reason, math.nan, math.nan)
    small, middle, large = estimates
    cosine = float(np.clip(np.dot(small.normal, large.normal), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    height_difference = abs(small.anchor_height_m - large.anchor_height_m)
    reason = None
    if middle.q95_residual_m > 0.08:
        reason = "residual_q95"
    elif middle.median_residual_m > 0.03:
        reason = "residual_median"
    elif angle_deg > 5.0:
        reason = "multiscale_normal"
    elif height_difference > 0.08:
        reason = "multiscale_height"
    return SupportPlaneQualification(
        tuple(estimates), reason, float(angle_deg), float(height_difference)
    )


def fit_support_plane(
    ground_points_world: np.ndarray,
    anchor_world: np.ndarray,
    *,
    radius_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the qualified central support plane used by object placement."""

    qualification = qualify_support_plane(ground_points_world, anchor_world, radius_m=radius_m)
    if not qualification.qualified:
        raise PlacementError(str(qualification.rejection_reason))
    middle = qualification.estimates[1]
    contact = np.asarray(anchor_world, dtype=np.float64).copy()
    contact[2] = middle.anchor_height_m
    return _freeze(contact), middle.normal, middle.rms_residual_m


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
        identity = (
            int(pool.frames[row]).to_bytes(4, "little", signed=False)
            + int(pool.slots[row]).to_bytes(4, "little", signed=False)
        )
        keys[index] = hashlib.sha256(prefix + identity).digest()
    return rows[np.argsort(keys, kind="stable")]


def _shape_outer_bounds(shape: InsertShape) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(shape, ShapeSpec):
        return shape.tight_continuous_outer_bounds(
            z_slabs=256, safety_margin_m=1.0e-6
        )
    return shape.local_bounds()


def _world_aabb(
    shape: InsertShape, rotation: np.ndarray, translation: np.ndarray, margin_m: float
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _shape_outer_bounds(shape)
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
) -> ObjectSpec:
    normal = np.asarray(patch.normal_world, dtype=np.float64)
    anchor = np.asarray(patch.anchor_world_m, dtype=np.float64)
    contact = anchor.copy()
    contact[2] = -(
        normal[0] * contact[0] + normal[1] * contact[1] + patch.offset
    ) / normal[2]
    rotation = _ground_rotation(normal, yaw_rad)
    lower_support = shape.minimum_z_m(xy_resolution=33, z_steps=129)
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
) -> tuple[bool, float, np.ndarray]:
    """Reject iff an actually observed return lies more than 5 cm inside."""

    threshold = _finite_scalar("penetration_m", penetration_m)
    rotation = np.asarray(proposed.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(proposed.translation_world_m, dtype=np.float64)
    lower, upper = _world_aabb(proposed.shape, rotation, translation, threshold)
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
    if isinstance(shape, NormalTemplateShape):
        center = np.mean(shape.vertices_m, axis=0)
        radius = float(np.max(np.linalg.norm(shape.vertices_m - center, axis=1)))
    else:
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


def _pair_witnesses(item: ObjectSpec) -> np.ndarray:
    shape = item.shape
    chunks = [_fibonacci_surface_points(shape, 8192)]
    lower, upper = _shape_outer_bounds(shape)
    unit = qmc.Sobol(d=3, scramble=False).random_base2(13)
    probes = lower + unit * (upper - lower)
    interior = probes[shape.signed_distance(probes) < 0.0]
    if interior.size:
        chunks.append(interior)
    if isinstance(shape, ShapeSpec):
        undeformed = list(shape.primitive_offsets_m)
        report = item.shape_generation_report
        if report is not None:
            undeformed.extend(report.shared_witnesses_undeformed_m)
        chunks.append(_forward_deform(shape, np.asarray(undeformed, dtype=np.float64)))
    elif isinstance(shape, NormalTemplateShape):
        chunks.extend((shape.vertices_m, np.mean(shape.vertices_m[shape.faces], axis=1)))
    local = np.concatenate(chunks, axis=0)
    rotation = np.asarray(item.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(item.translation_world_m, dtype=np.float64)
    return local @ rotation.T + translation


def obvious_pair_penetration(
    left: ObjectSpec, right: ObjectSpec, *, penetration_m: float = 0.05
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
        points = _pair_witnesses(source)
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
    shape_seed: int | None = None,
    template_identity: str | None = None,
    shape_generation_report: ShapeGenerationReport | None = None,
    existing_objects: Sequence[ObjectSpec] = (),
    allowed_support_semantics: Sequence[int] | None = None,
    proposal_rows: Sequence[int] | None = None,
    maximum_candidates: int = 128,
    grounding_eligibility: GroundingEligibility | None = None,
    yaw_for_support: Callable[[SupportPatch], float] | None = None,
) -> tuple[ObjectSpec, PlacementRecord]:
    """The sole support-pool-only E22/E23/E24/E25 placement pipeline."""

    if not isinstance(material, MaterialSpec) or not isinstance(
        support_pool, QualifiedSupportPool
    ) or not isinstance(obstacles, ObservedObstacleIndex):
        raise TypeError("placement inputs have unsupported types")
    label_value = str(label)
    if label_value == "normal-control":
        if not isinstance(shape, NormalTemplateShape):
            raise PlacementError("normal-control placement requires a train/206 template")
        allowed = normal_control_support_semantics(shape.raw_semantic_id)
    elif label_value == "anomaly-proxy":
        allowed = frozenset(SUPPORT_POOL_SEMANTICS)
    else:
        raise PlacementError(f"label must be one of {OBJECT_LABELS}")
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
        if order.ndim != 1 or order.size < 1 or np.any(
            (order < 0) | (order >= support_pool.pool_indices.shape[0])
        ):
            raise PlacementError("proposal_rows contains an invalid support row")
        if not np.isin(support_pool.semantics[order], tuple(allowed)).all():
            raise PlacementError("proposal_rows violates the support semantic policy")
    grounding = (
        qualify_grounding(shape)
        if grounding_eligibility is None else grounding_eligibility
    )
    if not isinstance(grounding, GroundingEligibility) or grounding.shape is not shape:
        raise PlacementError("grounding eligibility belongs to a different shape")
    if not grounding.passed:
        raise PlacementError("shape fails E22 grounding eligibility")
    rejections: list[str] = []
    proposal_pool_indices: list[int] = []
    proposal_minimum_sdf: list[float] = []
    for proposal, row in enumerate(order[:maximum_candidates]):
        patch = support_pool.patch(int(row))
        proposal_pool_indices.append(patch.pool_index)
        proposal_yaw = (
            yaw_rad if yaw_for_support is None else yaw_for_support(patch)
        )
        proposal_yaw = _finite_scalar("proposal_yaw", proposal_yaw)
        proposed = _grounded_object(
            shape, material, patch, object_id=object_id, label=label_value,  # type: ignore[arg-type]
            yaw_rad=proposal_yaw, shape_generation_report=shape_generation_report,
        )
        collision, minimum_sdf, _ = observed_normal_collision(proposed, obstacles)
        proposal_minimum_sdf.append(minimum_sdf)
        if collision:
            rejections.append("observed_normal_deep_penetration")
            continue
        pair_collision = False
        for other in existing_objects:
            if obvious_pair_penetration(proposed, other)[0]:
                pair_collision = True
                break
        if pair_collision:
            rejections.append("obvious_pair_penetration")
            continue
        return proposed, PlacementRecord(
            object_id, label_value, shape_seed, template_identity,
            _integer("material_seed", material_seed), _integer("yaw_seed", yaw_seed),
            proposal, patch.pool_index, patch.frame_id, patch.slot, patch.semantic,
            tuple(proposal_pool_indices), tuple(rejections),
            tuple(proposal_minimum_sdf), minimum_sdf,
            0, () if shape_seed is None else (shape_seed,), (),
        )
    raise PlacementError("no qualified support passed E22/E23/E24 within 128 proposals")


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
    normal_template_library: Sequence[NormalTemplateShape],
    support_pool: QualifiedSupportPool,
    obstacles: ObservedObstacleIndex,
    world_type: WorldType,
    seed: int,
    *,
    maximum_attempts: int = 48,
    trajectory_yaw_by_frame: Mapping[int, float] | None = None,
) -> tuple[WorldSpec, WorldGenerationReport]:
    """Build one immutable train/206 world through the sole qualified pipeline."""

    templates = tuple(normal_template_library)
    world_seed = _integer("seed", seed)
    if not isinstance(support_pool, QualifiedSupportPool) or not isinstance(
        obstacles, ObservedObstacleIndex
    ):
        raise TypeError("training world requires the qualified pool and obstacle index")
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise RenderError("maximum_attempts must be positive")
    normal_count, anomaly_count = _training_entity_counts(world_type, world_seed)
    if (normal_count or anomaly_count) and (
        not templates
        or any(not isinstance(item, NormalTemplateShape) for item in templates)
        or any(item.source_sequence_id != 206 for item in templates)
    ):
        raise RenderError("training entities require a 206 normal-template library")
    if normal_count == anomaly_count == 0:
        world = WorldSpec(world_seed, 206)
        return world, WorldGenerationReport(
            world_seed, 206, world_type, 0, 0, 0, world_seed, world_seed
        )
    if normal_count and trajectory_yaw_by_frame is None:
        raise RenderError("normal-control worlds require trajectory yaw by support frame")

    for attempt in range(maximum_attempts):
        attempt_seed = world_seed + 1_000_003 * attempt
        rng = np.random.default_rng(attempt_seed)
        objects: list[ObjectSpec] = []
        records: list[PlacementRecord] = []
        try:
            labels: list[ObjectLabel] = (
                ["normal-control"] * normal_count
                + ["anomaly-proxy"] * anomaly_count
            )
            rng.shuffle(labels)
            for entity_index, label in enumerate(labels):
                entity_seed = attempt_seed + 10_007 * (entity_index + 1)
                shape_seed: int | None = None
                template_identity: str | None = None
                template_seed: int | None = None
                scale_seed: int | None = None
                perturbation: float | None = None
                yaw_for_support: Callable[[SupportPatch], float] | None = None
                report: ShapeGenerationReport | None = None
                grounding: GroundingEligibility | None = None
                shape_proposals: tuple[int, ...] = ()
                grounding_rejections: tuple[int, ...] = ()
                if label == "normal-control":
                    template_seed = entity_seed + 1
                    scale_seed = entity_seed + 2
                    source = templates[int(
                        np.random.default_rng(template_seed).integers(0, len(templates))
                    )]
                    target_scale = np.random.default_rng(
                        np.random.SeedSequence([scale_seed, 2501])
                    ).uniform(0.9, 1.1, size=3)
                    shape = _aligned_scaled_template(source, target_scale)
                    semantic = source.raw_semantic_id
                    limit = (
                        math.pi if semantic == 30
                        else math.radians(30.0) if semantic in (11, 15, 31, 32)
                        else math.radians(15.0)
                    )
                    perturbation = float(
                        np.random.default_rng(
                            np.random.SeedSequence([entity_seed + 31, 2502])
                        ).uniform(-limit, limit)
                    )
                    template_identity = _normal_template_identity(source)

                    def support_yaw(
                        patch: SupportPatch, offset: float = perturbation
                    ) -> float:
                        assert trajectory_yaw_by_frame is not None
                        return float(trajectory_yaw_by_frame[patch.frame_id]) + offset

                    yaw_for_support = support_yaw
                else:
                    (
                        shape, report, grounding, shape_proposals,
                        grounding_rejections,
                    ) = _grounding_qualified_shape(
                        entity_seed + 3, stride=3072, maximum_proposals=64
                    )
                    shape_seed = shape_proposals[-1]
                material_seed = entity_seed + 11
                yaw_seed = entity_seed + 31
                yaw = (
                    float(perturbation) if perturbation is not None
                    else float(np.random.default_rng(yaw_seed).uniform(-math.pi, math.pi))
                )
                item, record = place_object(
                    shape, MaterialSpec.sample(material_seed), support_pool, obstacles,
                    object_id=entity_index + 1, label=label,
                    proposal_namespace="training-world-v1",
                    proposal_stream=entity_seed, yaw_rad=yaw,
                    material_seed=material_seed, yaw_seed=yaw_seed,
                    shape_seed=shape_seed, template_identity=template_identity,
                    shape_generation_report=report, existing_objects=objects,
                    grounding_eligibility=grounding,
                    yaw_for_support=yaw_for_support,
                )
                record = replace(
                    record,
                    template_seed=template_seed,
                    scale_seed=scale_seed,
                    pose_perturbation_rad=perturbation,
                )
                if label == "anomaly-proxy":
                    record = replace(
                        record,
                        accepted_shape_proposal=len(shape_proposals) - 1,
                        shape_proposal_seeds=shape_proposals,
                        grounding_rejection_seeds=grounding_rejections,
                    )
                objects.append(item)
                records.append(record)
            world = WorldSpec(world_seed, 206, tuple(objects))
            if world.world_type != world_type:
                raise AssertionError("training sampler produced the wrong world type")
        except (RenderError, PlacementError):
            continue
        return world, WorldGenerationReport(
            world_seed, 206, world_type, attempt, normal_count, anomaly_count,
            world_seed, attempt_seed, tuple(records)
        )
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


_E23_SUPPORT_POOL: QualifiedSupportPool | None = None
_E23_OBSTACLES: ObservedObstacleIndex | None = None
_E23_PROPOSALS: tuple[np.ndarray, ...] = ()


def _e23_fixture_errors() -> int:
    shapes = (
        ShapeSpec(((0.5, 0.5, 0.5),), ((0.0, 0.0, 0.0),), ((1.0, 1.0),), (0.0,), ("union",)),
        ShapeSpec(((0.7, 0.4, 0.3),), ((0.0, 0.0, 0.0),), ((1.0, 1.0),), (0.0,), ("union",)),
        ShapeSpec(((0.75, 0.65, 0.20),), ((0.0, 0.0, 0.0),), ((0.8, 1.2),), (0.0,), ("union",)),
        ShapeSpec(((0.75, 0.20, 0.20),), ((0.0, 0.0, 0.0),), ((0.8, 1.2),), (0.0,), ("union",)),
    )
    errors = 0
    for shape in shapes:
        upper = float(_shape_outer_bounds(shape)[1][0])
        for target in (0.10, 0.02, -0.02, -0.06, -0.15):
            root = brentq(
                lambda x: float(shape.signed_distance(np.asarray(((x, 0.0, 0.0),)))[0]) - target,
                0.0,
                2.0 * upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
            )
            measured = float(shape.signed_distance(np.asarray(((root, 0.0, 0.0),)))[0])
            expected_reject = target < -0.05
            if abs(measured - target) > 1.0e-10 or (measured < -0.05) != expected_reject:
                errors += 1
    return errors


def _e23_worker(index: int) -> dict[str, object]:
    pool, obstacles = _E23_SUPPORT_POOL, _E23_OBSTACLES
    if pool is None or obstacles is None or len(_E23_PROPOSALS) != 1024:
        raise RuntimeError("E23 worker state is not initialized")
    shape_seed = 2_000_000 + index
    semantic = 40 if index < 512 else 48 if index < 768 else 49
    material_seed = shape_seed + 2302
    yaw = float(np.random.default_rng(np.random.SeedSequence([shape_seed, 2301])).uniform(-math.pi, math.pi))
    try:
        shape, shape_report = ShapeSpec.sample_with_report(shape_seed)
        item, record = place_object(
            shape, MaterialSpec.sample(material_seed), pool, obstacles,
            object_id=index + 1, label="anomaly-proxy",
            proposal_namespace="E23-support-v1", proposal_stream=shape_seed,
            yaw_rad=yaw, material_seed=material_seed, yaw_seed=shape_seed,
            shape_seed=shape_seed, shape_generation_report=shape_report,
            allowed_support_semantics=(semantic,),
            proposal_rows=_E23_PROPOSALS[index], maximum_candidates=128,
        )
        collision, minimum_sdf, obstacle_ids = observed_normal_collision(item, obstacles)
        shape_json = json.dumps(shape.to_dict(), sort_keys=True, separators=(",", ":"))
        report_json = json.dumps(shape_report.to_dict(), sort_keys=True, separators=(",", ":"))
        object_json = json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":"))
        return {
            "hard_error": 0,
            "error": "",
            "shape_seed": shape_seed,
            "semantic": semantic,
            "shape_hash": hashlib.sha256(shape_json.encode()).hexdigest(),
            "shape_report_hash": hashlib.sha256(report_json.encode()).hexdigest(),
            "yaw_rad": yaw,
            "proposal_count": record.accepted_proposal + 1,
            "proposal_pool_indices": record.proposal_pool_indices,
            "rejection_reasons": record.rejection_reasons,
            "support_pool_index": record.support_pool_index,
            "support_frame": record.support_frame,
            "support_slot": record.support_slot,
            "translation": item.translation_world_m,
            "rotation": item.rotation_world_from_local,
            "minimum_sdf": minimum_sdf,
            "collision": collision,
            "obstacle_count": int(obstacle_ids.size),
            "obstacle_hash": hashlib.sha256(
                np.ascontiguousarray(obstacle_ids).tobytes()
            ).hexdigest(),
            "object_json": object_json,
        }
    except Exception as error:
        return {
            "hard_error": 1,
            "error": f"{type(error).__name__}: {error}",
            "shape_seed": shape_seed,
            "semantic": semantic,
        }


def _e23_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    maximum_proposals = 128
    proposals = np.full((1024, maximum_proposals), -1, dtype=np.int64)
    for index, record in enumerate(records):
        values = np.asarray(record.get("proposal_pool_indices", ()), dtype=np.int64)
        proposals[index, : values.size] = values
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)
    return {
        "shape_seed": values("shape_seed", np.int64, -1),
        "support_semantic": values("semantic", np.uint16, 0),
        "shape_hash": values("shape_hash", "S64", ""),
        "shape_report_hash": values("shape_report_hash", "S64", ""),
        "yaw_rad": values("yaw_rad", np.float64, math.nan),
        "proposal_count": values("proposal_count", np.int16, 0),
        "proposal_pool_indices": proposals,
        "rejection_reasons_json": np.asarray(
            [
                json.dumps(
                    item.get("rejection_reasons", ()),
                    separators=(",", ":"),
                )
                for item in records
            ],
            dtype="U4096",
        ),
        "support_pool_index": values("support_pool_index", np.int64, -1),
        "support_frame": values("support_frame", np.int32, -1),
        "support_slot": values("support_slot", np.int32, -1),
        "translation": np.asarray(
            [item.get("translation", (math.nan,) * 3) for item in records], np.float64
        ),
        "rotation": np.asarray(
            [item.get("rotation", ((math.nan,) * 3,) * 3) for item in records], np.float64
        ),
        "minimum_obstacle_sdf_m": values("minimum_sdf", np.float64, math.nan),
        "accepted_collision": values("collision", np.bool_, True),
        "obstacle_count": values("obstacle_count", np.int32, -1),
        "obstacle_identity_hash": values("obstacle_hash", "S64", ""),
        "object_json": values("object_json", "U16384", ""),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "error_message": values("error", "U512", ""),
    }


def _scientific_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def run_e23_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Execute the frozen two-run E23 qualification on real train/206 returns."""

    if processes != 24:
        raise PlacementError("formal E23 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    pool = load_qualified_support_pool(support_pool_path)
    obstacles = collect_observed_obstacle_index(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    orders = {
        semantic: _identity_order(pool, (semantic,), "E23-support-v1", 0)
        for semantic in SUPPORT_POOL_SEMANTICS
    }
    local_indices = {40: 0, 48: 0, 49: 0}
    quotas = {40: 512, 48: 256, 49: 256}
    proposal_rows: list[np.ndarray] = []
    for index in range(1024):
        semantic = 40 if index < 512 else 48 if index < 768 else 49
        local = local_indices[semantic]
        local_indices[semantic] += 1
        order = orders[semantic]
        proposal_rows.append(
            order[(local + np.arange(128, dtype=np.int64) * quotas[semantic]) % order.size]
        )
    global _E23_SUPPORT_POOL, _E23_OBSTACLES, _E23_PROPOSALS
    _E23_SUPPORT_POOL = pool
    _E23_OBSTACLES = obstacles
    _E23_PROPOSALS = tuple(proposal_rows)
    fixture_errors = _e23_fixture_errors()
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    context = mp.get_context("fork")
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e23_worker, range(1024))
        run_seconds.append(time.monotonic() - started)
        runs.append(_e23_arrays(records))
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    hard_errors = int(np.count_nonzero(first["hard_error_code"]))
    collisions = int(np.count_nonzero(first["accepted_collision"]))
    completed = int(np.count_nonzero(first["support_pool_index"] >= 0))
    passed = (
        fixture_errors == 0 and hard_errors == 0 and collisions == 0
        and completed == 1024 and reproduced
        and bool(np.all(first["proposal_count"] <= 128))
    )
    scientific_hash = _scientific_array_hash(first)
    metadata = {
        "experiment": "E23",
        "passed": passed,
        "fixture_errors": fixture_errors,
        "objects": 1024,
        "completed": completed,
        "hard_errors": hard_errors,
        "accepted_collisions": collisions,
        "elementwise_reproduced": reproduced,
        "scientific_array_hash": scientific_hash,
        "run_seconds": run_seconds,
        "obstacle_points": int(obstacles.points_world_m.shape[0]),
        "support_pool_sha256": SUPPORT_POOL_SHA256,
        "support_prefix": "E23-support-v1",
        "shape_seeds": [2_000_000, 2_001_023],
        "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **first, metadata_json=np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    ))
    os.replace(temporary, destination)
    return metadata


_E24_SUPPORT_POOL: QualifiedSupportPool | None = None
_E24_OBSTACLES: ObservedObstacleIndex | None = None


def _e24_fixture_errors() -> int:
    material = MaterialSpec(0.5, 0.2)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    sphere = ShapeSpec(
        ((0.5, 0.5, 0.5),), ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),), (0.0,), ("union",),
    )
    ellipsoid = ShapeSpec(
        ((0.6, 0.4, 0.5),), ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),), (0.0,), ("union",),
    )
    schema7 = ShapeSpec(
        ((0.5, 0.5, 0.5),), ((0.0, 0.0, 0.0),),
        ((0.7, 1.4),), (0.0,), ("union",),
    )
    hull = NormalTemplateShape(
        np.asarray(
            [
                (x, y, z)
                for x in (-0.5, 0.5)
                for y in (-0.5, 0.5)
                for z in (-0.5, 0.5)
            ],
            dtype=np.float64,
        ),
        np.empty((0, 3), dtype=np.int32),
        206, 0, 10, 1, (0.0, 0.0, 0.0),
    )

    def item(
        shape: InsertShape,
        identifier: int,
        translation: tuple[float, float, float],
    ) -> ObjectSpec:
        label: ObjectLabel = (
            "normal-control" if isinstance(shape, NormalTemplateShape)
            else "anomaly-proxy"
        )
        return ObjectSpec(identifier, label, shape, material, translation, identity)

    fixtures = (
        (sphere, sphere, 0, 1.0),
        (ellipsoid, ellipsoid, 1, 0.8),
        (schema7, schema7, 0, 1.0),
        (hull, schema7, 0, 1.0),
    )
    errors = 0
    for left_shape, right_shape, axis, span in fixtures:
        for overlap, expected in (
            (-0.10, False), (0.0, False), (0.02, False),
            (0.06, True), (0.15, True),
        ):
            translation = [0.0, 0.0, 0.0]
            translation[axis] = span - overlap
            measured, _ = obvious_pair_penetration(
                item(left_shape, 1, (0.0, 0.0, 0.0)),
                item(right_shape, 2, tuple(translation)),
            )
            if measured != expected:
                errors += 1
    return errors


def _e24_v2_worker(index: int) -> dict[str, object]:
    pool, obstacles = _E24_SUPPORT_POOL, _E24_OBSTACLES
    if pool is None or obstacles is None:
        raise RuntimeError("E24 worker state is not initialized")
    world_seed = 2_100_000 + index
    entity_count = int(
        np.random.default_rng(
            np.random.SeedSequence([world_seed, 2401])
        ).integers(2, 7)
    )
    try:
        objects: list[ObjectSpec] = []
        records: list[PlacementRecord] = []
        for entity_index in range(entity_count):
            entity_slot = index * 6 + entity_index
            (
                shape, report, grounding, shape_proposals,
                grounding_rejections,
            ) = _grounding_qualified_shape(
                3_000_000 + entity_slot,
                stride=3072,
                maximum_proposals=64,
            )
            shape_seed = shape_proposals[-1]
            material_seed = shape_seed + 2403
            yaw_seed = shape_seed + 2402
            yaw = float(
                np.random.default_rng(
                    np.random.SeedSequence([shape_seed, 2402])
                ).uniform(-math.pi, math.pi)
            )
            proposed, record = place_object(
                shape, MaterialSpec.sample(material_seed), pool, obstacles,
                object_id=entity_index + 1, label="anomaly-proxy",
                proposal_namespace="E24-support-v1",
                proposal_stream=entity_slot,
                yaw_rad=yaw, material_seed=material_seed, yaw_seed=yaw_seed,
                shape_seed=shape_seed, shape_generation_report=report,
                existing_objects=objects, maximum_candidates=128,
                grounding_eligibility=grounding,
            )
            record = replace(
                record,
                accepted_shape_proposal=len(shape_proposals) - 1,
                shape_proposal_seeds=shape_proposals,
                grounding_rejection_seeds=grounding_rejections,
            )
            objects.append(proposed)
            records.append(record)
        world = WorldSpec(world_seed, 206, tuple(objects))
        validation_errors = 0
        final_pair_penetrations = 0
        for item_value, record in zip(world.objects, records, strict=True):
            support_row = int(
                np.searchsorted(pool.pool_indices, record.support_pool_index)
            )
            if (
                support_row >= pool.pool_indices.size
                or int(pool.pool_indices[support_row]) != record.support_pool_index
            ):
                raise PlacementError("accepted support identity is absent from the pool")
            grounding = qualify_grounding(item_value.shape)
            if (
                not grounding.passed
                or observed_normal_collision(item_value, obstacles)[0]
                or record.accepted_proposal + 1 != len(record.proposal_pool_indices)
                or record.accepted_proposal != len(record.rejection_reasons)
                or record.accepted_shape_proposal + 1
                != len(record.shape_proposal_seeds)
                or record.accepted_shape_proposal
                != len(record.grounding_rejection_seeds)
                or record.shape_proposal_seeds[-1] != record.shape_seed
            ):
                validation_errors += 1
        for left in range(entity_count):
            for right in range(left + 1, entity_count):
                final_pair_penetrations += int(
                    obvious_pair_penetration(world.objects[left], world.objects[right])[0]
                )
        report_json = json.dumps(
            [record.to_dict() for record in records],
            sort_keys=True, separators=(",", ":"),
        )
        return {
            "hard_error": 0,
            "error": "",
            "world_seed": world_seed,
            "entity_count": entity_count,
            "support_proposal_count": sum(
                record.accepted_proposal + 1 for record in records
            ),
            "shape_proposal_count": sum(
                len(record.shape_proposal_seeds) for record in records
            ),
            "grounding_rejections": sum(
                len(record.grounding_rejection_seeds) for record in records
            ),
            "pair_rejections": sum(
                reason == "obvious_pair_penetration"
                for record in records for reason in record.rejection_reasons
            ),
            "validation_errors": validation_errors,
            "final_pair_penetrations": final_pair_penetrations,
            "world_hash": world.identity,
            "world_json": world.to_json(),
            "placement_report_json": report_json,
        }
    except PlacementError as error:
        shape_exhaustion = int(
            "no E22-qualified shape" in str(error)
        )
        return {
            "hard_error": 0,
            "shape_exhaustion": shape_exhaustion,
            "placement_exhaustion": 1 - shape_exhaustion,
            "failure_stage": (
                "shape_proposal_exhaustion" if shape_exhaustion
                else "placement_proposal_exhaustion"
            ),
            "error": f"PlacementError: {error}",
            "world_seed": world_seed,
            "entity_count": entity_count,
        }
    except Exception as error:
        return {
            "hard_error": 1,
            "shape_exhaustion": 0,
            "placement_exhaustion": 0,
            "failure_stage": "hard_error",
            "error": f"{type(error).__name__}: {error}",
            "world_seed": world_seed,
            "entity_count": entity_count,
        }


def _e24_v2_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)
    return {
        "world_seed": values("world_seed", np.int64, -1),
        "entity_count": values("entity_count", np.int8, 0),
        "support_proposal_count": values("support_proposal_count", np.int16, 0),
        "shape_proposal_count": values("shape_proposal_count", np.int16, 0),
        "grounding_rejections": values("grounding_rejections", np.int16, 0),
        "pair_rejections": values("pair_rejections", np.int16, 0),
        "validation_errors": values("validation_errors", np.int16, 0),
        "final_pair_penetrations": values("final_pair_penetrations", np.int16, 0),
        "world_hash": values("world_hash", "S64", ""),
        "world_json": np.asarray(
            [str(item.get("world_json", "")).encode() for item in records]
        ),
        "placement_report_json": np.asarray(
            [str(item.get("placement_report_json", "")).encode() for item in records]
        ),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "shape_exhaustion_code": values("shape_exhaustion", np.uint8, 0),
        "placement_exhaustion_code": values("placement_exhaustion", np.uint8, 0),
        "failure_stage": values("failure_stage", "U64", ""),
        "error_message": values("error", "U512", ""),
    }


def run_e24_v2_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Execute the frozen two-run E24-v2 multi-entity qualification."""

    if processes != 24:
        raise PlacementError("formal E24-v2 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    pool = load_qualified_support_pool(support_pool_path)
    obstacles = collect_observed_obstacle_index(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    global _E24_SUPPORT_POOL, _E24_OBSTACLES
    _E24_SUPPORT_POOL = pool
    _E24_OBSTACLES = obstacles
    fixture_errors = _e24_fixture_errors()
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    context = mp.get_context("fork")
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e24_v2_worker, range(512))
        run_seconds.append(time.monotonic() - started)
        runs.append(_e24_v2_arrays(records))
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    hard_errors = int(np.count_nonzero(first["hard_error_code"]))
    shape_exhaustions = int(np.count_nonzero(first["shape_exhaustion_code"]))
    placement_exhaustions = int(
        np.count_nonzero(first["placement_exhaustion_code"])
    )
    validation_errors = int(np.sum(first["validation_errors"]))
    final_pair_penetrations = int(np.sum(first["final_pair_penetrations"]))
    completed = int(np.count_nonzero(first["world_hash"] != b""))
    passed = (
        fixture_errors == 0 and completed == 512 and hard_errors == 0
        and shape_exhaustions == 0 and placement_exhaustions == 0
        and validation_errors == 0 and final_pair_penetrations == 0
        and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    metadata = {
        "experiment": "E24-v2",
        "passed": passed,
        "fixture_errors": fixture_errors,
        "worlds": 512,
        "completed": completed,
        "hard_errors": hard_errors,
        "shape_exhaustions": shape_exhaustions,
        "placement_exhaustions": placement_exhaustions,
        "shape_proposals": int(np.sum(first["shape_proposal_count"])),
        "grounding_rejections": int(np.sum(first["grounding_rejections"])),
        "support_proposals": int(np.sum(first["support_proposal_count"])),
        "pair_rejections": int(np.sum(first["pair_rejections"])),
        "validation_errors": validation_errors,
        "final_pair_penetrations": final_pair_penetrations,
        "elementwise_reproduced": reproduced,
        "scientific_array_hash": scientific_hash,
        "run_seconds": run_seconds,
        "obstacle_points": int(obstacles.points_world_m.shape[0]),
        "support_pool_sha256": SUPPORT_POOL_SHA256,
        "world_seeds": [2_100_000, 2_100_511],
        "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return metadata


_E25_SUPPORT_POOL: QualifiedSupportPool | None = None
_E25_OBSTACLES: ObservedObstacleIndex | None = None
_E25_TEMPLATES: dict[int, tuple[NormalTemplateShape, ...]] = {}
_E25_TRAJECTORY_YAW: dict[int, float] = {}


def _normal_template_identity(template: NormalTemplateShape) -> str:
    return hashlib.sha256(
        json.dumps(
            template.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _aligned_scaled_template(
    template: NormalTemplateShape, scale_xyz: Sequence[float]
) -> NormalTemplateShape:
    points = np.asarray(template.vertices_m, dtype=np.float64)
    covariance = np.cov(points[:, :2], rowvar=False, bias=True)
    _, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, -1]
    if principal[0] < 0.0 or (
        abs(principal[0]) <= EPSILON and principal[1] < 0.0
    ):
        principal = -principal
    angle = math.atan2(float(principal[1]), float(principal[0]))
    cosine, sine = math.cos(angle), math.sin(angle)
    alignment = np.asarray(((cosine, -sine), (sine, cosine)))
    aligned = points.copy()
    aligned[:, :2] = points[:, :2] @ alignment
    scale = np.asarray(_tuple_values("scale_xyz", scale_xyz, 3))
    return NormalTemplateShape(
        aligned * scale,
        template.faces,
        template.source_sequence_id,
        template.source_frame_id,
        template.raw_semantic_id,
        template.source_instance_id,
        template.source_center_sensor_m,
        tuple(map(float, scale)),
    )


def trajectory_yaw_by_frame(frames: Sequence[SourceFrame]) -> dict[int, float]:
    ordered = tuple(sorted(frames, key=lambda frame: frame.frame_id))
    if len(ordered) < 2:
        raise PlacementError("trajectory tangent requires at least two source frames")
    positions = np.asarray([_pose(frame)[1] for frame in ordered])
    result: dict[int, float] = {}
    for index, frame in enumerate(ordered):
        if index == 0:
            tangent = positions[1] - positions[0]
        elif index == len(ordered) - 1:
            tangent = positions[-1] - positions[-2]
        else:
            tangent = positions[index + 1] - positions[index - 1]
        horizontal = tangent[:2]
        if np.linalg.norm(horizontal) <= EPSILON:
            horizontal = _pose(frame)[0][:2, 0]
        if np.linalg.norm(horizontal) <= EPSILON:
            raise PlacementError("trajectory tangent and pose fallback are degenerate")
        result[int(frame.frame_id)] = math.atan2(
            float(horizontal[1]), float(horizontal[0])
        )
    return result


def _e25_template(index: int) -> tuple[NormalTemplateShape, str]:
    vehicle = tuple(
        semantic for semantic in (10, 11, 15, 18, 20)
        if semantic in _E25_TEMPLATES
    )
    person = tuple(
        semantic for semantic in (30, 31, 32)
        if semantic in _E25_TEMPLATES
    )
    groups = (vehicle, person)
    semantics = groups[index % 2]
    if not semantics:
        raise PlacementError("E25 selected an inactive broad group")
    group_index = index // 2
    semantic = semantics[group_index % len(semantics)]
    class_index = group_index // len(semantics)
    templates = _E25_TEMPLATES[semantic]
    template = templates[class_index % len(templates)]
    return template, _normal_template_identity(template)


def _e25_worker(index: int) -> dict[str, object]:
    pool, obstacles = _E25_SUPPORT_POOL, _E25_OBSTACLES
    if pool is None or obstacles is None or not _E25_TEMPLATES:
        raise RuntimeError("E25 worker state is not initialized")
    control_seed = 2_500_000 + index
    try:
        source, template_identity = _e25_template(index)
        scale = np.random.default_rng(
            np.random.SeedSequence([control_seed, 2501])
        ).uniform(0.9, 1.1, size=3)
        shape = _aligned_scaled_template(source, scale)
        semantic = shape.raw_semantic_id
        pose_rng = np.random.default_rng(
            np.random.SeedSequence([control_seed, 2502])
        )
        limit = math.pi if semantic == 30 else math.radians(15.0)
        perturbation = float(pose_rng.uniform(-limit, limit))
        material_seed = control_seed + 2503

        def yaw_for_support(patch: SupportPatch) -> float:
            return _E25_TRAJECTORY_YAW[patch.frame_id] + perturbation

        item, record = place_object(
            shape, MaterialSpec.sample(material_seed), pool, obstacles,
            object_id=index + 1, label="normal-control",
            proposal_namespace="E25-support-v1", proposal_stream=index,
            yaw_rad=perturbation, material_seed=material_seed,
            yaw_seed=control_seed, template_identity=template_identity,
            maximum_candidates=128, yaw_for_support=yaw_for_support,
        )
        support_row = int(np.searchsorted(pool.pool_indices, record.support_pool_index))
        if (
            support_row >= pool.pool_indices.size
            or int(pool.pool_indices[support_row]) != record.support_pool_index
        ):
            raise PlacementError("accepted E25 support identity is absent")
        patch = pool.patch(support_row)
        expected_rotation = _ground_rotation(
            np.asarray(patch.normal_world), yaw_for_support(patch)
        )
        semantic_violation = int(
            record.support_semantic not in normal_control_support_semantics(semantic)
        )
        scale_error = int(np.any((scale < 0.9) | (scale > 1.1)))
        pose_error = int(
            np.max(np.abs(expected_rotation - np.asarray(item.rotation_world_from_local)))
            > 1.0e-10
        )
        validation_error = int(
            not qualify_grounding(item.shape).passed
            or observed_normal_collision(item, obstacles)[0]
            or record.accepted_proposal + 1 != len(record.proposal_pool_indices)
            or record.accepted_proposal != len(record.rejection_reasons)
            or record.template_identity != template_identity
        )
        fixture_error = 0
        fixture_hash = ""
        if index == 0:
            proxy_shape, proxy_report, proxy_grounding, proxy_proposals, _ = (
                _grounding_qualified_shape(
                    5_000_000, stride=3072, maximum_proposals=64
                )
            )
            proxy_seed = proxy_proposals[-1]
            proxy_yaw = float(
                np.random.default_rng(
                    np.random.SeedSequence([proxy_seed, 2504])
                ).uniform(-math.pi, math.pi)
            )
            proxy, proxy_record = place_object(
                proxy_shape, MaterialSpec.sample(proxy_seed + 2505), pool, obstacles,
                object_id=2, label="anomaly-proxy",
                proposal_namespace="E25-mixed-fixture-v1", proposal_stream=0,
                yaw_rad=proxy_yaw, material_seed=proxy_seed + 2505,
                yaw_seed=proxy_seed, shape_seed=proxy_seed,
                shape_generation_report=proxy_report, existing_objects=(item,),
                grounding_eligibility=proxy_grounding, maximum_candidates=128,
            )
            fixture_error = int(
                observed_normal_collision(proxy, obstacles)[0]
                or obvious_pair_penetration(item, proxy)[0]
            )
            fixture_hash = hashlib.sha256(
                json.dumps(
                    item.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
                + json.dumps(
                    {"proxy": proxy.to_dict(), "record": proxy_record.to_dict()},
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest()
        return {
            "hard_error": 0,
            "error": "",
            "control_seed": control_seed,
            "semantic": semantic,
            "template_identity": template_identity,
            "scale": scale,
            "pose_perturbation_rad": perturbation,
            "support_semantic": record.support_semantic,
            "support_pool_index": record.support_pool_index,
            "support_proposal_count": record.accepted_proposal + 1,
            "semantic_violation": semantic_violation,
            "scale_error": scale_error,
            "pose_error": pose_error,
            "validation_error": validation_error,
            "fixture_error": fixture_error,
            "fixture_hash": fixture_hash,
            "object_json": json.dumps(
                item.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            "placement_report_json": json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":")
            ),
        }
    except PlacementError as error:
        return {
            "hard_error": 0,
            "placement_exhaustion": 1,
            "error": f"PlacementError: {error}",
            "control_seed": control_seed,
        }
    except Exception as error:
        return {
            "hard_error": 1,
            "placement_exhaustion": 0,
            "error": f"{type(error).__name__}: {error}",
            "control_seed": control_seed,
        }


def _e25_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)
    return {
        "control_seed": values("control_seed", np.int64, -1),
        "semantic": values("semantic", np.uint16, 0),
        "template_identity": values("template_identity", "S64", ""),
        "scale": np.asarray(
            [item.get("scale", (math.nan,) * 3) for item in records], np.float64
        ),
        "pose_perturbation_rad": values("pose_perturbation_rad", np.float64, math.nan),
        "support_semantic": values("support_semantic", np.uint16, 0),
        "support_pool_index": values("support_pool_index", np.int64, -1),
        "support_proposal_count": values("support_proposal_count", np.int16, 0),
        "semantic_violation": values("semantic_violation", np.uint8, 0),
        "scale_error": values("scale_error", np.uint8, 0),
        "pose_error": values("pose_error", np.uint8, 0),
        "validation_error": values("validation_error", np.uint8, 0),
        "fixture_error": values("fixture_error", np.uint8, 0),
        "fixture_hash": values("fixture_hash", "S64", ""),
        "object_json": np.asarray(
            [str(item.get("object_json", "")).encode() for item in records]
        ),
        "placement_report_json": np.asarray(
            [str(item.get("placement_report_json", "")).encode() for item in records]
        ),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "placement_exhaustion_code": values("placement_exhaustion", np.uint8, 0),
        "error_message": values("error", "U512", ""),
    }


def run_e25_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Execute the frozen two-run E25 normal-control qualification."""

    if processes != 24:
        raise PlacementError("formal E25 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    frames = tuple(sequence.source_frame(frame_id) for frame_id in sequence.frame_ids)
    templates = extract_normal_template_library(frames)
    by_semantic: dict[int, tuple[NormalTemplateShape, ...]] = {}
    for semantic in sorted(NORMAL_TEMPLATE_SEMANTICS):
        selected = tuple(
            item for item in templates if item.raw_semantic_id == semantic
        )
        if selected:
            by_semantic[semantic] = selected
    counts = {semantic: len(items) for semantic, items in by_semantic.items()}
    identities = [_normal_template_identity(item) for item in templates]
    library_hash = hashlib.sha256("".join(identities).encode()).hexdigest()
    if counts != {10: 64, 18: 64, 20: 64, 30: 64} or len(set(identities)) != 256:
        raise PlacementError("E25 observable-template precheck changed")
    pool = load_qualified_support_pool(support_pool_path)
    obstacles = collect_observed_obstacle_index(frames)
    global _E25_SUPPORT_POOL, _E25_OBSTACLES, _E25_TEMPLATES, _E25_TRAJECTORY_YAW
    _E25_SUPPORT_POOL = pool
    _E25_OBSTACLES = obstacles
    _E25_TEMPLATES = by_semantic
    _E25_TRAJECTORY_YAW = trajectory_yaw_by_frame(frames)
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    context = mp.get_context("fork")
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e25_worker, range(1024))
        run_seconds.append(time.monotonic() - started)
        runs.append(_e25_arrays(records))
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    hard_errors = int(np.count_nonzero(first["hard_error_code"]))
    placement_exhaustions = int(np.count_nonzero(first["placement_exhaustion_code"]))
    completed = int(np.count_nonzero(first["template_identity"] != b""))
    semantic_violations = int(np.sum(first["semantic_violation"]))
    scale_errors = int(np.sum(first["scale_error"]))
    pose_errors = int(np.sum(first["pose_error"]))
    validation_errors = int(np.sum(first["validation_error"]))
    fixture_errors = int(np.sum(first["fixture_error"]))
    passed = (
        library_hash == "de5dfd765ac7d4fe4bb4644c40ecafdd80cdc31a3d0b6fc4fccd8e84a9fd906b"
        and completed == 1024 and hard_errors == 0 and placement_exhaustions == 0
        and semantic_violations == 0 and scale_errors == 0 and pose_errors == 0
        and validation_errors == 0 and fixture_errors == 0 and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    metadata = {
        "experiment": "E25", "passed": passed, "templates": len(templates),
        "active_counts": counts, "library_sha256": library_hash,
        "placements": 1024, "completed": completed,
        "hard_errors": hard_errors, "placement_exhaustions": placement_exhaustions,
        "semantic_violations": semantic_violations, "scale_errors": scale_errors,
        "pose_errors": pose_errors, "validation_errors": validation_errors,
        "fixture_errors": fixture_errors, "elementwise_reproduced": reproduced,
        "scientific_array_hash": scientific_hash, "run_seconds": run_seconds,
        "support_pool_sha256": SUPPORT_POOL_SHA256, "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return metadata


_E26_SUPPORT_POOL: QualifiedSupportPool | None = None
_E26_OBSTACLES: ObservedObstacleIndex | None = None
_E26_TEMPLATES: tuple[NormalTemplateShape, ...] = ()
_E26_TRAJECTORY_YAW: dict[int, float] = {}
_E26_RENDERER_IDENTITY = ""


def _e26_request_identity(world_hash: str, frame_id: int) -> str:
    payload = f"{world_hash}:{frame_id}:{_E26_RENDERER_IDENTITY}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _e26_worker(index: int) -> dict[str, object]:
    pool, obstacles = _E26_SUPPORT_POOL, _E26_OBSTACLES
    if (
        pool is None or obstacles is None or not _E26_TEMPLATES
        or len(_E26_RENDERER_IDENTITY) != 64
    ):
        raise RuntimeError("E26 worker state is not initialized")
    world_seed = 2_600_000 + index
    world_type = WORLD_TYPES[index // 64]
    try:
        world, report = sample_training_world(
            _E26_TEMPLATES, pool, obstacles, world_type, world_seed,
            maximum_attempts=48,
            trajectory_yaw_by_frame=_E26_TRAJECTORY_YAW,
        )
        world_json = world.to_json()
        report_json = report.to_json()
        round_trip = WorldSpec.from_dict(json.loads(world_json))
        report_round_trip = WorldGenerationReport.from_dict(json.loads(report_json))
        round_trip_errors = int(
            round_trip.to_json() != world_json
            or round_trip.identity != world.identity
            or report_round_trip.to_json() != report_json
        )
        validation_errors = int(
            world.seed != world_seed
            or world.source_sequence_id != 206
            or world.world_type != world_type
            or report.world_seed != world_seed
            or report.world_type != world_type
            or report.normal_count != world.normal_control_count
            or report.anomaly_count != world.anomaly_proxy_count
            or len(report.placements) != len(world.objects)
            or [item.object_id for item in world.objects]
            != list(range(1, len(world.objects) + 1))
        )
        support_errors = 0
        pose_errors = 0
        material_errors = 0
        for item, record in zip(world.objects, report.placements, strict=True):
            row = int(np.searchsorted(pool.pool_indices, record.support_pool_index))
            if (
                row >= pool.pool_indices.size
                or int(pool.pool_indices[row]) != record.support_pool_index
            ):
                support_errors += 1
                continue
            patch = pool.patch(row)
            validation_errors += int(
                record.object_id != item.object_id
                or record.label != item.label
                or not qualify_grounding(item.shape).passed
                or observed_normal_collision(item, obstacles)[0]
                or record.accepted_proposal + 1
                != len(record.proposal_pool_indices)
                or record.accepted_proposal != len(record.rejection_reasons)
            )
            material_errors += int(
                item.material.to_dict()
                != MaterialSpec.sample(record.material_seed).to_dict()
            )
            if item.label == "normal-control":
                support_errors += int(
                    record.support_semantic
                    not in normal_control_support_semantics(item.shape.raw_semantic_id)
                )
                perturbation = record.pose_perturbation_rad
                if perturbation is None:
                    pose_errors += 1
                else:
                    expected = _ground_rotation(
                        np.asarray(patch.normal_world),
                        _E26_TRAJECTORY_YAW[patch.frame_id] + perturbation,
                    )
                    pose_errors += int(
                        np.max(np.abs(
                            expected - np.asarray(item.rotation_world_from_local)
                        )) > 1.0e-10
                    )
                validation_errors += int(
                    not isinstance(item.shape, NormalTemplateShape)
                    or record.template_identity is None
                    or record.template_seed is None
                    or record.scale_seed is None
                    or np.any(
                        (np.asarray(item.shape.scale_xyz) < 0.9)
                        | (np.asarray(item.shape.scale_xyz) > 1.1)
                    )
                )
            else:
                validation_errors += int(
                    not isinstance(item.shape, ShapeSpec)
                    or record.shape_seed is None
                    or not record.shape_proposal_seeds
                    or record.shape_proposal_seeds[-1] != record.shape_seed
                    or len(record.grounding_rejection_seeds)
                    != record.accepted_shape_proposal
                )
        pair_errors = sum(
            int(obvious_pair_penetration(world.objects[left], world.objects[right])[0])
            for left in range(len(world.objects))
            for right in range(left + 1, len(world.objects))
        )

        before_traversal = world.to_json()
        center = 2 + world_seed % 445
        forward = tuple(range(center - 2, center + 3))
        reverse = tuple(reversed(forward))
        random_order = tuple(
            np.random.default_rng(
                np.random.SeedSequence([world_seed, 2601])
            ).permutation(forward).tolist()
        )
        expected_requests = {
            frame_id: _e26_request_identity(world.identity, frame_id)
            for frame_id in forward
        }

        def traverse(order: Sequence[int], cache: dict[int, str]) -> dict[int, str]:
            for frame_id in order:
                cache.setdefault(
                    int(frame_id), _e26_request_identity(world.identity, int(frame_id))
                )
            return dict(cache)

        uncached = traverse(forward, {})
        cached: dict[int, str] = {}
        traverse(reverse, cached)
        cached_result = traverse(random_order, cached)
        cached.clear()
        rebuilt = traverse(random_order, cached)
        traversal_errors = int(
            uncached != expected_requests
            or cached_result != expected_requests
            or rebuilt != expected_requests
            or world.to_json() != before_traversal
        )
        return {
            "hard_error": 0,
            "error": "",
            "world_seed": world_seed,
            "world_type": world_type,
            "normal_count": world.normal_control_count,
            "anomaly_count": world.anomaly_proxy_count,
            "entity_count": len(world.objects),
            "world_attempt": report.world_attempt,
            "world_hash": world.identity,
            "world_json": world_json,
            "report_json": report_json,
            "round_trip_error": round_trip_errors,
            "validation_error": validation_errors,
            "support_error": support_errors,
            "pose_error": pose_errors,
            "material_error": material_errors,
            "pair_error": pair_errors,
            "traversal_error": traversal_errors,
            "request_manifest_hash": hashlib.sha256(
                json.dumps(
                    expected_requests, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
    except PlacementError as error:
        return {
            "hard_error": 0,
            "placement_exhaustion": 1,
            "error": f"PlacementError: {error}",
            "world_seed": world_seed,
            "world_type": world_type,
        }
    except Exception as error:
        return {
            "hard_error": 1,
            "placement_exhaustion": 0,
            "error": f"{type(error).__name__}: {error}",
            "world_seed": world_seed,
            "world_type": world_type,
        }


def _e26_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)

    return {
        "world_seed": values("world_seed", np.int64, -1),
        "world_type": values("world_type", "U16", ""),
        "normal_count": values("normal_count", np.int8, 0),
        "anomaly_count": values("anomaly_count", np.int8, 0),
        "entity_count": values("entity_count", np.int8, 0),
        "world_attempt": values("world_attempt", np.int8, -1),
        "world_hash": values("world_hash", "S64", ""),
        "world_json": np.asarray(
            [str(item.get("world_json", "")).encode() for item in records]
        ),
        "report_json": np.asarray(
            [str(item.get("report_json", "")).encode() for item in records]
        ),
        "request_manifest_hash": values("request_manifest_hash", "S64", ""),
        "round_trip_error": values("round_trip_error", np.uint8, 0),
        "validation_error": values("validation_error", np.uint8, 0),
        "support_error": values("support_error", np.uint8, 0),
        "pose_error": values("pose_error", np.uint8, 0),
        "material_error": values("material_error", np.uint8, 0),
        "pair_error": values("pair_error", np.uint8, 0),
        "traversal_error": values("traversal_error", np.uint8, 0),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "placement_exhaustion_code": values("placement_exhaustion", np.uint8, 0),
        "error_message": values("error", "U512", ""),
    }


def run_e26_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Run the frozen single-process and 24-process E26 world audit."""

    if processes != 24:
        raise PlacementError("formal E26 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    frames = tuple(sequence.source_frame(frame_id) for frame_id in sequence.frame_ids)
    templates = extract_normal_template_library(frames)
    pool = load_qualified_support_pool(support_pool_path)
    obstacles = collect_observed_obstacle_index(frames)
    renderer_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    global _E26_SUPPORT_POOL, _E26_OBSTACLES, _E26_TEMPLATES
    global _E26_TRAJECTORY_YAW, _E26_RENDERER_IDENTITY
    _E26_SUPPORT_POOL = pool
    _E26_OBSTACLES = obstacles
    _E26_TEMPLATES = templates
    _E26_TRAJECTORY_YAW = trajectory_yaw_by_frame(frames)
    _E26_RENDERER_IDENTITY = renderer_identity

    source = Path(__file__).read_text(encoding="utf-8")
    authority_errors = int(
        source.count("def place_object(") != 1
        or source.count("_grounded_object(") != 2
        or "def generate_fixed_development_worlds(" in source
    )
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    started = time.monotonic()
    runs.append(_e26_arrays([_e26_worker(index) for index in range(256)]))
    run_seconds.append(time.monotonic() - started)
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        runs.append(_e26_arrays(workers.map(_e26_worker, range(256))))
    run_seconds.append(time.monotonic() - started)
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    completed = int(np.count_nonzero(first["world_hash"] != b""))
    type_errors = int(np.count_nonzero(
        first["world_type"]
        != np.repeat(np.asarray(WORLD_TYPES, dtype="U16"), 64)
    ))
    error_fields = (
        "round_trip_error", "validation_error", "support_error", "pose_error",
        "material_error", "pair_error", "traversal_error", "hard_error_code",
        "placement_exhaustion_code",
    )
    errors = {name: int(np.sum(first[name])) for name in error_fields}
    passed = (
        completed == 256 and type_errors == 0 and authority_errors == 0
        and all(value == 0 for value in errors.values()) and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    metadata = {
        "experiment": "E26", "passed": passed, "worlds": 256,
        "completed": completed, "type_errors": type_errors,
        "authority_errors": authority_errors, **errors,
        "elementwise_reproduced": reproduced, "run_seconds": run_seconds,
        "renderer_identity": renderer_identity,
        "support_pool_sha256": SUPPORT_POOL_SHA256,
        "scientific_array_hash": scientific_hash, "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    os.replace(temporary, destination)
    return metadata


def _render_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AJAE authoritative renderer experiments")
    subcommands = parser.add_subparsers(dest="command", required=True)
    e23 = subcommands.add_parser("qualify-e23")
    e23.add_argument("--data-root", type=Path, required=True)
    e23.add_argument("--support-pool", type=Path, required=True)
    e23.add_argument("--output", type=Path, required=True)
    e23.add_argument("--processes", type=int, default=24)
    e24 = subcommands.add_parser("qualify-e24-v2")
    e24.add_argument("--data-root", type=Path, required=True)
    e24.add_argument("--support-pool", type=Path, required=True)
    e24.add_argument("--output", type=Path, required=True)
    e24.add_argument("--processes", type=int, default=24)
    e25 = subcommands.add_parser("qualify-e25")
    e25.add_argument("--data-root", type=Path, required=True)
    e25.add_argument("--support-pool", type=Path, required=True)
    e25.add_argument("--output", type=Path, required=True)
    e25.add_argument("--processes", type=int, default=24)
    e26 = subcommands.add_parser("qualify-e26")
    e26.add_argument("--data-root", type=Path, required=True)
    e26.add_argument("--support-pool", type=Path, required=True)
    e26.add_argument("--output", type=Path, required=True)
    e26.add_argument("--processes", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _render_parser().parse_args(argv)
    if args.command == "qualify-e23":
        result = run_e23_qualification(
            args.data_root, args.support_pool, args.output, processes=args.processes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e24-v2":
        result = run_e24_v2_qualification(
            args.data_root, args.support_pool, args.output, processes=args.processes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e25":
        result = run_e25_qualification(
            args.data_root, args.support_pool, args.output, processes=args.processes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e26":
        result = run_e26_qualification(
            args.data_root, args.support_pool, args.output, processes=args.processes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    raise AssertionError("unreachable renderer command")


if __name__ == "__main__":
    raise SystemExit(main())
