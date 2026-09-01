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
import ast
import gc
import math
import os
import time
import warnings
import multiprocessing as mp
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq, differential_evolution
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
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
FROZEN_SENSOR_CALIBRATION_SHA256 = (
    "b532b7e04d9025233b2768b8fb36287e477f62f20a3ff685a62f4a4a29bfefe0"
)
FROZEN_E25_NEW_ARTIFACT_SHA256 = (
    "30fc7d1ecd60d005cb18c60ac81b1c7335e2121fcd3f1da5f440b5387a747b19"
)
FROZEN_GATE1_SUPPORT_POOL_SHA256 = (
    "fc3646fbc145cdc29d2cf203835a3e0018bacbc6eaf714e091d21f7b93bfaf50"
)
FROZEN_E38_V2_ARTIFACT_SHA256 = (
    "914b185ae31d5509fa286208c26bb4271460d289a02ec398eaee715b7eeb7c9a"
)
FROZEN_E39_V2_ARTIFACT_SHA256 = (
    "e7cea1574638db2f7e41799fe3855519ea57a47e9f6adc04f1a5a37e8aa526e0"
)
FROZEN_E37_ARTIFACT_SHA256 = (
    "04e524a5428c9b906e9fefe253f7ec66533bd6cb3452ea6d9afdb830e1a94b34"
)
FROZEN_E57_SOURCE_BANK_SHA256 = (
    "d3088e29e4c6179999ccb34088dae558fa402bf6b1455394acdc99cac4118463"
)
FROZEN_E57_SOURCE_BANK_ARRAY_SHA256 = (
    "f4fb2081b346c686e2d6930a03e3f17bb6c6d3eee4fcfc16984c1a9c1d8de4f5"
)
FROZEN_E57_ARTIFACT_SHA256 = (
    "b14efc1aad86ac67b5bf7c8631f02b2e68664e071b747b7b210d5f7a30f5d123"
)
E58_TORUS_NAMESPACE = "E58-held-out-torus-v1"
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
            identity = f"206:{frame.frame_id}:{raw}:{identifier}".encode("ascii")
            identity_hash = hashlib.sha256(identity).digest()
            retained = candidates[raw]
            if (
                len(retained) >= maximum_templates_per_class
                and identity_hash >= max(retained, key=lambda item: item[0])[0]
            ):
                continue
            try:
                template = NormalTemplateShape.from_source_frame(
                    frame,
                    raw_semantic_id=raw,
                    instance_id=identifier,
                )
            except RenderError:
                continue
            retained.append((identity_hash, template))
            if len(retained) > maximum_templates_per_class:
                retained.remove(max(retained, key=lambda item: item[0]))
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
    grounding_standard_lower_support_m: float = math.nan
    grounding_strict_lower_support_m: float = math.nan
    grounding_buried_fraction: float = math.nan

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


def _frame_trace_context(
    source: SourceFrame, ray_grid: RayGrid,
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
        directions_sensor, directions_world, origins_sensor,
        origins_world, native_range,
    )


def render_frame(
    source: SourceFrame,
    world: WorldSpec,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    _trace_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
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
        directions_sensor, directions_world, origins_sensor,
        origins_world, normal_range,
    ) = (
        _frame_trace_context(source, ray_grid)
        if _trace_context is None else _trace_context
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
        if _competition is None else _competition
    )
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
        raise PlacementError("normal train sequence contains no observed obstacle returns")
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
    shape: InsertShape, rotation: np.ndarray, translation: np.ndarray, margin_m: float,
    *, local_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = (
        _shape_outer_bounds(shape) if local_bounds is None else local_bounds
    )
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
    contact[2] = -(
        normal[0] * contact[0] + normal[1] * contact[1] + patch.offset
    ) / normal[2]
    rotation = _ground_rotation(normal, yaw_rad)
    lower_support = (
        shape.minimum_z_m(xy_resolution=33, z_steps=129)
        if lower_support_m is None else _finite_scalar(
            "lower_support_m", lower_support_m
        )
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
        proposed.shape, rotation, translation, threshold,
        local_bounds=local_bounds,
    )
    points, identities = obstacles.within_aabb(lower, upper)
    if points.size == 0:
        return False, math.inf, identities
    local = (points - translation) @ rotation
    if isinstance(proposed.shape, NormalTemplateShape):
        # A point is deeply inside only if every hull half-space is below the
        # threshold. Eliminated points cannot contain the global minimum once
        # any deep point survives, so later planes only evaluate survivors.
        local_lower, local_upper = proposed.shape.local_bounds()
        candidates = np.flatnonzero(np.all(
            (local >= local_lower) & (local <= local_upper), axis=1
        ))
        maximum = np.full(local.shape[0], -np.inf, dtype=np.float64)
        normals = proposed.shape.plane_normals
        offsets = proposed.shape.plane_offsets
        for start in range(0, normals.shape[0], 16):
            stop = min(start + 16, normals.shape[0])
            values = (
                local[candidates] @ normals[start:stop].T
                + offsets[start:stop]
            )
            candidate_maximum = np.maximum(
                maximum[candidates], np.max(values, axis=1)
            )
            keep = candidate_maximum < -threshold
            maximum[candidates] = candidate_maximum
            candidates = candidates[keep]
            if candidates.size == 0:
                break
        if candidates.size:
            minimum = float(np.min(maximum[candidates]))
            return True, minimum, identities
    distance = proposed.shape.signed_distance(local)
    minimum = float(np.min(distance))
    deep = distance < -threshold
    return bool(np.any(deep)), minimum, identities


def _fibonacci_surface_points(shape: InsertShape, count: int = 8192) -> np.ndarray:
    identifiers = np.arange(_integer("surface point count", count, minimum=32))
    if isinstance(shape, HeldOutTorusShape):
        # A torus is not star-shaped about its center, so center-directed rays
        # miss the hole.  Two irrational rotations give deterministic surface
        # witnesses without imposing the star-shaped assumption.
        major_angle = 2.0 * math.pi * np.mod(
            identifiers * ((math.sqrt(5.0) - 1.0) / 2.0), 1.0
        )
        tube_angle = 2.0 * math.pi * np.mod(
            (identifiers + 0.5) * (math.sqrt(2.0) - 1.0), 1.0
        )
        radial = shape.major_radius_m + shape.tube_radius_m * np.cos(tube_angle)
        points = np.column_stack(
            (
                radial * np.cos(major_angle),
                radial * np.sin(major_angle),
                shape.tube_radius_m * np.sin(tube_angle),
            )
        )
        residual = np.abs(shape.signed_distance(points))
        if not np.isfinite(points).all() or float(np.max(residual)) > 1.0e-12:
            raise PlacementError("deterministic torus points are not on the geometry")
        return points
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


def _pair_local_witnesses(item: ObjectSpec) -> np.ndarray:
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
    shape_seed: int | None = None,
    template_identity: str | None = None,
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
    pair_witness_cache: dict[int, np.ndarray] = {}
    local_bounds = _shape_outer_bounds(shape)
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
            object_id, label_value, shape_seed, template_identity,
            _integer("material_seed", material_seed), _integer("yaw_seed", yaw_seed),
            proposal, patch.pool_index, patch.frame_id, patch.slot, patch.semantic,
            tuple(proposal_pool_indices), tuple(rejections),
            tuple(proposal_minimum_sdf), minimum_sdf,
            0, () if shape_seed is None else (shape_seed,), (),
        )
        return proposed, replace(
            record,
            grounding_standard_lower_support_m=grounding.standard_lower_support_m,
            grounding_strict_lower_support_m=grounding.strict_lower_support_m,
            grounding_buried_fraction=grounding.buried_fraction,
        )
    raise PlacementExhaustion(
        proposal_pool_indices, rejections, proposal_minimum_sdf
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
    normal_template_library: Sequence[NormalTemplateShape],
    support_pool: QualifiedSupportPool,
    obstacles: ObservedObstacleIndex,
    world_type: WorldType,
    seed: int,
    *,
    control_context: CoverageControlContext,
    maximum_attempts: int = 48,
) -> tuple[WorldSpec, WorldGenerationReport]:
    """Build one immutable train/206 world through the sole qualified pipeline."""

    templates = tuple(normal_template_library)
    world_seed = _integer("seed", seed)
    if not isinstance(support_pool, QualifiedSupportPool) or not isinstance(
        obstacles, ObservedObstacleIndex
    ):
        raise TypeError("training world requires the qualified pool and obstacle index")
    if (
        not isinstance(control_context, CoverageControlContext)
        or control_context.support_pool is not support_pool
        or control_context.source_sequence_id != 206
    ):
        raise TypeError(
            "training world requires the bound train/206 coverage-control context"
        )
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
    template_index_by_identity = {
        _normal_template_identity(item): index
        for index, item in enumerate(templates)
    }
    if len(template_index_by_identity) != len(templates):
        raise RenderError("training normal-template identities must be unique")

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
                template_index: int | None = None
                template_seed: int | None = None
                scale_seed: int | None = None
                perturbation: float | None = None
                yaw_for_support: Callable[[SupportPatch], float] | None = None
                report: ShapeGenerationReport | None = None
                grounding: GroundingEligibility | None = None
                shape_proposals: tuple[int, ...] = ()
                grounding_rejections: tuple[int, ...] = ()
                proposal_rows: np.ndarray | None = None
                post_placement_rejection: (
                    Callable[[ObjectSpec, SupportPatch], str | None] | None
                ) = None
                if label == "normal-control":
                    template_seed = entity_seed + 1
                    scale_seed = entity_seed + 2
                    template_index = int(
                        np.random.default_rng(template_seed).integers(0, len(templates))
                    )
                    source = templates[template_index]
                    assigned_range_bin = _e25_new_assigned_range_bin(template_index)
                    target_scale = np.random.default_rng(
                        np.random.SeedSequence([scale_seed, 2501])
                    ).uniform(0.9, 1.1, size=3)
                    shape = _aligned_scaled_template(source, target_scale)
                    grounding = qualify_grounding(shape)
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
                    proposal_rows = _coverage_control_support_stream(
                        control_context,
                        template_index,
                        semantic,
                        assigned_range_bin,
                    )
                    if proposal_rows.size == 0:
                        raise PlacementError(
                            "assigned range bin has no semantically legal E21 support"
                        )

                    def support_yaw(
                        patch: SupportPatch, offset: float = perturbation
                    ) -> float:
                        return float(
                            control_context.trajectory_yaw_by_frame[patch.frame_id]
                        ) + offset

                    yaw_for_support = support_yaw

                    def reject_control(
                        proposed: ObjectSpec,
                        patch: SupportPatch,
                        target_bin: int = assigned_range_bin,
                    ) -> str | None:
                        observation = _coverage_control_observation(
                            control_context,
                            proposed,
                            patch,
                            world_seed,
                            target_bin,
                            (*objects, proposed),
                        )
                        if observation.visible_returns < 1:
                            return "no_visible_normal_control_return"
                        if observation.range_bin != target_bin:
                            return "assigned_visible_range_bin_mismatch"
                        return None

                    post_placement_rejection = reject_control
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
                    proposal_namespace=(
                        "E25-new-support-v1"
                        if label == "normal-control" else "training-world-v1"
                    ),
                    proposal_stream=(
                        template_index
                        if template_index is not None else entity_seed
                    ),
                    yaw_rad=yaw,
                    material_seed=material_seed, yaw_seed=yaw_seed,
                    shape_seed=shape_seed, template_identity=template_identity,
                    shape_generation_report=report, existing_objects=objects,
                    proposal_rows=proposal_rows,
                    grounding_eligibility=grounding,
                    yaw_for_support=yaw_for_support,
                    post_placement_rejection=post_placement_rejection,
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
            for item, record in zip(world.objects, records, strict=True):
                if item.label != "normal-control":
                    continue
                assert record.template_identity is not None
                template_index = template_index_by_identity[record.template_identity]
                assigned_range_bin = _e25_new_assigned_range_bin(template_index)
                support_row = int(np.searchsorted(
                    support_pool.pool_indices, record.support_pool_index
                ))
                if (
                    support_row >= support_pool.pool_indices.size
                    or int(support_pool.pool_indices[support_row])
                    != record.support_pool_index
                ):
                    raise RenderError(
                        "training control references an unknown support row"
                    )
                observation = _coverage_control_observation(
                    control_context,
                    item,
                    support_pool.patch(support_row),
                    world_seed,
                    assigned_range_bin,
                    world.objects,
                )
                if (
                    observation.visible_returns < 1
                    or observation.range_bin != assigned_range_bin
                ):
                    raise PlacementError(
                        "complete world invalidated a coverage-control observation"
                    )
        except PlacementError:
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
    if slots.size < 1:
        raise RenderError("low-level audit requires at least one selected return")
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


CANONICAL_NORMAL_TEMPLATE_LIBRARY_SHA256 = (
    "de5dfd765ac7d4fe4bb4644c40ecafdd80cdc31a3d0b6fc4fccd8e84a9fd906b"
)


def canonical_normal_template_library_identity(
    normal_template_library: Sequence[NormalTemplateShape],
) -> tuple[tuple[str, ...], dict[int, int], str]:
    """Require the exact ordered train/206 library qualified by E25-new."""

    templates = tuple(normal_template_library)
    if any(not isinstance(item, NormalTemplateShape) for item in templates):
        raise PlacementError("canonical normal-template library has invalid values")
    identities = tuple(_normal_template_identity(item) for item in templates)
    counts = {
        semantic: sum(item.raw_semantic_id == semantic for item in templates)
        for semantic in (10, 18, 20, 30)
    }
    library_hash = hashlib.sha256("".join(identities).encode()).hexdigest()
    if (
        len(templates) != 256
        or counts != {10: 64, 18: 64, 20: 64, 30: 64}
        or len(set(identities)) != 256
        or library_hash != CANONICAL_NORMAL_TEMPLATE_LIBRARY_SHA256
    ):
        raise PlacementError("canonical train/206 normal-template library changed")
    return identities, counts, library_hash


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


def _trajectory_yaw_by_pose(
    pose_by_frame: Mapping[int, np.ndarray],
) -> dict[int, float]:
    ordered = tuple(sorted((int(key), np.asarray(value)) for key, value in pose_by_frame.items()))
    if len(ordered) < 2:
        raise PlacementError("trajectory tangent requires at least two source frames")
    positions = np.asarray([pose[:3, 3] for _, pose in ordered])
    result: dict[int, float] = {}
    for index, (frame_id, pose) in enumerate(ordered):
        if index == 0:
            tangent = positions[1] - positions[0]
        elif index == len(ordered) - 1:
            tangent = positions[-1] - positions[-2]
        else:
            tangent = positions[index + 1] - positions[index - 1]
        horizontal = tangent[:2]
        if np.linalg.norm(horizontal) <= EPSILON:
            horizontal = pose[:2, 0]
        if np.linalg.norm(horizontal) <= EPSILON:
            raise PlacementError("trajectory tangent and pose fallback are degenerate")
        result[frame_id] = math.atan2(
            float(horizontal[1]), float(horizontal[0])
        )
    return result


def trajectory_yaw_by_frame(frames: Sequence[SourceFrame]) -> dict[int, float]:
    return _trajectory_yaw_by_pose({
        int(frame.frame_id): np.asarray(frame.lidar_pose, dtype=np.float64)
        for frame in frames
    })


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
    identities, counts, library_hash = canonical_normal_template_library_identity(
        templates
    )
    by_semantic: dict[int, tuple[NormalTemplateShape, ...]] = {}
    for semantic in sorted(NORMAL_TEMPLATE_SEMANTICS):
        selected = tuple(
            item for item in templates if item.raw_semantic_id == semantic
        )
        if selected:
            by_semantic[semantic] = selected
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
        completed == 1024 and hard_errors == 0 and placement_exhaustions == 0
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


_E25V2_SEQUENCE: object | None = None
_E25V2_SUPPORT_POOL: QualifiedSupportPool | None = None
_E25V2_OBSTACLES: ObservedObstacleIndex | None = None
_E25V2_TEMPLATES: dict[int, tuple[NormalTemplateShape, ...]] = {}
_E25V2_TRAJECTORY_YAW: dict[int, float] = {}
_E25V2_RAY_GRID: RayGrid | None = None
_E25V2_SENSOR: SensorCalibration | None = None
_E25V2_TARGETS: dict[str, np.ndarray] = {}
_E25V2_SUPPORT_ROWS: tuple[np.ndarray, ...] = ()
_E25V2_TARGET_COVARIATES = np.empty((0, 5), dtype=np.float64)
_E25V2_TARGET_LOOKUP: dict[tuple[int, int, int, int], np.ndarray] = {}
_E25V2_SUPPORT_PROPOSABLE = np.empty(0, dtype=np.bool_)
_E25V2_FRAME_CACHE: dict[
    int,
    tuple[
        SourceFrame,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
] = {}
_E25V2_UNIFORM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_E25V2_CACHE_LIMIT = 8
_E25V3_FRAME_OFFSETS = (0, -1, 1, -2, 2)
_E25V3_TARGETS: dict[str, np.ndarray] = {}
_E25V3_SUPPORT_ROWS: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
_E25V3_PLANE_NORMALS = np.empty((0, 3, 3), dtype=np.float64)
_E25V3_PLANE_OFFSETS = np.empty((0, 3), dtype=np.float64)
_E25V3_PLANE_RADIUS = np.empty(0, dtype=np.float64)
_E45_UNIT_FIELDS = (
    "bank_seed", "source", "center_frame", "frame_id", "support_semantic",
    "range_bin", "azimuth_sector", "median_distance_m", "median_beam", "Nvis",
    "O_hat", "local_density", "geometry_hits", "point_count", "point_features",
    "unit_hash",
)
_E25_REAL_TARGET_FIELDS = _E45_UNIT_FIELDS + (
    "real_semantic", "real_instance", "reference_support_pool_index",
)


@dataclass(frozen=True, slots=True)
class _SingleObjectSensorTrace:
    """Compact exact object trace expanded to file slots only after precheck."""

    candidate_slots: np.ndarray
    distance_m: np.ndarray
    native_range_m: np.ndarray
    normal_world: np.ndarray
    valid: np.ndarray
    in_range: np.ndarray
    accepted: np.ndarray


def _xy_hull_distance(
    points_xy: np.ndarray, polygon_xy: np.ndarray, equations: np.ndarray,
) -> np.ndarray:
    """Return the exact Euclidean distance from points to a closed XY hull."""
    points = np.asarray(points_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    inside = np.all(
        points @ equations[:, :2].T + equations[:, 2] <= EPSILON, axis=1
    )
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - starts
    denominator = np.sum(np.square(edges), axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(
        np.divide(
            np.sum(relative * edges[None, :, :], axis=2),
            denominator[None, :],
            out=np.zeros((points.shape[0], edges.shape[0]), dtype=np.float64),
            where=denominator[None, :] > 0.0,
        ),
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + fraction[:, :, None] * edges[None, :, :]
    boundary_distance = np.min(
        np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1
    )
    return np.where(inside, 0.0, boundary_distance)


def _e25v3_support_filter_frame(
    task: tuple[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Find an E21 plane within its verified local range for each real object."""
    frame_id, target_indices = task
    sequence, pool = _E25V2_SEQUENCE, _E25V2_SUPPORT_POOL
    if (
        sequence is None or pool is None or not _E25V3_TARGETS
        or _E25V3_PLANE_NORMALS.shape[0] != pool.frames.size
    ):
        raise RuntimeError("E25-v3 target qualification is not initialized")
    frame = sequence.source_frame(frame_id)
    assert frame.labels is not None
    rotation, translation = _pose(frame)
    count = target_indices.size
    candidate_count = np.zeros(
        (count, len(_E25V3_FRAME_OFFSETS)), dtype=np.int32
    )
    local_candidate_count = np.zeros_like(candidate_count)
    nearest_distance = np.full_like(candidate_count, np.inf, dtype=np.float64)
    nearest_local_distance = np.full_like(candidate_count, np.inf, dtype=np.float64)
    compatible = np.zeros(count, dtype=np.bool_)
    rejection_code = np.ones(count, dtype=np.uint8)
    evaluated = np.zeros(count, dtype=np.int32)
    projection_rejections = np.zeros(count, dtype=np.int32)
    burial_rejections = np.zeros(count, dtype=np.int32)
    selected_row = np.full(count, -1, dtype=np.int64)
    selected_offset = np.full(count, 127, dtype=np.int8)
    selected_semantic = np.zeros(count, dtype=np.uint16)
    selected_anchor_distance = np.full(count, np.nan, dtype=np.float64)
    selected_radius = np.full(count, np.nan, dtype=np.float64)
    selected_normal = np.full((count, 3), np.nan, dtype=np.float64)
    selected_offset_value = np.full(count, np.nan, dtype=np.float64)
    projection_height_difference = np.full(count, np.nan, dtype=np.float64)
    buried_fraction = np.full(count, np.nan, dtype=np.float64)
    signed_height_summary = np.full((count, 6), np.nan, dtype=np.float64)
    lower_gap_over_visible_extent = np.full(count, np.nan, dtype=np.float64)
    anchor_distance_over_radius = np.full(count, np.nan, dtype=np.float64)
    plane_slope_deg = np.full(count, np.nan, dtype=np.float64)
    observed_points = np.empty(count, dtype=np.int32)
    for output_index, target_index in enumerate(target_indices):
        semantic = int(_E25V3_TARGETS["real_semantic"][target_index])
        instance = int(_E25V3_TARGETS["real_instance"][target_index])
        selected = (
            (frame.labels.semantic == np.uint16(semantic))
            & (frame.labels.instance == np.uint16(instance))
            & ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        )
        sensor_points = np.asarray(frame.xyzi[selected, :3], dtype=np.float64)
        if sensor_points.shape[0] < 4:
            raise RenderError("E25-v3 target has fewer than four observed points")
        world_points = sensor_points @ rotation.T + translation
        try:
            hull = ConvexHull(world_points[:, :2])
        except QhullError as error:
            raise RenderError("E25-v3 target has a degenerate XY hull") from error
        polygon = world_points[np.asarray(hull.vertices), :2]
        equations = np.asarray(hull.equations, dtype=np.float64)
        observed_points[output_index] = sensor_points.shape[0]
        policy = tuple(sorted(normal_control_support_semantics(semantic)))
        ordered_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for offset_index, offset in enumerate(_E25V3_FRAME_OFFSETS):
            rows = _E25V3_SUPPORT_ROWS.get(
                (frame_id + offset, policy), np.empty(0, dtype=np.int64)
            )
            candidate_count[output_index, offset_index] = rows.size
            if rows.size == 0:
                ordered_candidates.append(
                    (offset, np.empty(0, dtype=np.int64), np.empty(0))
                )
                continue
            distance = _xy_hull_distance(
                pool.anchors_world_m[rows, :2], polygon, equations,
            )
            order = np.lexsort((pool.selection_hashes[rows], distance))
            nearest_distance[output_index, offset_index] = distance[order[0]]
            ordered_rows = rows[order]
            ordered_distance = distance[order]
            local = (
                ordered_distance
                <= 1.25 * _E25V3_PLANE_RADIUS[ordered_rows]
            )
            local_candidate_count[output_index, offset_index] = int(
                np.count_nonzero(local)
            )
            if bool(local.any()):
                nearest_local_distance[output_index, offset_index] = float(
                    ordered_distance[local][0]
                )
            ordered_candidates.append(
                (offset, ordered_rows[local], ordered_distance[local])
            )
        for offset, ordered_rows, ordered_distance in ordered_candidates:
            for start in range(0, ordered_rows.size, 64):
                stop = min(start + 64, ordered_rows.size)
                batch_rows = ordered_rows[start:stop]
                normals = _E25V3_PLANE_NORMALS[batch_rows]
                offsets = _E25V3_PLANE_OFFSETS[batch_rows]
                small_height = -(
                    polygon @ normals[:, 0, :2].T + offsets[:, 0]
                ) / normals[:, 0, 2]
                large_height = -(
                    polygon @ normals[:, 2, :2].T + offsets[:, 2]
                ) / normals[:, 2, 2]
                height_difference = np.max(
                    np.abs(small_height - large_height), axis=0
                )
                stable = height_difference <= 0.08
                signed = (
                    world_points @ normals[:, 1, :].T + offsets[:, 1]
                )
                fraction = np.mean(signed < -0.02, axis=0)
                accepted = stable & (fraction <= 0.02)
                if bool(accepted.any()):
                    local = int(np.flatnonzero(accepted)[0])
                    prefix = local + 1
                    evaluated[output_index] += prefix
                    projection_rejections[output_index] += int(
                        np.count_nonzero(~stable[:prefix])
                    )
                    burial_rejections[output_index] += int(
                        np.count_nonzero(stable[:prefix] & (fraction[:prefix] > 0.02))
                    )
                    row = int(batch_rows[local])
                    current = signed[:, local]
                    quantiles = np.quantile(current, (0.0, 0.02, 0.05, 0.5, 1.0))
                    visible_extent = float(quantiles[-1] - quantiles[0])
                    compatible[output_index] = True
                    rejection_code[output_index] = 0
                    selected_row[output_index] = row
                    selected_offset[output_index] = offset
                    selected_semantic[output_index] = pool.semantics[row]
                    selected_anchor_distance[output_index] = ordered_distance[start + local]
                    selected_radius[output_index] = _E25V3_PLANE_RADIUS[row]
                    selected_normal[output_index] = normals[local, 1]
                    selected_offset_value[output_index] = offsets[local, 1]
                    projection_height_difference[output_index] = height_difference[local]
                    buried_fraction[output_index] = fraction[local]
                    signed_height_summary[output_index] = (
                        quantiles[0], quantiles[1], quantiles[2], quantiles[3],
                        quantiles[4], visible_extent,
                    )
                    lower_gap_over_visible_extent[output_index] = (
                        quantiles[0] / visible_extent
                        if visible_extent > EPSILON else np.inf
                    )
                    anchor_distance_over_radius[output_index] = (
                        ordered_distance[start + local] / _E25V3_PLANE_RADIUS[row]
                    )
                    plane_slope_deg[output_index] = math.degrees(
                        math.acos(float(np.clip(normals[local, 1, 2], -1.0, 1.0)))
                    )
                    break
                evaluated[output_index] += batch_rows.size
                projection_rejections[output_index] += int(np.count_nonzero(~stable))
                burial_rejections[output_index] += int(
                    np.count_nonzero(stable & (fraction > 0.02))
                )
            if compatible[output_index]:
                break
        if not compatible[output_index]:
            available = int(candidate_count[output_index].sum())
            local_available = int(local_candidate_count[output_index].sum())
            rejection_code[output_index] = (
                1 if available == 0
                else 2 if local_available == 0
                else 3 if burial_rejections[output_index] == 0
                else 4
            )
    return {
        "target_index": target_indices.astype(np.int64),
        "observed_point_count": observed_points,
        "support_candidate_count_by_offset": candidate_count,
        "local_support_candidate_count_by_offset": local_candidate_count,
        "nearest_support_distance_m_by_offset": nearest_distance,
        "nearest_local_support_distance_m_by_offset": nearest_local_distance,
        "compatible": compatible,
        "rejection_code": rejection_code,
        "evaluated_support_candidates": evaluated,
        "projection_stability_rejections": projection_rejections,
        "visible_burial_rejections": burial_rejections,
        "selected_support_row": selected_row,
        "selected_frame_offset": selected_offset,
        "selected_support_semantic": selected_semantic,
        "selected_anchor_distance_m": selected_anchor_distance,
        "selected_central_radius_m": selected_radius,
        "selected_plane_normal": selected_normal,
        "selected_plane_offset": selected_offset_value,
        "projection_height_difference_m": projection_height_difference,
        "visible_buried_fraction": buried_fraction,
        "signed_height_summary_m": signed_height_summary,
        "lower_gap_over_visible_extent": lower_gap_over_visible_extent,
        "anchor_distance_over_central_radius": anchor_distance_over_radius,
        "plane_slope_deg": plane_slope_deg,
    }


def _finite_quantiles(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"finite": 0, "nonfinite": int(np.asarray(values).size)}
    probabilities = np.asarray(
        (0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    )
    quantiles = np.quantile(finite, probabilities)
    names = (
        "minimum", "q05", "q25", "median", "q75", "q90", "q95", "q99",
        "maximum",
    )
    return {
        "finite": int(finite.size),
        "nonfinite": int(np.asarray(values).size - finite.size),
        **{name: float(value) for name, value in zip(names, quantiles, strict=True)},
    }


def run_e25_v3_target_qualification(
    data_root: Path | str, support_pool_path: Path | str,
    target_bank_path: Path | str, output_path: Path | str, *, processes: int = 24,
) -> dict[str, object]:
    """Qualify real targets against the finite E21-v4 local plane domain."""
    if not 1 <= processes <= 24:
        raise PlacementError("E25-v3 target qualification processes must be in [1,24]")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    target_path = Path(target_bank_path).expanduser().resolve(strict=True)
    required = {
        "frame_id", "real_semantic", "real_instance", "range_bin", "O_hat",
        "Nvis", "unit_hash", "metadata_json",
    }
    with np.load(target_path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise PlacementError("E25-v3 input target bank is missing required arrays")
        target_metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            target_metadata.get("experiment") != "E25-v2-real-normal-target-bank"
            or target_metadata.get("source_sequence") != "train/206"
        ):
            raise PlacementError("E25-v3 diagnostic requires the retained train/206 bank")
        targets = {
            name: np.asarray(payload[name]).copy()
            for name in required if name != "metadata_json"
        }
    count = int(targets["frame_id"].size)
    if count < 1 or any(value.shape[0] != count for value in targets.values()):
        raise PlacementError("E25-v3 target arrays are not aligned")
    if np.unique(targets["unit_hash"]).size != count:
        raise PlacementError("E25-v3 target identities are not unique")
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    pool = load_qualified_support_pool(support_pool_path)
    support_path = Path(support_pool_path).expanduser().resolve(strict=True)
    with np.load(support_path, allow_pickle=False) as payload:
        required_plane_fields = {
            "qualified", "normals", "offsets", "central_radius_m",
            "median_residual_m", "q95_residual_m", "normal_angle_deg",
            "height_difference_m",
        }
        if not required_plane_fields.issubset(payload.files):
            raise PlacementError("E25-v3 support pool lacks multiscale plane arrays")
        raw_rows = np.asarray(pool.pool_indices, dtype=np.int64)
        qualified = np.asarray(payload["qualified"], dtype=np.bool_)
        if not bool(np.all(qualified[raw_rows])):
            raise PlacementError("E25-v3 plane arrays include an unqualified patch")
        plane_normals = np.asarray(payload["normals"], dtype=np.float64)[raw_rows]
        plane_offsets = np.asarray(payload["offsets"], dtype=np.float64)[raw_rows]
        plane_radius = np.asarray(
            payload["central_radius_m"], dtype=np.float64
        )[raw_rows]
        median_residual = np.asarray(
            payload["median_residual_m"], dtype=np.float64
        )[raw_rows, 1]
        q95_residual = np.asarray(
            payload["q95_residual_m"], dtype=np.float64
        )[raw_rows, 1]
        normal_angle = np.asarray(
            payload["normal_angle_deg"], dtype=np.float64
        )[raw_rows]
        height_difference = np.asarray(
            payload["height_difference_m"], dtype=np.float64
        )[raw_rows]
    if (
        plane_normals.shape != (pool.frames.size, 3, 3)
        or plane_offsets.shape != (pool.frames.size, 3)
        or plane_radius.shape != (pool.frames.size,)
        or not np.isfinite(plane_normals).all()
        or not np.isfinite(plane_offsets).all()
        or not np.isfinite(plane_radius).all()
        or not np.allclose(
            plane_radius, np.clip(pool.ranges_m / 20.0, 1.0, 3.0),
            atol=1.0e-12, rtol=1.0e-12,
        )
        or np.any(plane_normals[:, :, 2] <= 0.0)
        or not np.allclose(
            np.linalg.norm(plane_normals, axis=2), 1.0,
            atol=1.0e-7, rtol=1.0e-7,
        )
        or np.any(median_residual > 0.03)
        or np.any(q95_residual > 0.08)
        or np.any(normal_angle > 5.0)
        or np.any(height_difference > 0.08)
    ):
        raise PlacementError("E25-v3 support pool violates E21-v4 qualification")
    policies = {
        tuple(sorted(normal_control_support_semantics(int(semantic))))
        for semantic in np.unique(targets["real_semantic"])
    }
    relevant_frames = {
        int(frame) + offset
        for frame in np.unique(targets["frame_id"])
        for offset in _E25V3_FRAME_OFFSETS
    }
    by_frame_semantic: dict[tuple[int, int], np.ndarray] = {}
    for support_semantic in sorted({item for policy in policies for item in policy}):
        rows = np.flatnonzero(pool.semantics == support_semantic)
        order = np.argsort(pool.frames[rows], kind="stable")
        rows = rows[order]
        frames = pool.frames[rows]
        unique_frames, starts = np.unique(frames, return_index=True)
        stops = np.r_[starts[1:], rows.size]
        for frame_id, start, stop in zip(unique_frames, starts, stops, strict=True):
            if int(frame_id) in relevant_frames:
                by_frame_semantic[(int(frame_id), support_semantic)] = rows[start:stop]
    support_rows: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    for frame_id in relevant_frames:
        for policy in policies:
            parts = [
                by_frame_semantic[(frame_id, semantic)]
                for semantic in policy if (frame_id, semantic) in by_frame_semantic
            ]
            support_rows[(frame_id, policy)] = (
                np.concatenate(parts).astype(np.int64)
                if parts else np.empty(0, dtype=np.int64)
            )
    global _E25V2_SEQUENCE, _E25V2_SUPPORT_POOL, _E25V3_TARGETS, _E25V3_SUPPORT_ROWS
    global _E25V3_PLANE_NORMALS, _E25V3_PLANE_OFFSETS, _E25V3_PLANE_RADIUS
    _E25V2_SEQUENCE, _E25V2_SUPPORT_POOL = sequence, pool
    _E25V3_TARGETS, _E25V3_SUPPORT_ROWS = targets, support_rows
    _E25V3_PLANE_NORMALS = np.ascontiguousarray(plane_normals)
    _E25V3_PLANE_OFFSETS = np.ascontiguousarray(plane_offsets)
    _E25V3_PLANE_RADIUS = np.ascontiguousarray(plane_radius)
    tasks = [
        (int(frame_id), np.flatnonzero(targets["frame_id"] == frame_id))
        for frame_id in np.unique(targets["frame_id"])
    ]
    started = time.monotonic()
    if processes == 1:
        chunks = [_e25v3_support_filter_frame(task) for task in tasks]
    else:
        with mp.get_context("fork").Pool(
            processes=processes, maxtasksperchild=24
        ) as workers:
            chunks = workers.map(
                _e25v3_support_filter_frame, tasks, chunksize=1
            )
    diagnostic_names = tuple(name for name in chunks[0] if name != "target_index")
    target_index = np.concatenate([chunk["target_index"] for chunk in chunks])
    order = np.argsort(target_index, kind="stable")
    if not np.array_equal(target_index[order], np.arange(count, dtype=np.int64)):
        raise PlacementError("E25-v3 qualification did not cover every target exactly once")
    diagnostic = {
        name: np.ascontiguousarray(
            np.concatenate([chunk[name] for chunk in chunks], axis=0)[order]
        )
        for name in diagnostic_names
    }
    scientific = {
        "frame_id": np.ascontiguousarray(targets["frame_id"]),
        "real_semantic": np.ascontiguousarray(targets["real_semantic"]),
        "real_instance": np.ascontiguousarray(targets["real_instance"]),
        "range_bin": np.ascontiguousarray(targets["range_bin"]),
        "occlusion": np.ascontiguousarray(targets["O_hat"]),
        "occlusion_layer": np.searchsorted(
            np.asarray((0.25, 0.75)), targets["O_hat"], side="right"
        ).astype(np.uint8),
        "visible_return_count": np.ascontiguousarray(targets["Nvis"]),
        "unit_hash": np.ascontiguousarray(targets["unit_hash"]),
        **diagnostic,
    }
    compatible = scientific["compatible"]
    class_names = {
        10: "car_10", 18: "truck_18", 20: "other_vehicle_20", 30: "person_30"
    }
    class_summary: dict[str, object] = {}
    for semantic in np.unique(scientific["real_semantic"]):
        mask = scientific["real_semantic"] == semantic
        accepted = mask & compatible
        class_summary[class_names.get(int(semantic), str(int(semantic)))] = {
            "targets": int(np.count_nonzero(mask)),
            "retained": int(np.count_nonzero(accepted)),
            "rejected": int(np.count_nonzero(mask & ~compatible)),
            "retained_range_count": np.bincount(
                scientific["range_bin"][accepted], minlength=5
            )[:5].tolist(),
            "retained_occlusion_layer_count": np.bincount(
                scientific["occlusion_layer"][accepted], minlength=3
            ).tolist(),
            "selected_anchor_distance_m": _finite_quantiles(
                scientific["selected_anchor_distance_m"][accepted]
            ),
            "projection_height_difference_m": _finite_quantiles(
                scientific["projection_height_difference_m"][accepted]
            ),
            "visible_buried_fraction": _finite_quantiles(
                scientific["visible_buried_fraction"][accepted]
            ),
            "minimum_visible_signed_height_m": _finite_quantiles(
                scientific["signed_height_summary_m"][accepted, 0]
            ),
            "lower_gap_over_visible_extent": _finite_quantiles(
                scientific["lower_gap_over_visible_extent"][accepted]
            ),
            "lower_gap_exceeds_visible_extent": int(np.count_nonzero(
                scientific["lower_gap_over_visible_extent"][accepted] > 1.0
            )),
        }
    accepted_distance = scientific["selected_anchor_distance_m"][compatible]
    accepted_gap_ratio = scientific["lower_gap_over_visible_extent"][compatible]
    rejection_names = (
        "accepted", "no_semantically_legal_patch",
        "outside_e21_local_validity", "no_projection_stable_patch",
        "visible_geometry_incompatible",
    )
    rejection_count = np.bincount(
        scientific["rejection_code"], minlength=len(rejection_names)
    )
    retained_range = np.bincount(
        scientific["range_bin"][compatible], minlength=5
    )[:5]
    retained_occlusion = np.bincount(
        scientific["occlusion_layer"][compatible], minlength=3
    )[:3]
    retained_frames = int(np.unique(scientific["frame_id"][compatible]).size)
    retained_instances = int(np.unique(np.column_stack((
        scientific["real_semantic"][compatible],
        scientific["real_instance"][compatible],
    )), axis=0).shape[0])
    retained_classes = {
        int(value) for value in scientific["real_semantic"][compatible]
    }
    required_classes = {10, 18, 20, 30}
    passed = (
        retained_classes == required_classes
        and bool(np.all(retained_range > 0))
        and bool(np.all(retained_occlusion > 0))
        and retained_frames >= 100
        and retained_instances >= 32
    )
    result: dict[str, object] = {
        "experiment": "E25-v3-target-support-qualification",
        "status": "qualification_complete",
        "passed": passed,
        "source_sequence": "train/206",
        "target_units": count,
        "frame_offsets_in_search_order": list(_E25V3_FRAME_OFFSETS),
        "candidate_order": "first frame offset, then exact anchor-to-closed-XY-hull distance, then E21-v4 selection hash",
        "compatibility_rule": {
            "source_patch": "E21-v4 qualified=true only; all original residual, normal and multiscale limits revalidated",
            "local_validity": "exact anchor-to-target-frame closed-XY-hull distance <=1.25*R(d), where the saved E21-v4 central_radius_m is R(d)=clip(d/20,1,3) m",
            "extrapolation_stability": "maximum absolute small-vs-large predicted ground-height difference over all target-frame XY-hull vertices <=0.08 m",
            "visible_geometry": "fraction of target-frame observed object returns more than 0.02 m below the central plane <=0.02",
            "lower_visible_gap_upper_gate": None,
        },
        "legal_support_semantics_unchanged": True,
        "old_reference_support_fields_used": False,
        "generator_executed": False,
        "train_201_accessed": False,
        "calipers_evaluated_or_changed": False,
        "Dxy_alpha_route_abandoned": True,
        "target_bank_input_role": "identity and real-observation covariates only; retained E25-v2 support bindings ignored",
        "coverage_requirements": {
            "active_classes": [10, 18, 20, 30],
            "all_five_range_bins_nonempty": True,
            "all_three_occlusion_layers_nonempty": True,
            "minimum_center_frames": 100,
            "minimum_real_instances": 32,
        },
        "retained": int(np.count_nonzero(compatible)),
        "rejected": int(np.count_nonzero(~compatible)),
        "rejection_count": {
            name: int(value)
            for name, value in zip(rejection_names, rejection_count, strict=True)
        },
        "class_summary": class_summary,
        "retained_center_frames": retained_frames,
        "retained_real_instances": retained_instances,
        "retained_range_count": retained_range.tolist(),
        "retained_occlusion_layer_count": retained_occlusion.tolist(),
        "selected_frame_offset_count": [
            int(np.count_nonzero(
                compatible & (scientific["selected_frame_offset"] == offset)
            ))
            for offset in _E25V3_FRAME_OFFSETS
        ],
        "selected_support_semantic_count": {
            str(semantic): int(np.count_nonzero(
                compatible
                & (scientific["selected_support_semantic"] == semantic)
            ))
            for semantic in (40, 48)
        },
        "support_candidates_available": int(np.sum(
            scientific["support_candidate_count_by_offset"], dtype=np.int64
        )),
        "local_support_candidates_available": int(np.sum(
            scientific["local_support_candidate_count_by_offset"], dtype=np.int64
        )),
        "support_candidates_evaluated": int(np.sum(
            scientific["evaluated_support_candidates"], dtype=np.int64
        )),
        "projection_stability_rejections": int(np.sum(
            scientific["projection_stability_rejections"], dtype=np.int64
        )),
        "visible_burial_rejections": int(np.sum(
            scientific["visible_burial_rejections"], dtype=np.int64
        )),
        "selected_anchor_distance_m": _finite_quantiles(accepted_distance),
        "selected_anchor_distance_over_central_radius": _finite_quantiles(
            scientific["anchor_distance_over_central_radius"][compatible]
        ),
        "selected_local_validity_margin_m": _finite_quantiles(
            1.25 * scientific["selected_central_radius_m"][compatible]
            - accepted_distance
        ),
        "projection_height_difference_m": _finite_quantiles(
            scientific["projection_height_difference_m"][compatible]
        ),
        "visible_buried_fraction": _finite_quantiles(
            scientific["visible_buried_fraction"][compatible]
        ),
        "minimum_visible_signed_height_m": _finite_quantiles(
            scientific["signed_height_summary_m"][compatible, 0]
        ),
        "lower_gap_over_visible_extent": _finite_quantiles(accepted_gap_ratio),
        "lower_gap_exceeds_visible_extent": int(np.count_nonzero(
            accepted_gap_ratio > 1.0
        )),
        "plane_slope_deg": _finite_quantiles(
            scientific["plane_slope_deg"][compatible]
        ),
        "run_seconds": float(time.monotonic() - started),
        "target_bank_sha256": _sha256_path(target_path),
        "support_pool_sha256": _sha256_path(support_path),
        "scientific_array_hash": _scientific_array_hash(scientific),
        "claim_limit": "PASS establishes only that the retained train/206 real targets have a legal E21-v4 patch inside its already-verified local radius, stable projected height, and no substantial cutting of visible object returns; it does not recover an unobserved tyre or foot contact point.",
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _load_e25_real_target_bank(
    path: Path | str, expected_experiment: str,
) -> tuple[Path, dict[str, np.ndarray], dict[str, object]]:
    """Load one immutable real-target bank and verify its scientific payload."""
    target_path = Path(path).expanduser().resolve(strict=True)
    required = set(_E25_REAL_TARGET_FIELDS) | {"metadata_json"}
    with np.load(target_path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise PlacementError("E25 real-target bank is missing required arrays")
        metadata = json.loads(str(payload["metadata_json"].item()))
        targets = {
            name: np.ascontiguousarray(np.asarray(payload[name]))
            for name in _E25_REAL_TARGET_FIELDS
        }
    count = int(targets["frame_id"].size)
    if (
        metadata.get("experiment") != expected_experiment
        or metadata.get("source_sequence") != "train/206"
        or not bool(metadata.get("passed"))
        or count < 1
        or any(value.shape[0] != count for value in targets.values())
        or int(metadata.get("target_units", -1)) != count
        or np.unique(targets["unit_hash"]).size != count
        or metadata.get("scientific_array_hash") != _scientific_array_hash(targets)
    ):
        raise PlacementError("E25 real-target bank failed identity or hash validation")
    return target_path, targets, metadata


def build_e25_v3_target_bank(
    source_target_bank_path: Path | str,
    target_qualification_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Materialize only E25-v3-qualified identities with their selected support."""
    source_path, source, source_metadata = _load_e25_real_target_bank(
        source_target_bank_path, "E25-v2-real-normal-target-bank"
    )
    qualification_path = Path(target_qualification_path).expanduser().resolve(
        strict=True
    )
    identity_fields = {
        "frame_id": "frame_id", "real_semantic": "real_semantic",
        "real_instance": "real_instance", "range_bin": "range_bin",
        "O_hat": "occlusion", "Nvis": "visible_return_count",
        "unit_hash": "unit_hash",
    }
    required = set(identity_fields.values()) | {
        "compatible", "rejection_code", "selected_support_row",
        "selected_support_semantic",
        "metadata_json",
    }
    with np.load(qualification_path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise PlacementError("E25-v3 qualification is missing required arrays")
        qualification_metadata = json.loads(str(payload["metadata_json"].item()))
        qualification_arrays = {
            name: np.ascontiguousarray(np.asarray(payload[name]))
            for name in payload.files if name != "metadata_json"
        }
    count = int(source["frame_id"].size)
    if (
        qualification_metadata.get("experiment")
        != "E25-v3-target-support-qualification"
        or not bool(qualification_metadata.get("passed"))
        or qualification_metadata.get("target_bank_sha256")
        != _sha256_path(source_path)
        or qualification_metadata.get("scientific_array_hash")
        != _scientific_array_hash(qualification_arrays)
        or any(
            qualification_arrays[qualification_name].shape[0] != count
            or not np.array_equal(
                source[source_name], qualification_arrays[qualification_name]
            )
            for source_name, qualification_name in identity_fields.items()
        )
    ):
        raise PlacementError("E25-v3 qualification does not match its source bank")
    compatible = np.asarray(qualification_arrays["compatible"], dtype=np.bool_)
    selected_rows = np.asarray(
        qualification_arrays["selected_support_row"], dtype=np.int64
    )
    selected_semantics = np.asarray(
        qualification_arrays["selected_support_semantic"], dtype=np.uint16
    )
    legal_semantics = np.asarray([
        int(selected_semantics[index]) in normal_control_support_semantics(
            int(source["real_semantic"][index])
        )
        for index in np.flatnonzero(compatible)
    ], dtype=np.bool_)
    if (
        int(np.count_nonzero(compatible))
        != int(qualification_metadata.get("retained", -1))
        or not np.array_equal(
            compatible, qualification_arrays["rejection_code"] == 0
        )
        or np.any(selected_rows[compatible] < 0)
        or np.any(selected_semantics[compatible] == 0)
        or not bool(np.all(legal_semantics))
        or np.any(selected_rows[~compatible] != -1)
        or np.any(selected_semantics[~compatible] != 0)
    ):
        raise PlacementError("E25-v3 qualification selection sentinels are invalid")
    targets = {
        name: np.ascontiguousarray(values[compatible])
        for name, values in source.items()
    }
    # Old E25-v2 support bindings are deliberately replaced, not repaired in place.
    targets["support_semantic"] = np.ascontiguousarray(
        selected_semantics[compatible]
    )
    targets["reference_support_pool_index"] = np.ascontiguousarray(
        selected_rows[compatible]
    )
    target_count = int(targets["frame_id"].size)
    center_frames = int(np.unique(targets["frame_id"]).size)
    real_instances = int(np.unique(np.column_stack((
        targets["real_semantic"], targets["real_instance"]
    )), axis=0).shape[0])
    range_count = np.bincount(targets["range_bin"], minlength=5)[:5]
    occlusion_count = np.bincount(np.searchsorted(
        np.asarray((0.25, 0.75)), targets["O_hat"], side="right"
    ), minlength=3)[:3]
    class_count = {
        str(int(semantic)): int(np.count_nonzero(
            targets["real_semantic"] == semantic
        ))
        for semantic in np.unique(targets["real_semantic"])
    }
    passed = (
        target_count == int(qualification_metadata["retained"])
        and set(map(int, np.unique(targets["real_semantic"]))) == {10, 18, 20, 30}
        and bool(np.all(range_count > 0)) and bool(np.all(occlusion_count > 0))
        and center_frames >= 100 and real_instances >= 32
    )
    result: dict[str, object] = {
        "experiment": "E25-v3-real-normal-target-bank",
        "passed": passed,
        "source_sequence": "train/206",
        "target_units": target_count,
        "center_frames": center_frames,
        "real_instances": real_instances,
        "class_count": class_count,
        "range_count": range_count.tolist(),
        "occlusion_layer_count": occlusion_count.tolist(),
        "support_semantic_count": {
            str(semantic): int(np.count_nonzero(
                targets["support_semantic"] == semantic
            ))
            for semantic in (40, 48)
        },
        "support_binding": "selected E25-v3 E21-v4 local-compatible support row",
        "real_observation_numeric_covariates_changed": False,
        "support_semantic_replaced": True,
        "old_e25_v2_support_bindings_used": False,
        "formal_repetitions": 1,
        "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "source_target_bank_sha256": _sha256_path(source_path),
        "source_target_bank_scientific_array_hash": source_metadata[
            "scientific_array_hash"
        ],
        "target_qualification_sha256": _sha256_path(qualification_path),
        "target_qualification_scientific_array_hash": qualification_metadata[
            "scientific_array_hash"
        ],
        "support_pool_sha256": qualification_metadata["support_pool_sha256"],
        "scientific_array_hash": _scientific_array_hash(targets),
        "claim_limit": "The bank contains only train/206 real-observation targets with a selected E21-v4 patch inside its verified local domain; it does not assert per-class full range or occlusion coverage.",
    }
    if not passed:
        raise PlacementError("E25-v3 target-bank build lost frozen aggregate coverage")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **targets,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e25v2_support_streams(
    sequence: object, pool: QualifiedSupportPool,
    targets: Mapping[str, np.ndarray], maximum_proposals: int = 128,
) -> tuple[np.ndarray, ...]:
    """Order legal supports around the reference object's observed environment."""
    cache: dict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    streams: list[np.ndarray] = []
    for target in range(targets["frame_id"].size):
        frame_id = int(targets["frame_id"][target])
        if frame_id not in cache:
            rows = np.flatnonzero(np.abs(pool.frames - frame_id) <= 2)
            frame = sequence.source_frame(frame_id)
            rotation, translation = _pose(frame)
            sensor = (pool.anchors_world_m[rows] - translation) @ rotation
            ranges = np.linalg.norm(sensor, axis=1)
            angles = np.arctan2(sensor[:, 1], sensor[:, 0]) % (2.0 * math.pi)
            sectors = (
                np.floor(angles / (math.pi / 4.0)).astype(np.int8) % 8
            )
            valid = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
            native_points = np.asarray(frame.xyzi[valid, :3], dtype=np.float64)
            neighbours = min(9, native_points.shape[0])
            distance, _ = cKDTree(native_points).query(
                sensor, k=neighbours, workers=1
            )
            radius = distance[:, -1] if distance.ndim == 2 else distance
            density = max(neighbours - 1, 1) / (
                (4.0 / 3.0) * math.pi * np.maximum(radius, 1.0e-3) ** 3
            )
            cache[frame_id] = rows, ranges, angles, sectors, density
        rows, ranges, angles, sectors, density = cache[frame_id]
        eligible = (
            (pool.semantics[rows] == targets["support_semantic"][target])
            & (_gate1_range_bin(ranges) == targets["range_bin"][target])
            & (sectors == targets["azimuth_sector"][target])
        )
        selected = rows[eligible]
        reference_row = int(targets["reference_support_pool_index"][target])
        environment_distance = np.linalg.norm(
            pool.anchors_world_m[selected, :2]
            - pool.anchors_world_m[reference_row, :2], axis=1,
        )
        range_error = np.abs(
            ranges[eligible] - float(targets["median_distance_m"][target])
        )
        density_targets = np.sort(np.log1p(targets["local_density"][
            (targets["real_semantic"] == targets["real_semantic"][target])
            & (targets["support_semantic"] == targets["support_semantic"][target])
            & (targets["range_bin"] == targets["range_bin"][target])
            & (targets["azimuth_sector"] == targets["azimuth_sector"][target])
        ]))
        proposal_density = np.log1p(density[eligible])
        insertion = np.searchsorted(density_targets, proposal_density)
        lower = density_targets[np.maximum(insertion - 1, 0)]
        upper = density_targets[np.minimum(insertion, density_targets.size - 1)]
        density_error = np.minimum(
            np.abs(proposal_density - lower), np.abs(proposal_density - upper)
        )
        target_angle = (
            (float(targets["azimuth_sector"][target]) + 0.5) * math.pi / 4.0
        )
        angle_error = np.abs(
            (angles[eligible] - target_angle + math.pi) % (2.0 * math.pi) - math.pi
        )
        order = np.lexsort(
            (
                pool.selection_hashes[selected], angle_error, range_error,
                environment_distance, density_error,
            )
        )
        streams.append(selected[order[:maximum_proposals]].astype(np.int64))
    return tuple(streams)


def _initialize_e25v2_target_index(
    targets: Mapping[str, np.ndarray], support_rows: tuple[np.ndarray, ...]
) -> None:
    """Cache immutable target strata and covariates for all worker proposals."""
    global _E25V2_TARGET_COVARIATES, _E25V2_TARGET_LOOKUP
    global _E25V2_SUPPORT_PROPOSABLE
    _E25V2_TARGET_COVARIATES = _e45_covariates(targets)
    _E25V2_SUPPORT_PROPOSABLE = np.asarray(
        [rows.size > 0 for rows in support_rows], dtype=np.bool_
    )
    lookup: dict[tuple[int, int, int, int], list[int]] = {}
    for index in range(targets["frame_id"].size):
        key = (
            int(targets["real_semantic"][index]),
            int(targets["support_semantic"][index]),
            int(targets["range_bin"][index]),
            int(targets["azimuth_sector"][index]),
        )
        lookup.setdefault(key, []).append(index)
    _E25V2_TARGET_LOOKUP = {
        key: np.asarray(rows, dtype=np.int64) for key, rows in lookup.items()
    }


def _e25v2_frame_context(
    frame_id: int, control_seed: int, object_id: int,
) -> tuple[
    SourceFrame,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]:
    """Reuse immutable frame rays and identity uniforms within one worker."""
    sequence, grid = _E25V2_SEQUENCE, _E25V2_RAY_GRID
    if sequence is None or grid is None:
        raise RuntimeError("E25-v2 frame context is not initialized")
    cached = _E25V2_FRAME_CACHE.pop(frame_id, None)
    if cached is None:
        frame = sequence.source_frame(frame_id)
        pose_rotation, lidar_origin = _pose(frame)
        directions_sensor = grid.directions_for(frame)
        directions_world = directions_sensor @ pose_rotation.T
        origins_sensor = grid.origins_for(frame)
        origins_world = origins_sensor @ pose_rotation.T + lidar_origin
        native_range = np.asarray(grid.ranges(frame)).copy()
        native_range[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = np.inf
        trace_context = (
            directions_sensor, directions_world, origins_sensor,
            origins_world, native_range,
        )
        cached = frame, trace_context
    _E25V2_FRAME_CACHE[frame_id] = cached
    while len(_E25V2_FRAME_CACHE) > _E25V2_CACHE_LIMIT:
        _E25V2_FRAME_CACHE.pop(next(iter(_E25V2_FRAME_CACHE)))
    uniform_key = control_seed, frame_id, object_id
    uniform = _E25V2_UNIFORM_CACHE.pop(uniform_key, None)
    if uniform is None:
        slots = np.arange(grid.slot_count, dtype=np.int32)
        uniform = _slot_uniform(
            WorldSpec(control_seed, 206), frame_id, slots,
            np.full(grid.slot_count, object_id, dtype=np.int32), channel=0,
        )
    _E25V2_UNIFORM_CACHE[uniform_key] = uniform
    while len(_E25V2_UNIFORM_CACHE) > _E25V2_CACHE_LIMIT:
        _E25V2_UNIFORM_CACHE.pop(next(iter(_E25V2_UNIFORM_CACHE)))
    return cached[0], cached[1], uniform


def _single_object_sensor_precheck(
    frame: SourceFrame, world: WorldSpec, ray_grid: RayGrid,
    sensor: SensorCalibration,
    trace_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    uniform: np.ndarray | None = None,
    candidate_slots_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, _SingleObjectSensorTrace]:
    """Compute exact pre-packing covariates without repeating a full render."""
    item = world.objects[0]
    if candidate_slots_hint is None:
        hinted_slots = np.arange(ray_grid.slot_count, dtype=np.int32)
    else:
        hinted_slots = np.asarray(candidate_slots_hint, dtype=np.int32)
        if (
            hinted_slots.ndim != 1
            or np.any((hinted_slots < 0) | (hinted_slots >= ray_grid.slot_count))
            or np.unique(hinted_slots).size != hinted_slots.size
        ):
            raise RenderError("candidate slot hint must contain unique valid slots")
    if trace_context is None:
        pose_rotation, lidar_origin = _pose(frame)
        directions_sensor = ray_grid.directions_for(frame)
        origins_sensor = ray_grid.origins_for(frame)
        hinted_direction_world = (
            directions_sensor[hinted_slots] @ pose_rotation.T
        )
        hinted_origins_world = (
            origins_sensor[hinted_slots] @ pose_rotation.T + lidar_origin
        )
        hinted_native_range = np.sum(
            (
                np.asarray(frame.xyzi[hinted_slots, :3], dtype=np.float64)
                - origins_sensor[hinted_slots]
            )
            * directions_sensor[hinted_slots],
            axis=1,
        )
        hinted_native_range[
            np.asarray(frame.zero_slot_mask, dtype=np.bool_)[hinted_slots]
        ] = np.inf
    else:
        (
            directions_sensor, directions_world, origins_sensor,
            origins_world, native_range,
        ) = trace_context
        hinted_direction_world = directions_world[hinted_slots]
        hinted_origins_world = origins_world[hinted_slots]
        hinted_native_range = native_range[hinted_slots]
    object_rotation = np.asarray(item.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(item.translation_world_m, dtype=np.float64)
    relative_origin = hinted_origins_world - translation
    unit_world_direction = hinted_direction_world / np.linalg.norm(
        hinted_direction_world, axis=1, keepdims=True
    )
    closest_parameter = np.maximum(
        -np.sum(relative_origin * unit_world_direction, axis=1), 0.0
    )
    closest = relative_origin + closest_parameter[:, None] * unit_world_direction
    # World-space distance is rotation invariant; the margin keeps this a strict
    # superset of the previous local-space broad phase under float64 roundoff.
    broad_phase = (
        np.linalg.norm(closest, axis=1) <= item.shape.bound_radius_m + 1.0e-6
    )
    candidate_slots = hinted_slots[broad_phase]
    candidate_native_range = hinted_native_range[broad_phase]
    if candidate_slots.size:
        local_origin = relative_origin[broad_phase] @ object_rotation
        local_direction = hinted_direction_world[broad_phase] @ object_rotation
        distance, local_normal, valid = item.shape.intersect(
            local_origin, local_direction
        )
    else:
        distance = np.empty(0, dtype=np.float64)
        local_normal = np.empty((0, 3), dtype=np.float64)
        valid = np.empty(0, dtype=np.bool_)
    official_distance = distance + ray_grid.official_range_offset_m
    in_range = (official_distance >= 2.5) & (official_distance <= 50.0)
    geometry = np.asarray(valid) & in_range
    normal_world = local_normal @ object_rotation.T
    incidence = np.zeros(candidate_slots.size, dtype=np.float64)
    incidence[valid] = np.arccos(np.clip(np.abs(np.sum(
        normal_world[valid]
        * -hinted_direction_world[broad_phase][valid],
        axis=1,
    )), 0.0, 1.0))
    chance = np.zeros(candidate_slots.size, dtype=np.float64)
    candidate_beam_ids = candidate_slots // ray_grid.columns
    chance[valid] = sensor.return_chance(
        candidate_beam_ids[valid], distance[valid], incidence[valid],
        item.material.return_bias,
    )
    if uniform is None:
        candidate_uniform = _slot_uniform(
            world,
            int(frame.frame_id),
            candidate_slots,
            np.full(candidate_slots.size, item.object_id, dtype=np.int32),
            channel=0,
        )
    else:
        uniform_array = np.asarray(uniform, dtype=np.float64)
        if uniform_array.shape == (ray_grid.slot_count,):
            candidate_uniform = uniform_array[candidate_slots]
        elif uniform_array.shape == (candidate_slots.size,):
            candidate_uniform = uniform_array
        else:
            raise RenderError("precomputed uniform values have an invalid shape")
    accepted_raw = valid & (candidate_uniform < chance)
    accepted = accepted_raw & in_range
    visible = accepted & (
        distance < candidate_native_range - world.tie_tolerance_m
    )
    returned_slots = candidate_slots[visible]
    if returned_slots.size == 0:
        exact = np.asarray((-1, -1, -1), dtype=np.int16)
        covariates = np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float64)
    else:
        angle = np.arctan2(
            directions_sensor[returned_slots, 1], directions_sensor[returned_slots, 0]
        )
        circular = math.atan2(float(np.sin(angle).sum()), float(np.cos(angle).sum()))
        exact = np.asarray((
            int(_gate1_range_bin(np.asarray([
                np.median(official_distance[visible])
            ]))[0]),
            int(math.floor((circular % (2.0 * math.pi)) / (math.pi / 4.0))) % 8,
            int(returned_slots.size),
        ), dtype=np.int16)
        covariates = np.asarray((
            float(np.median(official_distance[visible])),
            float(np.median(candidate_beam_ids[visible])),
            float(np.log1p(returned_slots.size)),
            float(1.0 - returned_slots.size / max(int(np.count_nonzero(geometry)), 1)),
        ), dtype=np.float64)
    trace = _SingleObjectSensorTrace(
        _freeze(candidate_slots), _freeze(distance),
        _freeze(candidate_native_range), _freeze(normal_world),
        _freeze(valid), _freeze(in_range), _freeze(accepted_raw),
    )
    return exact, covariates, trace


def _expand_single_object_trace(
    trace: _SingleObjectSensorTrace, item: ObjectSpec, slot_count: int,
) -> tuple[np.ndarray, _ObjectCompetition]:
    """Expand one compact exact trace only for a final full renderer check."""
    geometry = np.zeros(slot_count, dtype=np.bool_)
    geometry[trace.candidate_slots] = trace.valid & trace.in_range
    accepted_slots = trace.candidate_slots[trace.accepted]
    competition_distance = np.full(slot_count, np.inf, dtype=np.float64)
    competition_normal = np.zeros((slot_count, 3), dtype=np.float64)
    competition_object = np.full(slot_count, -1, dtype=np.int32)
    competition_distance[accepted_slots] = trace.distance_m[trace.accepted]
    competition_normal[accepted_slots] = trace.normal_world[trace.accepted]
    competition_object[accepted_slots] = item.object_id
    competition = _ObjectCompetition(
        _freeze(competition_distance), _freeze(competition_normal),
        _freeze(competition_object),
        {item.object_id: int(np.count_nonzero(trace.valid))},
        {item.object_id: int(np.count_nonzero(trace.accepted))},
    )
    return geometry, competition


def _e25v2_target_attempt(
    index: int, source: NormalTemplateShape, template_identity: str,
    shape: NormalTemplateShape, scale: np.ndarray, perturbation: float,
    material_seed: int, material: MaterialSpec,
    grounding: GroundingEligibility, target: int,
    placement_cache: dict[int, tuple[ObjectSpec, PlacementRecord] | None],
    sensor_cache: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[int, int, int] | None],
    ],
) -> dict[str, object]:
    sequence, pool, obstacles = (
        _E25V2_SEQUENCE, _E25V2_SUPPORT_POOL, _E25V2_OBSTACLES
    )
    grid, sensor = _E25V2_RAY_GRID, _E25V2_SENSOR
    if (
        sequence is None or pool is None or obstacles is None
        or grid is None or sensor is None or not _E25V2_TARGETS
        or not _E25V2_SUPPORT_ROWS or not _E25V2_TEMPLATES
    ):
        raise RuntimeError("E25-v2 worker state is not initialized")
    control_seed = 2_500_000 + index
    semantic = int(source.raw_semantic_id)
    frame_id = int(_E25V2_TARGETS["frame_id"][target])
    frame, trace_context, uniform = _e25v2_frame_context(
        frame_id, control_seed, index + 1
    )
    condition_rejections = 0
    placement_rejections = 0
    last_difference = np.full(5, np.nan, dtype=np.float64)
    violation_counts = np.zeros(5, dtype=np.int16)
    exact_mismatch_counts = np.zeros(3, dtype=np.int16)
    best_scaled_difference = np.full(5, np.inf, dtype=np.float64)
    calipers = np.asarray((2.0, 4.0, 0.25, 0.10, 0.25))
    def yaw_for_support(patch: SupportPatch) -> float:
        return _E25V2_TRAJECTORY_YAW[patch.frame_id] + perturbation

    for proposal, row in enumerate(_E25V2_SUPPORT_ROWS[target][:128]):
        support_row = int(row)
        if support_row not in placement_cache:
            try:
                placement_cache[support_row] = place_object(
                    shape, material, pool, obstacles,
                    object_id=index + 1, label="normal-control",
                    proposal_namespace="E25-v2-observation-conditioned",
                    proposal_stream=control_seed, yaw_rad=perturbation,
                    material_seed=material_seed, yaw_seed=control_seed,
                    template_identity=template_identity,
                    proposal_rows=(support_row,), maximum_candidates=1,
                    grounding_eligibility=grounding,
                    yaw_for_support=yaw_for_support,
                )
            except PlacementError:
                placement_cache[support_row] = None
        placement = placement_cache[support_row]
        if placement is None:
            placement_rejections += 1
            continue
        item, record = placement
        world = WorldSpec(control_seed, 206, (item,))
        sensor_key = frame_id, support_row
        sensor_value = sensor_cache.get(sensor_key)
        trace: _SingleObjectSensorTrace | None = None
        if sensor_value is None:
            (
                precheck_exact, precheck_covariates,
                trace,
            ) = _single_object_sensor_precheck(
                frame, world, grid, sensor, trace_context, uniform
            )
            control_covariates = None
            control_summary = None
        else:
            (
                precheck_exact, precheck_covariates,
                control_covariates, control_summary,
            ) = sensor_value
        comparable = _E25V2_TARGET_LOOKUP.get(
            (semantic, record.support_semantic,
             int(precheck_exact[0]), int(precheck_exact[1])),
            np.empty(0, dtype=np.int64),
        )
        comparable_covariates = _E25V2_TARGET_COVARIATES[comparable]
        precheck_differences = np.abs(
            comparable_covariates[:, :4] - precheck_covariates
        )
        precheck_pass = np.all(precheck_differences <= calipers[:4], axis=1)
        precheck_difference = (
            np.min(precheck_differences, axis=0)
            if comparable.size else np.full(4, np.inf, dtype=np.float64)
        )
        exact_mismatch_counts += np.asarray((
            precheck_exact[2] < 1,
            not bool(np.any(_E25V2_TARGETS["range_bin"][comparable]
                            == precheck_exact[0])),
            comparable.size == 0,
        ), dtype=np.int16)
        violation_counts[:4] += (precheck_difference > calipers[:4]).astype(np.int16)
        best_scaled_difference[:4] = np.minimum(
            best_scaled_difference[:4], precheck_difference / calipers[:4]
        )
        if (
            precheck_exact[2] < 1
            or not bool(precheck_pass.any())
        ):
            if sensor_value is None:
                sensor_cache[sensor_key] = (
                    precheck_exact, precheck_covariates, None, None
                )
            condition_rejections += 1
            continue
        if control_covariates is None or control_summary is None:
            assert trace is not None
            precheck_geometry, competition = _expand_single_object_trace(
                trace, item, grid.slot_count
            )
            rendered = render_frame(
                frame, world, grid, sensor,
                _trace_context=trace_context, _competition=competition,
            )
            returned = np.asarray(rendered.normal_control_mask, dtype=np.bool_)
            official = np.asarray(grid.official_ranges(rendered.source))
            returned = returned & (official >= 2.5) & (official <= 50.0)
            unit = _e45_unit_record(
                frame, grid, 1, control_seed, frame_id, semantic,
                int(_E25V2_TARGETS["real_instance"][target]),
                int(_E25V2_TARGETS["support_semantic"][target]),
                precheck_geometry, returned, rendered.source,
            )
            control_covariates = _e45_covariates({
                name: np.asarray(unit[name]).reshape(1)
                for name in (
                    "median_distance_m", "median_beam", "Nvis", "O_hat",
                    "local_density",
                )
            })[0]
            control_summary = (
                int(unit["point_count"]), int(unit["support_semantic"]),
                int(unit["range_bin"]), int(unit["azimuth_sector"]),
            )
            sensor_cache[sensor_key] = (
                precheck_exact, precheck_covariates,
                control_covariates, control_summary,
            )
        feasible_targets = comparable[precheck_pass]
        feasible_covariates = comparable_covariates[precheck_pass]
        feasible_differences = np.abs(feasible_covariates - control_covariates)
        fully_matched = np.all(feasible_differences <= calipers, axis=1)
        if bool(fully_matched.any()):
            candidates = np.flatnonzero(fully_matched)
            cost = np.sum(
                np.square(feasible_differences[candidates] / calipers), axis=1
            )
            selected = candidates[np.lexsort((
                _E25V2_TARGETS["unit_hash"][feasible_targets[candidates]], cost
            ))[0]]
            matched_target = int(feasible_targets[selected])
            last_difference = feasible_differences[selected]
        else:
            matched_target = target
            last_difference = np.min(feasible_differences, axis=0)
        best_scaled_difference[4] = min(
            best_scaled_difference[4], last_difference[4] / calipers[4]
        )
        point_count, unit_support, unit_range, unit_sector = control_summary
        exact = bool(fully_matched.any()) and (
            point_count > 0
            and unit_support
            == int(_E25V2_TARGETS["support_semantic"][target])
            and unit_range
            == int(_E25V2_TARGETS["range_bin"][matched_target])
            and unit_sector
            == int(_E25V2_TARGETS["azimuth_sector"][matched_target])
        )
        violation_counts[4] += np.int16(last_difference[4] > calipers[4])
        if not exact or not np.all(last_difference <= calipers):
            condition_rejections += 1
            continue
        return {
            "hard_error": 0, "placement_exhaustion": 0,
            "control_seed": control_seed, "target_index": matched_target,
            "target_frame": int(_E25V2_TARGETS["frame_id"][matched_target]),
            "target_semantic": semantic,
            "target_instance": int(_E25V2_TARGETS["real_instance"][matched_target]),
            "target_unit_hash": int(_E25V2_TARGETS["unit_hash"][matched_target]),
            "target_range_bin": int(_E25V2_TARGETS["range_bin"][matched_target]),
            "target_occlusion": float(_E25V2_TARGETS["O_hat"][matched_target]),
            "control_occlusion": float(control_covariates[3]),
            "condition_difference": last_difference,
            "support_proposal_count": proposal + 1,
            "placement_rejections": placement_rejections,
            "condition_rejections": condition_rejections,
            "violation_counts": violation_counts,
            "exact_mismatch_counts": exact_mismatch_counts,
            "best_scaled_difference": best_scaled_difference,
            "template_identity": template_identity,
            "scale": scale, "pose_perturbation_rad": perturbation,
            "support_semantic": record.support_semantic,
            "support_pool_index": record.support_pool_index,
            "object_json": json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")),
            "placement_report_json": json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")),
        }
    return {
        "hard_error": 0, "placement_exhaustion": 1,
        "control_seed": control_seed, "target_index": target,
        "target_frame": frame_id, "target_semantic": semantic,
        "target_instance": int(_E25V2_TARGETS["real_instance"][target]),
        "target_unit_hash": int(_E25V2_TARGETS["unit_hash"][target]),
        "target_range_bin": int(_E25V2_TARGETS["range_bin"][target]),
        "target_occlusion": float(_E25V2_TARGETS["O_hat"][target]),
        "condition_difference": last_difference,
        "violation_counts": violation_counts,
        "exact_mismatch_counts": exact_mismatch_counts,
        "best_scaled_difference": best_scaled_difference,
        "support_proposal_count": min(128, len(_E25V2_SUPPORT_ROWS[target])),
        "placement_rejections": placement_rejections,
        "condition_rejections": condition_rejections,
    }


def _e25v2_worker(index: int) -> dict[str, object]:
    if not _E25V2_TARGETS or not _E25V2_SUPPORT_ROWS or not _E25V2_TEMPLATES:
        raise RuntimeError("E25-v2 worker state is not initialized")
    control_seed = 2_500_000 + index
    templates = tuple(
        template for semantic in sorted(_E25V2_TEMPLATES)
        for template in _E25V2_TEMPLATES[semantic]
    )
    if len(templates) != 256 or not 0 <= index < len(templates):
        raise PlacementError("E25-v2 requires one fixture per frozen template")
    source = templates[index]
    template_identity = _normal_template_identity(source)
    semantic = int(source.raw_semantic_id)
    scale = np.random.default_rng(
        np.random.SeedSequence([control_seed + 2, 2501])
    ).uniform(0.9, 1.1, size=3)
    shape = _aligned_scaled_template(source, scale)
    limit = math.pi if semantic == 30 else math.radians(15.0)
    perturbation = float(np.random.default_rng(
        np.random.SeedSequence([control_seed, 2502])
    ).uniform(-limit, limit))
    material_seed = control_seed + 2503
    material = MaterialSpec.sample(material_seed)
    grounding = qualify_grounding(shape)
    placement_cache: dict[int, tuple[ObjectSpec, PlacementRecord] | None] = {}
    sensor_cache: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[int, int, int] | None],
    ] = {}
    eligible = np.flatnonzero(
        (_E25V2_TARGETS["real_semantic"] == semantic)
        & _E25V2_SUPPORT_PROPOSABLE
    )
    if eligible.size == 0:
        raise PlacementError(
            "E25-v2 has no support-proposable train/206 target for template semantic"
        )
    order = np.lexsort((
        _E25V2_TARGETS["unit_hash"][eligible],
        np.abs(
            _E25V2_TARGETS["frame_id"][eligible].astype(np.int64)
            - source.source_frame_id
        ),
        _E25V2_TARGETS["real_instance"][eligible] != source.source_instance_id,
    ))
    aggregate_placement = 0
    aggregate_condition = 0
    aggregate_violation = np.zeros(5, dtype=np.int32)
    aggregate_exact = np.zeros(3, dtype=np.int32)
    aggregate_best = np.full(5, np.inf, dtype=np.float64)
    last: dict[str, object] | None = None
    for target_proposal, target in enumerate(eligible[order[:128]]):
        attempt = _e25v2_target_attempt(
            index, source, template_identity, shape, scale, perturbation,
            material_seed, material, grounding, int(target),
            placement_cache, sensor_cache,
        )
        aggregate_placement += int(attempt["placement_rejections"])
        aggregate_condition += int(attempt["condition_rejections"])
        aggregate_violation += np.asarray(attempt["violation_counts"])
        aggregate_exact += np.asarray(attempt["exact_mismatch_counts"])
        aggregate_best = np.minimum(
            aggregate_best, np.asarray(attempt["best_scaled_difference"])
        )
        last = attempt
        if int(attempt["placement_exhaustion"]) == 0:
            attempt.update({
                "target_proposal_count": target_proposal + 1,
                "support_proposal_count": (
                    aggregate_placement + aggregate_condition + 1
                ),
                "placement_rejections": aggregate_placement,
                "condition_rejections": aggregate_condition,
                "violation_counts": aggregate_violation,
                "exact_mismatch_counts": aggregate_exact,
                "best_scaled_difference": aggregate_best,
            })
            return attempt
    assert last is not None
    last.update({
        "target_proposal_count": min(128, eligible.size),
        "support_proposal_count": aggregate_placement + aggregate_condition,
        "placement_rejections": aggregate_placement,
        "condition_rejections": aggregate_condition,
        "violation_counts": aggregate_violation,
        "exact_mismatch_counts": aggregate_exact,
        "best_scaled_difference": aggregate_best,
    })
    return last


def _e25v2_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)
    return {
        "control_seed": values("control_seed", np.int64, -1),
        "target_index": values("target_index", np.int32, -1),
        "target_frame": values("target_frame", np.int16, -1),
        "target_semantic": values("target_semantic", np.uint16, 0),
        "target_instance": values("target_instance", np.uint16, 0),
        "target_unit_hash": values("target_unit_hash", np.uint64, 0),
        "target_range_bin": values("target_range_bin", np.int8, -1),
        "target_occlusion": values("target_occlusion", np.float64, np.nan),
        "control_occlusion": values("control_occlusion", np.float64, np.nan),
        "condition_difference": np.asarray([
            item.get("condition_difference", (np.nan,) * 5) for item in records
        ], dtype=np.float64),
        "violation_counts": np.asarray([
            item.get("violation_counts", (0,) * 5) for item in records
        ], dtype=np.int16),
        "exact_mismatch_counts": np.asarray([
            item.get("exact_mismatch_counts", (0,) * 3) for item in records
        ], dtype=np.int16),
        "best_scaled_difference": np.asarray([
            item.get("best_scaled_difference", (np.inf,) * 5) for item in records
        ], dtype=np.float64),
        "support_proposal_count": values("support_proposal_count", np.int16, 0),
        "target_proposal_count": values("target_proposal_count", np.int16, 0),
        "placement_rejections": values("placement_rejections", np.int16, 0),
        "condition_rejections": values("condition_rejections", np.int16, 0),
        "template_identity": values("template_identity", "S64", ""),
        "scale": np.asarray([item.get("scale", (np.nan,) * 3) for item in records]),
        "pose_perturbation_rad": values("pose_perturbation_rad", np.float64, np.nan),
        "support_semantic": values("support_semantic", np.uint16, 0),
        "support_pool_index": values("support_pool_index", np.int64, -1),
        "object_json": np.asarray([str(item.get("object_json", "")).encode() for item in records]),
        "placement_report_json": np.asarray([
            str(item.get("placement_report_json", "")).encode() for item in records
        ]),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "placement_exhaustion_code": values("placement_exhaustion", np.uint8, 0),
    }


def _e25v2_work_order() -> tuple[int, ...]:
    """Start worst-case templates first so deterministic long tails overlap."""
    templates = tuple(
        template for semantic in sorted(_E25V2_TEMPLATES)
        for template in _E25V2_TEMPLATES[semantic]
    )
    costs: list[int] = []
    for index, source in enumerate(templates):
        semantic = int(source.raw_semantic_id)
        eligible = np.flatnonzero(
            (_E25V2_TARGETS["real_semantic"] == semantic)
            & _E25V2_SUPPORT_PROPOSABLE
        )
        order = np.lexsort((
            _E25V2_TARGETS["unit_hash"][eligible],
            np.abs(
                _E25V2_TARGETS["frame_id"][eligible].astype(np.int64)
                - source.source_frame_id
            ),
            _E25V2_TARGETS["real_instance"][eligible]
            != source.source_instance_id,
        ))
        proposals = sum(
            min(128, _E25V2_SUPPORT_ROWS[int(target)].size)
            for target in eligible[order[:128]]
        )
        costs.append(proposals * int(source.plane_normals.shape[0]))
    return tuple(sorted(range(len(templates)), key=lambda i: (-costs[i], i)))


def run_e25_v3_normal_control_qualification(
    data_root: Path | str, support_pool_path: Path | str,
    calibration_path: Path | str, target_bank_path: Path | str,
    output_path: Path | str, *, processes: int = 12,
) -> dict[str, object]:
    """Qualify formal controls conditioned on the E25-v3 train/206 target bank."""
    if processes != 12:
        raise PlacementError("formal E25-v3 requires exactly 12 control workers")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    templates = extract_normal_template_library(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    by_semantic = {
        semantic: tuple(item for item in templates if item.raw_semantic_id == semantic)
        for semantic in sorted({item.raw_semantic_id for item in templates})
    }
    pool = load_qualified_support_pool(support_pool_path)
    grid, sensor = load_sensor_calibration(calibration_path)
    target_path, targets, target_metadata = _load_e25_real_target_bank(
        target_bank_path, "E25-v3-real-normal-target-bank"
    )
    support_path = Path(support_pool_path).expanduser().resolve(strict=True)
    reference_rows = np.asarray(
        targets["reference_support_pool_index"], dtype=np.int64
    )
    if (
        target_metadata.get("support_pool_sha256") != _sha256_path(support_path)
        or np.any(reference_rows < 0)
        or np.any(reference_rows >= pool.frames.size)
    ):
        raise PlacementError("E25-v3 target support bindings do not match E21-v4")
    frame_offsets = pool.frames[reference_rows] - targets["frame_id"]
    if (
        not np.array_equal(
            pool.semantics[reference_rows], targets["support_semantic"]
        )
        or not bool(np.all(np.isin(frame_offsets, _E25V3_FRAME_OFFSETS)))
        or any(
            int(targets["support_semantic"][index])
            not in normal_control_support_semantics(
                int(targets["real_semantic"][index])
            )
            for index in range(targets["frame_id"].size)
        )
    ):
        raise PlacementError("E25-v3 target support bindings do not match E21-v4")
    sequence._frames.clear()
    gc.collect()
    obstacles = collect_observed_obstacle_index(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    frame_ids = tuple(map(int, sequence.frame_ids))
    positions = np.asarray([
        _pose(sequence.source_frame(frame_id))[1] for frame_id in frame_ids
    ])
    fallback_axes = np.asarray([
        _pose(sequence.source_frame(frame_id))[0][:2, 0] for frame_id in frame_ids
    ])
    trajectory_yaws: dict[int, float] = {}
    for index, frame_id in enumerate(frame_ids):
        if index == 0:
            tangent = positions[1] - positions[0]
        elif index == len(frame_ids) - 1:
            tangent = positions[-1] - positions[-2]
        else:
            tangent = positions[index + 1] - positions[index - 1]
        horizontal = tangent[:2]
        if np.linalg.norm(horizontal) <= EPSILON:
            horizontal = fallback_axes[index]
        if np.linalg.norm(horizontal) <= EPSILON:
            raise PlacementError("trajectory tangent and pose fallback are degenerate")
        trajectory_yaws[frame_id] = math.atan2(
            float(horizontal[1]), float(horizontal[0])
        )
    global _E25_TEMPLATES, _E25V2_SEQUENCE, _E25V2_SUPPORT_POOL
    global _E25V2_RAY_GRID, _E25V2_TEMPLATES, _E25V2_OBSTACLES
    global _E25V2_TRAJECTORY_YAW, _E25V2_SENSOR, _E25V2_TARGETS
    global _E25V2_SUPPORT_ROWS
    _E25V2_SEQUENCE = sequence
    _E25V2_SUPPORT_POOL = pool
    _E25V2_RAY_GRID = grid
    _E25V2_TEMPLATES = by_semantic
    _E25V2_OBSTACLES = obstacles
    _E25_TEMPLATES = by_semantic
    _E25V2_TRAJECTORY_YAW = trajectory_yaws
    _E25V2_SENSOR = sensor
    _E25V2_TARGETS = targets
    _E25V2_SUPPORT_ROWS = _e25v2_support_streams(sequence, pool, targets)
    _initialize_e25v2_target_index(targets, _E25V2_SUPPORT_ROWS)
    work_order = _e25v2_work_order()
    sequence._cache_frames = 1
    sequence._frames.clear()
    gc.collect()
    started = time.monotonic()
    with mp.get_context("fork").Pool(
        processes=processes, maxtasksperchild=16
    ) as workers:
        scheduled = workers.map(_e25v2_worker, work_order, chunksize=1)
    by_index = {int(item["control_seed"]) - 2_500_000: item for item in scheduled}
    records = [by_index[index] for index in range(256)]
    run_seconds = [time.monotonic() - started]
    first = _e25v2_arrays(records)
    completed_mask = (
        (first["template_identity"] != b"")
        & (first["placement_exhaustion_code"] == 0)
        & (first["hard_error_code"] == 0)
    )
    completed = int(np.count_nonzero(completed_mask))
    selected_range = np.bincount(
        first["target_range_bin"][completed_mask], minlength=5
    )[:5]
    selected_occlusion = np.bincount(np.searchsorted(
        np.asarray((0.25, 0.75)), first["target_occlusion"][completed_mask], side="right"
    ), minlength=3)
    selected_instances = int(np.unique(np.column_stack((
        first["target_semantic"][completed_mask],
        first["target_instance"][completed_mask],
    )), axis=0).shape[0])
    center_frames = int(np.unique(first["target_frame"][completed_mask]).size)
    hard_errors = int(np.sum(first["hard_error_code"]))
    exhaustions = int(np.sum(first["placement_exhaustion_code"]))
    condition_errors = int(np.count_nonzero(
        np.any(first["condition_difference"][completed_mask]
               > np.asarray((2.0, 4.0, 0.25, 0.10, 0.25)), axis=1)
    ))
    unique_templates = int(np.unique(
        first["template_identity"][completed_mask]
    ).size)
    passed = (
        completed == 256 and hard_errors == 0
        and exhaustions == 0 and condition_errors == 0
        and unique_templates == 256
        and center_frames >= 100 and selected_instances >= 32
        and bool(np.all(selected_range > 0)) and bool(np.all(selected_occlusion > 0))
    )
    result = {
        "experiment": "E25-v3-normal-control", "passed": passed,
        "failure_classification": None if passed else "local_support_conditioned_control_generation_failure",
        "templates": len(templates), "target_bank": target_metadata,
        "placements": 256, "completed": completed,
        "hard_errors": hard_errors, "placement_exhaustions": exhaustions,
        "condition_errors": condition_errors, "center_frames": center_frames,
        "selected_real_instances": selected_instances,
        "selected_range_count": selected_range.tolist(),
        "selected_occlusion_layer_count": selected_occlusion.tolist(),
        "target_proposals": int(np.sum(first["target_proposal_count"])),
        "support_proposals": int(np.sum(first["support_proposal_count"])),
        "placement_rejections": int(np.sum(first["placement_rejections"])),
        "condition_rejections": int(np.sum(first["condition_rejections"])),
        "unique_templates": unique_templates,
        "formal_repetitions": 1,
        "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "run_seconds": run_seconds,
        "scientific_array_hash": _scientific_array_hash(first),
        "target_bank_sha256": _sha256_path(target_path),
        "support_pool_sha256": _sha256_path(support_path),
        "target_qualification_sha256": target_metadata[
            "target_qualification_sha256"
        ],
        "formal_training_distribution_changed": True,
        "changed_component": "normal-control target and support-position selection only",
        "unchanged_components": [
            "256 real train/206 normal templates", "0.9-1.1 scale stream",
            "class semantics and pose stream", "material stream",
            "E22 grounding", "E23 observed collision", "E24 pair collision",
            "renderer", "return probability", "intensity",
            "E45A covariates and calipers", "schema 7 anomaly-proxy",
        ],
        "target_support_contract": "E25-v3 E21-v4 local validity plus projected stability plus visible-geometry compatibility",
        "claim_limit": "Qualification applies to the retained train/206 aggregate target domain; person has only four retained targets in one range and occlusion stratum.",
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E25_NEW_OBSTACLES: ObservedObstacleIndex | None = None
_E25_NEW_TEMPLATES: tuple[NormalTemplateShape, ...] = ()
_E25_NEW_CONTROL_CONTEXT: CoverageControlContext | None = None
_E25_NEW_NAMESPACE = int.from_bytes(
    hashlib.sha256(b"E25-new-support-v1").digest()[:8], "little"
)


def _e25_new_assigned_range_bin(fixture_index: int) -> int:
    """Assign every canonical template once by the frozen index cycle."""

    index = _integer("fixture_index", fixture_index)
    if index >= 256:
        raise PlacementError("E25-new fixture index must lie in [0,255]")
    return index % 5


@dataclass(frozen=True, slots=True)
class _E25NewObservation:
    frame_id: int
    visible_returns: int
    visible_in_range_returns: int
    accepted_in_range_returns: int
    geometry_in_range_hits: int
    median_official_range_m: float
    median_beam: float
    range_bin: int
    azimuth_sector: int
    occlusion: float


@dataclass(frozen=True, slots=True)
class _E25NewFrameContext:
    frame: SourceFrame
    trace: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(slots=True)
class CoverageControlContext:
    """Immutable production inputs plus bounded worker-local acceleration caches."""

    frames_by_id: dict[int, SourceFrame]
    support_pool: QualifiedSupportPool
    ray_grid: RayGrid
    sensor: SensorCalibration
    trajectory_yaw_by_frame: dict[int, float]
    support_rows: dict[tuple[int, int], np.ndarray]
    sensor_direction_tree: cKDTree
    maximum_ray_origin_offset_m: float
    frame_cache: dict[int, _E25NewFrameContext] = field(default_factory=dict)
    support_stream_cache: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict
    )
    observation_cache: dict[
        tuple[str, int, int, int], _E25NewObservation
    ] = field(default_factory=dict)
    frame_loader: Callable[[int], SourceFrame] | None = None
    available_frame_ids: frozenset[int] | None = None
    source_sequence_id_override: int | None = None

    @property
    def source_sequence_id(self) -> int:
        if self.frames_by_id:
            return next(iter(self.frames_by_id.values())).sequence_id
        if self.source_sequence_id_override is None:
            raise RenderError("coverage-control source sequence is unavailable")
        return self.source_sequence_id_override


def build_coverage_control_context(
    frames: Sequence[SourceFrame],
    support_pool: QualifiedSupportPool,
    ray_grid: RayGrid,
    sensor: SensorCalibration,
    *,
    frame_loader: Callable[[int], SourceFrame] | None = None,
    frame_ids: Sequence[int] | None = None,
    source_sequence_id: int | None = None,
    trajectory_yaws: Mapping[int, float] | None = None,
) -> CoverageControlContext:
    """Bind the sole selector to resident or bounded on-demand source frames."""

    source_frames = tuple(frames)
    frames_by_id = {int(frame.frame_id): frame for frame in source_frames}
    if source_frames:
        if len(frames_by_id) != len(source_frames):
            raise RenderError("coverage-control frame IDs must be unique")
        sequence_ids = {int(frame.sequence_id) for frame in source_frames}
        if len(sequence_ids) != 1:
            raise RenderError("coverage-control frames must use one source sequence")
        resolved_sequence_id = next(iter(sequence_ids))
        resolved_frame_ids = frozenset(frames_by_id)
        resolved_yaws = trajectory_yaw_by_frame(source_frames)
        if any(int(frame.xyzi.shape[0]) != ray_grid.slot_count for frame in source_frames):
            raise RenderError("coverage-control frame and ray-grid slots differ")
    else:
        if frame_loader is None or frame_ids is None or source_sequence_id is None:
            raise RenderError(
                "streaming coverage-control context requires loader, frame IDs and sequence"
            )
        ordered_ids = tuple(map(int, frame_ids))
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise RenderError("streaming coverage-control frame IDs are invalid")
        resolved_sequence_id = _integer(
            "source_sequence_id", source_sequence_id
        )
        resolved_frame_ids = frozenset(ordered_ids)
        if trajectory_yaws is None or set(trajectory_yaws) != set(ordered_ids):
            raise RenderError("streaming coverage-control trajectory yaw is incomplete")
        resolved_yaws = {int(key): float(value) for key, value in trajectory_yaws.items()}
    if not isinstance(support_pool, QualifiedSupportPool):
        raise TypeError("coverage-control context requires a qualified support pool")
    support_range_bin = _gate1_range_bin(support_pool.ranges_m)
    support_rows = {
        (semantic, range_bin): np.ascontiguousarray(
            np.flatnonzero(
                np.isin(
                    support_pool.semantics,
                    tuple(normal_control_support_semantics(semantic)),
                )
                & (support_range_bin == range_bin)
            ),
            dtype=np.int64,
        )
        for semantic in (10, 18, 20, 30)
        for range_bin in range(5)
    }
    unit_sensor_directions = ray_grid.directions_sensor / np.linalg.norm(
        ray_grid.directions_sensor, axis=1, keepdims=True
    )
    if ray_grid.origins_sensor is None:
        raise RenderError("coverage-control ray origins are unavailable")
    return CoverageControlContext(
        frames_by_id,
        support_pool,
        ray_grid,
        sensor,
        resolved_yaws,
        support_rows,
        cKDTree(unit_sensor_directions, compact_nodes=True),
        float(np.max(np.linalg.norm(ray_grid.origins_sensor, axis=1))),
        frame_loader=frame_loader,
        available_frame_ids=resolved_frame_ids,
        source_sequence_id_override=resolved_sequence_id,
    )


def _e25_new_frame_context(
    context: CoverageControlContext,
    frame_id: int,
) -> _E25NewFrameContext:
    """Cache only the immutable ray transforms needed by one worker."""

    grid = context.ray_grid
    cached = context.frame_cache.pop(frame_id, None)
    if cached is None:
        frame = context.frames_by_id.get(frame_id)
        if frame is None:
            if (
                context.frame_loader is None
                or context.available_frame_ids is None
                or frame_id not in context.available_frame_ids
            ):
                raise RenderError("coverage-control support frame is unavailable")
            frame = context.frame_loader(frame_id)
        if (
            frame.sequence_id != context.source_sequence_id
            or int(frame.frame_id) != frame_id
            or int(frame.xyzi.shape[0]) != grid.slot_count
        ):
            raise RenderError("coverage-control loaded frame identity changed")
        pose_rotation, lidar_origin = _pose(frame)
        directions_sensor = grid.directions_for(frame)
        directions_world = directions_sensor @ pose_rotation.T
        origins_sensor = grid.origins_for(frame)
        origins_world = origins_sensor @ pose_rotation.T + lidar_origin
        native_range = np.asarray(grid.ranges(frame)).copy()
        native_range[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = np.inf
        cached = _E25NewFrameContext(
            frame,
            (
                directions_sensor,
                directions_world,
                origins_sensor,
                origins_world,
                native_range,
            ),
        )
    context.frame_cache[frame_id] = cached
    while len(context.frame_cache) > 4:
        context.frame_cache.pop(next(iter(context.frame_cache)))
    return cached


def _e25_new_conservative_ray_slots(
    context: CoverageControlContext,
    item: ObjectSpec,
    pose_rotation: np.ndarray,
    lidar_origin_world: np.ndarray,
) -> np.ndarray:
    """Return a proven superset of rays that can meet the bounding sphere."""

    tree = context.sensor_direction_tree
    center_world = (
        np.asarray(item.translation_world_m, dtype=np.float64)
        - lidar_origin_world
    )
    center = center_world @ pose_rotation
    distance = float(np.linalg.norm(center))
    # Moving each calibrated beam origin to the lidar origin can enlarge the
    # line-to-centre distance by at most this origin-offset norm.
    radius = (
        item.shape.bound_radius_m
        + context.maximum_ray_origin_offset_m
        + 1.0e-6
    )
    if distance <= radius:
        return np.arange(context.ray_grid.slot_count, dtype=np.int32)
    angle = math.asin(min(radius / distance, 1.0))
    chord = 2.0 * math.sin(0.5 * angle) + 1.0e-12
    slots = tree.query_ball_point(
        center / distance, chord, workers=1
    )
    return np.asarray(sorted(slots), dtype=np.int32)


def _coverage_control_observation(
    context: CoverageControlContext,
    item: ObjectSpec,
    patch: SupportPatch,
    world_seed: int,
    assigned_range_bin: int,
    world_objects: Sequence[ObjectSpec],
) -> _E25NewObservation:
    """Adjudicate one control in its actual partial or complete world."""

    grid, sensor = context.ray_grid, context.sensor
    frame_context = _e25_new_frame_context(context, patch.frame_id)
    frame = frame_context.frame
    pose_rotation, lidar_origin_world = _pose(frame)
    objects = tuple(world_objects)
    if sum(other.object_id == item.object_id for other in objects) != 1:
        raise RenderError("coverage-control world must contain the candidate once")
    world = WorldSpec(world_seed, context.source_sequence_id, objects)
    cache_key = (
        world.identity, patch.frame_id, item.object_id, assigned_range_bin
    )
    cached = context.observation_cache.pop(cache_key, None)
    if cached is not None:
        context.observation_cache[cache_key] = cached
        return cached
    candidate_world = WorldSpec(
        world_seed, context.source_sequence_id, (item,)
    )
    _, _, trace = _single_object_sensor_precheck(
        frame,
        candidate_world,
        grid,
        sensor,
        trace_context=frame_context.trace,
        candidate_slots_hint=_e25_new_conservative_ray_slots(
            context, item, pose_rotation, lidar_origin_world
        ),
    )
    candidate_slots = trace.candidate_slots
    trace_context = frame_context.trace
    singleton = len(objects) == 1 and objects[0].object_id == item.object_id
    if singleton:
        _, full_competition = _expand_single_object_trace(
            trace, item, grid.slot_count
        )
        compact_distance = full_competition.distance_m[candidate_slots]
        compact_object_id = full_competition.object_id[candidate_slots]
    else:
        compact_competition = _accepted_object_hits(
            trace_context[3][candidate_slots],
            trace_context[1][candidate_slots],
            world,
            grid,
            sensor,
            patch.frame_id,
            slot_ids=candidate_slots,
        )
        compact_distance = compact_competition.distance_m
        compact_object_id = compact_competition.object_id
    visible = (
        compact_object_id == item.object_id
    ) & (
        compact_distance
        < trace.native_range_m - world.tie_tolerance_m
    )
    visible_slots = candidate_slots[visible]
    directions_sensor, origins_sensor = trace_context[0], trace_context[2]
    # Reproduce render_frame's float32 XYZ packing before the official range
    # projection so a range-bin edge is decided exactly as in the renderer.
    packed_visible_xyz = (
        origins_sensor[visible_slots]
        + compact_distance[visible, None]
        * directions_sensor[visible_slots]
    ).astype(np.float32)
    official_visible = np.sum(
        (
            packed_visible_xyz.astype(np.float64)
            - origins_sensor[visible_slots]
        )
        * directions_sensor[visible_slots],
        axis=1,
    ) + grid.official_range_offset_m
    visible_count = int(visible_slots.size)
    median_range = (
        float(np.median(official_visible)) if visible_count else math.nan
    )
    range_bin = (
        int(_gate1_range_bin(np.asarray((median_range,)))[0])
        if visible_count else -1
    )
    if visible_count:
        angle = np.arctan2(
            directions_sensor[visible_slots, 1],
            directions_sensor[visible_slots, 0],
        )
        circular = math.atan2(
            float(np.sin(angle).sum()), float(np.cos(angle).sum())
        )
        sector = int(
            math.floor((circular % (2.0 * math.pi)) / (math.pi / 4.0))
        ) % 8
        median_beam = float(np.median(visible_slots // grid.columns))
    else:
        sector, median_beam = -1, math.nan
    accepted_in_range = int(np.count_nonzero(trace.accepted & trace.in_range))
    visible_in_range = int(np.count_nonzero(
        (official_visible >= 2.5) & (official_visible <= 50.0)
    ))
    geometry_in_range = int(np.count_nonzero(trace.valid & trace.in_range))
    occlusion = (
        float(1.0 - visible_in_range / accepted_in_range)
        if accepted_in_range else math.nan
    )
    preliminary = _E25NewObservation(
        patch.frame_id,
        visible_count,
        visible_in_range,
        accepted_in_range,
        geometry_in_range,
        median_range,
        median_beam,
        range_bin,
        sector,
        occlusion,
    )
    if visible_count == 0 or range_bin != assigned_range_bin:
        return preliminary

    if not singleton:
        full_competition = _accepted_object_hits(
            trace_context[3],
            trace_context[1],
            world,
            grid,
            sensor,
            patch.frame_id,
        )
    rendered = render_frame(
        frame,
        world,
        grid,
        sensor,
        _trace_context=trace_context,
        _competition=full_competition,
    )
    returned = (
        np.asarray(rendered.normal_control_mask, dtype=np.bool_)
        & (np.asarray(rendered.object_id_internal) == item.object_id)
    )
    slots = np.flatnonzero(returned)
    rendered_xyz = np.asarray(rendered.source.xyzi[slots, :3], dtype=np.float64)
    official = np.sum(
        (rendered_xyz - trace_context[2][slots]) * trace_context[0][slots],
        axis=1,
    ) + grid.official_range_offset_m
    final_median = float(np.median(official)) if slots.size else math.nan
    final_bin = (
        int(_gate1_range_bin(np.asarray((final_median,)))[0])
        if slots.size else -1
    )
    visible_in_range_mask = (official >= 2.5) & (official <= 50.0)
    final_visible_in_range = int(np.count_nonzero(visible_in_range_mask))
    final_accepted_in_range = accepted_in_range
    final_occlusion = (
        float(1.0 - final_visible_in_range / final_accepted_in_range)
        if final_accepted_in_range else math.nan
    )
    if slots.size:
        angle = np.arctan2(
            trace_context[0][slots, 1], trace_context[0][slots, 0]
        )
        circular = math.atan2(
            float(np.sin(angle).sum()), float(np.cos(angle).sum())
        )
        final_sector = int(
            math.floor((circular % (2.0 * math.pi)) / (math.pi / 4.0))
        ) % 8
        final_beam = float(np.median(slots // grid.columns))
    else:
        final_sector, final_beam = -1, math.nan
    final = _E25NewObservation(
        patch.frame_id,
        int(slots.size),
        final_visible_in_range,
        final_accepted_in_range,
        geometry_in_range,
        final_median,
        final_beam,
        final_bin,
        final_sector,
        final_occlusion,
    )
    if (
        final.visible_returns != preliminary.visible_returns
        or final.visible_in_range_returns != preliminary.visible_in_range_returns
        or final.accepted_in_range_returns != preliminary.accepted_in_range_returns
        or final.range_bin != preliminary.range_bin
        or final.azimuth_sector != preliminary.azimuth_sector
        or not np.isclose(
            final.median_official_range_m,
            preliminary.median_official_range_m,
            atol=1.0e-10,
            rtol=0.0,
        )
        or not np.isclose(final.median_beam, preliminary.median_beam)
        or not np.isclose(
            final.occlusion, preliminary.occlusion, equal_nan=True
        )
        or not np.all(rendered.object_id_internal[slots] == item.object_id)
        or full_competition.geometric_hits[item.object_id]
        != int(np.count_nonzero(trace.valid))
        or full_competition.accepted_hits[item.object_id]
        != int(np.count_nonzero(trace.accepted))
    ):
        raise RenderError(
            "coverage-control compact competition and final renderer disagree"
        )
    context.observation_cache[cache_key] = final
    while len(context.observation_cache) > 128:
        context.observation_cache.pop(next(iter(context.observation_cache)))
    return final


def _e25_new_observation(
    item: ObjectSpec,
    patch: SupportPatch,
    control_seed: int,
    assigned_range_bin: int,
) -> _E25NewObservation:
    """Run the frozen single-fixture E25-new observation through production code."""

    context = _E25_NEW_CONTROL_CONTEXT
    if context is None:
        raise RuntimeError("E25-new control context is not initialized")
    return _coverage_control_observation(
        context,
        item,
        patch,
        control_seed,
        assigned_range_bin,
        (item,),
    )


def _coverage_control_support_stream(
    context: CoverageControlContext,
    fixture_index: int,
    semantic: int,
    assigned_range_bin: int,
) -> np.ndarray:
    """Select the deterministic top-128 E21 rows without result-based ordering."""

    if assigned_range_bin != _e25_new_assigned_range_bin(fixture_index):
        raise PlacementError(
            "coverage-control range bin differs from its canonical template index"
        )
    cache_key = (fixture_index, semantic)
    cached = context.support_stream_cache.get(cache_key)
    if cached is not None:
        return cached
    pool = context.support_pool
    rows = context.support_rows.get(
        (semantic, assigned_range_bin), np.empty(0, dtype=np.int64)
    )
    if rows.size == 0:
        return rows
    with np.errstate(over="ignore"):
        salt = (
            np.uint64(_E25_NEW_NAMESPACE)
            ^ np.uint64(fixture_index + 1) * np.uint64(0xD1B54A32D192ED03)
        )
    keys = _splitmix64(pool.selection_hashes[rows] ^ salt)
    limit = min(128, rows.size)
    selected = (
        np.argpartition(keys, limit - 1)[:limit]
        if limit < rows.size else np.arange(rows.size)
    )
    order = np.lexsort((pool.pool_indices[rows[selected]], keys[selected]))
    result = _freeze(np.ascontiguousarray(rows[selected[order]], dtype=np.int64))
    context.support_stream_cache[cache_key] = result
    return result


def precompute_coverage_control_support_streams(
    context: CoverageControlContext,
    normal_template_library: Sequence[NormalTemplateShape],
) -> None:
    """Populate all 256 frozen E25-new streams before worker forking."""

    templates = tuple(normal_template_library)
    canonical_normal_template_library_identity(templates)
    for index, template in enumerate(templates):
        _coverage_control_support_stream(
            context,
            index,
            int(template.raw_semantic_id),
            _e25_new_assigned_range_bin(index),
        )


def _e25_new_support_stream(
    fixture_index: int,
    semantic: int,
    assigned_range_bin: int,
) -> np.ndarray:
    context = _E25_NEW_CONTROL_CONTEXT
    if context is None:
        raise RuntimeError("E25-new control context is not initialized")
    return _coverage_control_support_stream(
        context, fixture_index, semantic, assigned_range_bin
    )


def _e25_new_worker(index: int) -> dict[str, object]:
    context, obstacles = _E25_NEW_CONTROL_CONTEXT, _E25_NEW_OBSTACLES
    if (
        context is None
        or obstacles is None
        or len(_E25_NEW_TEMPLATES) != 256
    ):
        raise RuntimeError("E25-new worker state is not initialized")
    pool = context.support_pool
    control_seed = 2_500_000 + index
    assigned_range_bin = _e25_new_assigned_range_bin(index)
    source = _E25_NEW_TEMPLATES[index]
    semantic = int(source.raw_semantic_id)
    template_identity = _normal_template_identity(source)
    observations: dict[int, _E25NewObservation] = {}
    sensor_attempts = 0
    no_visible_rejections = 0
    range_rejections = 0
    try:
        scale = np.random.default_rng(
            np.random.SeedSequence([control_seed + 2, 2501])
        ).uniform(0.9, 1.1, size=3)
        shape = _aligned_scaled_template(source, scale)
        grounding = qualify_grounding(shape)
        limit = math.pi if semantic == 30 else math.radians(15.0)
        perturbation = float(np.random.default_rng(
            np.random.SeedSequence([control_seed, 2502])
        ).uniform(-limit, limit))
        material_seed = control_seed + 2503
        material = MaterialSpec.sample(material_seed)
        proposal_rows = _e25_new_support_stream(
            index, semantic, assigned_range_bin
        )
        if proposal_rows.size == 0:
            raise PlacementError(
                "assigned range bin has no semantically legal E21 support"
            )

        def yaw_for_support(patch: SupportPatch) -> float:
            return context.trajectory_yaw_by_frame[patch.frame_id] + perturbation

        def reject_after_placement(
            proposed: ObjectSpec, patch: SupportPatch
        ) -> str | None:
            nonlocal sensor_attempts, no_visible_rejections, range_rejections
            sensor_attempts += 1
            observation = _e25_new_observation(
                proposed, patch, control_seed, assigned_range_bin
            )
            observations[patch.pool_index] = observation
            if observation.visible_returns < 1:
                no_visible_rejections += 1
                return "no_visible_normal_control_return"
            if observation.range_bin != assigned_range_bin:
                range_rejections += 1
                return "assigned_visible_range_bin_mismatch"
            return None

        item, record = place_object(
            shape,
            material,
            pool,
            obstacles,
            object_id=index + 1,
            label="normal-control",
            proposal_namespace="E25-new-support-v1",
            proposal_stream=index,
            yaw_rad=perturbation,
            material_seed=material_seed,
            yaw_seed=control_seed,
            template_identity=template_identity,
            proposal_rows=proposal_rows,
            maximum_candidates=128,
            grounding_eligibility=grounding,
            yaw_for_support=yaw_for_support,
            post_placement_rejection=reject_after_placement,
        )
        observation = observations[record.support_pool_index]
        support_row = int(
            np.searchsorted(pool.pool_indices, record.support_pool_index)
        )
        support_error = int(
            support_row >= pool.pool_indices.size
            or int(pool.pool_indices[support_row]) != record.support_pool_index
        )
        if support_error:
            patch = pool.patch(0)
        else:
            patch = pool.patch(support_row)
        expected_rotation = _ground_rotation(
            np.asarray(patch.normal_world), yaw_for_support(patch)
        )
        semantic_error = int(
            record.support_semantic
            not in normal_control_support_semantics(semantic)
        )
        scale_error = int(np.any((scale < 0.9) | (scale > 1.1)))
        pose_error = int(
            np.max(np.abs(
                expected_rotation
                - np.asarray(item.rotation_world_from_local)
            )) > 1.0e-10
        )
        grounding_error = int(
            not grounding.passed
            or record.grounding_standard_lower_support_m
            != grounding.standard_lower_support_m
            or record.grounding_strict_lower_support_m
            != grounding.strict_lower_support_m
            or record.grounding_buried_fraction != grounding.buried_fraction
        )
        collision_error = int(observed_normal_collision(item, obstacles)[0])
        material_error = int(
            item.material.to_dict() != MaterialSpec.sample(material_seed).to_dict()
        )
        range_identity_error = int(
            observation.range_bin != assigned_range_bin
        )
        visibility_error = int(observation.visible_returns < 1)
        physical_rejections = sum(
            reason in {
                "observed_normal_deep_penetration",
                "obvious_pair_penetration",
            }
            for reason in record.rejection_reasons
        )
        proposal_accounting_error = int(
            record.accepted_proposal + 1 != len(record.proposal_pool_indices)
            or record.accepted_proposal != len(record.rejection_reasons)
            or len(record.proposal_pool_indices)
            != len(record.proposal_minimum_obstacle_sdf_m)
            or record.accepted_proposal
            != physical_rejections + no_visible_rejections + range_rejections
            or sensor_attempts
            != no_visible_rejections + range_rejections + 1
            or record.accepted_proposal + 1
            != physical_rejections + sensor_attempts
        )
        return {
            "fixture_index": index,
            "hard_error": proposal_accounting_error,
            "placement_exhaustion": 0,
            "preplacement_rejection": 0,
            "error": "",
            "control_seed": control_seed,
            "semantic": semantic,
            "template_identity": template_identity,
            "assigned_range_bin": assigned_range_bin,
            "final_range_bin": observation.range_bin,
            "median_official_range_m": observation.median_official_range_m,
            "median_beam": observation.median_beam,
            "visible_returns": observation.visible_returns,
            "visible_in_range_returns": observation.visible_in_range_returns,
            "accepted_in_range_returns": observation.accepted_in_range_returns,
            "geometry_in_range_hits": observation.geometry_in_range_hits,
            "occlusion": observation.occlusion,
            "azimuth_sector": observation.azimuth_sector,
            "scale": scale,
            "pose_perturbation_rad": perturbation,
            "support_semantic": record.support_semantic,
            "support_pool_index": record.support_pool_index,
            "support_frame": record.support_frame,
            "support_proposal_count": record.accepted_proposal + 1,
            "physical_rejections": physical_rejections,
            "no_visible_rejections": no_visible_rejections,
            "range_rejections": range_rejections,
            "support_error": support_error,
            "semantic_error": semantic_error,
            "scale_error": scale_error,
            "pose_error": pose_error,
            "grounding_error": grounding_error,
            "collision_error": collision_error,
            "material_error": material_error,
            "range_identity_error": range_identity_error,
            "visibility_error": visibility_error,
            "proposal_accounting_error": proposal_accounting_error,
            "object_json": json.dumps(
                item.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            "placement_report_json": json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":")
            ),
        }
    except PlacementExhaustion as error:
        proposal_count = len(error.proposal_pool_indices)
        physical_rejections = sum(
            reason in {
                "observed_normal_deep_penetration",
                "obvious_pair_penetration",
            }
            for reason in error.rejection_reasons
        )
        proposal_accounting_error = int(
            proposal_count != len(error.rejection_reasons)
            or proposal_count != len(error.minimum_obstacle_sdf_m)
            or proposal_count
            != physical_rejections
            + no_visible_rejections
            + range_rejections
        )
        return {
            "fixture_index": index,
            "hard_error": proposal_accounting_error,
            "placement_exhaustion": 1,
            "preplacement_rejection": 0,
            "error": f"PlacementExhaustion: {error}",
            "control_seed": control_seed,
            "semantic": semantic,
            "template_identity": template_identity,
            "assigned_range_bin": assigned_range_bin,
            "support_proposal_count": proposal_count,
            "physical_rejections": physical_rejections,
            "no_visible_rejections": no_visible_rejections,
            "range_rejections": range_rejections,
            "proposal_accounting_error": proposal_accounting_error,
        }
    except PlacementError as error:
        return {
            "fixture_index": index,
            "hard_error": 0,
            "placement_exhaustion": 1,
            "preplacement_rejection": 1,
            "error": f"PlacementError: {error}",
            "control_seed": control_seed,
            "semantic": semantic,
            "template_identity": template_identity,
            "assigned_range_bin": assigned_range_bin,
            "support_proposal_count": 0,
            "physical_rejections": 0,
            "no_visible_rejections": 0,
            "range_rejections": 0,
        }
    except Exception as error:
        return {
            "fixture_index": index,
            "hard_error": 1,
            "placement_exhaustion": 0,
            "preplacement_rejection": 0,
            "error": f"{type(error).__name__}: {error}",
            "control_seed": control_seed,
            "semantic": semantic,
            "template_identity": template_identity,
            "assigned_range_bin": assigned_range_bin,
        }


def _e25_new_arrays(
    records: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object, default: object) -> np.ndarray:
        return np.asarray([item.get(name, default) for item in records], dtype=dtype)

    return {
        "fixture_index": values("fixture_index", np.int16, -1),
        "control_seed": values("control_seed", np.int64, -1),
        "semantic": values("semantic", np.uint16, 0),
        "template_identity": values("template_identity", "S64", ""),
        "assigned_range_bin": values("assigned_range_bin", np.int8, -1),
        "final_range_bin": values("final_range_bin", np.int8, -1),
        "median_official_range_m": values(
            "median_official_range_m", np.float64, math.nan
        ),
        "median_beam": values("median_beam", np.float64, math.nan),
        "visible_returns": values("visible_returns", np.int32, 0),
        "visible_in_range_returns": values(
            "visible_in_range_returns", np.int32, 0
        ),
        "accepted_in_range_returns": values(
            "accepted_in_range_returns", np.int32, 0
        ),
        "geometry_in_range_hits": values(
            "geometry_in_range_hits", np.int32, 0
        ),
        "occlusion": values("occlusion", np.float64, math.nan),
        "azimuth_sector": values("azimuth_sector", np.int8, -1),
        "scale": np.asarray(
            [item.get("scale", (math.nan,) * 3) for item in records],
            dtype=np.float64,
        ),
        "pose_perturbation_rad": values(
            "pose_perturbation_rad", np.float64, math.nan
        ),
        "support_semantic": values("support_semantic", np.uint16, 0),
        "support_pool_index": values("support_pool_index", np.int64, -1),
        "support_frame": values("support_frame", np.int16, -1),
        "support_proposal_count": values(
            "support_proposal_count", np.int16, 0
        ),
        "physical_rejections": values("physical_rejections", np.int16, 0),
        "no_visible_rejections": values(
            "no_visible_rejections", np.int16, 0
        ),
        "range_rejections": values("range_rejections", np.int16, 0),
        "support_error": values("support_error", np.uint8, 0),
        "semantic_error": values("semantic_error", np.uint8, 0),
        "scale_error": values("scale_error", np.uint8, 0),
        "pose_error": values("pose_error", np.uint8, 0),
        "grounding_error": values("grounding_error", np.uint8, 0),
        "collision_error": values("collision_error", np.uint8, 0),
        "material_error": values("material_error", np.uint8, 0),
        "range_identity_error": values(
            "range_identity_error", np.uint8, 0
        ),
        "visibility_error": values("visibility_error", np.uint8, 0),
        "proposal_accounting_error": values(
            "proposal_accounting_error", np.uint8, 0
        ),
        "object_json": np.asarray([
            str(item.get("object_json", "")).encode() for item in records
        ]),
        "placement_report_json": np.asarray([
            str(item.get("placement_report_json", "")).encode()
            for item in records
        ]),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "placement_exhaustion_code": values(
            "placement_exhaustion", np.uint8, 0
        ),
        "preplacement_rejection_code": values(
            "preplacement_rejection", np.uint8, 0
        ),
        "error_message": values("error", "U512", ""),
    }


def run_e25_new_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Run the single frozen coverage-oriented E25-new qualification."""

    if processes != 24:
        raise PlacementError("formal E25-new requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    frames = tuple(
        sequence.source_frame(frame_id) for frame_id in sequence.frame_ids
    )
    templates = extract_normal_template_library(frames)
    _, counts, library_hash = canonical_normal_template_library_identity(templates)
    pool = load_qualified_support_pool(support_pool_path)
    calibration_path_resolved = Path(calibration_path).expanduser().resolve(strict=True)
    if _sha256_path(calibration_path_resolved) != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise PlacementError("E25-new sensor calibration identity changed")
    grid, sensor = load_sensor_calibration(calibration_path_resolved)
    obstacles = collect_observed_obstacle_index(frames)
    control_context = build_coverage_control_context(
        frames, pool, grid, sensor
    )
    precompute_coverage_control_support_streams(control_context, templates)
    global _E25_NEW_CONTROL_CONTEXT, _E25_NEW_OBSTACLES
    global _E25_NEW_TEMPLATES
    _E25_NEW_CONTROL_CONTEXT = control_context
    _E25_NEW_OBSTACLES = obstacles
    _E25_NEW_TEMPLATES = templates

    work_order = tuple(sorted(
        range(256),
        key=lambda index: (
            -_e25_new_assigned_range_bin(index),
            -int(templates[index].plane_normals.shape[0]),
            index,
        ),
    ))
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        scheduled = workers.map(_e25_new_worker, work_order, chunksize=1)
    by_index = {int(item["fixture_index"]): item for item in scheduled}
    records = [by_index[index] for index in range(256)]
    run_seconds = time.monotonic() - started
    arrays = _e25_new_arrays(records)
    completed_mask = (
        (arrays["template_identity"] != b"")
        & (arrays["hard_error_code"] == 0)
        & (arrays["placement_exhaustion_code"] == 0)
    )
    completed = int(np.count_nonzero(completed_mask))
    attempted_unique_templates = int(np.unique(
        arrays["template_identity"][arrays["template_identity"] != b""]
    ).size)
    unique_templates = int(np.unique(
        arrays["template_identity"][completed_mask]
    ).size)
    expected_bins = np.asarray(
        [_e25_new_assigned_range_bin(index) for index in range(256)],
        dtype=np.int16,
    )
    assigned_count = np.bincount(expected_bins, minlength=5)[:5]
    accepted_count = np.bincount(
        arrays["final_range_bin"][completed_mask], minlength=5
    )[:5]
    semantic_values = np.asarray((10, 18, 20, 30), dtype=np.uint16)
    class_completed = {
        str(int(semantic)): int(np.count_nonzero(
            completed_mask & (arrays["semantic"] == semantic)
        ))
        for semantic in semantic_values
    }
    class_assigned_range = np.stack([
        np.bincount(
            expected_bins[arrays["semantic"] == semantic], minlength=5
        )[:5]
        for semantic in semantic_values
    ])
    class_accepted_range = np.stack([
        np.bincount(
            arrays["final_range_bin"][completed_mask & (
                arrays["semantic"] == semantic
            )],
            minlength=5,
        )[:5]
        for semantic in semantic_values
    ])
    azimuth_count = np.bincount(
        arrays["azimuth_sector"][completed_mask], minlength=8
    )[:8]
    class_azimuth_count = np.stack([
        np.bincount(
            arrays["azimuth_sector"][completed_mask & (
                arrays["semantic"] == semantic
            )],
            minlength=8,
        )[:8]
        for semantic in semantic_values
    ])
    maximum_azimuth_sector_count = int(np.max(azimuth_count, initial=0))
    maximum_azimuth_sector_share = (
        maximum_azimuth_sector_count / completed if completed else None
    )
    class_maximum_azimuth_sector_count = {
        str(int(semantic)): int(np.max(class_azimuth_count[row], initial=0))
        for row, semantic in enumerate(semantic_values)
    }
    class_maximum_azimuth_sector_share = {
        str(int(semantic)): (
            class_maximum_azimuth_sector_count[str(int(semantic))]
            / class_completed[str(int(semantic))]
            if class_completed[str(int(semantic))]
            else None
        )
        for semantic in semantic_values
    }
    finite_occlusion = completed_mask & np.isfinite(arrays["occlusion"])
    occlusion_layer = np.full(256, -1, dtype=np.int8)
    occlusion_layer[finite_occlusion] = np.searchsorted(
        np.asarray((0.25, 0.75)),
        arrays["occlusion"][finite_occlusion],
        side="right",
    )
    occlusion_count = np.bincount(
        occlusion_layer[finite_occlusion], minlength=3
    )[:3]
    class_occlusion_count = np.stack([
        np.bincount(
            occlusion_layer[finite_occlusion & (
                arrays["semantic"] == semantic
            )],
            minlength=3,
        )[:3]
        for semantic in semantic_values
    ])
    error_fields = (
        "support_error",
        "semantic_error",
        "scale_error",
        "pose_error",
        "grounding_error",
        "collision_error",
        "material_error",
        "range_identity_error",
        "visibility_error",
        "proposal_accounting_error",
        "hard_error_code",
        "placement_exhaustion_code",
        "preplacement_rejection_code",
    )
    errors = {name: int(np.sum(arrays[name])) for name in error_fields}
    pass_error_fields = (
        "support_error",
        "semantic_error",
        "scale_error",
        "pose_error",
        "grounding_error",
        "collision_error",
        "material_error",
        "range_identity_error",
        "visibility_error",
        "hard_error_code",
        "placement_exhaustion_code",
    )
    passed = (
        completed == 256
        and attempted_unique_templates == 256
        and unique_templates == 256
        and class_completed == {"10": 64, "18": 64, "20": 64, "30": 64}
        and np.array_equal(assigned_count, np.asarray((52, 51, 51, 51, 51)))
        and np.array_equal(accepted_count, assigned_count)
        and np.array_equal(arrays["fixture_index"], np.arange(256))
        and np.array_equal(arrays["control_seed"], 2_500_000 + np.arange(256))
        and np.array_equal(arrays["assigned_range_bin"], expected_bins)
        and all(errors[name] == 0 for name in pass_error_fields)
    )
    nvis = arrays["visible_returns"][completed_mask]
    nvis_summary = {
        "minimum": int(np.min(nvis)) if nvis.size else None,
        "median": float(np.median(nvis)) if nvis.size else None,
        "mean": float(np.mean(nvis)) if nvis.size else None,
        "q95": float(np.quantile(nvis, 0.95)) if nvis.size else None,
        "maximum": int(np.max(nvis)) if nvis.size else None,
    }
    support_path = Path(support_pool_path).expanduser().resolve(strict=True)
    calibration_path_resolved = Path(calibration_path).expanduser().resolve(strict=True)
    implementation_error_fields = (
        "support_error",
        "semantic_error",
        "scale_error",
        "pose_error",
        "grounding_error",
        "collision_error",
        "material_error",
        "range_identity_error",
        "visibility_error",
        "hard_error_code",
    )
    implementation_errors = sum(
        errors[name] for name in implementation_error_fields
    )
    metadata = {
        "experiment": "E25-new-normal-control",
        "passed": passed,
        "failure_classification": (
            None
            if passed
            else (
                "protocol_implementation_defect"
                if implementation_errors
                else "coverage_control_qualification_failure"
            )
        ),
        "templates": 256,
        "active_counts": counts,
        "library_sha256": library_hash,
        "fixtures": 256,
        "completed": completed,
        "attempted_unique_templates": attempted_unique_templates,
        "unique_templates": unique_templates,
        "class_completed": class_completed,
        "assigned_range_count": assigned_count.tolist(),
        "accepted_range_count": accepted_count.tolist(),
        "class_assigned_range_count": class_assigned_range.tolist(),
        "class_accepted_range_count": class_accepted_range.tolist(),
        "azimuth_sector_count": azimuth_count.tolist(),
        "class_azimuth_sector_count": class_azimuth_count.tolist(),
        "maximum_azimuth_sector_count": maximum_azimuth_sector_count,
        "maximum_azimuth_sector_share": maximum_azimuth_sector_share,
        "class_maximum_azimuth_sector_count": (
            class_maximum_azimuth_sector_count
        ),
        "class_maximum_azimuth_sector_share": (
            class_maximum_azimuth_sector_share
        ),
        "occlusion_layer_count": occlusion_count.tolist(),
        "class_occlusion_layer_count": class_occlusion_count.tolist(),
        "undefined_occlusion_count": int(np.count_nonzero(
            completed_mask & ~np.isfinite(arrays["occlusion"])
        )),
        "Nvis": nvis_summary,
        "support_proposals": int(np.sum(arrays["support_proposal_count"])),
        "physical_rejections": int(np.sum(arrays["physical_rejections"])),
        "no_visible_rejections": int(np.sum(arrays["no_visible_rejections"])),
        "range_rejections": int(np.sum(arrays["range_rejections"])),
        "implementation_errors": implementation_errors,
        **errors,
        "formal_repetitions": 1,
        "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "run_seconds": [run_seconds],
        "processes": processes,
        "numeric_library_threads_per_process": 1,
        "support_pool_sha256": _sha256_path(support_path),
        "calibration_sha256": _sha256_path(calibration_path_resolved),
        "scientific_array_hash": _scientific_array_hash(arrays),
        "train_201_used": False,
        "real_target_bank_used": False,
        "E45A_calipers_used_in_generation": False,
        "descriptive_only": ["azimuth sector", "occlusion layer", "Nvis"],
        "claim_limit": (
            "E25-new qualifies coverage-oriented legal and visible normal controls "
            "over the official 2.5--50 m range; it does not estimate the real-normal "
            "distance distribution or establish real/control common support or source "
            "indistinguishability."
        ),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        occlusion_layer=occlusion_layer,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    os.replace(temporary, destination)
    return metadata


_E26_SUPPORT_POOL: QualifiedSupportPool | None = None
_E26_OBSTACLES: ObservedObstacleIndex | None = None
_E26_TEMPLATES: tuple[NormalTemplateShape, ...] = ()
_E26_TEMPLATE_INDEX: dict[str, int] = {}
_E26_CONTROL_CONTEXT: CoverageControlContext | None = None
_E26_TRAJECTORY_YAW: dict[int, float] = {}
_E26_RENDERER_IDENTITY = ""


def _e26_request_identity(world_hash: str, frame_id: int) -> str:
    payload = f"{world_hash}:{frame_id}:{_E26_RENDERER_IDENTITY}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _e26_worker(index: int) -> dict[str, object]:
    pool, obstacles, control_context = (
        _E26_SUPPORT_POOL,
        _E26_OBSTACLES,
        _E26_CONTROL_CONTEXT,
    )
    if (
        pool is None or obstacles is None or not _E26_TEMPLATES
        or control_context is None or len(_E26_TEMPLATE_INDEX) != 256
        or len(_E26_RENDERER_IDENTITY) != 64
    ):
        raise RuntimeError("E26 worker state is not initialized")
    world_seed = 2_600_000 + index
    world_type = WORLD_TYPES[index // 64]
    try:
        world, report = sample_training_world(
            _E26_TEMPLATES, pool, obstacles, world_type, world_seed,
            control_context=control_context,
            maximum_attempts=48,
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
        expected_normal_count, expected_anomaly_count = _training_entity_counts(
            world_type, world_seed
        )
        expected_attempt_seed = world_seed + 1_000_003 * report.world_attempt
        expected_labels: list[ObjectLabel] = (
            ["normal-control"] * expected_normal_count
            + ["anomaly-proxy"] * expected_anomaly_count
        )
        np.random.default_rng(expected_attempt_seed).shuffle(expected_labels)
        world_random_stream_errors = int(
            report.count_seed != world_seed
            or report.label_order_seed != expected_attempt_seed
            or report.normal_count != expected_normal_count
            or report.anomaly_count != expected_anomaly_count
            or [item.label for item in world.objects] != expected_labels
        )
        support_errors = 0
        pose_errors = 0
        material_errors = 0
        grounding_errors = 0
        collision_errors = 0
        control_visibility_errors = 0
        control_range_errors = 0
        control_support_stream_errors = 0
        control_random_stream_errors = 0
        anomaly_random_stream_errors = 0
        assigned_range_count = np.zeros(5, dtype=np.int16)
        final_range_count = np.zeros(5, dtype=np.int16)
        control_observations: list[dict[str, object]] = []

        def expected_grounded_transform(
            shape: InsertShape, patch: SupportPatch, yaw_rad: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """Independently reconstruct the frozen support transform."""

            normal = np.asarray(patch.normal_world, dtype=np.float64)
            contact = np.asarray(patch.anchor_world_m, dtype=np.float64).copy()
            contact[2] = -(
                normal[0] * contact[0]
                + normal[1] * contact[1]
                + patch.offset
            ) / normal[2]
            rotation = _ground_rotation(normal, yaw_rad)
            translation = contact - normal * shape.minimum_z_m(
                xy_resolution=33, z_steps=129
            )
            return rotation, translation

        for item, record in zip(world.objects, report.placements, strict=True):
            expected_entity_seed = (
                expected_attempt_seed + 10_007 * item.object_id
            )
            row = int(np.searchsorted(pool.pool_indices, record.support_pool_index))
            if (
                row >= pool.pool_indices.size
                or int(pool.pool_indices[row]) != record.support_pool_index
            ):
                support_errors += 1
                continue
            patch = pool.patch(row)
            grounding = qualify_grounding(item.shape)
            collision, minimum_sdf, _ = observed_normal_collision(item, obstacles)
            validation_errors += int(
                record.object_id != item.object_id
                or record.label != item.label
                or record.accepted_proposal + 1
                != len(record.proposal_pool_indices)
                or record.accepted_proposal != len(record.rejection_reasons)
                or len(record.proposal_pool_indices)
                != len(record.proposal_minimum_obstacle_sdf_m)
            )
            support_errors += int(
                record.support_frame != patch.frame_id
                or record.support_slot != patch.slot
                or record.support_semantic != patch.semantic
            )
            grounding_errors += int(
                not grounding.passed
                or record.grounding_standard_lower_support_m
                != grounding.standard_lower_support_m
                or record.grounding_strict_lower_support_m
                != grounding.strict_lower_support_m
                or record.grounding_buried_fraction != grounding.buried_fraction
            )
            collision_errors += int(
                collision
                or not np.isclose(
                    record.minimum_obstacle_sdf_m,
                    minimum_sdf,
                    equal_nan=True,
                )
            )
            expected_material_seed = expected_entity_seed + 11
            material_errors += int(
                record.material_seed != expected_material_seed
                or item.material.to_dict()
                != MaterialSpec.sample(expected_material_seed).to_dict()
            )
            if item.label == "normal-control":
                expected_template_seed = expected_entity_seed + 1
                expected_scale_seed = expected_entity_seed + 2
                expected_yaw_seed = expected_entity_seed + 31
                expected_template_index = int(
                    np.random.default_rng(expected_template_seed).integers(
                        0, len(_E26_TEMPLATES)
                    )
                )
                expected_source = _E26_TEMPLATES[expected_template_index]
                expected_identity = _normal_template_identity(expected_source)
                expected_scale = np.random.default_rng(
                    np.random.SeedSequence([expected_scale_seed, 2501])
                ).uniform(0.9, 1.1, size=3)
                expected_shape = _aligned_scaled_template(
                    expected_source, expected_scale
                )
                semantic = int(expected_source.raw_semantic_id)
                perturbation_limit = (
                    math.pi if semantic == 30
                    else math.radians(30.0) if semantic in (11, 15, 31, 32)
                    else math.radians(15.0)
                )
                expected_perturbation = float(np.random.default_rng(
                    np.random.SeedSequence([expected_yaw_seed, 2502])
                ).uniform(-perturbation_limit, perturbation_limit))
                expected_yaw = (
                    _E26_TRAJECTORY_YAW[patch.frame_id]
                    + expected_perturbation
                )
                expected_rotation, expected_translation = expected_grounded_transform(
                    expected_shape,
                    patch,
                    expected_yaw,
                )
                support_errors += int(
                    record.support_semantic
                    not in normal_control_support_semantics(semantic)
                )
                pose_errors += int(
                    record.pose_perturbation_rad != expected_perturbation
                    or np.max(np.abs(
                        expected_rotation
                        - np.asarray(item.rotation_world_from_local)
                    )) > 1.0e-10
                    or np.max(np.abs(
                        expected_translation
                        - np.asarray(item.translation_world_m)
                    )) > 1.0e-10
                )
                validation_errors += int(
                    not isinstance(item.shape, NormalTemplateShape)
                    or np.any(
                        (np.asarray(item.shape.scale_xyz) < 0.9)
                        | (np.asarray(item.shape.scale_xyz) > 1.1)
                    )
                )
                control_random_stream_errors += int(
                    record.template_seed != expected_template_seed
                    or record.scale_seed != expected_scale_seed
                    or record.material_seed != expected_material_seed
                    or record.yaw_seed != expected_yaw_seed
                    or record.template_identity != expected_identity
                    or _E26_TEMPLATE_INDEX.get(record.template_identity or "", -1)
                    != expected_template_index
                    or item.shape.to_dict() != expected_shape.to_dict()
                )
                assigned_bin = _e25_new_assigned_range_bin(
                    expected_template_index
                )
                assigned_range_count[assigned_bin] += 1
                expected_rows = _coverage_control_support_stream(
                    control_context,
                    expected_template_index,
                    semantic,
                    assigned_bin,
                )
                expected_pool_indices = pool.pool_indices[
                    expected_rows[:len(record.proposal_pool_indices)]
                ]
                control_support_stream_errors += int(
                    not np.array_equal(
                        expected_pool_indices,
                        np.asarray(record.proposal_pool_indices, dtype=np.uint64),
                    )
                )
                observation = _coverage_control_observation(
                    control_context,
                    item,
                    patch,
                    world_seed,
                    assigned_bin,
                    world.objects,
                )
                control_visibility_errors += int(
                    observation.visible_returns < 1
                )
                control_range_errors += int(
                    observation.range_bin != assigned_bin
                )
                if 0 <= observation.range_bin < 5:
                    final_range_count[observation.range_bin] += 1
                control_observations.append({
                    "object_id": item.object_id,
                    "template_index": expected_template_index,
                    "support_frame": patch.frame_id,
                    "assigned_range_bin": assigned_bin,
                    "final_range_bin": observation.range_bin,
                    "visible_returns": observation.visible_returns,
                    "median_official_range_m": observation.median_official_range_m,
                })
            else:
                expected_yaw_seed = expected_entity_seed + 31
                shape_proposal_count = len(record.shape_proposal_seeds)
                expected_shape_seeds = tuple(
                    expected_entity_seed + 3 + 3072 * proposal
                    for proposal in range(shape_proposal_count)
                )
                if not 1 <= shape_proposal_count <= 64:
                    anomaly_random_stream_errors += 1
                    continue
                expected_shape_seed = expected_shape_seeds[-1]
                expected_shape, expected_shape_report = ShapeSpec.sample_with_report(
                    expected_shape_seed
                )
                expected_yaw = float(
                    np.random.default_rng(expected_yaw_seed).uniform(
                        -math.pi, math.pi
                    )
                )
                expected_rotation, expected_translation = expected_grounded_transform(
                    expected_shape,
                    patch,
                    expected_yaw,
                )
                rejected_shape_error = any(
                    qualify_grounding(ShapeSpec.sample(seed)).passed
                    for seed in expected_shape_seeds[:-1]
                )
                anomaly_random_stream_errors += int(
                    not isinstance(item.shape, ShapeSpec)
                    or record.shape_seed != expected_shape_seed
                    or record.shape_proposal_seeds != expected_shape_seeds
                    or record.accepted_shape_proposal
                    != shape_proposal_count - 1
                    or record.grounding_rejection_seeds
                    != expected_shape_seeds[:-1]
                    or record.material_seed != expected_material_seed
                    or record.yaw_seed != expected_yaw_seed
                    or item.shape.to_dict() != expected_shape.to_dict()
                    or item.shape_generation_report is None
                    or item.shape_generation_report.to_dict()
                    != expected_shape_report.to_dict()
                    or rejected_shape_error
                )
                pose_errors += int(
                    np.max(np.abs(
                        expected_rotation
                        - np.asarray(item.rotation_world_from_local)
                    )) > 1.0e-10
                    or np.max(np.abs(
                        expected_translation
                        - np.asarray(item.translation_world_m)
                    )) > 1.0e-10
                )
        final_witness_cache: dict[int, np.ndarray] = {}
        pair_errors = sum(
            int(obvious_pair_penetration(
                world.objects[left], world.objects[right],
                witness_cache=final_witness_cache,
            )[0])
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
            "world_random_stream_error": world_random_stream_errors,
            "support_error": support_errors,
            "pose_error": pose_errors,
            "material_error": material_errors,
            "grounding_error": grounding_errors,
            "collision_error": collision_errors,
            "control_visibility_error": control_visibility_errors,
            "control_range_error": control_range_errors,
            "control_support_stream_error": control_support_stream_errors,
            "control_random_stream_error": control_random_stream_errors,
            "anomaly_random_stream_error": anomaly_random_stream_errors,
            "assigned_range_count": assigned_range_count.tolist(),
            "final_range_count": final_range_count.tolist(),
            "control_observation_json": json.dumps(
                control_observations, sort_keys=True, separators=(",", ":")
            ),
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
        "control_observation_json": np.asarray([
            str(item.get("control_observation_json", "")).encode()
            for item in records
        ]),
        "assigned_range_count": np.asarray([
            item.get("assigned_range_count", (0, 0, 0, 0, 0))
            for item in records
        ], dtype=np.int16),
        "final_range_count": np.asarray([
            item.get("final_range_count", (0, 0, 0, 0, 0))
            for item in records
        ], dtype=np.int16),
        "request_manifest_hash": values("request_manifest_hash", "S64", ""),
        "round_trip_error": values("round_trip_error", np.uint8, 0),
        "validation_error": values("validation_error", np.uint8, 0),
        "world_random_stream_error": values(
            "world_random_stream_error", np.uint8, 0
        ),
        "support_error": values("support_error", np.uint8, 0),
        "pose_error": values("pose_error", np.uint8, 0),
        "material_error": values("material_error", np.uint8, 0),
        "grounding_error": values("grounding_error", np.uint8, 0),
        "collision_error": values("collision_error", np.uint8, 0),
        "control_visibility_error": values(
            "control_visibility_error", np.uint8, 0
        ),
        "control_range_error": values("control_range_error", np.uint8, 0),
        "control_support_stream_error": values(
            "control_support_stream_error", np.uint8, 0
        ),
        "control_random_stream_error": values(
            "control_random_stream_error", np.uint8, 0
        ),
        "anomaly_random_stream_error": values(
            "anomaly_random_stream_error", np.uint8, 0
        ),
        "pair_error": values("pair_error", np.uint8, 0),
        "traversal_error": values("traversal_error", np.uint8, 0),
        "hard_error_code": values("hard_error", np.uint8, 1),
        "placement_exhaustion_code": values("placement_exhaustion", np.uint8, 0),
        "error_message": values("error", "U512", ""),
    }


def _e26_single_manifest_errors(records: Sequence[Mapping[str, object]]) -> int:
    """Rebuild worker manifests serially without repeating geometry generation."""

    errors = 0
    for record in records:
        if record.get("hard_error", 1) or record.get("placement_exhaustion", 0):
            continue
        world = WorldSpec.from_dict(json.loads(str(record["world_json"])))
        report = WorldGenerationReport.from_dict(
            json.loads(str(record["report_json"]))
        )
        center = 2 + world.seed % 445
        requests = {
            frame_id: _e26_request_identity(world.identity, frame_id)
            for frame_id in range(center - 2, center + 3)
        }
        request_hash = hashlib.sha256(
            json.dumps(requests, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        errors += int(
            world.to_json() != record["world_json"]
            or report.to_json() != record["report_json"]
            or request_hash != record["request_manifest_hash"]
        )
    return errors


def _placement_authority_errors(source: str) -> int:
    """Audit function structure without counting audit string literals."""

    syntax = ast.parse(source)
    function_names = [
        node.name for node in ast.walk(syntax)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    grounded_calls = [
        node for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_grounded_object"
    ]
    return int(
        function_names.count("place_object") != 1
        or len(grounded_calls) != 1
        or "generate_fixed_development_worlds" in function_names
    )


def run_e26_v2_qualification(
    data_root: Path | str,
    support_pool_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Run the single frozen E26-v2 production-world qualification."""

    if processes != 24:
        raise PlacementError("formal E26-v2 requires exactly 24 worker processes")
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
    template_identities, template_counts, template_library_hash = (
        canonical_normal_template_library_identity(templates)
    )
    pool = load_qualified_support_pool(support_pool_path)
    calibration_path_resolved = Path(calibration_path).expanduser().resolve(strict=True)
    calibration_hash = _sha256_path(calibration_path_resolved)
    if calibration_hash != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise PlacementError("E26-v2 sensor calibration identity changed")
    grid, sensor = load_sensor_calibration(calibration_path_resolved)
    if (
        sensor.source_sequence_id != 206
        or grid.calibration_frame_ids != sequence.frame_ids
        or any(frame.slot_count != grid.slot_count for frame in frames)
    ):
        raise PlacementError("E26-v2 sensor calibration provenance changed")
    obstacles = collect_observed_obstacle_index(frames)
    control_context = build_coverage_control_context(
        frames, pool, grid, sensor
    )
    precompute_coverage_control_support_streams(control_context, templates)
    renderer_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    global _E26_SUPPORT_POOL, _E26_OBSTACLES, _E26_TEMPLATES
    global _E26_TEMPLATE_INDEX, _E26_CONTROL_CONTEXT
    global _E26_TRAJECTORY_YAW, _E26_RENDERER_IDENTITY
    _E26_SUPPORT_POOL = pool
    _E26_OBSTACLES = obstacles
    _E26_TEMPLATES = templates
    _E26_TEMPLATE_INDEX = {
        identity: index for index, identity in enumerate(template_identities)
    }
    _E26_CONTROL_CONTEXT = control_context
    _E26_TRAJECTORY_YAW = control_context.trajectory_yaw_by_frame
    _E26_RENDERER_IDENTITY = renderer_identity

    source = Path(__file__).read_text(encoding="utf-8")
    authority_errors = _placement_authority_errors(source)
    work_order = tuple(sorted(
        range(256),
        key=lambda index: (
            -sum(_training_entity_counts(
                WORLD_TYPES[index // 64], 2_600_000 + index
            )),
            index,
        ),
    ))
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        scheduled = workers.map(_e26_worker, work_order, chunksize=1)
    by_seed = {int(item["world_seed"]): item for item in scheduled}
    records = [by_seed[2_600_000 + index] for index in range(256)]
    run_seconds = time.monotonic() - started
    single_manifest_errors = _e26_single_manifest_errors(records)
    first = _e26_arrays(records)
    completed = int(np.count_nonzero(first["world_hash"] != b""))
    type_errors = int(np.count_nonzero(
        first["world_type"]
        != np.repeat(np.asarray(WORLD_TYPES, dtype="U16"), 64)
    ))
    error_fields = (
        "round_trip_error", "validation_error", "world_random_stream_error",
        "support_error", "pose_error", "material_error", "grounding_error",
        "collision_error", "control_visibility_error", "control_range_error",
        "control_support_stream_error", "control_random_stream_error",
        "anomaly_random_stream_error",
        "pair_error", "traversal_error", "hard_error_code",
        "placement_exhaustion_code",
    )
    errors = {name: int(np.sum(first[name])) for name in error_fields}
    passed = (
        completed == 256 and type_errors == 0 and authority_errors == 0
        and single_manifest_errors == 0
        and all(value == 0 for value in errors.values())
    )
    scientific_hash = _scientific_array_hash(first)
    assigned_range_count = np.sum(first["assigned_range_count"], axis=0)
    final_range_count = np.sum(first["final_range_count"], axis=0)
    observations = [
        observation
        for payload in first["control_observation_json"]
        if payload
        for observation in json.loads(payload.decode())
    ]
    nvis = np.asarray(
        [observation["visible_returns"] for observation in observations],
        dtype=np.int64,
    )
    implementation_errors = (
        authority_errors + single_manifest_errors + type_errors
        + sum(
            value for name, value in errors.items()
            if name != "placement_exhaustion_code"
        )
    )
    metadata = {
        "experiment": "E26-v2", "passed": passed,
        "failure_classification": (
            None if passed else (
                "protocol_implementation_defect"
                if implementation_errors else "multi_entity_world_sampling_failure"
            )
        ),
        "worlds": 256,
        "completed": completed, "type_errors": type_errors,
        "authority_errors": authority_errors,
        "single_manifest_errors": single_manifest_errors, **errors,
        "formal_repetitions": 1,
        "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "run_seconds": [run_seconds],
        "normal_controls": int(np.sum(first["normal_count"])),
        "anomaly_proxies": int(np.sum(first["anomaly_count"])),
        "assigned_range_count": assigned_range_count.tolist(),
        "final_range_count": final_range_count.tolist(),
        "Nvis": {
            "minimum": int(np.min(nvis)) if nvis.size else None,
            "median": float(np.median(nvis)) if nvis.size else None,
            "mean": float(np.mean(nvis)) if nvis.size else None,
            "maximum": int(np.max(nvis)) if nvis.size else None,
        },
        "renderer_identity": renderer_identity,
        "normal_template_counts": template_counts,
        "normal_template_library_sha256": template_library_hash,
        "support_pool_sha256": _sha256_path(
            Path(support_pool_path).expanduser().resolve(strict=True)
        ),
        "calibration_sha256": calibration_hash,
        "scientific_array_hash": scientific_hash, "processes": processes,
        "numeric_library_threads_per_process": 1,
        "gpu_used": False,
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


_E27_TEMPLATES: tuple[NormalTemplateShape, ...] = ()
_E27_SENSOR = SensorCalibration.constant(1.0, return_probability=1.0)
_E27_NVIS_UNITS: tuple[tuple[WorldSpec, int, SourceFrame], ...] = ()
_E27_RAY_GRID: RayGrid | None = None


def _e27_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2701]))
    yaw = float(rng.uniform(-math.pi, math.pi))
    pitch, roll = rng.uniform(-math.radians(15.0), math.radians(15.0), size=2)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    return rz @ ry @ rx


def _convex_entry_distance(
    shape: NormalTemplateShape, origin: np.ndarray, direction: np.ndarray
) -> float:
    numerator = -(origin @ shape.plane_normals.T + shape.plane_offsets)
    denominator = direction @ shape.plane_normals.T
    if np.any((np.abs(denominator) <= EPSILON) & (numerator < 0.0)):
        return math.inf
    lower = np.full(denominator.shape, -np.inf)
    upper = np.full(denominator.shape, np.inf)
    np.divide(numerator, denominator, out=lower, where=denominator < -EPSILON)
    np.divide(numerator, denominator, out=upper, where=denominator > EPSILON)
    entry, exit_distance = float(np.max(lower)), float(np.min(upper))
    return entry if exit_distance >= max(entry, 0.0) and entry > 1.0e-5 else math.inf


def _e27_worker(index: int) -> dict[str, object]:
    if len(_E27_TEMPLATES) != 256:
        raise RuntimeError("E27 template fixtures are not initialized")
    seed = 2_700_000 + index
    shape = _E27_TEMPLATES[index]
    target_slot = index
    beam_id, column_id = divmod(target_slot, 2)
    elevation = math.radians(-20.0 + 40.0 * beam_id / 127.0)
    azimuth = math.pi * column_id + math.radians((index % 17) - 8)
    target = np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    target /= np.linalg.norm(target)
    directions = np.tile(-target, (256, 1))
    directions[target_slot] = target
    grid = RayGrid(
        directions,
        np.linspace(-math.radians(20.0), math.radians(20.0), 128),
        np.asarray((0.0, math.pi)),
        beam_count=128,
    )
    rotation = _e27_rotation(seed)
    center = np.mean(shape.vertices_m, axis=0)
    desired_distance = 2.5 + 47.5 * index / 255.0
    translation = target * (desired_distance + shape.bound_radius_m) - center @ rotation.T
    local_origin = -translation @ rotation
    local_direction = target @ rotation
    provisional = _convex_entry_distance(shape, local_origin, local_direction)
    if not math.isfinite(provisional):
        raise PlacementError("E27 reference failed to bracket the target hull")
    translation += target * (desired_distance - provisional)
    local_origin = -translation @ rotation
    expected_distance = _convex_entry_distance(
        shape, local_origin, local_direction
    )
    item = ObjectSpec(
        index + 1, "normal-control", shape, MaterialSpec.sample(seed + 2702),
        tuple(map(float, translation)),
        tuple(tuple(map(float, row)) for row in rotation),
    )
    world = WorldSpec(seed, 206, (item,))
    competition = _accepted_object_hits(
        np.zeros((256, 3), dtype=np.float64), directions, world, grid,
        _E27_SENSOR, frame_id=index,
    )
    measured = float(competition.distance_m[target_slot])
    hit_error = int(not math.isfinite(measured) or measured <= 0.0)
    miss_error = int(
        np.count_nonzero(np.isfinite(competition.distance_m)) != 1
    )
    distance_error = (
        math.inf if hit_error else abs(measured - expected_distance)
    )
    point_world = measured * target if not hit_error else np.zeros(3)
    point_local = (point_world - translation) @ rotation
    residual = (
        math.inf if hit_error
        else abs(float(shape.signed_distance(point_local[None, :])[0]))
    )
    normal = competition.normal_world[target_slot]
    normal_error = (
        math.inf if hit_error else abs(float(np.linalg.norm(normal)) - 1.0)
    )
    outward_error = int(hit_error or float(np.dot(normal, target)) >= 0.0)
    object_id_error = int(
        competition.object_id[target_slot] != index + 1
        or np.any(np.delete(competition.object_id, target_slot) != -1)
    )
    return {
        "seed": seed,
        "semantic": shape.raw_semantic_id,
        "template_identity": _normal_template_identity(shape),
        "beam_id": beam_id,
        "column_id": column_id,
        "desired_distance_m": desired_distance,
        "measured_distance_m": measured,
        "distance_error_m": distance_error,
        "surface_residual_m": residual,
        "normal_norm_error": normal_error,
        "hit_error": hit_error,
        "miss_error": miss_error,
        "outward_error": outward_error,
        "object_id_error": object_id_error,
    }


def _e27_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object) -> np.ndarray:
        return np.asarray([item[name] for item in records], dtype=dtype)

    return {
        "seed": values("seed", np.int64),
        "semantic": values("semantic", np.uint16),
        "template_identity": values("template_identity", "S64"),
        "beam_id": values("beam_id", np.int16),
        "column_id": values("column_id", np.int8),
        "desired_distance_m": values("desired_distance_m", np.float64),
        "measured_distance_m": values("measured_distance_m", np.float64),
        "distance_error_m": values("distance_error_m", np.float64),
        "surface_residual_m": values("surface_residual_m", np.float64),
        "normal_norm_error": values("normal_norm_error", np.float64),
        "hit_error": values("hit_error", np.uint8),
        "miss_error": values("miss_error", np.uint8),
        "outward_error": values("outward_error", np.uint8),
        "object_id_error": values("object_id_error", np.uint8),
    }


def _e27_nvis_worker(index: int) -> int:
    if _E27_RAY_GRID is None or not _E27_NVIS_UNITS:
        raise RuntimeError("E27 real-placement visibility units are not initialized")
    world, object_id, frame = _E27_NVIS_UNITS[index]
    rotation, lidar_origin = _pose(frame)
    directions_world = _E27_RAY_GRID.directions_for(frame) @ rotation.T
    origins_world = _E27_RAY_GRID.origins_for(frame) @ rotation.T + lidar_origin
    competition = _accepted_object_hits(
        origins_world, directions_world, world, _E27_RAY_GRID,
        _E27_SENSOR, int(frame.frame_id),
    )
    native = np.asarray(_E27_RAY_GRID.ranges(frame)).copy()
    native[np.asarray(frame.zero_slot_mask, dtype=np.bool_)] = np.inf
    visible = (
        np.isfinite(competition.distance_m)
        & (competition.distance_m < native - world.tie_tolerance_m)
        & (competition.object_id == object_id)
    )
    return int(np.count_nonzero(visible))


def run_e27_qualification(
    e25_artifact_path: Path | str,
    e26_artifact_path: Path | str,
    data_root: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Run the frozen return-probability-one normal-hull hit qualification."""

    if processes != 24:
        raise RenderError("formal E27 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    with np.load(Path(e25_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E25" or metadata.get("passed") is not True:
            raise RenderError("E27 requires the passed formal E25 artifact")
        selected: dict[str, NormalTemplateShape] = {}
        for identity_bytes, object_bytes in zip(
            source["template_identity"], source["object_json"], strict=True
        ):
            identity = identity_bytes.decode()
            if identity and identity not in selected:
                item = ObjectSpec.from_dict(json.loads(object_bytes.decode()))
                if not isinstance(item.shape, NormalTemplateShape):
                    raise RenderError("E25 normal-control artifact contains a non-template")
                selected[identity] = item.shape
    templates = tuple(
        selected[identity]
        for identity in sorted(
            selected, key=lambda value: (selected[value].raw_semantic_id, value)
        )
    )
    counts = {
        semantic: sum(item.raw_semantic_id == semantic for item in templates)
        for semantic in (10, 18, 20, 30)
    }
    if len(templates) != 256 or counts != {10: 64, 18: 64, 20: 64, 30: 64}:
        raise RenderError("E27 active template coverage changed")
    global _E27_TEMPLATES
    _E27_TEMPLATES = templates
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    context = mp.get_context("fork")
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e27_worker, range(256))
        runs.append(_e27_arrays(records))
        run_seconds.append(time.monotonic() - started)
    with np.load(Path(e26_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        e26_metadata = json.loads(str(source["metadata_json"]))
        if e26_metadata.get("experiment") != "E26" or e26_metadata.get("passed") is not True:
            raise RenderError("E27 visibility description requires the passed E26 artifact")
        units_json = [
            (world_json.decode(), report_json.decode())
            for world_json, report_json, count in zip(
                source["world_json"], source["report_json"],
                source["entity_count"], strict=True,
            )
            if int(count) > 0
        ]
    if len(units_json) != 192:
        raise RenderError("E27 real-placement visibility sample changed")
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    parsed_units = []
    required_frames: set[int] = set()
    for world_json, report_json in units_json:
        world = WorldSpec.from_dict(json.loads(world_json))
        report = WorldGenerationReport.from_dict(json.loads(report_json))
        frame_id = report.placements[0].support_frame
        parsed_units.append((world, world.objects[0].object_id, frame_id))
        required_frames.add(frame_id)
    frames = {
        frame_id: sequence.source_frame(frame_id) for frame_id in required_frames
    }
    ray_grid, _ = load_sensor_calibration(calibration_path)
    global _E27_NVIS_UNITS, _E27_RAY_GRID
    _E27_NVIS_UNITS = tuple(
        (world, object_id, frames[frame_id])
        for world, object_id, frame_id in parsed_units
    )
    _E27_RAY_GRID = ray_grid
    with context.Pool(processes=processes) as workers:
        nvis = np.asarray(workers.map(_e27_nvis_worker, range(192)), dtype=np.int64)
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    error_names = ("hit_error", "miss_error", "outward_error", "object_id_error")
    errors = {name: int(np.sum(first[name])) for name in error_names}
    passed = (
        all(value == 0 for value in errors.values())
        and float(np.max(first["distance_error_m"])) <= 1.0e-8
        and float(np.max(first["surface_residual_m"])) <= 1.0e-8
        and float(np.max(first["normal_norm_error"])) <= 1.0e-10
        and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E27", "passed": passed, "fixtures": 256,
        "active_counts": counts, **errors,
        "maximum_distance_error_m": float(np.max(first["distance_error_m"])),
        "maximum_surface_residual_m": float(np.max(first["surface_residual_m"])),
        "maximum_normal_norm_error": float(np.max(first["normal_norm_error"])),
        "descriptive_real_placement_nvis": {
            "objects": 192,
            "minimum": int(np.min(nvis)),
            "median": float(np.median(nvis)),
            "q95_higher": int(np.quantile(nvis, 0.95, method="higher")),
            "maximum": int(np.max(nvis)),
            "zero_count": int(np.count_nonzero(nvis == 0)),
        },
        "elementwise_reproduced": reproduced, "run_seconds": run_seconds,
        "scientific_array_hash": scientific_hash, "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first, descriptive_real_placement_nvis=nvis,
        metadata_json=np.asarray(
            json.dumps(result, sort_keys=True, separators=(",", ":"))
        ),
    )
    os.replace(temporary, destination)
    return result


_E28_SHAPES: tuple[tuple[ShapeSpec, ShapeGenerationReport], ...] = ()


def _sdf_first_entry(
    shape: ShapeSpec,
    origin: np.ndarray,
    direction: np.ndarray,
    nodes: int,
) -> float:
    radius = shape.bound_radius_m
    projection = -float(np.dot(origin, direction))
    discriminant = projection * projection - (
        float(np.dot(origin, origin)) - radius * radius
    )
    if discriminant <= 0.0:
        return math.inf
    span = math.sqrt(discriminant)
    lower = max(0.0, projection - span)
    upper = projection + span
    if upper <= lower:
        return math.inf
    distances = np.linspace(lower, upper, nodes, dtype=np.float64)
    values = shape.signed_distance(origin + distances[:, None] * direction)
    inside = values <= 0.0
    entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
    if not entries.size:
        return math.inf
    right = int(entries[0])
    return float(brentq(
        lambda distance: float(shape.signed_distance(
            (origin + distance * direction)[None, :]
        )[0]),
        float(distances[right - 1]), float(distances[right]),
        xtol=1.0e-12, rtol=1.0e-14,
    ))


def _e28_worker(index: int) -> dict[str, object]:
    if len(_E28_SHAPES) != 256:
        raise RuntimeError("E28 schema-7 fixtures are not initialized")
    seed = 2_800_000 + index
    shape, report = _E28_SHAPES[index]
    target_slot = index
    beam_id, column_id = divmod(target_slot, 2)
    elevation = math.radians(-20.0 + 40.0 * beam_id / 127.0)
    azimuth = math.pi * column_id + math.radians((index % 17) - 8)
    target = np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    target /= np.linalg.norm(target)
    directions = np.tile(-target, (256, 1))
    directions[target_slot] = target
    rotation = _e27_rotation(seed)
    undeformed = (
        np.asarray(report.shared_witnesses_undeformed_m[0])
        if report.shared_witnesses_undeformed_m
        else np.asarray(shape.primitive_offsets_m[0])
    )
    witness = _forward_deform(shape, undeformed[None, :])[0]
    witness_margin = -float(shape.signed_distance(witness[None, :])[0])
    if not math.isfinite(witness_margin) or witness_margin <= 1.0e-8:
        raise PlacementError("E28 target witness is not strictly interior")
    desired_distance = 2.5 + 47.5 * index / 255.0
    translation = (
        target * (desired_distance + 2.0 * shape.bound_radius_m)
        - witness @ rotation.T
    )
    local_origin = -translation @ rotation
    local_direction = target @ rotation
    reference_standard = _sdf_first_entry(
        shape, local_origin, local_direction, 4097
    )
    reference_strict = _sdf_first_entry(
        shape, local_origin, local_direction, 16385
    )
    if not math.isfinite(reference_standard) or not math.isfinite(reference_strict):
        raise PlacementError("E28 reference failed to find the target entry")
    reference_disagreement = abs(reference_standard - reference_strict)
    translation += target * (desired_distance - reference_strict)
    local_origin = -translation @ rotation
    local_directions = directions @ rotation
    distance, local_normal, valid = shape.intersect(
        local_origin, local_directions
    )
    measured = float(distance[target_slot])
    hit_error = int(not valid[target_slot] or measured <= 0.0)
    miss_error = int(np.count_nonzero(valid) != 1)
    distance_error = math.inf if hit_error else abs(measured - desired_distance)
    point_world = measured * target if not hit_error else np.zeros(3)
    point_local = (point_world - translation) @ rotation
    residual = (
        math.inf if hit_error
        else abs(float(shape.signed_distance(point_local[None, :])[0]))
    )
    normal = local_normal[target_slot] @ rotation.T
    normal_error = math.inf if hit_error else abs(float(np.linalg.norm(normal)) - 1.0)
    outward_error = int(hit_error or float(np.dot(normal, target)) >= 0.0)
    return {
        "seed": seed,
        "shape_family": report.shape_family,
        "primitive_count": shape.primitive_count,
        "size_lower_m": report.accepted_size_lower_m,
        "size_upper_m": report.accepted_size_upper_m,
        "shape_identity": hashlib.sha256(
            json.dumps(shape.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "beam_id": beam_id,
        "column_id": column_id,
        "desired_distance_m": desired_distance,
        "measured_distance_m": measured,
        "reference_disagreement_m": reference_disagreement,
        "distance_error_m": distance_error,
        "surface_residual_m": residual,
        "normal_norm_error": normal_error,
        "witness_margin": witness_margin,
        "hit_error": hit_error,
        "miss_error": miss_error,
        "outward_error": outward_error,
    }


def _e28_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    def values(name: str, dtype: object) -> np.ndarray:
        return np.asarray([item[name] for item in records], dtype=dtype)

    return {
        "seed": values("seed", np.int64),
        "shape_family": values("shape_family", "U16"),
        "primitive_count": values("primitive_count", np.int8),
        "size_lower_m": values("size_lower_m", np.float64),
        "size_upper_m": values("size_upper_m", np.float64),
        "shape_identity": values("shape_identity", "S64"),
        "beam_id": values("beam_id", np.int16),
        "column_id": values("column_id", np.int8),
        "desired_distance_m": values("desired_distance_m", np.float64),
        "measured_distance_m": values("measured_distance_m", np.float64),
        "reference_disagreement_m": values("reference_disagreement_m", np.float64),
        "distance_error_m": values("distance_error_m", np.float64),
        "surface_residual_m": values("surface_residual_m", np.float64),
        "normal_norm_error": values("normal_norm_error", np.float64),
        "witness_margin": values("witness_margin", np.float64),
        "hit_error": values("hit_error", np.uint8),
        "miss_error": values("miss_error", np.uint8),
        "outward_error": values("outward_error", np.uint8),
    }


def run_e28_v2_qualification(
    e26_artifact_path: Path | str,
    data_root: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Run frozen E28-v2 directly at the continuous geometry interface."""

    if processes != 24:
        raise RenderError("formal E28-v2 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    context = mp.get_context("fork")
    with context.Pool(processes=processes) as workers:
        fixtures = workers.map(
            ShapeSpec.sample_with_report, range(2_800_000, 2_800_256)
        )
    global _E28_SHAPES
    _E28_SHAPES = tuple(fixtures)
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e28_worker, range(256))
        runs.append(_e28_arrays(records))
        run_seconds.append(time.monotonic() - started)

    with np.load(Path(e26_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E26" or metadata.get("passed") is not True:
            raise RenderError("E28-v2 visibility description requires the passed E26 artifact")
        units_json = []
        for world_bytes, report_bytes in zip(
            source["world_json"], source["report_json"], strict=True
        ):
            world = WorldSpec.from_dict(json.loads(world_bytes.decode()))
            if world.anomaly_proxy_count:
                units_json.append((world, WorldGenerationReport.from_dict(
                    json.loads(report_bytes.decode())
                )))
    if len(units_json) != 128:
        raise RenderError("E28-v2 real-proxy visibility sample changed")
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    parsed_units = []
    required_frames: set[int] = set()
    for world, report in units_json:
        item = next(value for value in world.objects if value.label == "anomaly-proxy")
        placement = next(value for value in report.placements if value.object_id == item.object_id)
        parsed_units.append((world, item.object_id, placement.support_frame))
        required_frames.add(placement.support_frame)
    frames = {frame_id: sequence.source_frame(frame_id) for frame_id in required_frames}
    ray_grid, _ = load_sensor_calibration(calibration_path)
    global _E27_NVIS_UNITS, _E27_RAY_GRID
    _E27_NVIS_UNITS = tuple(
        (world, object_id, frames[frame_id])
        for world, object_id, frame_id in parsed_units
    )
    _E27_RAY_GRID = ray_grid
    with context.Pool(processes=processes) as workers:
        nvis = np.asarray(workers.map(_e27_nvis_worker, range(128)), dtype=np.int64)

    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name], equal_nan=True)
        if np.issubdtype(runs[0][name].dtype, np.floating)
        else np.array_equal(runs[0][name], runs[1][name])
        for name in runs[0]
    )
    first = runs[0]
    error_names = ("hit_error", "miss_error", "outward_error")
    errors = {name: int(np.sum(first[name])) for name in error_names}
    passed = (
        all(value == 0 for value in errors.values())
        and float(np.max(first["reference_disagreement_m"])) < 5.0e-5
        and float(np.max(first["distance_error_m"])) <= 1.0e-4
        and float(np.max(first["surface_residual_m"])) <= 1.0e-6
        and float(np.max(first["normal_norm_error"])) <= 1.0e-10
        and reproduced
    )
    family_counts = {
        family: int(np.count_nonzero(first["shape_family"] == family))
        for family in SHAPE_FAMILIES
    }
    primitive_counts = {
        count: int(np.count_nonzero(first["primitive_count"] == count))
        for count in range(1, 6)
    }
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E28-v2", "passed": passed, "fixtures": 256,
        "family_counts": family_counts, "primitive_counts": primitive_counts,
        **errors,
        "maximum_reference_disagreement_m": float(np.max(first["reference_disagreement_m"])),
        "maximum_distance_error_m": float(np.max(first["distance_error_m"])),
        "maximum_surface_residual_m": float(np.max(first["surface_residual_m"])),
        "maximum_normal_norm_error": float(np.max(first["normal_norm_error"])),
        "descriptive_real_proxy_nvis": {
            "objects": 128, "minimum": int(np.min(nvis)),
            "median": float(np.median(nvis)),
            "q95_higher": int(np.quantile(nvis, 0.95, method="higher")),
            "maximum": int(np.max(nvis)),
            "zero_count": int(np.count_nonzero(nvis == 0)),
        },
        "elementwise_reproduced": reproduced, "run_seconds": run_seconds,
        "scientific_array_hash": scientific_hash, "processes": processes,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first, descriptive_real_proxy_nvis=nvis,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E29_SENSOR: SensorCalibration | None = None


def _e29_reference_uniform(
    world_seed: int,
    source_sequence_id: int,
    frame_id: int,
    slots: np.ndarray,
    object_ids: np.ndarray,
    channel: int,
) -> np.ndarray:
    """Independently reproduce the frozen 64-bit identity mixer."""

    mask = (1 << 64) - 1
    base = (
        world_seed
        ^ (source_sequence_id << 24)
        ^ (frame_id << 40)
        ^ (channel * 0xA24BAED4963EE407)
    ) & mask
    with np.errstate(over="ignore"):
        value = np.asarray(slots, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        value ^= np.asarray(object_ids, dtype=np.uint64) * np.uint64(
            0xD1B54A32D192ED03
        )
        value ^= np.uint64(base)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return (value.astype(np.float64) + 0.5) / float(2**64)


def _e29_worker(identity: int) -> dict[str, np.ndarray]:
    if _E29_SENSOR is None:
        raise RuntimeError("E29 sensor calibration is not initialized")
    sensor = _E29_SENSOR
    beam, range_bin, incidence_bin = np.indices(
        sensor.return_probability.shape, dtype=np.int64
    )
    beam = beam.ravel()
    range_bin = range_bin.ravel()
    incidence_bin = incidence_bin.ravel()
    ranges = 0.5 * (
        sensor.range_edges_m[range_bin] + sensor.range_edges_m[range_bin + 1]
    )
    incidence = 0.5 * (
        sensor.incidence_edges_rad[incidence_bin]
        + sensor.incidence_edges_rad[incidence_bin + 1]
    )
    bias = MaterialSpec.sample(2_900_000 + identity).return_bias
    probability = sensor.return_chance(beam, ranges, incidence, bias)
    base = sensor.return_probability[beam, range_bin, incidence_bin]
    clipped = np.clip(base, 1.0e-5, 1.0 - 1.0e-5)
    reference_probability = 1.0 / (
        1.0 + np.exp(-(np.log(clipped / (1.0 - clipped)) + 2.0 * bias))
    )
    slots = np.arange(base.size, dtype=np.int32)
    object_ids = np.full(base.size, identity + 1, dtype=np.int32)
    world = WorldSpec(2_900_000 + identity, 206, ())
    frame_id = 1_000 + identity
    uniform = _slot_uniform(
        world, frame_id, slots, object_ids, channel=0
    )
    reference_uniform = _e29_reference_uniform(
        world.seed, world.source_sequence_id, frame_id,
        slots, object_ids, 0,
    )
    return {
        "identity": np.full(base.size, identity, dtype=np.int16),
        "beam": beam.astype(np.int16),
        "range_bin": range_bin.astype(np.int8),
        "incidence_bin": incidence_bin.astype(np.int8),
        "material_bias": np.full(base.size, bias, dtype=np.float64),
        "probability": probability,
        "reference_probability": reference_probability,
        "uniform": uniform,
        "reference_uniform": reference_uniform,
        "accepted": uniform < probability,
        "reference_accepted": reference_uniform < reference_probability,
    }


def run_e29_qualification(
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Qualify return probabilities and deterministic identity sampling."""

    if processes != 24:
        raise RenderError("formal E29 requires exactly 24 worker processes")
    _, sensor = load_sensor_calibration(calibration_path)
    global _E29_SENSOR
    _E29_SENSOR = sensor
    context = mp.get_context("fork")
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e29_worker, range(24))
        names = tuple(records[0])
        runs.append({
            name: np.concatenate([record[name] for record in records])
            for name in names
        })
        run_seconds.append(time.monotonic() - started)
    first = runs[0]
    reproduced = all(
        np.array_equal(first[name], runs[1][name]) for name in first
    )
    base = sensor.return_probability
    pooled_opportunities = sensor.opportunity_counts.sum(axis=0)
    pooled_returns = sensor.return_counts.sum(axis=0)
    reference_base = np.empty_like(base)
    fallback = sensor.opportunity_counts < 64
    for beam in range(base.shape[0]):
        local = ~fallback[beam]
        reference_base[beam, local] = (
            sensor.return_counts[beam, local] + 0.5
        ) / (sensor.opportunity_counts[beam, local] + 1.0)
        reference_base[beam, ~local] = (
            pooled_returns[~local] + 0.5
        ) / (pooled_opportunities[~local] + 1.0)
    provenance = dict(sensor.provenance)
    fallback_traceable = (
        np.array_equal(sensor.fallback_mask, fallback)
        and provenance.get("return_estimator")
        == "jeffreys_beta_smoothed_binomial_rate"
        and provenance.get("return_low_count_fallback")
        == "cross_beam_same_range_incidence_below_64_opportunities"
    )
    probability_domain_errors = int(np.count_nonzero(
        ~np.isfinite(base) | (base < 0.0) | (base > 1.0)
    ))
    modulated_domain_errors = int(np.count_nonzero(
        ~np.isfinite(first["probability"])
        | (first["probability"] < 0.0)
        | (first["probability"] > 1.0)
    ))
    maximum_base_error = float(np.max(np.abs(base - reference_base)))
    maximum_probability_error = float(np.max(np.abs(
        first["probability"] - first["reference_probability"]
    )))
    maximum_uniform_error = float(np.max(np.abs(
        first["uniform"] - first["reference_uniform"]
    )))
    accepted_mask_errors = int(np.count_nonzero(
        first["accepted"] != first["reference_accepted"]
    ))
    p0_errors = int(np.count_nonzero(first["uniform"] < 0.0))
    p1_errors = int(np.count_nonzero(~(first["uniform"] < 1.0)))
    accepted = int(np.count_nonzero(first["accepted"]))
    rejected = int(first["accepted"].size - accepted)
    passed = (
        probability_domain_errors == 0
        and modulated_domain_errors == 0
        and maximum_base_error == 0.0
        and maximum_probability_error == 0.0
        and maximum_uniform_error == 0.0
        and accepted_mask_errors == 0
        and p0_errors == 0
        and p1_errors == 0
        and accepted > 0 and rejected > 0
        and fallback_traceable and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E29", "passed": passed,
        "calibration_cells": int(base.size),
        "identity_trials_per_cell": 24,
        "decisions": int(first["accepted"].size),
        "probability_domain_errors": probability_domain_errors,
        "modulated_domain_errors": modulated_domain_errors,
        "maximum_base_probability_error": maximum_base_error,
        "maximum_modulated_probability_error": maximum_probability_error,
        "maximum_uniform_error": maximum_uniform_error,
        "accepted_mask_errors": accepted_mask_errors,
        "p0_errors": p0_errors, "p1_errors": p1_errors,
        "intermediate_accepted": accepted,
        "intermediate_rejected": rejected,
        "fallback_cells": int(np.count_nonzero(sensor.fallback_mask)),
        "fallback_traceable": fallback_traceable,
        "elementwise_reproduced": reproduced,
        "run_seconds": run_seconds, "processes": processes,
        "scientific_array_hash": scientific_hash,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E30_SENSOR: SensorCalibration | None = None


def _e30_worker(index: int) -> dict[str, np.ndarray]:
    if len(_E27_TEMPLATES) != 256 or _E30_SENSOR is None:
        raise RuntimeError("E30 fixtures are not initialized")
    sensor = _E30_SENSOR
    shape = _E27_TEMPLATES[index]
    seed = 2_700_000 + index
    beam_id, column_id = divmod(index, 2)
    elevation = math.radians(-20.0 + 40.0 * beam_id / 127.0)
    azimuth = math.pi * column_id + math.radians((index % 17) - 8)
    target = np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    target /= np.linalg.norm(target)
    rotation = _e27_rotation(seed)
    center = np.mean(shape.vertices_m, axis=0)
    desired_distance = 2.5 + 47.5 * index / 255.0
    translation = (
        target * (desired_distance + shape.bound_radius_m)
        - center @ rotation.T
    )
    provisional = _convex_entry_distance(
        shape, -translation @ rotation, target @ rotation
    )
    translation += target * (desired_distance - provisional)
    distance, local_normal, valid = shape.intersect(
        -translation @ rotation, (target @ rotation)[None, :]
    )
    geometry_error = int(not valid[0] or distance[0] <= 0.0)
    world_normal = local_normal[0] @ rotation.T
    incidence = math.acos(np.clip(abs(float(world_normal @ -target)), 0.0, 1.0))
    material = MaterialSpec.sample(seed + 2702)
    probability = float(sensor.return_chance(
        np.asarray((beam_id,)), distance,
        np.asarray((incidence,)), material.return_bias,
    )[0])
    base = float(sensor.return_probability[
        beam_id,
        np.clip(np.searchsorted(sensor.range_edges_m, distance[0], side="right") - 1, 0, 5),
        np.clip(np.searchsorted(sensor.incidence_edges_rad, incidence, side="right") - 1, 0, 2),
    ])
    clipped = float(np.clip(base, 1.0e-5, 1.0 - 1.0e-5))
    reference_probability = 1.0 / (
        1.0 + math.exp(-(math.log(clipped / (1.0 - clipped)) + 2.0 * material.return_bias))
    )
    replicas = np.arange(24, dtype=np.int64)
    frame_ids = replicas * 256 + index
    slots = np.full(24, index, dtype=np.int32)
    object_ids = np.full(24, index + 1, dtype=np.int32)
    world = WorldSpec(seed, 206, ())
    uniform = np.asarray([
        _slot_uniform(
            world, int(frame_id), slots[replica : replica + 1],
            object_ids[replica : replica + 1], channel=0,
        )[0]
        for replica, frame_id in enumerate(frame_ids)
    ])
    reference_uniform = np.asarray([
        _e29_reference_uniform(
            world.seed, world.source_sequence_id, int(frame_id),
            slots[replica : replica + 1], object_ids[replica : replica + 1], 0,
        )[0]
        for replica, frame_id in enumerate(frame_ids)
    ])
    accepted = uniform < probability
    reference_accepted = reference_uniform < reference_probability
    points = np.full((24, 3), np.nan, dtype=np.float64)
    points[accepted] = distance[0] * target
    intensity = np.full(24, np.nan, dtype=np.float32)
    accepted_ids = np.flatnonzero(accepted)
    if accepted_ids.size:
        intensity_uniform = np.asarray([
            _slot_uniform(
                world, int(frame_ids[replica]),
                slots[replica : replica + 1], object_ids[replica : replica + 1],
                channel=1,
            )[0]
            for replica in accepted_ids
        ])
        intensity[accepted] = sensor.sample_intensity(
            np.full(accepted_ids.size, beam_id, dtype=np.int16),
            np.full(accepted_ids.size, distance[0], dtype=np.float64),
            np.full(accepted_ids.size, incidence, dtype=np.float64),
            intensity_uniform, material,
        )
    semantic = np.zeros(24, dtype=np.uint16)
    semantic[accepted] = np.uint16(shape.raw_semantic_id)
    return {
        "fixture_index": np.full(24, index, dtype=np.int16),
        "seed": np.full(24, seed, dtype=np.int64),
        "frame_id": frame_ids,
        "beam_id": np.full(24, beam_id, dtype=np.int16),
        "column_id": np.full(24, column_id, dtype=np.int8),
        "semantic_expected": np.full(24, shape.raw_semantic_id, dtype=np.uint16),
        "geometry_error": np.full(24, geometry_error, dtype=np.uint8),
        "distance_m": np.full(24, distance[0], dtype=np.float64),
        "probability": np.full(24, probability, dtype=np.float64),
        "reference_probability": np.full(24, reference_probability, dtype=np.float64),
        "uniform": uniform,
        "reference_uniform": reference_uniform,
        "accepted": accepted,
        "reference_accepted": reference_accepted,
        "point_world_m": points,
        "intensity": intensity,
        "semantic": semantic,
    }


def run_e30_qualification(
    e25_artifact_path: Path | str,
    e27_artifact_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Qualify normal-control accepted returns without native competition."""

    if processes != 24:
        raise RenderError("formal E30 requires exactly 24 worker processes")
    with np.load(Path(e27_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E27" or metadata.get("passed") is not True:
            raise RenderError("E30 requires the passed E27 artifact")
        e27_identity = source["template_identity"].copy()
    with np.load(Path(e25_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E25" or metadata.get("passed") is not True:
            raise RenderError("E30 requires the passed E25 artifact")
        selected: dict[str, NormalTemplateShape] = {}
        for identity_bytes, object_bytes in zip(
            source["template_identity"], source["object_json"], strict=True
        ):
            identity = identity_bytes.decode()
            if identity and identity not in selected:
                item = ObjectSpec.from_dict(json.loads(object_bytes.decode()))
                if not isinstance(item.shape, NormalTemplateShape):
                    raise RenderError("E30 E25 input contains a non-template")
                selected[identity] = item.shape
    identities = sorted(
        selected, key=lambda value: (selected[value].raw_semantic_id, value)
    )
    templates = tuple(selected[identity] for identity in identities)
    reconstructed_identity = np.asarray(
        [_normal_template_identity(template) for template in templates], dtype="S64"
    )
    if len(templates) != 256 or not np.array_equal(
        reconstructed_identity, e27_identity
    ):
        raise RenderError("E30 fixtures differ from E27")
    _, sensor = load_sensor_calibration(calibration_path)
    global _E27_TEMPLATES, _E30_SENSOR
    _E27_TEMPLATES = templates
    _E30_SENSOR = sensor
    context = mp.get_context("fork")
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e30_worker, range(256))
        runs.append({
            name: np.concatenate([record[name] for record in records])
            for name in records[0]
        })
        run_seconds.append(time.monotonic() - started)
    first = runs[0]
    reproduced = all(np.array_equal(
        first[name], runs[1][name], equal_nan=True
    ) for name in first)
    accepted = first["accepted"]
    rejected = ~accepted
    mask_errors = int(np.count_nonzero(
        accepted != first["reference_accepted"]
    ))
    probability_errors = int(np.count_nonzero(
        first["probability"] != first["reference_probability"]
    ))
    uniform_errors = int(np.count_nonzero(
        first["uniform"] != first["reference_uniform"]
    ))
    geometry_errors = int(np.count_nonzero(first["geometry_error"]))
    accepted_payload_errors = int(np.count_nonzero(
        ~np.isfinite(first["point_world_m"][accepted]).all(axis=1)
        | ~np.isfinite(first["intensity"][accepted])
        | (first["intensity"][accepted] < sensor.intensity_min)
        | (first["intensity"][accepted] > sensor.intensity_max)
        | (first["semantic"][accepted] != first["semantic_expected"][accepted])
    ))
    rejected_payload_errors = int(np.count_nonzero(
        np.isfinite(first["point_world_m"][rejected]).any(axis=1)
        | np.isfinite(first["intensity"][rejected])
        | (first["semantic"][rejected] != 0)
    ))
    accepted_count = int(np.count_nonzero(accepted))
    rejected_count = int(np.count_nonzero(rejected))
    passed = (
        geometry_errors == 0 and probability_errors == 0
        and uniform_errors == 0 and mask_errors == 0
        and accepted_payload_errors == 0 and rejected_payload_errors == 0
        and accepted_count > 0 and rejected_count > 0 and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E30", "passed": passed,
        "fixtures": 256, "identities_per_fixture": 24,
        "decisions": int(accepted.size),
        "geometry_errors": geometry_errors,
        "probability_errors": probability_errors,
        "uniform_errors": uniform_errors,
        "accepted_mask_errors": mask_errors,
        "accepted_payload_errors": accepted_payload_errors,
        "rejected_payload_errors": rejected_payload_errors,
        "accepted": accepted_count, "rejected": rejected_count,
        "elementwise_reproduced": reproduced,
        "run_seconds": run_seconds, "processes": processes,
        "scientific_array_hash": scientific_hash,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E31_SENSOR: SensorCalibration | None = None


def _e31_worker(index: int) -> dict[str, np.ndarray]:
    if len(_E28_SHAPES) != 256 or _E31_SENSOR is None:
        raise RuntimeError("E31 fixtures are not initialized")
    sensor = _E31_SENSOR
    shape, report = _E28_SHAPES[index]
    seed = 2_800_000 + index
    beam_id, column_id = divmod(index, 2)
    elevation = math.radians(-20.0 + 40.0 * beam_id / 127.0)
    azimuth = math.pi * column_id + math.radians((index % 17) - 8)
    target = np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    target /= np.linalg.norm(target)
    rotation = _e27_rotation(seed)
    undeformed = (
        np.asarray(report.shared_witnesses_undeformed_m[0])
        if report.shared_witnesses_undeformed_m
        else np.asarray(shape.primitive_offsets_m[0])
    )
    witness = _forward_deform(shape, undeformed[None, :])[0]
    desired_distance = 2.5 + 47.5 * index / 255.0
    translation = (
        target * (desired_distance + 2.0 * shape.bound_radius_m)
        - witness @ rotation.T
    )
    provisional = _sdf_first_entry(
        shape, -translation @ rotation, target @ rotation, 16385
    )
    translation += target * (desired_distance - provisional)
    distance, local_normal, valid = shape.intersect(
        -translation @ rotation, (target @ rotation)[None, :]
    )
    geometry_error = int(not valid[0] or distance[0] <= 0.0)
    world_normal = local_normal[0] @ rotation.T
    incidence = math.acos(np.clip(abs(float(world_normal @ -target)), 0.0, 1.0))
    material = MaterialSpec.sample(seed + 2802)
    probability = float(sensor.return_chance(
        np.asarray((beam_id,)), distance,
        np.asarray((incidence,)), material.return_bias,
    )[0])
    range_bin = int(np.clip(
        np.searchsorted(sensor.range_edges_m, distance[0], side="right") - 1,
        0, sensor.range_edges_m.size - 2,
    ))
    incidence_bin = int(np.clip(
        np.searchsorted(sensor.incidence_edges_rad, incidence, side="right") - 1,
        0, sensor.incidence_edges_rad.size - 2,
    ))
    base = float(sensor.return_probability[beam_id, range_bin, incidence_bin])
    clipped = float(np.clip(base, 1.0e-5, 1.0 - 1.0e-5))
    reference_probability = 1.0 / (
        1.0 + math.exp(-(math.log(clipped / (1.0 - clipped)) + 2.0 * material.return_bias))
    )
    replicas = np.arange(24, dtype=np.int64)
    frame_ids = replicas * 256 + index
    slots = np.full(24, index, dtype=np.int32)
    object_ids = np.full(24, index + 1, dtype=np.int32)
    world = WorldSpec(seed, 206, ())
    uniform = np.asarray([
        _slot_uniform(
            world, int(frame_id), slots[replica : replica + 1],
            object_ids[replica : replica + 1], channel=0,
        )[0]
        for replica, frame_id in enumerate(frame_ids)
    ])
    reference_uniform = np.asarray([
        _e29_reference_uniform(
            world.seed, world.source_sequence_id, int(frame_id),
            slots[replica : replica + 1], object_ids[replica : replica + 1], 0,
        )[0]
        for replica, frame_id in enumerate(frame_ids)
    ])
    accepted = uniform < probability
    reference_accepted = reference_uniform < reference_probability
    points = np.full((24, 3), np.nan, dtype=np.float64)
    points[accepted] = distance[0] * target
    intensity = np.full(24, np.nan, dtype=np.float32)
    accepted_ids = np.flatnonzero(accepted)
    if accepted_ids.size:
        intensity_uniform = np.asarray([
            _slot_uniform(
                world, int(frame_ids[replica]),
                slots[replica : replica + 1], object_ids[replica : replica + 1],
                channel=1,
            )[0]
            for replica in accepted_ids
        ])
        intensity[accepted] = sensor.sample_intensity(
            np.full(accepted_ids.size, beam_id, dtype=np.int16),
            np.full(accepted_ids.size, distance[0], dtype=np.float64),
            np.full(accepted_ids.size, incidence, dtype=np.float64),
            intensity_uniform, material,
        )
    semantic = np.zeros(24, dtype=np.uint16)
    semantic[accepted] = np.uint16(2)
    internal_object_id = np.full(24, -1, dtype=np.int32)
    internal_object_id[accepted] = index + 1
    return {
        "fixture_index": np.full(24, index, dtype=np.int16),
        "seed": np.full(24, seed, dtype=np.int64),
        "frame_id": frame_ids,
        "beam_id": np.full(24, beam_id, dtype=np.int16),
        "column_id": np.full(24, column_id, dtype=np.int8),
        "geometry_error": np.full(24, geometry_error, dtype=np.uint8),
        "distance_m": np.full(24, distance[0], dtype=np.float64),
        "probability": np.full(24, probability, dtype=np.float64),
        "reference_probability": np.full(24, reference_probability, dtype=np.float64),
        "uniform": uniform,
        "reference_uniform": reference_uniform,
        "accepted": accepted,
        "reference_accepted": reference_accepted,
        "point_world_m": points,
        "intensity": intensity,
        "semantic": semantic,
        "internal_object_id": internal_object_id,
    }


def run_e31_qualification(
    e28_artifact_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Qualify anomaly-proxy accepted returns without native competition."""

    if processes != 24:
        raise RenderError("formal E31 requires exactly 24 worker processes")
    with np.load(Path(e28_artifact_path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E28-v2" or metadata.get("passed") is not True:
            raise RenderError("E31 requires the passed E28-v2 artifact")
        expected_seed = source["seed"].copy()
        expected_identity = source["shape_identity"].copy()
    context = mp.get_context("fork")
    with context.Pool(processes=processes) as workers:
        fixtures = workers.map(
            ShapeSpec.sample_with_report, range(2_800_000, 2_800_256)
        )
    identity = np.asarray([
        hashlib.sha256(json.dumps(
            shape.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        for shape, _ in fixtures
    ], dtype="S64")
    if not np.array_equal(expected_seed, np.arange(2_800_000, 2_800_256)) or not np.array_equal(
        identity, expected_identity
    ):
        raise RenderError("E31 fixtures differ from E28-v2")
    _, sensor = load_sensor_calibration(calibration_path)
    global _E28_SHAPES, _E31_SENSOR
    _E28_SHAPES = tuple(fixtures)
    _E31_SENSOR = sensor
    runs: list[dict[str, np.ndarray]] = []
    run_seconds: list[float] = []
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e31_worker, range(256))
        runs.append({
            name: np.concatenate([record[name] for record in records])
            for name in records[0]
        })
        run_seconds.append(time.monotonic() - started)
    first = runs[0]
    reproduced = all(np.array_equal(
        first[name], runs[1][name], equal_nan=True
    ) for name in first)
    accepted = first["accepted"]
    rejected = ~accepted
    geometry_errors = int(np.count_nonzero(first["geometry_error"]))
    probability_errors = int(np.count_nonzero(
        first["probability"] != first["reference_probability"]
    ))
    uniform_errors = int(np.count_nonzero(
        first["uniform"] != first["reference_uniform"]
    ))
    mask_errors = int(np.count_nonzero(
        accepted != first["reference_accepted"]
    ))
    accepted_payload_errors = int(np.count_nonzero(
        ~np.isfinite(first["point_world_m"][accepted]).all(axis=1)
        | ~np.isfinite(first["intensity"][accepted])
        | (first["intensity"][accepted] < sensor.intensity_min)
        | (first["intensity"][accepted] > sensor.intensity_max)
        | (first["semantic"][accepted] != 2)
        | (first["internal_object_id"][accepted]
           != first["fixture_index"][accepted].astype(np.int32) + 1)
    ))
    rejected_payload_errors = int(np.count_nonzero(
        np.isfinite(first["point_world_m"][rejected]).any(axis=1)
        | np.isfinite(first["intensity"][rejected])
        | (first["semantic"][rejected] != 0)
        | (first["internal_object_id"][rejected] != -1)
    ))
    accepted_count = int(np.count_nonzero(accepted))
    rejected_count = int(np.count_nonzero(rejected))
    passed = (
        geometry_errors == 0 and probability_errors == 0
        and uniform_errors == 0 and mask_errors == 0
        and accepted_payload_errors == 0 and rejected_payload_errors == 0
        and accepted_count > 0 and rejected_count > 0 and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E31", "passed": passed,
        "fixtures": 256, "identities_per_fixture": 24,
        "decisions": int(accepted.size),
        "geometry_errors": geometry_errors,
        "probability_errors": probability_errors,
        "uniform_errors": uniform_errors,
        "accepted_mask_errors": mask_errors,
        "accepted_payload_errors": accepted_payload_errors,
        "rejected_payload_errors": rejected_payload_errors,
        "accepted": accepted_count, "rejected": rejected_count,
        "elementwise_reproduced": reproduced,
        "run_seconds": run_seconds, "processes": processes,
        "scientific_array_hash": scientific_hash,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e32_run_once() -> dict[str, np.ndarray]:
    half = 0.25
    vertices = np.asarray([
        (x, y, z)
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
    ], dtype=np.float64)
    shape = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32), 206, 0, 10, 1,
        (0.0, 0.0, 0.0),
    )
    grid = RayGrid(
        np.asarray(((1.0, 0.0, 0.0),)), np.asarray((0.0,)),
        np.asarray((0.0,)), beam_count=1,
    )
    sensor = SensorCalibration.constant(0.5, return_probability=1.0)
    tie = 1.0e-6
    gaps = np.asarray((0.5e-6, 1.0e-6, 2.0e-6), dtype=np.float64)
    relation = np.asarray((-1, 0, 1), dtype=np.int8)
    records: dict[str, list[object]] = {
        name: [] for name in (
            "relation", "gap_m", "inserted_distance_m", "native_distance_m",
            "expected_inserted", "actual_inserted", "occluded_original",
            "single_return_error", "distance_error_m", "semantic_error",
            "object_id_error", "mask_error", "packed_error",
        )
    }
    for case, gap in enumerate(gaps):
        native_distance = float(np.float32(5.0))
        inserted_distance = native_distance - float(gap)
        item = ObjectSpec(
            1, "normal-control", shape, MaterialSpec(0.5, 0.1, 0.35),
            (inserted_distance + half, 0.0, 0.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        world = WorldSpec(3_200_000 + case, 206, (item,))
        semantic = np.asarray((10,), dtype=np.uint16)
        instance = np.asarray((1,), dtype=np.uint16)
        packed = semantic.astype(np.uint32) | (instance.astype(np.uint32) << 16)
        labels = PointLabels(
            packed, semantic, instance,
            np.asarray((NORMAL_SEMANTIC_TARGET[10],), dtype=np.uint8),
        )
        frame = make_source_frame(
            case, np.asarray(((native_distance, 0.0, 0.0, 0.25),), dtype=np.float32),
            np.eye(4, dtype=np.float64), labels,
            partition="train", sequence_id=206,
        )
        rendered = render_frame(frame, world, grid, sensor)
        expected = inserted_distance < native_distance - tie
        actual = bool(rendered.inserted_mask[0])
        measured = float(np.linalg.norm(rendered.source.xyzi[0, :3]))
        expected_distance = inserted_distance if expected else native_distance
        expected_semantic = 10
        expected_object = 1 if expected else -1
        records["relation"].append(int(relation[case]))
        records["gap_m"].append(float(gap))
        records["inserted_distance_m"].append(inserted_distance)
        records["native_distance_m"].append(native_distance)
        records["expected_inserted"].append(expected)
        records["actual_inserted"].append(actual)
        records["occluded_original"].append(bool(rendered.occluded_original_mask[0]))
        records["single_return_error"].append(int(rendered.source.xyzi.shape[0] != 1))
        records["distance_error_m"].append(abs(measured - expected_distance))
        records["semantic_error"].append(int(rendered.source.labels is None or int(rendered.source.labels.semantic[0]) != expected_semantic))
        records["object_id_error"].append(int(int(rendered.object_id_internal[0]) != expected_object))
        records["mask_error"].append(int(actual != expected or bool(rendered.occluded_original_mask[0]) != expected))
        expected_packed = int(rendered.source.labels.packed[0]) if rendered.source.labels is not None else -1
        records["packed_error"].append(int(expected_packed != int(rendered.packed_labels[0])))
    dtypes = {
        "relation": np.int8, "gap_m": np.float64,
        "inserted_distance_m": np.float64, "native_distance_m": np.float64,
        "expected_inserted": np.bool_, "actual_inserted": np.bool_,
        "occluded_original": np.bool_, "single_return_error": np.uint8,
        "distance_error_m": np.float64, "semantic_error": np.uint8,
        "object_id_error": np.uint8, "mask_error": np.uint8,
        "packed_error": np.uint8,
    }
    return {name: np.asarray(values, dtype=dtypes[name]) for name, values in records.items()}


def run_e32_qualification(output_path: Path | str) -> dict[str, object]:
    """Qualify inserted-front/native-background competition boundaries."""

    started = time.monotonic()
    runs = [_e32_run_once(), _e32_run_once()]
    run_seconds = [time.monotonic() - started]
    reproduced = all(np.array_equal(runs[0][name], runs[1][name]) for name in runs[0])
    first = runs[0]
    discrete = ("single_return_error", "semantic_error", "object_id_error", "mask_error", "packed_error")
    errors = {name: int(np.sum(first[name])) for name in discrete}
    passed = (
        all(value == 0 for value in errors.values())
        and float(np.max(first["distance_error_m"])) <= 1.0e-6
        and np.array_equal(first["actual_inserted"], np.asarray((False, False, True)))
        and reproduced
    )
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E32", "passed": passed, "fixtures": 3,
        **errors,
        "maximum_distance_error_m": float(np.max(first["distance_error_m"])),
        "elementwise_reproduced": reproduced,
        "run_seconds": run_seconds,
        "scientific_array_hash": scientific_hash,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e33_run_once() -> dict[str, np.ndarray]:
    half = 0.25
    vertices = np.asarray([
        (x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)
    ], dtype=np.float64)
    shape = NormalTemplateShape(
        vertices, np.empty((0, 3), dtype=np.int32), 206, 0, 10, 1,
        (0.0, 0.0, 0.0),
    )
    grid = RayGrid(
        np.asarray(((1.0, 0.0, 0.0),)), np.asarray((0.0,)),
        np.asarray((0.0,)), beam_count=1,
    )
    sensor = SensorCalibration.constant(0.5, return_probability=1.0)
    gaps = np.asarray((0.5e-6, 1.0e-6, 2.0e-6), dtype=np.float64)
    output = {name: [] for name in (
        "gap_m", "inserted_mask_error", "foreground_distance_error_m",
        "foreground_semantic_error", "foreground_instance_error",
        "object_id_error", "occlusion_error", "single_return_error",
    )}
    for case, gap in enumerate(gaps):
        native_distance = 5.0
        inserted_distance = native_distance + float(gap)
        item = ObjectSpec(
            1, "normal-control", shape, MaterialSpec(0.5, 0.1, 0.35),
            (inserted_distance + half, 0.0, 0.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        semantic = np.asarray((10,), dtype=np.uint16)
        instance = np.asarray((7,), dtype=np.uint16)
        packed = semantic.astype(np.uint32) | (instance.astype(np.uint32) << 16)
        labels = PointLabels(
            packed, semantic, instance,
            np.asarray((NORMAL_SEMANTIC_TARGET[10],), dtype=np.uint8),
        )
        frame = make_source_frame(
            case, np.asarray(((5.0, 0.0, 0.0, 0.25),), dtype=np.float32),
            np.eye(4, dtype=np.float64), labels,
            partition="train", sequence_id=206,
        )
        rendered = render_frame(
            frame, WorldSpec(3_300_000 + case, 206, (item,)), grid, sensor
        )
        measured = float(np.linalg.norm(rendered.source.xyzi[0, :3]))
        rendered_labels = rendered.source.labels
        output["gap_m"].append(float(gap))
        output["inserted_mask_error"].append(int(rendered.inserted_mask[0]))
        output["foreground_distance_error_m"].append(abs(measured - native_distance))
        output["foreground_semantic_error"].append(int(rendered_labels is None or int(rendered_labels.semantic[0]) != 10))
        output["foreground_instance_error"].append(int(rendered_labels is None or int(rendered_labels.instance[0]) != 7))
        output["object_id_error"].append(int(rendered.object_id_internal[0] != -1))
        output["occlusion_error"].append(int(rendered.occluded_original_mask[0]))
        output["single_return_error"].append(int(rendered.source.xyzi.shape[0] != 1))
    dtypes = {name: np.float64 if name.endswith("_m") else np.uint8 for name in output}
    return {name: np.asarray(values, dtype=dtypes[name]) for name, values in output.items()}


def run_e33_qualification(output_path: Path | str) -> dict[str, object]:
    """Qualify native-foreground occlusion of accepted inserted returns."""

    started = time.monotonic()
    runs = [_e33_run_once(), _e33_run_once()]
    elapsed = time.monotonic() - started
    reproduced = all(np.array_equal(runs[0][name], runs[1][name]) for name in runs[0])
    first = runs[0]
    error_names = tuple(name for name in first if name != "gap_m" and name != "foreground_distance_error_m")
    errors = {name: int(np.sum(first[name])) for name in error_names}
    passed = all(value == 0 for value in errors.values()) and float(np.max(first["foreground_distance_error_m"])) <= 1e-7 and reproduced
    scientific_hash = _scientific_array_hash(first)
    result = {
        "experiment": "E33", "passed": passed, "fixtures": 3, **errors,
        "maximum_foreground_distance_error_m": float(np.max(first["foreground_distance_error_m"])),
        "elementwise_reproduced": reproduced, "run_seconds": [elapsed],
        "scientific_array_hash": scientific_hash,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **first, metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))))
    os.replace(temporary, destination)
    return result


def _e34_run_once() -> dict[str, np.ndarray]:
    half = 0.25
    vertices = np.asarray([
        (x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)
    ], dtype=np.float64)
    shape = NormalTemplateShape(vertices, np.empty((0, 3), dtype=np.int32), 206, 0, 10, 1, (0.0, 0.0, 0.0))
    grid = RayGrid(np.asarray(((1.0, 0.0, 0.0),)), np.asarray((0.0,)), np.asarray((0.0,)), beam_count=1)
    output = {name: [] for name in ("case", "occupancy_error", "semantic_error", "object_id_error", "mask_error", "intensity_payload_error")}
    for case in range(3):
        item = ObjectSpec(1, "normal-control", shape, MaterialSpec(0.5, 0.1, 0.0), (5.0 + half, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        objects = () if case == 2 else (item,)
        sensor = SensorCalibration.constant(0.5, return_probability=1.0 if case == 0 else 0.0)
        source = make_source_frame(case, np.asarray(((0.0, 0.0, 0.0, 17.0),), dtype=np.float32), np.eye(4, dtype=np.float64), None, partition="train", sequence_id=206)
        rendered = render_frame(source, WorldSpec(3_400_000 + case, 206, objects), grid, sensor)
        expected = case == 0
        occupied = bool(np.linalg.norm(rendered.source.xyzi[0, :3]) > 0.0)
        semantic = int(rendered.packed_labels[0] & np.uint32(0xFFFF))
        output["case"].append(case)
        output["occupancy_error"].append(int(occupied != expected))
        output["semantic_error"].append(int(semantic != (10 if expected else 0)))
        output["object_id_error"].append(int(int(rendered.object_id_internal[0]) != (1 if expected else -1)))
        output["mask_error"].append(int(bool(rendered.inserted_mask[0]) != expected or bool(rendered.occluded_original_mask[0])))
        output["intensity_payload_error"].append(int((not np.isfinite(rendered.source.xyzi[0, 3])) if expected else rendered.source.xyzi[0, 3] != np.float32(17.0)))
    return {name: np.asarray(values, dtype=np.int8 if name == "case" else np.uint8) for name, values in output.items()}


def run_e34_qualification(output_path: Path | str) -> dict[str, object]:
    """Qualify new-return creation and rejection on empty native slots."""

    started = time.monotonic(); runs = [_e34_run_once(), _e34_run_once()]; elapsed = time.monotonic() - started
    reproduced = all(np.array_equal(runs[0][name], runs[1][name]) for name in runs[0])
    first = runs[0]; names = tuple(name for name in first if name != "case")
    errors = {name: int(np.sum(first[name])) for name in names}
    passed = all(value == 0 for value in errors.values()) and reproduced
    scientific_hash = _scientific_array_hash(first)
    result = {"experiment": "E34", "passed": passed, "fixtures": 3, **errors, "elementwise_reproduced": reproduced, "run_seconds": [elapsed], "scientific_array_hash": scientific_hash}
    destination = Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **first, metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))))
    os.replace(temporary, destination); return result


_E35_SENSOR: SensorCalibration | None = None


def _e35_worker(identity: int) -> dict[str, np.ndarray]:
    if _E35_SENSOR is None:
        raise RuntimeError("E35 calibration is not initialized")
    sensor = _E35_SENSOR
    beam, range_bin, incidence_bin = np.indices(sensor.return_probability.shape, dtype=np.int64)
    beam = beam.ravel(); range_bin = range_bin.ravel(); incidence_bin = incidence_bin.ravel()
    ranges = 0.5 * (sensor.range_edges_m[range_bin] + sensor.range_edges_m[range_bin + 1])
    incidence = 0.5 * (sensor.incidence_edges_rad[incidence_bin] + sensor.incidence_edges_rad[incidence_bin + 1])
    slots = np.arange(beam.size, dtype=np.int32); object_ids = np.full(beam.size, identity + 1, dtype=np.int32)
    world = WorldSpec(3_500_000 + identity, 206, ())
    uniform = _slot_uniform(world, 2_000 + identity, slots, object_ids, channel=1)
    reference_uniform = _e29_reference_uniform(world.seed, 206, 2_000 + identity, slots, object_ids, 1)
    material = MaterialSpec.sample(3_500_000 + identity)
    intensity = sensor.sample_intensity(beam, ranges, incidence, uniform, material)
    quantile_raw = material.intensity_quantile + material.roughness * (reference_uniform - 0.5)
    quantile = np.clip(quantile_raw, 0.0, 1.0)
    reference = np.asarray([
        np.interp(quantile[i], sensor.quantile_levels, sensor.intensity_quantiles[int(beam[i]), int(range_bin[i]), int(incidence_bin[i])])
        for i in range(beam.size)
    ], dtype=np.float32)
    np.clip(reference, sensor.intensity_min, sensor.intensity_max, out=reference)
    return {
        "identity": np.full(beam.size, identity, dtype=np.int16), "beam": beam.astype(np.int16),
        "range_bin": range_bin.astype(np.int8), "incidence_bin": incidence_bin.astype(np.int8),
        "uniform": uniform, "reference_uniform": reference_uniform,
        "quantile_raw": quantile_raw, "quantile": quantile,
        "intensity": intensity, "reference_intensity": reference,
        "quantile_low_clipped": quantile_raw < 0.0, "quantile_high_clipped": quantile_raw > 1.0,
    }


def run_e35_qualification(calibration_path: Path | str, output_path: Path | str, *, processes: int = 24) -> dict[str, object]:
    """Qualify conditional intensity generation against an independent reference."""
    if processes != 24:
        raise RenderError("formal E35 requires exactly 24 worker processes")
    _, sensor = load_sensor_calibration(calibration_path)
    global _E35_SENSOR; _E35_SENSOR = sensor
    context = mp.get_context("fork"); runs = []; run_seconds = []
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers: records = workers.map(_e35_worker, range(24))
        runs.append({name: np.concatenate([record[name] for record in records]) for name in records[0]})
        run_seconds.append(time.monotonic() - started)
    first = runs[0]; reproduced = all(np.array_equal(first[name], runs[1][name]) for name in first)
    maximum_error = float(np.max(np.abs(first["intensity"].astype(np.float64) - first["reference_intensity"].astype(np.float64))))
    maximum_uniform_error = float(np.max(np.abs(first["uniform"] - first["reference_uniform"])))
    undefined = int(np.count_nonzero(~np.isfinite(first["intensity"])))
    support_errors = int(np.count_nonzero((first["intensity"] < sensor.intensity_min) | (first["intensity"] > sensor.intensity_max)))
    low = int(np.count_nonzero(first["quantile_low_clipped"])); high = int(np.count_nonzero(first["quantile_high_clipped"])); total = int(first["intensity"].size)
    passed = maximum_error <= 1e-6 and maximum_uniform_error == 0.0 and undefined == 0 and support_errors == 0 and reproduced
    scientific_hash = _scientific_array_hash(first)
    result = {"experiment": "E35", "passed": passed, "calibration_cells": int(sensor.return_probability.size), "identities_per_cell": 24, "samples": total, "maximum_intensity_error": maximum_error, "maximum_uniform_error": maximum_uniform_error, "undefined_cells": undefined, "support_errors": support_errors, "quantile_low_clipped": low, "quantile_high_clipped": high, "quantile_clipping_fraction": (low + high) / total, "elementwise_reproduced": reproduced, "run_seconds": run_seconds, "processes": processes, "scientific_array_hash": scientific_hash}
    destination = Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **first, metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":")))); os.replace(temporary, destination); return result


def run_e36_qualification(output_path: Path | str) -> dict[str, object]:
    """Audit label-independent sensor stages and paired-fixture constructibility."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    audited = {"_accepted_object_hits", "return_chance", "sample_intensity"}
    label_branches = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in audited:
            label_branches += sum(
                isinstance(child, ast.Attribute) and child.attr == "label"
                or isinstance(child, ast.Constant) and child.value in OBJECT_LABELS
                for child in ast.walk(node)
            )
    half = 0.25
    vertices = np.asarray([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)], dtype=np.float64)
    normal_shape = NormalTemplateShape(vertices, np.empty((0, 3), dtype=np.int32), 206, 0, 10, 1, (0.0, 0.0, 0.0))
    proxy_shape, proxy_report = ShapeSpec.sample_with_report(3_600_000)
    material = MaterialSpec(0.5, 0.1, 0.0); rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    failures = []
    try:
        ObjectSpec(1, "anomaly-proxy", normal_shape, material, (5.0, 0.0, 0.0), rotation)
    except RenderError as error:
        failures.append(str(error))
    try:
        ObjectSpec(1, "normal-control", proxy_shape, material, (5.0, 0.0, 0.0), rotation, proxy_report)
    except RenderError as error:
        failures.append(str(error))
    paired_trace_completed = len(failures) == 0
    passed = label_branches == 0 and paired_trace_completed
    result = {
        "experiment": "E36", "passed": passed,
        "audited_functions": sorted(audited),
        "sensor_intermediate_label_branches": int(label_branches),
        "paired_fixture_constructible": paired_trace_completed,
        "paired_fixture_construction_errors": failures,
        "paired_trace_completed": paired_trace_completed,
        "failure_classification": None if passed else "protocol design conflict",
    }
    destination = Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":")))); os.replace(temporary, destination); return result


_E36_V2_SENSOR: SensorCalibration | None = None


def _e36_v2_worker(task: tuple[int, str]) -> dict[str, np.ndarray]:
    identity, virtual_label = task
    if _E36_V2_SENSOR is None or virtual_label not in OBJECT_LABELS:
        raise RuntimeError("E36-v2 fixture is not initialized")
    sensor = _E36_V2_SENSOR
    beam, range_bin, incidence_bin = np.indices(sensor.return_probability.shape, dtype=np.int64)
    beam = beam.ravel(); range_bin = range_bin.ravel(); incidence_bin = incidence_bin.ravel()
    distance = 0.5 * (sensor.range_edges_m[range_bin] + sensor.range_edges_m[range_bin + 1])
    incidence = 0.5 * (sensor.incidence_edges_rad[incidence_bin] + sensor.incidence_edges_rad[incidence_bin + 1])
    slots = np.arange(beam.size, dtype=np.int32); object_ids = np.full(beam.size, identity + 1, dtype=np.int32)
    world = WorldSpec(3_600_000 + identity, 206, ())
    frame_id = 3_000 + identity
    material = MaterialSpec.sample(3_600_000 + identity)
    probability = sensor.return_chance(beam, distance, incidence, material.return_bias)
    return_uniform = _slot_uniform(world, frame_id, slots, object_ids, channel=0)
    accepted = return_uniform < probability
    intensity_uniform = _slot_uniform(world, frame_id, slots, object_ids, channel=1)
    intensity = np.full(beam.size, np.nan, dtype=np.float32)
    intensity[accepted] = sensor.sample_intensity(
        beam[accepted], distance[accepted], incidence[accepted],
        intensity_uniform[accepted], material,
    )
    mode = slots % 3
    native_range = np.where(mode == 0, distance + 1.0, np.where(mode == 1, distance - 1.0, np.inf))
    competition_input = np.where(accepted, distance, np.inf)
    inserted = accepted & (distance < native_range - 1.0e-6)
    final_distance = np.where(inserted, distance, native_range)
    occupancy = np.isfinite(final_distance)
    semantic = np.zeros(beam.size, dtype=np.uint16)
    semantic[inserted] = np.uint16(10 if virtual_label == "normal-control" else 2)
    normal_mask = inserted & (virtual_label == "normal-control")
    anomaly_mask = inserted & (virtual_label == "anomaly-proxy")
    return {
        "identity": np.full(beam.size, identity, dtype=np.int16),
        "beam": beam.astype(np.int16), "range_bin": range_bin.astype(np.int8),
        "incidence_bin": incidence_bin.astype(np.int8), "distance_m": distance,
        "incidence_rad": incidence, "material_bias": np.full(beam.size, material.return_bias),
        "native_range_m": native_range, "return_probability": probability,
        "return_uniform": return_uniform, "accepted": accepted,
        "intensity_uniform": intensity_uniform, "sampled_intensity": intensity,
        "competition_input_m": competition_input, "final_distance_m": final_distance,
        "occupancy": occupancy, "inserted": inserted,
        "semantic": semantic, "normal_control_mask": normal_mask,
        "anomaly_proxy_mask": anomaly_mask,
    }


def _e36_static_label_audit() -> tuple[int, int, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    sensor_names = {"_accepted_object_hits", "return_chance", "sample_intensity"}
    sensor_label_reads = 0; render_label_reads = 0; render_pre_competition_reads = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reads = [child for child in ast.walk(node) if isinstance(child, ast.Attribute) and child.attr == "label"]
        if node.name in sensor_names:
            sensor_label_reads += len(reads)
        if node.name == "render_frame":
            render_label_reads = len(reads)
            calls = [child.lineno for child in ast.walk(node) if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "_accepted_object_hits"]
            boundary = min(calls) if calls else 10**9
            render_pre_competition_reads = sum(read.lineno < boundary for read in reads)
    return sensor_label_reads, render_pre_competition_reads, render_label_reads


def run_e36_v2_qualification(calibration_path: Path | str, output_path: Path | str, *, processes: int = 24) -> dict[str, object]:
    """Qualify label independence below ObjectSpec at the sensor interface."""
    if processes != 24:
        raise RenderError("formal E36-v2 requires exactly 24 worker processes")
    _, sensor = load_sensor_calibration(calibration_path)
    global _E36_V2_SENSOR; _E36_V2_SENSOR = sensor
    context = mp.get_context("fork"); run_seconds = []; paired_runs = []
    labels = ("normal-control", "anomaly-proxy")
    for _ in range(2):
        started = time.monotonic()
        with context.Pool(processes=processes) as workers:
            records = workers.map(_e36_v2_worker, [(i, label) for label in labels for i in range(24)])
        conditions = []
        for label_index in range(2):
            selected = records[label_index * 24 : (label_index + 1) * 24]
            conditions.append({name: np.concatenate([record[name] for record in selected]) for name in selected[0]})
        paired_runs.append(conditions); run_seconds.append(time.monotonic() - started)
    allowed = {"semantic", "normal_control_mask", "anomaly_proxy_mask"}
    intermediate = tuple(name for name in paired_runs[0][0] if name not in allowed)
    paired_errors = {name: int(np.count_nonzero(~np.isclose(paired_runs[0][0][name], paired_runs[0][1][name], equal_nan=True))) if np.issubdtype(paired_runs[0][0][name].dtype, np.floating) else int(np.count_nonzero(paired_runs[0][0][name] != paired_runs[0][1][name])) for name in intermediate}
    reproduced = all(np.array_equal(paired_runs[0][condition][name], paired_runs[1][condition][name], equal_nan=True) for condition in range(2) for name in paired_runs[0][condition])
    normal, anomaly = paired_runs[0]
    bookkeeping_errors = int(np.count_nonzero(normal["normal_control_mask"] != normal["inserted"])) + int(np.count_nonzero(normal["anomaly_proxy_mask"])) + int(np.count_nonzero(anomaly["anomaly_proxy_mask"] != anomaly["inserted"])) + int(np.count_nonzero(anomaly["normal_control_mask"])) + int(np.count_nonzero(normal["semantic"][normal["inserted"]] != 10)) + int(np.count_nonzero(anomaly["semantic"][anomaly["inserted"]] != 2))
    sensor_reads, pre_competition_reads, render_reads = _e36_static_label_audit()
    passed = all(value == 0 for value in paired_errors.values()) and bookkeeping_errors == 0 and sensor_reads == 0 and pre_competition_reads == 0 and reproduced
    scientific = {f"normal_{name}": normal[name] for name in normal}
    scientific.update({f"anomaly_{name}": anomaly[name] for name in allowed})
    scientific_hash = _scientific_array_hash(scientific)
    result = {"experiment": "E36-v2", "passed": passed, "sensor_inputs": int(normal["accepted"].size), "intermediate_array_errors": int(sum(paired_errors.values())), "intermediate_error_by_field": paired_errors, "bookkeeping_errors": bookkeeping_errors, "sensor_function_label_reads": sensor_reads, "render_pre_competition_label_reads": pre_competition_reads, "render_final_bookkeeping_label_reads": render_reads, "elementwise_reproduced": reproduced, "run_seconds": run_seconds, "processes": processes, "scientific_array_hash": scientific_hash}
    destination = Path(output_path).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **scientific, metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":")))); os.replace(temporary, destination); return result


_E37_WORLDS: tuple[WorldSpec, ...] = ()
_E37_FRAMES: dict[int, SourceFrame] = {}
_E37_RAY_GRID: RayGrid | None = None
_E37_SENSOR: SensorCalibration | None = None
_E37_RENDERER_IDENTITY = ""
_E37_SOURCE_IDENTITY = ""
_E37_STU_IDENTITY = ""
_E37_FRAME_CACHE: type | None = None
_E37_FRAME_CACHE_KEY: type | None = None
_E37_FRAME_IDS = tuple(range(98, 104))
_E37_FORWARD_REQUESTS = tuple(range(98, 103)) + tuple(range(99, 104))


def _e37_arrays(frame: RenderedFrame) -> dict[str, np.ndarray]:
    """Expose every E37-qualified slot-aligned renderer output."""
    return {
        "xyzi": np.asarray(frame.xyzi),
        "occupancy": ~np.asarray(frame.source.zero_slot_mask),
        "packed_labels": np.asarray(frame.packed_labels),
        "normal_control_mask": np.asarray(frame.normal_control_mask),
        "anomaly_proxy_mask": np.asarray(frame.anomaly_proxy_mask),
        "inserted_mask": np.asarray(frame.inserted_mask),
        "occluded_original_mask": np.asarray(frame.occluded_original_mask),
        "unchanged_normal_mask": np.asarray(frame.unchanged_normal_mask),
        "object_id_internal": np.asarray(frame.object_id_internal),
    }


def _e37_array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _e37_frame_digests(frame: RenderedFrame) -> dict[str, str]:
    return {name: _e37_array_digest(value) for name, value in _e37_arrays(frame).items()}


def _e37_cache_key(world: WorldSpec, frame_id: int) -> object:
    if _E37_FRAME_CACHE_KEY is None:
        raise RuntimeError("E37 production cache key is not initialized")
    frame_identity = hashlib.sha256(
        f"{_E37_SOURCE_IDENTITY}:train:206:{frame_id}".encode("ascii")
    ).hexdigest()
    return _E37_FRAME_CACHE_KEY(
        world.identity,
        frame_identity,
        _E37_RENDERER_IDENTITY,
        _E37_STU_IDENTITY,
    )


def _e37_request_order(world: WorldSpec, mode: str) -> tuple[int, ...]:
    if mode in {"serial_cached_forward", "parallel_uncached_forward"}:
        return _E37_FORWARD_REQUESTS
    if mode == "parallel_cached_reverse":
        return tuple(reversed(_E37_FORWARD_REQUESTS))
    if mode == "parallel_cached_random":
        rng = np.random.default_rng(np.random.SeedSequence([world.seed, 3701]))
        return tuple(np.asarray(_E37_FORWARD_REQUESTS)[rng.permutation(10)].tolist())
    raise RuntimeError(f"unknown E37 execution mode {mode!r}")


def _e37_worker(task: tuple[int, str]) -> dict[str, object]:
    world_index, mode = task
    if (
        len(_E37_WORLDS) != 128
        or set(_E37_FRAMES) != set(_E37_FRAME_IDS)
        or _E37_RAY_GRID is None
        or _E37_SENSOR is None
        or _E37_FRAME_CACHE is None
    ):
        raise RuntimeError("E37 fixtures are not initialized")
    world = _E37_WORLDS[world_index]
    requests = _e37_request_order(world, mode)
    cached = "uncached" not in mode
    cache = _E37_FRAME_CACHE(7) if cached else None
    first: dict[int, RenderedFrame] = {}
    digests: dict[int, dict[str, str]] = {}
    duplicate_bit_errors = 0
    render_calls = 0

    def produce(frame_id: int) -> RenderedFrame:
        nonlocal render_calls
        render_calls += 1
        return render_frame(
            _E37_FRAMES[frame_id], world, _E37_RAY_GRID, _E37_SENSOR
        )

    for frame_id in requests:
        if cache is None:
            rendered = produce(frame_id)
        else:
            rendered = cache.rendered_frame(
                _e37_cache_key(world, frame_id),
                lambda frame_id=frame_id: produce(frame_id),
            )
        if frame_id not in first:
            first[frame_id] = rendered
            digests[frame_id] = _e37_frame_digests(rendered)
            continue
        for name, value in _e37_arrays(rendered).items():
            reference = _e37_arrays(first[frame_id])[name]
            left = np.ascontiguousarray(value).view(np.uint8)
            right = np.ascontiguousarray(reference).view(np.uint8)
            duplicate_bit_errors += int(np.unpackbits(np.bitwise_xor(left, right)).sum())
    if set(first) != set(_E37_FRAME_IDS):
        raise RuntimeError("E37 request stream did not cover the six frozen frames")
    return {
        "world_index": world_index,
        "world_hash": world.identity,
        "world_type": world.world_type,
        "mode": mode,
        "render_calls": render_calls,
        "duplicate_bit_errors": duplicate_bit_errors,
        "digests": digests,
    }


def _e37_cross_world_worker(pair_index: int) -> dict[str, int]:
    if _E37_FRAME_CACHE is None:
        raise RuntimeError("E37 production frame cache is not initialized")
    left_index = pair_index
    right_index = pair_index + 64
    left_world = _E37_WORLDS[left_index]
    right_world = _E37_WORLDS[right_index]
    frame_id = _E37_FRAME_IDS[pair_index % len(_E37_FRAME_IDS)]
    cache = _E37_FRAME_CACHE(7)
    calls = {left_index: 0, right_index: 0}

    def request(index: int, world: WorldSpec) -> RenderedFrame:
        def produce() -> RenderedFrame:
            calls[index] += 1
            assert _E37_RAY_GRID is not None and _E37_SENSOR is not None
            return render_frame(
                _E37_FRAMES[frame_id], world, _E37_RAY_GRID, _E37_SENSOR
            )
        return cache.rendered_frame(_e37_cache_key(world, frame_id), produce)

    left_first = request(left_index, left_world)
    right_first = request(right_index, right_world)
    left_second = request(left_index, left_world)
    right_second = request(right_index, right_world)
    errors = int(calls[left_index] != 1 or calls[right_index] != 1)
    errors += int(left_first is not left_second or right_first is not right_second)
    errors += int(left_first is right_first)
    return {"cache_miss_errors": errors, "factory_calls": sum(calls.values())}


def _e37_static_window_audit() -> tuple[int, int]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    render_parameters = -1
    uniform_window_reads = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "render_frame":
            names = [item.arg for item in node.args.args]
            render_parameters = sum("window" in name for name in names)
        elif node.name == "_slot_uniform":
            uniform_window_reads += sum(
                isinstance(child, ast.Name) and "window" in child.id
                or isinstance(child, ast.Attribute) and "window" in child.attr
                for child in ast.walk(node)
            )
    return render_parameters, uniform_window_reads


def run_e37_qualification(
    e26_artifact_path: Path | str,
    data_root: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Qualify world/frame identity across overlapping window request paths."""
    if processes != 24:
        raise RenderError("formal E37 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
        from .train import FrameCache, FrameCacheKey
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
        from train import FrameCache, FrameCacheKey  # type: ignore[no-redef]

    with np.load(
        Path(e26_artifact_path).expanduser().resolve(strict=True), allow_pickle=False
    ) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E26" or metadata.get("passed") is not True:
            raise RenderError("E37 requires the passed formal E26 artifact")
        records = [
            (str(world_type), int(seed), WorldSpec.from_dict(json.loads(payload.decode())))
            for world_type, seed, payload in zip(
                source["world_type"], source["world_seed"], source["world_json"],
                strict=True,
            )
        ]
    selected = tuple(
        world
        for world_type in WORLD_TYPES
        for _, _, world in sorted(
            (item for item in records if item[0] == world_type),
            key=lambda item: item[1],
        )[:32]
    )
    if len(selected) != 128 or {
        world_type: sum(world.world_type == world_type for world in selected)
        for world_type in WORLD_TYPES
    } != {world_type: 32 for world_type in WORLD_TYPES}:
        raise RenderError("E37 fixed E26 world coverage changed")

    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    frames = {frame_id: sequence.source_frame(frame_id) for frame_id in _E37_FRAME_IDS}
    ray_grid, sensor = load_sensor_calibration(calibration_path)
    render_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    renderer_payload = {
        "generator_schema": PROCEDURAL_GENERATOR_SCHEMA,
        "render_source_sha256": render_source_sha256,
    }
    renderer_identity = hashlib.sha256(
        json.dumps(renderer_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_digest = hashlib.sha256()
    for frame_id in _E37_FRAME_IDS:
        frame = frames[frame_id]
        source_digest.update(np.asarray(frame.xyzi).tobytes())
        source_digest.update(np.asarray(frame.lidar_pose).tobytes())
        assert frame.labels is not None
        source_digest.update(np.asarray(frame.labels.packed).tobytes())
    source_identity = source_digest.hexdigest()
    stu_identity = hashlib.sha256(
        Path(calibration_path).expanduser().resolve(strict=True).read_bytes()
    ).hexdigest()
    global _E37_WORLDS, _E37_FRAMES, _E37_RAY_GRID, _E37_SENSOR
    global _E37_RENDERER_IDENTITY, _E37_SOURCE_IDENTITY, _E37_STU_IDENTITY
    global _E37_FRAME_CACHE, _E37_FRAME_CACHE_KEY
    _E37_WORLDS = selected
    _E37_FRAMES = frames
    _E37_RAY_GRID = ray_grid
    _E37_SENSOR = sensor
    _E37_RENDERER_IDENTITY = renderer_identity
    _E37_SOURCE_IDENTITY = source_identity
    _E37_STU_IDENTITY = stu_identity
    _E37_FRAME_CACHE = FrameCache
    _E37_FRAME_CACHE_KEY = FrameCacheKey

    context = mp.get_context("fork")
    modes = (
        ("serial_cached_forward", 1),
        ("parallel_uncached_forward", processes),
        ("parallel_cached_reverse", processes),
        ("parallel_cached_random", processes),
    )
    executions: dict[str, list[dict[str, object]]] = {}
    run_seconds: dict[str, float] = {}
    for mode, worker_count in modes:
        started = time.monotonic()
        tasks = [(index, mode) for index in range(128)]
        with context.Pool(processes=worker_count) as workers:
            values = workers.map(_e37_worker, tasks)
        executions[mode] = sorted(values, key=lambda item: int(item["world_index"]))
        run_seconds[mode] = time.monotonic() - started
    with context.Pool(processes=processes) as workers:
        cross_world = workers.map(_e37_cross_world_worker, range(64))

    reference = executions["serial_cached_forward"]
    field_names = tuple(next(iter(reference))["digests"][_E37_FRAME_IDS[0]])
    digest_errors = {name: 0 for name in field_names}
    identity_errors = 0
    for mode, values in executions.items():
        for expected, actual in zip(reference, values, strict=True):
            identity_errors += int(
                expected["world_hash"] != actual["world_hash"]
                or expected["world_type"] != actual["world_type"]
            )
            for frame_id in _E37_FRAME_IDS:
                for name in field_names:
                    digest_errors[name] += int(
                        expected["digests"][frame_id][name]
                        != actual["digests"][frame_id][name]
                    )
    duplicate_bit_errors = sum(
        int(value["duplicate_bit_errors"])
        for values in executions.values() for value in values
    )
    expected_calls = {
        "serial_cached_forward": 128 * 6,
        "parallel_uncached_forward": 128 * 10,
        "parallel_cached_reverse": 128 * 6,
        "parallel_cached_random": 128 * 6,
    }
    render_calls = {
        mode: sum(int(value["render_calls"]) for value in values)
        for mode, values in executions.items()
    }
    render_call_errors = sum(
        render_calls[mode] != expected for mode, expected in expected_calls.items()
    )
    cross_world_cache_errors = sum(
        int(value["cache_miss_errors"]) for value in cross_world
    )
    cross_world_factory_calls = sum(int(value["factory_calls"]) for value in cross_world)
    render_window_parameters, uniform_window_reads = _e37_static_window_audit()
    passed = (
        identity_errors == 0
        and all(value == 0 for value in digest_errors.values())
        and duplicate_bit_errors == 0
        and render_call_errors == 0
        and cross_world_cache_errors == 0
        and cross_world_factory_calls == 128
        and render_window_parameters == 0
        and uniform_window_reads == 0
    )
    world_hashes = np.asarray([world.identity for world in selected], dtype="S64")
    output_digests = np.asarray(
        [
            reference[world_index]["digests"][frame_id][name]
            for world_index in range(128)
            for frame_id in _E37_FRAME_IDS
            for name in field_names
        ],
        dtype="S64",
    ).reshape(128, 6, len(field_names))
    scientific = {
        "world_hash": world_hashes,
        "frame_id": np.asarray(_E37_FRAME_IDS, dtype=np.int16),
        "field_name": np.asarray(field_names, dtype="U32"),
        "output_sha256": output_digests,
    }
    result = {
        "experiment": "E37", "passed": passed, "worlds": 128,
        "world_type_counts": {world_type: 32 for world_type in WORLD_TYPES},
        "window_centers": [100, 101], "window_frame_ids": [[98, 99, 100, 101, 102], [99, 100, 101, 102, 103]],
        "unique_frames_per_world": 6, "frame_requests_per_world": 10,
        "qualified_fields": list(field_names), "identity_errors": identity_errors,
        "field_digest_errors": digest_errors,
        "duplicate_request_bit_errors": duplicate_bit_errors,
        "render_calls": render_calls, "expected_render_calls": expected_calls,
        "render_call_errors": int(render_call_errors),
        "cross_world_cache_errors": cross_world_cache_errors,
        "cross_world_factory_calls": cross_world_factory_calls,
        "render_frame_window_parameters": render_window_parameters,
        "slot_uniform_window_reads": uniform_window_reads,
        "execution_processes": {mode: count for mode, count in modes},
        "run_seconds": run_seconds, "render_source_sha256": render_source_sha256,
        "renderer_generator_identity": renderer_identity,
        "source_identity": source_identity, "stu_identity": stu_identity,
        "scientific_array_hash": _scientific_array_hash(scientific),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_GATE1_SEQUENCE: object | None = None
_GATE1_BANK_SEED_BASE = 3_800_000
_GATE1_BANK_CAPACITY_LIMIT = 256


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Apply the frozen unsigned SplitMix64 permutation elementwise."""
    value = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _gate1_support_center(center_frame: int) -> dict[str, np.ndarray]:
    sequence = _GATE1_SEQUENCE
    if sequence is None:
        raise RuntimeError("Gate 1 sequence is not initialized")
    frames = tuple(sequence.source_frame(frame_id) for frame_id in range(center_frame - 2, center_frame + 3))
    center = frames[2]
    if center.labels is None:
        raise RuntimeError("Gate 1 support qualification requires labels")
    context_points: list[np.ndarray] = []
    context_semantics: list[np.ndarray] = []
    for frame in frames:
        assert frame.labels is not None
        real = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        semantic = np.asarray(frame.labels.semantic)
        ground = real & np.isin(semantic, SUPPORT_POOL_SEMANTICS)
        rotation, translation = _pose(frame)
        context_points.append(
            np.asarray(frame.xyzi[ground, :3], dtype=np.float64) @ rotation.T + translation
        )
        context_semantics.append(semantic[ground])
    points = np.concatenate(context_points)
    semantics = np.concatenate(context_semantics)
    center_real = ~np.asarray(center.zero_slot_mask, dtype=np.bool_)
    selected = center_real & np.isin(center.labels.semantic, SUPPORT_POOL_SEMANTICS)
    slots = np.flatnonzero(selected).astype(np.int32)
    rotation, translation = _pose(center)
    anchors = np.asarray(center.xyzi[slots, :3], dtype=np.float64) @ rotation.T + translation
    ranges = np.linalg.norm(np.asarray(center.xyzi[slots, :3], dtype=np.float64), axis=1)
    anchor_semantics = np.asarray(center.labels.semantic[slots], dtype=np.uint16)
    cell_x = np.floor(anchors[:, 0] / 0.5).astype(np.int64)
    cell_y = np.floor(anchors[:, 1] / 0.5).astype(np.int64)
    packed = (np.uint64(center_frame) << np.uint64(32)) | slots.astype(np.uint64)
    hashes = _splitmix64(packed ^ (anchor_semantics.astype(np.uint64) << np.uint64(48)))
    order = np.lexsort((slots, hashes, cell_y, cell_x, anchor_semantics))
    ordered_group = np.column_stack((anchor_semantics[order], cell_x[order], cell_y[order]))
    keep = np.ones(order.size, dtype=np.bool_)
    keep[1:] = np.any(ordered_group[1:] != ordered_group[:-1], axis=1)
    retained = order[keep]
    trees = {
        semantic: cKDTree(points[semantics == semantic, :2], compact_nodes=True)
        for semantic in SUPPORT_POOL_SEMANTICS
    }
    semantic_points = {
        semantic: points[semantics == semantic] for semantic in SUPPORT_POOL_SEMANTICS
    }
    output: dict[str, list[object]] = {
        "semantic": [], "frame": [], "slot": [], "range_m": [],
        "selection_hash": [], "anchor_world": [], "normal": [], "offset": [],
    }
    for index in retained:
        semantic = int(anchor_semantics[index])
        radius = float(np.clip(ranges[index] / 20.0, 1.0, 3.0))
        neighbours = trees[semantic].query_ball_point(anchors[index, :2], 1.25 * radius)
        local = semantic_points[semantic][np.asarray(neighbours, dtype=np.int64)]
        qualification = qualify_support_plane(local, anchors[index], radius_m=radius)
        if not qualification.qualified:
            continue
        middle = qualification.estimates[1]
        output["semantic"].append(semantic)
        output["frame"].append(center_frame)
        output["slot"].append(int(slots[index]))
        output["range_m"].append(float(ranges[index]))
        output["selection_hash"].append(int(hashes[index]))
        output["anchor_world"].append(anchors[index])
        output["normal"].append(middle.normal)
        output["offset"].append(middle.offset)
    count = len(output["frame"])
    return {
        "semantic": np.asarray(output["semantic"], dtype=np.uint16),
        "frame": np.asarray(output["frame"], dtype=np.int32),
        "slot": np.asarray(output["slot"], dtype=np.int32),
        "range_m": np.asarray(output["range_m"], dtype=np.float64),
        "selection_hash": np.asarray(output["selection_hash"], dtype=np.uint64),
        "anchor_world": np.asarray(output["anchor_world"], dtype=np.float64).reshape(count, 3),
        "normal": np.asarray(output["normal"], dtype=np.float64).reshape(count, 3),
        "offset": np.asarray(output["offset"], dtype=np.float64),
    }


def build_gate1_support_pool(
    sequence: object, output_path: Path | str, *, processes: int = 24
) -> tuple[QualifiedSupportPool, dict[str, object]]:
    """Apply the unchanged E21-v4 estimator to train/201 frames 4--681."""
    if processes != 24:
        raise RenderError("formal Gate 1 support-pool construction requires 24 processes")
    global _GATE1_SEQUENCE
    _GATE1_SEQUENCE = sequence
    centers = tuple(range(6, 680))
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        records = workers.map(_gate1_support_center, centers, chunksize=1)
    arrays = {
        name: np.concatenate([record[name] for record in records])
        for name in records[0]
    }
    count = int(arrays["frame"].size)
    pool = QualifiedSupportPool(
        np.arange(count, dtype=np.int64), arrays["semantic"], arrays["frame"],
        arrays["slot"], arrays["range_m"], arrays["selection_hash"],
        arrays["anchor_world"], arrays["normal"], arrays["offset"],
    )
    metadata = {
        "experiment": "Gate1-201-support-pool", "passed": count > 0,
        "source_sequence": "train/201", "source_frames": [4, 681],
        "center_frames": [6, 679], "pool_size": count,
        "semantic_counts": {
            str(semantic): int(np.count_nonzero(arrays["semantic"] == semantic))
            for semantic in SUPPORT_POOL_SEMANTICS
        },
        "covered_center_frames": int(np.unique(arrays["frame"]).size),
        "processes": processes, "elapsed_seconds": time.monotonic() - started,
        "scientific_array_hash": _scientific_array_hash(arrays),
        "estimator": "E21-v4 unchanged three-scale trimmed-SVD",
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return pool, metadata


def load_gate1_support_pool(path: Path | str) -> tuple[QualifiedSupportPool, dict[str, object]]:
    source = Path(path).expanduser().resolve(strict=True)
    if _sha256_path(source) != FROZEN_GATE1_SUPPORT_POOL_SHA256:
        raise PlacementError("Gate 1 support-pool artifact identity changed")
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        if (
            metadata.get("experiment") != "Gate1-201-support-pool"
            or metadata.get("passed") is not True
            or metadata.get("source_frames") != [4, 681]
            or metadata.get("center_frames") != [6, 679]
            or metadata.get("estimator")
            != "E21-v4 unchanged three-scale trimmed-SVD"
        ):
            raise PlacementError("Gate 1 support-pool artifact is not qualified")
        arrays = {name: np.asarray(payload[name]) for name in (
            "semantic", "frame", "slot", "range_m", "selection_hash",
            "anchor_world", "normal", "offset",
        )}
    count = arrays["frame"].size
    if (
        int(metadata.get("pool_size", -1)) != count
        or metadata.get("scientific_array_hash")
        != _scientific_array_hash(arrays)
    ):
        raise PlacementError("Gate 1 support-pool contents changed")
    return QualifiedSupportPool(
        np.arange(count, dtype=np.int64), arrays["semantic"], arrays["frame"],
        arrays["slot"], arrays["range_m"], arrays["selection_hash"],
        arrays["anchor_world"], arrays["normal"], arrays["offset"],
    ), metadata


def _gate1_frame_keys_and_obstacles(
    sequence: object, *, build_obstacles: bool,
) -> tuple[dict[int, set[tuple[int, int]]], ObservedObstacleIndex | None]:
    """Read train/201 once for persistent-instance keys and optional obstacles."""

    active = frozenset((10, 18, 20, 30))
    frame_keys: dict[int, set[tuple[int, int]]] = {}
    point_chunks: list[np.ndarray] = []
    identity_chunks: list[np.ndarray] = []
    for frame_id in range(4, 682):
        frame = sequence.source_frame(frame_id)
        if frame.partition != "train" or frame.sequence_id != 201 or frame.labels is None:
            raise PlacementError("Gate 1 frames must be labelled train/201 data")
        semantic = np.asarray(frame.labels.semantic)
        instance = np.asarray(frame.labels.instance)
        real = ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        range_m = np.linalg.norm(np.asarray(frame.xyzi[:, :3], dtype=np.float64), axis=1)
        eligible = real & (range_m >= 2.5) & (range_m <= 50.0)
        packed = semantic.astype(np.uint32) | (
            instance.astype(np.uint32) << np.uint32(16)
        )
        keys: set[tuple[int, int]] = set()
        for value, count in zip(
            *np.unique(packed[eligible], return_counts=True), strict=True
        ):
            raw = int(value & np.uint32(0xFFFF))
            identifier = int(value >> np.uint32(16))
            if raw in active and identifier > 0 and int(count) >= 16:
                keys.add((raw, identifier))
        frame_keys[frame_id] = keys
        if build_obstacles:
            slots = np.flatnonzero(
                real & (semantic != 0) & ~np.isin(semantic, GROUND_SEMANTIC_IDS)
            )
            rotation, translation = _pose(frame)
            sensor = np.asarray(frame.xyzi[slots, :3], dtype=np.float64)
            point_chunks.append(sensor @ rotation.T + translation)
            identity_chunks.append(
                (np.uint64(frame_id) << np.uint64(32))
                | slots.astype(np.uint64)
            )
    obstacles = None
    if build_obstacles:
        obstacles = ObservedObstacleIndex(
            np.concatenate(point_chunks, axis=0),
            np.concatenate(identity_chunks, axis=0),
        )
    return frame_keys, obstacles


def _gate1_real_candidates(
    sequence: object, pool: QualifiedSupportPool,
    *, frame_keys: Mapping[int, set[tuple[int, int]]] | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Select persistent real instances with an observable legal support semantic."""
    resolved_keys = (
        _gate1_frame_keys_and_obstacles(sequence, build_obstacles=False)[0]
        if frame_keys is None else frame_keys
    )
    if set(resolved_keys) != set(range(4, 682)):
        raise PlacementError("Gate 1 real-instance frame keys are incomplete")
    pool_frames = np.asarray(pool.frames)
    frame_order = (
        None
        if bool(np.all(pool_frames[1:] >= pool_frames[:-1]))
        else np.argsort(pool_frames, kind="stable")
    )
    ordered_frames = pool_frames if frame_order is None else pool_frames[frame_order]
    candidates: list[tuple[bytes, tuple[int, int, int, int]]] = []
    for center in range(6, 680):
        persistent = set.intersection(*(
            resolved_keys[frame] for frame in range(center - 2, center + 3)
        ))
        if not persistent:
            continue
        source = sequence.source_frame(center)
        assert source.labels is not None
        rotation, translation = _pose(source)
        lower = int(np.searchsorted(ordered_frames, center, side="left"))
        upper = int(np.searchsorted(ordered_frames, center, side="right"))
        rows = (
            np.arange(lower, upper, dtype=np.int64)
            if frame_order is None else frame_order[lower:upper]
        )
        if rows.size == 0:
            continue
        for semantic, instance in sorted(persistent):
            mask = (
                (source.labels.semantic == np.uint16(semantic))
                & (source.labels.instance == np.uint16(instance))
                & ~np.asarray(source.zero_slot_mask, dtype=np.bool_)
            )
            sensor_points = np.asarray(source.xyzi[mask, :3], dtype=np.float64)
            ranges = np.linalg.norm(sensor_points, axis=1)
            if sensor_points.shape[0] < 32 or not np.any((ranges >= 2.5) & (ranges <= 50.0)):
                continue
            world_points = sensor_points @ rotation.T + translation
            try:
                horizontal_hull = ConvexHull(world_points[:, :2])
            except QhullError:
                continue
            polygon = world_points[np.asarray(horizontal_hull.vertices), :2]
            allowed = normal_control_support_semantics(semantic)
            legal = rows[np.isin(pool.semantics[rows], tuple(allowed))]
            if legal.size == 0:
                continue
            patch_xy = pool.anchors_world_m[legal, :2]
            equations = np.asarray(horizontal_hull.equations, dtype=np.float64)
            expanded_inside = np.all(
                patch_xy @ equations[:, :2].T + equations[:, 2] <= 0.5,
                axis=1,
            )
            if bool(expanded_inside.any()):
                eligible_rows = legal[expanded_inside]
                center_xy = np.mean(polygon, axis=0)
                distance = np.linalg.norm(
                    pool.anchors_world_m[eligible_rows, :2] - center_xy, axis=1
                )
            else:
                starts = polygon
                ends = np.roll(polygon, -1, axis=0)
                edges = ends - starts
                denominator = np.sum(np.square(edges), axis=1)
                relative = patch_xy[:, None, :] - starts[None, :, :]
                fraction = np.clip(
                    np.sum(relative * edges[None, :, :], axis=2)
                    / denominator[None, :],
                    0.0, 1.0,
                )
                closest = starts[None, :, :] + fraction[:, :, None] * edges[None, :, :]
                distance = np.min(
                    np.linalg.norm(patch_xy[:, None, :] - closest, axis=2), axis=1
                )
                within = distance <= 1.0
                if not bool(within.any()):
                    continue
                eligible_rows = legal[within]
                distance = distance[within]
            row = int(
                eligible_rows[
                    np.lexsort((pool.selection_hashes[eligible_rows], distance))[0]
                ]
            )
            support_semantic = int(pool.semantics[row])
            identity = f"201:{center}:{semantic}:{instance}".encode("ascii")
            candidates.append((hashlib.sha256(identity).digest(), (center, semantic, instance, support_semantic)))
    ordered = tuple(value for _, value in sorted(candidates, key=lambda item: item[0]))
    if len(ordered) < 256:
        raise PlacementError("train/201 has fewer than 256 persistent supported real instances")
    return ordered


_GATE1_POOL: QualifiedSupportPool | None = None
_GATE1_OBSTACLES: ObservedObstacleIndex | None = None
_GATE1_TEMPLATES: tuple[NormalTemplateShape, ...] = ()
_GATE1_TEMPLATE_IDENTITIES: tuple[str, ...] = ()
_GATE1_CONTROL_CONTEXT: CoverageControlContext | None = None
_GATE1_REAL_CANDIDATES: tuple[tuple[int, int, int, int], ...] = ()
_GATE1_RAY_GRID: RayGrid | None = None
_GATE1_SENSOR: SensorCalibration | None = None
_GATE1_RENDERER_IDENTITY = ""


def _gate1_control_rows(
    context: CoverageControlContext,
    template_index: int,
    semantic: int,
    assigned_range_bin: int,
    center_frame: int,
) -> np.ndarray:
    """Keep the frozen global stream order inside the paired five-frame unit."""

    rows = _coverage_control_support_stream(
        context, template_index, semantic, assigned_range_bin
    )
    return rows[np.abs(context.support_pool.frames[rows] - center_frame) <= 2]


def _gate1_control_template_assignment(attempt_seed: int) -> tuple[int, int]:
    """Draw one canonical template and derive its immutable range-bin identity."""

    template_index = int(
        np.random.default_rng(_integer("attempt_seed", attempt_seed) + 1).integers(
            0, 256
        )
    )
    return template_index, _e25_new_assigned_range_bin(template_index)


def _gate1_observation_dict(
    observation: _E25NewObservation,
) -> dict[str, int | float]:
    return {
        "frame_id": observation.frame_id,
        "visible_returns": observation.visible_returns,
        "visible_in_range_returns": observation.visible_in_range_returns,
        "accepted_in_range_returns": observation.accepted_in_range_returns,
        "geometry_in_range_hits": observation.geometry_in_range_hits,
        "median_official_range_m": observation.median_official_range_m,
        "median_beam": observation.median_beam,
        "range_bin": observation.range_bin,
        "azimuth_sector": observation.azimuth_sector,
        "occlusion": observation.occlusion,
    }


def _gate1_bank_worker(index: int) -> dict[str, object]:
    pool, obstacles, control_context = (
        _GATE1_POOL, _GATE1_OBSTACLES, _GATE1_CONTROL_CONTEXT
    )
    sequence, grid, sensor = _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR
    if (
        pool is None or obstacles is None or sequence is None or grid is None or sensor is None
        or control_context is None or len(_GATE1_TEMPLATES) != 256
        or len(_GATE1_TEMPLATE_IDENTITIES) != 256 or not _GATE1_REAL_CANDIDATES
    ):
        raise RuntimeError("Gate 1 candidate-bank fixtures are not initialized")
    if not 0 <= index < _GATE1_BANK_CAPACITY_LIMIT:
        raise RuntimeError("Gate 1 bank index exceeds its frozen capacity")
    bank_seed = _GATE1_BANK_SEED_BASE + index
    for attempt in range(48):
        attempt_seed = bank_seed + 1_000_003 * attempt
        real_index = int(
            np.random.default_rng(np.random.SeedSequence([attempt_seed, 3801])).integers(
                0, len(_GATE1_REAL_CANDIDATES)
            )
        )
        center, real_semantic, real_instance, real_support_semantic = (
            _GATE1_REAL_CANDIDATES[real_index]
        )
        try:
            template_seed = attempt_seed + 1
            scale_seed = attempt_seed + 2
            template_index, assigned_range_bin = (
                _gate1_control_template_assignment(attempt_seed)
            )
            source_template = _GATE1_TEMPLATES[template_index]
            rows = _gate1_control_rows(
                control_context,
                template_index,
                int(source_template.raw_semantic_id),
                assigned_range_bin,
                center,
            )
            if rows.size == 0:
                continue
            scale = np.random.default_rng(
                np.random.SeedSequence([scale_seed, 2501])
            ).uniform(0.9, 1.1, size=3)
            control_shape = _aligned_scaled_template(source_template, scale)
            grounding = qualify_grounding(control_shape)
            perturbation = float(
                np.random.default_rng(np.random.SeedSequence([attempt_seed + 31, 2502])).uniform(
                    -math.pi if control_shape.raw_semantic_id == 30 else -math.radians(15.0),
                    math.pi if control_shape.raw_semantic_id == 30 else math.radians(15.0),
                )
            )

            def control_yaw(patch: SupportPatch, offset: float = perturbation) -> float:
                return float(control_context.trajectory_yaw_by_frame[patch.frame_id]) + offset

            observations: dict[int, _E25NewObservation] = {}

            def reject_control(
                proposed: ObjectSpec, patch: SupportPatch,
            ) -> str | None:
                observation = _coverage_control_observation(
                    control_context,
                    proposed,
                    patch,
                    bank_seed,
                    assigned_range_bin,
                    (proposed,),
                )
                observations[patch.pool_index] = observation
                if observation.visible_returns < 1:
                    return "no_visible_normal_control_return"
                if observation.range_bin != assigned_range_bin:
                    return "assigned_visible_range_bin_mismatch"
                return None

            control_material_seed = attempt_seed + 11
            control, control_record = place_object(
                control_shape, MaterialSpec.sample(control_material_seed), pool, obstacles,
                object_id=1, label="normal-control", proposal_namespace="E25-new-support-v1",
                proposal_stream=template_index, yaw_rad=perturbation,
                material_seed=control_material_seed, yaw_seed=attempt_seed + 31,
                template_identity=_GATE1_TEMPLATE_IDENTITIES[template_index],
                proposal_rows=rows, grounding_eligibility=grounding,
                yaw_for_support=control_yaw, post_placement_rejection=reject_control,
            )
            control_record = replace(
                control_record,
                template_seed=template_seed,
                scale_seed=scale_seed,
                pose_perturbation_rad=perturbation,
            )
            control_observation = observations[control_record.support_pool_index]
            proxy_shape, proxy_report, proxy_grounding, shape_proposals, grounding_rejections = _grounding_qualified_shape(
                attempt_seed + 3, stride=3072, maximum_proposals=64
            )
            proxy_material_seed = attempt_seed + 12
            proxy_yaw_seed = attempt_seed + 32
            proxy_yaw = float(np.random.default_rng(proxy_yaw_seed).uniform(-math.pi, math.pi))
            proxy, proxy_record = place_object(
                proxy_shape, MaterialSpec.sample(proxy_material_seed), pool, obstacles,
                object_id=1, label="anomaly-proxy", proposal_namespace="gate1-proxy-v2",
                proposal_stream=attempt_seed, yaw_rad=proxy_yaw,
                material_seed=proxy_material_seed, yaw_seed=proxy_yaw_seed,
                shape_seed=shape_proposals[-1], shape_generation_report=proxy_report,
                proposal_rows=rows, grounding_eligibility=proxy_grounding,
            )
            control_world = WorldSpec(bank_seed, 201, (control,))
            proxy_world = WorldSpec(bank_seed, 201, (proxy,))
            # The accepted control already won a final renderer slot in its
            # support frame, which lies inside this five-frame unit. Check the
            # proxy lazily and stop at its first final visible return.
            proxy_visible = False
            for frame_id in range(center - 2, center + 3):
                frame = sequence.source_frame(frame_id)
                _, _, _, rendered = _gate1_single_object_trace(
                    frame, proxy_world, grid, sensor
                )
                if bool(np.asarray(rendered.anomaly_proxy_mask).any()):
                    proxy_visible = True
                    break
            if not proxy_visible:
                raise PlacementError(
                    "anomaly-proxy has no final visible return in its five-frame unit"
                )
        except PlacementExhaustion:
            continue
        except PlacementError as error:
            if str(error) in {
                "no E22-qualified shape in 64 deterministic proposals",
                "shape fails E22 grounding eligibility",
                "anomaly-proxy has no final visible return in its five-frame unit",
            }:
                continue
            raise
        return {
            "bank_seed": bank_seed, "attempt": attempt, "center_frame": center,
            "real_semantic": real_semantic, "real_instance": real_instance,
            "real_support_semantic": real_support_semantic,
            "control_support_semantic": control_record.support_semantic,
            "proxy_support_semantic": proxy_record.support_semantic,
            "control_support_frame": control_record.support_frame,
            "proxy_support_frame": proxy_record.support_frame,
            "control_template_index": template_index,
            "control_template_identity": _GATE1_TEMPLATE_IDENTITIES[template_index],
            "control_assigned_range_bin": assigned_range_bin,
            "control_final_range_bin": control_observation.range_bin,
            "control_visible_returns": control_observation.visible_returns,
            "control_observation_json": json.dumps(
                _gate1_observation_dict(control_observation),
                sort_keys=True, separators=(",", ":"),
            ),
            "control_world_json": control_world.to_json(),
            "proxy_world_json": proxy_world.to_json(),
            "control_record_json": json.dumps(control_record.to_dict(), sort_keys=True, separators=(",", ":")),
            "proxy_record_json": json.dumps(
                replace(
                    proxy_record, accepted_shape_proposal=len(shape_proposals) - 1,
                    shape_proposal_seeds=shape_proposals,
                    grounding_rejection_seeds=grounding_rejections,
                ).to_dict(), sort_keys=True, separators=(",", ":"),
            ),
            "error": "",
        }
    return {
        "bank_seed": bank_seed, "attempt": 48, "center_frame": -1,
        "real_semantic": 0, "real_instance": 0,
        "real_support_semantic": 0, "control_support_semantic": 0,
        "proxy_support_semantic": 0, "control_support_frame": -1,
        "proxy_support_frame": -1, "control_template_index": -1,
        "control_template_identity": "", "control_assigned_range_bin": -1,
        "control_final_range_bin": -1, "control_visible_returns": 0,
        "control_observation_json": "",
        "control_world_json": "", "proxy_world_json": "",
        "control_record_json": "", "proxy_record_json": "",
        "error": "placement_exhaustion",
    }


def _gate1_bank_arrays(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    return {
        "bank_seed": np.asarray([record["bank_seed"] for record in records], dtype=np.int64),
        "attempt": np.asarray([record["attempt"] for record in records], dtype=np.int16),
        "center_frame": np.asarray([record["center_frame"] for record in records], dtype=np.int16),
        "real_semantic": np.asarray([record["real_semantic"] for record in records], dtype=np.uint16),
        "real_instance": np.asarray([record["real_instance"] for record in records], dtype=np.uint16),
        "real_support_semantic": np.asarray([record["real_support_semantic"] for record in records], dtype=np.uint16),
        "control_support_semantic": np.asarray([record["control_support_semantic"] for record in records], dtype=np.uint16),
        "proxy_support_semantic": np.asarray([record["proxy_support_semantic"] for record in records], dtype=np.uint16),
        "control_support_frame": np.asarray([record["control_support_frame"] for record in records], dtype=np.int16),
        "proxy_support_frame": np.asarray([record["proxy_support_frame"] for record in records], dtype=np.int16),
        "control_template_index": np.asarray([record["control_template_index"] for record in records], dtype=np.int16),
        "control_template_identity": np.asarray([record["control_template_identity"] for record in records]),
        "control_assigned_range_bin": np.asarray([record["control_assigned_range_bin"] for record in records], dtype=np.int8),
        "control_final_range_bin": np.asarray([record["control_final_range_bin"] for record in records], dtype=np.int8),
        "control_visible_returns": np.asarray([record["control_visible_returns"] for record in records], dtype=np.int32),
        "control_observation_json": np.asarray([record["control_observation_json"] for record in records]),
        "control_world_json": np.asarray([record["control_world_json"] for record in records]),
        "proxy_world_json": np.asarray([record["proxy_world_json"] for record in records]),
        "control_record_json": np.asarray([record["control_record_json"] for record in records]),
        "proxy_record_json": np.asarray([record["proxy_record_json"] for record in records]),
        "error": np.asarray([record["error"] for record in records]),
    }


def _initialize_gate1_candidate_generation(
    sequence: object, control_context: CoverageControlContext,
    obstacles: ObservedObstacleIndex, templates: Sequence[NormalTemplateShape],
    real_candidates: Sequence[tuple[int, int, int, int]] | None = None,
) -> None:
    template_tuple = tuple(templates)
    identities, _, _ = canonical_normal_template_library_identity(template_tuple)
    if control_context.source_sequence_id != 201:
        raise RenderError("Gate 1 control context must be bound to train/201")
    precompute_coverage_control_support_streams(control_context, template_tuple)
    global _GATE1_SEQUENCE, _GATE1_POOL, _GATE1_OBSTACLES, _GATE1_TEMPLATES
    global _GATE1_TEMPLATE_IDENTITIES, _GATE1_CONTROL_CONTEXT
    global _GATE1_REAL_CANDIDATES, _GATE1_RAY_GRID, _GATE1_SENSOR
    _GATE1_SEQUENCE = sequence
    _GATE1_POOL = control_context.support_pool
    _GATE1_OBSTACLES = obstacles
    _GATE1_TEMPLATES = template_tuple
    _GATE1_TEMPLATE_IDENTITIES = identities
    _GATE1_CONTROL_CONTEXT = control_context
    _GATE1_REAL_CANDIDATES = (
        tuple(real_candidates) if real_candidates is not None
        else _gate1_real_candidates(sequence, control_context.support_pool)
    )
    _GATE1_RAY_GRID = control_context.ray_grid
    _GATE1_SENSOR = control_context.sensor


def _write_gate1_candidate_bank(
    arrays: Mapping[str, np.ndarray], output_path: Path | str,
    *, processes: int, elapsed_seconds: float, support_pool_sha256: str,
    calibration_sha256: str, normal_template_library_sha256: str,
) -> dict[str, object]:
    errors = int(np.count_nonzero(arrays["error"] != ""))
    capacity = int(arrays["bank_seed"].size)
    completed = arrays["error"] == ""
    contract_errors = int(
        np.count_nonzero(
            completed & (
                (arrays["control_template_index"] < 0)
                | (arrays["control_template_index"] >= 256)
                | (arrays["control_assigned_range_bin"] != arrays["control_template_index"] % 5)
                | (arrays["control_final_range_bin"] != arrays["control_assigned_range_bin"])
                | (arrays["control_visible_returns"] < 1)
                | (np.abs(arrays["control_support_frame"] - arrays["center_frame"]) > 2)
                | (np.abs(arrays["proxy_support_frame"] - arrays["center_frame"]) > 2)
            )
        )
    )
    seed_identity_errors = int(not np.array_equal(
        arrays["bank_seed"], np.arange(3_800_000, 3_800_000 + capacity)
    ))
    metadata = {
        "experiment": "Gate1-candidate-bank-v2",
        "schema": "gate1-candidate-bank-v2",
        "passed": (
            capacity == 256 and errors == 0
            and contract_errors == 0 and seed_identity_errors == 0
        ),
        "capacity": capacity, "completed": capacity - errors, "errors": errors,
        "contract_errors": contract_errors,
        "seed_identity_errors": seed_identity_errors,
        "source_sequence": "train/201", "paired_seed_range": [3800000, 3800255],
        "real_candidate_count": len(_GATE1_REAL_CANDIDATES),
        "center_frame_count": int(np.unique(
            arrays["center_frame"][arrays["center_frame"] >= 0]
        ).size),
        "support_semantic_counts": {
            source_name: {
                str(semantic): int(np.count_nonzero(arrays[field] == semantic))
                for semantic in (40, 48)
            }
            for source_name, field in (
                ("real-normal", "real_support_semantic"),
                ("normal-control", "control_support_semantic"),
                ("anomaly-proxy", "proxy_support_semantic"),
            )
        },
        "control_template_unique_count": int(np.unique(
            arrays["control_template_index"][completed]
        ).size),
        "control_assigned_range_count": np.bincount(
            arrays["control_assigned_range_bin"][completed], minlength=5
        )[:5].tolist(),
        "control_final_range_count": np.bincount(
            arrays["control_final_range_bin"][completed], minlength=5
        )[:5].tolist(),
        "control_visible_returns_minimum": (
            int(np.min(arrays["control_visible_returns"][completed]))
            if bool(np.any(completed)) else None
        ),
        "normal_template_library_sha256": normal_template_library_sha256,
        "support_pool_sha256": support_pool_sha256,
        "calibration_sha256": calibration_sha256,
        "renderer_identity": _GATE1_RENDERER_IDENTITY,
        "maximum_attempt": int(np.max(arrays["attempt"])),
        "processes": processes, "elapsed_seconds": elapsed_seconds,
        "scientific_array_hash": _scientific_array_hash(arrays),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return metadata


def build_gate1_candidate_bank(
    sequence: object,
    control_context: CoverageControlContext,
    obstacles: ObservedObstacleIndex,
    templates: Sequence[NormalTemplateShape],
    output_path: Path | str,
    *,
    processes: int = 24,
    capacity: int = 256,
    support_pool_sha256: str,
    calibration_sha256: str,
    normal_template_library_sha256: str,
    real_candidates: Sequence[tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    """Build a frozen independent three-source candidate bank."""
    if processes != 24:
        raise RenderError("formal Gate 1 candidate bank requires 24 processes")
    if capacity != 256:
        raise RenderError("Gate 1 v2 candidate bank has exactly 256 paired seeds")
    global _GATE1_BANK_SEED_BASE, _GATE1_BANK_CAPACITY_LIMIT
    _GATE1_BANK_SEED_BASE = 3_800_000
    _GATE1_BANK_CAPACITY_LIMIT = 256
    _initialize_gate1_candidate_generation(
        sequence, control_context, obstacles, templates, real_candidates
    )
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        records = workers.map(_gate1_bank_worker, range(capacity), chunksize=1)
    return _write_gate1_candidate_bank(
        _gate1_bank_arrays(records), output_path, processes=processes,
        elapsed_seconds=time.monotonic() - started,
        support_pool_sha256=support_pool_sha256,
        calibration_sha256=calibration_sha256,
        normal_template_library_sha256=normal_template_library_sha256,
    )


def extend_gate1_candidate_bank(
    existing_path: Path | str, output_path: Path | str, target_capacity: int,
    *, processes: int = 24,
) -> dict[str, object]:
    del existing_path, output_path, target_capacity, processes
    raise RenderError(
        "Gate 1 v2 has no three-source capacity ladder; E45A-new and E45B-v2 use separate banks"
    )


@dataclass(frozen=True, slots=True)
class _Gate1BankUnit:
    bank_seed: int
    center_frame: int
    real_semantic: int
    real_instance: int
    real_support_semantic: int
    control_support_semantic: int
    proxy_support_semantic: int
    control_template_index: int
    control_assigned_range_bin: int
    control_final_range_bin: int
    control_visible_returns: int
    control_world: WorldSpec
    proxy_world: WorldSpec


_E38_BANK: tuple[_Gate1BankUnit, ...] = ()


def _gate1_object_geometry(
    frame: SourceFrame, world: WorldSpec, item: ObjectSpec, ray_grid: RayGrid
) -> tuple[np.ndarray, np.ndarray]:
    rotation, lidar_origin = _pose(frame)
    directions_world = ray_grid.directions_for(frame) @ rotation.T
    origins_world = ray_grid.origins_for(frame) @ rotation.T + lidar_origin
    object_rotation = np.asarray(item.rotation_world_from_local, dtype=np.float64)
    translation = np.asarray(item.translation_world_m, dtype=np.float64)
    local_origin = (origins_world - translation) @ object_rotation
    local_direction = directions_world @ object_rotation
    distance, _, valid = item.shape.intersect(local_origin, local_direction)
    official_distance = np.asarray(distance) + ray_grid.official_range_offset_m
    valid = np.asarray(valid) & (official_distance >= 2.5) & (official_distance <= 50.0)
    return valid, official_distance


def _gate1_real_geometry(
    frame: SourceFrame, semantic: int, instance: int, ray_grid: RayGrid,
    trace_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert frame.labels is not None
    returned = (
        (frame.labels.semantic == np.uint16(semantic))
        & (frame.labels.instance == np.uint16(instance))
        & ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
    )
    if trace_context is None:
        ranges = np.asarray(ray_grid.official_ranges(frame))
    else:
        ranges = np.asarray(trace_context[4]).copy()
        ranges[np.isfinite(ranges)] += ray_grid.official_range_offset_m
    returned &= (ranges >= 2.5) & (ranges <= 50.0)
    points = np.asarray(frame.xyzi[returned, :3], dtype=np.float64)
    if points.shape[0] < 4:
        raise RenderError("Gate 1 real instance has fewer than four in-range returns")
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise RenderError("Gate 1 real instance hull is not volumetric") from error
    equations = np.asarray(hull.equations, dtype=np.float64)
    if trace_context is None:
        origins = ray_grid.origins_for(frame)
        directions = ray_grid.directions_for(frame)
    else:
        directions, _, origins, _, _ = trace_context
    distance = np.full(ray_grid.slot_count, np.inf, dtype=np.float64)
    valid = np.zeros(ray_grid.slot_count, dtype=np.bool_)
    # A finite sphere enclosing all observed instance points is a conservative
    # broad phase for their convex hull; excluded rays cannot enter that hull.
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = float(np.max(np.linalg.norm(points - center, axis=1))) + 1.0e-6
    unit_direction = directions / np.linalg.norm(
        directions, axis=1, keepdims=True
    )
    relative_origin = origins - center
    closest_parameter = np.maximum(
        -np.sum(relative_origin * unit_direction, axis=1), 0.0
    )
    closest = relative_origin + closest_parameter[:, None] * unit_direction
    broad_phase = np.linalg.norm(closest, axis=1) <= radius
    broad_phase |= returned
    candidate_slots = np.flatnonzero(broad_phase)
    for start in range(0, candidate_slots.size, 4096):
        selected = candidate_slots[start:start + 4096]
        numerator = -(origins[selected] @ equations[:, :3].T + equations[:, 3])
        denominator = directions[selected] @ equations[:, :3].T
        parallel_outside = np.any(
            (np.abs(denominator) <= EPSILON) & (numerator < 0.0), axis=1
        )
        lower_terms = np.full_like(denominator, -np.inf)
        upper_terms = np.full_like(denominator, np.inf)
        np.divide(numerator, denominator, out=lower_terms, where=denominator < -EPSILON)
        np.divide(numerator, denominator, out=upper_terms, where=denominator > EPSILON)
        enter = np.max(lower_terms, axis=1)
        exit_distance = np.min(upper_terms, axis=1)
        current = (
            ~parallel_outside & (exit_distance >= np.maximum(enter, 0.0))
            & (enter + ray_grid.official_range_offset_m >= 2.5)
            & (enter + ray_grid.official_range_offset_m <= 50.0)
        )
        distance[selected] = enter + ray_grid.official_range_offset_m
        valid[selected] = current
    opportunity = valid | returned
    distance = np.where(valid, distance, ranges)
    # Observed instance slots are authoritative members of the opportunity
    # union and retain their measured official return distance.
    distance[returned] = ranges[returned]
    return opportunity, returned, distance


def _e38_bootstrap(
    opportunities: np.ndarray, returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([3801, 2000]))
    clusters = opportunities.shape[0]
    weights = rng.multinomial(
        clusters, np.full(clusters, 1.0 / clusters), size=2000
    ).astype(np.float64)
    denominator = weights @ opportunities
    numerator = weights @ returns
    rate = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return (
        np.quantile(rate, 0.025, axis=0),
        np.quantile(rate, 0.975, axis=0),
    )


def _e38_trace_contract_errors(
    trace: Mapping[str, np.ndarray], ray_grid: RayGrid,
    units: Sequence[_Gate1BankUnit],
) -> int:
    """Cross-check every shared E38--E44 count and per-return identity."""

    beam_count = ray_grid.beam_count
    expected_shapes = {
        "bank_seed": (256, 3, 5), "source": (256, 3, 5),
        "frame_id": (256, 3, 5), "support_semantic": (256, 3, 5),
        "opportunity": (256, 3, 5, beam_count),
        "return_count": (256, 3, 5, beam_count),
        "median_distance_m": (256, 3, 5), "median_beam": (256, 3, 5),
        "range_opportunity": (256, 3, 5, 5),
        "range_return_count": (256, 3, 5, 5),
        "geometry_hits": (256, 3, 5), "accepted_hits": (256, 3, 5),
        "visible_returns": (256, 3, 5), "visible_distance_m": (256, 3, 5),
        "empty_slots": (256, 2, 5, beam_count),
        "empty_geometry": (256, 2, 5, beam_count, 5),
        "empty_accepted": (256, 2, 5, beam_count, 5),
        "empty_final_new": (256, 2, 5, beam_count, 5),
    }
    if len(units) != 256 or any(
        name not in trace or np.asarray(trace[name]).shape != shape
        for name, shape in expected_shapes.items()
    ):
        return 1
    per_return_names = (
        "intensity_source", "intensity_bank_seed", "intensity_frame",
        "intensity_slot", "intensity_beam", "intensity_official_range_m",
        "intensity_range_bin", "intensity_value",
    )
    if any(name not in trace or np.asarray(trace[name]).ndim != 1 for name in per_return_names):
        return 1
    return_count = int(np.asarray(trace[per_return_names[0]]).size)
    if any(np.asarray(trace[name]).size != return_count for name in per_return_names):
        return 1
    errors = 0
    errors += int(np.count_nonzero(
        np.sum(trace["opportunity"], axis=-1) != trace["geometry_hits"]
    ))
    errors += int(np.count_nonzero(
        np.sum(trace["return_count"], axis=-1) != trace["visible_returns"]
    ))
    errors += int(np.count_nonzero(
        np.sum(trace["range_opportunity"], axis=-1) != trace["geometry_hits"]
    ))
    errors += int(np.count_nonzero(
        np.sum(trace["range_return_count"], axis=-1) != trace["visible_returns"]
    ))
    errors += int(np.count_nonzero(trace["accepted_hits"] > trace["geometry_hits"]))
    errors += int(np.count_nonzero(trace["visible_returns"] > trace["accepted_hits"]))
    errors += int(np.count_nonzero(trace["empty_final_new"] > trace["empty_accepted"]))
    errors += int(np.count_nonzero(trace["empty_accepted"] > trace["empty_geometry"]))
    errors += int(np.count_nonzero(
        np.sum(trace["empty_geometry"], axis=-1) > trace["empty_slots"]
    ))
    expected_seed = np.broadcast_to(
        np.asarray([unit.bank_seed for unit in units], dtype=np.int64)[:, None, None],
        (256, 3, 5),
    )
    expected_source = np.broadcast_to(
        np.arange(3, dtype=np.uint8)[None, :, None], (256, 3, 5)
    )
    expected_frame = np.asarray([
        np.broadcast_to(
            np.arange(unit.center_frame - 2, unit.center_frame + 3)[None, :],
            (3, 5),
        )
        for unit in units
    ])
    errors += int(np.count_nonzero(trace["bank_seed"] != expected_seed))
    errors += int(np.count_nonzero(trace["source"] != expected_source))
    errors += int(np.count_nonzero(trace["frame_id"] != expected_frame))
    errors += int(np.count_nonzero(~np.isin(trace["support_semantic"], (40, 48))))
    if return_count:
        source = np.asarray(trace["intensity_source"])
        seed = np.asarray(trace["intensity_bank_seed"])
        frame = np.asarray(trace["intensity_frame"])
        slot = np.asarray(trace["intensity_slot"])
        beam = np.asarray(trace["intensity_beam"])
        official_range = np.asarray(trace["intensity_official_range_m"])
        range_bin = np.asarray(trace["intensity_range_bin"])
        errors += int(np.count_nonzero((source < 0) | (source > 2)))
        errors += int(np.count_nonzero((seed < 3_800_000) | (seed > 3_800_255)))
        errors += int(np.count_nonzero((frame < 4) | (frame > 681)))
        errors += int(np.count_nonzero((slot < 0) | (slot >= ray_grid.slot_count)))
        errors += int(np.count_nonzero(beam != slot // ray_grid.columns))
        errors += int(np.count_nonzero(
            ~np.isfinite(official_range)
            | (official_range < 2.5) | (official_range > 50.0)
        ))
        errors += int(np.count_nonzero(
            range_bin != _gate1_range_bin(official_range)
        ))
        observed_by_source = np.bincount(source, minlength=3)[:3]
        errors += int(np.count_nonzero(
            observed_by_source != np.sum(trace["visible_returns"], axis=(0, 2))
        ))
    return errors


def run_e38_v2_qualification(
    data_root: Path | str,
    e25_new_artifact_path: Path | str,
    calibration_path: Path | str,
    support_pool_path: Path | str,
    candidate_bank_output: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Build E25-new Gate 1 units and save one shared E38--E44 trace."""
    if processes != 24:
        raise RenderError("formal E38 requires exactly 24 processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    e25_new_path = Path(e25_new_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(e25_new_path) != FROZEN_E25_NEW_ARTIFACT_SHA256:
        raise RenderError("E38-v2 E25-new artifact identity changed")
    sequence_206 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    templates = extract_normal_template_library(
        sequence_206.source_frame(frame_id) for frame_id in sequence_206.frame_ids
    )
    template_identities, template_counts, template_library_hash = (
        canonical_normal_template_library_identity(templates)
    )
    with np.load(e25_new_path, allow_pickle=False) as source:
        e25_new_metadata = json.loads(str(source["metadata_json"]))
        e25_fixture = np.asarray(source["fixture_index"])
        e25_identity = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["template_identity"]
        )
        e25_assigned = np.asarray(source["assigned_range_bin"])
    if (
        e25_new_metadata.get("experiment") != "E25-new-normal-control"
        or e25_new_metadata.get("passed") is not True
        or tuple(e25_fixture.tolist()) != tuple(range(256))
        or e25_identity != template_identities
        or not np.array_equal(e25_assigned, np.arange(256) % 5)
    ):
        raise RenderError("E38-v2 E25-new template contract changed")
    del sequence_206
    gc.collect()

    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    support_path = Path(support_pool_path).expanduser().resolve(strict=True)
    pool, pool_metadata = load_gate1_support_pool(support_path)
    calibration_resolved = Path(calibration_path).expanduser().resolve(strict=True)
    calibration_hash = _sha256_path(calibration_resolved)
    if calibration_hash != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise RenderError("E38-v2 sensor calibration identity changed")
    ray_grid, sensor = load_sensor_calibration(calibration_resolved)
    frame_ids = tuple(range(4, 682))
    trajectory_yaws = _trajectory_yaw_by_pose({
        frame_id: sequence.lidar_pose(frame_id) for frame_id in frame_ids
    })
    frame_keys, obstacles = _gate1_frame_keys_and_obstacles(
        sequence, build_obstacles=True,
    )
    assert obstacles is not None
    real_candidates = _gate1_real_candidates(
        sequence, pool, frame_keys=frame_keys
    )
    control_context = build_coverage_control_context(
        (), pool, ray_grid, sensor,
        frame_loader=sequence.source_frame,
        frame_ids=frame_ids,
        source_sequence_id=201,
        trajectory_yaws=trajectory_yaws,
    )
    renderer_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    global _GATE1_RENDERER_IDENTITY
    _GATE1_RENDERER_IDENTITY = renderer_identity
    bank_metadata = build_gate1_candidate_bank(
        sequence, control_context, obstacles, templates, candidate_bank_output,
        processes=processes,
        support_pool_sha256=FROZEN_GATE1_SUPPORT_POOL_SHA256,
        calibration_sha256=calibration_hash,
        normal_template_library_sha256=template_library_hash,
        real_candidates=real_candidates,
    )
    if bank_metadata["passed"] is not True:
        result = {
            "experiment": "E38-v2", "passed": False,
            "failure_classification": "candidate_bank_construction_failure",
            "support_pool": pool_metadata, "candidate_bank": bank_metadata,
            "formal_repetitions": 1,
            "elementwise_reproduced": None,
            "reproducibility_check": "not_run_by_owner_decision",
        }
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(
                json.dumps(result, sort_keys=True, separators=(",", ":"))
            ),
        )
        os.replace(temporary, destination)
        return result
    global _E38_BANK
    _E38_BANK = _load_gate1_bank(candidate_bank_output)
    global _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR
    _GATE1_SEQUENCE = sequence
    _GATE1_RAY_GRID = ray_grid
    _GATE1_SENSOR = sensor
    # Candidate generation state is not used by the shared render trace. Drop
    # the large obstacle index and support context before the second fork pool.
    global _GATE1_POOL, _GATE1_OBSTACLES, _GATE1_TEMPLATES
    global _GATE1_TEMPLATE_IDENTITIES, _GATE1_CONTROL_CONTEXT
    global _GATE1_REAL_CANDIDATES
    _GATE1_POOL = None
    _GATE1_OBSTACLES = None
    _GATE1_TEMPLATES = ()
    _GATE1_TEMPLATE_IDENTITIES = ()
    _GATE1_CONTROL_CONTEXT = None
    _GATE1_REAL_CANDIDATES = ()
    del pool, obstacles, templates, control_context, frame_keys, real_candidates
    gc.collect()
    fixed = {
        "bank_seed", "source", "frame_id", "support_semantic",
        "opportunity", "return_count", "median_distance_m", "median_beam",
        "range_opportunity", "range_return_count", "geometry_hits",
        "accepted_hits", "visible_returns", "visible_distance_m", "empty_slots",
        "empty_geometry", "empty_accepted", "empty_final_new",
    }
    work_order = tuple(sorted(
        range(256),
        key=lambda index: (
            -sum(
                int(item.shape.plane_normals.shape[0])
                if isinstance(item.shape, NormalTemplateShape)
                else item.shape.primitive_count
                for item in (
                    _E38_BANK[index].control_world.objects[0],
                    _E38_BANK[index].proxy_world.objects[0],
                )
            ),
            index,
        ),
    ))
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        scheduled = workers.map(_e39_worker, work_order, chunksize=1)
    trace_seconds = time.monotonic() - started
    by_index = {
        int(record["bank_seed"][0, 0]) - 3_800_000: record
        for record in scheduled
    }
    records = [by_index[index] for index in range(256)]
    trace = {
        name: (
            np.stack([record[name] for record in records])
            if name in fixed else np.concatenate([record[name] for record in records])
        )
        for name in records[0]
    }
    conservation_errors = int(np.count_nonzero(
        trace["return_count"] > trace["opportunity"]
    ))
    trace_contract_errors = _e38_trace_contract_errors(
        trace, ray_grid, _E38_BANK
    )
    total_opportunity = trace["opportunity"].sum(axis=(0, 2))
    total_returns = trace["return_count"].sum(axis=(0, 2))
    rates = np.divide(
        total_returns, total_opportunity, out=np.zeros_like(total_returns, dtype=np.float64),
        where=total_opportunity > 0,
    )
    ci_low = np.zeros((3, ray_grid.beam_count), dtype=np.float64)
    ci_high = np.zeros_like(ci_low)
    for source_index in range(3):
        groups_opportunity = trace["opportunity"][:, source_index].reshape(-1, ray_grid.beam_count)
        groups_return = trace["return_count"][:, source_index].reshape(-1, ray_grid.beam_count)
        ci_low[source_index], ci_high[source_index] = _e38_bootstrap(
            groups_opportunity, groups_return
        )
    finite_errors = int(
        np.count_nonzero(~np.isfinite(rates))
        + np.count_nonzero(~np.isfinite(ci_low))
        + np.count_nonzero(~np.isfinite(ci_high))
        + np.count_nonzero(~np.isfinite(trace["median_distance_m"]))
        + np.count_nonzero(~np.isfinite(trace["median_beam"]))
        + np.count_nonzero(~np.isfinite(trace["visible_distance_m"]))
        + np.count_nonzero(~np.isfinite(trace["intensity_official_range_m"]))
        + np.count_nonzero(~np.isfinite(trace["intensity_value"]))
    )
    source_nonzero = [int(total_returns[index].sum()) for index in range(3)]
    passed = (
        conservation_errors == 0 and trace_contract_errors == 0
        and finite_errors == 0
        and all(value > 0 for value in source_nonzero)
    )
    scientific = {
        **trace, "beam_opportunity": total_opportunity,
        "beam_return_count": total_returns, "beam_return_rate": rates,
        "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
    }
    result = {
        "experiment": "E38-v2", "passed": passed, "bank_seeds": 256,
        "failure_classification": None if passed else "per_beam_qualification_failure",
        "entity_frame_groups_per_source": 1280,
        "sources": ["real-normal", "normal-control", "anomaly-proxy"],
        "total_opportunities": [int(total_opportunity[index].sum()) for index in range(3)],
        "total_returns": source_nonzero, "conservation_errors": conservation_errors,
        "trace_contract_errors": trace_contract_errors,
        "finite_errors": finite_errors,
        "formal_repetitions": 1,
        "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "bootstrap_clusters_per_source": 1280, "bootstrap_replicates": 2000,
        "run_seconds": [trace_seconds], "processes": processes,
        "numeric_library_threads_per_process": 1, "gpu_used": False,
        "support_pool_size": int(pool_metadata["pool_size"]),
        "support_pool_reused": True,
        "support_pool_sha256": FROZEN_GATE1_SUPPORT_POOL_SHA256,
        "candidate_bank_completed": int(bank_metadata["completed"]),
        "candidate_bank_sha256": _sha256_path(
            Path(candidate_bank_output).expanduser().resolve(strict=True)
        ),
        "e25_new_artifact_sha256": FROZEN_E25_NEW_ARTIFACT_SHA256,
        "normal_template_counts": template_counts,
        "normal_template_library_sha256": template_library_hash,
        "calibration_sha256": calibration_hash,
        "renderer_identity": renderer_identity,
        "shared_trace_for_E39_E44": True,
        "scientific_array_hash": _scientific_array_hash(scientific),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_GATE1_RANGE_EDGES = np.asarray((2.5, 10.0, 20.0, 30.0, 40.0, 50.0), dtype=np.float64)


def _gate1_range_bin(distance_m: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=np.float64)
    bins = np.searchsorted(_GATE1_RANGE_EDGES, distance, side="right") - 1
    bins[distance == 50.0] = 4
    bins[(distance < 2.5) | (distance > 50.0)] = -1
    return bins.astype(np.int8)


def _gate1_single_object_trace(
    frame: SourceFrame, world: WorldSpec, ray_grid: RayGrid,
    sensor: SensorCalibration,
    trace_context: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RenderedFrame]:
    item = world.objects[0]
    shared = (
        _frame_trace_context(frame, ray_grid)
        if trace_context is None else trace_context
    )
    _, _, compact = _single_object_sensor_precheck(
        frame, world, ray_grid, sensor, trace_context=shared,
    )
    geometry, competition = _expand_single_object_trace(
        compact, item, ray_grid.slot_count
    )
    accepted = np.zeros(ray_grid.slot_count, dtype=np.bool_)
    accepted[compact.candidate_slots] = compact.accepted & compact.in_range
    official_distance = np.full(ray_grid.slot_count, np.inf, dtype=np.float64)
    official_distance[compact.candidate_slots] = (
        compact.distance_m + ray_grid.official_range_offset_m
    )
    rendered = render_frame(
        frame, world, ray_grid, sensor,
        _trace_context=shared, _competition=competition,
    )
    return geometry, accepted, official_distance, rendered


def _e39_worker(index: int) -> dict[str, np.ndarray]:
    sequence, grid, sensor = _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR
    if sequence is None or grid is None or sensor is None or len(_E38_BANK) != 256:
        raise RuntimeError("E39 shared-trace fixtures are not initialized")
    unit = _E38_BANK[index]
    bank_seed = unit.bank_seed
    center = unit.center_frame
    real_semantic = unit.real_semantic
    real_instance = unit.real_instance
    control_world = unit.control_world
    proxy_world = unit.proxy_world
    source_support_semantic = np.asarray(
        (
            unit.real_support_semantic,
            unit.control_support_semantic,
            unit.proxy_support_semantic,
        ),
        dtype=np.uint16,
    )
    beam_opportunity = np.zeros((3, 5, grid.beam_count), dtype=np.int32)
    beam_returns = np.zeros_like(beam_opportunity)
    median_distance_m = np.zeros((3, 5), dtype=np.float64)
    median_beam = np.zeros((3, 5), dtype=np.float64)
    range_opportunity = np.zeros((3, 5, 5), dtype=np.int32)
    range_returns = np.zeros_like(range_opportunity)
    geometry_hits = np.zeros((3, 5), dtype=np.int32)
    accepted_hits = np.zeros_like(geometry_hits)
    visible_returns = np.zeros_like(geometry_hits)
    visible_distance_m = np.zeros((3, 5), dtype=np.float64)
    empty_slots = np.zeros((2, 5, grid.beam_count), dtype=np.int32)
    empty_geometry = np.zeros((2, 5, grid.beam_count, 5), dtype=np.int32)
    empty_accepted = np.zeros_like(empty_geometry)
    empty_final_new = np.zeros_like(empty_geometry)
    intensity_source: list[np.ndarray] = []
    intensity_bank_seed: list[np.ndarray] = []
    intensity_frame: list[np.ndarray] = []
    intensity_slot: list[np.ndarray] = []
    intensity_beam: list[np.ndarray] = []
    intensity_official_range_m: list[np.ndarray] = []
    intensity_range_bin: list[np.ndarray] = []
    intensity_value: list[np.ndarray] = []
    for frame_offset, frame_id in enumerate(range(center - 2, center + 3)):
        frame = sequence.source_frame(frame_id)
        trace_context = _frame_trace_context(frame, grid)
        real_geometry, real_return, real_distance = _gate1_real_geometry(
            frame, real_semantic, real_instance, grid, trace_context
        )
        real_bin = _gate1_range_bin(real_distance)
        beam_opportunity[0, frame_offset] = np.bincount(
            grid.beam_ids[real_geometry], minlength=grid.beam_count
        )
        beam_returns[0, frame_offset] = np.bincount(
            grid.beam_ids[real_return], minlength=grid.beam_count
        )
        median_distance_m[0, frame_offset] = float(
            np.median(real_distance[real_geometry])
        )
        median_beam[0, frame_offset] = float(
            np.median(grid.beam_ids[real_geometry])
        )
        for range_bin in range(5):
            range_opportunity[0, frame_offset, range_bin] = int(
                np.count_nonzero(real_geometry & (real_bin == range_bin))
            )
            range_returns[0, frame_offset, range_bin] = int(
                np.count_nonzero(real_return & (real_bin == range_bin))
            )
        geometry_hits[0, frame_offset] = int(np.count_nonzero(real_geometry))
        accepted_hits[0, frame_offset] = geometry_hits[0, frame_offset]
        visible_returns[0, frame_offset] = int(np.count_nonzero(real_return))
        visible_distance_m[0, frame_offset] = float(np.median(real_distance[real_return]))
        real_slots = np.flatnonzero(real_return)
        real_return_distance = np.asarray(grid.official_ranges(frame))[real_slots]
        intensity_source.append(np.zeros(real_slots.size, dtype=np.uint8))
        intensity_bank_seed.append(
            np.full(real_slots.size, bank_seed, dtype=np.int64)
        )
        intensity_frame.append(np.full(real_slots.size, frame_id, dtype=np.int16))
        intensity_slot.append(real_slots.astype(np.int32))
        intensity_beam.append(grid.beam_ids[real_slots].astype(np.int16))
        intensity_official_range_m.append(real_return_distance.astype(np.float64))
        intensity_range_bin.append(_gate1_range_bin(real_return_distance))
        intensity_value.append(np.asarray(frame.xyzi[real_slots, 3], dtype=np.float32))
        native_empty = np.asarray(frame.zero_slot_mask, dtype=np.bool_)
        for source_index, world in enumerate((control_world, proxy_world), start=1):
            geometry, accepted, distance, rendered = _gate1_single_object_trace(
                frame, world, grid, sensor, trace_context
            )
            returned = np.array(
                rendered.normal_control_mask if source_index == 1
                else rendered.anomaly_proxy_mask,
                dtype=np.bool_, copy=True,
            )
            rendered_distance = np.asarray(grid.official_ranges(rendered.source))
            returned &= (rendered_distance >= 2.5) & (rendered_distance <= 50.0)
            distance_bin = _gate1_range_bin(distance)
            returned_bin = _gate1_range_bin(rendered_distance)
            beam_opportunity[source_index, frame_offset] = np.bincount(
                grid.beam_ids[geometry], minlength=grid.beam_count
            )
            beam_returns[source_index, frame_offset] = np.bincount(
                grid.beam_ids[returned], minlength=grid.beam_count
            )
            if bool(geometry.any()):
                median_distance_m[source_index, frame_offset] = float(
                    np.median(distance[geometry])
                )
                median_beam[source_index, frame_offset] = float(
                    np.median(grid.beam_ids[geometry])
                )
            else:
                pose_rotation, lidar_origin = _pose(frame)
                center_sensor = (
                    np.asarray(world.objects[0].translation_world_m) - lidar_origin
                ) @ pose_rotation
                center_distance = float(np.linalg.norm(center_sensor))
                direction = center_sensor / center_distance
                nearest = int(np.argmax(grid.directions_for(frame) @ direction))
                median_distance_m[source_index, frame_offset] = (
                    center_distance + grid.official_range_offset_m
                )
                median_beam[source_index, frame_offset] = float(
                    grid.beam_ids[nearest]
                )
            for range_bin in range(5):
                range_opportunity[source_index, frame_offset, range_bin] = int(
                    np.count_nonzero(geometry & (distance_bin == range_bin))
                )
                range_returns[source_index, frame_offset, range_bin] = int(
                    np.count_nonzero(returned & (returned_bin == range_bin))
                )
            geometry_hits[source_index, frame_offset] = int(np.count_nonzero(geometry))
            accepted_hits[source_index, frame_offset] = int(np.count_nonzero(accepted))
            visible_returns[source_index, frame_offset] = int(np.count_nonzero(returned))
            if bool(returned.any()):
                visible_distance_m[source_index, frame_offset] = float(
                    np.median(rendered_distance[returned])
                )
            elif bool(geometry.any()):
                visible_distance_m[source_index, frame_offset] = float(
                    np.median(distance[geometry])
                )
            else:
                pose_rotation, lidar_origin = _pose(frame)
                center_sensor = (
                    np.asarray(world.objects[0].translation_world_m) - lidar_origin
                ) @ pose_rotation
                visible_distance_m[source_index, frame_offset] = float(
                    np.linalg.norm(center_sensor) + grid.official_range_offset_m
                )
            label_index = source_index - 1
            empty_slots[label_index, frame_offset] = np.bincount(
                grid.beam_ids[native_empty], minlength=grid.beam_count
            )
            for range_bin in range(5):
                selection = native_empty & (distance_bin == range_bin)
                empty_geometry[label_index, frame_offset, :, range_bin] = np.bincount(
                    grid.beam_ids[selection & geometry], minlength=grid.beam_count
                )
                empty_accepted[label_index, frame_offset, :, range_bin] = np.bincount(
                    grid.beam_ids[selection & accepted], minlength=grid.beam_count
                )
                empty_final_new[label_index, frame_offset, :, range_bin] = np.bincount(
                    grid.beam_ids[selection & returned], minlength=grid.beam_count
                )
            slots = np.flatnonzero(returned)
            intensity_source.append(np.full(slots.size, source_index, dtype=np.uint8))
            intensity_bank_seed.append(
                np.full(slots.size, bank_seed, dtype=np.int64)
            )
            intensity_frame.append(np.full(slots.size, frame_id, dtype=np.int16))
            intensity_slot.append(slots.astype(np.int32))
            intensity_beam.append(grid.beam_ids[slots].astype(np.int16))
            intensity_official_range_m.append(
                rendered_distance[slots].astype(np.float64)
            )
            intensity_range_bin.append(returned_bin[slots])
            intensity_value.append(np.asarray(rendered.xyzi[slots, 3], dtype=np.float32))
    return {
        "bank_seed": np.full((3, 5), bank_seed, dtype=np.int64),
        "source": np.broadcast_to(np.arange(3, dtype=np.uint8)[:, None], (3, 5)).copy(),
        "frame_id": np.broadcast_to(np.arange(center - 2, center + 3, dtype=np.int16), (3, 5)).copy(),
        "support_semantic": np.broadcast_to(
            source_support_semantic[:, None], (3, 5)
        ).copy(),
        "opportunity": beam_opportunity,
        "return_count": beam_returns,
        "median_distance_m": median_distance_m,
        "median_beam": median_beam,
        "range_opportunity": range_opportunity, "range_return_count": range_returns,
        "geometry_hits": geometry_hits, "accepted_hits": accepted_hits,
        "visible_returns": visible_returns, "visible_distance_m": visible_distance_m,
        "empty_slots": empty_slots, "empty_geometry": empty_geometry,
        "empty_accepted": empty_accepted, "empty_final_new": empty_final_new,
        "intensity_source": np.concatenate(intensity_source),
        "intensity_bank_seed": np.concatenate(intensity_bank_seed),
        "intensity_frame": np.concatenate(intensity_frame),
        "intensity_slot": np.concatenate(intensity_slot),
        "intensity_beam": np.concatenate(intensity_beam),
        "intensity_official_range_m": np.concatenate(
            intensity_official_range_m
        ),
        "intensity_range_bin": np.concatenate(intensity_range_bin),
        "intensity_value": np.concatenate(intensity_value),
    }


def _load_gate1_bank(path: Path | str) -> tuple[_Gate1BankUnit, ...]:
    source_path = Path(path).expanduser().resolve(strict=True)
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if (
            metadata.get("experiment") != "Gate1-candidate-bank-v2"
            or metadata.get("schema") != "gate1-candidate-bank-v2"
            or metadata.get("passed") is not True
            or metadata.get("normal_template_library_sha256")
            != CANONICAL_NORMAL_TEMPLATE_LIBRARY_SHA256
            or metadata.get("support_pool_sha256")
            != FROZEN_GATE1_SUPPORT_POOL_SHA256
            or metadata.get("calibration_sha256")
            != FROZEN_SENSOR_CALIBRATION_SHA256
            or int(metadata.get("capacity", -1)) != 256
            or metadata.get("scientific_array_hash") is None
            or (
                _GATE1_RENDERER_IDENTITY
                and metadata.get("renderer_identity") != _GATE1_RENDERER_IDENTITY
            )
        ):
            raise RenderError("Gate 1 candidate bank is not qualified")
        required = (
            "bank_seed", "attempt", "center_frame", "real_semantic",
            "real_instance", "real_support_semantic",
            "control_support_semantic", "proxy_support_semantic",
            "control_support_frame", "proxy_support_frame",
            "control_template_index", "control_template_identity",
            "control_assigned_range_bin", "control_final_range_bin",
            "control_visible_returns", "control_observation_json",
            "control_world_json", "proxy_world_json",
            "control_record_json", "proxy_record_json", "error",
        )
        try:
            arrays = {name: np.asarray(source[name]) for name in required}
        except KeyError as error:
            raise RenderError(
                f"Gate 1 candidate bank is missing {error.args[0]}"
            ) from error
    if any(value.shape != (256,) for value in arrays.values()):
        raise RenderError("Gate 1 candidate bank arrays must each contain 256 units")
    if _scientific_array_hash(arrays) != metadata["scientific_array_hash"]:
        raise RenderError("Gate 1 candidate bank scientific arrays changed")
    if bool(np.any(arrays["error"] != "")):
        raise RenderError("Gate 1 candidate bank contains an incomplete unit")
    units: list[_Gate1BankUnit] = []
    for row in range(256):
        seed = int(arrays["bank_seed"][row])
        attempt = int(arrays["attempt"][row])
        center = int(arrays["center_frame"][row])
        real_semantic = int(arrays["real_semantic"][row])
        real_instance = int(arrays["real_instance"][row])
        real_support = int(arrays["real_support_semantic"][row])
        control_support = int(arrays["control_support_semantic"][row])
        proxy_support = int(arrays["proxy_support_semantic"][row])
        control_support_frame = int(arrays["control_support_frame"][row])
        proxy_support_frame = int(arrays["proxy_support_frame"][row])
        template_index = int(arrays["control_template_index"][row])
        template_identity = str(arrays["control_template_identity"][row])
        assigned_bin = int(arrays["control_assigned_range_bin"][row])
        final_bin = int(arrays["control_final_range_bin"][row])
        visible_returns = int(arrays["control_visible_returns"][row])
        try:
            observation = json.loads(str(arrays["control_observation_json"][row]))
            control_world = WorldSpec.from_dict(
                json.loads(str(arrays["control_world_json"][row]))
            )
            proxy_world = WorldSpec.from_dict(
                json.loads(str(arrays["proxy_world_json"][row]))
            )
            control_record = PlacementRecord.from_dict(
                json.loads(str(arrays["control_record_json"][row]))
            )
            proxy_record = PlacementRecord.from_dict(
                json.loads(str(arrays["proxy_record_json"][row]))
            )
        except (json.JSONDecodeError, TypeError, RenderError) as error:
            raise RenderError(
                f"Gate 1 candidate bank unit {row} cannot be decoded"
            ) from error
        attempt_seed = seed + 1_000_003 * attempt
        expected_template_index, expected_assigned_bin = (
            _gate1_control_template_assignment(attempt_seed)
        )
        expected_real: tuple[int, int, int, int] | None = None
        if _GATE1_REAL_CANDIDATES:
            real_index = int(np.random.default_rng(
                np.random.SeedSequence([attempt_seed, 3801])
            ).integers(0, len(_GATE1_REAL_CANDIDATES)))
            expected_real = _GATE1_REAL_CANDIDATES[real_index]
        expected_pool_prefix: np.ndarray | None = None
        if (
            _GATE1_CONTROL_CONTEXT is not None
            and len(_GATE1_TEMPLATES) == 256
            and 0 <= template_index < 256
            and 0 <= assigned_bin < 5
        ):
            frozen_rows = _gate1_control_rows(
                _GATE1_CONTROL_CONTEXT,
                template_index,
                int(_GATE1_TEMPLATES[template_index].raw_semantic_id),
                assigned_bin,
                center,
            )
            expected_pool_prefix = _GATE1_CONTROL_CONTEXT.support_pool.pool_indices[
                frozen_rows
            ]
        shared_prefix = min(
            len(control_record.proposal_pool_indices),
            len(proxy_record.proposal_pool_indices),
        )
        if (
            seed != 3_800_000 + row
            or not 0 <= attempt < 48
            or not 6 <= center <= 679
            or real_semantic not in {10, 18, 20, 30}
            or real_instance <= 0
            or real_support not in {40, 48}
            or control_support not in {40, 48}
            or proxy_support not in {40, 48}
            or abs(control_support_frame - center) > 2
            or abs(proxy_support_frame - center) > 2
            or not 0 <= template_index < 256
            or template_index != expected_template_index
            or len(template_identity) != 64
            or assigned_bin != template_index % 5
            or assigned_bin != expected_assigned_bin
            or final_bin != assigned_bin
            or visible_returns < 1
            or not isinstance(observation, Mapping)
            or int(observation.get("frame_id", -1)) != control_support_frame
            or int(observation.get("visible_returns", -1)) != visible_returns
            or int(observation.get("range_bin", -1)) != final_bin
            or control_world.seed != seed
            or proxy_world.seed != seed
            or control_world.source_sequence_id != 201
            or proxy_world.source_sequence_id != 201
            or len(control_world.objects) != 1
            or len(proxy_world.objects) != 1
            or control_world.objects[0].object_id != 1
            or proxy_world.objects[0].object_id != 1
            or control_world.objects[0].label != "normal-control"
            or proxy_world.objects[0].label != "anomaly-proxy"
            or not isinstance(control_world.objects[0].shape, NormalTemplateShape)
            or not isinstance(proxy_world.objects[0].shape, ShapeSpec)
            or control_record.object_id != 1
            or proxy_record.object_id != 1
            or control_record.label != "normal-control"
            or proxy_record.label != "anomaly-proxy"
            or control_record.template_identity != template_identity
            or control_record.template_seed != attempt_seed + 1
            or control_record.scale_seed != attempt_seed + 2
            or control_record.material_seed != attempt_seed + 11
            or control_record.yaw_seed != attempt_seed + 31
            or proxy_record.material_seed != attempt_seed + 12
            or proxy_record.yaw_seed != attempt_seed + 32
            or control_record.support_frame != control_support_frame
            or proxy_record.support_frame != proxy_support_frame
            or control_record.support_semantic != control_support
            or proxy_record.support_semantic != proxy_support
            or control_record.accepted_proposal + 1
            != len(control_record.proposal_pool_indices)
            or proxy_record.accepted_proposal + 1
            != len(proxy_record.proposal_pool_indices)
            or control_record.proposal_pool_indices[:shared_prefix]
            != proxy_record.proposal_pool_indices[:shared_prefix]
            or (
                expected_real is not None
                and (center, real_semantic, real_instance, real_support)
                != expected_real
            )
            or (
                expected_pool_prefix is not None
                and (
                    len(control_record.proposal_pool_indices)
                    > expected_pool_prefix.size
                    or len(proxy_record.proposal_pool_indices)
                    > expected_pool_prefix.size
                    or control_record.proposal_pool_indices
                    != tuple(map(
                        int,
                        expected_pool_prefix[
                            :len(control_record.proposal_pool_indices)
                        ],
                    ))
                    or proxy_record.proposal_pool_indices
                    != tuple(map(
                        int,
                        expected_pool_prefix[
                            :len(proxy_record.proposal_pool_indices)
                        ],
                    ))
                )
            )
            or (
                len(_GATE1_TEMPLATE_IDENTITIES) == 256
                and template_identity != _GATE1_TEMPLATE_IDENTITIES[template_index]
            )
        ):
            raise RenderError(f"Gate 1 candidate bank unit {row} changed identity")
        units.append(_Gate1BankUnit(
            seed, center, real_semantic, real_instance,
            real_support, control_support, proxy_support,
            template_index, assigned_bin, final_bin, visible_returns,
            control_world, proxy_world,
        ))
    return tuple(units)


def run_e39_qualification(
    e38_artifact_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Qualify per-range rates by reading the frozen E38-v2 shared trace."""

    source = Path(e38_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source) != FROZEN_E38_V2_ARTIFACT_SHA256:
        raise RenderError("E39-v2 E38-v2 shared trace identity changed")
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        trace = {
            name: np.asarray(payload[name])
            for name in payload.files if name != "metadata_json"
        }
    if (
        metadata.get("experiment") != "E38-v2"
        or metadata.get("passed") is not True
        or metadata.get("scientific_array_hash") != _scientific_array_hash(trace)
        or metadata.get("shared_trace_for_E39_E44") is not True
    ):
        raise RenderError("E39-v2 requires the qualified E38-v2 shared trace")
    started = time.monotonic()
    opportunity = trace["range_opportunity"].sum(axis=(0, 2))
    returned = trace["range_return_count"].sum(axis=(0, 2))
    rate = np.divide(returned, opportunity, out=np.zeros_like(returned, dtype=np.float64), where=opportunity > 0)
    conservation_errors = int(np.count_nonzero(
        trace["range_return_count"] > trace["range_opportunity"]
    ))
    observation_groups = np.count_nonzero(
        trace["range_return_count"] > 0, axis=(0, 2)
    )
    finite_errors = int(
        np.count_nonzero(~np.isfinite(rate))
        + np.count_nonzero(~np.isfinite(trace["visible_distance_m"]))
    )
    first_four_coverage_errors = int(np.count_nonzero(observation_groups[:, :4] == 0))
    passed = (
        conservation_errors == 0
        and finite_errors == 0
        and first_four_coverage_errors == 0
    )
    scientific = {
        **trace, "range_edges_m": _GATE1_RANGE_EDGES,
        "source_range_opportunity": opportunity,
        "source_range_return_count": returned,
        "source_range_return_rate": rate,
        "source_range_entity_frame_groups": observation_groups,
    }
    result = {
        "experiment": "E39-v2", "passed": passed, "bank_seeds": 256,
        "entity_frame_groups_per_source": 1280,
        "range_edges_m": _GATE1_RANGE_EDGES.tolist(),
        "source_range_opportunity": opportunity.tolist(),
        "source_range_return_count": returned.tolist(),
        "source_range_entity_frame_groups": observation_groups.tolist(),
        "conservation_errors": conservation_errors, "finite_errors": finite_errors,
        "first_four_coverage_errors": first_four_coverage_errors,
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e38_v2_sha256": FROZEN_E38_V2_ARTIFACT_SHA256,
        "run_seconds": [time.monotonic() - started],
        "scientific_array_hash": _scientific_array_hash(scientific),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _ecdf_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    values = np.sort(np.concatenate((left, right)))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_ecdf = np.searchsorted(left_sorted, values, side="right") / left_sorted.size
    right_ecdf = np.searchsorted(right_sorted, values, side="right") / right_sorted.size
    return float(np.max(np.abs(left_ecdf - right_ecdf)))


def _e40_statistics(
    source: np.ndarray, beam: np.ndarray, range_bin: np.ndarray, intensity: np.ndarray,
    sensor: SensorCalibration,
) -> dict[str, np.ndarray]:
    shape = (3, 128, 5)
    cell_key = (source.astype(np.int64) * 128 + beam) * 5 + range_bin
    order = np.argsort(cell_key, kind="stable")
    ordered_intensity = np.asarray(intensity[order], dtype=np.float64)
    flat_count = np.bincount(cell_key, minlength=3 * 128 * 5)
    count = flat_count.reshape(shape).astype(np.int64)
    offsets = np.concatenate((np.asarray((0,), dtype=np.int64), np.cumsum(flat_count)))
    quantiles = np.zeros(shape + (5,), dtype=np.float64)
    ecdf = np.zeros((3, 128, 5), dtype=np.float64)
    ecdf_valid = np.zeros((3, 128, 5), dtype=np.bool_)
    clipping = np.zeros((2, 128, 5, 2), dtype=np.int64)
    pairs = ((0, 1), (0, 2), (1, 2))
    probabilities = (0.05, 0.25, 0.5, 0.75, 0.95)
    for beam_id in range(128):
        for distance_id in range(5):
            values = []
            for source_id in range(3):
                key = (source_id * 128 + beam_id) * 5 + distance_id
                current = ordered_intensity[offsets[key]:offsets[key + 1]]
                values.append(current)
                if current.size:
                    quantiles[source_id, beam_id, distance_id] = np.quantile(
                        current, probabilities
                    )
                if source_id > 0:
                    clipping[source_id - 1, beam_id, distance_id, 0] = int(
                        np.count_nonzero(current <= sensor.intensity_min)
                    )
                    clipping[source_id - 1, beam_id, distance_id, 1] = int(
                        np.count_nonzero(current >= sensor.intensity_max)
                    )
            for pair_id, (left, right) in enumerate(pairs):
                if values[left].size and values[right].size:
                    ecdf_valid[pair_id, beam_id, distance_id] = True
                    ecdf[pair_id, beam_id, distance_id] = _ecdf_distance(
                        values[left], values[right]
                    )
    return {
        "cell_count": count, "conditional_quantiles": quantiles,
        "ecdf_distance": ecdf, "ecdf_valid": ecdf_valid,
        "generated_clipping_count": clipping,
    }


def run_e40_qualification(
    e39_artifact_path: Path | str, calibration_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Audit beam-by-range conditional intensity from the shared E39 trace."""
    source_path = Path(e39_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E39_V2_ARTIFACT_SHA256:
        raise RenderError("E40-v2 E39-v2 shared trace identity changed")
    with np.load(source_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        if metadata.get("experiment") != "E39-v2" or metadata.get("passed") is not True:
            raise RenderError("E40-v2 requires the passed formal E39-v2 shared trace")
        source = np.asarray(trace["intensity_source"])
        beam = np.asarray(trace["intensity_beam"])
        range_bin = np.asarray(trace["intensity_range_bin"])
        intensity = np.asarray(trace["intensity_value"])
        expected_range_count = np.asarray(trace["source_range_return_count"])
    calibration = Path(calibration_path).expanduser().resolve(strict=True)
    if _sha256_path(calibration) != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise RenderError("E40-v2 sensor calibration identity changed")
    _, sensor = load_sensor_calibration(calibration)
    started = time.monotonic()
    first = _e40_statistics(source, beam, range_bin, intensity, sensor)
    elapsed = time.monotonic() - started
    observed_range_count = first["cell_count"].sum(axis=1)
    count_errors = int(np.count_nonzero(observed_range_count != expected_range_count))
    identity_errors = int(
        np.count_nonzero((source < 0) | (source > 2))
        + np.count_nonzero((beam < 0) | (beam >= 128))
        + np.count_nonzero((range_bin < 0) | (range_bin >= 5))
    )
    finite_errors = int(np.count_nonzero(~np.isfinite(intensity)))
    generated = source > 0
    support_errors = int(np.count_nonzero(
        (intensity[generated] < sensor.intensity_min)
        | (intensity[generated] > sensor.intensity_max)
    ))
    generated_count = [int(np.count_nonzero(source == value)) for value in (1, 2)]
    clipping_count = first["generated_clipping_count"].sum(axis=(1, 2))
    clipping_fraction = np.divide(
        clipping_count, np.asarray(generated_count)[:, None],
        out=np.zeros((2, 2), dtype=np.float64),
        where=np.asarray(generated_count)[:, None] > 0,
    )
    passed = (
        count_errors == 0 and identity_errors == 0 and finite_errors == 0
        and support_errors == 0
    )
    scientific = {**first, "generated_clipping_fraction": clipping_fraction}
    result = {
        "experiment": "E40-v2", "passed": passed, "intensity_returns": int(intensity.size),
        "source_counts": [int(np.count_nonzero(source == value)) for value in range(3)],
        "nonempty_cells": [int(np.count_nonzero(first["cell_count"][value])) for value in range(3)],
        "count_errors": count_errors, "identity_errors": identity_errors,
        "finite_errors": finite_errors, "generated_support_errors": support_errors,
        "generated_clipping_count_low_high": clipping_count.tolist(),
        "generated_clipping_fraction_low_high": clipping_fraction.tolist(),
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e39_v2_sha256": FROZEN_E39_V2_ARTIFACT_SHA256,
        "run_seconds": [elapsed],
        "scientific_array_hash": _scientific_array_hash(scientific),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e41_statistics(
    empty_slots: np.ndarray, empty_geometry: np.ndarray,
    empty_accepted: np.ndarray, empty_final_new: np.ndarray,
) -> dict[str, np.ndarray]:
    """Audit the frozen empty-slot sensor chain without rerendering."""
    geometry_by_beam = empty_geometry.sum(axis=-1)
    totals = np.stack(
        (
            empty_slots.sum(axis=(0, 2, 3)),
            empty_geometry.sum(axis=(0, 2, 3, 4)),
            empty_accepted.sum(axis=(0, 2, 3, 4)),
            empty_final_new.sum(axis=(0, 2, 3, 4)),
        ),
        axis=1,
    ).astype(np.int64)
    violations = np.asarray(
        (
            np.count_nonzero(geometry_by_beam > empty_slots),
            np.count_nonzero(empty_accepted > empty_geometry),
            np.count_nonzero(empty_final_new > empty_accepted),
            np.count_nonzero(empty_slots < 0)
            + np.count_nonzero(empty_geometry < 0)
            + np.count_nonzero(empty_accepted < 0)
            + np.count_nonzero(empty_final_new < 0),
        ),
        dtype=np.int64,
    )
    return {
        "source_chain_total": totals,
        "source_return_rejection": totals[:, 1] - totals[:, 2],
        "source_post_acceptance_rejection": totals[:, 2] - totals[:, 3],
        "chain_violation_count": violations,
        "beam_range_geometry": empty_geometry.sum(axis=(0, 2)),
        "beam_range_accepted": empty_accepted.sum(axis=(0, 2)),
        "beam_range_final_new": empty_final_new.sum(axis=(0, 2)),
        "beam_empty_opportunity": empty_slots.sum(axis=(0, 2)),
    }


def run_e41_qualification(
    e39_artifact_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Qualify empty-to-valid accounting from the passed shared E39 trace."""
    source_path = Path(e39_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E39_V2_ARTIFACT_SHA256:
        raise RenderError("E41-v2 E39-v2 shared trace identity changed")
    with np.load(source_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        if metadata.get("experiment") != "E39-v2" or metadata.get("passed") is not True:
            raise RenderError("E41-v2 requires the passed formal E39-v2 shared trace")
        arrays = tuple(
            np.asarray(trace[name])
            for name in (
                "empty_slots", "empty_geometry", "empty_accepted", "empty_final_new"
            )
        )
    started = time.monotonic()
    first = _e41_statistics(*arrays)
    elapsed = time.monotonic() - started
    totals = first["source_chain_total"]
    return_rejection = first["source_return_rejection"]
    branch_coverage_errors = int(
        np.count_nonzero(totals[:, 3] == 0) + np.count_nonzero(return_rejection == 0)
    )
    chain_errors = int(first["chain_violation_count"].sum())
    passed = chain_errors == 0 and branch_coverage_errors == 0
    result = {
        "experiment": "E41-v2", "passed": passed,
        "source_order": ["normal-control", "anomaly-proxy"],
        "chain_order": ["empty opportunity", "geometry hit", "return accepted", "final new"],
        "source_chain_total": totals.tolist(),
        "source_return_rejection": return_rejection.tolist(),
        "source_post_acceptance_rejection": first["source_post_acceptance_rejection"].tolist(),
        "chain_violation_count": first["chain_violation_count"].tolist(),
        "chain_errors": chain_errors,
        "branch_coverage_errors": branch_coverage_errors,
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e39_v2_sha256": FROZEN_E39_V2_ARTIFACT_SHA256,
        "run_seconds": [elapsed],
        "scientific_array_hash": _scientific_array_hash(first),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e42_statistics(
    support_semantic: np.ndarray, geometry_hits: np.ndarray,
    accepted_hits: np.ndarray, visible_returns: np.ndarray,
    visible_distance_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign the frozen positive-Nvis strata and audit their accounting."""
    layer = np.full(visible_returns.shape, -1, dtype=np.int8)
    layer[(visible_returns >= 1) & (visible_returns < 8)] = 0
    layer[(visible_returns >= 8) & (visible_returns < 32)] = 1
    layer[(visible_returns >= 32) & (visible_returns < 128)] = 2
    layer[visible_returns >= 128] = 3
    range_bin = _gate1_range_bin(visible_distance_m)
    layer_count = np.zeros((3, 4), dtype=np.int64)
    zero_count = np.zeros(3, dtype=np.int64)
    for source_id in range(3):
        zero_count[source_id] = np.count_nonzero(layer[:, source_id] < 0)
        for layer_id in range(4):
            layer_count[source_id, layer_id] = np.count_nonzero(
                layer[:, source_id] == layer_id
            )
    shared_count = np.zeros((2, 5, 4, 3), dtype=np.int64)
    for support_id, semantic in enumerate((40, 48)):
        for range_id in range(5):
            for layer_id in range(4):
                for source_id in range(3):
                    shared_count[support_id, range_id, layer_id, source_id] = np.count_nonzero(
                        (support_semantic[:, source_id] == semantic)
                        & (range_bin[:, source_id] == range_id)
                        & (layer[:, source_id] == layer_id)
                    )
    return {
        "geometry_hits": geometry_hits,
        "accepted_hits": accepted_hits,
        "visible_returns": visible_returns,
        "visible_distance_m": visible_distance_m,
        "range_bin": range_bin,
        "nvis_layer": layer,
        "source_layer_count": layer_count,
        "source_zero_visible_count": zero_count,
        "shared_stratum_source_count": shared_count,
        "shared_stratum_valid": np.all(shared_count > 0, axis=-1),
    }


def run_e42_qualification(
    e39_artifact_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Qualify entity-frame Nvis strata and preliminary matching support."""
    source_path = Path(e39_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E39_V2_ARTIFACT_SHA256:
        raise RenderError("E42-v2 E39-v2 shared trace identity changed")
    with np.load(source_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        if metadata.get("experiment") != "E39-v2" or metadata.get("passed") is not True:
            raise RenderError("E42-v2 requires the passed formal E39-v2 shared trace")
        support = np.asarray(trace["support_semantic"])
        geometry = np.asarray(trace["geometry_hits"])
        accepted = np.asarray(trace["accepted_hits"])
        visible = np.asarray(trace["visible_returns"])
        distance = np.asarray(trace["visible_distance_m"])
    started = time.monotonic()
    first = _e42_statistics(support, geometry, accepted, visible, distance)
    elapsed = time.monotonic() - started
    definition_errors = int(
        np.count_nonzero(geometry < 0)
        + np.count_nonzero(accepted < 0)
        + np.count_nonzero(visible < 0)
        + np.count_nonzero(accepted > geometry)
        + np.count_nonzero(visible > accepted)
        + np.count_nonzero(~np.isfinite(distance))
        + np.count_nonzero((first["range_bin"] < 0) | (first["range_bin"] >= 5))
    )
    group_count = visible.shape[0] * visible.shape[2]
    count_errors = int(np.count_nonzero(
        first["source_layer_count"].sum(axis=1)
        + first["source_zero_visible_count"] != group_count
    ))
    generated_layer_coverage = np.count_nonzero(
        first["source_layer_count"][1:] > 0, axis=1
    )
    coverage_errors = int(np.count_nonzero(generated_layer_coverage < 3))
    shared_strata = int(np.count_nonzero(first["shared_stratum_valid"]))
    matching_errors = int(shared_strata == 0)
    passed = (
        definition_errors == 0 and count_errors == 0 and coverage_errors == 0
        and matching_errors == 0
    )
    result = {
        "experiment": "E42-v2", "passed": passed,
        "nvis_layers": [[1, 8], [8, 32], [32, 128], [128, None]],
        "entity_frame_groups_per_source": group_count,
        "source_layer_count": first["source_layer_count"].tolist(),
        "source_zero_visible_count": first["source_zero_visible_count"].tolist(),
        "generated_layer_coverage": generated_layer_coverage.tolist(),
        "shared_support_range_nvis_strata": shared_strata,
        "definition_errors": definition_errors, "count_errors": count_errors,
        "coverage_errors": coverage_errors, "matching_errors": matching_errors,
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e39_v2_sha256": FROZEN_E39_V2_ARTIFACT_SHA256,
        "run_seconds": [elapsed],
        "scientific_array_hash": _scientific_array_hash(first),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e43_statistics(visible_returns: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize genuine adjacent-frame visibility changes for fixed entities."""
    change = np.diff(visible_returns.astype(np.int64), axis=2)
    previous = visible_returns[:, :, :-1].astype(np.float64)
    relative = np.abs(change) / np.maximum(previous, 1.0)
    visible_frames = np.count_nonzero(visible_returns > 0, axis=2).astype(np.int8)
    v_count = np.zeros((3, 6), dtype=np.int64)
    quantiles = np.zeros((3, 5), dtype=np.float64)
    transitions = np.zeros((3, 2), dtype=np.int64)
    for source_id in range(3):
        v_count[source_id] = np.bincount(
            visible_frames[:, source_id], minlength=6
        )[:6]
        quantiles[source_id] = np.quantile(
            relative[:, source_id].ravel(), (0.05, 0.25, 0.5, 0.75, 0.95)
        )
        transitions[source_id, 0] = np.count_nonzero(
            (visible_returns[:, source_id, :-1] == 0)
            & (visible_returns[:, source_id, 1:] > 0)
        )
        transitions[source_id, 1] = np.count_nonzero(
            (visible_returns[:, source_id, :-1] > 0)
            & (visible_returns[:, source_id, 1:] == 0)
        )
    return {
        "visible_returns": visible_returns,
        "adjacent_nvis_change": change,
        "adjacent_nvis_relative_change": relative,
        "visible_frame_count_V": visible_frames,
        "source_V_count": v_count,
        "source_relative_change_quantiles": quantiles,
        "source_appearance_disappearance": transitions,
    }


def run_e43_qualification(
    e37_artifact_path: Path | str, e39_artifact_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Qualify deterministic five-frame visibility and finite temporal changes."""
    e37_path = Path(e37_artifact_path).expanduser().resolve(strict=True)
    e39_path = Path(e39_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(e37_path) != FROZEN_E37_ARTIFACT_SHA256:
        raise RenderError("E43-v2 E37 window-audit identity changed")
    if _sha256_path(e39_path) != FROZEN_E39_V2_ARTIFACT_SHA256:
        raise RenderError("E43-v2 E39-v2 shared trace identity changed")
    with np.load(e37_path, allow_pickle=False) as source:
        e37 = json.loads(str(source["metadata_json"]))
    with np.load(e39_path, allow_pickle=False) as trace:
        e39 = json.loads(str(trace["metadata_json"]))
        visible = np.asarray(trace["visible_returns"])
    if e37.get("experiment") != "E37" or e37.get("passed") is not True:
        raise RenderError("E43 requires the passed formal E37 window audit")
    if e39.get("experiment") != "E39-v2" or e39.get("passed") is not True:
        raise RenderError("E43-v2 requires the passed formal E39-v2 shared trace")
    started = time.monotonic()
    first = _e43_statistics(visible)
    elapsed = time.monotonic() - started
    field_errors = sum(int(value) for value in e37["field_digest_errors"].values())
    window_identity_errors = int(
        e37["duplicate_request_bit_errors"] + e37["identity_errors"]
        + e37["render_call_errors"] + e37["cross_world_cache_errors"]
        + e37["render_frame_window_parameters"] + e37["slot_uniform_window_reads"]
        + field_errors
    )
    repeated_render_errors = int(e37["duplicate_request_bit_errors"])
    finite_errors = int(
        np.count_nonzero(~np.isfinite(first["adjacent_nvis_relative_change"]))
        + np.count_nonzero(~np.isfinite(first["source_relative_change_quantiles"]))
    )
    definition_errors = int(
        np.count_nonzero(visible < 0)
        + np.count_nonzero(
            (first["visible_frame_count_V"] < 0)
            | (first["visible_frame_count_V"] > 5)
        )
        + np.count_nonzero(first["source_V_count"].sum(axis=1) != visible.shape[0])
    )
    passed = (
        window_identity_errors == 0 and repeated_render_errors == 0
        and finite_errors == 0 and definition_errors == 0
    )
    result = {
        "experiment": "E43-v2", "passed": passed,
        "relative_change_definition": "abs(N_t-N_tminus1)/max(N_tminus1,1)",
        "source_V_count_for_V_0_to_5": first["source_V_count"].tolist(),
        "source_relative_change_quantiles_Q05_Q25_Q50_Q75_Q95": (
            first["source_relative_change_quantiles"].tolist()
        ),
        "source_appearance_disappearance": first["source_appearance_disappearance"].tolist(),
        "window_identity_errors": window_identity_errors,
        "repeated_render_errors": repeated_render_errors,
        "finite_errors": finite_errors, "definition_errors": definition_errors,
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e37_sha256": FROZEN_E37_ARTIFACT_SHA256,
        "input_e39_v2_sha256": FROZEN_E39_V2_ARTIFACT_SHA256,
        "run_seconds": [elapsed],
        "scientific_array_hash": _scientific_array_hash(first),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e44_statistics(
    support_semantic: np.ndarray, accepted_hits: np.ndarray,
    visible_returns: np.ndarray, visible_distance_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute frozen occlusion strata only where the denominator exists."""
    valid = accepted_hits > 0
    occlusion = np.zeros(accepted_hits.shape, dtype=np.float64)
    occlusion[valid] = 1.0 - visible_returns[valid] / accepted_hits[valid]
    layer = np.full(accepted_hits.shape, -1, dtype=np.int8)
    layer[valid & (occlusion >= 0.0) & (occlusion < 0.25)] = 0
    layer[valid & (occlusion >= 0.25) & (occlusion < 0.75)] = 1
    layer[valid & (occlusion >= 0.75) & (occlusion <= 1.0)] = 2
    range_bin = _gate1_range_bin(visible_distance_m)
    layer_count = np.zeros((3, 3), dtype=np.int64)
    for source_id in range(3):
        for layer_id in range(3):
            layer_count[source_id, layer_id] = np.count_nonzero(
                layer[:, source_id] == layer_id
            )
    shared_count = np.zeros((2, 5, 3, 3), dtype=np.int64)
    for support_id, semantic in enumerate((40, 48)):
        for range_id in range(5):
            for layer_id in range(3):
                for source_id in range(3):
                    shared_count[support_id, range_id, layer_id, source_id] = np.count_nonzero(
                        (support_semantic[:, source_id] == semantic)
                        & (range_bin[:, source_id] == range_id)
                        & (layer[:, source_id] == layer_id)
                    )
    return {
        "accepted_hits": accepted_hits,
        "visible_returns": visible_returns,
        "visible_distance_m": visible_distance_m,
        "occlusion_valid": valid,
        "occlusion_rate": occlusion,
        "occlusion_layer": layer,
        "source_layer_count": layer_count,
        "source_undefined_count": np.count_nonzero(~valid, axis=(0, 2)),
        "shared_stratum_source_count": shared_count,
        "shared_stratum_valid": np.all(shared_count > 0, axis=-1),
    }


def run_e44_qualification(
    e39_artifact_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Qualify frozen occlusion rates and preliminary matching support."""
    source_path = Path(e39_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E39_V2_ARTIFACT_SHA256:
        raise RenderError("E44-v2 E39-v2 shared trace identity changed")
    with np.load(source_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        if metadata.get("experiment") != "E39-v2" or metadata.get("passed") is not True:
            raise RenderError("E44-v2 requires the passed formal E39-v2 shared trace")
        support = np.asarray(trace["support_semantic"])
        accepted = np.asarray(trace["accepted_hits"])
        visible = np.asarray(trace["visible_returns"])
        distance = np.asarray(trace["visible_distance_m"])
    started = time.monotonic()
    first = _e44_statistics(support, accepted, visible, distance)
    elapsed = time.monotonic() - started
    valid = first["occlusion_valid"]
    definition_errors = int(
        np.count_nonzero(accepted < 0)
        + np.count_nonzero(visible < 0)
        + np.count_nonzero(visible > accepted)
        + np.count_nonzero(~np.isfinite(first["occlusion_rate"][valid]))
        + np.count_nonzero(
            (first["occlusion_rate"][valid] < 0.0)
            | (first["occlusion_rate"][valid] > 1.0)
        )
        + np.count_nonzero(first["occlusion_layer"][valid] < 0)
    )
    valid_count = np.count_nonzero(valid, axis=(0, 2))
    count_errors = int(np.count_nonzero(
        first["source_layer_count"].sum(axis=1) != valid_count
    ))
    generated_layer_coverage = np.count_nonzero(
        first["source_layer_count"][1:] > 0, axis=1
    )
    coverage_errors = int(np.count_nonzero(generated_layer_coverage < 3))
    shared_strata = int(np.count_nonzero(first["shared_stratum_valid"]))
    matching_errors = int(shared_strata == 0)
    passed = (
        definition_errors == 0 and count_errors == 0 and coverage_errors == 0
        and matching_errors == 0
    )
    result = {
        "experiment": "E44-v2", "passed": passed,
        "occlusion_layers": [[0.0, 0.25], [0.25, 0.75], [0.75, 1.0]],
        "source_valid_count": valid_count.tolist(),
        "source_undefined_count": first["source_undefined_count"].tolist(),
        "source_layer_count": first["source_layer_count"].tolist(),
        "generated_layer_coverage": generated_layer_coverage.tolist(),
        "shared_support_range_occlusion_strata": shared_strata,
        "definition_errors": definition_errors, "count_errors": count_errors,
        "coverage_errors": coverage_errors, "matching_errors": matching_errors,
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "input_e39_v2_sha256": FROZEN_E39_V2_ARTIFACT_SHA256,
        "run_seconds": [elapsed],
        "scientific_array_hash": _scientific_array_hash(first),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **first,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _e45_real_candidate_capacity(
    sequence: object, pool: QualifiedSupportPool,
) -> dict[str, np.ndarray]:
    """Enumerate the complete frozen real-normal candidate universe."""
    candidates = _gate1_real_candidates(sequence, pool)
    rows: list[tuple[int, int, int, int, int, int, float]] = []
    for candidate_id, (center, semantic, instance, support) in enumerate(candidates):
        for frame_id in range(center - 2, center + 3):
            frame = sequence.source_frame(frame_id)
            assert frame.labels is not None
            selected = (
                (frame.labels.semantic == np.uint16(semantic))
                & (frame.labels.instance == np.uint16(instance))
                & ~np.asarray(frame.zero_slot_mask, dtype=np.bool_)
            )
            ranges = np.linalg.norm(
                np.asarray(frame.xyzi[selected, :3], dtype=np.float64), axis=1
            )
            ranges = ranges[(ranges >= 2.5) & (ranges <= 50.0)]
            if ranges.size < 16:
                raise RenderError("frozen Gate 1 real candidate lost persistent coverage")
            rows.append(
                (
                    candidate_id, center, frame_id, semantic, instance, support,
                    float(np.median(ranges)),
                )
            )
    integer = np.asarray([row[:6] for row in rows], dtype=np.int64)
    distance = np.asarray([row[6] for row in rows], dtype=np.float64)
    return {
        "candidate_id": integer[:, 0], "center_frame": integer[:, 1],
        "frame_id": integer[:, 2], "real_semantic": integer[:, 3],
        "real_instance": integer[:, 4], "support_semantic": integer[:, 5],
        "median_visible_distance_m": distance,
        "range_bin": _gate1_range_bin(distance),
    }


def run_e45_qualification(
    data_root: Path | str, support_pool_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Apply necessary candidate-domain checks before frozen triplet matching."""
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    pool, pool_metadata = load_gate1_support_pool(support_pool_path)
    started = time.monotonic()
    universe = _e45_real_candidate_capacity(sequence, pool)
    valid_range = (universe["range_bin"] >= 0) & (universe["range_bin"] < 5)
    range_count_runs = [
        np.bincount(universe["range_bin"][valid_range], minlength=5)[:5].astype(np.int64)
        for _ in range(2)
    ]
    reproduced = np.array_equal(range_count_runs[0], range_count_runs[1])
    range_count = range_count_runs[0]
    identity_errors = int(
        np.count_nonzero((universe["range_bin"] < 0) | (universe["range_bin"] >= 5))
        + np.count_nonzero(~np.isfinite(universe["median_visible_distance_m"]))
        + np.count_nonzero(~np.isin(universe["support_semantic"], (40, 48)))
    )
    real_candidates = int(np.unique(universe["candidate_id"]).size)
    candidate_frame_errors = int(universe["frame_id"].size != real_candidates * 5)
    required_range_triplets = np.asarray((128, 128, 128, 128, 32), dtype=np.int64)
    upper_bound = range_count.copy()
    coverage_shortfall = np.maximum(required_range_triplets - upper_bound, 0)
    necessary_coverage_errors = int(np.count_nonzero(coverage_shortfall))
    far_range_impossible = bool(upper_bound[4] < required_range_triplets[4])
    if necessary_coverage_errors == 0:
        raise RenderError("E45 necessary coverage passed; full triplet matching must run")
    passed = False
    failure_classification = "scientific_candidate_domain_failure"
    result = {
        "experiment": "E45", "passed": passed,
        "failure_classification": failure_classification,
        "support_pool_size": int(pool_metadata["pool_size"]),
        "complete_real_candidate_entities": real_candidates,
        "complete_real_candidate_entity_frames": int(universe["frame_id"].size),
        "real_candidate_range_count": range_count.tolist(),
        "required_triplets_by_range": required_range_triplets.tolist(),
        "necessary_range_shortfall": coverage_shortfall.tolist(),
        "maximum_possible_triplets_by_range": upper_bound.tolist(),
        "capacity_ladder": [256, 512, 1024, 2048],
        "capacity_expansion_executed": False,
        "capacity_expansion_short_circuit": (
            "the complete frozen real-normal universe has zero 40--50 m units; "
            "every capacity-ladder bank is a subset sampled from this universe"
            if far_range_impossible else None
        ),
        "triplet_matching_executed": False,
        "triplet_matching_skip_reason": (
            "a necessary frozen range-coverage condition is impossible"
            if necessary_coverage_errors else None
        ),
        "identity_errors": identity_errors,
        "candidate_frame_errors": candidate_frame_errors,
        "necessary_coverage_errors": necessary_coverage_errors,
        "elementwise_reproduced": reproduced,
        "elapsed_seconds": time.monotonic() - started,
        "scientific_array_hash": _scientific_array_hash(universe),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **universe,
        required_triplets_by_range=required_range_triplets,
        maximum_possible_triplets_by_range=upper_bound,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E45_BANK: tuple[
    tuple[int, int, int, int, int, WorldSpec, WorldSpec] | _Gate1BankUnit, ...
] = ()


def _point_identity_order(frame_id: int, slots: np.ndarray, grid: RayGrid) -> np.ndarray:
    """Hash canonical frame/beam/column identities for fixed point subsampling."""
    beam = grid.beam_ids[slots].astype(np.uint64)
    column = grid.column_ids[slots].astype(np.uint64)
    # SplitMix64 intentionally wraps modulo 2^64.
    with np.errstate(over="ignore"):
        value = (
            np.uint64(frame_id) * np.uint64(0x9E3779B185EBCA87)
            ^ beam * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ column * np.uint64(0x165667B19E3779F9)
        )
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return np.argsort(value, kind="stable")


def _e45_unit_hash(
    source_id: int, bank_seed: int, frame_id: int,
    real_semantic: int, real_instance: int,
) -> np.uint64:
    identity = (
        f"real:{frame_id}:{real_semantic}:{real_instance}"
        if source_id == 0 else f"generated:{source_id}:{bank_seed}:{frame_id}"
    )
    return np.uint64(int.from_bytes(hashlib.sha256(identity.encode("ascii")).digest()[:8], "little"))


def _e45_unit_record(
    frame: SourceFrame, grid: RayGrid, source_id: int, bank_seed: int,
    center_frame: int, real_semantic: int, real_instance: int,
    support_semantic: int, geometry: np.ndarray, returned: np.ndarray,
    rendered_frame: SourceFrame,
) -> dict[str, np.ndarray]:
    slots = np.flatnonzero(returned)
    features = np.zeros((64, 7), dtype=np.float64)
    point_count = min(int(slots.size), 64)
    median_distance = 0.0
    median_beam = 0.0
    local_density = 0.0
    azimuth_sector = -1
    if slots.size:
        all_features = np.asarray(
            low_level_return_features(rendered_frame, grid, returned), dtype=np.float64
        )
        chosen = _point_identity_order(int(frame.frame_id), slots, grid)[:64]
        features[:point_count] = all_features[chosen]
        median_distance = float(np.median(grid.official_ranges(rendered_frame)[slots]))
        median_beam = float(np.median(grid.beam_ids[slots]))
        local_density = float(np.median(all_features[:, 6]))
        angle = np.arctan2(all_features[:, 1], all_features[:, 0])
        circular = math.atan2(float(np.sin(angle).sum()), float(np.cos(angle).sum()))
        azimuth_sector = int(math.floor((circular % (2.0 * math.pi)) / (math.pi / 4.0))) % 8
    geometry_count = int(np.count_nonzero(geometry))
    occlusion = (
        float(1.0 - slots.size / geometry_count) if geometry_count else 0.0
    )
    return {
        "bank_seed": np.asarray(bank_seed, dtype=np.int64),
        "source": np.asarray(source_id, dtype=np.uint8),
        "center_frame": np.asarray(center_frame, dtype=np.int16),
        "frame_id": np.asarray(frame.frame_id, dtype=np.int16),
        "support_semantic": np.asarray(support_semantic, dtype=np.uint16),
        "range_bin": np.asarray(
            int(_gate1_range_bin(np.asarray([median_distance]))[0]) if slots.size else -1,
            dtype=np.int8,
        ),
        "azimuth_sector": np.asarray(azimuth_sector, dtype=np.int8),
        "median_distance_m": np.asarray(median_distance, dtype=np.float64),
        "median_beam": np.asarray(median_beam, dtype=np.float64),
        "Nvis": np.asarray(slots.size, dtype=np.int32),
        "O_hat": np.asarray(occlusion, dtype=np.float64),
        "local_density": np.asarray(local_density, dtype=np.float64),
        "geometry_hits": np.asarray(geometry_count, dtype=np.int32),
        "point_count": np.asarray(point_count, dtype=np.int16),
        "point_features": features,
        "unit_hash": np.asarray(
            _e45_unit_hash(source_id, bank_seed, int(frame.frame_id), real_semantic, real_instance),
            dtype=np.uint64,
        ),
    }


def _e45_worker(index: int) -> dict[str, np.ndarray]:
    sequence, grid, sensor = _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR
    if sequence is None or grid is None or sensor is None or index >= len(_E45_BANK):
        raise RuntimeError("E45 candidate fixtures are not initialized")
    unit = _E45_BANK[index]
    if isinstance(unit, _Gate1BankUnit):
        bank_seed = unit.bank_seed
        center = unit.center_frame
        real_semantic = unit.real_semantic
        real_instance = unit.real_instance
        source_support = (
            unit.real_support_semantic,
            unit.control_support_semantic,
            unit.proxy_support_semantic,
        )
        control_world = unit.control_world
        proxy_world = unit.proxy_world
    else:
        (
            bank_seed, center, real_semantic, real_instance, support,
            control_world, proxy_world,
        ) = unit
        source_support = (support, support, support)
    records: list[dict[str, np.ndarray]] = []
    for frame_id in range(center - 2, center + 3):
        frame = sequence.source_frame(frame_id)
        real_geometry, real_return, _ = _gate1_real_geometry(
            frame, real_semantic, real_instance, grid
        )
        records.append(_e45_unit_record(
            frame, grid, 0, bank_seed, center, real_semantic, real_instance,
            source_support[0], real_geometry, real_return, frame,
        ))
        for source_id, world in enumerate((control_world, proxy_world), start=1):
            geometry, _, _, rendered = _gate1_single_object_trace(frame, world, grid, sensor)
            returned = np.asarray(
                rendered.normal_control_mask if source_id == 1 else rendered.anomaly_proxy_mask,
                dtype=np.bool_,
            )
            official = np.asarray(grid.official_ranges(rendered.source))
            returned = returned & (official >= 2.5) & (official <= 50.0)
            records.append(_e45_unit_record(
                frame, grid, source_id, bank_seed, center, real_semantic,
                real_instance, source_support[source_id], geometry, returned,
                rendered.source,
            ))
    return {
        name: np.stack([record[name] for record in records])
        for name in records[0]
    }


def _write_e45_units(
    arrays: Mapping[str, np.ndarray], path: Path, bank_hash: str,
    extraction_seconds: float,
) -> None:
    metadata = {
        "experiment": "E45-v2-units", "passed": True,
        "capacity": int(arrays["bank_seed"].shape[0]), "bank_hash": bank_hash,
        "extraction_seconds": extraction_seconds,
        "scientific_array_hash": _scientific_array_hash(arrays),
    }
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, path)


def _e45_covariates(units: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((
        units["median_distance_m"], units["median_beam"],
        np.log1p(units["Nvis"]), units["O_hat"],
        np.log1p(units["local_density"]),
    ))


def _e45_smd(values: np.ndarray) -> np.ndarray:
    output = np.zeros((3, 5), dtype=np.float64)
    for pair_id, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        left_values = values[:, left]
        right_values = values[:, right]
        pooled = np.sqrt((left_values.var(axis=0, ddof=1) + right_values.var(axis=0, ddof=1)) / 2.0)
        difference = np.abs(left_values.mean(axis=0) - right_values.mean(axis=0))
        output[pair_id] = np.divide(
            difference, pooled, out=np.where(difference == 0.0, 0.0, np.inf),
            where=pooled > 0.0,
        )
    return output


def _e45_match(units: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    flat = {name: np.asarray(value).reshape((-1,) + value.shape[2:]) for name, value in units.items()}
    covariates = _e45_covariates(flat)
    valid = (
        (flat["point_count"] > 0) & (flat["range_bin"] >= 0)
        & (flat["range_bin"] < 4) & (flat["azimuth_sector"] >= 0)
        & np.isfinite(covariates).all(axis=1)
    )
    by_source: list[np.ndarray] = []
    for source_id in range(3):
        candidates = np.flatnonzero(valid & (flat["source"] == source_id))
        order = np.argsort(flat["unit_hash"][candidates], kind="stable")
        candidates = candidates[order]
        _, first = np.unique(flat["unit_hash"][candidates], return_index=True)
        by_source.append(candidates[np.sort(first)])
    caliper = np.asarray((2.0, 4.0, 0.25, 0.10, 0.25), dtype=np.float64)
    matched: list[tuple[int, int, int]] = []
    for support in (40, 48):
        for range_bin in range(4):
            for sector in range(8):
                groups = [
                    source[
                        (flat["support_semantic"][source] == support)
                        & (flat["range_bin"][source] == range_bin)
                        & (flat["azimuth_sector"][source] == sector)
                    ]
                    for source in by_source
                ]
                if any(group.size == 0 for group in groups):
                    continue
                used_control: set[int] = set()
                used_proxy: set[int] = set()
                real_order: list[tuple[int, int]] = []
                for real in groups[0]:
                    control_count = int(np.count_nonzero(
                        np.all(np.abs(covariates[groups[1]] - covariates[real]) <= caliper, axis=1)
                    ))
                    proxy_count = int(np.count_nonzero(
                        np.all(np.abs(covariates[groups[2]] - covariates[real]) <= caliper, axis=1)
                    ))
                    real_order.append((control_count * proxy_count, int(real)))
                real_order.sort(key=lambda item: (item[0], int(flat["unit_hash"][item[1]])))
                for possible_count, real in real_order:
                    if possible_count == 0:
                        continue
                    control = np.asarray(
                        [item for item in groups[1] if int(item) not in used_control],
                        dtype=np.int64,
                    )
                    proxy = np.asarray(
                        [item for item in groups[2] if int(item) not in used_proxy],
                        dtype=np.int64,
                    )
                    control = control[np.all(np.abs(covariates[control] - covariates[real]) <= caliper, axis=1)]
                    proxy = proxy[np.all(np.abs(covariates[proxy] - covariates[real]) <= caliper, axis=1)]
                    if control.size == 0 or proxy.size == 0:
                        continue
                    best: tuple[float, int, int, int, int] | None = None
                    for control_id in control:
                        compatible = proxy[np.all(
                            np.abs(covariates[proxy] - covariates[control_id]) <= caliper,
                            axis=1,
                        )]
                        if compatible.size == 0:
                            continue
                        score = (
                            np.square((covariates[compatible] - covariates[real]) / caliper).sum(axis=1)
                            + np.square((covariates[compatible] - covariates[control_id]) / caliper).sum(axis=1)
                            + float(np.square((covariates[control_id] - covariates[real]) / caliper).sum())
                        )
                        proxy_id = int(compatible[np.lexsort((
                            flat["unit_hash"][compatible], score,
                        ))[0]])
                        candidate = (
                            float(np.min(score)), int(flat["unit_hash"][control_id]),
                            int(flat["unit_hash"][proxy_id]), int(control_id), proxy_id,
                        )
                        if best is None or candidate < best:
                            best = candidate
                    if best is None:
                        continue
                    control_id, proxy_id = best[3], best[4]
                    matched.append((real, control_id, proxy_id))
                    used_control.add(control_id)
                    used_proxy.add(proxy_id)
    matched_index = np.asarray(matched, dtype=np.int64).reshape(-1, 3)
    matched_covariates = covariates[matched_index] if matched else np.empty((0, 3, 5))
    smd = _e45_smd(matched_covariates) if len(matched) > 1 else np.full((3, 5), np.inf)
    range_count = (
        np.bincount(flat["range_bin"][matched_index[:, 0]], minlength=4)[:4]
        if matched else np.zeros(4, dtype=np.int64)
    )
    caliper_errors = 0
    if matched:
        for left, right in ((0, 1), (0, 2), (1, 2)):
            caliper_errors += int(np.count_nonzero(
                np.abs(matched_covariates[:, left] - matched_covariates[:, right]) > caliper
            ))
    duplicate_errors = int(sum(
        len(matched) - np.unique(flat["unit_hash"][matched_index[:, source]]).size
        for source in range(3)
    )) if matched else 0
    return {
        "matched_flat_index": matched_index,
        "matched_covariates": matched_covariates,
        "pairwise_smd": smd,
        "matched_range_count": range_count.astype(np.int64),
        "matched_center_frames": np.unique(
            flat["center_frame"][matched_index[:, 0]]
        ).astype(np.int16) if matched else np.empty(0, dtype=np.int16),
        "caliper_errors": np.asarray(caliper_errors, dtype=np.int64),
        "duplicate_errors": np.asarray(duplicate_errors, dtype=np.int64),
    }


def _load_e25_templates(path: Path | str) -> tuple[NormalTemplateShape, ...]:
    with np.load(Path(path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E25" or metadata.get("passed") is not True:
            raise RenderError("Gate 1 requires the passed E25 template artifact")
        unique: dict[str, NormalTemplateShape] = {}
        for identity, payload in zip(source["template_identity"], source["object_json"], strict=True):
            key = identity.decode()
            if key and key not in unique:
                item = ObjectSpec.from_dict(json.loads(payload.decode()))
                if not isinstance(item.shape, NormalTemplateShape):
                    raise RenderError("E25 artifact contains a non-template control")
                unique[key] = item.shape
    return tuple(unique[key] for key in sorted(unique))


def _e45_v1_real_candidates(path: Path | str) -> tuple[tuple[int, int, int, int], ...]:
    with np.load(Path(path).expanduser().resolve(strict=True), allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if metadata.get("experiment") != "E45":
            raise RenderError("E45-v2 requires the formal E45-v1 candidate-universe artifact")
        candidate = np.asarray(source["candidate_id"])
        center = np.asarray(source["center_frame"])
        semantic = np.asarray(source["real_semantic"])
        instance = np.asarray(source["real_instance"])
        support = np.asarray(source["support_semantic"])
    rows: list[tuple[int, int, int, int]] = []
    for identifier in np.unique(candidate):
        selected = np.flatnonzero(candidate == identifier)
        if selected.size != 5:
            raise RenderError("E45-v1 candidate universe does not contain five frames per entity")
        first = int(selected[0])
        rows.append((
            int(center[first]), int(semantic[first]), int(instance[first]), int(support[first])
        ))
    return tuple(rows)


def _load_e45_bank_with_metadata(
    path: Path | str,
) -> tuple[tuple[tuple[int, int, int, int, int, WorldSpec, WorldSpec], ...], dict[str, object]]:
    source_path = Path(path).expanduser().resolve(strict=True)
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
    return _load_gate1_bank(source_path), metadata


def _load_e45_unit_cache(
    path: Path, capacity: int, bank_hash: str,
) -> tuple[dict[str, np.ndarray], float] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if (
            metadata.get("experiment") != "E45-v2-units"
            or metadata.get("passed") is not True
            or int(metadata.get("capacity", -1)) != capacity
            or metadata.get("bank_hash") != bank_hash
        ):
            return None
        return (
            {name: np.asarray(source[name]) for name in _E45_UNIT_FIELDS},
            float(metadata.get("extraction_seconds", 0.0)),
        )


def _e45_selected_scientific(
    units: Mapping[str, np.ndarray], match: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    flat = {
        name: np.asarray(value).reshape((-1,) + value.shape[2:])
        for name, value in units.items()
    }
    index = match["matched_flat_index"]
    selected = {
        f"matched_{name}": flat[name][index]
        for name in (
            "bank_seed", "source", "center_frame", "frame_id", "support_semantic",
            "range_bin", "azimuth_sector", "median_distance_m", "median_beam",
            "Nvis", "O_hat", "local_density", "geometry_hits", "point_count",
            "point_features", "unit_hash",
        )
    }
    return {**match, **selected}


def _e45_pair_smd(values: np.ndarray) -> np.ndarray:
    left, right = values[:, 0], values[:, 1]
    pooled = np.sqrt((left.var(axis=0, ddof=1) + right.var(axis=0, ddof=1)) / 2.0)
    difference = np.abs(left.mean(axis=0) - right.mean(axis=0))
    return np.divide(
        difference, pooled, out=np.where(difference == 0.0, 0.0, np.inf),
        where=pooled > 0.0,
    )


def _e45_hash_tie_cost(primary: np.ndarray, tie_hash: np.ndarray) -> np.ndarray:
    """Order exactly equal edge costs by the frozen unit hashes."""
    cost = np.asarray(primary, dtype=np.float64).copy()
    order = np.lexsort((tie_hash, cost))
    start = 0
    while start < order.size:
        stop = start + 1
        value = cost[order[start]]
        while stop < order.size and cost[order[stop]] == value:
            stop += 1
        adjusted = value
        for position in order[start:stop]:
            cost[position] = adjusted
            adjusted = np.nextafter(adjusted, np.inf)
        start = stop
    return cost


def _e45_pair_match(
    units: Mapping[str, np.ndarray], left_source: int, right_source: int,
) -> dict[str, np.ndarray]:
    """Find the maximum legal pair set, then minimize normalized imbalance."""
    flat = {
        name: np.asarray(value).reshape((-1,) + value.shape[2:])
        for name, value in units.items()
    }
    covariates = _e45_covariates(flat)
    valid = (
        (flat["point_count"] > 0) & (flat["range_bin"] >= 0)
        & (flat["range_bin"] < 4) & (flat["azimuth_sector"] >= 0)
        & np.isfinite(covariates).all(axis=1)
    )
    by_source: list[np.ndarray] = []
    for source_id in (left_source, right_source):
        candidates = np.flatnonzero(valid & (flat["source"] == source_id))
        candidates = candidates[np.argsort(flat["unit_hash"][candidates], kind="stable")]
        _, first = np.unique(flat["unit_hash"][candidates], return_index=True)
        by_source.append(candidates[np.sort(first)])

    caliper = np.asarray((2.0, 4.0, 0.25, 0.10, 0.25), dtype=np.float64)
    matched: list[tuple[int, int]] = []
    legal_edge_count = 0
    strata_with_edges = 0
    for support in (40, 48):
        for range_bin in range(4):
            for sector in range(8):
                groups = [
                    source[
                        (flat["support_semantic"][source] == support)
                        & (flat["range_bin"][source] == range_bin)
                        & (flat["azimuth_sector"][source] == sector)
                    ]
                    for source in by_source
                ]
                left, right = groups
                if left.size == 0 or right.size == 0:
                    continue
                right_tree = cKDTree(covariates[right] / caliper, compact_nodes=True)
                neighbours = right_tree.query_ball_point(
                    covariates[left] / caliper, r=1.0, p=np.inf, workers=1,
                )
                edge_left: list[int] = []
                edge_right: list[int] = []
                for left_local, possible in enumerate(neighbours):
                    if not possible:
                        continue
                    possible_array = np.asarray(possible, dtype=np.int64)
                    legal = possible_array[np.all(
                        np.abs(covariates[right[possible_array]] - covariates[left[left_local]])
                        <= caliper,
                        axis=1,
                    )]
                    edge_left.extend([left_local] * legal.size)
                    edge_right.extend(legal.tolist())
                if not edge_left:
                    continue
                rows = np.asarray(edge_left, dtype=np.int64)
                columns = np.asarray(edge_right, dtype=np.int64)
                differences = (
                    covariates[left[rows]] - covariates[right[columns]]
                ) / caliper
                primary = 1.0 + np.square(differences).sum(axis=1)
                with np.errstate(over="ignore"):
                    tie_hash = (
                        flat["unit_hash"][left[rows]].astype(np.uint64)
                        * np.uint64(0x9E3779B97F4A7C15)
                        + flat["unit_hash"][right[columns]].astype(np.uint64)
                    )
                real_cost = _e45_hash_tie_cost(primary, tie_hash)
                penalty = float((left.size + 1) * (float(real_cost.max()) + 1.0))
                dummy_rows = np.arange(left.size, dtype=np.int64)
                graph = coo_matrix(
                    (
                        np.concatenate((real_cost, np.full(left.size, penalty))),
                        (
                            np.concatenate((rows, dummy_rows)),
                            np.concatenate((columns, right.size + dummy_rows)),
                        ),
                    ),
                    shape=(left.size, right.size + left.size),
                ).tocsr()
                row_match, column_match = min_weight_full_bipartite_matching(graph)
                real = column_match < right.size
                pairs = np.column_stack((left[row_match[real]], right[column_match[real]]))
                pairs = pairs[np.argsort(flat["unit_hash"][pairs[:, 0]], kind="stable")]
                matched.extend((int(pair[0]), int(pair[1])) for pair in pairs)
                legal_edge_count += rows.size
                strata_with_edges += 1

    matched_index = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    matched_covariates = (
        covariates[matched_index] if matched
        else np.empty((0, 2, 5), dtype=np.float64)
    )
    smd = (
        _e45_pair_smd(matched_covariates)
        if len(matched) > 1 else np.full(5, np.inf)
    )
    range_count = (
        np.bincount(flat["range_bin"][matched_index[:, 0]], minlength=4)[:4]
        if matched else np.zeros(4, dtype=np.int64)
    )
    caliper_errors = int(np.count_nonzero(
        np.abs(matched_covariates[:, 0] - matched_covariates[:, 1]) > caliper
    )) if matched else 0
    duplicate_errors = int(sum(
        len(matched) - np.unique(flat["unit_hash"][matched_index[:, side]]).size
        for side in range(2)
    )) if matched else 0
    return {
        "matched_flat_index": matched_index,
        "matched_covariates": matched_covariates,
        "pairwise_smd": smd,
        "matched_range_count": range_count.astype(np.int64),
        "matched_center_frames": np.unique(
            flat["center_frame"][matched_index[:, 0]]
        ).astype(np.int16) if matched else np.empty(0, dtype=np.int16),
        "legal_edge_count": np.asarray(legal_edge_count, dtype=np.int64),
        "strata_with_edges": np.asarray(strata_with_edges, dtype=np.int64),
        "caliper_errors": np.asarray(caliper_errors, dtype=np.int64),
        "duplicate_errors": np.asarray(duplicate_errors, dtype=np.int64),
    }


def run_e45_pair_qualification(
    unit_cache_path: Path | str, output_path: Path | str, *, experiment: str,
    left_source: int, right_source: int,
) -> dict[str, object]:
    """Qualify one frozen pairwise common-support estimand."""
    cache_path = Path(unit_cache_path).expanduser().resolve(strict=True)
    with np.load(cache_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if (
            metadata.get("experiment") != "E45-v2-units"
            or metadata.get("passed") is not True
            or int(metadata.get("capacity", -1)) != 2048
        ):
            raise RenderError(f"{experiment} requires the formal 2,048-capacity E45-v2 cache")
        units = {name: np.asarray(source[name]) for name in _E45_UNIT_FIELDS}
    started = time.monotonic()
    runs = [_e45_pair_match(units, left_source, right_source) for _ in range(2)]
    matching_seconds = time.monotonic() - started
    reproduced = all(
        np.array_equal(runs[0][name], runs[1][name]) for name in runs[0]
    )
    match = runs[0]
    matched_count = int(match["matched_flat_index"].shape[0])
    center_frames = int(match["matched_center_frames"].size)
    range_count = np.asarray(match["matched_range_count"])
    maximum_smd = float(np.max(match["pairwise_smd"]))
    caliper_errors = int(match["caliper_errors"])
    duplicate_errors = int(match["duplicate_errors"])
    passed = (
        matched_count >= 1024 and center_frames >= 100
        and bool(np.all(range_count > 0)) and caliper_errors == 0
        and duplicate_errors == 0 and maximum_smd <= 0.10 and reproduced
    )
    scientific = _e45_selected_scientific(units, match)
    result = {
        "experiment": experiment,
        "passed": passed,
        "failure_classification": None if passed else "insufficient_pairwise_common_support",
        "source_pair": [left_source, right_source],
        "unit_cache_sha256": _sha256_path(cache_path),
        "unit_cache_scientific_array_hash": metadata["scientific_array_hash"],
        "capacity": 2048,
        "estimand_range_m": [2.5, 40.0],
        "range_40_50_status": "unobservable_for_real-vs-rendered-object matching in train/201",
        "matching_objective": "maximum_cardinality_then_minimum_sum_squared_normalized_covariate_difference",
        "matched_pairs": matched_count,
        "center_frames": center_frames,
        "matched_range_count_2p5_to_40": range_count.tolist(),
        "legal_edges": int(match["legal_edge_count"]),
        "exact_strata_with_edges": int(match["strata_with_edges"]),
        "pairwise_smd": match["pairwise_smd"].tolist(),
        "maximum_pairwise_smd": maximum_smd,
        "caliper_errors": caliper_errors,
        "duplicate_errors": duplicate_errors,
        "elementwise_reproduced": reproduced,
        "two_run_matching_seconds": matching_seconds,
        "scientific_array_hash": _scientific_array_hash(scientific),
        "claim_limit": (
            "The pairwise source audit is limited to train/201 real-object support "
            "from 2.5 m through 40 m; 40--50 m has no direct real-object matching evidence."
        ),
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output)
    return result


def run_e45a_qualification(
    unit_cache_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    return run_e45_pair_qualification(
        unit_cache_path, output_path, experiment="E45A",
        left_source=0, right_source=1,
    )


def _e45_weighted_balance(
    units: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Estimate deterministic overlap weights on the frozen E46 confounders."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    flat = {
        name: np.asarray(value).reshape((-1,) + value.shape[2:])
        for name, value in units.items()
    }
    covariates = _e45_covariates(flat)
    valid = (
        np.isin(flat["source"], (0, 1))
        & (flat["point_count"] > 0)
        & (flat["range_bin"] >= 0)
        & (flat["range_bin"] < 4)
        & (flat["azimuth_sector"] >= 0)
        & np.isfinite(covariates).all(axis=1)
        & (flat["O_hat"] >= 0.0)
        & (flat["O_hat"] <= 1.0)
    )
    candidates: list[np.ndarray] = []
    for source_id in (0, 1):
        index = np.flatnonzero(valid & (flat["source"] == source_id))
        index = index[np.argsort(flat["unit_hash"][index], kind="stable")]
        _, first = np.unique(flat["unit_hash"][index], return_index=True)
        candidates.append(index[np.sort(first)])

    occlusion = np.digitize(flat["O_hat"], (0.25, 0.75), right=False).astype(np.int8)
    cell_key = np.column_stack((
        flat["support_semantic"].astype(np.int64),
        flat["range_bin"].astype(np.int64),
        flat["azimuth_sector"].astype(np.int64),
        occlusion.astype(np.int64),
    ))
    source_cells = [
        {tuple(map(int, row)) for row in cell_key[index]}
        for index in candidates
    ]
    common_keys = sorted(source_cells[0] & source_cells[1])
    common_lookup = {key: cell_id for cell_id, key in enumerate(common_keys)}
    selected_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    for index in candidates:
        keep = np.asarray(
            [tuple(map(int, cell_key[item])) in common_lookup for item in index],
            dtype=np.bool_,
        )
        selected = index[keep]
        selected_parts.append(selected)
        cell_parts.append(np.asarray(
            [common_lookup[tuple(map(int, cell_key[item]))] for item in selected],
            dtype=np.int32,
        ))
    selected_index = np.concatenate(selected_parts)
    selected_cell = np.concatenate(cell_parts)
    selected_source = flat["source"][selected_index].astype(np.uint8)
    selected_covariates = covariates[selected_index]
    if len(common_keys) == 0 or min(map(len, selected_parts)) < 2:
        return {
            "selected_flat_index": selected_index.astype(np.int64),
            "selected_source": selected_source,
            "selected_cell": selected_cell,
            "common_cell_key": np.asarray(common_keys, dtype=np.int64).reshape(-1, 4),
            "selected_covariates": selected_covariates,
            "unit_weight": np.empty(selected_index.size, dtype=np.float64),
            "propensity": np.empty(selected_index.size, dtype=np.float64),
            "weighted_smd": np.full(5, np.inf),
            "weighted_ks": np.full(5, np.inf),
            "effective_sample_size": np.zeros(2, dtype=np.float64),
            "center_frame_count": np.zeros(2, dtype=np.int64),
            "maximum_cell_mass_difference": np.asarray(np.inf),
            "maximum_basis_balance_error": np.asarray(np.inf),
            "maximum_weight_fraction": np.ones(2, dtype=np.float64),
            "optimizer_iterations": np.asarray(0, dtype=np.int64),
        }

    rows = np.arange(selected_index.size, dtype=np.int64)
    cell_design = csr_matrix(
        (np.ones(rows.size), (rows, selected_cell)),
        shape=(rows.size, len(common_keys)),
    )
    mean = selected_covariates.mean(axis=0)
    scale = selected_covariates.std(axis=0)
    scale[scale == 0.0] = 1.0
    continuous = (selected_covariates - mean) / scale
    quantiles = np.quantile(
        selected_covariates, np.linspace(0.05, 0.95, 19), axis=0,
        method="linear",
    )
    distribution_basis = np.column_stack([
        selected_covariates[:, feature] <= threshold
        for feature in range(5) for threshold in np.unique(quantiles[:, feature])
    ]).astype(np.float64)
    design = hstack(
        (cell_design, csr_matrix(continuous), csr_matrix(distribution_basis)),
        format="csr",
    )
    model = LogisticRegression(
        penalty="l2", C=np.inf, l1_ratio=0.0,
        fit_intercept=False, solver="lbfgs",
        tol=1.0e-9, max_iter=10_000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(design, selected_source)
    propensity = model.predict_proba(design)[:, 1]
    weight = np.where(selected_source == 0, propensity, 1.0 - propensity)
    for source_id in (0, 1):
        source = selected_source == source_id
        total = float(weight[source].sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RenderError("E45A-overlap produced a source with zero total weight")
        weight[source] /= total

    def weighted_mean_variance(
        values: np.ndarray, weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        average = np.sum(values * weights[:, None], axis=0)
        variance = np.sum(np.square(values - average) * weights[:, None], axis=0)
        return average, variance

    source_mask = [selected_source == source_id for source_id in (0, 1)]
    summaries = [
        weighted_mean_variance(selected_covariates[mask], weight[mask])
        for mask in source_mask
    ]
    difference = np.abs(summaries[0][0] - summaries[1][0])
    pooled = np.sqrt((summaries[0][1] + summaries[1][1]) / 2.0)
    smd = np.divide(
        difference, pooled, out=np.where(difference == 0.0, 0.0, np.inf),
        where=pooled > 0.0,
    )
    ks = np.zeros(5, dtype=np.float64)
    for feature in range(5):
        order = np.argsort(selected_covariates[:, feature], kind="stable")
        sorted_values = selected_covariates[order, feature]
        sorted_source = selected_source[order]
        sorted_weight = weight[order]
        _, starts = np.unique(sorted_values, return_index=True)
        left_cdf = np.cumsum(np.add.reduceat(
            np.where(sorted_source == 0, sorted_weight, 0.0), starts,
        ))
        right_cdf = np.cumsum(np.add.reduceat(
            np.where(sorted_source == 1, sorted_weight, 0.0), starts,
        ))
        ks[feature] = float(np.max(np.abs(left_cdf - right_cdf)))
    ess = np.asarray([
        1.0 / np.square(weight[mask]).sum() for mask in source_mask
    ])
    center_count = np.asarray([
        np.unique(flat["center_frame"][selected_index[mask]][weight[mask] > 0.0]).size
        for mask in source_mask
    ], dtype=np.int64)
    cell_mass = np.asarray([
        np.bincount(selected_cell[mask], weights=weight[mask], minlength=len(common_keys))
        for mask in source_mask
    ])
    fitted_basis = np.asarray(design[:, len(common_keys):].toarray())
    basis_mean = np.asarray([
        np.sum(fitted_basis[mask] * weight[mask, None], axis=0)
        for mask in source_mask
    ])
    return {
        "selected_flat_index": selected_index.astype(np.int64),
        "selected_source": selected_source,
        "selected_cell": selected_cell,
        "common_cell_key": np.asarray(common_keys, dtype=np.int64),
        "selected_covariates": selected_covariates,
        "unit_weight": weight,
        "propensity": propensity,
        "weighted_smd": smd,
        "weighted_ks": ks,
        "effective_sample_size": ess,
        "center_frame_count": center_count,
        "maximum_cell_mass_difference": np.asarray(
            np.max(np.abs(cell_mass[0] - cell_mass[1]))
        ),
        "maximum_basis_balance_error": np.asarray(
            np.max(np.abs(basis_mean[0] - basis_mean[1]))
        ),
        "maximum_weight_fraction": np.asarray([
            np.max(weight[mask]) for mask in source_mask
        ]),
        "optimizer_iterations": np.asarray(int(model.n_iter_[0]), dtype=np.int64),
    }


def run_e45a_overlap_qualification(
    unit_cache_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Qualify the weighted common-overlap population used by E46."""
    cache_path = Path(unit_cache_path).expanduser().resolve(strict=True)
    if _sha256_path(cache_path) != "92fe629be31a7b5a5eb97bd1ee6a7d402d69fc507b1fbd23e925a19cab1be6cf":
        raise RenderError("E45A-overlap requires the frozen E45A-new 2,048-unit cache")
    with np.load(cache_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        if (
            metadata.get("experiment") != "E45-v2-units"
            or metadata.get("passed") is not True
            or int(metadata.get("capacity", -1)) != 2048
            or metadata.get("scientific_array_hash")
            != "39c2d55e9cd9a6acb5337d6d1eae0bf815de40e3a9c8ac1d1827af8a1f64f3d1"
        ):
            raise RenderError("E45A-overlap unit-cache metadata changed")
        units = {name: np.asarray(source[name]) for name in _E45_UNIT_FIELDS}
    started = time.monotonic()
    runs = [_e45_weighted_balance(units) for _ in range(2)]
    elapsed = time.monotonic() - started
    reproduced = all(np.array_equal(runs[0][name], runs[1][name]) for name in runs[0])
    balance = runs[0]
    common_cells = int(balance["common_cell_key"].shape[0])
    source_units = np.bincount(balance["selected_source"], minlength=2).astype(np.int64)
    ess = np.asarray(balance["effective_sample_size"])
    center_frames = np.asarray(balance["center_frame_count"])
    maximum_smd = float(np.max(balance["weighted_smd"]))
    maximum_ks = float(np.max(balance["weighted_ks"]))
    passed = (
        common_cells > 0 and bool(np.all(source_units > 0))
        and bool(np.all(ess >= 256.0)) and int(center_frames[0]) >= 100
        and maximum_smd <= 0.10 and maximum_ks <= 0.10 and reproduced
    )
    if passed:
        failure_classification = None
        failure_reason = None
    elif common_cells == 0 or int(center_frames[0]) < 100:
        failure_classification = "sample_or_observability_defect"
        failure_reason = "insufficient_exact_common_support"
    elif bool(np.any(ess < 256.0)):
        failure_classification = "scientific_failure"
        failure_reason = "insufficient_effective_overlap"
    else:
        failure_classification = "scientific_failure"
        failure_reason = "structural_covariate_imbalance"
    scientific = {
        **balance,
        "selected_unit_hash": np.asarray(units["unit_hash"]).reshape(-1)[
            balance["selected_flat_index"]
        ],
        "selected_center_frame": np.asarray(units["center_frame"]).reshape(-1)[
            balance["selected_flat_index"]
        ],
        "selected_point_count": np.asarray(units["point_count"]).reshape(-1)[
            balance["selected_flat_index"]
        ],
    }
    result = {
        "experiment": "E45A-overlap",
        "passed": passed,
        "failure_classification": failure_classification,
        "failure_reason": failure_reason,
        "unit_cache_sha256": _sha256_path(cache_path),
        "unit_cache_scientific_array_hash": metadata["scientific_array_hash"],
        "estimand_range_m": [2.5, 40.0],
        "exact_strata": [
            "support_semantic", "range_bin", "45_degree_azimuth_sector",
            "occlusion_stratum_[0,0.25)_[0.25,0.75)_[0.75,1]",
        ],
        "weighting": (
            "unpenalized_logistic_overlap_weights_with_full_cell_indicators_"
            "five_continuous_covariates_and_pooled_5_to_95_percentile_ecdf_basis"
        ),
        "common_exact_cells": common_cells,
        "common_support_units_by_source": source_units.tolist(),
        "center_frames_by_source": center_frames.tolist(),
        "effective_sample_size_by_source": ess.tolist(),
        "maximum_weight_fraction_by_source": balance["maximum_weight_fraction"].tolist(),
        "weighted_smd": balance["weighted_smd"].tolist(),
        "maximum_weighted_smd": maximum_smd,
        "weighted_ks": balance["weighted_ks"].tolist(),
        "maximum_weighted_ks": maximum_ks,
        "maximum_exact_cell_mass_difference": float(balance["maximum_cell_mass_difference"]),
        "maximum_fitted_basis_balance_error": float(balance["maximum_basis_balance_error"]),
        "optimizer_iterations": int(balance["optimizer_iterations"]),
        "elementwise_reproduced": reproduced,
        "two_run_seconds": elapsed,
        "scientific_array_hash": _scientific_array_hash(scientific),
        "claim_limit": (
            "E45A-overlap qualifies only the weighted train/201 real-normal and "
            "E25-new control population from 2.5 m through 40 m for E46; it does "
            "not establish renderer indistinguishability."
        ),
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output)
    return result


def run_e45b_qualification(
    unit_cache_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    return run_e45_pair_qualification(
        unit_cache_path, output_path, experiment="E45B",
        left_source=1, right_source=2,
    )


def _e45_v2_unit_from_record(record: Mapping[str, object]) -> _Gate1BankUnit:
    """Decode a freshly generated pair-bank record without rerunning placement."""
    return _Gate1BankUnit(
        int(record["bank_seed"]), int(record["center_frame"]),
        int(record["real_semantic"]), int(record["real_instance"]),
        int(record["real_support_semantic"]),
        int(record["control_support_semantic"]),
        int(record["proxy_support_semantic"]),
        int(record["control_template_index"]),
        int(record["control_assigned_range_bin"]),
        int(record["control_final_range_bin"]),
        int(record["control_visible_returns"]),
        WorldSpec.from_dict(json.loads(str(record["control_world_json"]))),
        WorldSpec.from_dict(json.loads(str(record["proxy_world_json"]))),
    )


def run_e45_pair_v2_qualification(
    data_root: Path | str, e25_new_artifact_path: Path | str,
    calibration_path: Path | str, support_pool_path: Path | str,
    output_path: Path | str, *, experiment: str, processes: int = 24,
) -> dict[str, object]:
    """Build one independent pair bank and stop at the first passing capacity."""
    if experiment not in {"E45A-new", "E45B-v2"} or processes != 24:
        raise RenderError("formal E45 pair qualification requires a known pair and 24 processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    project_root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(project_root / "protocol.json")
    e25_path = Path(e25_new_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(e25_path) != FROZEN_E25_NEW_ARTIFACT_SHA256:
        raise RenderError("E45 pair bank E25-new artifact identity changed")
    sequence_206 = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=206,
        label_mode=LabelMode.REQUIRED,
    )
    templates = extract_normal_template_library(
        sequence_206.source_frame(frame_id) for frame_id in sequence_206.frame_ids
    )
    _, _, template_hash = canonical_normal_template_library_identity(templates)
    del sequence_206
    gc.collect()
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    pool, _ = load_gate1_support_pool(support_pool_path)
    calibration = Path(calibration_path).expanduser().resolve(strict=True)
    if _sha256_path(calibration) != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise RenderError("E45 pair bank calibration identity changed")
    grid, sensor = load_sensor_calibration(calibration)
    frame_ids = tuple(range(4, 682))
    trajectory_yaws = _trajectory_yaw_by_pose({
        frame_id: sequence.lidar_pose(frame_id) for frame_id in frame_ids
    })
    frame_keys, obstacles = _gate1_frame_keys_and_obstacles(
        sequence, build_obstacles=True,
    )
    assert obstacles is not None
    real_candidates = _gate1_real_candidates(sequence, pool, frame_keys=frame_keys)
    context = build_coverage_control_context(
        (), pool, grid, sensor, frame_loader=sequence.source_frame,
        frame_ids=frame_ids, source_sequence_id=201,
        trajectory_yaws=trajectory_yaws,
    )
    _initialize_gate1_candidate_generation(
        sequence, context, obstacles, templates, real_candidates,
    )
    global _GATE1_BANK_SEED_BASE, _GATE1_BANK_CAPACITY_LIMIT, _E45_BANK
    _GATE1_BANK_SEED_BASE = 4_500_000 if experiment == "E45A-new" else 4_600_000
    _GATE1_BANK_CAPACITY_LIMIT = 2048
    left_source, right_source = ((0, 1) if experiment == "E45A-new" else (1, 2))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    units: dict[str, np.ndarray] | None = None
    history: list[dict[str, object]] = []
    final_match: dict[str, np.ndarray] | None = None
    for previous, capacity in zip((0, 512, 1024), (512, 1024, 2048)):
        bank_started = time.monotonic()
        with mp.get_context("fork").Pool(processes=processes) as workers:
            suffix_records = workers.map(
                _gate1_bank_worker, range(previous, capacity), chunksize=1
            )
        bank_seconds = time.monotonic() - bank_started
        records.extend(suffix_records)
        bank_arrays = _gate1_bank_arrays(records)
        bank_errors = int(np.count_nonzero(bank_arrays["error"] != ""))
        if bank_errors:
            raise RenderError(f"{experiment} pair bank exhausted {bank_errors} units")
        bank_hash = _scientific_array_hash(bank_arrays)
        bank_path = output.parent / f"{experiment.lower()}_bank_{capacity}.npz"
        bank_metadata = {
            "experiment": f"{experiment}-pair-bank", "passed": True,
            "capacity": capacity, "seed_base": _GATE1_BANK_SEED_BASE,
            "processes": processes, "elapsed_seconds": bank_seconds,
            "scientific_array_hash": bank_hash,
            "support_pool_sha256": FROZEN_GATE1_SUPPORT_POOL_SHA256,
            "calibration_sha256": FROZEN_SENSOR_CALIBRATION_SHA256,
            "normal_template_library_sha256": template_hash,
        }
        temporary = bank_path.with_suffix(bank_path.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary, **bank_arrays,
            metadata_json=np.asarray(json.dumps(
                bank_metadata, sort_keys=True, separators=(",", ":")
            )),
        )
        os.replace(temporary, bank_path)
        _E45_BANK = tuple(_e45_v2_unit_from_record(row) for row in suffix_records)
        extraction_started = time.monotonic()
        with mp.get_context("fork").Pool(processes=processes) as workers:
            extracted = workers.map(
                _e45_worker, range(len(suffix_records)), chunksize=1
            )
        extraction_seconds = time.monotonic() - extraction_started
        suffix = {
            name: np.stack([record[name] for record in extracted])
            for name in _E45_UNIT_FIELDS
        }
        units = suffix if units is None else {
            name: np.concatenate((units[name], suffix[name]))
            for name in _E45_UNIT_FIELDS
        }
        unit_path = output.parent / f"{experiment.lower()}_units_{capacity}.npz"
        _write_e45_units(units, unit_path, bank_hash, extraction_seconds)
        matching_started = time.monotonic()
        match = _e45_pair_match(units, left_source, right_source)
        matching_seconds = time.monotonic() - matching_started
        matched_count = int(match["matched_flat_index"].shape[0])
        center_frames = int(match["matched_center_frames"].size)
        range_count = np.asarray(match["matched_range_count"])
        maximum_smd = float(np.max(match["pairwise_smd"]))
        passed = (
            matched_count >= 1024 and center_frames >= 100
            and bool(np.all(range_count > 0))
            and int(match["caliper_errors"]) == 0
            and int(match["duplicate_errors"]) == 0 and maximum_smd <= 0.10
        )
        history.append({
            "capacity": capacity, "matched_pairs": matched_count,
            "center_frames": center_frames, "range_count": range_count.tolist(),
            "legal_edges": int(match["legal_edge_count"]),
            "pairwise_smd": match["pairwise_smd"].tolist(),
            "maximum_pairwise_smd": maximum_smd,
            "caliper_errors": int(match["caliper_errors"]),
            "duplicate_errors": int(match["duplicate_errors"]),
            "bank_seconds": bank_seconds,
            "unit_extraction_seconds": extraction_seconds,
            "matching_seconds": matching_seconds, "passed": passed,
        })
        final_match = match
        if passed:
            break
    assert units is not None and final_match is not None
    scientific = _e45_selected_scientific(units, final_match)
    result = {
        "experiment": experiment, "passed": bool(history[-1]["passed"]),
        "failure_classification": (
            None if history[-1]["passed"] else "insufficient_pairwise_common_support"
        ),
        "capacity_ladder": history, "final_capacity": history[-1]["capacity"],
        "matched_pairs": history[-1]["matched_pairs"],
        "center_frames": history[-1]["center_frames"],
        "matched_range_count_2p5_to_40": history[-1]["range_count"],
        "pairwise_smd": history[-1]["pairwise_smd"],
        "maximum_pairwise_smd": history[-1]["maximum_pairwise_smd"],
        "caliper_errors": history[-1]["caliper_errors"],
        "duplicate_errors": history[-1]["duplicate_errors"],
        "formal_repetitions": 1, "elementwise_reproduced": None,
        "reproducibility_check": "not_run_by_owner_decision",
        "seed_base": _GATE1_BANK_SEED_BASE,
        "scientific_array_hash": _scientific_array_hash(scientific),
    }
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(
            result, sort_keys=True, separators=(",", ":")
        )),
    )
    os.replace(temporary, output)
    return result


FROZEN_E45B_V2_ARTIFACT_SHA256 = (
    "19ecbc843cc5325e3f12497c50e5855388f0f5caa581179f6fd6639613a8ecfd"
)
E48_FOLD_NAMESPACE = "E48-center-v1"
E48_BOOTSTRAP_SEED = (4800, 2000)


def _e48_center_fold(frame_id: int) -> int:
    """Assign one center frame without reading features or labels."""
    digest = hashlib.sha256(
        f"{E48_FOLD_NAMESPACE}:{frame_id}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") % 5


def _e48_fold_plan(center_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Keep matched pairs intact and embargo every test center from training."""
    centers = np.asarray(center_frame, dtype=np.int64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RenderError("E48 center-frame identity must be pair x source")
    frames = np.unique(centers)
    frame_fold = np.asarray(
        [_e48_center_fold(int(frame)) for frame in frames], dtype=np.int8
    )
    lookup = {int(frame): int(fold) for frame, fold in zip(frames, frame_fold, strict=True)}
    pair_fold = np.asarray(
        [[lookup[int(left)], lookup[int(right)]] for left, right in centers],
        dtype=np.int8,
    )
    test = np.zeros((5, centers.shape[0]), dtype=np.bool_)
    train = np.zeros_like(test)
    excluded = np.zeros_like(test)
    for fold in range(5):
        test[fold] = (pair_fold[:, 0] == fold) & (pair_fold[:, 1] == fold)
        train[fold] = (pair_fold[:, 0] != fold) & (pair_fold[:, 1] != fold)
        excluded[fold] = ~(test[fold] | train[fold])
    return {
        "unique_center_frame": frames.astype(np.int16),
        "center_frame_fold": frame_fold,
        "pair_center_frame_fold": pair_fold,
        "fold_test_pair": test,
        "fold_train_pair": train,
        "fold_excluded_pair": excluded,
    }


def _e48_points(
    features: np.ndarray, point_count: np.ndarray, pair_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten selected pairs while giving every entity-frame total weight one."""
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for local_pair, pair in enumerate(np.asarray(pair_index, dtype=np.int64)):
        for source in range(2):
            count = int(point_count[pair, source])
            if count < 1 or count > 64:
                raise RenderError("E48 unit point count is outside 1..64")
            values = np.asarray(features[pair, source, :count], dtype=np.float64)
            if values.shape != (count, 7) or not np.isfinite(values).all():
                raise RenderError("E48 low-level point features are invalid")
            rows.append(values)
            labels.append(np.full(count, source, dtype=np.uint8))
            pairs.append(np.full(count, local_pair, dtype=np.int64))
            weights.append(np.full(count, 1.0 / count, dtype=np.float64))
    return (
        np.concatenate(rows), np.concatenate(labels),
        np.concatenate(pairs), np.concatenate(weights),
    )


def _e48_metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted AUC, balanced accuracy, and both class recalls."""
    from sklearn.metrics import roc_auc_score

    recall = np.asarray([
        np.sum(weights[(labels == source) & (predictions == source)])
        / np.sum(weights[labels == source])
        for source in (0, 1)
    ], dtype=np.float64)
    return np.asarray((
        float(roc_auc_score(labels, scores, sample_weight=weights)),
        float(recall.mean()), float(recall[0]), float(recall[1]),
    ))


def _e48_pair_bootstrap(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray,
    point_pair: np.ndarray, base_weight: np.ndarray, pair_count: int,
) -> np.ndarray:
    """Resample matched pair identities and carry both entity-frame units together."""
    rng = np.random.default_rng(np.random.SeedSequence(E48_BOOTSTRAP_SEED))
    output = np.empty((2000, 4), dtype=np.float64)
    probability = np.full(pair_count, 1.0 / pair_count, dtype=np.float64)
    for replicate in range(2000):
        multiplicity = rng.multinomial(pair_count, probability)
        weights = base_weight * multiplicity[point_pair]
        output[replicate] = _e48_metrics(labels, scores, predictions, weights)
    return output


def run_e48_qualification(
    e45b_v2_artifact_path: Path | str, output_path: Path | str,
) -> dict[str, object]:
    """Audit near-saturated rendered label prediction with two low-capacity models."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    source_path = Path(e45b_v2_artifact_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E45B_V2_ARTIFACT_SHA256:
        raise RenderError("E48 requires the frozen E45B-v2 artifact")
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        scientific = {
            name: np.asarray(source[name])
            for name in source.files if name != "metadata_json"
        }
    if (
        metadata.get("experiment") != "E45B-v2"
        or metadata.get("passed") is not True
        or metadata.get("scientific_array_hash") != _scientific_array_hash(scientific)
    ):
        raise RenderError("E48 input metadata or scientific arrays changed")
    source_id = np.asarray(scientific["matched_source"])
    expected_source = np.tile(
        np.asarray((1, 2), dtype=np.uint8), (source_id.shape[0], 1)
    )
    if source_id.shape != (1347, 2) or not np.array_equal(source_id, expected_source):
        raise RenderError("E48 requires 1,347 ordered control/proxy pairs")
    point_count = np.asarray(scientific["matched_point_count"], dtype=np.int16)
    point_features = np.asarray(scientific["matched_point_features"], dtype=np.float64)
    if point_features.shape != (1347, 2, 64, 7) or point_count.shape != (1347, 2):
        raise RenderError("E48 point-feature contract changed")

    plan = _e48_fold_plan(scientific["matched_center_frame"])
    test_count = plan["fold_test_pair"].sum(axis=1)
    train_count = plan["fold_train_pair"].sum(axis=1)
    excluded_count = plan["fold_excluded_pair"].sum(axis=1)
    if (
        plan["unique_center_frame"].size != 294
        or not np.array_equal(test_count, (61, 46, 91, 74, 97))
        or not np.array_equal(train_count, (913, 1004, 800, 861, 832))
        or not np.array_equal(excluded_count, (373, 297, 456, 412, 418))
        or int(test_count.sum()) != 369
        or np.any(plan["fold_test_pair"].sum(axis=0) > 1)
    ):
        raise RenderError("E48 frozen fold identity changed")
    centers = np.asarray(scientific["matched_center_frame"], dtype=np.int64)
    for fold in range(5):
        train_pairs = np.flatnonzero(plan["fold_train_pair"][fold])
        test_pairs = np.flatnonzero(plan["fold_test_pair"][fold])
        if np.intersect1d(centers[train_pairs], centers[test_pairs]).size:
            raise RenderError("E48 center frame crosses train and test")

    model_names = ("l2_logistic_regression", "depth3_decision_tree")
    oof_score = np.full((2, 1347, 2, 64), np.nan, dtype=np.float64)
    oof_prediction = np.full((2, 1347, 2, 64), 255, dtype=np.uint8)
    model_iterations = np.zeros((2, 5), dtype=np.int64)
    started = time.monotonic()
    for fold in range(5):
        train_pairs = np.flatnonzero(plan["fold_train_pair"][fold])
        test_pairs = np.flatnonzero(plan["fold_test_pair"][fold])
        train_x, train_y, _, train_weight = _e48_points(
            point_features, point_count, train_pairs
        )
        test_x, test_y, _, _ = _e48_points(point_features, point_count, test_pairs)
        scaler = StandardScaler().fit(train_x, sample_weight=train_weight)
        train_scaled = scaler.transform(train_x)
        test_scaled = scaler.transform(test_x)
        logistic = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True,
            tol=1.0e-4, max_iter=5000, class_weight=None, random_state=4800,
        )
        tree = DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=3,
            min_samples_leaf=64, class_weight=None, random_state=4801,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            logistic.fit(train_scaled, train_y, sample_weight=train_weight)
        tree.fit(train_x, train_y, sample_weight=train_weight)
        models = ((logistic, test_scaled), (tree, test_x))
        model_iterations[0, fold] = int(logistic.n_iter_[0])
        model_iterations[1, fold] = int(tree.tree_.node_count)
        offset = 0
        for pair in test_pairs:
            pair_points = int(point_count[pair].sum())
            pair_slice = slice(offset, offset + pair_points)
            for model_id, (model, values) in enumerate(models):
                score = model.predict_proba(values[pair_slice])[:, 1]
                prediction = (score >= 0.5).astype(np.uint8)
                local = 0
                for source_index in range(2):
                    count = int(point_count[pair, source_index])
                    oof_score[model_id, pair, source_index, :count] = score[local:local + count]
                    oof_prediction[model_id, pair, source_index, :count] = prediction[local:local + count]
                    local += count
            offset += pair_points
        if offset != test_x.shape[0]:
            raise RenderError("E48 test prediction packing changed")
    elapsed = time.monotonic() - started

    oof_pair_index = np.flatnonzero(plan["fold_test_pair"].any(axis=0))
    if oof_pair_index.size != 369:
        raise RenderError("E48 OOF pair count changed")
    metric = np.empty((2, 4), dtype=np.float64)
    bootstrap = np.empty((2, 2000, 4), dtype=np.float64)
    for model_id in range(2):
        labels_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        pair_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for local_pair, pair in enumerate(oof_pair_index):
            for source_index in range(2):
                count = int(point_count[pair, source_index])
                labels_parts.append(np.full(count, source_index, dtype=np.uint8))
                score_parts.append(oof_score[model_id, pair, source_index, :count])
                prediction_parts.append(oof_prediction[model_id, pair, source_index, :count])
                pair_parts.append(np.full(count, local_pair, dtype=np.int64))
                weight_parts.append(np.full(count, 1.0 / count, dtype=np.float64))
        labels = np.concatenate(labels_parts)
        scores = np.concatenate(score_parts)
        predictions = np.concatenate(prediction_parts)
        point_pair = np.concatenate(pair_parts)
        base_weight = np.concatenate(weight_parts)
        if not np.isfinite(scores).all():
            raise RenderError("E48 OOF scores are incomplete")
        metric[model_id] = _e48_metrics(labels, scores, predictions, base_weight)
        bootstrap[model_id] = _e48_pair_bootstrap(
            labels, scores, predictions, point_pair, base_weight, 369
        )
    ci_low = np.quantile(bootstrap, 0.025, axis=1, method="linear")
    ci_high = np.quantile(bootstrap, 0.975, axis=1, method="linear")
    model_fail = (ci_low[:, 0] >= 0.95) & (ci_low[:, 1] >= 0.90)
    passed = not bool(np.any(model_fail))
    saved = {
        **plan,
        "oof_pair_index": oof_pair_index.astype(np.int64),
        "oof_score": oof_score,
        "oof_prediction": oof_prediction,
        "model_metric": metric,
        "bootstrap_metric": bootstrap,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "model_fail": model_fail,
        "model_iterations_or_nodes": model_iterations,
    }
    result = {
        "experiment": "E48", "passed": passed,
        "failure_classification": None if passed else "near_saturated_low_level_label_shortcut",
        "input_artifact_sha256": FROZEN_E45B_V2_ARTIFACT_SHA256,
        "input_scientific_array_hash": metadata["scientific_array_hash"],
        "fold_namespace": E48_FOLD_NAMESPACE,
        "unique_center_frames": 294,
        "fold_test_pairs": test_count.tolist(),
        "fold_train_pairs": train_count.tolist(),
        "fold_excluded_pairs": excluded_count.tolist(),
        "oof_pairs": 369,
        "models": list(model_names),
        "features": ["x", "y", "z", "intensity", "beam", "range", "local_density"],
        "maximum_points_per_unit": 64,
        "unit_total_weight": 1.0,
        "metric_order": ["roc_auc", "balanced_accuracy", "control_recall", "proxy_recall"],
        "metric": metric.tolist(),
        "bootstrap_replicates": 2000,
        "bootstrap_seed_sequence": list(E48_BOOTSTRAP_SEED),
        "bootstrap_cluster": "matched_pair",
        "bootstrap_ci_low": ci_low.tolist(),
        "bootstrap_ci_high": ci_high.tolist(),
        "model_fail": model_fail.tolist(),
        "fail_rule": "any_model_LCB95_AUC_ge_0.95_and_LCB95_BA_ge_0.90",
        "formal_repetitions": 1,
        "feature_ablation_or_attribution_run": False,
        "elapsed_seconds": elapsed,
        "scientific_array_hash": _scientific_array_hash(saved),
        "claim_limit": (
            "E48 only tests whether frozen low-level point observations nearly "
            "saturate rendered control/proxy label prediction; it does not test "
            "proxy supervision utility or distinguish semantic geometry from artifacts."
        ),
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **saved,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output)
    return result


_E57_SEQUENCE: object | None = None
_E57_GRID: RayGrid | None = None
_E57_SENSOR: SensorCalibration | None = None
_E57_BANK: dict[str, np.ndarray] = {}


def _e57_candidate_worker(index: int) -> dict[str, object]:
    """Build and characterize one mixed world without reading model output."""

    sequence, grid, sensor = _E57_SEQUENCE, _E57_GRID, _E57_SENSOR
    if sequence is None or grid is None or sensor is None or not _E57_BANK:
        raise RuntimeError("E57 candidate fixtures are not initialized")
    try:
        bank_seed = int(_E57_BANK["bank_seed"][index])
        center = int(_E57_BANK["center_frame"][index])
        attempt = int(_E57_BANK["attempt"][index])
        if str(_E57_BANK["error"][index]):
            raise RenderError("E57 source bank contains an incomplete row")
        control_world = WorldSpec.from_dict(
            json.loads(str(_E57_BANK["control_world_json"][index]))
        )
        proxy_world = WorldSpec.from_dict(
            json.loads(str(_E57_BANK["proxy_world_json"][index]))
        )
        control_record = PlacementRecord.from_dict(
            json.loads(str(_E57_BANK["control_record_json"][index]))
        )
        proxy_record = PlacementRecord.from_dict(
            json.loads(str(_E57_BANK["proxy_record_json"][index]))
        )
        if (
            control_world.seed != bank_seed
            or proxy_world.seed != bank_seed
            or control_world.source_sequence_id != 201
            or proxy_world.source_sequence_id != 201
            or control_world.world_type != "control_only"
            or proxy_world.world_type != "anomaly_only"
            or len(control_world.objects) != 1
            or len(proxy_world.objects) != 1
            or not 6 <= center <= 679
        ):
            raise RenderError("E57 source-bank row identity changed")
        control = replace(control_world.objects[0], object_id=1)
        proxy = replace(proxy_world.objects[0], object_id=2)
        control_record = replace(control_record, object_id=1)
        proxy_record = replace(proxy_record, object_id=2)
        world = WorldSpec(bank_seed, 201, (control, proxy))
        # This world composes two independently generated source-bank rows.  A
        # dedicated provenance record avoids inventing a single-world RNG trace.
        report = {
            "format": "ajae-e57-v2-composed-world-report",
            "source_bank_scientific_array_sha256": (
                FROZEN_E57_SOURCE_BANK_ARRAY_SHA256
            ),
            "source_bank_index": index,
            "bank_seed": bank_seed,
            "source_bank_attempt": attempt,
            "center_frame": center,
            "composition": "same-row control plus proxy with object IDs 1 and 2",
            "placement_record_json": [
                json.dumps(
                    control_record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(
                    proxy_record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
        }
        penetration, minimum_pair_sdf = obvious_pair_penetration(control, proxy)
        if penetration:
            return {
                "index": index,
                "bank_seed": bank_seed,
                "center_frame": center,
                "eligible": False,
                "reason": "pair_penetration",
                "minimum_pair_sdf_m": minimum_pair_sdf,
            }
        frames = tuple(
            sequence.source_frame(frame_id)
            for frame_id in range(center - 2, center + 3)
        )
        diagnostics = five_frame_world_diagnostics(world, frames, grid, sensor)
        objects = diagnostics["objects"]
        if not isinstance(objects, list) or len(objects) != 2:
            raise RenderError("E57 mixed-world diagnostics changed structure")
        by_id = {int(item["object_id"]): item for item in objects}
        if set(by_id) != {1, 2}:
            raise RenderError("E57 diagnostics lost an object identity")
        descriptor = np.asarray(
            [
                by_id[1]["Nvis"], by_id[1]["O"], by_id[1]["d"], by_id[1]["V"],
                by_id[2]["Nvis"], by_id[2]["O"], by_id[2]["d"], by_id[2]["V"],
            ],
            dtype=np.float64,
        )
        visible = (
            np.isfinite(descriptor).all()
            and descriptor[0] > 0.0
            and descriptor[4] > 0.0
            and descriptor[3] >= 1.0
            and descriptor[7] >= 1.0
        )
        rendered = render_frame(frames[2], world, grid, sensor)
        labels = rendered.source.labels
        if labels is None:
            raise RenderError("E57 center render has no evaluation labels")
        ranges = np.linalg.norm(
            np.asarray(rendered.source.xyzi[:, :3], dtype=np.float32), axis=1
        )
        valid = (
            ~np.asarray(rendered.source.zero_slot_mask, dtype=np.bool_)
            & (ranges >= 2.5)
            & (ranges <= 50.0)
            & (np.asarray(labels.semantic) != 0)
        )
        anomaly_points = int(np.count_nonzero(valid & (labels.semantic == 2)))
        normal_points = int(np.count_nonzero(valid & (labels.semantic != 2)))
        evaluable = anomaly_points >= 5 and normal_points > 0
        world_json = world.to_json()
        report_json = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        diagnostics_json = json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if (
            WorldSpec.from_dict(json.loads(world_json)).to_json() != world_json
            or json.dumps(
                json.loads(report_json),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) != report_json
        ):
            raise RenderError("E57 canonical world or report round trip changed")
        candidate_hash = hashlib.sha256(
            (
                FROZEN_E57_SOURCE_BANK_ARRAY_SHA256
                + f":{index}:"
                + world_json
                + report_json
                + diagnostics_json
            ).encode("utf-8")
        ).hexdigest()
        return {
            "index": index,
            "bank_seed": bank_seed,
            "center_frame": center,
            "eligible": bool(visible and evaluable),
            "reason": "" if visible and evaluable else (
                "not_evaluable" if visible else "label_not_visible"
            ),
            "minimum_pair_sdf_m": minimum_pair_sdf,
            "descriptor": descriptor,
            "anomaly_points": anomaly_points,
            "normal_points": normal_points,
            "world_json": world_json,
            "report_json": report_json,
            "diagnostics_json": diagnostics_json,
            "candidate_hash": candidate_hash,
        }
    except Exception as error:
        return {
            "index": index,
            "bank_seed": int(_E57_BANK["bank_seed"][index]),
            "center_frame": int(_E57_BANK["center_frame"][index]),
            "eligible": False,
            "reason": f"hard_error:{type(error).__name__}:{error}",
            "hard_error": 1,
        }


def _e57_rank_coordinates(
    descriptor: np.ndarray, candidate_hash: np.ndarray,
) -> np.ndarray:
    """Map each generator descriptor to a deterministic empirical rank."""

    values = np.asarray(descriptor, dtype=np.float64)
    hashes = np.asarray(candidate_hash)
    if values.ndim != 2 or hashes.shape != (values.shape[0],):
        raise RenderError("E57 descriptor and hash arrays are not aligned")
    if values.shape[0] < 24 or not np.isfinite(values).all():
        raise RenderError("E57 needs at least 24 finite eligible descriptors")
    coordinates = np.empty_like(values)
    denominator = max(values.shape[0] - 1, 1)
    for column in range(values.shape[1]):
        order = np.lexsort((hashes, values[:, column]))
        coordinates[order, column] = np.arange(values.shape[0]) / denominator
    return coordinates


def _e57_select(
    descriptor: np.ndarray, candidate_hash: np.ndarray, count: int = 24,
) -> np.ndarray:
    """Choose a model-independent maximin span without cell-count targets."""

    coordinates = _e57_rank_coordinates(descriptor, candidate_hash)
    hashes = np.asarray(candidate_hash)
    if type(count) is not int or not 1 <= count <= coordinates.shape[0]:
        raise RenderError("E57 selected-world count is invalid")
    centrality = np.sum(np.square(coordinates - 0.5), axis=1)
    first_order = np.lexsort((hashes, centrality))
    selected = [int(first_order[0])]
    available = np.ones(coordinates.shape[0], dtype=np.bool_)
    available[selected[0]] = False
    minimum_distance = np.sum(
        np.square(coordinates - coordinates[selected[0]]), axis=1
    )
    while len(selected) < count:
        candidates = np.flatnonzero(available)
        best_distance = float(np.max(minimum_distance[candidates]))
        tied = candidates[np.isclose(
            minimum_distance[candidates], best_distance, rtol=0.0, atol=1.0e-15
        )]
        chosen = int(tied[np.argsort(hashes[tied], kind="stable")[0]])
        selected.append(chosen)
        available[chosen] = False
        distance = np.sum(np.square(coordinates - coordinates[chosen]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return np.asarray(selected, dtype=np.int64)


def _e57_characterization(descriptor: np.ndarray) -> dict[str, object]:
    """Report the retained d/Nvis/O/V bins without turning them into gates."""

    values = np.asarray(descriptor, dtype=np.float64).reshape(-1, 2, 4)
    output: dict[str, object] = {}
    for label, source in (("normal_control", 0), ("anomaly_proxy", 1)):
        current = values[:, source]
        output[label] = {
            "Nvis": np.bincount(
                np.searchsorted((8.0, 32.0, 128.0), current[:, 0], side="right"),
                minlength=4,
            ).tolist(),
            "occlusion": np.bincount(
                np.searchsorted((0.25, 0.50, 0.75), current[:, 1], side="right"),
                minlength=4,
            ).tolist(),
            "distance": np.bincount(
                np.searchsorted((10.0, 20.0, 30.0), current[:, 2], side="right"),
                minlength=4,
            ).tolist(),
            "V": np.bincount(current[:, 3].astype(np.int64), minlength=6)[1:6].tolist(),
        }
    return output


def run_e57_qualification(
    data_root: Path | str,
    protocol_path: Path | str,
    source_bank_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Freeze 24 legal, evaluable and model-independent development worlds."""

    if processes != 24:
        raise RenderError("formal E57 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    source_path = Path(source_bank_path).expanduser().resolve(strict=True)
    if _sha256_path(source_path) != FROZEN_E57_SOURCE_BANK_SHA256:
        raise RenderError("E57 source bank identity changed")
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        arrays = {
            name: np.asarray(source[name])
            for name in source.files if name != "metadata_json"
        }
    if (
        metadata.get("experiment") != "E45B-v2-pair-bank"
        or metadata.get("passed") is not True
        or metadata.get("capacity") != 1024
        or metadata.get("scientific_array_hash")
        != FROZEN_E57_SOURCE_BANK_ARRAY_SHA256
        or _scientific_array_hash(arrays) != FROZEN_E57_SOURCE_BANK_ARRAY_SHA256
    ):
        raise RenderError("E57 source bank metadata or arrays changed")
    calibration = Path(calibration_path).expanduser().resolve(strict=True)
    if _sha256_path(calibration) != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise RenderError("E57 calibration identity changed")
    grid, sensor = load_sensor_calibration(calibration)
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    global _E57_SEQUENCE, _E57_GRID, _E57_SENSOR, _E57_BANK
    _E57_SEQUENCE, _E57_GRID, _E57_SENSOR, _E57_BANK = (
        sequence, grid, sensor, arrays
    )
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        records = workers.map(_e57_candidate_worker, range(1024), chunksize=1)
    elapsed = time.monotonic() - started
    records.sort(key=lambda item: int(item["index"]))
    hard_errors = sum(int(item.get("hard_error", 0)) for item in records)
    if hard_errors:
        examples = [str(item["reason"]) for item in records if item.get("hard_error")][:3]
        raise RenderError(f"E57 candidate implementation errors: {examples}")
    eligible = np.asarray([bool(item["eligible"]) for item in records])
    eligible_index = np.flatnonzero(eligible)
    descriptor = (
        np.stack([records[index]["descriptor"] for index in eligible_index])
        if eligible_index.size
        else np.empty((0, 8), dtype=np.float64)
    )
    candidate_hash = np.asarray(
        [records[index]["candidate_hash"] for index in eligible_index], dtype="S64"
    )
    local_selection = (
        _e57_select(descriptor, candidate_hash, 24)
        if eligible_index.size >= 24
        else np.empty(0, dtype=np.int64)
    )
    repeated_selection = (
        _e57_select(descriptor, candidate_hash, 24)
        if eligible_index.size >= 24
        else np.empty(0, dtype=np.int64)
    )
    reproduction_errors = int(not np.array_equal(local_selection, repeated_selection))
    selected_index = eligible_index[local_selection]
    selected_descriptor = descriptor[local_selection]
    selected_records = [records[index] for index in selected_index]
    control_multiframe_worlds = int(np.count_nonzero(selected_descriptor[:, 3] >= 2))
    proxy_multiframe_worlds = int(np.count_nonzero(selected_descriptor[:, 7] >= 2))
    qualification_errors = int(
        selected_index.size != 24
        or control_multiframe_worlds < 12
        or proxy_multiframe_worlds < 12
        or any(int(item["anomaly_points"]) < 5 for item in selected_records)
        or any(int(item["normal_points"]) < 1 for item in selected_records)
    )
    selected_hash = hashlib.sha256(b"".join(
        np.asarray([item["candidate_hash"] for item in selected_records], dtype="S64")
    )).hexdigest()
    saved = {
        "candidate_bank_index": np.arange(1024, dtype=np.int32),
        "candidate_bank_seed": np.asarray(
            [item["bank_seed"] for item in records], dtype=np.int64
        ),
        "candidate_center_frame": np.asarray(
            [item["center_frame"] for item in records], dtype=np.int16
        ),
        "candidate_eligible": eligible,
        "candidate_rejection_reason": np.asarray(
            [item["reason"] for item in records], dtype="U64"
        ),
        "eligible_bank_index": eligible_index.astype(np.int32),
        "eligible_descriptor": descriptor,
        "eligible_candidate_sha256": candidate_hash,
        "selected_bank_index": selected_index.astype(np.int32),
        "selected_descriptor": selected_descriptor,
        "selected_candidate_sha256": np.asarray(
            [item["candidate_hash"] for item in selected_records], dtype="S64"
        ),
        "selected_center_frame": np.asarray(
            [item["center_frame"] for item in selected_records], dtype=np.int16
        ),
        "selected_world_id": np.arange(selected_index.size, dtype=np.int16),
        "selected_object_id": np.tile(
            np.asarray([[1, 2]], dtype=np.int16), (selected_index.size, 1)
        ),
        "selected_frame_id": np.asarray(
            [
                [int(item["center_frame"]) + offset for offset in (-2, -1, 0, 1, 2)]
                for item in selected_records
            ],
            dtype=np.int16,
        ),
        "selected_anomaly_points": np.asarray(
            [item["anomaly_points"] for item in selected_records], dtype=np.int32
        ),
        "selected_normal_points": np.asarray(
            [item["normal_points"] for item in selected_records], dtype=np.int32
        ),
        "selected_world_json": np.asarray(
            [item["world_json"] for item in selected_records]
        ),
        "selected_report_json": np.asarray(
            [item["report_json"] for item in selected_records]
        ),
        "selected_diagnostics_json": np.asarray(
            [item["diagnostics_json"] for item in selected_records]
        ),
    }
    reason_count = {
        reason: int(np.count_nonzero(saved["candidate_rejection_reason"] == reason))
        for reason in np.unique(saved["candidate_rejection_reason"])
        if reason
    }
    passed = qualification_errors == 0 and reproduction_errors == 0
    result = {
        "experiment": "E57-v2",
        "passed": passed,
        "failure_classification": None if passed else "development_testbed_non_degeneracy_failure",
        "source_bank": str(source_path),
        "source_bank_sha256": FROZEN_E57_SOURCE_BANK_SHA256,
        "source_bank_scientific_array_sha256": FROZEN_E57_SOURCE_BANK_ARRAY_SHA256,
        "source_bank_use": "raw generator worlds and placement records only",
        "e45b_matching_or_e48_scores_read": False,
        "candidate_worlds": 1024,
        "eligible_candidate_worlds": int(eligible_index.size),
        "candidate_rejection_count": reason_count,
        "selected_worlds": int(selected_index.size),
        "selected_world_sha256": selected_hash,
        "selection_rule": "rank_normalized_generator_descriptors_center_then_greedy_maximin_hash_tie",
        "selection_descriptors": [
            "control_Nvis", "control_O", "control_d", "control_V",
            "proxy_Nvis", "proxy_O", "proxy_d", "proxy_V",
        ],
        "selection_reproduction_errors": reproduction_errors,
        "control_multiframe_worlds": control_multiframe_worlds,
        "proxy_multiframe_worlds": proxy_multiframe_worlds,
        "minimum_multiframe_worlds_per_label": 12,
        "minimum_center_anomaly_points": (
            int(np.min(saved["selected_anomaly_points"]))
            if selected_index.size else 0
        ),
        "minimum_center_normal_points": (
            int(np.min(saved["selected_normal_points"]))
            if selected_index.size else 0
        ),
        "qualification_errors": qualification_errors,
        "saved_identity_domain": (
            "selected_world_id x selected_object_id x selected_frame_id x "
            "calibration-bound canonical (beam_id,azimuth_column) ray"
        ),
        "descriptive_characterization": _e57_characterization(selected_descriptor),
        "descriptive_bins_are_nonblocking": True,
        "processes": processes,
        "elapsed_seconds": elapsed,
        "protocol_sha256": _sha256_path(protocol_file),
        "scientific_array_hash": _scientific_array_hash(saved),
        "claim_limit": (
            "E57 freezes a model-independent and non-degenerate 24-world development "
            "testbed; d, Nvis, occlusion and V strata are descriptive and do not "
            "establish model utility or balanced synthetic coverage."
        ),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **saved,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


_E58_SEQUENCE: object | None = None
_E58_GRID: RayGrid | None = None
_E58_SENSOR: SensorCalibration | None = None
_E58_BASE: dict[str, np.ndarray] = {}


def _e58_seed(candidate_hash: str) -> int:
    payload = f"{E58_TORUS_NAMESPACE}:{candidate_hash}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _e58_replacement_world(
    source_world: WorldSpec, source_hash: str,
) -> tuple[WorldSpec, int]:
    """Replace only proxy geometry while preserving renderer RNG identity."""

    if (
        source_world.source_sequence_id != 201
        or source_world.world_type != "mixed"
        or tuple(item.object_id for item in source_world.objects) != (1, 2)
        or tuple(item.label for item in source_world.objects)
        != ("normal-control", "anomaly-proxy")
    ):
        raise RenderError("E58 source world identity changed")
    control, source_proxy = source_world.objects
    if not isinstance(source_proxy.shape, ShapeSpec):
        raise RenderError("E58 source proxy is not in-generator geometry")
    torus_seed = _e58_seed(source_hash)
    source_diameter = float(np.clip(
        2.0 * source_proxy.shape.bound_radius_m, 0.4, 3.0
    ))
    torus = sample_held_out_anomaly_shape(
        torus_seed, size_m_range=(source_diameter, source_diameter)
    )
    rotation = np.asarray(
        source_proxy.rotation_world_from_local, dtype=np.float64
    )
    old_lower = source_proxy.shape.minimum_z_m(
        xy_resolution=33, z_steps=129
    )
    new_lower = torus.minimum_z_m()
    translation = np.asarray(
        source_proxy.translation_world_m, dtype=np.float64
    ) + rotation[:, 2] * (old_lower - new_lower)
    proxy = replace(
        source_proxy,
        shape=torus,
        translation_world_m=tuple(map(float, translation)),
        shape_generation_report=None,
    )
    return WorldSpec(
        source_world.seed,
        source_world.source_sequence_id,
        (control, proxy),
        source_world.tie_tolerance_m,
    ), torus_seed


def _e58_candidate_worker(index: int) -> dict[str, object]:
    """Replace one frozen proxy by a score-blind held-out torus."""

    sequence, grid, sensor = _E58_SEQUENCE, _E58_GRID, _E58_SENSOR
    if sequence is None or grid is None or sensor is None or not _E58_BASE:
        raise RuntimeError("E58 candidate fixtures are not initialized")
    try:
        source_hash = bytes(_E58_BASE["selected_candidate_sha256"][index]).decode()
        source_world = WorldSpec.from_dict(
            json.loads(str(_E58_BASE["selected_world_json"][index]))
        )
        center = int(_E58_BASE["selected_center_frame"][index])
        world, torus_seed = _e58_replacement_world(source_world, source_hash)
        control, proxy = world.objects
        semantic_identity_errors = int(
            world.seed != source_world.seed
            or world.source_sequence_id != source_world.source_sequence_id
            or world.tie_tolerance_m != source_world.tie_tolerance_m
            or control.to_dict() != source_world.objects[0].to_dict()
            or proxy.object_id != source_world.objects[1].object_id
            or proxy.label != source_world.objects[1].label
            or proxy.material != source_world.objects[1].material
            or proxy.rotation_world_from_local
            != source_world.objects[1].rotation_world_from_local
            or not isinstance(proxy.shape, HeldOutTorusShape)
            or proxy.shape_generation_report is not None
        )
        slots = np.arange(grid.directions_sensor.shape[0], dtype=np.int64)
        sensor_stream_errors = 0
        for channel in (0, 1):
            for object_id in (1, 2):
                object_ids = np.full(slots.size, object_id, dtype=np.int32)
                sensor_stream_errors += int(not np.array_equal(
                    _slot_uniform(
                        source_world, center, slots, object_ids, channel=channel
                    ),
                    _slot_uniform(world, center, slots, object_ids, channel=channel),
                ))
        cache_identity_errors = int(world.identity == source_world.identity)
        penetration, minimum_pair_sdf = obvious_pair_penetration(control, proxy)
        frames = tuple(
            sequence.source_frame(frame_id)
            for frame_id in range(center - 2, center + 3)
        )
        diagnostics = five_frame_world_diagnostics(world, frames, grid, sensor)
        objects = diagnostics.get("objects")
        if not isinstance(objects, list) or len(objects) != 2:
            raise RenderError("E58 diagnostics changed structure")
        by_id = {int(item["object_id"]): item for item in objects}
        descriptor = np.asarray(
            [
                by_id[1]["Nvis"], by_id[1]["O"], by_id[1]["d"], by_id[1]["V"],
                by_id[2]["Nvis"], by_id[2]["O"], by_id[2]["d"], by_id[2]["V"],
            ],
            dtype=np.float64,
        )
        visible = (
            np.isfinite(descriptor).all()
            and descriptor[0] > 0.0
            and descriptor[4] > 0.0
        )
        rendered = render_frame(frames[2], world, grid, sensor)
        labels = rendered.source.labels
        if labels is None:
            raise RenderError("E58 center render has no labels")
        ranges = np.linalg.norm(
            np.asarray(rendered.source.xyzi[:, :3], dtype=np.float32), axis=1
        )
        valid = (
            ~np.asarray(rendered.source.zero_slot_mask, dtype=np.bool_)
            & (ranges >= 2.5)
            & (ranges <= 50.0)
            & (np.asarray(labels.semantic) != 0)
        )
        anomaly_points = int(np.count_nonzero(valid & (labels.semantic == 2)))
        normal_points = int(np.count_nonzero(valid & (labels.semantic != 2)))
        world_json = world.to_json()
        diagnostics_json = json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        identity = hashlib.sha256(
            (
                FROZEN_E57_ARTIFACT_SHA256
                + f":{index}:{source_hash}:{torus_seed}:"
                + world_json
                + diagnostics_json
            ).encode("utf-8")
        ).hexdigest()
        training_shape = sample_training_anomaly_shape(torus_seed)
        training_sampler_torus_error = int(
            isinstance(training_shape, HeldOutTorusShape)
            or not isinstance(training_shape, ShapeSpec)
        )
        eligible = (
            not penetration
            and visible
            and anomaly_points >= 5
            and normal_points >= 1
        )
        return {
            "index": index,
            "source_candidate_hash": source_hash,
            "center_frame": center,
            "torus_seed": torus_seed,
            "eligible": bool(eligible),
            "reason": "" if eligible else (
                "pair_penetration" if penetration else
                "not_visible" if not visible else "not_evaluable"
            ),
            "minimum_pair_sdf_m": minimum_pair_sdf,
            "descriptor": descriptor,
            "anomaly_points": anomaly_points,
            "normal_points": normal_points,
            "world_json": world_json,
            "diagnostics_json": diagnostics_json,
            "identity": identity,
            "training_sampler_torus_error": training_sampler_torus_error,
            "semantic_identity_errors": semantic_identity_errors,
            "sensor_stream_errors": sensor_stream_errors,
            "cache_identity_errors": cache_identity_errors,
        }
    except Exception as error:
        return {
            "index": index,
            "eligible": False,
            "reason": f"hard_error:{type(error).__name__}:{error}",
            "hard_error": 1,
            "training_sampler_torus_error": 1,
            "semantic_identity_errors": 1,
            "sensor_stream_errors": 1,
            "cache_identity_errors": 1,
        }


def _e58_selected_indices(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    eligible = [item for item in records if item.get("eligible") is True]
    ordered = sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            f"{E58_TORUS_NAMESPACE}:select:{item['identity']}".encode("ascii")
        ).digest(),
    )
    return np.asarray([int(item["index"]) for item in ordered[:6]], dtype=np.int16)


def run_e58_qualification(
    data_root: Path | str,
    protocol_path: Path | str,
    e57_path: Path | str,
    calibration_path: Path | str,
    output_path: Path | str,
    *,
    processes: int = 24,
) -> dict[str, object]:
    """Freeze six torus diagnostics and prove selection-path isolation."""

    if processes != 24:
        raise RenderError("formal E58 requires exactly 24 worker processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol_file = Path(protocol_path).expanduser().resolve(strict=True)
    protocol = load_protocol(protocol_file)
    if (
        protocol.development["held_out_affects_selection"] is not False
        or protocol.development["checkpoint_selection"][
            "held_out_input_forbidden"
        ] is not True
    ):
        raise RenderError("E58 held-out selection isolation changed")
    e57_file = Path(e57_path).expanduser().resolve(strict=True)
    if _sha256_path(e57_file) != FROZEN_E57_ARTIFACT_SHA256:
        raise RenderError("E58 E57-v2 input identity changed")
    with np.load(e57_file, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        arrays = {
            name: np.asarray(source[name])
            for name in source.files if name != "metadata_json"
        }
    if (
        metadata.get("experiment") != "E57-v2"
        or metadata.get("passed") is not True
        or metadata.get("scientific_array_hash") != _scientific_array_hash(arrays)
        or arrays.get("selected_world_json", np.empty(0)).shape != (24,)
    ):
        raise RenderError("E58 received invalid E57-v2 evidence")
    calibration = Path(calibration_path).expanduser().resolve(strict=True)
    if _sha256_path(calibration) != FROZEN_SENSOR_CALIBRATION_SHA256:
        raise RenderError("E58 calibration identity changed")
    grid, sensor = load_sensor_calibration(calibration)
    sequence = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="train",
        sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    global _E58_SEQUENCE, _E58_GRID, _E58_SENSOR, _E58_BASE
    _E58_SEQUENCE, _E58_GRID, _E58_SENSOR, _E58_BASE = (
        sequence, grid, sensor, arrays
    )
    started = time.monotonic()
    with mp.get_context("fork").Pool(processes=processes) as workers:
        records = workers.map(_e58_candidate_worker, range(24), chunksize=1)
    elapsed = time.monotonic() - started
    records.sort(key=lambda item: int(item["index"]))
    hard_errors = sum(int(item.get("hard_error", 0)) for item in records)
    if hard_errors:
        examples = [str(item["reason"]) for item in records if item.get("hard_error")][:3]
        raise RenderError(f"E58 candidate implementation errors: {examples}")
    first_selection = _e58_selected_indices(records)
    second_selection = _e58_selected_indices(records)
    selection_errors = int(not np.array_equal(first_selection, second_selection))
    selected = [records[int(index)] for index in first_selection]
    isolation_errors = sum(
        int(item["training_sampler_torus_error"]) for item in records
    )
    semantic_identity_errors = sum(
        int(item["semantic_identity_errors"]) for item in records
    )
    sensor_stream_errors = sum(
        int(item["sensor_stream_errors"]) for item in records
    )
    cache_identity_errors = sum(
        int(item["cache_identity_errors"]) for item in records
    )
    qualification_errors = int(len(selected) != 6)
    saved = {
        "candidate_e57_index": np.arange(24, dtype=np.int16),
        "candidate_eligible": np.asarray(
            [item["eligible"] for item in records], dtype=np.bool_
        ),
        "candidate_rejection_reason": np.asarray(
            [item["reason"] for item in records], dtype="U64"
        ),
        "candidate_identity_sha256": np.asarray(
            [item["identity"] for item in records], dtype="S64"
        ),
        "selected_e57_index": first_selection,
        "selected_world_id": np.arange(24, 30, dtype=np.int16)[:len(selected)],
        "selected_center_frame": np.asarray(
            [item["center_frame"] for item in selected], dtype=np.int16
        ),
        "selected_frame_id": np.asarray(
            [
                [int(item["center_frame"]) + offset for offset in (-2, -1, 0, 1, 2)]
                for item in selected
            ],
            dtype=np.int16,
        ).reshape(len(selected), 5),
        "selected_object_id": np.tile(
            np.asarray([[1, 2]], dtype=np.int16), (len(selected), 1)
        ),
        "selected_torus_seed": np.asarray(
            [item["torus_seed"] for item in selected], dtype=np.uint32
        ),
        "selected_descriptor": np.asarray(
            [item["descriptor"] for item in selected], dtype=np.float64
        ).reshape(len(selected), 8),
        "selected_anomaly_points": np.asarray(
            [item["anomaly_points"] for item in selected], dtype=np.int32
        ),
        "selected_normal_points": np.asarray(
            [item["normal_points"] for item in selected], dtype=np.int32
        ),
        "selected_world_json": np.asarray(
            [item["world_json"] for item in selected]
        ),
        "selected_diagnostics_json": np.asarray(
            [item["diagnostics_json"] for item in selected]
        ),
        "selected_identity_sha256": np.asarray(
            [item["identity"] for item in selected], dtype="S64"
        ),
    }
    passed = (
        qualification_errors == 0
        and selection_errors == 0
        and isolation_errors == 0
        and semantic_identity_errors == 0
        and sensor_stream_errors == 0
        and cache_identity_errors == 0
    )
    result = {
        "experiment": "E58",
        "passed": passed,
        "failure_classification": None if passed else "held_out_identity_or_isolation_failure",
        "e57_artifact_sha256": FROZEN_E57_ARTIFACT_SHA256,
        "e57_scientific_array_hash": metadata["scientific_array_hash"],
        "candidate_worlds": 24,
        "eligible_candidate_worlds": int(np.count_nonzero(saved["candidate_eligible"])),
        "selected_worlds": len(selected),
        "selection_namespace": E58_TORUS_NAMESPACE,
        "selection_reproduction_errors": selection_errors,
        "training_sampler_torus_errors": isolation_errors,
        "semantic_identity_errors": semantic_identity_errors,
        "sensor_stream_errors": sensor_stream_errors,
        "cache_identity_errors": cache_identity_errors,
        "held_out_affects_checkpoint_or_threshold": False,
        "model_scores_read": False,
        "qualification_errors": qualification_errors,
        "minimum_center_anomaly_points": (
            int(np.min(saved["selected_anomaly_points"])) if selected else 0
        ),
        "minimum_center_normal_points": (
            int(np.min(saved["selected_normal_points"])) if selected else 0
        ),
        "processes": processes,
        "elapsed_seconds": elapsed,
        "protocol_sha256": _sha256_path(protocol_file),
        "scientific_array_hash": _scientific_array_hash(saved),
        "claim_limit": (
            "E58 proves fixed held-out torus identity and exclusion from training, "
            "checkpoint, threshold and PASS paths; it makes no model-quality claim."
        ),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **saved,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, destination)
    return result


def _phase6_characterization_arrays(
    descriptor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Aggregate frozen E57 descriptors without rerendering or reselection."""

    values = np.asarray(descriptor, dtype=np.float64)
    if values.shape != (24, 8) or not np.isfinite(values).all():
        raise RenderError("E59/E60 require the finite 24-by-8 E57 descriptor")
    shaped = values.reshape(24, 2, 4)
    nvis, occlusion, distance, visible_frames = np.moveaxis(shaped, 2, 0)
    if (
        np.any(nvis < 1.0)
        or np.any((occlusion < 0.0) | (occlusion > 1.0))
        or np.any((distance < 2.5) | (distance > 50.0))
        or np.any((visible_frames < 1.0) | (visible_frames > 5.0))
        or not np.array_equal(visible_frames, np.floor(visible_frames))
    ):
        raise RenderError("E59/E60 E57 descriptors are outside frozen domains")
    distance_bin = np.digitize(distance, (10.0, 20.0, 30.0)).astype(np.int8)
    nvis_bin = np.digitize(nvis, (8.0, 32.0, 128.0)).astype(np.int8)
    occlusion_bin = np.digitize(
        occlusion, (0.25, 0.50, 0.75)
    ).astype(np.int8)
    distance_count = np.stack([
        np.bincount(distance_bin[:, label], minlength=4)
        for label in range(2)
    ]).astype(np.int16)
    nvis_count = np.stack([
        np.bincount(nvis_bin[:, label], minlength=4)
        for label in range(2)
    ]).astype(np.int16)
    occlusion_count = np.stack([
        np.bincount(occlusion_bin[:, label], minlength=4)
        for label in range(2)
    ]).astype(np.int16)
    v_count = np.stack([
        np.bincount(
            visible_frames[:, label].astype(np.int64), minlength=6
        )[1:6]
        for label in range(2)
    ]).astype(np.int16)
    common = {
        "Nvis": nvis,
        "occlusion": occlusion,
        "distance_m": distance,
        "visible_frames": visible_frames.astype(np.int8),
    }
    e59 = {
        **common,
        "distance_bin": distance_bin,
        "Nvis_bin": nvis_bin,
        "occlusion_bin": occlusion_bin,
        "distance_count": distance_count,
        "Nvis_count": nvis_count,
        "occlusion_count": occlusion_count,
    }
    e60 = {**common, "visible_frame_count": v_count}
    return e59, e60


def run_e59_e60_characterization(
    e57_path: Path | str,
    e59_output_path: Path | str,
    e60_output_path: Path | str,
) -> dict[str, object]:
    """Write both nonblocking Phase-6 characterizations from one E57 read."""

    source_path = Path(e57_path).expanduser().resolve(strict=True)
    source_hash = _sha256_path(source_path)
    if source_hash != FROZEN_E57_ARTIFACT_SHA256:
        raise RenderError("E59/E60 E57-v2 input identity changed")
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        source_arrays = {
            name: np.asarray(source[name])
            for name in source.files if name != "metadata_json"
        }
    if (
        metadata.get("experiment") != "E57-v2"
        or metadata.get("passed") is not True
        or metadata.get("scientific_array_hash")
        != _scientific_array_hash(source_arrays)
    ):
        raise RenderError("E59/E60 received invalid E57-v2 evidence")
    e59, e60 = _phase6_characterization_arrays(
        source_arrays["selected_descriptor"]
    )
    identity = {
        "world_id": np.asarray(source_arrays["selected_world_id"], dtype=np.int16),
        "center_frame": np.asarray(
            source_arrays["selected_center_frame"], dtype=np.int16
        ),
        "object_id": np.asarray(
            source_arrays["selected_object_id"], dtype=np.int16
        ),
        "candidate_sha256": np.asarray(
            source_arrays["selected_candidate_sha256"], dtype="S64"
        ),
        "label": np.asarray(("normal-control", "anomaly-proxy"), dtype="U16"),
    }
    e59 = {**identity, **e59}
    e60 = {**identity, **e60}
    if (
        not all(np.sum(e59[name], axis=1).tolist() == [24, 24]
                for name in ("distance_count", "Nvis_count", "occlusion_count"))
        or np.sum(e60["visible_frame_count"], axis=1).tolist() != [24, 24]
    ):
        raise RenderError("E59/E60 count conservation failed")
    shared = {
        "e57_artifact_sha256": source_hash,
        "e57_scientific_array_hash": metadata["scientific_array_hash"],
        "worlds": 24,
        "units_per_label": 24,
        "labels": ["normal-control", "anomaly-proxy"],
        "rerendered_worlds": 0,
        "model_scores_read": False,
        "scientific_fail_verdict": False,
        "claim_limit": (
            "Descriptive marginal support only; sparse bins limit interpretation "
            "but cannot block E61 or training."
        ),
    }
    e59_result = {
        "experiment": "E59",
        "completed": True,
        **shared,
        "distance_count": e59["distance_count"].tolist(),
        "Nvis_count": e59["Nvis_count"].tolist(),
        "occlusion_count": e59["occlusion_count"].tolist(),
        "scientific_array_hash": _scientific_array_hash(e59),
    }
    e60_result = {
        "experiment": "E60",
        "completed": True,
        **shared,
        "visible_frame_count": e60["visible_frame_count"].tolist(),
        "scientific_array_hash": _scientific_array_hash(e60),
    }
    destinations = (
        Path(e59_output_path).expanduser().resolve(),
        Path(e60_output_path).expanduser().resolve(),
    )
    temporaries = tuple(
        path.with_suffix(path.suffix + ".tmp.npz") for path in destinations
    )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    for temporary, arrays, result in zip(
        temporaries, (e59, e60), (e59_result, e60_result), strict=True
    ):
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json=np.asarray(
                json.dumps(result, sort_keys=True, separators=(",", ":"))
            ),
        )
    for temporary, destination in zip(temporaries, destinations, strict=True):
        os.replace(temporary, destination)
    return {"E59": e59_result, "E60": e60_result}


_E45A2_TARGET_UNITS: dict[str, np.ndarray] = {}
_E45A2_SUPPORT_ROWS: tuple[np.ndarray, ...] = ()


def _e45a2_real_targets(units: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Freeze the unique valid real units already exposed by E45-v2."""
    flat = {
        name: np.asarray(value).reshape((-1,) + value.shape[2:])
        for name, value in units.items()
    }
    covariates = _e45_covariates(flat)
    valid = (
        (flat["source"] == 0) & (flat["point_count"] > 0)
        & (flat["range_bin"] >= 0) & (flat["range_bin"] < 4)
        & (flat["azimuth_sector"] >= 0) & np.isfinite(covariates).all(axis=1)
    )
    candidates = np.flatnonzero(valid)
    candidates = candidates[np.argsort(flat["unit_hash"][candidates], kind="stable")]
    _, first = np.unique(flat["unit_hash"][candidates], return_index=True)
    selected = candidates[np.sort(first)]
    return {name: flat[name][selected] for name in _E45_UNIT_FIELDS}


def _e45a2_support_streams(
    sequence: object, pool: QualifiedSupportPool,
    targets: Mapping[str, np.ndarray], maximum_proposals: int,
) -> tuple[np.ndarray, ...]:
    """Order audit placements by frozen target range, sector, and support."""
    streams: list[np.ndarray] = []
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for target in range(targets["source"].size):
        frame_id = int(targets["frame_id"][target])
        cached = frame_cache.get(frame_id)
        if cached is None:
            rows = np.flatnonzero(np.abs(pool.frames - frame_id) <= 2)
            frame = sequence.source_frame(frame_id)
            rotation, translation = _pose(frame)
            sensor = (pool.anchors_world_m[rows] - translation) @ rotation
            ranges = np.linalg.norm(sensor, axis=1)
            sectors = (
                np.floor(
                    (np.arctan2(sensor[:, 1], sensor[:, 0]) % (2.0 * math.pi))
                    / (math.pi / 4.0)
                ).astype(np.int8) % 8
            )
            cached = rows, ranges, sectors
            frame_cache[frame_id] = cached
        rows, ranges, sectors = cached
        eligible = (
            (pool.semantics[rows] == targets["support_semantic"][target])
            & (_gate1_range_bin(ranges) == targets["range_bin"][target])
            & (sectors == targets["azimuth_sector"][target])
        )
        selected = rows[eligible]
        distance_error = np.abs(
            ranges[eligible] - float(targets["median_distance_m"][target])
        )
        order = np.lexsort((pool.selection_hashes[selected], distance_error))
        streams.append(selected[order[:maximum_proposals]].astype(np.int64))
    return tuple(streams)


def _e45a2_worker(task: tuple[int, int]) -> dict[str, object]:
    target, proposal = task
    pool, obstacles = _GATE1_POOL, _GATE1_OBSTACLES
    sequence, grid, sensor = _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR
    if (
        pool is None or obstacles is None or sequence is None or grid is None
        or sensor is None or not _E45A2_TARGET_UNITS or not _E45A2_SUPPORT_ROWS
    ):
        raise RuntimeError("E45A-v2 targeted fixtures are not initialized")
    if proposal >= _E45A2_SUPPORT_ROWS[target].size:
        return {"target": target, "proposal": proposal, "status": 0}
    row = int(_E45A2_SUPPORT_ROWS[target][proposal])
    seed = 4_500_000 + 128 * target + proposal
    try:
        support_semantic = int(_E45A2_TARGET_UNITS["support_semantic"][target])
        source_template = _gate1_template(seed + 1, support_semantic)
        scale = np.random.default_rng(
            np.random.SeedSequence([seed + 2, 2501])
        ).uniform(0.9, 1.1, size=3)
        shape = _aligned_scaled_template(source_template, scale)
        limit = math.pi if shape.raw_semantic_id == 30 else math.radians(15.0)
        perturbation = float(
            np.random.default_rng(np.random.SeedSequence([seed + 3, 2502])).uniform(
                -limit, limit
            )
        )

        def yaw_for_support(patch: SupportPatch) -> float:
            return float(_GATE1_TRAJECTORY_YAW[patch.frame_id]) + perturbation

        material_seed = seed + 4
        item, placement = place_object(
            shape, MaterialSpec.sample(material_seed), pool, obstacles,
            object_id=1, label="normal-control",
            proposal_namespace="E45A-v2-targeted-control",
            proposal_stream=seed, yaw_rad=perturbation,
            material_seed=material_seed, yaw_seed=seed + 3,
            template_identity=_normal_template_identity(source_template),
            proposal_rows=(row,), maximum_candidates=1,
            grounding_eligibility=qualify_grounding(shape),
            yaw_for_support=yaw_for_support,
        )
        validation_error = int(
            placement.support_semantic != support_semantic
            or not qualify_grounding(item.shape).passed
            or observed_normal_collision(item, obstacles)[0]
            or placement.accepted_proposal != 0
            or not isinstance(item.shape, NormalTemplateShape)
        )
        if validation_error:
            return {
                "target": target, "proposal": proposal, "status": 6,
                "support_pool_index": placement.support_pool_index,
            }
        frame_id = int(_E45A2_TARGET_UNITS["frame_id"][target])
        center_frame = int(_E45A2_TARGET_UNITS["center_frame"][target])
        frame = sequence.source_frame(frame_id)
        world = WorldSpec(seed, 201, (item,))
        geometry, _, _, rendered = _gate1_single_object_trace(frame, world, grid, sensor)
        returned = np.asarray(rendered.normal_control_mask, dtype=np.bool_)
        official = np.asarray(grid.official_ranges(rendered.source))
        returned = returned & (official >= 2.5) & (official <= 50.0)
        unit = _e45_unit_record(
            frame, grid, 1, seed, center_frame, 0, 0, support_semantic,
            geometry, returned, rendered.source,
        )
        if int(unit["point_count"]) == 0:
            return {
                "target": target, "proposal": proposal, "status": 2,
                "support_pool_index": placement.support_pool_index,
            }
        target_covariates = _e45_covariates({
            name: np.asarray(_E45A2_TARGET_UNITS[name][target]).reshape(1)
            for name in (
                "median_distance_m", "median_beam", "Nvis", "O_hat", "local_density"
            )
        })[0]
        control_covariates = _e45_covariates({
            name: np.asarray(unit[name]).reshape(1)
            for name in (
                "median_distance_m", "median_beam", "Nvis", "O_hat", "local_density"
            )
        })[0]
        exact = (
            int(unit["support_semantic"]) == support_semantic
            and int(unit["range_bin"]) == int(_E45A2_TARGET_UNITS["range_bin"][target])
            and int(unit["azimuth_sector"])
            == int(_E45A2_TARGET_UNITS["azimuth_sector"][target])
        )
        covariate_difference = np.abs(control_covariates - target_covariates)
        if not exact:
            status = 3
        else:
            caliper = np.asarray((2.0, 4.0, 0.25, 0.10, 0.25), dtype=np.float64)
            status = 5 if np.all(covariate_difference <= caliper) else 4
        return {
            "target": target, "proposal": proposal, "status": status,
            "support_pool_index": placement.support_pool_index,
            "covariate_difference": covariate_difference,
            "unit": unit if status == 5 else None,
        }
    except PlacementError:
        return {"target": target, "proposal": proposal, "status": 1, "support_pool_index": row}
    except Exception as error:
        return {
            "target": target, "proposal": proposal, "status": 7,
            "support_pool_index": row,
            "error": f"{type(error).__name__}: {error}",
        }


def _e45a2_candidate_cache(
    results: Sequence[Mapping[str, object]], targets: int, proposal_limit: int,
    output_path: Path, elapsed_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    status = np.zeros((targets, proposal_limit), dtype=np.uint8)
    support_index = np.full((targets, proposal_limit), -1, dtype=np.int64)
    covariate_difference = np.full((targets, proposal_limit, 5), np.nan, dtype=np.float64)
    eligible = [result for result in results if int(result["status"]) == 5]
    for result in results:
        target = int(result["target"])
        proposal = int(result["proposal"])
        status[target, proposal] = int(result["status"])
        support_index[target, proposal] = int(result.get("support_pool_index", -1))
        if "covariate_difference" in result:
            covariate_difference[target, proposal] = result["covariate_difference"]
    units = {
        name: np.stack([result["unit"][name] for result in eligible])
        if eligible else np.empty((0,) + np.asarray(_E45A2_TARGET_UNITS[name][0]).shape,
                                  dtype=np.asarray(_E45A2_TARGET_UNITS[name]).dtype)
        for name in _E45_UNIT_FIELDS
    }
    arrays = {
        **units,
        "eligible_target_index": np.asarray(
            [result["target"] for result in eligible], dtype=np.int32
        ),
        "eligible_proposal_index": np.asarray(
            [result["proposal"] for result in eligible], dtype=np.int16
        ),
        "proposal_status": status,
        "proposal_support_pool_index": support_index,
        "proposal_covariate_difference": covariate_difference,
    }
    status_count = np.bincount(status.reshape(-1), minlength=8).astype(np.int64)
    metadata = {
        "experiment": "E45A-v2-targeted-control-bank",
        "passed": int(status_count[7]) == 0,
        "target_units": targets,
        "proposal_limit": proposal_limit,
        "proposal_status_count_0_to_7": status_count.tolist(),
        "eligible_controls": len(eligible),
        "elapsed_seconds": elapsed_seconds,
        "scientific_array_hash": _scientific_array_hash(arrays),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output_path)
    return arrays, metadata


def run_e45a_v2_qualification(
    data_root: Path | str, support_pool_path: Path | str,
    e25_artifact_path: Path | str, calibration_path: Path | str,
    unit_cache_path: Path | str, output_path: Path | str, *, processes: int = 24,
) -> dict[str, object]:
    """Build an audit-only targeted control bank and requalify E45A."""
    if type(processes) is not int or not 1 <= processes <= 24:
        raise RenderError("formal E45A-v2 processes must lie in [1,24]")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    cache_path = Path(unit_cache_path).expanduser().resolve(strict=True)
    with np.load(cache_path, allow_pickle=False) as source:
        cache_metadata = json.loads(str(source["metadata_json"]))
        if (
            cache_metadata.get("experiment") != "E45-v2-units"
            or cache_metadata.get("passed") is not True
            or int(cache_metadata.get("capacity", -1)) != 2048
            or cache_metadata.get("scientific_array_hash")
            != "a3b63d11107edb1b1dce6c052e188a92879131fe20bec5568db10275c83a6160"
        ):
            raise RenderError("E45A-v2 requires the frozen formal E45-v2 unit cache")
        source_units = {name: np.asarray(source[name]) for name in _E45_UNIT_FIELDS}
    targets = _e45a2_real_targets(source_units)
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    pool, _ = load_gate1_support_pool(support_pool_path)
    grid, sensor = load_sensor_calibration(calibration_path)
    frames = tuple(sequence.source_frame(frame_id) for frame_id in range(4, 682))
    obstacles = collect_observed_obstacle_index(frames, source_sequence_id=201)
    trajectory_yaws = trajectory_yaw_by_frame(frames)
    del frames
    gc.collect()
    _initialize_gate1_candidate_generation(
        sequence, pool, obstacles, _load_e25_templates(e25_artifact_path),
        grid, sensor, trajectory_yaws, (),
    )
    global _E45A2_TARGET_UNITS, _E45A2_SUPPORT_ROWS
    _E45A2_TARGET_UNITS = targets
    _E45A2_SUPPORT_ROWS = _e45a2_support_streams(sequence, pool, targets, 64)
    # Each targeted task reads one frame; retaining the training cache wastes RAM.
    sequence._cache_frames = 1
    sequence._frames.clear()
    gc.collect()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    all_results: list[dict[str, object]] = []
    final_units: dict[str, np.ndarray] | None = None
    final_match: dict[str, np.ndarray] | None = None
    final_reproduced = False
    previous = 0
    for proposal_limit in (4, 8, 16, 32, 64):
        tasks = [
            (target, proposal)
            for proposal in range(previous, proposal_limit)
            for target in range(targets["source"].size)
        ]
        started = time.monotonic()
        with mp.get_context("fork").Pool(
            processes=processes, maxtasksperchild=8
        ) as workers:
            suffix = workers.map(_e45a2_worker, tasks, chunksize=4)
        generation_seconds = time.monotonic() - started
        hard_errors = [result for result in suffix if int(result["status"]) == 7]
        if hard_errors:
            raise RenderError(f"E45A-v2 hard errors: {hard_errors[:3]}")
        all_results.extend(suffix)
        candidate_path = output.parent / f"e45a_v2_targeted_controls_{proposal_limit}.npz"
        candidate_arrays, candidate_metadata = _e45a2_candidate_cache(
            all_results, int(targets["source"].size), proposal_limit,
            candidate_path, generation_seconds,
        )
        control_units = {name: candidate_arrays[name] for name in _E45_UNIT_FIELDS}
        combined = {
            name: np.concatenate((targets[name], control_units[name]), axis=0)[None, ...]
            for name in _E45_UNIT_FIELDS
        }
        matching_started = time.monotonic()
        runs = [_e45_pair_match(combined, 0, 1) for _ in range(2)]
        matching_seconds = time.monotonic() - matching_started
        reproduced = all(
            np.array_equal(runs[0][name], runs[1][name]) for name in runs[0]
        )
        match = runs[0]
        matched_count = int(match["matched_flat_index"].shape[0])
        center_frames = int(match["matched_center_frames"].size)
        range_count = np.asarray(match["matched_range_count"])
        maximum_smd = float(np.max(match["pairwise_smd"]))
        passed = (
            matched_count >= 1024 and center_frames >= 100
            and bool(np.all(range_count > 0))
            and int(match["caliper_errors"]) == 0
            and int(match["duplicate_errors"]) == 0
            and maximum_smd <= 0.10 and reproduced
        )
        history.append({
            "proposal_limit": proposal_limit,
            "eligible_controls": int(candidate_metadata["eligible_controls"]),
            "proposal_status_count_0_to_7": candidate_metadata[
                "proposal_status_count_0_to_7"
            ],
            "matched_pairs": matched_count,
            "center_frames": center_frames,
            "range_count_2p5_to_40": range_count.tolist(),
            "maximum_pairwise_smd": maximum_smd,
            "caliper_errors": int(match["caliper_errors"]),
            "duplicate_errors": int(match["duplicate_errors"]),
            "elementwise_reproduced": reproduced,
            "generation_seconds": generation_seconds,
            "two_run_matching_seconds": matching_seconds,
            "passed": passed,
        })
        final_units, final_match, final_reproduced = combined, match, reproduced
        if passed:
            break
        previous = proposal_limit
    assert final_units is not None and final_match is not None
    final = history[-1]
    passed = bool(final["passed"])
    scientific = _e45_selected_scientific(final_units, final_match)
    result = {
        "experiment": "E45A-v2", "passed": passed,
        "failure_classification": None if passed else "targeted_control_common_support_failure",
        "audit_only_targeted_bank": True,
        "formal_training_distribution_changed": False,
        "source_pair": ["real-normal", "normal-control"],
        "target_units": int(targets["source"].size),
        "target_center_frames": int(np.unique(targets["center_frame"]).size),
        "target_range_count_2p5_to_40": np.bincount(
            targets["range_bin"], minlength=4
        )[:4].tolist(),
        "proposal_ladder": [4, 8, 16, 32, 64],
        "history": history,
        "final_proposal_limit": int(final["proposal_limit"]),
        "matched_pairs": int(final["matched_pairs"]),
        "center_frames": int(final["center_frames"]),
        "matched_range_count_2p5_to_40": final["range_count_2p5_to_40"],
        "pairwise_smd": final_match["pairwise_smd"].tolist(),
        "maximum_pairwise_smd": float(final["maximum_pairwise_smd"]),
        "caliper_errors": int(final["caliper_errors"]),
        "duplicate_errors": int(final["duplicate_errors"]),
        "elementwise_reproduced": final_reproduced,
        "unit_cache_sha256": _sha256_path(cache_path),
        "unit_cache_scientific_array_hash": cache_metadata["scientific_array_hash"],
        "scientific_array_hash": _scientific_array_hash(scientific),
        "claim_limit": (
            "E45A-v2 is an audit-only conditional matching bank. It does not estimate "
            "the random-placement training distribution and does not change E26."
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output)
    return result


def run_e45_v2_qualification(
    data_root: Path | str, support_pool_path: Path | str,
    e25_artifact_path: Path | str, calibration_path: Path | str,
    candidate_bank_256_path: Path | str, e45_v1_artifact_path: Path | str,
    output_path: Path | str, *, processes: int = 24,
) -> dict[str, object]:
    """Build and qualify strict triplets on the observable 2.5--40 m domain."""
    if processes != 24:
        raise RenderError("formal E45-v2 requires exactly 24 processes")
    try:
        from .protocol import load_protocol
        from .scene import LabelMode, STUSequence
    except ImportError:
        from protocol import load_protocol  # type: ignore[no-redef]
        from scene import LabelMode, STUSequence  # type: ignore[no-redef]
    protocol = load_protocol(Path(__file__).resolve().parents[1] / "protocol.json")
    sequence = STUSequence.open(
        data_root, protocol=protocol, partition="train", sequence_id=201,
        label_mode=LabelMode.REQUIRED,
    )
    pool, _ = load_gate1_support_pool(support_pool_path)
    grid, sensor = load_sensor_calibration(calibration_path)
    real_candidates = _e45_v1_real_candidates(e45_v1_artifact_path)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    capacities = (256, 512, 1024, 2048)
    bank_path = Path(candidate_bank_256_path).expanduser().resolve(strict=True)
    previous_units: dict[str, np.ndarray] | None = None
    previous_capacity = 0
    generator_initialized = False
    history: list[dict[str, object]] = []
    final_units: dict[str, np.ndarray] | None = None
    final_match: dict[str, np.ndarray] | None = None
    final_reproduced = False
    for capacity in capacities:
        if capacity > 256:
            target = output.parent / f"gate1_candidate_bank_{capacity}.npz"
            if target.exists():
                _, existing_metadata = _load_e45_bank_with_metadata(target)
                if existing_metadata.get("passed") is not True or int(existing_metadata.get("capacity", -1)) != capacity:
                    raise RenderError("existing expanded Gate 1 bank is invalid")
            else:
                if not generator_initialized:
                    frames = tuple(sequence.source_frame(frame_id) for frame_id in range(4, 682))
                    obstacles = collect_observed_obstacle_index(frames, source_sequence_id=201)
                    _initialize_gate1_candidate_generation(
                        sequence, pool, obstacles, _load_e25_templates(e25_artifact_path),
                        grid, sensor, trajectory_yaw_by_frame(frames), real_candidates,
                    )
                    generator_initialized = True
                extend_gate1_candidate_bank(
                    bank_path, target, capacity, processes=processes
                )
            bank_path = target
        bank, bank_metadata = _load_e45_bank_with_metadata(bank_path)
        if len(bank) != capacity or bank_metadata.get("passed") is not True:
            raise RenderError("E45-v2 candidate bank has the wrong capacity or status")
        global _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR, _E45_BANK
        _GATE1_SEQUENCE, _GATE1_RAY_GRID, _GATE1_SENSOR = sequence, grid, sensor
        _E45_BANK = bank
        unit_path = output.parent / f"e45_v2_units_{capacity}.npz"
        cached_units = _load_e45_unit_cache(
            unit_path, capacity, str(bank_metadata["scientific_array_hash"])
        )
        extraction_seconds = 0.0
        units = None if cached_units is None else cached_units[0]
        if cached_units is not None:
            extraction_seconds = cached_units[1]
        if units is None:
            started = time.monotonic()
            with mp.get_context("fork").Pool(processes=processes) as workers:
                records = workers.map(
                    _e45_worker, range(previous_capacity, capacity), chunksize=1
                )
            suffix = {
                name: np.stack([record[name] for record in records])
                for name in _E45_UNIT_FIELDS
            }
            units = (
                suffix if previous_units is None else {
                    name: np.concatenate((previous_units[name], suffix[name]))
                    for name in _E45_UNIT_FIELDS
                }
            )
            extraction_seconds = time.monotonic() - started
            _write_e45_units(
                units, unit_path, str(bank_metadata["scientific_array_hash"]),
                extraction_seconds,
            )
        matching_started = time.monotonic()
        runs = [_e45_match(units) for _ in range(2)]
        matching_seconds = time.monotonic() - matching_started
        reproduced = all(
            np.array_equal(runs[0][name], runs[1][name]) for name in runs[0]
        )
        match = runs[0]
        matched_count = int(match["matched_flat_index"].shape[0])
        center_frames = int(match["matched_center_frames"].size)
        range_count = match["matched_range_count"]
        maximum_smd = float(np.max(match["pairwise_smd"]))
        caliper_errors = int(match["caliper_errors"])
        duplicate_errors = int(match["duplicate_errors"])
        passed = (
            matched_count >= 1024 and center_frames >= 100
            and bool(np.all(range_count > 0)) and caliper_errors == 0
            and duplicate_errors == 0 and maximum_smd <= 0.10 and reproduced
        )
        history.append({
            "capacity": capacity, "matched_triplets": matched_count,
            "center_frames": center_frames, "range_count_2p5_to_40": range_count.tolist(),
            "maximum_pairwise_smd": maximum_smd,
            "caliper_errors": caliper_errors, "duplicate_errors": duplicate_errors,
            "elementwise_reproduced": reproduced,
            "bank_extension_seconds": (
                0.0 if capacity == 256
                else float(bank_metadata.get("elapsed_seconds", 0.0))
            ),
            "unit_extraction_seconds": extraction_seconds,
            "two_run_matching_seconds": matching_seconds,
            "passed": passed,
        })
        final_units, final_match, final_reproduced = units, match, reproduced
        if passed:
            break
        previous_units, previous_capacity = units, capacity
    assert final_units is not None and final_match is not None
    final = history[-1]
    passed = bool(final["passed"])
    scientific = _e45_selected_scientific(final_units, final_match)
    result = {
        "experiment": "E45-v2", "passed": passed,
        "failure_classification": None if passed else "insufficient_three_source_common_support",
        "estimand_range_m": [2.5, 40.0],
        "range_40_50_status": "unobservable_for_real-vs-rendered-object matching in train/201",
        "capacity_history": history,
        "final_capacity": int(final["capacity"]),
        "matched_triplets": int(final["matched_triplets"]),
        "center_frames": int(final["center_frames"]),
        "matched_range_count_2p5_to_40": final["range_count_2p5_to_40"],
        "pairwise_smd": final_match["pairwise_smd"].tolist(),
        "maximum_pairwise_smd": float(final["maximum_pairwise_smd"]),
        "caliper_errors": int(final["caliper_errors"]),
        "duplicate_errors": int(final["duplicate_errors"]),
        "elementwise_reproduced": final_reproduced,
        "scientific_array_hash": _scientific_array_hash(scientific),
        "claim_limit": (
            "Source-leakage adjudication is limited to strictly matched real-normal "
            "object support in train/201 from 2.5 m through 40 m; 40--50 m has no "
            "real-object matching evidence."
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, **scientific,
        metadata_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"))),
    )
    os.replace(temporary, output)
    return result


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
    e25v3 = subcommands.add_parser("qualify-e25-v3-targets")
    e25v3.add_argument("--data-root", type=Path, required=True)
    e25v3.add_argument("--support-pool", type=Path, required=True)
    e25v3.add_argument("--target-bank", type=Path, required=True)
    e25v3.add_argument("--output", type=Path, required=True)
    e25v3.add_argument("--processes", type=int, default=24)
    e25v3_bank = subcommands.add_parser("build-e25-v3-target-bank")
    e25v3_bank.add_argument("--source-target-bank", type=Path, required=True)
    e25v3_bank.add_argument("--target-qualification", type=Path, required=True)
    e25v3_bank.add_argument("--output", type=Path, required=True)
    e25v3_control = subcommands.add_parser("qualify-e25-v3-normal-control")
    e25v3_control.add_argument("--data-root", type=Path, required=True)
    e25v3_control.add_argument("--support-pool", type=Path, required=True)
    e25v3_control.add_argument("--calibration", type=Path, required=True)
    e25v3_control.add_argument("--target-bank", type=Path, required=True)
    e25v3_control.add_argument("--output", type=Path, required=True)
    e25v3_control.add_argument("--processes", type=int, default=12)
    e25_new = subcommands.add_parser("qualify-e25-new-normal-control")
    e25_new.add_argument("--data-root", type=Path, required=True)
    e25_new.add_argument("--support-pool", type=Path, required=True)
    e25_new.add_argument("--calibration", type=Path, required=True)
    e25_new.add_argument("--output", type=Path, required=True)
    e25_new.add_argument("--processes", type=int, default=24)
    e26 = subcommands.add_parser("qualify-e26-v2")
    e26.add_argument("--data-root", type=Path, required=True)
    e26.add_argument("--support-pool", type=Path, required=True)
    e26.add_argument("--calibration", type=Path, required=True)
    e26.add_argument("--output", type=Path, required=True)
    e26.add_argument("--processes", type=int, default=24)
    e27 = subcommands.add_parser("qualify-e27")
    e27.add_argument("--e25-artifact", type=Path, required=True)
    e27.add_argument("--e26-artifact", type=Path, required=True)
    e27.add_argument("--data-root", type=Path, required=True)
    e27.add_argument("--calibration", type=Path, required=True)
    e27.add_argument("--output", type=Path, required=True)
    e27.add_argument("--processes", type=int, default=24)
    e28 = subcommands.add_parser("qualify-e28-v2")
    e28.add_argument("--e26-artifact", type=Path, required=True)
    e28.add_argument("--data-root", type=Path, required=True)
    e28.add_argument("--calibration", type=Path, required=True)
    e28.add_argument("--output", type=Path, required=True)
    e28.add_argument("--processes", type=int, default=24)
    e29 = subcommands.add_parser("qualify-e29")
    e29.add_argument("--calibration", type=Path, required=True)
    e29.add_argument("--output", type=Path, required=True)
    e29.add_argument("--processes", type=int, default=24)
    e30 = subcommands.add_parser("qualify-e30")
    e30.add_argument("--e25-artifact", type=Path, required=True)
    e30.add_argument("--e27-artifact", type=Path, required=True)
    e30.add_argument("--calibration", type=Path, required=True)
    e30.add_argument("--output", type=Path, required=True)
    e30.add_argument("--processes", type=int, default=24)
    e31 = subcommands.add_parser("qualify-e31")
    e31.add_argument("--e28-artifact", type=Path, required=True)
    e31.add_argument("--calibration", type=Path, required=True)
    e31.add_argument("--output", type=Path, required=True)
    e31.add_argument("--processes", type=int, default=24)
    e32 = subcommands.add_parser("qualify-e32")
    e32.add_argument("--output", type=Path, required=True)
    e33 = subcommands.add_parser("qualify-e33")
    e33.add_argument("--output", type=Path, required=True)
    e34 = subcommands.add_parser("qualify-e34")
    e34.add_argument("--output", type=Path, required=True)
    e35 = subcommands.add_parser("qualify-e35")
    e35.add_argument("--calibration", type=Path, required=True)
    e35.add_argument("--output", type=Path, required=True)
    e35.add_argument("--processes", type=int, default=24)
    e36 = subcommands.add_parser("qualify-e36")
    e36.add_argument("--output", type=Path, required=True)
    e36v2 = subcommands.add_parser("qualify-e36-v2")
    e36v2.add_argument("--calibration", type=Path, required=True)
    e36v2.add_argument("--output", type=Path, required=True)
    e36v2.add_argument("--processes", type=int, default=24)
    e37 = subcommands.add_parser("qualify-e37")
    e37.add_argument("--e26-artifact", type=Path, required=True)
    e37.add_argument("--data-root", type=Path, required=True)
    e37.add_argument("--calibration", type=Path, required=True)
    e37.add_argument("--output", type=Path, required=True)
    e37.add_argument("--processes", type=int, default=24)
    e38 = subcommands.add_parser("qualify-e38-v2")
    e38.add_argument("--data-root", type=Path, required=True)
    e38.add_argument("--e25-new-artifact", type=Path, required=True)
    e38.add_argument("--calibration", type=Path, required=True)
    e38.add_argument("--support-pool", type=Path, required=True)
    e38.add_argument("--candidate-bank-output", type=Path, required=True)
    e38.add_argument("--output", type=Path, required=True)
    e38.add_argument("--processes", type=int, default=24)
    e39 = subcommands.add_parser("qualify-e39-v2")
    e39.add_argument("--e38-artifact", type=Path, required=True)
    e39.add_argument("--output", type=Path, required=True)
    e40 = subcommands.add_parser("qualify-e40-v2")
    e40.add_argument("--e39-artifact", type=Path, required=True)
    e40.add_argument("--calibration", type=Path, required=True)
    e40.add_argument("--output", type=Path, required=True)
    e41 = subcommands.add_parser("qualify-e41-v2")
    e41.add_argument("--e39-artifact", type=Path, required=True)
    e41.add_argument("--output", type=Path, required=True)
    e42 = subcommands.add_parser("qualify-e42-v2")
    e42.add_argument("--e39-artifact", type=Path, required=True)
    e42.add_argument("--output", type=Path, required=True)
    e43 = subcommands.add_parser("qualify-e43-v2")
    e43.add_argument("--e37-artifact", type=Path, required=True)
    e43.add_argument("--e39-artifact", type=Path, required=True)
    e43.add_argument("--output", type=Path, required=True)
    e44 = subcommands.add_parser("qualify-e44-v2")
    e44.add_argument("--e39-artifact", type=Path, required=True)
    e44.add_argument("--output", type=Path, required=True)
    e45 = subcommands.add_parser("qualify-e45")
    e45.add_argument("--data-root", type=Path, required=True)
    e45.add_argument("--support-pool", type=Path, required=True)
    e45.add_argument("--output", type=Path, required=True)
    e45v2 = subcommands.add_parser("qualify-e45-v2")
    e45v2.add_argument("--data-root", type=Path, required=True)
    e45v2.add_argument("--support-pool", type=Path, required=True)
    e45v2.add_argument("--e25-artifact", type=Path, required=True)
    e45v2.add_argument("--calibration", type=Path, required=True)
    e45v2.add_argument("--candidate-bank-256", type=Path, required=True)
    e45v2.add_argument("--e45-v1-artifact", type=Path, required=True)
    e45v2.add_argument("--output", type=Path, required=True)
    e45v2.add_argument("--processes", type=int, default=24)
    e45a = subcommands.add_parser("qualify-e45a")
    e45a.add_argument("--unit-cache", type=Path, required=True)
    e45a.add_argument("--output", type=Path, required=True)
    e45b = subcommands.add_parser("qualify-e45b")
    e45b.add_argument("--unit-cache", type=Path, required=True)
    e45b.add_argument("--output", type=Path, required=True)
    for command in ("qualify-e45a-new", "qualify-e45b-v2"):
        pair = subcommands.add_parser(command)
        pair.add_argument("--data-root", type=Path, required=True)
        pair.add_argument("--e25-new-artifact", type=Path, required=True)
        pair.add_argument("--calibration", type=Path, required=True)
        pair.add_argument("--support-pool", type=Path, required=True)
        pair.add_argument("--output", type=Path, required=True)
        pair.add_argument("--processes", type=int, default=24)
    e48 = subcommands.add_parser("qualify-e48")
    e48.add_argument("--e45b-v2-artifact", type=Path, required=True)
    e48.add_argument("--output", type=Path, required=True)
    e57 = subcommands.add_parser("qualify-e57-v2")
    e57.add_argument("--data-root", type=Path, required=True)
    e57.add_argument("--protocol", type=Path, required=True)
    e57.add_argument("--source-bank", type=Path, required=True)
    e57.add_argument("--calibration", type=Path, required=True)
    e57.add_argument("--output", type=Path, required=True)
    e57.add_argument("--processes", type=int, default=24)
    e58 = subcommands.add_parser("qualify-e58")
    e58.add_argument("--data-root", type=Path, required=True)
    e58.add_argument("--protocol", type=Path, required=True)
    e58.add_argument("--e57", type=Path, required=True)
    e58.add_argument("--calibration", type=Path, required=True)
    e58.add_argument("--output", type=Path, required=True)
    e58.add_argument("--processes", type=int, default=24)
    e59_e60 = subcommands.add_parser("characterize-e59-e60")
    e59_e60.add_argument("--e57", type=Path, required=True)
    e59_e60.add_argument("--e59-output", type=Path, required=True)
    e59_e60.add_argument("--e60-output", type=Path, required=True)
    e45a2 = subcommands.add_parser("qualify-e45a-v2")
    e45a2.add_argument("--data-root", type=Path, required=True)
    e45a2.add_argument("--support-pool", type=Path, required=True)
    e45a2.add_argument("--e25-artifact", type=Path, required=True)
    e45a2.add_argument("--calibration", type=Path, required=True)
    e45a2.add_argument("--unit-cache", type=Path, required=True)
    e45a2.add_argument("--output", type=Path, required=True)
    e45a2.add_argument("--processes", type=int, default=24)
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
    if args.command == "qualify-e25-v3-targets":
        result = run_e25_v3_target_qualification(
            args.data_root, args.support_pool, args.target_bank,
            args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "build-e25-v3-target-bank":
        result = build_e25_v3_target_bank(
            args.source_target_bank, args.target_qualification, args.output,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e25-v3-normal-control":
        result = run_e25_v3_normal_control_qualification(
            args.data_root, args.support_pool, args.calibration,
            args.target_bank, args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e25-new-normal-control":
        result = run_e25_new_qualification(
            args.data_root,
            args.support_pool,
            args.calibration,
            args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e26-v2":
        result = run_e26_v2_qualification(
            args.data_root,
            args.support_pool,
            args.calibration,
            args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e27":
        result = run_e27_qualification(
            args.e25_artifact, args.e26_artifact, args.data_root,
            args.calibration, args.output, processes=args.processes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e28-v2":
        result = run_e28_v2_qualification(
            args.e26_artifact, args.data_root, args.calibration,
            args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e29":
        result = run_e29_qualification(
            args.calibration, args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e30":
        result = run_e30_qualification(
            args.e25_artifact, args.e27_artifact, args.calibration,
            args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e31":
        result = run_e31_qualification(
            args.e28_artifact, args.calibration, args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e32":
        result = run_e32_qualification(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e33":
        result = run_e33_qualification(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e34":
        result = run_e34_qualification(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e35":
        result = run_e35_qualification(args.calibration, args.output, processes=args.processes)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e36":
        result = run_e36_qualification(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e36-v2":
        result = run_e36_v2_qualification(args.calibration, args.output, processes=args.processes)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e37":
        result = run_e37_qualification(
            args.e26_artifact, args.data_root, args.calibration, args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e38-v2":
        result = run_e38_v2_qualification(
            args.data_root, args.e25_new_artifact, args.calibration,
            args.support_pool, args.candidate_bank_output, args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e39-v2":
        result = run_e39_qualification(
            args.e38_artifact, args.output,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e40-v2":
        result = run_e40_qualification(
            args.e39_artifact, args.calibration, args.output,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e41-v2":
        result = run_e41_qualification(args.e39_artifact, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e42-v2":
        result = run_e42_qualification(args.e39_artifact, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e43-v2":
        result = run_e43_qualification(
            args.e37_artifact, args.e39_artifact, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e44-v2":
        result = run_e44_qualification(args.e39_artifact, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e45":
        result = run_e45_qualification(
            args.data_root, args.support_pool, args.output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e45-v2":
        result = run_e45_v2_qualification(
            args.data_root, args.support_pool, args.e25_artifact,
            args.calibration, args.candidate_bank_256, args.e45_v1_artifact,
            args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e45a":
        result = run_e45a_qualification(args.unit_cache, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e45b":
        result = run_e45b_qualification(args.unit_cache, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command in {"qualify-e45a-new", "qualify-e45b-v2"}:
        result = run_e45_pair_v2_qualification(
            args.data_root, args.e25_new_artifact, args.calibration,
            args.support_pool, args.output,
            experiment=("E45A-new" if args.command.endswith("a-new") else "E45B-v2"),
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e48":
        result = run_e48_qualification(args.e45b_v2_artifact, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e57-v2":
        result = run_e57_qualification(
            args.data_root, args.protocol, args.source_bank,
            args.calibration, args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "qualify-e58":
        result = run_e58_qualification(
            args.data_root, args.protocol, args.e57,
            args.calibration, args.output, processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "characterize-e59-e60":
        result = run_e59_e60_characterization(
            args.e57, args.e59_output, args.e60_output
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "qualify-e45a-v2":
        result = run_e45a_v2_qualification(
            args.data_root, args.support_pool, args.e25_artifact,
            args.calibration, args.unit_cache, args.output,
            processes=args.processes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    raise AssertionError("unreachable renderer command")


if __name__ == "__main__":
    raise SystemExit(main())
