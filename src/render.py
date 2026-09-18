"""Fixed-world single-scan observation, selected from AJAE/src/render.py.

All geometry, response tables, tolerances and seeds are supplied explicitly.
This module provides no V4 sampling distribution or fitted sensor default.
"""

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.special import expit
from scipy.stats import qmc

from .data import Frame, readonly, rigid, supervision
from .shape import Shape, unresolved_penetration


@dataclass(frozen=True, slots=True)
class Material:
    quantile: float
    roughness: float
    return_bias: float

    def __post_init__(self):
        if not np.isfinite((self.quantile, self.roughness, self.return_bias)).all():
            raise ValueError("material parameters must be finite")
        if not 0 <= self.quantile <= 1 or self.roughness < 0:
            raise ValueError("invalid material quantile or roughness")


@dataclass(frozen=True, slots=True)
class Response:
    """Explicit beam/range/incidence response tables and their stated provenance."""

    range_edges: np.ndarray
    incidence_edges: np.ndarray
    quantiles: np.ndarray
    probability: np.ndarray
    intensity: np.ndarray
    intensity_bounds: tuple
    intensity_step: float | None
    provenance: str

    def __post_init__(self):
        for name in ("range_edges", "incidence_edges", "quantiles"):
            array = np.asarray(getattr(self, name), np.float64)
            if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all() or np.any(np.diff(array) <= 0):
                raise ValueError(f"invalid response {name}")
            object.__setattr__(self, name, readonly(array.copy()))
        if self.range_edges[0] < 0 or self.incidence_edges[0] < 0 or self.incidence_edges[-1] > math.pi / 2:
            raise ValueError("range/incidence edges are outside their physical domain")
        if self.quantiles[0] != 0 or self.quantiles[-1] != 1:
            raise ValueError("intensity quantiles must span [0,1]")
        probability, intensity = np.asarray(self.probability, np.float64), np.asarray(self.intensity, np.float64)
        bins = (len(self.range_edges) - 1, len(self.incidence_edges) - 1)
        if probability.ndim != 3 or probability.shape[0] < 1 or probability.shape[1:] != bins:
            raise ValueError("return probability must be [beam,range,incidence]")
        if intensity.shape != probability.shape + (len(self.quantiles),):
            raise ValueError("intensity quantile table has incompatible shape")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("return probabilities must be finite in [0,1]")
        bounds = np.asarray(self.intensity_bounds, np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("invalid intensity bounds")
        if not np.isfinite(intensity).all() or np.any(np.diff(intensity, axis=-1) < 0) or np.any((intensity < bounds[0]) | (intensity > bounds[1])):
            raise ValueError("intensity quantiles must be ordered within the supplied bounds")
        if self.intensity_step is not None and (not np.isfinite(self.intensity_step) or self.intensity_step <= 0):
            raise ValueError("intensity step must be positive or explicitly None")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("state the source or diagnostic purpose of the response tables")
        object.__setattr__(self, "probability", readonly(probability.copy()))
        object.__setattr__(self, "intensity", readonly(intensity.copy()))
        object.__setattr__(self, "intensity_bounds", tuple(bounds))

    def sample(self, beam, distance, incidence, material, return_uniform, intensity_uniform):
        beam = np.asarray(beam)
        arrays = [np.asarray(value, np.float64) for value in (distance, incidence, return_uniform, intensity_uniform)]
        if beam.ndim != 1 or not np.issubdtype(beam.dtype, np.integer) or any(value.shape != beam.shape or not np.isfinite(value).all() for value in arrays):
            raise ValueError("response queries must be aligned finite one-dimensional arrays")
        distance, incidence, return_uniform, intensity_uniform = arrays
        if np.any((beam < 0) | (beam >= self.probability.shape[0])) or np.any(distance < 0) or np.any((incidence < 0) | (incidence > math.pi / 2)) or any(np.any((value < 0) | (value > 1)) for value in arrays[2:]):
            raise ValueError("response query outside its domain")
        # The original model extends edge bins; it does not extrapolate intensity.
        r = np.clip(np.searchsorted(self.range_edges, distance, side="right") - 1, 0, len(self.range_edges) - 2)
        a = np.clip(np.searchsorted(self.incidence_edges, incidence, side="right") - 1, 0, len(self.incidence_edges) - 2)
        base = self.probability[beam, r, a]
        chance = base.copy()
        interior = (base > 0) & (base < 1)
        chance[interior] = expit(np.log(base[interior] / (1 - base[interior])) + 2 * material.return_bias)
        # Preserve exact deterministic 0/1 endpoints instead of clipping them inward.
        returned = (chance == 1) | (return_uniform < chance)
        quantile = np.clip(material.quantile + material.roughness * (intensity_uniform - .5), 0, 1)
        upper = np.clip(np.searchsorted(self.quantiles, quantile, side="right"), 1, len(self.quantiles) - 1)
        lower = upper - 1
        weight = (quantile - self.quantiles[lower]) / (self.quantiles[upper] - self.quantiles[lower])
        values = self.intensity[beam, r, a]
        row = np.arange(len(beam))
        intensity = values[row, lower] * (1 - weight) + values[row, upper] * weight
        # Preserve the original float32 return format before optional quantization.
        intensity = np.clip(intensity.astype(np.float32), *self.intensity_bounds).astype(np.float64)
        if self.intensity_step is not None:
            intensity = np.rint(intensity / self.intensity_step) * self.intensity_step
        return returned, intensity


@dataclass(frozen=True, slots=True)
class Object:
    object_id: int
    geometry_id: str
    shape: Shape
    material: Material
    pose: np.ndarray

    def __post_init__(self):
        if type(self.object_id) is not int or not 0 < self.object_id <= np.iinfo(np.int64).max:
            raise ValueError("object_id must be a positive int64")
        if not isinstance(self.geometry_id, str) or not self.geometry_id.strip():
            raise ValueError("geometry_id must identify the supplied geometry")
        if not isinstance(self.shape, Shape) or not isinstance(self.material, Material):
            raise TypeError("object requires Shape and Material")
        pose = np.asarray(self.pose, np.float64)
        rigid(pose)
        object.__setattr__(self, "pose", readonly(pose.copy()))

    def bounds(self):
        lower, upper = self.shape.bounds()
        corners = np.asarray(list(product(*zip(lower, upper)))) @ self.pose[:3, :3].T + self.pose[:3, 3]
        return corners.min(axis=0), corners.max(axis=0)


@dataclass(frozen=True, slots=True)
class World:
    version: str
    sequence_id: int
    seed: int
    objects: tuple
    tie_tolerance_m: float

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version.strip() or self.sequence_id != 206:
            raise ValueError("a V4 world must identify its version and source sequence 206")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("world seed must be a uint64 integer")
        if not np.isfinite(self.tie_tolerance_m) or self.tie_tolerance_m < 0:
            raise ValueError("tie tolerance must be finite and nonnegative")
        objects = tuple(sorted(self.objects, key=lambda item: item.object_id))
        if len({item.object_id for item in objects}) != len(objects):
            raise ValueError("duplicate object identity in a world")
        geometries = {}
        for item in objects:
            if item.geometry_id in geometries and geometries[item.geometry_id] != item.shape:
                raise ValueError("one geometry_id refers to different shapes")
            geometries[item.geometry_id] = item.shape
        # A world has no per-frame visibility switch. Windows select observations later.
        object.__setattr__(self, "objects", objects)


def slot_uniform(world, frame_id, slots, object_id, channel):
    """Original stateless stream, keyed by source frame, file slot, object and channel."""
    base = (world.seed ^ (world.sequence_id << 24) ^ (frame_id << 40) ^ (channel * 0xA24BAED4963EE407)) & ((1 << 64) - 1)
    with np.errstate(over="ignore"):
        value = np.asarray(slots, np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        value ^= np.uint64(object_id) * np.uint64(0xD1B54A32D192ED03) ^ np.uint64(base)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return (value.astype(np.float64) + .5) / float(2**64)


@dataclass(frozen=True, slots=True)
class Observation:
    frame: Frame
    world: World
    inserted: np.ndarray
    occluded_original: np.ndarray
    visible_normal: np.ndarray
    object_ids: np.ndarray

    def effective_counts(self):
        """Apply frame selection first; keep 1-4 point objects within eligible frames."""
        selected = supervision(self.frame)
        return {item.object_id: int(((self.object_ids == item.object_id) & (selected.targets == 1)).sum())
                for item in self.world.objects}


def render_frame(source, world, rays, response, trace):
    """Resolve all opaque surfaces jointly, then sample the winning surface's return."""
    if source.partition != "train" or source.sequence_id != world.sequence_id:
        raise ValueError("the supplied world belongs to training sequence 206")
    if source.labels is None:
        raise ValueError("background construction requires original labels")
    if len(source.xyzi) != len(rays.directions) or response.probability.shape[0] != len(rays.local):
        raise ValueError("scan, rays and response beam counts do not align")
    rotation, translation = source.pose[:3, :3], source.pose[:3, 3]
    world_origins = rays.origins @ rotation.T + translation
    world_directions = rays.directions @ rotation.T
    direction_norm = np.linalg.norm(world_directions, axis=1)
    unit_world = world_directions / direction_norm[:, None]
    # Compare distances on one world-space unit ray despite source pose rounding.
    native_world = source.xyzi[:, :3] @ rotation.T + translation
    native_t = np.sum((native_world - world_origins) * unit_world, axis=1)
    native_t[~source.actual] = np.inf
    if np.any(native_t <= 0):
        raise ValueError("a native return lies behind its ray origin")
    count = len(source.xyzi)
    nearest = np.full(count, np.inf)
    owner = np.full(count, -1, dtype=np.int64)
    incidence = np.zeros(count)
    for item in world.objects:
        object_rotation = item.pose[:3, :3]
        local_origins = (world_origins - item.pose[:3, 3]) @ object_rotation
        local_directions = unit_world @ object_rotation
        distances, normals, valid = item.shape.intersect(local_origins, local_directions, trace)
        # Sorted IDs give reproducible ownership to coincident surfaces.
        take = valid & (distances < nearest - world.tie_tolerance_m)
        nearest[take], owner[take] = distances[take], item.object_id
        unit_local = local_directions[take] / np.linalg.norm(local_directions[take], axis=1, keepdims=True)
        incidence[take] = np.arccos(np.clip(np.abs(np.sum(normals[take] * -unit_local, axis=1)), 0, 1))
    foreground = (owner >= 0) & (nearest < native_t - world.tie_tolerance_m)
    occluded = foreground & source.actual
    inserted = np.zeros(count, dtype=bool)
    xyzi, labels = source.xyzi.copy(), source.labels.copy()
    # No-return foreground is still opaque; neither a rear object nor background reappears.
    xyzi[occluded], labels[occluded] = 0, 0
    object_ids = np.full(count, -1, dtype=np.int64)
    columns = count // len(rays.local)
    for item in world.objects:
        slots = np.flatnonzero(foreground & (owner == item.object_id))
        if not len(slots):
            continue
        returned, intensity = response.sample(
            slots // columns, nearest[slots], incidence[slots], item.material,
            slot_uniform(world, source.frame_id, slots, item.object_id, 0),
            slot_uniform(world, source.frame_id, slots, item.object_id, 1),
        )
        kept = slots[returned]
        inserted[kept], object_ids[kept] = True, item.object_id
        ray_parameter = nearest[kept] / direction_norm[kept]
        xyzi[kept, :3] = rays.origins[kept] + ray_parameter[:, None] * rays.directions[kept]
        xyzi[kept, 3] = intensity[returned]
        # Synthetic object IDs stay in metadata, never collide with native instance IDs.
        labels[kept] = 2
    frame = Frame(source.frame_id, xyzi, source.pose, labels, source.sequence_id, source.partition)
    normal = source.actual & (source.semantic != 0) & (source.semantic != 2) & ~foreground
    return Observation(frame, world, readonly(inserted), readonly(occluded), readonly(normal), readonly(object_ids))


@dataclass(frozen=True, slots=True)
class Grounding:
    shape: Shape
    lower_z: float
    refined_lower_z: float
    buried_fraction: float
    surface_points: np.ndarray
    accepted: bool


def check_grounding(shape, *, coarse, fine, trace, surface_count, residual_tolerance,
                    convergence_m, buried_depth_m, max_buried_fraction):
    if not np.isfinite((convergence_m, buried_depth_m, max_buried_fraction)).all() or convergence_m < 0 or buried_depth_m < 0 or not 0 <= max_buried_fraction <= 1:
        raise ValueError("invalid explicit grounding thresholds")
    lower = shape.minimum_z(**coarse)
    refined = shape.minimum_z(**fine)
    surface = shape.surface_points(surface_count, trace, residual_tolerance)
    fraction = float(np.mean(surface[:, 2] - lower < -buried_depth_m))
    accepted = abs(refined - lower) <= convergence_m and fraction <= max_buried_fraction
    return Grounding(shape, lower, refined, fraction, surface, bool(accepted))


def ground_object(grounding, material, *, object_id, geometry_id, anchor_world,
                  normal_world, plane_offset, yaw):
    """Orient local z along a supplied support normal and place the lower surface on it."""
    if not grounding.accepted:
        raise ValueError("shape failed the caller's grounding thresholds")
    anchor, up = np.asarray(anchor_world, np.float64), np.asarray(normal_world, np.float64)
    if anchor.shape != (3,) or up.shape != (3,) or not np.isfinite((anchor, up)).all() or not np.isfinite((plane_offset, yaw)).all():
        raise ValueError("invalid support plane")
    norm = np.linalg.norm(up)
    if norm == 0 or up[2] <= 0:
        raise ValueError("support normal must point upward")
    up, offset = up / norm, plane_offset / norm
    contact = anchor.copy()
    contact[2] = -(up[0] * contact[0] + up[1] * contact[1] + offset) / up[2]
    heading = np.array((math.cos(yaw), math.sin(yaw), 0.))
    x = heading - up * np.dot(heading, up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = np.column_stack((x, y, up)), contact - up * grounding.lower_z
    return Object(object_id, geometry_id, grounding.shape, material, pose)


def observed_collision(item, points_world, *, allowance_m, gradient_step_m, witness_fraction):
    """Return offending supplied point indices; unseen scene geometry remains unknown."""
    points = np.asarray(points_world, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("observed obstacle points must be finite [N,3]")
    lower, upper = item.bounds()
    selected = np.flatnonzero(((points >= lower) & (points <= upper)).all(axis=1))
    local = (points[selected] - item.pose[:3, 3]) @ item.pose[:3, :3]
    unresolved, values = unresolved_penetration(item.shape, local, allowance_m=allowance_m,
                                               gradient_step_m=gradient_step_m, witness_fraction=witness_fraction)
    return selected[unresolved], float(values.min()) if len(values) else math.inf


def pair_collision(left, right, *, trace, surface_count, residual_tolerance,
                   interior_power, allowance_m, gradient_step_m, witness_fraction):
    """Test sampled surfaces/interiors on both sides; absence of witnesses is not proof."""
    if type(interior_power) is not int or interior_power < 0:
        raise ValueError("interior_power must be a nonnegative integer")
    left_lower, left_upper = left.bounds()
    right_lower, right_upper = right.bounds()
    if np.any(left_upper < right_lower) or np.any(right_upper < left_lower):
        return False, math.inf
    minimum = math.inf
    for source, target in ((left, right), (right, left)):
        shape = source.shape
        surface = shape.surface_points(surface_count, trace, residual_tolerance)
        lower, upper = shape.bounds()
        probes = lower + qmc.Sobol(d=3, scramble=False).random_base2(interior_power) * (upper - lower)
        centers = shape.deform(np.asarray(shape.offsets))
        probes = np.concatenate((probes, centers))
        # A subtracted primitive's center may be empty and is not a collision witness.
        interior = probes[shape.level(probes) < 0]
        world = np.concatenate((surface, interior)) @ source.pose[:3, :3].T + source.pose[:3, 3]
        local = (world - target.pose[:3, 3]) @ target.pose[:3, :3]
        unresolved, values = unresolved_penetration(target.shape, local, allowance_m=allowance_m,
                                                   gradient_step_m=gradient_step_m, witness_fraction=witness_fraction)
        minimum = min(minimum, float(values.min()))
        if unresolved.any():
            return True, minimum
    return False, minimum
