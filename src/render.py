"""Fixed-world observation and the bounded 206 normal-reference pilot generator.

The core ray renderer accepts explicit geometry, response tables and tolerances.
The pilot uses only training placements and the existing 206 response calibration.
"""

from dataclasses import asdict, dataclass
from itertools import product
import math

import numpy as np
from scipy.special import expit
from scipy.stats import qmc

from .data import (Frame, PILOT_VERSION, STUSequence, legacy_source_identity,
                   point_targets, readonly, rigid, supervision, read_rays, write_json,
                   nuscenes_rays, nuscenes_poses, read_nuscenes, identity, NATIVE_VERSION,
                   DIVERSITY, context_groups, observation_descriptor, normal_descriptor,
                   select_observations, native_keyframes, expand_stu, load_manifest)
from .shape import Shape, Trace, unresolved_penetration


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
        if not isinstance(self.version, str) or not self.version.strip() or self.sequence_id not in (0, 206):
            raise ValueError("a V4 world must identify its version and nuScenes or STU 206 source")
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
    sampling: tuple = ()

    def effective_counts(self):
        """Apply frame selection first; keep 1-4 point objects within eligible frames."""
        selected = supervision(self.frame)
        return {item.object_id: int(((self.object_ids == item.object_id) & (selected.targets == 1)).sum())
                for item in self.world.objects}


def render_frame(source, world, rays, response, trace):
    """Resolve all opaque surfaces jointly, then sample the winning surface's return."""
    if source.partition != "train" or source.sequence_id != world.sequence_id:
        raise ValueError("the supplied world and training scan belong to different sensors")
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
    if rays.returned is not None:
        native_t[~rays.returned] = np.inf
    if np.any(native_t <= 0):
        raise ValueError("a native return lies behind its ray origin")
    count = len(source.xyzi)
    nearest = np.full(count, np.inf)
    owner = np.full(count, -1, dtype=np.int64)
    incidence = np.zeros(count)
    potential = {}
    for item in world.objects:
        object_rotation = item.pose[:3, :3]
        local_origins = (world_origins - item.pose[:3, 3]) @ object_rotation
        local_directions = unit_world @ object_rotation
        distances, normals, valid = item.shape.intersect(local_origins, local_directions, trace)
        potential[item.object_id] = int(valid.sum())
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
    sampling = []
    for item in world.objects:
        slots = np.flatnonzero(foreground & (owner == item.object_id))
        info = dict(object_id=item.object_id, potential_surface_rays=potential[item.object_id],
                    foreground_surface_rays=len(slots), returned_rays=0,
                    incidence_degrees=[] if not len(slots) else np.rad2deg(np.quantile(incidence[slots], [.1,.5,.9])).tolist())
        sampling.append(info)
        if not len(slots):
            continue
        returned, intensity = response.sample(
            slots // columns if rays.beam_ids is None else rays.beam_ids[slots],
            nearest[slots], incidence[slots], item.material,
            slot_uniform(world, source.frame_id, slots, item.object_id, 0),
            slot_uniform(world, source.frame_id, slots, item.object_id, 1),
        )
        kept = slots[returned]
        info["returned_rays"] = len(kept)
        inserted[kept], object_ids[kept] = True, item.object_id
        ray_parameter = nearest[kept] / direction_norm[kept]
        xyzi[kept, :3] = rays.origins[kept] + ray_parameter[:, None] * rays.directions[kept]
        xyzi[kept, 3] = intensity[returned]
        # Synthetic object IDs stay in metadata, never collide with native instance IDs.
        labels[kept] = 2
    frame = Frame(source.frame_id, xyzi, source.pose, labels, source.sequence_id, source.partition)
    normal = source.actual & (source.semantic != 0) & (source.semantic != 2) & ~foreground
    # Potential/foreground count ray opportunities, not visible surface area.
    return Observation(frame, world, readonly(inserted), readonly(occluded), readonly(normal), readonly(object_ids), tuple(sampling))


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


def observation_features(xyzi):
    """Measured local descriptors nominate candidates; they never determine labels."""
    xyz = xyzi[:, :3].astype(np.float64)
    eigen = np.linalg.eigvalsh(np.cov(xyz.T))[::-1]
    eigen = np.maximum(eigen, 0)
    return dict(count=len(xyzi), range_m=float(np.median(np.linalg.norm(xyz, axis=1))),
                spread_m=(2 * np.sqrt(eigen)).tolist(),
                linearity=float((eigen[0] - eigen[1]) / max(eigen[0], 1e-12)),
                planarity=float((eigen[1] - eigen[2]) / max(eigen[0], 1e-12)),
                intensity=np.quantile(xyzi[:, 3], [.1, .5, .9]).tolist())


def candidate_distance(left, right):
    # Each quantity has fixed units; this is a ranking heuristic, not a difficulty label.
    a = np.r_[left["count"], left["range_m"], np.asarray(left["spread_m"]) + .02]
    b = np.r_[right["count"], right["range_m"], np.asarray(right["spread_m"]) + .02]
    return float(np.mean(np.abs(np.log(a / b))) +
                 abs(left["linearity"] - right["linearity"]) +
                 abs(left["planarity"] - right["planarity"]) +
                 np.mean(np.abs(np.asarray(left["intensity"]) - right["intensity"])))


def _generate_candidate(task):
    import json
    from pathlib import Path
    geometry_id, subset, seed, output = task
    saved = Path(output) / geometry_id / "world.json"
    if saved.exists():
        row = json.loads(saved.read_text())
        if (row["version"], row["geometry"], row["subset"], row["seed"]) != (PILOT_VERSION, geometry_id, subset, seed):
            raise ValueError("existing candidate has a different identity")
        if not all(Path(frame["delta"]).is_file() for frame in row["frames"]):
            raise ValueError("completed candidate has missing observations")
        return row
    rng = np.random.default_rng(seed)
    index = int(geometry_id.rsplit("-", 1)[1])
    # Detached solids expose unknown obstacles without assigning real anomaly species.
    extents = np.asarray(((.8, .12, .14), (.55, .38, .09), (.24, .22, .28))[index % 3])
    extents *= rng.uniform(.75, 1.3, 3)
    shape = Shape((tuple(extents / 2),), ((0., 0., 0.),),
                  (tuple(rng.uniform(.5, 1.3, 2)),), (0.,), ("union",),
                  0., (0., 0.), (0., 0.), 0., (1., 1., 1.), (0., 0., 0.))
    material = Material(float(rng.uniform(.15, .85)), float(rng.uniform(.1, .35)), 0.)
    grounding = check_grounding(shape, coarse=dict(xy_resolution=33, z_steps=129, bisections=24, refinements=5),
                                fine=dict(xy_resolution=65, z_steps=257, bisections=24, refinements=5),
                                trace=_trace, surface_count=128, residual_tolerance=1e-5,
                                convergence_m=1e-4, buried_depth_m=1e-4, max_buried_fraction=0.)
    if not grounding.accepted:
        return dict(geometry=geometry_id, subset=subset, accepted=False, reason="grounding")
    proposals = []
    rejected = dict(reference=0, support=0, collision=0, visibility=0)
    for anchor_id in rng.permutation(len(_anchors))[:32]:
        anchor = _anchors[anchor_id]
        ref = anchor["generation"].get("normal_reference")
        if not ref or "support_plane" not in anchor["generation"] or "geometry" not in anchor["generation"]:
            rejected["reference"] += 1
            continue
        source = _frames[ref["frame"]]
        slots = np.asarray(ref["slots"], dtype=int)
        normal_slots = np.flatnonzero(point_targets(source) == 0)
        center = np.median(source.xyzi[slots, :3], axis=0)
        central = normal_slots[np.argmin(np.linalg.norm(source.xyzi[normal_slots, :3] - center, axis=1))]
        center = source.xyzi[central, :3]
        slots = normal_slots[(source.semantic[normal_slots] == source.semantic[central]) &
                            (np.linalg.norm(source.xyzi[normal_slots, :3] - center, axis=1) <= .35)]
        if len(slots) < 5:
            rejected["reference"] += 1
            continue
        plane = anchor["generation"]["support_plane"]
        radius = float(np.linalg.norm(extents[:2] / 2))
        if radius > anchor["generation"]["geometry"]["footprint_radius_m"]:
            rejected["support"] += 1
            continue
        item = ground_object(grounding, material, object_id=1, geometry_id=geometry_id,
                             anchor_world=plane["anchor_world_m"], normal_world=plane["normal_world"],
                             plane_offset=plane["offset"], yaw=anchor["generation"]["yaw_rad"])
        # Check every original scan, including moving obstacles, before rendering.
        if any(len(observed_collision(item, points, allowance_m=.05, gradient_step_m=1e-6,
                                      witness_fraction=1 - 1e-6)[0]) for points in _obstacles):
            rejected["collision"] += 1
            continue
        world = World(PILOT_VERSION, 206, seed, (item,), 1e-6)
        observed = render_frame(source, world, _rays, _response, _trace)
        selected = supervision(observed.frame)
        anomaly = np.flatnonzero(selected.targets == 1)
        if len(anomaly) < 5 or not np.all(observed.visible_normal[slots]):
            rejected["visibility"] += 1
            continue
        xyzi = observed.frame.xyzi
        nearest = anomaly[np.argmin(np.linalg.norm(xyzi[anomaly, :3] - np.median(xyzi[anomaly, :3], axis=0), axis=1))]
        local = anomaly[np.linalg.norm(xyzi[anomaly, :3] - xyzi[nearest, :3], axis=1) <= .35]
        if len(local) < 5:
            rejected["visibility"] += 1
            continue
        normal_feature, anomaly_feature = observation_features(source.xyzi[slots]), observation_features(xyzi[local])
        score = candidate_distance(normal_feature, anomaly_feature)
        proposals.append((score, world, dict(frame=source.frame_id, normal_slots=slots.tolist(),
                          anomaly_slots=local.tolist(), normal=normal_feature, anomaly=anomaly_feature,
                          semantic=int(source.semantic[central]), radius_m=.35,
                          support_plane=plane, source_anchor=int(anchor_id))))
        if len(proposals) == 6:
            break
    if not proposals:
        return dict(geometry=geometry_id, subset=subset, accepted=False,
                    reason="no_legal_visible_candidate", rejected=rejected)
    score, world, reference = min(proposals, key=lambda item: item[0])
    directory = Path(output) / geometry_id
    directory.mkdir()
    item = world.objects[0]
    metadata = dict(version=PILOT_VERSION, geometry=geometry_id, subset=subset,
                    accepted=True, seed=seed, shape=asdict(shape), material=asdict(material),
                    pose=item.pose.tolist(), reference=reference, candidate_score=score,
                    compared_proposals=len(proposals), rejected=rejected,
                    label="anomaly-proxy", collision="all observed non-ground returns in all 449 frames",
                    unobserved_geometry="not certified", frames=[], skipped={"zero": 0, "one_to_four": 0})
    for source in _frames:
        observed = render_frame(source, world, _rays, _response, _trace)
        selected = supervision(observed.frame)
        if not selected.eligible:
            metadata["skipped"]["zero" if selected.anomaly_count == 0 else "one_to_four"] += 1
            continue
        slots = np.flatnonzero(observed.inserted | observed.occluded_original).astype(np.int32)
        path = directory / f"{source.frame_id:06d}.npz"
        np.savez_compressed(path, slots=slots, xyzi=observed.frame.xyzi[slots],
                            labels=observed.frame.labels[slots],
                            inserted=np.flatnonzero(observed.inserted).astype(np.int32),
                            occluded=np.flatnonzero(observed.occluded_original).astype(np.int32),
                            frame=np.int64(source.frame_id), source_identity=_source_ids[source.frame_id])
        metadata["frames"].append(dict(source="targeted", group="targeted", subset=subset,
            geometry=geometry_id, frame=source.frame_id, delta=str(path.resolve()),
            normal=selected.normal_count, anomaly=selected.anomaly_count,
            points=int(observed.frame.actual.sum()), slots=len(source.xyzi)))
    write_json(directory / "world.json", metadata)
    return metadata


def generate_candidates(output, *, data_root, pool_root, workers=8):
    import json
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    import torch
    global _frames, _obstacles, _anchors, _rays, _response, _trace, _source_ids
    output, pool_root = Path(output), Path(pool_root)
    split = json.loads((output / "split.json").read_text())
    sequence = STUSequence(data_root)
    _frames = [sequence[i] for i in range(len(sequence))]
    _source_ids = [legacy_source_identity(frame) for frame in _frames]
    _obstacles = []
    for frame in _frames:
        selected = frame.actual & ~np.isin(frame.semantic, [40, 44, 48, 49, 60])
        _obstacles.append(readonly(frame.xyzi[selected, :3].astype(np.float64) @
                                  frame.pose[:3, :3].T + frame.pose[:3, 3]))
    pool = json.loads((pool_root / "manifest.json").read_text())
    _anchors = [json.loads((pool_root / row["path"] / "world.json").read_text())
                for row in pool["splits"]["train"]["worlds"]]
    if not any(row["generation"].get("normal_reference") for row in _anchors):
        raise ValueError("no training placement has a measured normal reference")
    if pool["splits"]["train"]["source_sequence"] != 206:
        raise ValueError("placement references must come only from STU 206")
    calibration = torch.load(pool_root / "calibration.pt", map_location="cpu", weights_only=False)
    sensor = calibration["sensor"]
    if sensor["source_sequence_id"] != 206:
        raise ValueError("response calibration must use only STU 206")
    _rays = read_rays()
    _response = Response(sensor["range_edges_m"], sensor["incidence_edges_rad"], sensor["quantile_levels"],
                         sensor["return_probability"], sensor["intensity_quantiles"],
                         (sensor["intensity_min"], sensor["intensity_max"]), None,
                         str(pool_root / "calibration.pt") + "; fitted from original STU 206 only")
    _trace = Trace(96, 8, 24, 1e-5, 4., 1e-5, 1e-5, 1e-9)
    tasks = [(geometry, subset, split["seed"] + 10000 + int(geometry.rsplit("-", 1)[1]), str(output))
             for subset in ("train", "check") for geometry in split["geometry_" + subset]]
    if any((output / geometry).exists() and not (output / geometry / "world.json").exists()
           for geometry, _, _, _ in tasks):
        raise FileExistsError("incomplete candidate outputs require inspection before restarting")
    records = []
    # Read-only source arrays are shared after fork; no repeated 206 decoding per worker.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as executor:
        for row in executor.map(_generate_candidate, tasks):
            records.append(dict(geometry=row["geometry"], subset=row["subset"], accepted=row["accepted"],
                                scans=len(row.get("frames", [])), candidate_score=row.get("candidate_score"),
                                path=str((output / row["geometry"] / "world.json").resolve()))
                           if row["accepted"] else row)
            print(json.dumps(dict(geometry=row["geometry"], accepted=row["accepted"],
                                  scans=len(row.get("frames", [])), score=row.get("candidate_score"))), flush=True)
    write_json(output / "targeted.json", dict(version=PILOT_VERSION, trace=asdict(_trace),
               response=_response.provenance, worlds=records,
               interpretation="normal-reference candidates; local confusion and context utility unproven"))


def _native_statistics(task):
    """Estimate response only inside locally supported, same-class planar patches."""
    record, mapping = task
    frame = read_nuscenes(record, mapping)
    rays, diagnostic = nuscenes_rays(record)
    raw = np.fromfile(record["label"], np.uint8)
    count = len(raw)
    slots = np.arange(33, count - 33)
    slots = slots[(slots % 32 > 0) & (slots % 32 < 31)]
    neighbor = slots[:, None] + np.array([-32, -1, 32, 1])
    keep = (point_targets(frame)[neighbor] == 0).all(1) & (raw[neighbor] == raw[neighbor[:, :1]]).all(1)
    slots, neighbor = slots[keep], neighbor[keep]
    points = frame.xyzi[neighbor, :3].astype(float)
    center = points.mean(1)
    centered = points - center[:, None]
    eig, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered) / 4)
    normal = vectors[:, :, 0]
    denom = (rays.directions[slots] * normal).sum(1)
    t = ((center - rays.origins[slots]) * normal).sum(1) / np.where(abs(denom) > 1e-8, denom, np.nan)
    intersection = rays.origins[slots] + t[:, None] * rays.directions[slots]
    radius = np.linalg.norm(centered, axis=-1).max(1)
    valid = (eig[:, 0] < .02**2) & (eig[:, 1] > 1e-5) & (radius < 2.) & (t >= 2.5) & (t <= 50)
    valid &= np.linalg.norm(intersection - center, axis=1) < radius * .7
    depth = ((frame.xyzi[slots, :3] - rays.origins[slots]) * rays.directions[slots]).sum(1)
    hit = rays.returned[slots] & (abs(depth - t) <= .08) & (point_targets(frame)[slots] == 0)
    # A real foreground/edge is not a failed return from the extrapolated plane.
    valid &= hit | ~rays.returned[slots]
    slots, t, denom, hit = slots[valid], t[valid], denom[valid], hit[valid]
    return dict(scene=record["scene"], token=record["token"], rays=diagnostic,
                beam=slots % 32, distance=t, incidence=np.arccos(np.clip(abs(denom), 0, 1)),
                returned=hit, intensity=frame.xyzi[slots, 3])


def fit_native_response(records, mapping, workers):
    """Fit training-only empirical response; unobserved bin support is explicit."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    by_log = {}
    for row in records:
        by_log.setdefault(row["log_token"], row)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(_native_statistics, [(r, mapping) for r in by_log.values()]))
    values = {k: np.concatenate([r[k] for r in rows]) for k in
              ("beam", "distance", "incidence", "returned", "intensity")}
    ranges, angles, quantiles = np.array([2.5, 5., 10., 20., 35., 50.]), np.deg2rad([0., 30., 60., 90.]), np.linspace(0, 1, 9)
    rbin = np.clip(np.searchsorted(ranges, values["distance"], side="right") - 1, 0, 4)
    abin = np.clip(np.searchsorted(angles, values["incidence"], side="right") - 1, 0, 2)
    probability, intensity = np.empty((32, 5, 3)), np.empty((32, 5, 3, 9))
    support, fallback = np.zeros((32, 5, 3), int), np.zeros((32, 5, 3), bool)
    for beam, r, a in product(range(32), range(5), range(3)):
        selected = (values["beam"] == beam) & (rbin == r) & (abin == a)
        support[beam, r, a] = selected.sum()
        if selected.sum() < 32:
            selected = (rbin == r) & (abin == a)
            fallback[beam, r, a] = True
        if selected.sum() < 32:
            selected = rbin == r
        if selected.sum() < 32:
            selected = np.ones(len(rbin), bool)
        probability[beam, r, a] = values["returned"][selected].mean()
        emitted = values["intensity"][selected & values["returned"]]
        if not len(emitted):
            raise ValueError("no real normal intensity supports a native response bin")
        intensity[beam, r, a] = np.quantile(emitted, quantiles)
    response = Response(ranges, angles, quantiles, probability, intensity, (0., 1.), 1 / 255,
                        "nuScenes training logs; supported planar interiors; empirical counterfactual approximation")
    report = dict(sources=[{k: r[k] for k in ("scene", "token", "rays")} for r in rows],
                  opportunities=len(rbin), observed_returns=int(values["returned"].sum()),
                  support=support.tolist(), pooled_fallback=fallback.tolist(),
                  probability=probability.tolist(), intensity=intensity.tolist(),
                  range_edges=ranges.tolist(), incidence_edges=angles.tolist(), quantiles=quantiles.tolist(),
                  limitation="Planar interior response is not ground truth for arbitrary new materials or missing geometry.")
    return response, report


def _native_geometry(task):
    name, shape_record = task
    shape = Shape(shape_record["primitive_scales_m"], shape_record["primitive_offsets_m"],
                  shape_record["primitive_exponents"], shape_record["primitive_yaws_rad"], shape_record["operations"],
                  shape_record["twist_rad_per_m"], shape_record["bend_per_m"], shape_record["taper_per_m"],
                  shape_record["surface_amplitude_m"], shape_record["surface_frequency_per_m"], shape_record["surface_phase_rad"])
    grounding = check_grounding(shape, coarse=dict(xy_resolution=33, z_steps=129, bisections=24, refinements=5),
        fine=dict(xy_resolution=65, z_steps=257, bisections=24, refinements=5),
        trace=Trace(96, 8, 24, 1e-5, 4., 1e-5, 1e-5, 1e-9), surface_count=128,
        residual_tolerance=1e-5, convergence_m=1e-4, buried_depth_m=1e-4, max_buried_fraction=0.)
    return name, grounding


def _native_scene(task):
    """Place a world once; observe its objects jointly from four original keyframes."""
    import json
    from pathlib import Path
    from scipy.spatial import cKDTree, ConvexHull
    scene_index, records, output = task
    destination = Path(output) / records[0]["scene"]
    metadata_path = destination / "world.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    rng = np.random.default_rng(41000 + scene_index)
    frames = [read_nuscenes(r, _native_mapping) for r in records]
    truth = [np.fromfile(r["label"], np.uint8) for r in records]
    ground_ids = [r["raw"] for r in _native_mapping if r["name"] in ("flat.driveable_surface", "flat.sidewalk", "flat.other")]
    world_points = [f.xyzi[:, :3].astype(float) @ f.pose[:3, :3].T + f.pose[:3, 3] for f in frames]
    obstacles = np.concatenate([p[f.actual & (f.range_m >= 2.5) & ~np.isin(t, ground_ids)]
                               for p, f, t in zip(world_points, frames, truth)])
    objects, proposals, diagnostics = [], [], []
    chosen = (1, 3, 5, 7)
    rays = {}
    for ordinal, frame_index in enumerate(chosen):
        source, points = frames[frame_index], world_points[frame_index]
        ray, diagnostic = nuscenes_rays(records[frame_index])
        rays[frame_index] = ray
        diagnostics.append(dict(token=records[frame_index]["token"], **diagnostic))
        geometry_name, grounding = _native_shapes[(scene_index * 4 + ordinal) % len(_native_shapes)]
        lower, upper = grounding.shape.bounds()
        footprint = float(np.linalg.norm(np.maximum(abs(lower[:2]), abs(upper[:2]))))
        ground = points[np.isin(truth[frame_index], ground_ids) & source.actual & (source.range_m >= 2.5)]
        tree = cKDTree(ground[:, :2]) if len(ground) else None
        available = np.flatnonzero(np.isin(truth[frame_index], ground_ids) & (source.range_m >= 4) & (source.range_m <= 35))
        accepted, reasons = None, dict(support=0, collision=0, visibility=0)
        for slot in rng.permutation(available)[:96]:
            anchor = points[slot]
            nearby = ground[tree.query_ball_point(anchor[:2], footprint + .3)]
            if len(nearby) < 12:
                reasons["support"] += 1
                continue
            center = nearby.mean(0)
            _, _, vh = np.linalg.svd(nearby - center, full_matrices=False)
            up = vh[-1] * (1 if vh[-1, 2] > 0 else -1)
            residual = abs((nearby - center) @ up)
            if up[2] < .95 or np.quantile(residual, .95) > .04:
                reasons["support"] += 1
                continue
            hull = ConvexHull(nearby[:, :2])
            # The complete circular footprint must be supported, not just its center.
            if np.max(hull.equations[:, :2] @ anchor[:2] + hull.equations[:, 2] + footprint) > 0:
                reasons["support"] += 1
                continue
            material = Material(float(rng.uniform(.1, .9)), float(rng.uniform(.1, .35)), 0.)
            item = ground_object(grounding, material, object_id=ordinal + 1, geometry_id=geometry_name,
                anchor_world=anchor, normal_world=up, plane_offset=-float(up @ center), yaw=float(rng.uniform(-np.pi, np.pi)))
            if len(observed_collision(item, obstacles, allowance_m=.03, gradient_step_m=1e-6, witness_fraction=1-1e-6)[0]):
                reasons["collision"] += 1
                continue
            lo, hi = item.bounds()
            if any(np.all(hi >= other.bounds()[0]) and np.all(other.bounds()[1] >= lo) for other in objects):
                reasons["collision"] += 1
                continue
            world = World(NATIVE_VERSION, 0, 41000 + scene_index, tuple(objects + [item]), 1e-6)
            observation = render_frame(source, world, ray, _native_response, _native_trace)
            if np.count_nonzero((observation.object_ids == item.object_id) & (point_targets(observation.frame) == 1)) < 5:
                reasons["visibility"] += 1
                continue
            accepted = item
            proposals.append(dict(object_id=item.object_id, geometry=geometry_name, anchor_frame=frame_index,
                support_points=len(nearby), support_p95_m=float(np.quantile(residual, .95)),
                normal_world=up.tolist(), pose=item.pose.tolist(), material=asdict(material), rejected=reasons))
            objects.append(item)
            break
        if accepted is None:
            proposals.append(dict(object_id=ordinal + 1, geometry=geometry_name, accepted=False, rejected=reasons))
    world = World(NATIVE_VERSION, 0, 41000 + scene_index, tuple(objects), 1e-6)
    destination.mkdir(parents=True, exist_ok=True)
    output_records, skipped = [], []
    for index, record in enumerate(records):
        if index not in chosen:
            output_records.append(record)
            continue
        observed = render_frame(frames[index], world, rays[index], _native_response, _native_trace)
        selected = supervision(observed.frame)
        if not selected.eligible:
            skipped.append(dict(token=record["token"], anomaly=selected.anomaly_count))
            continue
        slots = np.flatnonzero(observed.inserted | observed.occluded_original).astype(np.int32)
        path = destination / f"{index}.npz"
        np.savez_compressed(path, token=record["token"], slots=slots, xyzi=observed.frame.xyzi[slots],
                            labels=observed.frame.labels[slots], inserted=np.flatnonzero(observed.inserted),
                            occluded=np.flatnonzero(observed.occluded_original), object_ids=observed.object_ids[slots])
        output_records.append(dict(record, group="anomaly_nuscenes", delta=str(path.resolve()),
            geometry=[item.geometry_id for item in objects], world=str(metadata_path.resolve()),
            normal=selected.normal_count, anomaly=selected.anomaly_count, points=int(observed.frame.actual.sum())))
    result = dict(scene=records[0]["scene"], log_token=records[0]["log_token"], seed=world.seed,
                  objects=proposals, rays=diagnostics, records=output_records, skipped=skipped,
                  collision_scope="all measured non-ground points in the eight selected scans; unseen surfaces unknown")
    write_json(metadata_path, result)
    return result


def _expand_native_scene(task):
    """Observe one fixed world along its real trajectory, then remove near repeats."""
    import json
    from pathlib import Path
    from scipy.spatial import cKDTree
    records, baseline, destination = task
    destination = Path(destination)
    metadata_path = destination/"world.json"
    specification = identity(dict(records=records, baseline=baseline, context=_expanded_key))
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved.get("specification") != specification:
            raise ValueError("cached trajectory selection has different inputs or construction settings")
        return saved
    frames = [read_nuscenes(r, _native_mapping) for r in records]
    truth = [np.fromfile(r["label"], np.uint8) for r in records]
    inherited = json.loads(Path(next(r["world"] for r in baseline if r["anomaly"])).read_text())
    objects = [Object(o["object_id"], o["geometry"], _expanded_shapes[o["geometry"]].shape,
                      Material(**o["material"]), np.asarray(o["pose"]))
               for o in inherited["objects"] if o.get("accepted", True)]
    seed, proposals = inherited["seed"], inherited["objects"]
    world = World(NATIVE_VERSION, 0, seed, tuple(objects), 1e-6)
    old_normal = {r["token"] for r in baseline if not r["anomaly"]}
    old_anomaly = {r["token"]: r for r in baseline if r["anomaly"]}
    normal_rows, histograms = [], []
    for record, frame, raw in zip(records, frames, truth):
        targets = point_targets(frame)
        if (targets == 1).any() or not (targets == 0).any():
            raise ValueError("expanded native normal supervision is not trustworthy")
        normal_rows.append(dict(record, points=int(frame.actual.sum()), slots=len(frame.xyzi),
                                normal=int((targets == 0).sum()), anomaly=0))
        histograms.append(np.bincount(raw[targets == 0], minlength=32))
    keep_normal, _ = select_observations([normal_descriptor(f,t) for f,t in zip(frames,truth)],
        [i for i,r in enumerate(records) if r["token"] in old_normal], normal=True)
    extra_normal = [normal_rows[i] for i in keep_normal if records[i]["token"] not in old_normal]
    candidates, payloads, observations, skipped, ray_checks, sampling = [], {}, {}, [], [], []
    ground_ids = [r["raw"] for r in _native_mapping if r["name"] in ("flat.driveable_surface", "flat.sidewalk", "flat.other")]
    for record, source, raw in zip(records, frames, truth):
        token = record["token"]
        if token in old_anomaly:
            observed = read_nuscenes(old_anomaly[token], _native_mapping)
            with np.load(old_anomaly[token]["delta"], allow_pickle=False) as delta:
                owners = np.full(len(source.xyzi), -1, dtype=np.int64)
                owners[delta["slots"]] = delta["object_ids"]
            candidate = old_anomaly[token]
        else:
            # Bounds only skip frames that cannot contain supervised anomaly hits.
            if not any(np.linalg.norm(source.pose[:3,3]-o.pose[:3,3])-o.shape.radius <= 50 for o in objects):
                skipped.append(dict(token=token, reason="outside_supervised_range"))
                continue
            obstacle = source.xyzi[source.actual & (source.range_m >= 2.5) & ~np.isin(raw, ground_ids), :3].astype(float)
            obstacle = obstacle @ source.pose[:3,:3].T+source.pose[:3,3]
            collisions = [o.object_id for o in objects if len(observed_collision(o, obstacle,
                allowance_m=.03, gradient_step_m=1e-6, witness_fraction=1-1e-6)[0])]
            if collisions:
                skipped.append(dict(token=token, reason="observed_collision", objects=collisions))
                continue
            rays, diagnostic = nuscenes_rays(record)
            result = render_frame(source, world, rays, _native_response, _native_trace)
            ray_checks.append(dict(token=token, **diagnostic))
            sampling.append(dict(token=token, objects=result.sampling))
            selected = supervision(result.frame)
            if not selected.eligible:
                skipped.append(dict(token=token, reason="zero_or_1_to_4_anomalies", anomaly=selected.anomaly_count))
                continue
            observed, owners = result.frame, result.object_ids
            slots = np.flatnonzero(result.inserted | result.occluded_original).astype(np.int32)
            path = destination/f"{record['frame']}.npz"
            payloads[token] = dict(token=token, slots=slots, xyzi=observed.xyzi[slots], labels=observed.labels[slots],
                inserted=np.flatnonzero(result.inserted), occluded=np.flatnonzero(result.occluded_original), object_ids=owners[slots])
            candidate = dict(record, group="anomaly_nuscenes", delta=str(path.resolve()), world=str(metadata_path.resolve()),
                geometry=[o.geometry_id for o in objects], points=int(observed.actual.sum()), slots=len(observed.xyzi),
                normal=selected.normal_count, anomaly=selected.anomaly_count)
        candidate_index = len(candidates)
        candidates.append(candidate)
        normal_ids = np.flatnonzero(observed.actual & (owners < 0))
        tree = cKDTree(observed.xyzi[normal_ids, :3])
        groups = context_groups(raw, _native_mapping)
        targets = point_targets(observed)
        for obj in objects:
            mask = (owners == obj.object_id) & (targets == 1)
            if not mask.any():
                continue
            center = (obj.pose[:3,3]-source.pose[:3,3]) @ source.pose[:3,:3]
            neighbors = normal_ids[tree.query_ball_point(center, DIVERSITY["context_radius_m"])]
            descriptor = observation_descriptor(observed.xyzi[mask], obj.pose, source.pose,
                                               np.bincount(groups[neighbors], minlength=8))
            observations.setdefault(obj.object_id, []).append(dict(index=candidate_index, token=token,
                geometry=obj.geometry_id, count=int(mask.sum()), descriptor=descriptor))
    chosen = {i for i,r in enumerate(candidates) if r["token"] in old_anomaly}
    objects_report = []
    for number, rows in observations.items():
        keep, _ = select_observations([r["descriptor"] for r in rows],
                                     [i for i,r in enumerate(rows) if r["token"] in old_anomaly])
        chosen.update(rows[i]["index"] for i in keep)
        objects_report.append(dict(object_id=number, geometry=rows[0]["geometry"], candidates=len(rows),
                                   observations=rows))
    selected_tokens = {candidates[i]["token"] for i in chosen}
    for row in objects_report:
        row["retained"] = sum(r["token"] in selected_tokens for r in row["observations"])
    destination.mkdir(parents=True, exist_ok=True)
    extra_anomaly = []
    for i in sorted(chosen):
        record = candidates[i]
        if record["token"] not in old_anomaly:
            np.savez_compressed(record["delta"], **payloads[record["token"]])
            extra_anomaly.append(record)
    result = dict(scene=records[0]["scene"], log_token=records[0]["log_token"], seed=seed, specification=specification,
        objects=proposals, records=extra_normal+extra_anomaly, tolerances=DIVERSITY,
        candidate_keyframes=len(records), eligible_anomaly_frames=len(candidates),
        retained_anomaly_frames=len(chosen), retained_normal_frames=len(keep_normal),
        normal_semantic_counts=np.sum(np.asarray(histograms)[keep_normal], axis=0).tolist(),
        observation_selection=objects_report, skipped=skipped, rays=ray_checks, sampling=sampling,
        collision_scope="all newly used original scans; unseen geometry remains unknown",
        sampling_definition="Potential surface rays precede external occlusion; foreground rays precede response. Counts cover all ranges, not visible surface area.")
    write_json(metadata_path, result)
    return result


def expand_native(output, reference_path, workers, limit=None):
    import json
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from collections import Counter
    global _native_mapping, _native_response, _native_trace, _expanded_shapes, _expanded_key
    output, reference_path = Path(output), Path(reference_path)
    reference = load_manifest(reference_path, "train")
    output.mkdir(parents=True, exist_ok=True)
    stu = expand_stu(reference, output, workers)
    pose_path = output/"poses.json"
    if not pose_path.exists():
        write_json(pose_path, dict(reference=reference["sha256"], records=native_keyframes(reference)))
    poses = json.loads(pose_path.read_text())
    if poses["reference"] != reference["sha256"]:
        raise ValueError("expanded trajectories belong to another base dataset")
    _native_mapping = reference["mapping"]
    response = json.loads((reference_path.parent/"response.json").read_text())
    _native_response = Response(response["range_edges"], response["incidence_edges"], response["quantiles"],
        response["probability"], response["intensity"], (0.,1.), 1/255, "unchanged native training response")
    previous_geometry = json.loads((reference_path.parent/"geometry.json").read_text())
    _native_trace = Trace(**previous_geometry["trace"])
    _expanded_shapes = {r["id"]: Grounding(Shape(**r["shape"]), r["lower_z"], r["refined_lower_z"],
        r["buried_fraction"], np.empty((0,3)), True) for r in previous_geometry["records"]}
    _expanded_key = identity(dict(reference=reference["sha256"], response=response, trace=asdict(_native_trace),
        geometry=previous_geometry, tolerances=DIVERSITY,
        construction="fixed existing worlds, all labeled keyframes, unchanged geometry and response"))
    scenes, baseline = {}, {}
    for r in poses["records"]:
        scenes.setdefault(r["scene"], []).append(r)
    for r in reference["records"]:
        if r.get("source") == "nuscenes":
            baseline.setdefault(r["scene"], []).append(r)
    ordered = sorted(scenes)
    if limit is not None:
        ordered = ordered[:limit]
    tasks = [(sorted(scenes[s], key=lambda r:r["timestamp"]), baseline[s], str(output/s))
             for s in ordered]
    completed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        for row in pool.map(_expand_native_scene, tasks):
            completed.append(row)
            if len(completed) % 50 == 0 or len(completed) == len(tasks):
                print(json.dumps(dict(scenes=len(completed), additions=sum(len(r["records"]) for r in completed))), flush=True)
    if limit is not None:
        return
    old_stu = {(r["world"],r["frame"]) for r in reference["records"] if r["group"] == "anomaly_stu"}
    records = reference["records"]+[r for r in stu["records"] if (r["world"],r["frame"]) not in old_stu]
    records += [r for row in completed for r in row["records"]]
    if any(1 <= r["anomaly"] <= 4 or r["normal"]+r["anomaly"] == 0 for r in records):
        raise ValueError("ineligible expanded training scan")
    result = dict(reference, records=records, selection=stu["worlds"],
        representative_rule="preserve all base records; retain measured changes without frame quotas",
        dataset_revision="expanded_observations", expansion=dict(reference=str(reference_path), reference_identity=reference["sha256"],
        tolerances=DIVERSITY, method="retain baseline; add measured changes without per-world or per-scene frame caps",
        stu_selection=str((output/"stu.json").resolve()), geometry=str((reference_path.parent/"geometry.json").resolve()),
        trajectories=str(pose_path.resolve()), new_geometry=0,
        independent_train_logs=len({r["log_token"] for r in records if r.get("source") == "nuscenes"}),
        groups=dict(Counter(r["group"] for r in records)), base_records=len(reference["records"]),
        train_records=len(records), complete_pass_visits=len(records), two_pass_updates=math.ceil(2*len(records)/8)))
    result["base_native_manifest"] = result.pop("native_manifest")
    result.pop("sha256")
    result["sha256"] = identity(result)
    write_json(output/"train.json", result, indent=None)
    print(json.dumps(result["expansion"]), flush=True)


def generate_native(output, *, pool_root, workers=8, limit=None, extend_from=None):
    if extend_from is not None:
        return expand_native(output, extend_from, workers, limit)
    import json
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    global _native_mapping, _native_shapes, _native_response, _native_trace
    output, pool_root = Path(output), Path(pool_root)
    normal = json.loads((output.parent / "background.json").read_text())
    pose_path = output / "poses.json"
    if not pose_path.exists():
        write_json(pose_path, dict(records=nuscenes_poses(normal["records"]), source_manifest=normal["sha256"]))
    poses = json.loads(pose_path.read_text())
    if poses["source_manifest"] != normal["sha256"]:
        raise ValueError("native poses belong to another background pool")
    _native_mapping = normal["mapping"]
    response_path = output / "response.json"
    if response_path.exists():
        response = json.loads(response_path.read_text())
        _native_response = Response(response["range_edges"], response["incidence_edges"], response["quantiles"],
            response["probability"], response["intensity"], (0., 1.), 1 / 255, "saved nuScenes training-only planar response")
    else:
        _native_response, response = fit_native_response(poses["records"], normal["mapping"], workers)
        write_json(response_path, response)
    base = json.loads(Path("assets/train.json").read_text())
    shapes = {}
    for world in base["worlds"]:
        saved = json.loads((pool_root / world["paths"][0] / "world.json").read_text())
        for item in saved["world"]["objects"]:
            shape = item["shape"]
            shapes.setdefault(identity(shape), shape)
    _native_trace = Trace(96, 8, 24, 1e-5, 4., 1e-5, 1e-5, 1e-9)
    geometry_path = output / "geometry.json"
    cached = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    if cached is not None and cached.get("source_shapes") == sorted(shapes) and cached["trace"] == asdict(_native_trace):
        _native_shapes = [(r["id"], Grounding(Shape(**r["shape"]), r["lower_z"], r["refined_lower_z"],
                           r["buried_fraction"], np.empty((0, 3)), True)) for r in cached["records"]]
        rejected_shapes = cached["rejected"]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            checked = list(pool.map(_native_geometry, sorted(shapes.items())))
        _native_shapes = [(name, grounding) for name, grounding in checked if grounding.accepted]
        rejected_shapes = [name for name, grounding in checked if not grounding.accepted]
    if not _native_shapes:
        raise ValueError("no reused geometry passed direct grounding checks")
    write_json(geometry_path, dict(records=[dict(id=name, shape=asdict(g.shape), lower_z=g.lower_z,
        refined_lower_z=g.refined_lower_z, buried_fraction=g.buried_fraction) for name, g in _native_shapes],
        candidate_geometries=len(shapes), source_shapes=sorted(shapes), rejected=rejected_shapes, trace=asdict(_native_trace)))
    scenes = {}
    for record in poses["records"]:
        scenes.setdefault(record["scene"], []).append(record)
    tasks = [(i, sorted(records, key=lambda r: r["timestamp"]), str(output)) for i, (_, records) in enumerate(sorted(scenes.items()))]
    if any(len(records) != 8 for _, records, _ in tasks):
        raise ValueError("the 4+4 native design requires exactly eight selected keyframes per scene")
    if limit is not None:
        tasks = tasks[:limit]
    completed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        for row in pool.map(_native_scene, tasks):
            completed.append(row)
            print(json.dumps(dict(scene=row["scene"], scans=len(row["records"]), skipped=len(row["skipped"]))), flush=True)
    result = dict(version=NATIVE_VERSION, kind="native", mapping=normal["mapping"], split=normal["split"],
                  normal_manifest=normal["sha256"], scenes=[r["scene"] for r in completed],
                  records=[r for row in completed for r in row["records"]],
                  skipped=[r for row in completed for r in row["skipped"]],
                  response=str(response_path.resolve()), geometry=str((output / "geometry.json").resolve()))
    result["sha256"] = identity(result)
    write_json(output / "manifest.json", result)


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Build a bounded 206 normal-reference candidate batch.")
    parser.add_argument("--output", type=Path, default=Path("results/data"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--pool-root", type=Path, default=Path("/home/jasongao/Study/AJAE/results/synthetic"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--extend-from", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers must fit the current CPU affinity")
    if args.native:
        generate_native(args.output, pool_root=args.pool_root, workers=args.workers, limit=args.limit, extend_from=args.extend_from)
    else:
        generate_candidates(args.output, data_root=args.data_root, pool_root=args.pool_root, workers=args.workers)
