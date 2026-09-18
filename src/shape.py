"""Selected implicit geometry and intersection calculations from AJAE/src/render.py.

Shape parameters and numerical resolutions are explicit; no legacy shape sampler,
size distribution, connectivity acceptance or grounding threshold is a V4 default.
"""

from dataclasses import dataclass
import math

import numpy as np

from .data import readonly


@dataclass(frozen=True, slots=True)
class Trace:
    """Explicit ray sampling, refinement and normal-estimation accuracy controls."""

    steps: int
    adaptive_depth: int
    bisections: int
    near_m: float
    proximity_factor: float
    gradient_step_m: float
    gradient_relative: float
    gradient_min: float

    def __post_init__(self):
        for name, minimum in (("steps", 2), ("adaptive_depth", 0), ("bisections", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"invalid trace {name}")
        for name in ("near_m", "proximity_factor", "gradient_step_m", "gradient_relative", "gradient_min"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"trace {name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class Shape:
    """CSG superquadrics with the original twist, bend, taper and surface modulation.

The level is a scaled implicit function, NOT a Euclidean signed distance.
    These parameters describe one supplied shape; they do not define a V4 distribution.
    Scales are half extents in metres; offsets are local metres; yaws are radians.
    Each exponent pair is (vertical, horizontal). Taper retains the original
    normalized-z convention and [0.25, 4] factor clamp as part of the geometry.
"""

    scales: tuple
    offsets: tuple
    exponents: tuple
    yaws: tuple
    operations: tuple
    twist: float
    bend: tuple
    taper: tuple
    amplitude: float
    frequency: tuple
    phase: tuple

    def __post_init__(self):
        count = len(self.scales)
        if count == 0 or len(self.operations) != count or self.operations[0] != "union":
            raise ValueError("a shape must start with a union primitive")
        if any(op not in ("union", "difference", "intersection") for op in self.operations):
            raise ValueError("unknown CSG operation")
        for name, shape in (("scales", (count, 3)), ("offsets", (count, 3)),
                            ("exponents", (count, 2)), ("yaws", (count,)),
                            ("bend", (2,)), ("taper", (2,)), ("frequency", (3,)), ("phase", (3,))):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid shape {name}")
            value = tuple(map(tuple, array)) if array.ndim == 2 else tuple(array)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "operations", tuple(self.operations))
        if np.any(np.asarray(self.scales) <= 0) or np.any(np.asarray(self.exponents) <= 0):
            raise ValueError("primitive scales and exponents must be positive")
        if not np.isfinite((self.twist, self.amplitude)).all() or self.amplitude < 0:
            raise ValueError("invalid deformation parameters")

    def undeform(self, points):
        result = np.asarray(points, dtype=np.float64).copy()
        z = result[..., 2]
        scale_z = max(item[2] for item in self.scales)
        x = (result[..., 0] - self.bend[0] * z**2) / np.clip(1 + self.taper[0] * z / scale_z, .25, 4)
        y = (result[..., 1] - self.bend[1] * z**2) / np.clip(1 + self.taper[1] * z / scale_z, .25, 4)
        cosine, sine = np.cos(-self.twist * z), np.sin(-self.twist * z)
        result[..., 0], result[..., 1] = cosine * x - sine * y, sine * x + cosine * y
        return result

    def deform(self, points):
        result = np.asarray(points, dtype=np.float64).copy()
        z = result[..., 2]
        x, y = result[..., 0].copy(), result[..., 1].copy()
        cosine, sine = np.cos(self.twist * z), np.sin(self.twist * z)
        scale_z = max(item[2] for item in self.scales)
        result[..., 0] = np.clip(1 + self.taper[0] * z / scale_z, .25, 4) * (cosine * x - sine * y) + self.bend[0] * z**2
        result[..., 1] = np.clip(1 + self.taper[1] * z / scale_z, .25, 4) * (sine * x + cosine * y) + self.bend[1] * z**2
        return result

    def level(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("shape queries must be finite [...,3]")
        points = self.undeform(points)
        result = None
        for scale, offset, exponents, yaw, operation in zip(
            self.scales, self.offsets, self.exponents, self.yaws, self.operations, strict=True
        ):
            local = points - offset
            cosine, sine = math.cos(yaw), math.sin(yaw)
            x = cosine * local[..., 0] + sine * local[..., 1]
            y = -sine * local[..., 0] + cosine * local[..., 1]
            vertical, horizontal = exponents
            xy = (np.abs(x / scale[0]) ** (2 / horizontal) + np.abs(y / scale[1]) ** (2 / horizontal)) ** (horizontal / vertical)
            value = ((xy + np.abs(local[..., 2] / scale[2]) ** (2 / vertical)) ** (vertical / 2) - 1) * min(scale)
            if result is None:
                result = value
            elif operation == "union":
                result = np.minimum(result, value)
            else:
                result = np.maximum(result, -value if operation == "difference" else value)
        if self.amplitude:
            result -= self.amplitude * np.sin(points * self.frequency + self.phase).mean(axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite implicit level; revise the supplied geometry")
        return result

    def bounds(self):
        """Conservative outer box propagated through CSG and the exact deformation."""
        lower = upper = None
        for scale, offset, yaw, operation in zip(self.scales, self.offsets, self.yaws, self.operations, strict=True):
            a, b, c = (1 + self.amplitude / min(scale)) * np.asarray(scale)
            cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
            half = np.asarray((cosine * a + sine * b, sine * a + cosine * b, c))
            lo, hi = np.asarray(offset) - half, np.asarray(offset) + half
            if lower is None:
                lower, upper = lo, hi
            elif operation == "union":
                lower, upper = np.minimum(lower, lo), np.maximum(upper, hi)
            elif operation == "intersection":
                lower, upper = np.maximum(lower, lo), np.minimum(upper, hi)
        if np.any(lower >= upper):
            raise ValueError("empty CSG outer bounds")
        z = np.array((lower[2], upper[2]))
        if self.twist != 0:
            radial = np.linalg.norm(np.maximum(np.abs(lower[:2]), np.abs(upper[:2])))
            lower[:2], upper[:2] = -radial, radial
        z_squared = (0 if z[0] <= 0 <= z[1] else np.min(z**2), np.max(z**2))
        scale_z = max(item[2] for item in self.scales)
        for axis in (0, 1):
            factors = np.clip(1 + self.taper[axis] * z / scale_z, .25, 4)
            products = np.array((lower[axis], upper[axis]))[:, None] * factors
            bends = self.bend[axis] * np.asarray(z_squared)
            lower[axis], upper[axis] = products.min() + bends.min(), products.max() + bends.max()
        if not np.isfinite((lower, upper)).all():
            raise ValueError("nonfinite deformed bounds")
        return lower, upper

    @property
    def radius(self):
        lower, upper = self.bounds()
        # Unlike a heuristic deformation radius, the outer box encloses all offsets.
        return float(np.nextafter(np.linalg.norm(np.maximum(abs(lower), abs(upper))), np.inf))

    def intersect(self, origins, directions, trace):
        """Find entry/exit sign changes with the legacy adaptive sampling algorithm.

Finite sampling is not a proof of finding every arbitrarily thin component.
The caller must validate the chosen resolution for its supplied geometry.
"""
        origins, directions = np.asarray(origins, np.float64), np.asarray(directions, np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("ray directions must be [N,3]")
        origins = np.broadcast_to(origins, directions.shape)
        norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(origins).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("rays must be finite and nonzero")
        directions = directions / norms[:, None]
        radius = self.radius
        projection = np.sum(origins * directions, axis=1)
        discriminant = projection**2 - (np.sum(origins**2, axis=1) - radius**2)
        root = np.sqrt(np.maximum(discriminant, 0))
        near, far = np.maximum(-projection - root, trace.near_m), -projection + root
        candidates = np.flatnonzero((discriminant >= 0) & (far > near))
        distance, normals = np.full(len(directions), np.inf), np.zeros_like(directions)
        fractions = np.linspace(0, 1, trace.steps)
        for start in range(0, len(candidates), 2048):
            ids = candidates[start:start + 2048]
            times = near[ids, None] + (far[ids] - near[ids])[:, None] * fractions
            values = self.level(origins[ids, None] + times[..., None] * directions[ids, None])
            inside = values <= 0
            starts_inside = inside[:, 0]
            crossing = np.where(starts_inside[:, None], inside[:, :-1] & ~inside[:, 1:], ~inside[:, :-1] & inside[:, 1:])
            hit = crossing.any(axis=1)
            lo, hi = np.full(len(ids), np.inf), np.full(len(ids), np.inf)
            rows = np.flatnonzero(hit)
            first = np.argmax(crossing[rows], axis=1)
            lo[rows], hi[rows] = times[rows, first], times[rows, first + 1]
            # Proximity only selects intervals to refine; only a sign change is a hit.
            rows = np.flatnonzero(~starts_inside)
            ray = np.repeat(rows, trace.steps - 1)
            left, right = times[rows, :-1].ravel(), times[rows, 1:].ravel()
            vl, vr = values[rows, :-1].ravel(), values[rows, 1:].ravel()
            for _ in range(trace.adaptive_depth):
                keep = (vl > 0) & (vr > 0) & (np.minimum(vl, vr) <= trace.proximity_factor * (right - left)) & (left < lo[ray])
                ray, left, right, vl, vr = (array[keep] for array in (ray, left, right, vl, vr))
                if not len(ray):
                    break
                middle = .5 * (left + right)
                vm = self.level(origins[ids[ray]] + middle[:, None] * directions[ids[ray]])
                crossed = vm <= 0
                for candidate in np.flatnonzero(crossed):
                    row = ray[candidate]
                    if left[candidate] < lo[row]:
                        lo[row], hi[row] = left[candidate], middle[candidate]
                outside = ~crossed
                ray = np.concatenate((ray[outside], ray[outside]))
                left, right, vl, vr = (
                    np.concatenate((left[outside], middle[outside])),
                    np.concatenate((middle[outside], right[outside])),
                    np.concatenate((vl[outside], vm[outside])),
                    np.concatenate((vm[outside], vr[outside])),
                )
            hit = np.isfinite(lo)
            hit_ids = ids[hit]
            lo, hi, starts = lo[hit], hi[hit], starts_inside[hit]
            for _ in range(trace.bisections):
                middle = .5 * (lo + hi)
                middle_inside = self.level(origins[hit_ids] + middle[:, None] * directions[hit_ids]) <= 0
                advance = middle_inside == starts
                lo, hi = np.where(advance, middle, lo), np.where(advance, hi, middle)
            distance[hit_ids] = .5 * (lo + hi)
        valid = np.isfinite(distance)
        ids = np.flatnonzero(valid)
        points = origins[ids] + distance[ids, None] * directions[ids]
        delta = max(trace.gradient_step_m, radius * trace.gradient_relative)
        offsets = delta * np.eye(3)
        gradient = (self.level(points[:, None] + offsets) - self.level(points[:, None] - offsets)) / (2 * delta)
        length = np.linalg.norm(gradient, axis=1)
        keep = np.isfinite(length) & (length > trace.gradient_min)
        distance[ids[~keep]], valid[ids[~keep]] = np.inf, False
        normals[ids[keep]] = gradient[keep] / length[keep, None]
        return readonly(distance), readonly(normals), readonly(valid)

    def minimum_z(self, *, xy_resolution, z_steps, bisections, refinements):
        """Numerical support search; convergence must be checked at a finer resolution."""
        if any(type(v) is not int for v in (xy_resolution, z_steps, bisections, refinements)):
            raise ValueError("support search resolutions must be integers")
        if xy_resolution < 3 or xy_resolution % 2 == 0 or z_steps < 3 or bisections < 1 or refinements < 0:
            raise ValueError("invalid support search resolutions")
        radius = self.radius
        z_axis = np.linspace(-radius, radius, z_steps)

        def roots(xy):
            points = np.empty((len(xy), z_steps, 3))
            points[..., :2], points[..., 2] = xy[:, None], z_axis
            inside = self.level(points) <= 0
            first = np.argmax(inside, axis=1)
            ids = np.flatnonzero(inside.any(axis=1) & (first > 0))
            values = np.full(len(xy), np.inf)
            lo, hi = z_axis[first[ids] - 1], z_axis[first[ids]]
            for _ in range(bisections):
                middle = .5 * (lo + hi)
                found = self.level(np.column_stack((xy[ids], middle))) <= 0
                lo, hi = np.where(found, lo, middle), np.where(found, middle, hi)
            values[ids] = .5 * (lo + hi)
            return values

        axis = np.linspace(-radius, radius, xy_resolution)
        xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        values = roots(xy)
        if not np.isfinite(values).any():
            raise ValueError("support search missed the shape; revise geometry or resolution")
        index = np.argmin(values)
        center, best = xy[index], float(values[index])
        step = 2 * radius / (xy_resolution - 1)
        for _ in range(refinements):
            offsets = np.linspace(-step, step, 5)
            candidates = center + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
            values = roots(candidates)
            index = np.argmin(values)
            if values[index] < best:
                center, best = candidates[index], float(values[index])
            step *= .25
        return best

    def surface_points(self, count, trace, residual_tolerance):
        """Sample radial first surfaces, not a complete mesh or connectivity certificate."""
        if type(count) is not int or count < 1 or not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("invalid surface sampling parameters")
        ids = np.arange(count)
        z, angle = 1 - 2 * (ids + .5) / count, math.pi * (3 - math.sqrt(5)) * ids
        radial = np.sqrt(np.maximum(0, 1 - z**2))
        direction = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))
        origins = 1.05 * self.radius * direction
        distance, _, valid = self.intersect(origins, -direction, trace)
        if not valid.all():
            raise ValueError("radial surface sampling missed geometry; cannot qualify this shape")
        points = origins - distance[:, None] * direction
        if np.max(np.abs(self.level(points))) > residual_tolerance:
            raise ValueError("surface sampling exceeds the supplied implicit residual tolerance")
        return readonly(points)


def unresolved_penetration(shape, points, *, allowance_m, gradient_step_m, witness_fraction):
    """Reject interiors without an exterior witness within a Euclidean allowance.

An exterior probe certifies a boundary crossing within the allowance. Failure to
find one is uncertainty, not a measurement of penetration depth.
"""
    if not np.isfinite((allowance_m, gradient_step_m, witness_fraction)).all() or allowance_m < 0 or gradient_step_m <= 0 or not 0 < witness_fraction < 1:
        raise ValueError("invalid penetration witness parameters")
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("penetration queries must be [N,3]")
    values = shape.level(points)
    unresolved = values < 0
    if allowance_m == 0:
        return unresolved, values
    directions = np.array([(x, y, z) for x in (-1., 0., 1.) for y in (-1., 0., 1.) for z in (-1., 0., 1.) if x or y or z])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ids = np.flatnonzero(unresolved)
    for start in range(0, len(ids), 2048):
        selected = ids[start:start + 2048]
        local, offsets = points[selected], gradient_step_m * np.eye(3)
        gradient = shape.level(local[:, None] + offsets) - shape.level(local[:, None] - offsets)
        norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        gradient = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm > 0)
        vectors = np.concatenate((np.broadcast_to(directions, (len(local), len(directions), 3)), gradient[:, None]), axis=1)
        probes = local[:, None] + allowance_m * witness_fraction * vectors
        exterior = shape.level(probes) > 0
        within = np.linalg.norm(probes - local[:, None], axis=-1) <= allowance_m
        unresolved[selected] = ~np.any(exterior & within, axis=1)
    return unresolved, values
