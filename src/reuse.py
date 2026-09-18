"""Measure existing AJAE/V3 samples and index eligible 206 observations.

No geometry, signal parameters, or training distribution is generated here.
Historical collision allowances are replayed as historical diagnostics only.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.spatial import cKDTree

from .analyze import REFERENCES, quantiles, write_csv
from .data import (STUSequence, legacy_source_identity, point_targets, read_delta,
                   read_rays, validate_delta)
from .shape import Shape, unresolved_penetration


def load_worlds(root, mirror):
    manifest = json.loads((root / "manifest.json").read_text())
    if (root / "manifest.json").read_bytes() != (mirror / "manifest.json").read_bytes():
        raise ValueError("main/V3 inventories differ; inspect both pools separately")
    configs = [manifest["configuration"]] + [v["configuration"] for v in manifest["generation_configs"].values()]
    allowances = sorted({c["placement"]["deep_penetration_m"] for c in configs})
    if len(allowances) != 1:
        raise ValueError("legacy collision allowances differ; resolve per-world provenance")
    worlds, inventory = [], []
    for split, part in manifest["splits"].items():
        for entry in part["worlds"]:
            directory = root / entry["path"]
            for name in ("world.json", "manifest.json"):
                if (directory / name).read_bytes() != (mirror / entry["path"] / name).read_bytes():
                    raise ValueError(f"main/V3 world metadata differs: {entry['path']}")
            definition = json.loads((directory / "world.json").read_text())
            measured = json.loads((directory / "manifest.json").read_text())
            world, generation = definition["world"], definition["generation"]
            expected = [f"{row['frame']:06d}.npz" for row in measured["frames"]]
            if sorted(p.name for p in (directory / "frames").glob("*.npz")) != expected:
                raise ValueError(f"incomplete saved world: {entry['path']}")
            if sorted(p.name for p in (mirror / entry["path"] / "frames").glob("*.npz")) != expected:
                raise ValueError(f"incomplete V3 world: {entry['path']}")
            if len(world["objects"]) != 1 or world["objects"][0]["object_id"] != 1:
                raise ValueError("expected the existing single-object worlds")
            obj = world["objects"][0]
            s = obj["shape"]
            shape = Shape(s["primitive_scales_m"], s["primitive_offsets_m"], s["primitive_exponents"],
                          s["primitive_yaws_rad"], s["operations"], s["twist_rad_per_m"],
                          s["bend_per_m"], s["taper_per_m"], s["surface_amplitude_m"],
                          s["surface_frequency_per_m"], s["surface_phase_rad"])
            geom = generation["geometry"]
            item = dict(path=entry["path"], source=part["source_sequence"],
                        world_id=entry["world_identity"], parent=entry.get("family_id"),
                        shape=shape, shape_key=json.dumps(s, sort_keys=True),
                        rotation=np.asarray(obj["rotation_world_from_local"]),
                        translation=np.asarray(obj["translation_world_m"]), radius=shape.radius,
                        frames=measured["frames"], generation=generation)
            reference = generation.get("normal_reference")
            plane = generation["support_plane"]
            normal = np.asarray(plane["normal_world"])
            # Existing unions have only primitive yaws: their local z minimum is analytic.
            simple = not (shape.twist or shape.amplitude or any(shape.bend) or any(shape.taper)) and set(shape.operations) == {"union"}
            contact = None
            if simple:
                zmin = min(o[2] - a[2] for o, a in zip(shape.offsets, shape.scales))
                contact = float((np.dot(item["translation"], normal) + plane["offset"] + zmin) / np.linalg.norm(normal))
            row = dict(world=entry["path"], source_sequence=item["source"],
                       world_identity=item["world_id"],
                       historical_combination=entry["combination"], historical_shape_family=generation["shape_family"],
                       geometry_parent=item["parent"], length_local_m=geom["length_m"],
                       width_local_m=geom["width_m"], height_local_m=geom["height_m"],
                       primitives=len(shape.scales), continuous_deformation=not simple,
                       vertical_exponent_min=min(e[0] for e in shape.exponents),
                       vertical_exponent_max=max(e[0] for e in shape.exponents),
                       horizontal_exponent_min=min(e[1] for e in shape.exponents),
                       horizontal_exponent_max=max(e[1] for e in shape.exponents),
                       yaw_rad=generation["yaw_rad"],
                       intensity_quantile=obj["material"]["intensity_quantile"],
                       material_roughness=obj["material"]["roughness"],
                       support_frame=generation["placement"]["support_frame"],
                       support_semantic=generation["placement"]["support_semantic"],
                       support_plane_rmse_m=generation["plane_rmse_m"], contact_residual_m=contact,
                       local_z_support_angle_deg=float(np.degrees(np.arccos(np.clip(np.dot(item["rotation"][:, 2], normal) / np.linalg.norm(normal), -1, 1)))),
                       historical_reference_frame=None if reference is None else reference["frame"],
                       historical_reference_points=0 if reference is None else len(reference["slots"]),
                       stored_frames=len(expected), source_binding_verified=False,
                       historical_eligible_frames=sum(f["in_range"] >= 5 for f in measured["frames"]),
                       historical_one_to_four_frames=sum(1 <= f["in_range"] <= 4 for f in measured["frames"]),
                       compared_files=0)
            inventory.append(row)
            if split == "train":
                if item["source"] != 206 or [f["frame"] for f in item["frames"]] != list(range(449)):
                    raise ValueError("training pool is not the complete 206 source")
                worlds.append(item)
            else:
                # Other source sequences remain outside the 206 training pool.
                row["main_v3_equal"] = all((directory / "frames" / name).read_bytes() ==
                    (mirror / entry["path"] / "frames" / name).read_bytes() for name in expected)
                row["compared_files"] = len(expected) if row["main_v3_equal"] else None
    return worlds, inventory, allowances[0]


def initialize(root, mirror, data_root, worlds, allowance):
    global _root, _mirror, _sequence, _worlds, _rays, _allowance
    _root, _mirror, _sequence, _worlds = root, mirror, STUSequence(data_root), worlds
    _rays, _allowance = read_rays(), allowance


def analyze_frame(frame_id):
    source = _sequence[frame_id]
    binding = legacy_source_identity(source)
    targets = point_targets(source)
    normal = targets == 0
    native_depth = np.sum((source.xyzi[:, :3].astype(float) - _rays.origins) * _rays.directions, axis=1)
    native_depth[~source.actual] = np.inf
    # Frame-local instance groups are candidates, not claims of reliable temporal identity.
    instance_keys = (source.semantic.astype(np.uint32) << np.uint32(16)) | source.instance
    groups = {int(key): np.flatnonzero(normal & (instance_keys == key))
              for key in np.unique(instance_keys[normal & (source.instance != 0)])}
    obstacles = np.flatnonzero(source.actual & (source.semantic != 0) & ~np.isin(source.semantic, (40, 44, 48, 49, 60)))
    world_points = source.xyzi[obstacles, :3].astype(float) @ source.pose[:3, :3].T + source.pose[:3, 3]
    tree = cKDTree(world_points)
    neighborhoods = tree.query_ball_point(np.asarray([w["translation"] for w in _worlds]),
                                         np.asarray([w["radius"] for w in _worlds]))
    rows, equivalent = [], {}
    for w, nearby in zip(_worlds, neighborhoods):
        relative = Path(w["path"]) / "frames" / f"{frame_id:06d}.npz"
        content = (_root / relative).read_bytes()
        duplicate = content == (_mirror / relative).read_bytes()
        delta = read_delta(BytesIO(content))
        validate_delta(delta, source, w["world_id"], source_identity=binding)
        inserted, occluded = delta["inserted_slot"], delta["occluded_slot"]
        anomaly = delta["xyzi"][np.searchsorted(delta["source_slot"], inserted)]
        distances = np.linalg.norm(anomaly[:, :3], axis=1)
        valid = (distances >= 2.5) & (distances <= 50)
        valid_ranges = distances[valid]
        count = int(valid.sum())
        duplicate_of = None
        if count >= 5:
            # Ignore world names: identical changes to the same source are one scan.
            key = tuple(delta[k].tobytes() for k in ("source_slot", "xyzi", "packed_labels"))
            duplicate_of = equivalent.get(key)
            equivalent.setdefault(key, w["path"])
        old = w["frames"][frame_id]
        changed_normal = int(normal[occluded].sum())
        replaced = int(np.count_nonzero(source.actual[inserted]))
        # Count untouched original points once, then subtract exactly the changed slots.
        row = dict(world=w["path"], frame=frame_id, anomaly_points=count, eligible=count >= 5,
                   duplicate_of=duplicate_of, selected=count >= 5 and duplicate_of is None,
                   anomaly_points_all=len(inserted), normal_points=int(normal.sum()) - changed_normal,
                   range_min_m=float(valid_ranges.min()) if count else None,
                   range_median_m=float(np.median(valid_ranges)) if count else None,
                   range_max_m=float(valid_ranges.max()) if count else None,
                   intensity_min=float(anomaly[valid, 3].min()) if count else None,
                   intensity_max=float(anomaly[valid, 3].max()) if count else None,
                   occluded_original=len(occluded), occluded_valid_normal=changed_normal,
                   replaced_original=replaced, lost_original=len(occluded) - replaced,
                   new_return_in_empty_slot=len(inserted) - replaced,
                   main_v3_equal=duplicate,
                   historical_count_equal=count == old["in_range"] and len(inserted) == old["count"] and len(occluded) == old["occluded"],
                   ray_residual_max_m=0., ray_roundoff_violations=0,
                   foreground_order_violations=0, surface_level_abs_max=0.,
                   background_interior_points=0, historical_collision_unresolved=0,
                   same_count_normal_id=None, same_count_normal_range_m=None, same_count_range_difference_m=None)
        if len(inserted):
            xyz = anomaly[:, :3].astype(float)
            dirs, origins = _rays.directions[inserted], _rays.origins[inserted]
            depth = np.sum((xyz - origins) * dirs, axis=1) / np.sum(dirs * dirs, axis=1)
            residual = np.linalg.norm(xyz - origins - depth[:, None] * dirs, axis=1)
            # The saved coordinates are rounded to float32; this is a numeric error bound.
            rounding = np.linalg.norm(np.spacing(np.abs(anomaly[:, :3])).astype(float) / 2, axis=1)
            rounding += 64 * np.finfo(float).eps * np.maximum(1, np.linalg.norm(xyz, axis=1))
            row["ray_residual_max_m"] = float(residual.max())
            row["ray_roundoff_violations"] = int(np.count_nonzero(residual > rounding))
            row["foreground_order_violations"] = int(np.count_nonzero(depth - native_depth[inserted] > rounding))
            world_xyz = xyz @ source.pose[:3, :3].T + source.pose[:3, 3]
            local = (world_xyz - w["translation"]) @ w["rotation"]
            row["surface_level_abs_max"] = float(np.max(np.abs(w["shape"].level(local))))
        if nearby:
            local = (world_points[nearby] - w["translation"]) @ w["rotation"]
            unresolved, levels = unresolved_penetration(w["shape"], local, allowance_m=_allowance,
                                                        gradient_step_m=max(1e-7, _allowance * 1e-4),
                                                        witness_fraction=1 - 1e-6)
            row["background_interior_points"] = int(np.count_nonzero(levels < 0))
            row["historical_collision_unresolved"] = int(unresolved.sum())
        if count >= 5:
            matches = []
            for key, slots in groups.items():
                kept = slots[~np.isin(slots, occluded, assume_unique=True)]
                if len(kept) == count:
                    median = float(np.median(source.range_m[kept]))
                    matches.append((abs(median - row["range_median_m"]), key, median))
            if matches:
                difference, key, median = min(matches)
                row.update(same_count_normal_id=f"206:{key >> 16}:{key & 65535}",
                           same_count_normal_range_m=median, same_count_range_difference_m=difference)
        reference = w["generation"].get("normal_reference")
        if reference and reference["frame"] == frame_id:
            slots = np.asarray(reference["slots"], int)
            row["historical_reference_changed_points"] = int(np.isin(slots, delta["source_slot"]).sum())
            row["historical_reference_valid_points"] = int(normal[slots].sum())
        rows.append(row)
    return rows, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def summarize(rows, worlds, inventory, allowance, seconds, workers, rss):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["world"]].append(r)
    for row in inventory:
        if row["world"] not in grouped:
            continue
        values = grouped[row["world"]]
        row.update(source_binding_verified=len(values) == row["stored_frames"],
                   compared_files=len(values),
                   verified_frames=len(values), eligible_frames=sum(r["eligible"] for r in values),
                   one_to_four_frames=sum(1 <= r["anomaly_points"] <= 4 for r in values),
                   anomaly_points=sum(r["anomaly_points"] for r in values if r["eligible"]),
                   main_v3_equal=all(r["main_v3_equal"] for r in values),
                   surface_level_abs_max=max(r["surface_level_abs_max"] for r in values),
                   background_interior_points=sum(r["background_interior_points"] for r in values),
                   historical_collision_unresolved=sum(r["historical_collision_unresolved"] for r in values))
        row["eligible_segments"] = sum(r["eligible"] and (i == 0 or not values[i-1]["eligible"] or values[i-1]["frame"] != r["frame"]-1) for i, r in enumerate(values))
    eligible = [r for r in rows if r["eligible"]]
    distance_rows = []
    # Display bins are descriptive, not acceptance thresholds or generation quotas.
    edges = [2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [r for r in eligible if lo <= r["range_median_m"] and (r["range_median_m"] <= hi if i == len(edges)-2 else r["range_median_m"] < hi)]
        distance_rows.append(dict(min_m=lo, max_m=hi, frames=len(selected),
                                  worlds=len({r["world"] for r in selected}),
                                  points=sum(r["anomaly_points"] for r in selected),
                                  frames_5_19=sum(r["anomaly_points"] < 20 for r in selected)))
    eligible_worlds = [r for r in inventory if r["source_sequence"] == 206]
    overlap_worlds = {r["world"] for r in eligible_worlds if r["background_interior_points"] > 0}
    identical_geometry = defaultdict(list)
    for w in worlds:
        identical_geometry[w["shape_key"]].append(w["path"])
    return dict(scope="single targeted sample pool; current user instruction overrides two-stage data text",
                source_sequence=206, original_frames=449, worlds=len(worlds),
                unique_shape_definitions=len(identical_geometry),
                recorded_geometry_parents=len({w["parent"] for w in worlds if w["parent"] is not None}),
                worlds_without_recorded_parent=sum(w["parent"] is None for w in worlds),
                equal_geometry_groups=[v for v in identical_geometry.values() if len(v) > 1],
                read_frames=len(rows), eligible_frames=len(eligible),
                selected_frames=sum(r["selected"] for r in rows),
                repeated_eligible_scans=sum(r["duplicate_of"] is not None for r in rows),
                zero_anomaly_frames=sum(r["anomaly_points"] == 0 for r in rows),
                one_to_four_frames=sum(1 <= r["anomaly_points"] <= 4 for r in rows),
                eligible_anomaly_points=sum(r["anomaly_points"] for r in eligible),
                eligible_normal_observations=sum(r["normal_points"] for r in eligible),
                anomaly_count=quantiles([r["anomaly_points"] for r in eligible]),
                intensity_min=min(r["intensity_min"] for r in eligible),
                intensity_max=max(r["intensity_max"] for r in eligible),
                eligible_frames_per_world=quantiles([r["eligible_frames"] for r in eligible_worlds]),
                eligible_segments=sum(r["eligible_segments"] for r in eligible_worlds),
                duplicate_files=sum(r["main_v3_equal"] for r in rows),
                metadata_count_mismatches=sum(not r["historical_count_equal"] for r in rows),
                ray_roundoff_violations=sum(r["ray_roundoff_violations"] for r in rows),
                foreground_order_violations=sum(r["foreground_order_violations"] for r in rows),
                ray_residual_max_m=max(r["ray_residual_max_m"] for r in rows),
                surface_level_abs_max=max(r["surface_level_abs_max"] for r in rows),
                historical_collision_allowance_m=allowance,
                historical_collision_unresolved=sum(r["historical_collision_unresolved"] for r in rows),
                background_interior_points=sum(r["background_interior_points"] for r in rows),
                background_overlap_worlds=sorted(overlap_worlds),
                background_overlap_frames=sum(r["background_interior_points"] > 0 for r in rows),
                eligible_frames_with_background_overlap=sum(r["background_interior_points"] > 0 for r in eligible),
                eligible_frames_in_overlap_worlds=sum(r["world"] in overlap_worlds for r in eligible),
                eligible_frames_without_world_overlap=sum(r["world"] not in overlap_worlds for r in eligible),
                historical_reference_changed_points=sum(r.get("historical_reference_changed_points", 0) for r in rows),
                same_count_normal_frames=sum(r["same_count_normal_id"] is not None for r in eligible),
                same_count_range_difference_m=quantiles([r["same_count_range_difference_m"] for r in eligible if r["same_count_range_difference_m"] is not None]),
                changed_background_count=quantiles([r["occluded_original"] for r in eligible]),
                unchanged_background_frames=sum(r["occluded_original"] == 0 for r in eligible),
                lost_original_returns=sum(r["lost_original"] for r in eligible),
                new_returns_in_empty_slots=sum(r["new_return_in_empty_slot"] for r in eligible),
                other_sequences=[dict(source_sequence=seq, worlds=len(other),
                    stored_frames=sum(r["stored_frames"] for r in other),
                    compared_files=sum(r["compared_files"] or 0 for r in other),
                    main_v3_equal=all(r["main_v3_equal"] for r in other),
                    historical_eligible_frames=sum(r["historical_eligible_frames"] for r in other),
                    original_source_recomputed=False, included_in_training=False)
                    for seq in sorted({r["source_sequence"] for r in inventory if r["source_sequence"] != 206})
                    for other in [[r for r in inventory if r["source_sequence"] == seq]]],
                distance=distance_rows, seconds=seconds, workers=workers, worker_peak_rss_bytes=rss)


def compare_references(rows, data_root, root):
    """Measure coarse observation overlap, without setting a matching tolerance."""
    candidates = [r for r in rows if r["selected"]]
    by_count, by_frame = defaultdict(list), defaultdict(list)
    for row in candidates:
        by_count[row["anomaly_points"]].append(row)
        by_frame[row["frame"]].append(row)
    sequence, matches, summaries = STUSequence(data_root), [], []
    for semantic, instance, start, end in REFERENCES:
        observations = []
        for frame_id in range(start, end + 1):
            original = sequence[frame_id]
            slots = np.flatnonzero((point_targets(original) == 0) &
                                  (original.semantic == semantic) & (original.instance == instance))
            if not len(slots):
                continue
            median = float(np.median(original.range_m[slots]))
            result = dict(reference=f"206:{semantic}:{instance}", normal_frame=frame_id,
                          normal_points=len(slots), normal_range_m=median, normal_sample_world=None,
                          anomaly_world=None, anomaly_frame=None, anomaly_points=None,
                          anomaly_range_m=None, range_difference_m=None)
            # The normal structure must still exist in an eligible implanted scan.
            for kept in sorted(by_frame[frame_id], key=lambda r: r["occluded_original"]):
                delta = read_delta(root / kept["world"] / "frames" / f"{frame_id:06d}.npz")
                if not np.intersect1d(slots, delta["occluded_slot"], assume_unique=True).size:
                    result["normal_sample_world"] = kept["world"]
                    break
            # Equal counts provide a strict existence check, not a new training rule.
            if by_count[len(slots)]:
                nearest = min(by_count[len(slots)], key=lambda r: abs(r["range_median_m"] - median))
                result.update(anomaly_world=nearest["world"], anomaly_frame=nearest["frame"],
                              anomaly_points=nearest["anomaly_points"], anomaly_range_m=nearest["range_median_m"],
                              range_difference_m=abs(nearest["range_median_m"] - median))
            observations.append(result)
        counts = [r["normal_points"] for r in observations]
        distances = [r["normal_range_m"] for r in observations]
        joint = [r for r in candidates if min(counts) <= r["anomaly_points"] <= max(counts) and
                 min(distances) <= r["range_median_m"] <= max(distances)]
        matched = [r for r in observations if r["anomaly_world"] is not None]
        summaries.append(dict(reference=f"206:{semantic}:{instance}", start_frame=start, end_frame=end,
                              normal_observations=len(observations), normal_count_range=[min(counts), max(counts)],
                              normal_distance_range_m=[min(distances), max(distances)],
                              normal_one_to_four=sum(c < 5 for c in counts),
                              normal_preserved_in_eligible_scan=sum(r["normal_sample_world"] is not None for r in observations),
                              exact_count_nearest_observations=len(matched),
                              nearest_range_gap_m=quantiles([r["range_difference_m"] for r in matched]),
                              unique_nearest_scans=len({(r["anomaly_world"], r["anomaly_frame"]) for r in matched}),
                              joint_range_candidate_scans=len(joint),
                              joint_range_candidate_worlds=len({r["world"] for r in joint})))
        matches.extend(observations)
    return matches, summaries


def run(args):
    started = time.monotonic()
    worlds, inventory, allowance = load_worlds(args.root, args.mirror)
    frames = list(range(449)) if args.frames is None else args.frames
    if args.output.exists() and any(args.output.iterdir()) and args.frames is not None:
        raise ValueError("pilot must not overwrite a full analysis")
    rows, peak = [], 0
    with ProcessPoolExecutor(args.workers, initializer=initialize,
                             initargs=(args.root, args.mirror, args.data_root, worlds, allowance)) as pool:
        for i, (part, rss) in enumerate(pool.map(analyze_frame, frames), 1):
            rows.extend(part)
            peak = max(peak, rss)
            if i % 25 == 0 or i == len(frames):
                print(f"source_frames={i}/{len(frames)} saved_samples={len(rows)} seconds={time.monotonic()-started:.1f}", flush=True)
    rows.sort(key=lambda r: (r["world"], r["frame"]))
    summary = summarize(rows, worlds, inventory, allowance, time.monotonic()-started, args.workers, peak)
    summary.update(main_root=str(args.root.resolve()), mirror_root=str(args.mirror.resolve()),
                   data_root=str(args.data_root.resolve()), complete_source_frames=len(frames) == 449)
    matches = []
    if args.frames is None:
        matches, summary["reference_comparison"] = compare_references(rows, args.data_root, args.root)
        summary["seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "frames.csv", rows)
    write_csv(args.output / "worlds.csv", inventory)
    if matches:
        write_csv(args.output / "matches.csv", matches)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("read_frames", "eligible_frames", "metadata_count_mismatches", "ray_roundoff_violations", "foreground_order_violations", "surface_level_abs_max", "historical_collision_unresolved", "same_count_normal_frames", "seconds", "worker_peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    run(parser.parse_args())
