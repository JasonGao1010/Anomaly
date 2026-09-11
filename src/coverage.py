"""World-level content coverage from frozen observations and targeted full scans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time

import numpy as np
from numba import set_num_threads
from scipy.spatial import cKDTree

from .data import FrozenDataset, _atomic_json, host_disk
from .render import calibrated_ray_grid, shape_from_dict, shape_geometry
from .geometry import (
    ScanGeometry, boundary_targets, local_evidence, sampling_targets,
    surface_probe, surface_targets, summarize_view, thinning_pair,
)


def conditions(world, row):
    """Range is the recorded median; eligibility uses the distance-filtered count."""
    count, eligible = row["count"], row["in_range"] >= 5
    distance = row.get("range", -1)
    far, low = 35 <= distance <= 50, world["height_m"] <= .2
    return dict(
        all=True, visible=count > 0, eligible=eligible,
        few_eligible=5 <= row["in_range"] < 20,
        far=far, far_eligible=far and eligible,
        far_few_eligible=far and 5 <= row["in_range"] < 20,
        far_denser=far and row["in_range"] >= 20,
        low_visible=low and count > 0, low_eligible=low and eligible,
        low_few_eligible=low and 5 <= row["in_range"] < 20,
        low_far_eligible=low and far and eligible,
        low_near_eligible=low and eligible and 2.5 <= distance < 10,
        low_middle_eligible=low and eligible and 10 <= distance < 35,
    )


def concentration(counts):
    counts = {key: int(value) for key, value in counts.items() if value > 0}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    return dict(total=total, worlds=len(counts), by_world=dict(ordered),
                top_one_share=ordered[0][1] / total if total else None,
                top_five_share=sum(n for _, n in ordered[:5]) / total if total else None)


def inventory(directory):
    root = json.loads((directory / "manifest.json").read_text())
    worlds = []
    # Root entries define membership; rejected physical worlds remain outside it.
    for split, part in root["splits"].items():
        for entry in part["worlds"]:
            path = directory / entry["path"]
            definition = json.loads((path / "world.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
            obj = definition["world"]["objects"][0]
            shape, generation = obj["shape"], definition["generation"]
            geometry = shape_geometry(shape_from_dict(shape))
            normal = np.asarray(generation["support_plane"]["normal_world"])
            world = dict(
                split=split, world=path.name, source_sequence=part["source_sequence"],
                identity=entry["world_identity"], seed=definition["world"]["seed"],
                primitives=len(shape["primitive_scales_m"]), **geometry,
                shape_family=generation.get("shape_family", "single"),
                background=generation.get("background", "not_stratified"),
                candidate_category=generation.get("candidate_category", "development_pool"),
                origin=entry.get("origin", "base"),
                content_check_frames=manifest.get("content_check_frames", []),
                exponents=shape["primitive_exponents"], material=obj["material"],
                support_semantic=generation["placement"]["support_semantic"],
                support_slope_degrees=float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
                support_rmse_m=generation["plane_rmse_m"],
                closest_trajectory_distance_m=generation["closest_trajectory_distance_m"],
                rows=manifest["frames"],
            )
            assert entry["world_identity"] == manifest["world_identity"]
            frame_count = root["configuration"]["dataset"]["splits"][split]["frames_per_world"]
            assert [r["frame"] for r in world["rows"]] == list(range(frame_count))
            visible = [r for r in world["rows"] if r["count"]]
            world["observation_context"] = dict(
                ground_status=dict(Counter(r.get("ground_status", "missing") for r in visible)),
                nearest_road=dict(Counter("all_road" if r.get("nearest_road_fraction") == 1
                                         else "no_road" if r.get("nearest_road_fraction") == 0
                                         else "mixed" if r.get("nearest_road_fraction") is not None
                                         else "missing" for r in visible)),
                median_normal_neighbors=dict(Counter("missing" if r.get("background_neighbors") is None
                                                     else "below_3" if r["background_neighbors"] < 3
                                                     else "3_to_19" if r["background_neighbors"] < 20
                                                     else "at_least_20" for r in visible)),
            )
            worlds.append(world)
        assert sum(len(w["rows"]) for w in worlds if w["split"] == split) == part["samples"]
    totals = {}
    for split in root["splits"]:
        selected = [w for w in worlds if w["split"] == split]
        totals[split] = {}
        for name in conditions(dict(height_m=0), dict(count=0, in_range=0)):
            frames, points, inside = {}, {}, {}
            for w in selected:
                rows = [r for r in w["rows"] if conditions(w, r)[name]]
                frames[w["world"]] = len(rows)
                points[w["world"]] = sum(r["count"] for r in rows)
                inside[w["world"]] = sum(r["in_range"] for r in rows)
            totals[split][name] = dict(frames=concentration(frames),
                                       anomaly_returns=concentration(points),
                                       in_range_anomaly_returns=concentration(inside))
    return root, worlds, totals


def select_checks(worlds, far_limit=None):
    selections = {}
    for world in worlds:
        eligible = [r for r in world["rows"] if r["in_range"] >= 5]
        few = sorted((r for r in eligible if r["in_range"] < 20),
                     key=lambda r: (r["in_range"], r["frame"]))
        chosen = []
        chosen.extend((world["rows"][f], "declared_content_witness") for f in world.get("content_check_frames", []))
        if eligible:
            chosen.append((min(eligible, key=lambda r: (-r["in_range"], r["frame"])), "densest"))
        elif world["rows"]:
            chosen.append((min(world["rows"], key=lambda r: (-r["count"], r["frame"])), "no_eligible_representative"))
        if few:
            chosen.append((few[(len(few) - 1) // 2], "few_representative"))
        far = [r for r in eligible if conditions(world, r)["far_eligible"]]
        if far_limit is None:
            chosen.extend((r, "all_far_eligible") for r in far)
        else:
            for dense in (False, True):
                rows = sorted((r for r in far if (r["in_range"] >= 20) == dense), key=lambda r: (r["range"], r["frame"]))
                ids = np.linspace(0, len(rows) - 1, min(far_limit, len(rows)), dtype=int)
                chosen.extend((rows[i], "far_denser_representative" if dense else "far_few_representative") for i in ids)
            weak = sorted((r for r in world["rows"] if 1 <= r["in_range"] < 5 and 35 <= r.get("range", -1) <= 50), key=lambda r: r["frame"])
            if weak:
                chosen.append((weak[(len(weak) - 1) // 2], "far_below_official_threshold"))
        for row, reason in chosen:
            key = (world["split"], world["world"], row["frame"])
            selections.setdefault(key, dict(split=key[0], world=key[1], frame=key[2], reasons=[]))
            selections[key]["reasons"].append(reason)
    return [selections[key] for key in sorted(selections)]


def _initialize(protocol, data_root, threads):
    global _protocol, _datasets, _grid, _threads
    _protocol, _threads = protocol, threads
    set_num_threads(threads)
    _datasets = {s: FrozenDataset(protocol["dataset"]["directory"], data_root, s, allow_candidates=True)
                 for s in ("train", "validation")}
    _grid = calibrated_ray_grid(protocol["calibration"]["rays"])


def _counts(values):
    return {str(k): int(v) for k, v in sorted(Counter(np.asarray(values).tolist()).items())}


def check_scan(selection):
    start, cpu = time.monotonic(), time.process_time()
    dataset, config = _datasets[selection["split"]], _protocol["geometry"]
    index = next(i for i, (p, _, f) in enumerate(dataset.samples)
                 if p.parent.parent.name == selection["world"] and f == selection["frame"])
    sample = dataset[index]
    original = dataset.sequence[selection["frame"]]
    path = dataset.samples[index][0].parent.parent
    definition = json.loads((path / "world.json").read_text())["world"]
    observed = json.loads((path / "manifest.json").read_text())["frames"][selection["frame"]]
    source, slots = sample.source, sample.source.real_slots
    y = sample.anomaly_target[slots]
    assert np.count_nonzero(y == 1) == observed["count"]
    ranges = np.linalg.norm(source.xyzi[slots, :3], axis=1)
    assert np.sum((y == 1) & (ranges >= 2.5) & (ranges <= 50)) == observed["in_range"]
    pairs = [thinning_pair(sample, original, _grid, definition["seed"], level, _protocol["seed"])
             for level in config["levels"]]
    geometries = [ScanGeometry(source.xyzi[slots], slots, config["sampling_scale"], _threads)]
    labels = [y]
    for pair in pairs:
        rows = pair["dense_row"]
        geometries.append(ScanGeometry(geometries[0].xyzi[rows], pair["sparse_source_slot"],
                                       config["sampling_scale"], _threads))
        labels.append(y[rows])
        assert np.array_equal(geometries[-1].xyzi, source.xyzi[pair["sparse_source_slot"]])
    targets = [dict(**geometries[0].scale_arrays(),
                    **boundary_targets(geometries[0], y, config["boundary"]))]
    evidence = local_evidence(geometries[0], y, config["sampling"]["radius_m"])
    for g, sy, pair in zip(geometries[1:], labels[1:], pairs):
        targets.append(dict(**g.scale_arrays(), **sampling_targets(
            geometries[0], g, y, sy, pair, config["sampling"], evidence)))
    surfaces = surface_targets(original, sample, geometries, labels, pairs, config["surface"])
    for target, surface in zip(targets, surfaces[config["surface"]["minimum_visible_support_points"]]):
        target.update(surface)
    available = targets[0]["boundary_valid"] | targets[0]["surface_valid"]
    for pair, target in zip(pairs, targets[1:]):
        available[pair["dense_row"][target["sampling_consistency_valid"]]] = True
    views = [summarize_view(g, sy, target, y if v else None)
             for v, (g, sy, target) in enumerate(zip(geometries, labels, targets))]
    edges = targets[0]["boundary_edges"]
    assert np.all(sample.anomaly_target[edges[:, 0]] == 0)
    assert np.all(sample.anomaly_target[edges[:, 1]] == 1)
    # Nearby normals are queried against the full inserted object observation.
    # Their reference-surface offsets were computed using the complete original/current scans.
    anomaly_tree = cKDTree(source.xyzi[slots[y == 1], :3].astype(float))
    distance = anomaly_tree.query(source.xyzi[slots, :3], workers=_threads)[0]
    near = (y == 0) & (distance <= 2)
    contexts = []
    for view, (g, sy, target) in enumerate(zip(geometries, labels, targets)):
        dense_rows = np.arange(len(y)) if view == 0 else pairs[view - 1]["dense_row"]
        nearby = near[dense_rows]
        # This is a measured normal protrusion proxy, not a curb identity or model difficulty label.
        raised = (sy == 0) & target["surface_valid"] & (target["surface_offset_z"] <= -.05) & (target["surface_offset_z"] >= -.2)
        contexts.append(dict(normal_within_2m=int(nearby.sum()),
                             normal_semantics_within_2m=_counts(source.labels.semantic[g.slots[nearby]]),
                             raised_normal_all=int(raised.sum()),
                             raised_normal_within_2m=int(np.sum(raised & nearby)),
                             raised_normal_semantics_within_2m=_counts(source.labels.semantic[g.slots[raised & nearby]])))
    probes = []
    t = targets[0]
    for name, mask in (("anomaly", (y == 1) & t["surface_valid"]),
                       ("nearby_raised_normal", near & t["surface_valid"]
                        & (t["surface_offset_z"] <= -.05) & (t["surface_offset_z"] >= -.2))):
        rows = np.flatnonzero(mask)
        if len(rows):
            row = int(rows[len(rows) // 2])
            probe = surface_probe(source.xyzi[slots[row], :3].astype(float), original, sample, slots, config["surface"])
            assert probe["valid"] and abs(probe["offset_z_m"] - float(t["surface_offset_z"][row])) < 1e-6
            probes.append(dict(kind=name, source_slot=int(slots[row]), label=int(y[row]),
                               semantic=int(source.labels.semantic[slots[row]]), **probe))
    return dict(sample=selection, observation=observed, views=views, context=contexts,
                base_anomaly_without_checked_geometry=int(np.sum((y == 1) & ~available)),
                anomaly_reference_positions=_counts(targets[0]["surface_reference_count"][y == 1]),
                boundary_edges=len(edges), boundary_normal_semantics=_counts(source.labels.semantic[edges[:, 0]]),
                actual_far_anomaly_returns=int(np.sum((y == 1) & (ranges >= 35) & (ranges <= 50))),
                probes=probes, seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def summarize_checks(records, worlds):
    lookup = {(w["split"], w["world"]): w for w in worlds}
    output = {}
    for split in ("train", "validation"):
        output[split] = {}
        for group in ("all", "densest", "few_representative", "far_eligible", "far_few_eligible",
                      "far_denser", "far_below_official_threshold", "low_eligible", "low_far_eligible"):
            chosen = []
            for r in records:
                s = r["sample"]
                if s["split"] != split:
                    continue
                w = lookup[(split, s["world"])]
                if group == "all" or group in s["reasons"] or conditions(w, r["observation"]).get(group, False):
                    chosen.append(r)
            metrics = {}
            accessors = {"boundary_edges": lambda r: r["boundary_edges"],
                         "base_anomaly_without_checked_geometry": lambda r: r["base_anomaly_without_checked_geometry"],
                         "actual_far_anomaly_returns": lambda r: r["actual_far_anomaly_returns"]}
            for view in range(3):
                for label in ("normal", "anomaly"):
                    for metric in ("total", "surface.valid", "sampling.valid", "boundary.valid", "boundary.non_saturated"):
                        if (metric.startswith("sampling") and view == 0) or (metric.startswith("boundary") and view != 0):
                            continue
                        def access(r, v=view, name=label, keys=metric.split(".")):
                            item = r["views"][v][name]
                            for key in keys:
                                item = item[key]
                            return item
                        accessors[f"view{view}.{label}.{metric}"] = access
                for metric in ("normal_within_2m", "raised_normal_all", "raised_normal_within_2m"):
                    accessors[f"view{view}.{metric}"] = lambda r, v=view, key=metric: r["context"][v][key]
            for metric, access in accessors.items():
                counts = Counter()
                for r in chosen:
                    counts[r["sample"]["world"]] += access(r)
                metrics[metric] = dict(concentration(counts), frames=sum(access(r) > 0 for r in chosen))
            reasons = {}
            for view in range(3):
                for label in ("normal", "anomaly"):
                    for target in (("boundary", "surface") if view == 0 else ("sampling", "surface")):
                        counts = Counter()
                        for r in chosen:
                            counts.update(r["views"][view][label][target]["reasons"])
                        reasons[f"view{view}.{label}.{target}"] = dict(counts)
            output[split][group] = dict(frames=len(chosen), worlds=len({r["sample"]["world"] for r in chosen}),
                                       metrics=metrics, reasons=reasons)
    return output


GEOMETRY_FIELDS = ("surface_residual", "normal_change")


def geometry_selection(worlds):
    """Select existing observations before computing their local geometry."""
    selections = select_checks(worlds, far_limit=2)
    chosen = {(s["split"], s["world"], s["frame"]): dict(s, representative=True, trajectory=False)
              for s in selections}
    trajectories = {}
    for split in ("train", "validation"):
        candidates = [w for w in worlds if w["split"] == split]
        world = min(candidates, key=lambda w: (
            -sum(r["count"] > 0 and 35 <= r.get("range", -1) <= 50 for r in w["rows"]), w["world"]))
        trajectories[split] = dict(world=world["world"], identity=world["identity"],
                                   far_visible_frames=sum(r["count"] > 0 and 35 <= r.get("range", -1) <= 50
                                                          for r in world["rows"]))
        for row in world["rows"]:
            key = (split, world["world"], row["frame"])
            selected = chosen.setdefault(key, dict(split=split, world=world["world"], frame=row["frame"],
                                                   reasons=[], representative=False, trajectory=False))
            selected["trajectory"] = True
    return [chosen[key] for key in sorted(chosen)], trajectories


def geometry_reference(directory=Path("results/geometry")):
    """Reuse the original train/206 fit, then retain only the two required fields."""
    import ctypes
    import gc
    from .geometry import as_data
    from .probes import fit_reference

    directory = Path(directory)
    normal = np.concatenate([np.load(p, allow_pickle=False)
                             for p in sorted((directory / "features/train/206").glob("*.npy"))])
    if np.any(normal["target"] != 0) or len(np.unique(normal["frame"])) != 449:
        raise ValueError("Conditional geometry requires the unchanged train/206 normal fit")
    fitted = fit_reference(as_data(normal))
    metadata = json.loads((directory / "reference.json").read_text())
    if fitted["metadata"] != metadata:
        raise ValueError("Paired geometry reference differs from the original normal-only fit")
    thresholds = json.loads((directory / "thresholds.json").read_text())["values"]
    result = dict(edges=fitted["edges"], conditional={
        cell: {field: values[field] for field in GEOMETRY_FIELDS}
        for cell, values in fitted["conditional"]["direction"].items()},
        thresholds={field: thresholds[f"feature/{field}/direction|C_direction"] for field in GEOMETRY_FIELDS})
    result["quantiles"] = {cell: {field: np.quantile(values[field], [.1, .9]) if len(values[field]) else None
                                  for field in GEOMETRY_FIELDS} for cell, values in result["conditional"].items()}
    del normal, fitted
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return result


def _geometry_initialize(directory, data_root):
    global _geometry_datasets, _geometry_paths
    set_num_threads(1)
    _geometry_datasets = {s: FrozenDataset(directory, data_root, s) for s in ("train", "validation")}
    _geometry_paths = {s: {(p.parent.parent.name, f): (p, identity) for p, identity, f in dataset.samples}
                       for s, dataset in _geometry_datasets.items()}


def _geometry_condition(values):
    """Use the original inclusive two-tail score and field-specific support."""
    from .probes import _cells, tail_score

    reference = _geometry_reference
    cells = _cells(values, reference["edges"])["direction"]
    result = {}
    for field in GEOMETRY_FIELDS:
        x = values[field]
        score = np.full(len(x), np.nan, np.float32)
        low, high = (np.full(len(x), np.nan) for _ in range(2))
        for cell in np.unique(cells):
            ordered = reference["conditional"].get(int(cell), {}).get(field)
            if ordered is None or not len(ordered):
                continue
            take = cells == cell
            score[take] = tail_score(x[take], ordered)
            low[take], high[take] = reference["quantiles"][int(cell)][field]
        covered = np.isfinite(score)
        threshold = reference["thresholds"][field]
        result[field] = dict(valid=np.isfinite(x), covered=covered,
                             hit=covered & (score >= threshold) if threshold is not None else np.zeros(len(x), bool),
                             lower=covered & (x < low), upper=covered & (x > high),
                             central=covered & (x >= low) & (x <= high))
    return result


def _geometry_frame(job):
    """Compute the complete original scan once, then each selected fixed-world view."""
    from .data import FrozenFrame
    from .profile import observed_geometry

    split, frame, selections = job
    started, cpu = time.monotonic(), time.process_time()
    original = _geometry_datasets[split].sequence[frame]
    original_slots = original.real_slots
    before = observed_geometry(original.xyzi[original_slots], original_slots)
    before_condition = _geometry_condition(before)
    original_normal = ~original.zero_slot_mask & (original.labels.semantic_target != 255)
    records, populations, changes = [], [], []
    for selection in selections:
        path, identity = _geometry_paths[split][selection["world"], frame]
        sample = FrozenFrame.load(path, original, identity)
        source, inserted, removed = sample.source, sample.inserted_mask, sample.occluded_original_mask
        slots = source.real_slots
        changed = inserted | removed
        kept = original_normal & ~changed
        if not (np.array_equal(original.xyzi[kept].view(np.uint32), source.xyzi[kept].view(np.uint32))
                and np.array_equal(original.labels.packed[kept], source.labels.packed[kept])
                and np.all(sample.anomaly_target[kept] == 0)):
            raise ValueError("Paired normal returns changed their XYZI, labels or target")
        after = observed_geometry(source.xyzi[slots], slots) if changed.any() else before
        conditional = _geometry_condition(after)
        inserted_xyz = source.xyzi[inserted, :3].astype(float)
        removed_xyz = original.xyzi[removed, :3].astype(float)
        distance = (cKDTree(inserted_xyz).query(source.xyzi[slots, :3], workers=1)[0]
                    if len(inserted_xyz) else np.full(len(slots), np.inf))
        ranges = after["range"]
        inside = (ranges >= 2.5) & (ranges <= 50)
        # New foreground and removed background positions both change local support.
        changed_xyz = np.concatenate((inserted_xyz, removed_xyz))
        kept_slots = np.flatnonzero(kept)
        before_rows, after_rows = np.searchsorted(original_slots, kept_slots), np.searchsorted(slots, kept_slots)
        for name in ("range", "ray_z"):
            if not np.array_equal(before[name][before_rows], after[name][after_rows]):
                raise ValueError("Unchanged normal return has different sensing conditions")
        changed_distance = (cKDTree(changed_xyz).query(original.xyzi[kept_slots, :3], workers=1)[0]
                            if len(changed_xyz) else np.full(len(kept_slots), np.inf))
        band = np.searchsorted([.5, 1., 2., 4.], changed_distance, side="left")
        observed = json.loads((path.parent.parent / "manifest.json").read_text())["frames"][frame]
        if int(inserted.sum()) != observed["count"] or int(np.sum(inserted[slots] & inside)) != observed["in_range"]:
            raise ValueError("Restored anomaly identities differ from the fixed observation metadata")
        common = {key: selection[key] for key in ("split", "world", "frame", "representative", "trajectory")}
        common["reasons"] = ";".join(selection["reasons"])
        records.append(dict(common, source_sequence=original.sequence_id, world_identity=identity,
                            anomaly_total=int(inserted.sum()), anomaly_in_range=observed["in_range"],
                            anomaly_outside_range=int(inserted.sum())-observed["in_range"],
                            observed_range_median=observed.get("range"), actual_returns=len(slots),
                            kept_normal=int(kept.sum()), removed_original=int(removed.sum()),
                            removed_normal=int(np.sum(removed & original_normal)),
                            occluded_no_return=int(np.sum(removed & ~inserted)),
                            inserted_original_empty=int(np.sum(inserted & original.zero_slot_mask)),
                            source_xyzi_equal=True))
        for population, population_mask in (("anomaly", inserted[slots]),
                                             ("nearby_kept_normal", kept[slots] & (distance <= 2))):
            for range_group, range_mask in (("in_range", inside), ("outside_range", ~inside)):
                mask = population_mask & range_mask
                for field in GEOMETRY_FIELDS:
                    x, state = after[field], conditional[field]
                    finite = mask & state["valid"]
                    row = dict(common, population=population, range_group=range_group, field=field,
                               total=int(mask.sum()), valid=int(finite.sum()),
                               **{key: int(np.sum(mask & state[key])) for key in ("covered", "hit", "lower", "central", "upper")},
                               hit_lower=int(np.sum(mask & state["hit"] & state["lower"])),
                               hit_upper=int(np.sum(mask & state["hit"] & state["upper"])),
                               value_sum=float(x[finite].sum(dtype=np.float64)),
                               neighbor_count_sum=int(after["neighbor_count"][mask].sum()),
                               condition_valid=int(np.sum(mask & after["condition_valid"])))
                    quantiles = np.quantile(x[finite], [.1, .5, .9]) if finite.any() else [None]*3
                    row.update(zip(("p10", "median", "p90"), quantiles))
                    populations.append(row)
        for field in GEOMETRY_FIELDS:
            old, new = before[field][before_rows], after[field][after_rows]
            old_valid, new_valid = np.isfinite(old), np.isfinite(new)
            both = old_valid & new_valid
            delta = new.astype(np.float64) - old.astype(np.float64)
            old_covered = before_condition[field]["covered"][before_rows]
            new_covered = conditional[field]["covered"][after_rows]
            old_hit = before_condition[field]["hit"][before_rows]
            new_hit = conditional[field]["hit"][after_rows]
            shared = old_covered & new_covered
            for group, label in enumerate(("[0,0.5]", "(0.5,1]", "(1,2]", "(2,4]", "(4,inf]")):
                mask = band == group
                usable = mask & both
                absolute = np.abs(delta[usable])
                quantiles = np.quantile(absolute, [.1, .5, .9]) if usable.any() else [None]*3
                changes.append(dict(common, field=field, distance_group=label, total=int(mask.sum()),
                                    both_valid=int(usable.sum()), before_only=int(np.sum(mask & old_valid & ~new_valid)),
                                    after_only=int(np.sum(mask & ~old_valid & new_valid)),
                                    both_missing=int(np.sum(mask & ~old_valid & ~new_valid)),
                                    changed=int(np.sum(usable & (old != new))),
                                    before_covered=int(np.sum(mask & old_covered)),
                                    after_covered=int(np.sum(mask & new_covered)),
                                    both_covered=int(np.sum(mask & shared)),
                                    hit_before=int(np.sum(mask & old_hit)), hit_after=int(np.sum(mask & new_hit)),
                                    new_hit=int(np.sum(mask & shared & ~old_hit & new_hit)),
                                    lost_hit=int(np.sum(mask & shared & old_hit & ~new_hit)),
                                    delta_sum=float(delta[usable].sum()), abs_delta_sum=float(np.abs(delta[usable]).sum()),
                                    maximum_abs_delta=float(np.max(absolute)) if usable.any() else None,
                                    abs_delta_p10=quantiles[0], abs_delta_median=quantiles[1], abs_delta_p90=quantiles[2]))
            # Two nested 2 m neighborhoods cannot change beyond 4 m of every edit.
            far = band == 4
            if np.any(old_valid[far] != new_valid[far]) or np.any(old[far & both] != new[far & both]):
                raise ValueError("Unchanged normal geometry changed outside its full 4 m support")
    return dict(frames=records, features=populations, changes=changes,
                seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def geometry_check(protocol, data_root, output, workers):
    """Describe selected worlds and complete preselected trajectories, without refitting rules."""
    import csv
    from .profile import GEOMETRY_PARAMETERS

    global _geometry_reference
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(protocol["dataset"]["directory"])
    _, worlds, _ = inventory(directory)
    selections, trajectories = geometry_selection(worlds)
    grouped = defaultdict(list)
    for selected in selections:
        grouped[selected["split"], selected["frame"]].append(selected)
    jobs = [(split, frame, chosen) for (split, frame), chosen in sorted(grouped.items())]
    disk = host_disk()
    if disk["SizeRemaining"] - 100_000_000 < disk["reserve_bytes"]:
        raise OSError("Compact paired geometry outputs would invade the physical E: reserve")
    selection = dict(dataset=str(directory), geometry_parameters=GEOMETRY_PARAMETERS, trajectories=trajectories,
                     selections=selections, source_frames=len(jobs), world_frames=len(selections),
                     representative_frames=sum(s["representative"] for s in selections),
                     scope="155 selected observations across 40 worlds plus two complete manifest-selected trajectories",
                     selection_rule="maximum count of visible frames with median range in [35,50] m; world-name tie break",
                     normal_definition="actual original return with semantic_target !=255, neither inserted nor occluded",
                     scoring="unchanged train/206 direction-conditional inclusive two-tail rarity and per-field thresholds",
                     neighborhoods="all actual returns; labels only identify output groups; original geometry once per source frame",
                     missing="retain zero-return frames and each field's unavailable geometry and reference support",
                     distance_groups="distance to union of new inserted and original removed coordinates; upper endpoints included",
                     hit_transitions="new_hit/lost_hit use only pairs covered before and after; hit_before/after use each view's coverage",
                     background_weighting="paired worlds share backgrounds; do not interpret world repetitions as independent source scans",
                     peak_write_budget_bytes=100_000_000, initial_disk=disk)
    _atomic_json(output / "selection.json", selection)
    started = time.monotonic()
    _geometry_reference = geometry_reference()
    selection["thresholds"] = _geometry_reference["thresholds"]
    _atomic_json(output / "selection.json", selection)
    # Keep this parent free of loaded scans; fork only the small read-only reference.
    pilot_jobs = [next(j for j in jobs if j[0] == split and
                       any(w["rows"][j[1]]["count"] for w in worlds
                           if w["split"] == split and any(s["world"] == w["world"] for s in j[2])))
                  for split in ("train", "validation")]
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork"),
                             initializer=_geometry_initialize, initargs=(directory, data_root)) as pool:
        pilot = list(pool.map(_geometry_frame, pilot_jobs))
    print(json.dumps(dict(stage="geometry_pilot", source_frames=2,
                          seconds=[r["seconds"] for r in pilot], peak_rss_bytes=[r["peak_rss_bytes"] for r in pilot])), flush=True)
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    peak_per_worker = max(r["peak_rss_bytes"] for r in pilot)
    if available - workers * peak_per_worker < 3_000_000_000:
        raise MemoryError("Pilot worker RSS leaves less than 3 GB available at requested concurrency")
    host_disk()
    totals, completed, cpu, peak = {}, 0, 0., 0
    def accumulate(rows, names, numeric):
        for row in rows:
            for scope in ("representative", "trajectory"):
                if not row[scope]:
                    continue
                key = (scope, row["split"], *[row[name] for name in names])
                value = totals.setdefault(key, dict(scope=scope, split=row["split"],
                                                   **{name: row[name] for name in names}, observations=0))
                value["observations"] += 1
                for name in numeric:
                    value[name] = value.get(name, 0) + row[name]
    handles, writers = {}, {}
    try:
        for name in ("frames", "features", "changes"):
            handles[name] = (output / (name + ".csv")).open("w", newline="")
        pilot_ids = {(j[0], j[1]) for j in pilot_jobs}
        pending = [j for j in jobs if (j[0], j[1]) not in pilot_ids]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
                                 initializer=_geometry_initialize, initargs=(directory, data_root)) as pool:
            futures = [pool.submit(_geometry_frame, job) for job in pending]
            def results():
                yield from pilot
                for future in as_completed(futures):
                    yield future.result()
            for result in results():
                for name in handles:
                    if name not in writers:
                        writers[name] = csv.DictWriter(handles[name], fieldnames=list(result[name][0]))
                        writers[name].writeheader()
                    writers[name].writerows(result[name])
                accumulate(result["features"], ("population", "range_group", "field"),
                           ("total", "valid", "covered", "hit", "hit_lower", "hit_upper", "lower", "central", "upper", "value_sum", "neighbor_count_sum", "condition_valid"))
                accumulate(result["changes"], ("distance_group", "field"),
                           ("total", "both_valid", "before_only", "after_only", "both_missing", "changed", "delta_sum", "abs_delta_sum",
                            "before_covered", "after_covered", "both_covered", "hit_before", "hit_after", "new_hit", "lost_hit"))
                completed += 1
                cpu += result["cpu_seconds"]
                peak = max(peak, result["peak_rss_bytes"])
                if completed % 50 == 0 or completed == len(jobs):
                    volume = host_disk()
                    print(json.dumps(dict(stage="paired_geometry", completed=completed, total=len(jobs),
                                          seconds=round(time.monotonic()-started, 2), disk_remaining=volume["SizeRemaining"])), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    _atomic_json(output / "summary.json", dict(scope=selection["scope"], trajectories=trajectories,
                 source_frames=completed, world_frames=len(selections), seconds=time.monotonic()-started,
                 worker_cpu_seconds=cpu, maximum_worker_rss_bytes=peak, workers=workers, library_threads=1,
                 initial_disk=disk, final_disk=host_disk(), pilot=[{k:v for k,v in r.items() if k not in handles} for r in pilot],
                 aggregates=list(totals.values()), thresholds=selection["thresholds"]))


def _coverage_cell(rows, field):
    """Count world observations separately from unique source frames and returns."""
    by_world = Counter()
    for row in rows:
        by_world[row["world"]] += int(row[field])
    count = sum(by_world.values())
    top = min(by_world, key=lambda w: (-by_world[w], w)) if count else None
    frame_counts = Counter(w for w, _ in {(r["world"], int(r["frame"])) for r in rows})
    return dict(worlds=len({r["world"] for r in rows}),
                source_frames=len({(r["split"], int(r["frame"])) for r in rows}),
                world_frames=len({(r["world"], int(r["frame"])) for r in rows}),
                returns=count, return_field=field, world_ids=sorted(by_world),
                top_world=top, top_world_share=by_world[top] / count if count else None,
                top_world_frame_share=max(frame_counts.values()) / sum(frame_counts.values()) if frame_counts else None)


def summarize_existing(directory, coverage):
    """Close coverage from recorded metadata; never render or recompute geometry."""
    import csv

    root = json.loads((directory / "manifest.json").read_text())
    previous = json.loads((coverage / "experiment.json").read_text())
    metadata = {(w["split"], w["world"]): w for w in previous["worlds"]}
    inserted = coverage / "geometry/inserted"
    selection = json.loads((inserted / "selection.json").read_text())
    if root["status"] != "frozen" or Path(previous["dataset"]).resolve() != directory.resolve():
        raise ValueError("coverage must describe the current frozen experiment")
    if Path(selection["dataset"]).resolve() != directory.resolve():
        raise ValueError("geometry selection belongs to another dataset")
    bands = ("near", "middle", "far")

    def distance_band(row):
        if not row["count"]:
            return "no_return"
        r = row["range"]
        return ("below_range" if r < 2.5 else "near" if r < 10 else
                "middle" if r < 35 else "far" if r <= 50 else "beyond_range")

    worlds, observations = [], []
    for split, part in root["splits"].items():
        for entry in part["worlds"]:
            path = directory / entry["path"]
            w = metadata[split, path.name]
            if w["source_sequence"] != part["source_sequence"]:
                raise ValueError("world and source sequence differ from existing coverage")
            manifest = json.loads((path / "manifest.json").read_text())
            if w["identity"] != entry["world_identity"] or manifest["world_identity"] != w["identity"]:
                raise ValueError("world identity differs from the retained coverage")
            if [r["frame"] for r in manifest["frames"]] != list(range(part["samples"] // len(part["worlds"]))):
                raise ValueError("full source-frame order is required")
            worlds.append(w)
            observations.extend(dict(r, split=split, world=w["world"],
                                     band=distance_band(r)) for r in manifest["frames"])
    if set(metadata) != {(w["split"], w["world"]) for w in worlds}:
        raise ValueError("retained coverage has different world membership")

    def read_rows(name):
        with (inserted / (name + ".csv")).open(encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    frames, features, changes = (read_rows(k) for k in ("frames", "features", "changes"))
    frame_keys = {(r["split"], r["world"], int(r["frame"])) for r in frames}
    selected_keys = {(r["split"], r["world"], r["frame"]) for r in selection["selections"]}
    if len(frame_keys) != len(frames) or frame_keys != selected_keys:
        raise ValueError("recorded geometry observations differ from their fixed selection")
    recorded = {(r["split"], r["world"], r["frame"]): r for r in observations}
    for row in frames:
        source = recorded[row["split"], row["world"], int(row["frame"])]
        if (int(row["anomaly_total"]) != source["count"] or
                int(row["anomaly_in_range"]) != source["in_range"] or
                int(row["removed_original"]) != source["occluded"]):
            raise ValueError("geometry rows disagree with frozen observation counts")

    report = dict(dataset=str(directory), scope="existing_pool_and_preselected_geometry_evidence_only",
                  definitions=dict(source_frames="distinct source sequence and frame, not independent environments",
                                   returns="world-frame return occurrences; normal backgrounds can repeat across worlds",
                                   distance="median range of all inserted returns; near [2.5,10), middle [10,35), far [35,50] m",
                                   in_range="individual return range in [2.5,50] m; eligible requires at least five",
                                   dimensions="whole continuous object bounds in local axes, with original 1e-6 m padding",
                                   low="local height <=0.2 m; not gravity height or occlusion",
                                   small="no frozen binary definition; retain individual physical dimensions",
                                   central="each field separately within inclusive conditional normal p10-p90",
                                   change_neighborhood="within 2 m of inserted OR removed positions, all sensor ranges",
                                   missing="null means unavailable or inapplicable; zero means measured absence"),
                  sources=[str(coverage / "experiment.json"), str(directory / "manifest.json"),
                           str(coverage / "geometry/normal_cases.json"), str(inserted / "selection.json"),
                           *[str(inserted / (k + ".csv")) for k in ("frames", "features", "changes")]],
                  splits={})
    for split, part in root["splits"].items():
        ws = [w for w in worlds if w["split"] == split]
        rows = [r for r in observations if r["split"] == split]
        expected = previous["inventory"][split]["all"]
        if (len(rows) != part["samples"] or
                sum(r["count"] for r in rows) != expected["anomaly_returns"]["total"] or
                sum(r["in_range"] for r in rows) != expected["in_range_anomaly_returns"]["total"]):
            raise ValueError("pool totals changed from existing coverage")
        cells = dict(all=_coverage_cell(rows, "count"), ranges={}, low={}, shapes={}, count_distance={})
        for band in bands:
            chosen = [r for r in rows if r["band"] == band]
            cells["ranges"][band] = dict(visible=_coverage_cell(chosen, "in_range"),
                                        eligible=_coverage_cell([r for r in chosen if r["in_range"] >= 5], "in_range"))
        low = {w["world"] for w in ws if w["height_m"] <= .2}
        for name, ids in [("low", low), *[(s, {w["world"] for w in ws if w["shape_family"] == s})
                                           for s in sorted({w["shape_family"] for w in ws})]]:
            group = dict(physical_worlds=len(ids))
            for band in ("all", *bands):
                chosen = [r for r in rows if r["world"] in ids and r["in_range"] >= 5
                          and (band == "all" or r["band"] == band)]
                group[band] = _coverage_cell(chosen, "in_range")
            if name == "low":
                cells["low"] = group
            else:
                cells["shapes"][name] = group
        states = dict(no_return=lambda r: r["count"] == 0,
                      outside_only=lambda r: r["count"] > 0 and r["in_range"] == 0,
                      one_to_four=lambda r: 1 <= r["in_range"] <= 4,
                      five_to_nineteen=lambda r: 5 <= r["in_range"] <= 19,
                      at_least_twenty=lambda r: r["in_range"] >= 20)
        for state, predicate in states.items():
            chosen = [r for r in rows if predicate(r)]
            field = "count" if state in ("no_return", "outside_only") else "in_range"
            cells["count_distance"][state] = dict(all=_coverage_cell(chosen, field),
                **{b: None if state == "no_return" else
                   _coverage_cell([r for r in chosen if r["band"] == b], "in_range") for b in bands})
        if sum(c["all"]["world_frames"] for c in cells["count_distance"].values()) != len(rows):
            raise ValueError("return-count states must partition all world frames")
        cells["occlusion"] = dict(
            removed_original=_coverage_cell([r for r in rows if r["occluded"] > 0], "occluded"),
            removed_with_no_anomaly_return=_coverage_cell([r for r in rows if r["occluded"] > 0 and r["count"] == 0], "occluded"),
            object_occlusion_fraction=None,
            limitation="removed original returns do not measure how much of the inserted object is occluded")
        cells["physical_worlds"] = []
        for w in ws:
            chosen = [r for r in rows if r["world"] == w["world"]]
            cells["physical_worlds"].append(dict(
                world=w["world"], shape=w["shape_family"],
                dimensions_m=[w[k] for k in ("length_m", "width_m", "height_m")],
                low=w["world"] in low,
                eligible_bands={b: sum(r["band"] == b and r["in_range"] >= 5 for r in chosen) for b in bands},
                visible_bands={b: sum(r["band"] == b for r in chosen) for b in bands}))
        cells["all_three_eligible_bands"] = [w["world"] for w in cells["physical_worlds"] if all(w["eligible_bands"].values())]
        cells["all_three_visible_bands"] = [w["world"] for w in cells["physical_worlds"] if all(w["visible_bands"].values())]
        evidence = {}
        for scope in ("representative", "trajectory"):
            selected = [r for r in frames if r["split"] == split and r[scope] == "True"]
            detail = dict(observations=_coverage_cell(selected, "anomaly_in_range"), fields={}, changes={})
            for field in GEOMETRY_FIELDS:
                populations = {}
                for population in ("anomaly", "nearby_kept_normal"):
                    chosen = [r for r in features if r["split"] == split and r[scope] == "True"
                              and r["field"] == field and r["population"] == population and r["range_group"] == "in_range"]
                    populations[population] = {k: _coverage_cell([r for r in chosen if int(r[k]) > 0], k)
                                               for k in ("total", "valid", "central", "upper", "hit")}
                    missing = [dict(r, missing=int(r["total"]) - int(r["valid"])) for r in chosen]
                    populations[population]["missing"] = _coverage_cell([r for r in missing if r["missing"] > 0], "missing")
                detail["fields"][field] = populations
                chosen = [r for r in changes if r["split"] == split and r[scope] == "True" and r["field"] == field
                          and r["distance_group"] in ("[0,0.5]", "(0.5,1]", "(1,2]")]
                # Distance bins partition returns; frame identities still count once.
                detail["changes"][field] = {k: _coverage_cell([r for r in chosen if int(r[k]) > 0], k)
                                              for k in ("total", "both_valid", "changed", "before_only", "after_only", "new_hit")}
            evidence[scope] = detail
        selected = [r for r in frames if r["split"] == split]
        evidence["union"] = _coverage_cell(selected, "anomaly_in_range")
        evidence["trajectory_world"] = selection["trajectories"][split]["world"]
        report["splits"][split] = dict(source_sequence=part["source_sequence"], normal_source_sequences=1,
                                       pool=cells, geometry=evidence)
    normal = json.loads((coverage / "geometry/normal_cases.json").read_text())
    report["original_normals"] = dict(cache_frames=normal["cache_frames"], totals=normal["source_totals"],
                                     cases=[{k: r[k] for k in ("sequence", "frame", "slot", "kind", "range", "ray_z", "direction_cell")}
                                            for r in normal["cases"]],
                                     world_count=None, per_frame_concentration=None,
                                     limitation="original sources have no inserted world; cached totals lack per-frame tail counts; cases are six selected anchors")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/data.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--far-limit", type=int, help="check at most this many frames per world and far count stratum")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--new-only", action="store_true", help="inventory the full experiment and measure geometric support only for supplements")
    parser.add_argument("--geometry", action="store_true", help="measure fixed-world geometry and paired unchanged normal returns")
    parser.add_argument("--summarize", action="store_true", help="summarize existing coverage records without reading scans or computing geometry")
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > len(os.sched_getaffinity(0)):
        parser.error("workers times threads must fit the available CPUs")
    protocol = json.loads(args.protocol.read_text())
    if args.dataset:
        protocol["dataset"]["directory"] = str(args.dataset)
    if args.summarize:
        if args.geometry or args.inventory_only or args.new_only or args.far_limit is not None:
            parser.error("--summarize only reads existing full-pool and fixed-selection records")
        coverage = Path(protocol["content_coverage"]["output"]).parent
        report = summarize_existing(Path(protocol["dataset"]["directory"]), coverage)
        output = args.output or coverage / "summary.json"
        _atomic_json(output, report)
        print(json.dumps(dict(output=str(output), scope=report["scope"])), flush=True)
        return
    if args.geometry:
        if args.threads != 1:
            parser.error("paired geometry uses one numerical-library thread per worker")
        geometry_check(protocol, args.data_root, args.output or Path("results/coverage/geometry/inserted"), args.workers)
        return
    args.output = args.output or Path(protocol["content_coverage"]["output"])
    if args.far_limit is not None and args.far_limit < 1:
        parser.error("far limit must be positive")
    root, worlds, totals = inventory(Path(protocol["dataset"]["directory"]))
    selections = select_checks([w for w in worlds if not args.new_only or w["origin"] == "supplement"], args.far_limit)
    report = dict(scope="full_pool_inventory_and_targeted_scan_checks_not_full_pool_geometric_support_coverage",
                  dataset=protocol["dataset"]["directory"], pool_status=root["status"],
                  local_scope="supplement_only" if args.new_only else "all_selected_worlds",
                  geometry_parameters=dict(seed=protocol["seed"],
                                              scale=protocol["geometry"]["sampling_scale"],
                                              boundary=protocol["geometry"]["boundary"],
                                              levels=protocol["geometry"]["levels"],
                                              sampling=protocol["geometry"]["sampling"],
                                              surface=protocol["geometry"]["surface"]),
                  definitions=dict(far="median recorded anomaly range in [35,50] m",
                                   eligible="at least 5 anomaly returns with sensor range in [2.5,50] m",
                                   low="whole continuous object outer local z extent <= 0.2 m, not gravity height",
                                   few="5 to 19 distance-filtered anomaly returns",
                                   selection="per world: earliest maximum count; median eligible few-point frame; all far frames or equally spaced range ranks in separate 5-19 and >=20 point strata; with a limit also retain a weak far representative",
                                   far_checks_per_world_per_count_stratum=args.far_limit,
                                   nearby="normal return within 2 m Euclidean distance of an inserted return",
                                   raised_normal="known normal with valid reference-surface fit and offset in [-0.20,-0.05] m; descriptive proxy, no curb annotation or learned difficulty claim",
                                   counts="return observations, not distinct objects or independent trials; repeated backgrounds remain correlated"),
                  worlds=[{k: v for k, v in w.items() if k != "rows"} for w in worlds],
                  inventory=totals, selection=selections, checks=[])
    _atomic_json(args.output, report)
    print(json.dumps(dict(worlds=len(worlds), frames=sum(len(w["rows"]) for w in worlds), selected=len(selections))), flush=True)
    if args.inventory_only:
        return
    disk, start = host_disk(), time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_initialize, initargs=(protocol, args.data_root, args.threads)) as pool:
        tasks = {pool.submit(check_scan, selection): selection for selection in selections}
        for future in as_completed(tasks):
            record = future.result()
            records.append(record)
            print(json.dumps(dict(done=len(records), total=len(selections), sample=record["sample"],
                                  seconds=round(record["seconds"], 2))), flush=True)
    records.sort(key=lambda r: (r["sample"]["split"], r["sample"]["world"], r["sample"]["frame"]))
    report.update(checks=records, targeted=summarize_checks(records, worlds),
                  execution=dict(workers=args.workers, threads=args.threads, seconds=time.monotonic() - start,
                                 worker_cpu_seconds=sum(r["cpu_seconds"] for r in records),
                                 maximum_worker_rss_bytes=max((r["peak_rss_bytes"] for r in records), default=0),
                                 disk_before=disk, disk_after=host_disk()))
    _atomic_json(args.output, report)


if __name__ == "__main__":
    main()
