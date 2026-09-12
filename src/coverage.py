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

from .data import FrozenDataset, _atomic_json, host_disk, source_identity
from .render import calibrated_ray_grid, shape_from_dict, shape_geometry, shape_relations, primary_structure, canonical_ray_slots_for_source
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
                cohort=entry.get("cohort", "original"),
                family_id=entry.get("family_id"), paired=entry.get("paired", False),
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
    return root, worlds, inventory_totals(worlds, root["splits"])


def inventory_totals(worlds, splits):
    totals = {}
    for split in splits:
        selected = [w for w in worlds if w["split"] == split]
        totals[split] = {}
        for name in conditions(dict(height_m=0), dict(count=0, in_range=0)):
            frames, points, inside, source_frames = {}, {}, {}, set()
            for w in selected:
                rows = [r for r in w["rows"] if conditions(w, r)[name]]
                frames[w["world"]] = len(rows)
                points[w["world"]] = sum(r["count"] for r in rows)
                inside[w["world"]] = sum(r["in_range"] for r in rows)
                source_frames.update((w["source_sequence"], r["frame"]) for r in rows)
            totals[split][name] = dict(frames=concentration(frames),
                                       anomaly_returns=concentration(points),
                                       in_range_anomaly_returns=concentration(inside),
                                       unique_source_frames=len(source_frames))
    return totals


def visibility_events(inserted, occluded, ray_slots, grid):
    """Separate physical-ray events; duplicated file slots are never extra evidence."""
    from scipy.ndimage import label

    a = np.unique(ray_slots[np.asarray(inserted, np.int32)])
    b = np.unique(ray_slots[np.asarray(occluded, np.int32)])
    replaced = np.intersect1d(a, b, assume_unique=True)
    new = np.setdiff1d(a, b, assume_unique=True)
    lost = np.setdiff1d(b, a, assume_unique=True)
    result = dict(anomaly_rays=len(a), changed_native_rays=len(b), new_hit_rays=len(new),
                  replaced_native_rays=len(replaced), lost_native_rays=len(lost),
                  changed_overlap_fraction=len(b) / (len(a) + len(lost)) if len(a) + len(lost) else None)
    for name, ids in (("new", new), ("replaced", replaced), ("lost", lost), ("changed", b)):
        if not len(ids):
            result.update({name + "_" + k: None for k in ("elevation_span_rad", "azimuth_span_rad")})
            result[name + "_components"] = 0
            continue
        directions = grid.directions_sensor[ids]
        angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi))
        result[name + "_azimuth_span_rad"] = float(2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
        result[name + "_elevation_span_rad"] = float(np.ptp(np.arcsin(directions[:, 2])))
        beams = np.argsort(np.argsort(grid.beam_elevation_rad))[grid.beam_ids[ids]]
        columns = grid.column_ids[ids]
        unique_columns = np.sort(np.unique(columns))
        gaps = np.diff(np.r_[unique_columns, unique_columns[0] + grid.columns])
        start = unique_columns[(int(np.argmax(gaps)) + 1) % len(unique_columns)]
        columns = (columns - start) % grid.columns
        lattice = np.zeros((int(np.ptp(beams)) + 1, int(columns.max()) + 1), bool)
        lattice[beams - beams.min(), columns] = True
        result[name + "_components"] = int(label(lattice, np.ones((3, 3), int))[1])
    return result


def _research_initialize(data_root, rays):
    from .scene import STUSequence
    from .protocol import load_protocol

    global _research_sources, _research_grid
    set_num_threads(1)
    _research_sources = {s: STUSequence.open(data_root, protocol=load_protocol(), partition="train",
                           sequence_id=n, label_mode="required") for s, n in (("train", 206), ("validation", 201))}
    _research_grid = calibrated_ray_grid(rays)


def _research_observation(job):
    split, frame, entries = job
    original = _research_sources[split][frame]
    grid = _research_grid
    mapping = canonical_ray_slots_for_source(original, grid)
    identity = source_identity(original)
    rows = []
    for world, path, world_identity, observed, cached in entries:
        if cached is not None:
            if cached["source_identity"] != identity or cached["world_identity"] != world_identity:
                raise ValueError("cached physical events belong to changed source inputs")
            rows.append(cached)
            continue
        with np.load(path, allow_pickle=False) as delta:
            if str(delta["source_identity"]) != identity or str(delta["world_identity"]) != world_identity:
                raise ValueError("visibility delta does not belong to the unchanged source and world")
            inserted, occluded = delta["inserted_slot"], delta["occluded_slot"]
            if len(inserted) != observed["count"] or len(occluded) != observed["occluded"]:
                raise ValueError("frozen event counts disagree with the world manifest")
            events = visibility_events(inserted, occluded, mapping, grid)
            changed_slots = delta["source_slot"]
            xyz = delta["xyzi"][np.searchsorted(changed_slots, inserted), :3]
            _, first = np.unique(mapping[inserted], return_index=True)
            distances = np.linalg.norm(xyz[first], axis=1)
            inside = (distances >= 2.5) & (distances <= 50)
            native_xyz = original.xyzi[occluded[np.unique(mapping[occluded], return_index=True)[1]], :3]
            rows.append(dict(split=split, world=world, world_identity=world_identity, frame=frame,
                source_identity=identity, source_sequence=original.sequence_id,
                official_anomaly_slots=observed["count"], official_in_range_slots=observed["in_range"],
                in_range_rays=int(inside.sum()), range=float(np.median(distances)) if len(distances) else None,
                changed_native_range_median=float(np.median(np.linalg.norm(native_xyz, axis=1))) if len(native_xyz) else None,
                **events))
    return rows


def research_inventory(protocol, data_root, workers):
    """Measure retained content before proposing replacements; never modify frozen samples."""
    import csv
    import hashlib

    config = protocol["research_coverage"]
    directory = Path(protocol["dataset"]["directory"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    root, worlds, _ = inventory(directory)
    disk, started = host_disk(), time.monotonic()
    ray_identity = hashlib.sha256(Path(protocol["calibration"]["rays"]).read_bytes()).hexdigest()
    previous, cached = {}, {}
    if (output / "inventory.json").exists() and (output / "observations.csv").exists():
        old, old_rows = research_records(output)
        if old.get("ray_identity") == ray_identity:
            cached = {(r["world_identity"], r["frame"]): r for r in old_rows}
        if old["parameters"]["morphology"] == config["morphology"]:
            previous = {w["parent"]: w["relations"] for w in old["worlds"]}
    jobs, metadata, relations = defaultdict(list), [], {}
    for split, part in root["splits"].items():
        for entry in part["worlds"]:
            path = directory / entry["path"]
            definition = json.loads((path / "world.json").read_text())
            obj = definition["world"]["objects"][0]
            shape_key = hashlib.sha256(json.dumps(obj["shape"], sort_keys=True).encode()).hexdigest()
            if shape_key not in relations:
                relations[shape_key] = obj["shape"]
            anchor = np.asarray(definition["generation"]["support_plane"]["anchor_world_m"])
            regions = ["/".join(map(str, np.floor((anchor[:2] + shift) / config["regions"]["grid_m"]).astype(int)))
                       for shift in config["regions"]["xy_shifts_m"]]
            meta = next(w for w in worlds if w["split"] == split and w["world"] == path.name)
            metadata.append(dict(split=split, world=path.name, path=str(path), identity=entry["world_identity"],
                parent=shape_key, cohort=entry.get("cohort", "original"), regions=regions,
                anchor_world_m=anchor.tolist(), height_m=meta["height_m"],
                dimensions_m=[meta[k] for k in ("length_m", "width_m", "height_m")],
                relations=None, shape_family=meta["shape_family"], assigned_cell=entry.get("combination"),
                content_check_frames=meta.get("content_check_frames", [])))
            for row in meta["rows"]:
                jobs[split, row["frame"]].append((path.name, str(path / "frames" / f"{row['frame']:06d}.npz"),
                    entry["world_identity"], row, cached.get((entry["world_identity"], row["frame"]))))
    record = dict(dataset=str(directory), dataset_identity=root["configuration_identity"],
        ray_identity=ray_identity, parameters=config, worlds=metadata,
        scope="complete_pool_physical_ray_events_and_world_geometry_not_full_pool_local_features")
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_research_initialize, initargs=(str(data_root), protocol["calibration"]["rays"])) as pool:
        shapes = {pool.submit(shape_relations, shape_from_dict(shape), config["morphology"]): key
                  for key, shape in relations.items() if key not in previous}
        relations.update({key: previous[key] for key in relations if key in previous})
        for future in as_completed(shapes):
            relations[shapes[future]] = future.result()
        for meta in metadata:
            meta["relations"] = relations[meta["parent"]]
        print(json.dumps(dict(event="physical_relations", geometry_parents=len(relations))), flush=True)
        futures = [pool.submit(_research_observation, (split, frame, entries)) for (split, frame), entries in sorted(jobs.items())]
        for done, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if done % 100 == 0 or done == len(futures):
                print(json.dumps(dict(event="ray_events", source_frames=done, total=len(futures))), flush=True)
    rows.sort(key=lambda r: (r["split"], r["world"], r["frame"]))
    with (output / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record["execution"] = dict(workers=workers, seconds=time.monotonic() - started,
                                 host_before=disk, host_after=host_disk(), world_frames=len(rows),
                                 reused_world_frames=sum(e[4] is not None for entries in jobs.values() for e in entries))
    _atomic_json(output / "inventory.json", record)
    print(json.dumps(dict(event="research_inventory", worlds=len(metadata), world_frames=len(rows),
                          seconds=record["execution"]["seconds"])), flush=True)


def research_records(output):
    import csv
    output = Path(output)
    record = json.loads((output / "inventory.json").read_text())
    integer = {"frame", "source_sequence", "official_anomaly_slots", "official_in_range_slots"}
    rows = []
    with (output / "observations.csv").open() as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for k, v in raw.items():
                if k in ("split", "world", "world_identity", "source_identity"):
                    row[k] = v
                elif not v:
                    row[k] = None
                else:
                    row[k] = int(v) if k in integer or k.endswith(("_rays", "_components")) else float(v)
            rows.append(row)
    return record, rows


def research_cells(row, world, config):
    n, distance = row["in_range_rays"], row["range"]
    band = ("near" if 2.5 <= distance < 10 else "middle" if 10 <= distance < 35
            else "far" if 35 <= distance <= 50 else "outside") if distance is not None else "unobservable"
    visible = n >= 5
    low = world["height_m"] <= config["low_height_m"]
    result = dict(near_sparse_1_4=band == "near" and 1 <= n < 5,
        far_at_least_20=band == "far" and n >= 20,
        low_near=low and visible and band == "near", low_middle=low and visible and band == "middle",
        low_far=low and visible and band == "far",
        weak_background_visible=visible and row["changed_native_rays"] <= config["background_change"]["weak_max_changed_rays"])
    for name in ("compact", "elongated", "sheet", "multi_branch", "multiple_contact"):
        result[name + "_visible"] = visible and world["relations"][name] is True
    return band, result


def research_geometry_selection(record, rows, config):
    metadata = {(w["split"], w["world"]): w for w in record["worlds"]}
    strata = defaultdict(list)
    for row in rows:
        band, cells = research_cells(row, metadata[row["split"], row["world"]], config)
        count_bin = int(np.searchsorted(config["return_bins"], row["in_range_rays"], side="right"))
        strata[row["split"], row["world"], band, count_bin, cells["weak_background_visible"]].append(row)
    selected = []
    for key, observations in sorted(strata.items()):
        observations.sort(key=lambda r: r["frame"])
        ids = np.unique(np.linspace(0, len(observations)-1,
                         min(len(observations), config["geometry"]["frames_per_world_joint_stratum"]), dtype=int))
        selected.extend(dict(observations[i], stratum=list(key[2:])) for i in ids)
    lookup = {(r["split"], r["world"], r["frame"]): r for r in rows}
    for meta in metadata.values():
        for anchor in meta.get("content_check_frames", []):
            for frame in range(max(0, anchor-2), anchor+3):
                row = lookup.get((meta["split"], meta["world"], frame))
                if row is not None:
                    selected.append(dict(row, stratum=["native_witness_window"]))
    return list({(r["split"], r["world"], r["frame"]): r for r in selected}.values())


def native_context_types(values, reference, sparse_threshold):
    """Original normal geometry categories; no inserted geometry or task score."""
    from .probes import _cells
    kinds = {k: np.zeros(len(values["range"]), bool) for k in ("smooth", "rough", "sparse")}
    cells = _cells(values, reference["edges"])["direction"]
    for cell in np.unique(cells):
        q = reference["quantiles"].get(int(cell))
        if q and all(q[f] is not None for f in GEOMETRY_FIELDS):
            take = cells == cell
            kinds["smooth"][take] = ((values["surface_residual"][take] <= q["surface_residual"][0])
                                    & (values["normal_change"][take] <= q["normal_change"][0]))
            kinds["rough"][take] = ((values["surface_residual"][take] >= q["surface_residual"][1])
                                   | (values["normal_change"][take] >= q["normal_change"][1]))
    kinds["rough"] &= ~kinds["smooth"]
    kinds["sparse"] = values["neighbor_count"] < sparse_threshold
    return kinds


def _research_geometry_frame(job):
    from .data import FrozenFrame
    from .profile import observed_geometry
    from .probes import _cells

    split, frame, entries = job
    original = _research_sources[split][frame]
    mapping = canonical_ray_slots_for_source(original, _research_grid)
    _, first = np.unique(mapping[original.real_slots], return_index=True)
    native_slots = original.real_slots[first]
    normal_slots = native_slots[original.labels.semantic_target[native_slots] != 255]
    normal_xyz = original.xyzi[normal_slots, :3]
    queries, prepared = [], []
    config = _research_config
    for observed, meta in entries:
        path = Path(meta["path"]) / "frames" / f"{frame:06d}.npz"
        sample = FrozenFrame.load(path, original, meta["identity"])
        source = sample.source
        anomaly = np.flatnonzero(sample.inserted_mask)
        anomaly = anomaly[np.unique(mapping[anomaly], return_index=True)[1]]
        edits = np.concatenate((source.xyzi[anomaly, :3], original.xyzi[sample.occluded_original_mask, :3]))
        distances = cKDTree(edits).query(normal_xyz)[0] if len(edits) else np.full(len(normal_slots), np.inf)
        keep = ~(sample.inserted_mask | sample.occluded_original_mask)[normal_slots]
        all_nearby = normal_slots[keep & (distances <= config["normal_contrasts"]["nearby_m"])]
        nearby = all_nearby
        if len(nearby) > config["geometry"]["normal_queries_per_frame"]:
            nearby = nearby[np.linspace(0, len(nearby)-1, config["geometry"]["normal_queries_per_frame"], dtype=int)]
        if not (np.array_equal(original.xyzi[nearby], source.xyzi[nearby])
                and np.array_equal(original.labels.packed[nearby], source.labels.packed[nearby])
                and np.all(sample.anomaly_target[nearby] == 0)):
            raise ValueError("retained normal identity or label changed")
        queries.append(all_nearby)
        prepared.append((observed, meta, sample, anomaly, nearby, all_nearby))
    query = np.unique(np.concatenate(queries))
    before = observed_geometry(original.xyzi[original.real_slots], original.real_slots, query)
    results = []
    for observed, meta, sample, anomaly, nearby, all_nearby in prepared:
        source = sample.source
        query_after = np.r_[anomaly, nearby]
        after = observed_geometry(source.xyzi[source.real_slots], source.real_slots, query_after)
        old = {k: v[np.searchsorted(query, nearby)] for k, v in before.items()}
        ac = _geometry_condition(after)
        cells = _cells(after, _geometry_reference["edges"])["direction"]
        old_cells = _cells(old, _geometry_reference["edges"])["direction"]
        inside = (after["range"] >= 2.5) & (after["range"] <= 50)
        is_anomaly = np.arange(len(query_after)) < len(anomaly)
        central = inside & is_anomaly & ac["surface_residual"]["central"] & ac["normal_change"]["central"]
        threshold = config["normal_contrasts"]["sparse_neighbor_threshold"]
        normal_types = native_context_types(old, _geometry_reference, threshold)
        full_old = {k: v[np.searchsorted(query, all_nearby)] for k, v in before.items()}
        full_types = native_context_types(full_old, _geometry_reference, threshold)
        distance = (cKDTree(source.xyzi[anomaly, :3]).query(original.xyzi[all_nearby, :3])[0]
                    if len(anomaly) else np.full(len(all_nearby), np.inf))
        native = {}
        for kind, mask in full_types.items():
            ids = all_nearby[mask & (distance <= config["normal_contrasts"]["nearby_m"])
                             & (full_old["range"] >= 2.5) & (full_old["range"] <= 50)]
            ids = ids[np.unique(original.xyzi[ids, :3], axis=0, return_index=True)[1]]
            native[kind] = dict(positions=len(ids), source_slots=ids.tolist())
        contrasts = {}
        for kind, mask in normal_types.items():
            matched_normal, matched_anomaly = set(), set()
            for cell in np.unique(cells[central]):
                ai = np.flatnonzero(central & (cells == cell))
                ni = np.flatnonzero(mask & (old_cells == cell))
                if not len(ni):
                    continue
                neighbors = cKDTree(source.xyzi[nearby[ni], :3]).query_ball_point(
                    source.xyzi[query_after[ai], :3], config["normal_contrasts"]["nearby_m"])
                for a, ns in zip(ai, neighbors):
                    if ns:
                        matched_anomaly.add(int(a))
                        matched_normal.update(ni[ns].tolist())
            normal_ids = nearby[sorted(matched_normal)]
            world_xyz = original.xyzi[normal_ids, :3].astype(float) @ original.lidar_pose[:3, :3].T + original.lidar_pose[:3, 3]
            regions = [["/".join(map(str, x)) for x in np.floor((world_xyz[:, :2] + shift) /
                         config["regions"]["grid_m"]).astype(int)] for shift in config["regions"]["xy_shifts_m"]]
            contrasts[kind] = dict(normal_queries=len(matched_normal), central_anomaly_queries=len(matched_anomaly),
                normal_source_slots=normal_ids.tolist(), normal_regions=regions,
                central_anomaly_slots=query_after[sorted(matched_anomaly)].tolist())
        changes = np.zeros(len(nearby), bool)
        for field, threshold in (("surface_residual", config["normal_contrasts"]["changed_residual_m"]),
                                 ("normal_change", config["normal_contrasts"]["changed_normal_rad"])):
            a, b = old[field], after[field][len(anomaly):]
            changes |= np.isfinite(a) & np.isfinite(b) & (np.abs(b.astype(float)-a) > threshold)
        populations = {}
        for label, mask in (("anomaly", is_anomaly & inside), ("nearby_original_normal", ~is_anomaly & inside)):
            populations[label] = dict(total=int(mask.sum()),
                insufficient_neighbors=int(np.sum(mask & (after["neighbor_count"] < 8))),
                **{field: dict(valid=int(np.sum(mask & ac[field]["valid"])),
                               covered=int(np.sum(mask & ac[field]["covered"])),
                               missing=int(np.sum(mask & ~ac[field]["valid"])),
                               valid_without_reference=int(np.sum(mask & ac[field]["valid"] & ~ac[field]["covered"])))
                   for field in GEOMETRY_FIELDS})
        obj = json.loads((Path(meta["path"]) / "world.json").read_text())["world"]["objects"][0]
        rotation, center = np.asarray(obj["rotation_world_from_local"]), np.asarray(obj["translation_world_m"])
        view = (original.lidar_pose[:3, 3] - center) @ rotation
        view /= np.linalg.norm(view)
        xyz = source.xyzi[anomaly, :3].astype(float) @ original.lidar_pose[:3, :3].T + original.lidar_pose[:3, 3]
        local = (xyz-center) @ rotation
        octants = np.unique(((local >= 0) * [1, 2, 4]).sum(axis=1))
        results.append(dict(split=split, world=meta["world"], frame=frame, world_identity=meta["identity"],
            source_identity=observed["source_identity"], stratum=observed["stratum"],
            anomaly_in_range=int(np.sum(is_anomaly & inside)), central_anomaly=int(central.sum()),
            nearby_original_normal_queries=len(nearby), changed_normal_queries=int(changes.sum()),
            changed_normal_source_slots=nearby[changes].tolist(),
            native_context=native, contrasts=contrasts, populations=populations, view_local=view.tolist(),
            visible_octants=octants.tolist(), visible_local_span_m=np.ptp(local, axis=0).tolist() if len(local) else None))
    return results


def research_geometry(protocol, data_root, workers):
    global _geometry_reference, _research_config
    config = protocol["research_coverage"]
    output = Path(config["output"])
    record, rows = research_records(output)
    selected = research_geometry_selection(record, rows, config)
    metadata = {(w["split"], w["world"]): w for w in record["worlds"]}
    cached = {}
    path = output / "geometry.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["parameters"] == config:
            cached = {(r["world_identity"], r["source_identity"]): r for r in previous["observations"]
                      if "native_context" in r}
    jobs, results = defaultdict(list), []
    for row in selected:
        identity = row["world_identity"], row["source_identity"]
        if identity in cached:
            results.append(dict(cached[identity], stratum=row["stratum"]))
        else:
            jobs[row["split"], row["frame"]].append((row, metadata[row["split"], row["world"]]))
    started = time.monotonic()
    _geometry_reference, _research_config = geometry_reference(), config
    _atomic_json(output / "selection.json", dict(dataset=record["dataset"], parameters=config,
        selections=selected, selection="three_evenly_spaced_source_frames_per_world_joint_stratum_plus_native_witness_windows",
        scope="selected_observations_only; proportions_do_not_estimate_complete_pool_geometry"))
    print(json.dumps(dict(event="research_geometry_start", selected=len(selected), reused=len(results), source_jobs=len(jobs))), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
            initializer=_research_initialize, initargs=(str(data_root), protocol["calibration"]["rays"])) as pool:
        futures = [pool.submit(_research_geometry_frame, (s, f, entries)) for (s, f), entries in sorted(jobs.items())]
        for done, future in enumerate(as_completed(futures), 1):
            results.extend(future.result())
            if done % 50 == 0 or done == len(futures):
                print(json.dumps(dict(event="research_geometry", source_jobs=done, total=len(futures))), flush=True)
    results.sort(key=lambda r: (r["split"], r["world"], r["frame"]))
    _atomic_json(path, dict(dataset=record["dataset"], parameters=config, observations=results,
        execution=dict(seconds=time.monotonic()-started, workers=workers),
        scope="systematic_joint_stratum_observations_not_a_full_pool_geometry_census"))


def research_cell_summary(rows, metadata, limits, config):
    parents, regions, frames = defaultdict(int), [defaultdict(int), defaultdict(int)], defaultdict(int)
    parent_frames = defaultdict(set)
    for row in rows:
        meta = metadata[row["split"], row["world"]]
        parents[meta["parent"]] += row["in_range_rays"]
        parent_frames[meta["parent"]].add(row["frame"])
        frames[row["frame"]] += row["in_range_rays"]
        for i, region in enumerate(meta["regions"]):
            regions[i][region] += row["in_range_rays"]
    pc, rc, fc = concentration(parents), [concentration(x) for x in regions], concentration(frames)
    repeated = sum(len(fs) >= config["minimum_frames_per_parent"] for fs in parent_frames.values())
    measured = dict(geometry_parents=len(parents), recurring_parents=repeated,
        support_regions=min(len(x) for x in regions), source_frames=len(frames), world_frames=len(rows),
        anomaly_returns=pc["total"], maximum_parent_share=pc["top_one_share"],
        maximum_region_share=max((r["top_one_share"] or 0 for r in rc), default=0) if rows else None,
        maximum_source_frame_share=fc["top_one_share"], parent_returns=pc, region_returns=rc,
        source_frame_returns=fc)
    failures = [key for key, value in limits.items() if measured[key] < value]
    if repeated < limits["geometry_parents"]:
        failures.append("recurring_geometry_parents")
    if measured["maximum_parent_share"] is None or measured["maximum_parent_share"] > config["maximum_parent_return_share"]:
        failures.append("parent_concentration")
    if measured["maximum_region_share"] is None or measured["maximum_region_share"] > config["maximum_region_return_share"]:
        failures.append("region_concentration")
    return dict(**measured, passed=not failures, failures=failures)


def research_summary(protocol):
    config = protocol["research_coverage"]
    output = Path(config["output"])
    record, rows = research_records(output)
    metadata = {(w["split"], w["world"]): w for w in record["worlds"]}
    geometry_path = output / "geometry.json"
    geometry_record = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    if geometry_record and geometry_record["parameters"] != config:
        raise ValueError("geometry summary cannot use measurements from different declared conditions")
    geometry = geometry_record["observations"] if geometry_record else []
    measured_geometry = {(r["world_identity"], r["source_identity"]): r for r in geometry}
    report = dict(dataset=record["dataset"], parameters=config, data_preparation_accepted=False, groups={})
    train_cells = None
    for group, limits in config["minimum"].items():
        chosen = [r for r in rows if (r["split"] == "train" if group == "train" else
                  r["split"] == "validation")]
        cell_rows, world_bands = defaultdict(list), defaultdict(lambda: defaultdict(set))
        world_views, world_fragments = defaultdict(list), defaultdict(set)
        geometry_rows = []
        for row in chosen:
            world = metadata[row["split"], row["world"]]
            band, cells = research_cells(row, world, config)
            if row["in_range_rays"] >= 5 and band in ("near", "middle", "far"):
                world_bands[world["identity"]][band].add(row["frame"])
            for key, active in cells.items():
                if active:
                    cell_rows[key].append(row)
            measured = measured_geometry.get((row["world_identity"], row["source_identity"]))
            if measured is None:
                continue
            geometry_rows.append(measured)
            if row["in_range_rays"] >= 5:
                world_views[world["identity"]].append(measured["view_local"])
                world_fragments[world["identity"]].add(tuple(measured["visible_octants"]))
            for kind, contrast in measured["contrasts"].items():
                if (contrast["normal_queries"] >= config["normal_contrasts"]["minimum_normal_queries"]
                        and contrast["central_anomaly_queries"] >= config["normal_contrasts"]["minimum_central_anomaly_queries"]):
                    cell_rows[f"normal_{kind}_with_nonextreme_anomaly"].append(dict(row,
                        in_range_rays=contrast["central_anomaly_queries"], normal_queries=contrast["normal_queries"],
                        normal_source_slots=contrast.get("normal_source_slots"), normal_regions=contrast.get("normal_regions")))
            if (row["in_range_rays"] >= 5 and measured["changed_normal_queries"] >=
                    config["normal_contrasts"]["minimum_changed_normal_queries"]):
                cell_rows["changed_neighborhood_retained_normal"].append(dict(row, normal_queries=measured["changed_normal_queries"]))
        longitudinal = {w for w, bands in world_bands.items()
                        if all(len(bands[b]) >= config["minimum_frames_per_parent"] for b in ("near", "middle", "far"))}
        # A yaw variant cannot supply a missing view to another world's trajectory.
        longitudinal = {w for w in longitudinal if len(world_views[w]) >= 2
                        and np.min(np.asarray(world_views[w]) @ np.asarray(world_views[w]).T)
                        <= np.cos(np.deg2rad(config["longitudinal"]["minimum_view_span_degrees"]))
                        and len(world_fragments[w]) >= config["longitudinal"]["minimum_visible_octant_patterns"]}
        cell_rows["same_parent_near_middle_far"] = [r for r in chosen if r["in_range_rays"] >= 5
                                      and r["world_identity"] in longitudinal]
        cells = {}
        for key in config["core_cells"]:
            if not geometry and (key.startswith("normal_") or key == "changed_neighborhood_retained_normal"):
                cells[key] = dict(passed=False, failures=["geometry_measurement_pending"], measured=None)
            else:
                cells[key] = research_cell_summary(cell_rows[key], metadata, limits, config)
                if key.startswith("normal_"):
                    bands = {research_cells(r, metadata[r["split"], r["world"]], config)[0] for r in cell_rows[key]}
                    cells[key]["distance_bins"] = sorted(bands)
                    cells[key]["normal_queries"] = sum(r["normal_queries"] for r in cell_rows[key])
                    if all(r["normal_source_slots"] is not None for r in cell_rows[key]):
                        cells[key]["unique_normal_source_positions"] = len({(r["source_identity"], slot)
                            for r in cell_rows[key] for slot in r["normal_source_slots"]})
                        cells[key]["normal_spatial_regions"] = [len({region for r in cell_rows[key]
                            for region in r["normal_regions"][i]}) for i in range(len(config["regions"]["xy_shifts_m"]))]
                    else:
                        cells[key]["unique_normal_source_positions"] = None
                        cells[key]["normal_spatial_regions"] = None
                    if len(bands) < config["normal_contrasts"]["minimum_distance_bins"]:
                        cells[key]["failures"].append("distance_crossing")
                        cells[key]["passed"] = False
        report["groups"][group] = dict(cells=cells, complete_world_frames=len(chosen),
            invisible_frames=sum(r["anomaly_rays"] == 0 for r in chosen),
            no_unique_official_support=sum(r["in_range_rays"] < 5 for r in chosen),
            normal_source_sequences=1, selected_geometry_frames=len(geometry_rows),
            geometry_scope="systematic_selected_observations_only", geometry_missingness={
                population: {field: {key: sum(r["populations"][population][field][key] for r in geometry_rows)
                                    for key in ("valid", "covered", "missing", "valid_without_reference")}
                             for field in GEOMETRY_FIELDS}
                for population in ("anomaly", "nearby_original_normal")})
        if group == "train":
            train_cells = cell_rows
    report["sampling"] = research_sampling(rows, metadata, train_cells, config, output)
    report["generation_matrix"] = generation_matrix(rows, metadata, measured_geometry, config)
    report["data_preparation_accepted"] = (all(c["passed"] for g in report["groups"].values()
                                               for c in g["cells"].values()) and report["sampling"]["passed"]
        and report["generation_matrix"]["quotas_satisfied"])
    _atomic_json(output / "summary.json", report)
    print(json.dumps({g: {k: v["failures"] for k, v in x["cells"].items() if not v["passed"]}
                      for g, x in report["groups"].items()}), flush=True)
    return report


def generation_matrix(rows, metadata, measured, config):
    """Separate complete trajectories from selected native-geometry witnesses."""
    groups = defaultdict(list)
    for row in rows:
        cell = metadata[row["split"], row["world"]].get("assigned_cell")
        if cell is not None:
            groups[row["split"], cell].append(row)
    result = dict(groups={}, quotas_satisfied=len(groups) == 2*config["generation_cells"]["count"],
        geometry_scope="selected_declared_scans; native_positions_are_source_slot_identities_not_independent_environments")
    for (split, cell), observations in sorted(groups.items()):
        worlds = {r["world"] for r in observations}
        count = config["generation_cells"]["worlds_per_cell"][split]
        result["quotas_satisfied"] &= len(worlds) == count
        parents, regions, states = Counter(), [Counter(), Counter()], defaultdict(list)
        witnesses, native_ids, native_frames = defaultdict(set), set(), set()
        missing = {f: Counter() for f in GEOMETRY_FIELDS}
        selected = 0
        for row in observations:
            meta = metadata[split, row["world"]]
            parents[meta["parent"]] += row["in_range_rays"]
            for i, region in enumerate(meta["regions"]):
                regions[i][region] += row["in_range_rays"]
            band, flags = research_cells(row, meta, config)
            support = int(np.searchsorted(config["return_bins"], row["in_range_rays"], side="right"))
            states[f"{band}/{support}/{int(flags['weak_background_visible'])}"].append(row)
            g = measured.get((row["world_identity"], row["source_identity"]))
            if g is None:
                continue
            selected += 1
            for field in GEOMETRY_FIELDS:
                missing[field].update(g["populations"]["anomaly"][field])
            native = g.get("native_context", {}).get(cell.rsplit("/", 1)[1])
            if native and native["positions"] >= 5 and row["in_range_rays"] >= 5:
                witnesses[meta["parent"]].add(row["frame"])
                native_frames.add(row["frame"])
                native_ids.update((row["source_identity"], slot) for slot in native["source_slots"])
        state_table = {}
        for state, rs in states.items():
            state_table[state] = dict(worlds=len({r["world"] for r in rs}),
                geometry_parents=len({metadata[split, r["world"]]["parent"] for r in rs}),
                source_frames=len({r["frame"] for r in rs}), world_frames=len(rs),
                in_range_anomaly_rays=sum(r["in_range_rays"] for r in rs))
        result["groups"].setdefault(split, {})[cell] = dict(worlds=len(worlds), geometry_parents=len(parents),
            support_regions=[len(r) for r in regions], source_frames=len({r["frame"] for r in observations}),
            complete_world_frames=len(observations), in_range_anomaly_rays=sum(parents.values()),
            maximum_parent_return_share=concentration(parents)["top_one_share"],
            maximum_region_return_share=[concentration(r)["top_one_share"] for r in regions],
            zero_anomaly_world_frames=sum(r["anomaly_rays"] == 0 for r in observations),
            selected_geometry_frames=selected, geometry_missingness={f: dict(v) for f, v in missing.items()},
            native_witness_source_frames=len(native_frames), native_witness_unique_source_positions=len(native_ids),
            recurring_native_witness_parents=sum(len(fs) >= config["minimum_frames_per_parent"] for fs in witnesses.values()),
            observation_states=state_table)
    return result


def research_sampling(rows, metadata, cells, config, output):
    """Inspect the exact probability of every full input scan, without training."""
    train = [r for r in rows if r["split"] == "train"]
    indices = {(r["world_identity"], r["frame"]): i for i, r in enumerate(train)}
    count = np.array([r["in_range_rays"] for r in train])
    probabilities = np.zeros(len(train))

    def distribute(chosen, mass, balanced_counts):
        regions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in chosen:
            meta = metadata[row["split"], row["world"]]
            regions[meta["regions"][0]][meta["parent"]][row["world_identity"]].append(row)
        for parents in regions.values():
            means = {p: np.mean([count[indices[r["world_identity"], r["frame"]]] for rs in worlds.values() for r in rs])
                     for p, worlds in parents.items()}
            weights = {p: 1 / np.sqrt(max(1, means[p])) if balanced_counts else 1 for p in parents}
            for parent, worlds in parents.items():
                pmass = mass / len(regions) * weights[parent] / sum(weights.values())
                for rs in worlds.values():
                    for row in rs:
                        probabilities[indices[row["world_identity"], row["frame"]]] += pmass / len(worlds) / len(rs)

    assignments = {metadata[r["split"], r["world"]].get("assigned_cell") for r in train}
    if None not in assignments:
        for cell in sorted(assignments):
            observations = [r for r in train if metadata[r["split"], r["world"]]["assigned_cell"] == cell]
            states = defaultdict(list)
            for r in observations:
                band, flags = research_cells(r, metadata[r["split"], r["world"]], config)
                support = int(np.searchsorted(config["return_bins"], r["in_range_rays"], side="right"))
                states[band, support, flags["weak_background_visible"]].append(r)
            for selected in states.values():
                distribute(selected, config["sampling"]["core_mixture"] / len(assignments) / len(states), True)
            distribute(observations, config["sampling"]["all_frames_mixture"] / len(assignments), False)
    else:
        active = [cells[k] for k in config["core_cells"] if cells[k]]
        if active:
            for selected in active:
                distribute(selected, config["sampling"]["core_mixture"] / len(active), True)
        distribute(train, config["sampling"]["all_frames_mixture"] if active else 1., False)
    if not np.isclose(probabilities.sum(), 1., atol=1e-12) or not np.all(probabilities > 0):
        raise ValueError("sampling must preserve every world-frame with positive normalized mass")
    def exposure(probability):
        totals = {name: defaultdict(float) for name in ("parent", "region", "shifted_region", "source_frame")}
        for row, p in zip(train, probability * count):
            meta = metadata[row["split"], row["world"]]
            for name, key in (("parent", meta["parent"]), ("region", meta["regions"][0]),
                              ("shifted_region", meta["regions"][1]), ("source_frame", str(row["frame"]))):
                totals[name][key] += float(p)
        total = float(np.dot(probability, count))
        return dict(expected_anomaly_returns_per_scan=total,
            maximum_positive_share={name: max(values.values(), default=0) / total if total else None
                                    for name, values in totals.items()},
            positive_exposure={name: dict(v) for name, v in totals.items()},
            invisible_frame_probability=float(sum(p for r, p in zip(train, probability) if r["anomaly_rays"] == 0)))
    measured = exposure(probabilities)
    shares = measured["maximum_positive_share"]
    passed = (shares["parent"] <= config["maximum_parent_return_share"] and
              max(shares["region"], shares["shifted_region"]) <= config["maximum_region_return_share"])
    np.savez_compressed(output / "sampling.npz", world_identity=np.array([r["world_identity"] for r in train]),
        frame=np.array([r["frame"] for r in train], np.int32), probability=probabilities)
    combination_probability = {cell: float(sum(p for r, p in zip(train, probabilities)
        if metadata[r["split"], r["world"]].get("assigned_cell") == cell)) for cell in assignments if cell is not None}
    return dict(passed=passed, rule=config["sampling"], training_executed=False,
        combination_probability=combination_probability,
        uniform_world_frame=exposure(np.full(len(train), 1 / len(train))), proposed=measured,
        inactive_core_cells=[k for k in config["core_cells"] if not cells[k]],
        weights=str(output / "sampling.npz"), scope="expected_full_scan_and_anomaly_return_exposure; not_a_model_loss_or_gradient_measure")


def balanced_assignment(candidates, quotas, minimum_regions):
    """Assign complete worlds once, with exact cell quotas and spatial diversity."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_array
    import warnings

    entries = [(w, cell) for w in candidates for cell in sorted(w["eligible_cells"])]
    if any(sum(cell in w["eligible_cells"] for w in candidates) < quota for cell, quota in quotas.items()):
        return None, dict(status=2, message="insufficient eligible worlds for one or more declared cells")
    cell_indices, world_indices, parent_indices = defaultdict(list), defaultdict(list), defaultdict(list)
    region_indices, states = defaultdict(list), defaultdict(list)
    scores = []
    for i, (w, cell) in enumerate(entries):
        cell_indices[cell].append(i)
        world_indices[w["identity"]].append(i)
        parent_indices[cell, w["parent"]].append(i)
        for grid, region in enumerate(w["regions"]):
            region_indices[cell, grid, region].append(i)
        for state in w["states"]:
            states[cell, state].append(i)
        scores.append(-.01 * min(w["eligible_cells"][cell], 10) - 1e-8 * (len(entries)-i))
    columns, row_ids, values, lower, upper = [], [], [], [], []
    def constrain(ids, lo, hi, weights=None):
        row_ids.extend([len(lower)] * len(ids)); columns.extend(ids)
        values.extend([1.] * len(ids) if weights is None else weights)
        lower.append(lo); upper.append(hi)
    for cell, quota in quotas.items():
        constrain(cell_indices[cell], quota, quota)
    for ids in world_indices.values():
        constrain(ids, 0, 1)
    for ids in parent_indices.values():
        constrain(ids, 0, 1)
    region_presence = defaultdict(list)
    for (cell, grid, region), ids in region_indices.items():
        j = len(scores); scores.append(-.001)
        constrain(ids + [j], 0, np.inf, [1.] * len(ids) + [-1.])
        constrain(ids + [j], -np.inf, 0, [1.] * len(ids) + [-quotas[cell]])
        constrain(ids, 0, int(np.ceil(quotas[cell]/2)))
        region_presence[cell, grid].append(j)
    for cell in quotas:
        for grid in (0, 1):
            constrain(region_presence[cell, grid], minimum_regions, np.inf)
    for (cell, state), ids in states.items():
        j = len(scores)
        # Complementary physical observations take priority over repeated witnesses.
        scores.append(-4. if state.startswith("far/4") else -1.)
        constrain(ids + [j], 0, np.inf, [1.] * len(ids) + [-1.])
    matrix = csc_array((np.asarray(values), (np.asarray(row_ids, np.int32), np.asarray(columns, np.int32))),
                       shape=(len(lower), len(scores)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*")
        result = milp(np.asarray(scores), integrality=np.ones(len(scores)), bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, lower, upper),
            options=dict(time_limit=60, mip_rel_gap=1e-4, threads=1))
    if result.x is None:
        return None, dict(status=int(result.status), message=result.message)
    chosen = [(w["identity"], cell) for (w, cell), x in zip(entries, result.x) if x > .5]
    if Counter(c for _, c in chosen) != Counter(quotas) or len({w for w, _ in chosen}) != len(chosen):
        raise ValueError("balanced selection did not satisfy exact world quotas")
    return chosen, dict(status=int(result.status), message=result.message, objective=float(result.fun))


def research_select(protocol, data_root):
    """Select from measured old and new candidates; physical samples are not rewritten."""
    config = protocol["research_coverage"]
    output, directory = Path(config["output"]), Path(protocol["dataset"]["directory"])
    record, rows = research_records(output)
    if Path(record["dataset"]).resolve() != directory.resolve():
        raise ValueError("candidate measurements belong to another pool")
    geometry = json.loads((output / "geometry.json").read_text())
    if geometry["parameters"] != config:
        raise ValueError("candidate geometry uses different declared conditions")
    expected = {(r["world_identity"], r["source_identity"]) for r in research_geometry_selection(record, rows, config)}
    actual = {(r["world_identity"], r["source_identity"]) for r in geometry["observations"]}
    if expected != actual:
        raise ValueError("complete the declared geometry observations for every candidate before selection")
    root = json.loads((directory / "manifest.json").read_text())
    if root["status"] != "candidates_complete":
        raise ValueError("finish candidate generation before balanced selection")
    observed, measured = defaultdict(list), defaultdict(list)
    for r in rows:
        observed[r["world_identity"]].append(r)
    for g in geometry["observations"]:
        measured[g["world_identity"]].append(g)
    report = dict(dataset=str(directory), rules=config["generation_cells"], groups={}, selected=[], rejected=[])
    entries = {e["world_identity"]: e for part in root["splits"].values() for e in part["worlds"]}
    all_cells = [f"{s}/{h}/{b}" for s in protocol["proposals"]["structures"]
                 for h in protocol["proposals"]["heights"] for b in protocol["proposals"]["backgrounds"]]
    selected = {}
    for split, part in root["splits"].items():
        candidates = []
        for meta in record["worlds"]:
            if meta["split"] != split:
                continue
            structure = primary_structure(meta["relations"])
            height = "low" if meta["height_m"] <= config["low_height_m"] else "raised"
            eligible = {}
            for kind in protocol["proposals"]["backgrounds"]:
                frames = {g["frame"] for g in measured[meta["identity"]] if g["anomaly_in_range"] >= 5
                          and g["native_context"][kind]["positions"] >= 5}
                if frames and structure is not None:
                    eligible[f"{structure}/{height}/{kind}"] = len(frames)
            states, core = set(), Counter()
            for r in observed[meta["identity"]]:
                band, flags = research_cells(r, meta, config)
                support = int(np.searchsorted(config["return_bins"], r["in_range_rays"], side="right"))
                states.add(f"{band}/{support}/{int(flags['weak_background_visible'])}")
                core.update(k for k, active in flags.items() if active)
            for g in measured[meta["identity"]]:
                core.update(f"normal_{k}_with_nonextreme_anomaly" for k, c in g["contrasts"].items()
                            if c["normal_queries"] >= config["normal_contrasts"]["minimum_normal_queries"]
                            and c["central_anomaly_queries"] >= config["normal_contrasts"]["minimum_central_anomaly_queries"])
                if g["anomaly_in_range"] >= 5 and g["changed_normal_queries"] >= config["normal_contrasts"]["minimum_changed_normal_queries"]:
                    core["changed_neighborhood_retained_normal"] += 1
            states.update("core/" + k for k, n in core.items() if n >= config["minimum_frames_per_parent"])
            candidates.append(dict(meta, eligible_cells=eligible, states=sorted(states)))
        quotas = {c: config["generation_cells"]["worlds_per_cell"][split] for c in all_cells}
        counts = {cell: sum(cell in w["eligible_cells"] for w in candidates) for cell in all_cells}
        chosen, solver = balanced_assignment(candidates, quotas, config["generation_cells"]["minimum_regions_per_cell"])
        report["groups"][split] = dict(candidate_worlds=len(candidates), eligible_worlds_by_cell=counts,
            quotas=quotas, solver=solver, selected_worlds=len(chosen) if chosen else 0)
        if chosen is None:
            continue
        selected[split] = sorted(chosen, key=lambda x: (x[1], x[0]))
    if len(selected) != len(root["splits"]):
        report["selection_complete"] = False
        _atomic_json(output / "balance.json", report)
        print(json.dumps(report["groups"], ensure_ascii=False), flush=True)
        return report
    identities = {identity for chosen in selected.values() for identity, _ in chosen}
    parents = {split: {w["parent"] for w in record["worlds"] if w["split"] == split and w["identity"] in identities}
               for split in selected}
    if parents["train"] & parents["validation"]:
        raise ValueError("a geometry parent crosses normal source splits")
    for meta in record["worlds"]:
        (report["selected"] if meta["identity"] in identities else report["rejected"]).append(
            {k: meta[k] for k in ("split", "world", "path", "identity", "parent")})
    for split, chosen in selected.items():
        dataset = FrozenDataset(directory, data_root, split, allow_candidates=True)
        chosen_ids, seen = {identity for identity, _ in chosen}, set()
        for path, identity, frame in dataset.samples:
            if identity in chosen_ids and not path.is_file():
                raise ValueError("a selected complete-world frame is missing")
        for identity, cell in chosen:
            if cell in seen:
                continue
            witness = next(g for g in measured[identity] if g["anomaly_in_range"] >= 5
                and g["native_context"][cell.rsplit("/", 1)[1]]["positions"] >= 5)
            from .data import FrozenFrame
            path = directory / entries[identity]["path"] / "frames" / f"{witness['frame']:06d}.npz"
            _ = FrozenFrame.load(path, dataset.sequence[witness["frame"]], identity)
            seen.add(cell)
        report["groups"][split]["readable_cells"] = len(seen)
        report["groups"][split]["all_frame_references_exist"] = True
        root["splits"][split]["worlds"] = [dict(entries[identity], combination=cell) for identity, cell in chosen]
        root["splits"][split]["samples"] = len(chosen) * protocol["dataset"]["splits"][split]["frames_per_world"]
        root["splits"][split]["selection"] = dict(rule=config["generation_cells"], worlds=len(chosen))
    report["selection_complete"] = True
    root["status"] = "frozen"
    root["balanced_selection"] = str(output / "balance.json")
    root["information_use"] = "normal_206_201; old_and_new_share_exact_240_120_world_quotas; no_val19_fitting"
    _atomic_json(output / "balance.json", report)
    _atomic_json(directory / "manifest.json", root)
    print(json.dumps({s: g["selected_worlds"] for s, g in report["groups"].items()}), flush=True)
    return report


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
    parser.add_argument("--research", choices=("inventory", "geometry", "summary", "select"), help="measure content or select balanced complete worlds")
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > len(os.sched_getaffinity(0)):
        parser.error("workers times threads must fit the available CPUs")
    protocol = json.loads(args.protocol.read_text())
    if args.dataset:
        protocol["dataset"]["directory"] = str(args.dataset)
    if args.research:
        if args.research == "inventory":
            research_inventory(protocol, args.data_root, args.workers)
        elif args.research == "geometry":
            research_geometry(protocol, args.data_root, args.workers)
        elif args.research == "select":
            research_select(protocol, args.data_root)
        else:
            research_summary(protocol)
        return
    if args.summarize:
        if args.geometry or args.inventory_only or args.new_only or args.far_limit is not None:
            parser.error("--summarize only reads existing full-pool and fixed-selection records")
        coverage = Path(protocol["content_coverage"]["output"]).parent
        # Fixed geometry evidence retains its original population after pool expansion.
        fixed = json.loads((coverage / "geometry/inserted/selection.json").read_text())["dataset"]
        report = summarize_existing(args.dataset or Path(fixed), coverage)
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
    selections = select_checks([w for w in worlds if not args.new_only or w["origin"] in ("supplement", "expanded")], args.far_limit)
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
                  inventory=totals,
                  cohorts={c: inventory_totals([w for w in worlds if w["cohort"] == c], root["splits"])
                           for c in sorted({w["cohort"] for w in worlds})},
                  selection=selections, checks=[])
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
