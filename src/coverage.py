"""World-level content coverage from frozen observations and targeted full scans."""

from __future__ import annotations

import argparse
from collections import Counter
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
from .supervision import (
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
    dataset, config = _datasets[selection["split"]], _protocol["supervision"]
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
             for level in config["C2"]["levels"]]
    geometries = [ScanGeometry(source.xyzi[slots], slots, config["common"]["sampling_scale"], _threads)]
    labels = [y]
    for pair in pairs:
        rows = pair["dense_row"]
        geometries.append(ScanGeometry(geometries[0].xyzi[rows], pair["sparse_source_slot"],
                                       config["common"]["sampling_scale"], _threads))
        labels.append(y[rows])
        assert np.array_equal(geometries[-1].xyzi, source.xyzi[pair["sparse_source_slot"]])
    targets = [dict(**geometries[0].scale_arrays(),
                    **boundary_targets(geometries[0], y, config["C1"]["parameters"]))]
    evidence = local_evidence(geometries[0], y, config["C2"]["evidence_parameters"]["radius_m"])
    for g, sy, pair in zip(geometries[1:], labels[1:], pairs):
        targets.append(dict(**g.scale_arrays(), **sampling_targets(
            geometries[0], g, y, sy, pair, config["C2"]["evidence_parameters"], evidence)))
    surfaces = surface_targets(original, sample, geometries, labels, pairs, config["C3"]["parameters"])
    for target, surface in zip(targets, surfaces[config["C3"]["parameters"]["minimum_visible_support_points"]]):
        target.update(surface)
    auxiliary = targets[0]["boundary_valid"] | targets[0]["surface_valid"]
    for pair, target in zip(pairs, targets[1:]):
        auxiliary[pair["dense_row"][target["sampling_consistency_valid"]]] = True
    views = [summarize_view(g, sy, target, y if v else None)
             for v, (g, sy, target) in enumerate(zip(geometries, labels, targets))]
    edges = targets[0]["boundary_edges"]
    assert np.all(sample.anomaly_target[edges[:, 0]] == 0)
    assert np.all(sample.anomaly_target[edges[:, 1]] == 1)
    # Nearby normals are queried against the full inserted object observation.
    # Their C3 targets were computed using the complete original/current scans.
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
            probe = surface_probe(source.xyzi[slots[row], :3].astype(float), original, sample, slots, config["C3"]["parameters"])
            assert probe["valid"] and abs(probe["offset_z_m"] - float(t["surface_offset_z"][row])) < 1e-6
            probes.append(dict(kind=name, source_slot=int(slots[row]), label=int(y[row]),
                               semantic=int(source.labels.semantic[slots[row]]), **probe))
    return dict(sample=selection, observation=observed, views=views, context=contexts,
                base_anomaly_without_checked_auxiliary=int(np.sum((y == 1) & ~auxiliary)),
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
                         "base_anomaly_without_checked_auxiliary": lambda r: r["base_anomaly_without_checked_auxiliary"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/v1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--far-limit", type=int, help="check at most this many frames per world and far count stratum")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > len(os.sched_getaffinity(0)):
        parser.error("workers times threads must fit the available CPUs")
    protocol = json.loads(args.protocol.read_text())
    args.output = args.output or Path(protocol["content_coverage"]["output"])
    if args.dataset:
        protocol["dataset"]["directory"] = str(args.dataset)
    if args.far_limit is not None and args.far_limit < 1:
        parser.error("far limit must be positive")
    root, worlds, totals = inventory(Path(protocol["dataset"]["directory"]))
    selections = select_checks(worlds, args.far_limit)
    report = dict(scope="full_pool_inventory_and_targeted_scan_checks_not_full_pool_supervision_coverage",
                  dataset=protocol["dataset"]["directory"], pool_status=root["status"],
                  supervision_parameters=dict(seed=protocol["seed"],
                                              scale=protocol["supervision"]["common"]["sampling_scale"],
                                              boundary=protocol["supervision"]["C1"]["parameters"],
                                              levels=protocol["supervision"]["C2"]["levels"],
                                              sampling=protocol["supervision"]["C2"]["evidence_parameters"],
                                              surface=protocol["supervision"]["C3"]["parameters"]),
                  definitions=dict(far="median recorded anomaly range in [35,50] m",
                                   eligible="at least 5 anomaly returns with sensor range in [2.5,50] m",
                                   low="whole continuous object outer local z extent <= 0.2 m, not gravity height",
                                   few="5 to 19 distance-filtered anomaly returns",
                                   selection="per world: earliest maximum count; median eligible few-point frame; all far frames or equally spaced range ranks in separate 5-19 and >=20 point strata; with a limit also retain a weak far representative",
                                   far_checks_per_world_per_count_stratum=args.far_limit,
                                   nearby="normal return within 2 m Euclidean distance of an inserted return",
                                   raised_normal="known normal with valid C3 and offset in [-0.20,-0.05] m; descriptive proxy, no curb annotation or learned difficulty claim",
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
                                 maximum_worker_rss_bytes=max(r["peak_rss_bytes"] for r in records),
                                 disk_before=disk, disk_after=host_disk()))
    _atomic_json(args.output, report)


if __name__ == "__main__":
    main()
