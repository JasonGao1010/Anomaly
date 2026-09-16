"""Trace proposed or actually optimized queries using existing coverage evidence."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from .coverage import research_cells, research_records
from .data import source_identity
from .protocol import PROJECT_ROOT
from .train import Requests, TrainingFrames, auxiliary_fraction


PHYSICAL = ("near_sparse_1_4", "far_at_least_20", "weak_background_visible")
SPARSE = "normal_sparse_with_nonextreme_anomaly"
CELLS = (*PHYSICAL, SPARSE)


def failure_support(log_path, data_root):
    """Replay only capped positive queries; inspect actual near, dense and dark support."""
    log_path = Path(log_path)
    config = json.loads((log_path.parent / "config.json").read_text())
    if "group_queries" not in config["training"]:
        raise ValueError("failure support requires the V2 frame-weighted query definition")
    dataset = TrainingFrames(config, data_root)
    inventory = json.loads((PROJECT_ROOT / "results/coverage/inventory.json").read_text())
    worlds = {row["identity"]: row for row in inventory["worlds"]}
    names = ("near", "near_dense", "near_dark", "near_dark_dense", "near_bright_dense", "middle_dense")
    groups = {name: dict(draws=0, queries=0, actual_positive_mass=0., point_pooled_mass=0.,
        samples=set(), worlds=set(), parents=set(), source_frames=set(), cells=Counter(), world_mass=Counter(),
        intensity=[]) for name in names}
    counts = {key: dict(draws=0, queries=0, actual_positive_mass=0., point_pooled_mass=0.)
              for key in ("1-4", "5-19", "20-99", "100-499", "500+")}
    cache, capped, steps, queries_total, positive_mass, sparse_mass = {}, 0, 0, 0, 0., 0.
    started = time.perf_counter()
    for line in log_path.open():
        batch = json.loads(line)
        if batch["step"] != steps + 1:
            raise ValueError("training log is not a complete ordered update prefix")
        rows = batch["exposure"]
        positives, queries = sum(row["anomaly"] > 0 for row in rows), sum(row["anomaly"] for row in rows)
        steps += 1
        queries_total += queries
        positive_mass += .5 * bool(positives)
        for row in rows:
            sparse_mass += row["conditions"]["normal_condition_mass"]["sparse"]
            n = row["anomaly"]
            if not n:
                continue
            sample, draw = row["sample"], row["draw"]
            path, world, frame = dataset.dataset.samples[sample]
            if world != row["world"] or frame != row["frame"]:
                raise ValueError("logged query no longer addresses its original world/frame")
            if sample not in cache:
                with np.load(path, allow_pickle=False) as values:
                    if values["world_identity"].item() != world or values["source_identity"].item() != row["source_identity"]:
                        raise ValueError("stored positive returns differ from the logged source identity")
                    slots, inserted = values["source_slot"], values["inserted_slot"]
                    positions = np.searchsorted(slots, inserted)
                    if not np.array_equal(slots[positions], inserted):
                        raise ValueError("inserted slots are absent from the stored physical difference")
                    cache[sample] = values["xyzi"][positions]
            full = cache[sample]
            full_range = np.linalg.norm(full[:, :3], axis=1)
            dense = np.count_nonzero((full_range >= 2.5) & (full_range <= 50)) >= 100
            if len(full) > config["training"]["anomaly_queries"]:
                frozen, _, selected = dataset.queries(sample, draw)
                detection = selected["query"][selected["detection_index"]].numpy()
                mask = selected["target"].numpy() == 1
                xyzi = frozen.source.xyzi[frozen.source.real_slots[detection[mask]]]
                if int((~mask).sum()) != row["normal"] or len(selected["keep_slot"]) != row["keep"]:
                    raise ValueError("replayed query stream disagrees with logged normal or keep queries")
                capped += 1
            else:
                xyzi = full
            distance = np.linalg.norm(xyzi[:, :3], axis=1)
            near = (distance >= 2.5) & (distance < 10)
            middle = (distance >= 10) & (distance < 20)
            dark = xyzi[:, 3] < .05
            if (len(xyzi) != n or int(near.sum()) != row["anomaly_queries_near"]
                    or int(((distance >= 35) & (distance <= 50)).sum()) != row["anomaly_queries_far"]
                    or int(((distance >= 2.5) & (distance <= 50)).sum()) != row["anomaly_queries_in_range"]):
                raise ValueError("actual positive queries disagree with saved exposure counts")
            actual, pooled = .5 / positives, .5 * n / queries
            group = counts[list(counts)[int(np.searchsorted([5, 20, 100, 500], n, side="right"))]]
            group["draws"] += 1
            group["queries"] += n
            group["actual_positive_mass"] += actual
            group["point_pooled_mass"] += pooled
            masks = (near, near & dense, near & dark, near & dark & dense, near & ~dark & dense, middle & dense)
            for name, mask in zip(names, masks, strict=True):
                selected_n = int(mask.sum())
                if not selected_n:
                    continue
                group = groups[name]
                group["draws"] += 1
                group["queries"] += selected_n
                group["actual_positive_mass"] += actual * selected_n / n
                group["point_pooled_mass"] += pooled * selected_n / n
                group["samples"].add(sample)
                group["worlds"].add(world)
                group["parents"].add(row["parent"])
                group["source_frames"].add(frame)
                group["cells"][worlds[world]["assigned_cell"]] += selected_n
                group["world_mass"][world] += actual * selected_n / n
                group["intensity"].append(xyzi[mask, 3])
        if steps % 1024 == 0:
            print(f"支持核对 更新={steps} 异常查询={queries_total} 查询重放={capped} 用时={time.perf_counter()-started:.1f}s", flush=True)
    for group in groups.values():
        for name in ("samples", "worlds", "parents", "source_frames"):
            group[name] = len(group[name])
        group["maximum_world_mass_share"] = max(group["world_mass"].values(), default=0) / (group["actual_positive_mass"] or 1)
        del group["world_mass"]
        values = np.concatenate(group.pop("intensity")) if group["queries"] else np.empty(0)
        group["intensity_quantiles_10_50_90"] = np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None
    for group in (*groups.values(), *counts.values()):
        group["actual_positive_mass_fraction"] = group["actual_positive_mass"] / positive_mass
        group["point_pooled_mass_fraction"] = group["point_pooled_mass"] / positive_mass
    return dict(log=str(log_path), steps=steps, anomaly_queries=queries_total, groups=groups,
        frame_query_counts=counts, capped_query_replays=capped,
        normal_sparse_risk_fraction=sparse_mass / (steps * config["training"]["batch_frames"]),
        seconds=time.perf_counter()-started,
        definitions="near=[2.5,10)m; middle=[10,20)m; dense=at least100 actual inserted returns in[2.5,50]m; dark=raw intensity<0.05",
        limits="coarse observed-return support, not semantic equivalence or gradient influence; point-pooled masses are a mathematical counterfactual on identical draws, not an executed training control")


def _initialize(config, data_root, records, worlds, geometry):
    global _frames, _records, _worlds, _geometry, _config
    torch.set_num_threads(1)
    _frames = TrainingFrames(config, data_root)
    _records, _worlds, _geometry, _config = records, worlds, geometry, config


def _intersection(slots, reference):
    return int(np.isin(slots, reference).sum())


def _measure(request):
    index, draw, need = request
    frozen, original, queries = _frames.queries(index, draw)
    return measure_queries(request, frozen, original, queries, _config, _records, _worlds, _geometry)


def measure_queries(request, frozen, original, queries, config, records, worlds, geometry):
    """Observe the actual selected slots; never resample queries or change model input."""
    index, draw, need = request
    observed = records[frozen.world_identity, original.frame_id]
    identity = source_identity(original)
    if observed["source_identity"] != identity:
        raise ValueError("coverage and query source identities differ")
    world = worlds[frozen.world_identity]
    measured = geometry.get((frozen.world_identity, identity))
    _, physical = research_cells(observed, world, config["coverage"])
    flags = {name: physical[name] for name in PHYSICAL}
    normal_rules = config["coverage"]["normal_contrasts"]
    contrast = measured["contrasts"]["sparse"] if measured is not None else None
    flags[SPARSE] = None if contrast is None else (
        contrast["normal_queries"] >= normal_rules["minimum_normal_queries"]
        and contrast["central_anomaly_queries"] >= normal_rules["minimum_central_anomaly_queries"])
    detection = queries["query"][queries["detection_index"]].numpy()
    targets = queries["target"].numpy()
    slots = frozen.source.real_slots[detection]
    normal, anomaly, kept = slots[targets == 0], slots[targets == 1], queries["keep_slot"]
    # Cached lists are the two matched sides, not a saved set of pairwise edges.
    sides = None if contrast is None else dict(
        normal=_intersection(normal, contrast["normal_source_slots"]),
        anomaly=_intersection(anomaly, contrast["central_anomaly_slots"]),
        keep=_intersection(kept, contrast["normal_source_slots"]))
    indices = None if contrast is None else {
        kind: np.flatnonzero((targets == label) & np.isin(slots, contrast[field])).tolist()
        for kind, label, field in (("normal", 0, "normal_source_slots"),
                                   ("anomaly", 1, "central_anomaly_slots"))}
    changed = None if measured is None else dict(
        normal=_intersection(normal, measured["changed_normal_source_slots"]),
        keep=_intersection(kept, measured["changed_normal_source_slots"]))
    distance = np.linalg.norm(frozen.source.xyzi[anomaly, :3], axis=1)
    result = dict(sample=index, draw=draw, step=draw // config["training"]["batch_frames"],
        world=frozen.world_identity, name=world["world"], frame=original.frame_id,
        source_identity=identity, parent=world["parent"], regions=world["regions"],
        flags=flags, geometry_measured=measured is not None,
        normal=len(normal), anomaly=len(anomaly), keep=len(kept), keep_active=bool(need),
        near_keep=queries["near_pairs"], sparse_sides=sides, sparse_detection_index=indices,
        changed_normal=changed,
        anomaly_queries_in_range=int(((distance >= 2.5) & (distance <= 50)).sum()),
        anomaly_queries_near=int(((distance >= 2.5) & (distance < 10)).sum()),
        anomaly_queries_far=int(((distance >= 35) & (distance <= 50)).sum()),
        physical={name: observed[name] for name in
                  ("range", "in_range_rays", "anomaly_rays", "changed_native_rays")})
    if "condition_exposure" in queries:
        conditions = queries["condition_exposure"]
        result["conditions"] = {k: v for k, v in conditions.items() if not k.endswith("_mask")}
        union_slots = frozen.source.real_slots[queries["query"].numpy()]
        coefficients = {}
        for kind, identity_slots in (("normal", union_slots), ("keep", kept)):
            mass = np.zeros(len(identity_slots), np.float64)
            for group, weight in zip(queries[kind + "_groups"], queries[kind + "_weights"], strict=True):
                if len(group):
                    mass[group.numpy()] += weight / len(group)
            coefficients[kind] = mass
            result["conditions"][kind + "_condition_mass"] = {
                name: float(mass[conditions[kind + "_" + name + "_mask"]].sum()) for name in ("sparse", "changed")}
            sparse_mask = conditions[kind + "_sparse_mask"] & (mass > 0)
            result["conditions"][kind + "_sparse_slots"] = identity_slots[sparse_mask].tolist()
            # Distance and nominal angular sampling both grow with range; report the actual queried mass.
            distance = np.linalg.norm(frozen.source.xyzi[identity_slots, :3], axis=1)
            bands = np.searchsorted([2.5, 10., 35., 50.], distance, side="right")
            result["conditions"][kind + "_range_mass"] = [float(mass[bands == b].sum()) for b in range(5)]
        # Historical geometry groups retain their own meanings under the new point coefficients.
        coefficient_mass = {}
        for name in CELLS:
            if flags[name] is not True:
                continue
            if name == SPARSE:
                normal_mask = np.isin(union_slots, contrast["normal_source_slots"])
                keep_mask = np.isin(kept, contrast["normal_source_slots"])
                positive_fraction = sides["anomaly"] / len(anomaly) if len(anomaly) else 0.
            else:
                normal_mask, keep_mask = np.ones(len(union_slots), bool), np.ones(len(kept), bool)
                positive_fraction = float(bool(len(anomaly)))
            coefficient_mass[name] = dict(normal=float(coefficients["normal"][normal_mask].sum()),
                keep=float(coefficients["keep"][keep_mask].sum()), anomaly=positive_fraction)
        result["frame_risk_mass"] = coefficient_mass
    return result


def _share(counter):
    total = sum(counter.values())
    return max(counter.values()) / total if total else None


def summarize(rows, config, steps):
    """Class means and the actual warm-up schedule define coefficient exposure."""
    batch_size = config["training"]["batch_frames"]
    for first in range(0, len(rows), batch_size):
        batch = rows[first:first + batch_size]
        fraction = auxiliary_fraction(batch[0]["step"], config["training"])
        totals = {kind: sum(r[kind] for r in batch) for kind in ("normal", "anomaly")}
        totals["keep"] = sum(r["keep"] for r in batch if r["keep_active"])
        for row in batch:
            row["auxiliary_fraction"] = fraction
            row["coefficient_mass"] = {}
            for name in CELLS:
                if row["flags"][name] is not True:
                    continue
                if "frame_risk_mass" in row:
                    counts = row["frame_risk_mass"][name]
                    positives = sum(r["anomaly"] > 0 for r in batch)
                    mass = dict(normal=.5 * counts["normal"] / batch_size,
                        anomaly=.5 * counts["anomaly"] / max(positives, 1),
                        keep=counts["keep"] / batch_size if row["keep_active"] else 0.)
                else:
                    counts = row["sparse_sides"] if name == SPARSE else row
                    mass = {kind: .5 * counts[kind] / totals[kind] if totals[kind] else 0.
                            for kind in ("normal", "anomaly")}
                    mass["keep"] = (fraction * config["loss"]["keep_weight"] * counts["keep"]
                                    / totals["keep"] if row["keep_active"] and totals["keep"] else 0.)
                row["coefficient_mass"][name] = mass
    groups = {}
    for name in CELLS:
        selected = [r for r in rows if r["flags"][name] is True]
        counts = [r["sparse_sides"] if name == SPARSE else r for r in selected]
        mass = {kind: sum(r["coefficient_mass"][name][kind] for r in selected)
                for kind in ("normal", "anomaly", "keep")}
        concentrations = {}
        for kind in mass:
            parents, sources, regions = Counter(), Counter(), [Counter(), Counter()]
            for row in selected:
                value = row["coefficient_mass"][name][kind]
                parents[row["parent"]] += value
                sources[row["source_identity"]] += value
                for i in range(2):
                    regions[i][row["regions"][i]] += value
            concentrations[kind] = dict(maximum_parent_share=_share(parents),
                maximum_source_frame_share=_share(sources),
                maximum_world_support_region_share=[_share(r) for r in regions])
        groups[name] = dict(draws=len(selected), unique_world_frames=len({(r["world"], r["frame"]) for r in selected}),
            worlds=len({r["world"] for r in selected}), parents=len({r["parent"] for r in selected}),
            source_frames=len({r["source_identity"] for r in selected}),
            world_support_regions=[len({r["regions"][i] for r in selected}) for i in range(2)],
            queries={kind: sum(c[kind] for c in counts) for kind in mass},
            active_keep_queries=sum(c["keep"] for r, c in zip(selected, counts) if r["keep_active"]),
            matched_sides_both_queried=(sum(c["normal"] > 0 and c["anomaly"] > 0 for c in counts)
                                        if name == SPARSE else None),
            coefficient_mass=mass, mean_coefficient_mass_per_step={k: v / steps for k, v in mass.items()},
            concentration=concentrations)
    if "group_queries" in config["training"]:
        groups["v2_conditions"] = condition_summary(rows, steps, batch_size)
    return groups


def condition_summary(rows, steps, batch_size):
    """Actual group exposure, including overlap and repeated source slots; never gradient influence."""
    output = {}
    for kind, scale in (("normal", .5 / batch_size), ("keep", 1. / batch_size)):
        active = [r for r in rows if kind == "normal" or r["keep_active"]]
        sources, sparse_slots = Counter(), Counter()
        mass = Counter()
        ranges = np.zeros(5)
        for row in active:
            item = row["conditions"]
            sources[row["source_identity"]] += 1
            sparse_slots.update((row["source_identity"], s) for s in item[kind + "_sparse_slots"])
            mass.update({k: scale * v for k, v in item[kind + "_condition_mass"].items()})
            ranges += scale * np.asarray(item[kind + "_range_mass"])
        output[kind] = dict(group_query_seats=np.sum([r["conditions"][kind + "_group_queries"] for r in active], axis=0).tolist() if active else [0, 0, 0],
            condition_coefficient_mass=dict(mass), mean_condition_mass_per_update={k: v / steps for k, v in mass.items()},
            range_coefficient_mass=ranges.tolist(), source_frames=len(sources),
            maximum_source_frame_draw_share=_share(sources), unique_queried_sparse_slots=len(sparse_slots),
            sparse_slot_visits=sum(sparse_slots.values()), maximum_sparse_slot_visits=max(sparse_slots.values(), default=0))
    output["range_bins_m"] = ["below2.5", "2.5_to10", "10_to35", "35_to50", "at_least50"]
    output["scope"] = "actual source-slot exposure; conditions overlap; nominal coefficient mass is not gradient influence"
    return output


def coverage_context(config, wanted):
    """Read existing evidence once; retain only fields used by query exposure."""
    record, observations = research_records(PROJECT_ROOT / "results/coverage")
    records = {(r["world_identity"], r["frame"]): r for r in observations
               if (r["world_identity"], r["frame"]) in wanted and r["split"] == "train"}
    if set(records) != wanted:
        raise ValueError("sampled frames are missing their existing physical coverage records")
    worlds = {w["identity"]: w for w in record["worlds"] if w["split"] == "train"}
    cached = json.loads((PROJECT_ROOT / "results/coverage/geometry.json").read_text())
    geometry = {(r["world_identity"], r["source_identity"]): dict(
                    contrasts={"sparse": r["contrasts"]["sparse"]},
                    changed_normal_source_slots=r["changed_normal_source_slots"])
                for r in cached["observations"]
                if (r["world_identity"], r["frame"]) in wanted and r["split"] == "train"}
    return dict(config, coverage=record["parameters"]), records, worlds, geometry


def training_summary(path, config, steps):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            update = json.loads(line)
            if update["step"] <= steps:
                rows.extend(update["exposure"])
    if [r["draw"] for r in rows] != list(range(steps * config["training"]["batch_frames"])):
        raise ValueError("actual exposure log is not the complete optimized request prefix")
    return dict(steps=steps, draws=len(rows),
        unique_world_frames=len({(r["world"], r["frame"]) for r in rows}),
        worlds=len({r["world"] for r in rows}), source_frames=len({r["source_identity"] for r in rows}),
        geometry_known_draws=sum(r["geometry_measured"] for r in rows),
        geometry_unknown_draws=sum(not r["geometry_measured"] for r in rows),
        groups=summarize(rows, config, steps),
        scope="actual completed updates; groups overlap; unknown geometry is not absent hard content; coefficient mass is not gradient influence; support regions describe placement, not normal-point locations")


def preview(config, data_root, steps=1024, workers=8):
    if steps < 1 or workers < 1:
        raise ValueError("preview requires positive fixed step and worker budgets")
    started = time.perf_counter()
    frames = TrainingFrames(config, data_root)
    probabilities = frames.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
    requests = list(Requests(probabilities, config, steps))
    wanted = {(frames.dataset.samples[i][1], frames.dataset.samples[i][2]) for i, _, _ in requests}
    worker_config, records, worlds, geometry = coverage_context(config, wanted)
    del frames
    arguments = (worker_config, str(Path(data_root)), records, worlds, geometry)
    if workers == 1:
        _initialize(*arguments)
        rows = list(map(_measure, requests))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                 initializer=_initialize, initargs=arguments) as executor:
            rows = list(executor.map(_measure, requests, chunksize=4))
    groups = summarize(rows, config, steps)
    return dict(steps=steps, draws=len(rows), seed=config["training"]["seed"], workers=workers,
        seconds=time.perf_counter() - started, groups=groups, records=rows,
        geometry_known_draws=sum(r["geometry_measured"] for r in rows),
        geometry_unknown_draws=sum(not r["geometry_measured"] for r in rows),
        tail="query_candidates_only; no_scores_or_tail_membership_computed",
        scope="fixed_training_prefix; no_parameter_updates; cached_geometry_only; unknown_is_not_zero; "
              "cached_contrast_sides_do_not_store_pairwise_edges; coefficient_mass_is_not_gradient_influence; "
              "groups_overlap; support_regions_describe_world_placement_not_normal_point_locations")
