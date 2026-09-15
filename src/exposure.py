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
