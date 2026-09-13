"""Trace frozen training queries without feature extraction, scoring or updates."""

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
    observed = _records[frozen.world_identity, original.frame_id]
    identity = source_identity(original)
    if observed["source_identity"] != identity:
        raise ValueError("coverage and query source identities differ")
    world = _worlds[frozen.world_identity]
    measured = _geometry.get((frozen.world_identity, identity))
    _, physical = research_cells(observed, world, _config["coverage"])
    flags = {name: physical[name] for name in PHYSICAL}
    normal_rules = _config["coverage"]["normal_contrasts"]
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
    return dict(sample=index, draw=draw, step=draw // _config["training"]["batch_frames"],
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
    return groups


def preview(config, data_root, steps=1024, workers=8):
    if steps < 1 or workers < 1:
        raise ValueError("preview requires positive fixed step and worker budgets")
    started = time.perf_counter()
    frames = TrainingFrames(config, data_root)
    probabilities = frames.dataset.sampling_probabilities(PROJECT_ROOT / config["training"]["sampling"])
    requests = list(Requests(probabilities, config, steps))
    record, observations = research_records(PROJECT_ROOT / "results/coverage")
    wanted = {(frames.dataset.samples[i][1], frames.dataset.samples[i][2]) for i, _, _ in requests}
    records = {(r["world_identity"], r["frame"]): r for r in observations
               if (r["world_identity"], r["frame"]) in wanted and r["split"] == "train"}
    if set(records) != wanted:
        raise ValueError("sampled frames are missing their existing physical coverage records")
    worlds = {w["identity"]: w for w in record["worlds"] if w["split"] == "train"}
    cached = json.loads((PROJECT_ROOT / "results/coverage/geometry.json").read_text())
    geometry = {(r["world_identity"], r["source_identity"]): r for r in cached["observations"]
                if (r["world_identity"], r["frame"]) in wanted and r["split"] == "train"}
    del cached, observations, frames
    worker_config = dict(config, coverage=record["parameters"])
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
