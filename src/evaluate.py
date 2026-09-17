"""Fixed V3 panels, shared empirical thresholds, and source-separated evaluation."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .data import DEFAULT_SAMPLES, FrozenDataset, detection_targets, read_real_frame
from .scene import STUSequence


VAL19 = (
    125,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    169,
)
SEED = 20260917
NORMAL_LABELS = {"pole": (80,), "vegetation": (70,), "trunk": (71,)}
WORKPOINTS = {"0.1pct": 0.001, "1pct": 0.01}
EXPOSURE_GROUPS = ("few_returns", "low_height", "far_range", "weak_disappearance")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def count_bin(count):
    return (
        "1-4"
        if count < 5
        else "5-19"
        if count < 20
        else "20-99"
        if count < 100
        else "100+"
    )


def distance_bin(distance):
    for lower, upper in ((2.5, 10), (10, 20), (20, 35), (35, 50)):
        if lower <= distance < upper or (upper == 50 and distance == 50):
            return f"{lower:g}-{upper:g}"
    raise ValueError("instance median must be in the detection range")


def instance_rows(source, target, metadata=None, records=False):
    """A row is one sequence/frame/instance observation, never an inferred cluster."""
    slots = source.real_slots if records else source.observation_slots
    ids = source.labels.instance[slots]
    observation_index = (
        np.searchsorted(slots, source.observation_slots)
        if records
        else np.arange(len(slots))
    )
    observed_ids = ids[observation_index]
    observed_target = target[observation_index]
    radius = np.linalg.norm(source.xyzi[source.observation_slots, :3], axis=1)
    rows = []
    for identity in np.unique(ids[target == 1]):
        if identity == 0:
            continue
        chosen = (target == 1) & (ids == identity)
        extra = (metadata or {}).get(str(int(identity)), {})
        height = extra.get("height_m")
        if height is not None and (
            not extra.get("height_source") or not np.isfinite(height) or height <= 0
        ):
            raise ValueError(
                "physical height requires a positive value and an independent source"
            )
        observed = (observed_target == 1) & (observed_ids == identity)
        count, returns = int(chosen.sum()), int(observed.sum())
        distance = float(np.median(radius[observed]))
        rows.append(
            dict(
                instance=int(identity),
                points=count,
                observed_returns=returns,
                distance_m=distance,
                returns=count_bin(returns),
                distance=distance_bin(distance),
                height_m=height,
                height=(
                    None
                    if height is None
                    else "<=0.30"
                    if height <= 0.30
                    else "0.30-0.50"
                    if height <= 0.50
                    else ">0.50"
                ),
                height_source=extra.get("height_source"),
                object_id=extra.get("object_id"),
            )
        )
    return rows


def _catalog_frame(request):
    root, sequence, frame, extra = request
    source = read_real_frame(root, sequence, frame)
    target = detection_targets(source, real_anomalies=True, records=True)
    raw = source.labels.semantic[source.real_slots]
    return dict(
        sequence=sequence,
        frame=frame,
        anomaly_points=int((target == 1).sum()),
        normal_points=int((target == 0).sum()),
        instances=instance_rows(source, target, extra.get("instances"), records=True),
        unknown_instance_points=int(
            ((target == 1) & (source.labels.instance[source.real_slots] == 0)).sum()
        ),
        normal_structures=[
            name
            for name, labels in NORMAL_LABELS.items()
            if np.any((target == 0) & np.isin(raw, labels))
        ],
    )


def stratified(frames, count, rng):
    if not frames:
        return []
    # Split the time axis, not the list of eligible observations.
    bins = np.linspace(min(frames), max(frames) + 1, count + 1)
    return [
        int(rng.choice(values))
        for a, b in zip(bins[:-1], bins[1:])
        if (values := [frame for frame in frames if a <= frame < b])
    ]


def select_hard(rows, excluded, rng, budget):
    chosen, sequences, attributes, objects = [], {}, {}, set()
    candidates = [
        row for row in rows if (row["sequence"], row["frame"]) not in excluded
    ]
    rng.shuffle(candidates)
    while candidates and len(chosen) < budget:

        def priority(row):
            tags = {"returns/" + i["returns"] for i in row["instances"]}
            tags |= {"distance/" + i["distance"] for i in row["instances"]}
            tags |= {"height/" + i["height"] for i in row["instances"] if i["height"]}
            tags |= set(row["normal_structures"])
            fresh = sum(1 / (1 + attributes.get(tag, 0)) for tag in tags)
            fresh += sum(
                i["object_id"] is not None and i["object_id"] not in objects
                for i in row["instances"]
            )
            adjacent = any(
                r["sequence"] == row["sequence"] and abs(r["frame"] - row["frame"]) < 5
                for r in chosen
            )
            return (not adjacent, -sequences.get(row["sequence"], 0), fresh)

        row = max(candidates, key=priority)
        candidates.remove(row)
        chosen.append(row)
        sequences[row["sequence"]] = sequences.get(row["sequence"], 0) + 1
        for instance in row["instances"]:
            for kind in ("returns", "distance", "height"):
                tag = f"{kind}/{instance[kind]}"
                attributes[tag] = attributes.get(tag, 0) + 1
            if instance["object_id"]:
                objects.add(instance["object_id"])
    return chosen


def prepare_panels(data_root, samples, output, annotations=None, workers=4, seed=SEED):
    """Select by GT before training, with optional independently sourced attributes.

    Annotation JSON uses observations["sequence/frame"].instances["packed_id"]
    with height_m, height_source, and optional cross-frame object_id. Optional
    normal_groups[name] contains original file slots and its annotation source.
    Missing physical heights and cross-frame identities remain unknown.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("panel already exists; reuse it for every candidate")
    annotation = json.loads(Path(annotations).read_text()) if annotations else {}
    observations = annotation.get("observations", {})
    requests = []
    for sequence in VAL19:
        directory = Path(data_root) / "val" / str(sequence) / "velodyne"
        files = sorted(directory.glob("*.bin"))
        if not files:
            raise FileNotFoundError(directory)
        for path in files:
            frame = int(path.stem)
            requests.append(
                (
                    str(data_root),
                    sequence,
                    frame,
                    observations.get(f"{sequence}/{frame}", {}),
                )
            )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_catalog_frame, requests, chunksize=16))
    rng = np.random.default_rng(seed)
    official = [row for row in rows if row["anomaly_points"] >= 5]
    micro = [row for row in rows if 1 <= row["anomaly_points"] <= 4]
    g, g_micro = [], []
    for sequence in VAL19:
        frames = [r["frame"] for r in official if r["sequence"] == sequence]
        picked = stratified(frames, 3, rng)
        g.extend([[sequence, f] for f in picked])
        if picked:
            g_micro.append([sequence, picked[len(picked) // 2]])
    h_micro_rows = select_hard(micro, set(), rng, 12)
    weak = [
        r
        for r in official
        if r["normal_structures"]
        or any(
            i["points"] < 20 or i["distance_m"] >= 35 or i["height"] == "<=0.30"
            for i in r["instances"]
        )
    ]
    hard = (
        select_hard(weak, set(map(tuple, g)), rng, 48 - len(h_micro_rows))
        + h_micro_rows
    )
    h = [[r["sequence"], r["frame"]] for r in hard]
    normal = STUSequence(data_root, 201)
    n_time = stratified(list(normal.frame_ids), 32, rng)
    normal_candidates = []
    for frame in normal.frame_ids:
        if frame in n_time:
            continue
        source = normal[frame]
        target = detection_targets(source)
        raw = source.labels.semantic[source.observation_slots]
        support = sum(
            int(((target == 0) & np.isin(raw, labels)).sum())
            for labels in NORMAL_LABELS.values()
        )
        if support:
            normal_candidates.append(frame)
    n_hard = stratified(normal_candidates, 16, rng)
    synthetic = FrozenDataset(samples, data_root, "validation")
    # Legacy counts nominate candidates only. Actual masks determine D and return counts.
    buckets = {"weak": [], "strong": [], "zero": []}
    for world_index, world in enumerate(synthetic.worlds):
        for row in world["frames"]:
            key = (
                "zero"
                if row.get("in_range", 0) == 0
                else "weak"
                if row.get("occluded", 0) < 5
                else "strong"
            )
            buckets[key].append(world_index * len(normal) + row["frame"])
    selected = []
    for values in buckets.values():
        rng.shuffle(values)
    for kind in ("weak", "strong", "zero") * 6:
        if len(selected) == 16:
            break
        while buckets[kind]:
            index = buckets[kind].pop()
            original, rendered = synthetic.pair(index)
            target = detection_targets(rendered.source, inserted=rendered.inserted_mask)
            d = disappeared(original, rendered)
            actual = (
                "zero" if not np.any(target == 1) else "weak" if d < 5 else "strong"
            )
            if actual != kind:
                continue
            selected.append(
                dict(
                    index=int(index),
                    world=rendered.world_identity,
                    frame=original.frame_id,
                    anomaly_points=int((target == 1).sum()),
                    disappeared=d,
                    denominator=None,
                    condition=kind,
                )
            )
            break
    gaps = dict(
        real_height_observations=sum(
            i["height_m"] is not None for r in rows for i in r["instances"]
        ),
        unknown_instance_points=sum(r["unknown_instance_points"] for r in rows),
        real_occlusion="unavailable: no paired real background",
        synthetic_occlusion_denominator="unavailable: affected-ray opportunities were not saved",
        G_requested=57,
        G_available=len(g),
        H_requested_max=48,
        H_available=len(h),
        N_temporal_requested=32,
        N_temporal_available=len(n_time),
        N_hard_requested=16,
        N_hard_available=len(n_hard),
        S_requested=16,
        S_available=len(selected),
        micro_frames_available=len(micro),
    )
    result = dict(
        format="ajae-v3-panels",
        seed=seed,
        data_root=str(Path(data_root).resolve()),
        annotations=annotation,
        catalog=rows,
        G=g,
        H=h,
        N_temporal=n_time,
        N_hard=n_hard,
        S=selected,
        micro=dict(
            G=g_micro,
            H=h[:: max(1, math.ceil(len(h) / 16))][:16],
            N_temporal=n_time[:: max(1, len(n_time) // 6)][:6],
            N_hard=n_hard[:2],
            S=selected[:4],
        ),
        gaps=gaps,
    )
    write_json(output, result)
    return result


def disappeared(original, rendered):
    slots = original.observation_slots
    return int(
        (
            (detection_targets(original) == 0) & rendered.occluded_original_mask[slots]
        ).sum()
    )


def operating_threshold(scores, labels, fpr=0.01):
    normal = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(normal) or not np.isfinite(normal).all():
        raise ValueError("a shared threshold requires finite normal scores")
    allowed = math.floor(fpr * len(normal))
    if not 0 <= allowed < len(normal):
        raise ValueError("FPR must lie in [0,1)")
    boundary = np.partition(normal, len(normal) - allowed - 1)[
        len(normal) - allowed - 1
    ]
    # Scores >= tau are positive. Exclude the complete boundary tie conservatively.
    return float(np.nextafter(np.float64(boundary), np.inf))


def recall_threshold(scores, labels, strict=False):
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    if not len(positive):
        return None
    required = (
        math.floor(0.95 * len(positive)) + 1
        if strict
        else math.ceil(0.95 * len(positive))
    )
    return float(
        np.partition(positive, len(positive) - required)[len(positive) - required]
    )


def rank_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (
        not len(scores)
        or not np.isfinite(scores).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("ranking requires finite scores and binary valid labels")
    positive, negative = int(labels.sum()), int((labels == 0).sum())
    if not positive or not negative:
        return {
            "AP": None,
            "AUROC": None,
            "FPR95": None,
            "R_at_0.1pct_FPR": None,
            "R_at_1pct_FPR": None,
        }
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1
    ]
    tp = np.cumsum(sorted_labels, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    recall, fpr = tp / positive, fp / negative
    ap = np.sum(np.diff(np.r_[0.0, recall]) * tp / (tp + fp))
    auc = np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fpr])
    # Official STU uses the first empirical point strictly above 95% recall.
    index95 = np.flatnonzero(recall > 0.95)[0]
    low = {
        name: operating_threshold(scores, labels, rate)
        for name, rate in WORKPOINTS.items()
    }
    normal = scores[labels == 0]
    return dict(
        AP=float(ap),
        AUROC=float(auc),
        FPR95=float(fpr[index95]),
        **{
            f"R_at_{name}_FPR": float(
                np.mean(scores[labels == 1].astype(np.float64) >= tau)
            )
            for name, tau in low.items()
        },
        thresholds=low,
        threshold_boundary_ties={
            name: int((normal == np.nextafter(tau, -np.inf)).sum())
            for name, tau in low.items()
        },
        tau_95=float(sorted_scores[ends[index95]]),
        achieved_recall95=float(recall[index95]),
        recall95_rule="first empirical recall strictly above 0.95, as in official STU",
        points=len(scores),
        anomaly_points=positive,
        normal_points=negative,
        units="fraction",
    )


@dataclass
class Prediction:
    sequence: int
    frame: int
    scores: np.ndarray
    target: np.ndarray
    instance: np.ndarray
    rows: list
    normal: dict


def predict(model, source, metadata=None, inserted=None, prepared=None):
    import torch
    from .model import prepare_scan

    if model.training:
        raise ValueError("evaluation must not update training BN statistics")
    if prepared is None:
        prepared = prepare_scan(source.xyzi[source.observation_slots])
    with torch.inference_mode():
        values = model([prepared])[0].cpu().numpy()
    # Restore known duplicate file records only after one computation per observation.
    scores = values[source.record_inverse]
    target = detection_targets(
        source,
        inserted=inserted,
        real_anomalies=source.partition == "val",
        records=True,
    )
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite model scores")
    slots, metadata = source.real_slots, metadata or {}
    valid = target >= 0
    normal = {
        name: (target[valid] == 0)
        & np.isin(source.labels.semantic[slots][valid], labels)
        for name, labels in NORMAL_LABELS.items()
    }
    for name, group in metadata.get("normal_groups", {}).items():
        if not group.get("source"):
            raise ValueError("normal structure annotations require a source")
        selected = np.asarray(group["slots"], dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= source.slot_count):
            raise ValueError("normal annotation slot outside the source scan")
        mask = np.isin(slots, selected)
        if np.any(mask & (target != 0)):
            raise ValueError("normal annotations must identify valid normal returns")
        normal[name] = mask[valid]
    return Prediction(
        source.sequence_id,
        source.frame_id,
        scores[valid],
        target[valid],
        source.labels.instance[slots][valid],
        instance_rows(source, target, metadata.get("instances"), records=True),
        normal,
    ), scores


def low_support(source):
    positions, inverse = np.unique(
        source.xyzi[source.observation_slots, :3], axis=0, return_inverse=True
    )
    distances, _ = cKDTree(positions).query(
        positions, k=9, distance_upper_bound=np.nextafter(2.0, np.inf), workers=1
    )
    return (np.isfinite(distances).sum(1)[inverse] - 1 < 8)[source.record_inverse]


def prediction_inputs(sources, workers):
    """Overlap bounded CPU geometry work; retain source order and GPU call order."""
    from .model import prepare_scan

    def prepare(source):
        scan = prepare_scan(source.xyzi[source.observation_slots])
        return source, scan, low_support(source)

    # Read sources in the consumer thread: STUSequence's LRU is not thread-safe.
    sources = iter(sources)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(prepare, s) for s in islice(sources, workers))
        while pending:
            result = pending.popleft().result()
            source = next(sources, None)
            if source is not None:
                pending.append(pool.submit(prepare, source))
            yield result


def summarize(predictions, threshold):
    groups, normal = {}, {}
    for prediction in predictions:
        positive = prediction.scores.astype(np.float64) >= threshold
        for row in prediction.rows:
            mask = (prediction.target == 1) & (prediction.instance == row["instance"])
            tags = [
                "all",
                "returns/" + row["returns"],
                "distance/" + row["distance"],
                "returns_distance/" + row["returns"] + "/" + row["distance"],
            ]
            if row["height"]:
                tags += [
                    "height/" + row["height"],
                    "height_distance/" + row["height"] + "/" + row["distance"],
                ]
            if "occlusion" in row:
                tags.append("disappeared/" + row["occlusion"])
            if (
                row.get("observed_returns", row["points"]) < 20
                or row["distance_m"] >= 35
                or row["height"] == "<=0.30"
            ):
                tags.append("target_union")
            for tag in tags:
                entry = groups.setdefault(
                    tag,
                    dict(
                        points=0,
                        observed_returns=0,
                        detected=0,
                        recalls=[],
                        sequences=set(),
                        objects=set(),
                        unknown_objects=0,
                    ),
                )
                detected = int(positive[mask].sum())
                entry["points"] += row["points"]
                entry["observed_returns"] += row.get("observed_returns", row["points"])
                entry["detected"] += detected
                entry["recalls"].append(detected / row["points"])
                entry["sequences"].add(prediction.sequence)
                if row["object_id"] is None:
                    entry["unknown_objects"] += 1
                else:
                    entry["objects"].add((prediction.sequence, row["object_id"]))
        for name, mask in {"all": prediction.target == 0, **prediction.normal}.items():
            entry = normal.setdefault(
                name, dict(points=0, false_positives=0, frames=[])
            )
            points, fp = int(mask.sum()), int((positive & mask).sum())
            entry["points"] += points
            entry["false_positives"] += fp
            entry["frames"].append(
                dict(
                    sequence=prediction.sequence,
                    frame=prediction.frame,
                    normal_points=points,
                    false_positives=fp,
                    fpr=fp / points if points else None,
                )
            )
    for entry in groups.values():
        entry["instance_observations"] = len(entry["recalls"])
        entry["instance_mean_point_recall"] = float(np.mean(entry.pop("recalls")))
        entry["point_recall"] = entry["detected"] / entry["points"]
        entry["sequences"] = len(entry["sequences"])
        entry["distinct_objects"] = (
            None if entry.pop("unknown_objects") else len(entry["objects"])
        )
        del entry["objects"]
    for entry in normal.values():
        entry["fpr"] = (
            entry["false_positives"] / entry["points"] if entry["points"] else None
        )
        frames = [r for r in entry["frames"] if r["normal_points"]]
        for field in ("false_positives", "fpr"):
            values = [r[field] for r in frames]
            entry[field + "_median"] = float(np.median(values)) if values else None
            entry[field + "_p95"] = float(np.quantile(values, 0.95)) if values else None
    total = sum(int((p.target == 1).sum()) for p in predictions)
    detected = sum(
        int(((p.target == 1) & (p.scores.astype(np.float64) >= threshold)).sum())
        for p in predictions
    )
    distributions = {}
    for label, name in ((0, "normal"), (1, "anomaly")):
        values = [p.scores[p.target == label] for p in predictions]
        count = sum(len(value) for value in values)
        if count:
            mean = sum(float(value.sum(dtype=np.float64)) for value in values) / count
            second = (
                sum(
                    float(np.einsum("i,i->", value, value, dtype=np.float64))
                    for value in values
                )
                / count
            )
            distributions[name] = dict(
                points=count,
                mean=mean,
                std=max(0.0, second - mean**2) ** 0.5,
                minimum=min(float(value.min()) for value in values if len(value)),
                maximum=max(float(value.max()) for value in values if len(value)),
            )
    fp = normal.get("all", {}).get("false_positives", 0)
    negative = normal.get("all", {}).get("points", 0)
    return dict(
        threshold=threshold,
        segmentation=dict(
            TP=detected,
            FP=fp,
            FN=total - detected,
            TN=negative - fp,
            Precision=detected / (detected + fp) if detected + fp else None,
            Recall=detected / total if total else None,
            IoU=detected / (total + fp) if total + fp else None,
        ),
        anomaly_points=total,
        detected_points=detected,
        point_recall=detected / total if total else None,
        unknown_instance_points=sum(
            int(((p.target == 1) & (p.instance == 0)).sum()) for p in predictions
        ),
        anomaly_groups=groups,
        normal_groups=normal,
        score_distribution=distributions,
        counting="metric points are restored file records; return-count and distance attributes use unique observations",
    )


def summarize_at(predictions, thresholds):
    return {name: summarize(predictions, tau) for name, tau in thresholds.items()}


def instance_predictions(predictions, thresholds):
    """Retain matched GT identities and counts for symmetric win/loss diagnostics."""
    records = []
    for prediction in predictions:
        for row in prediction.rows:
            mask = (prediction.target == 1) & (prediction.instance == row["instance"])
            values = prediction.scores[mask].astype(np.float64)
            if len(values) != row["points"]:
                raise ValueError("instance diagnostic and metric point identities differ")
            records.append(
                dict(
                    sequence=prediction.sequence,
                    frame=prediction.frame,
                    **row,
                    detected={
                        name: int((values >= tau).sum())
                        for name, tau in thresholds.items()
                    },
                )
            )
    return records


def coverage_status(exposure, plan, step):
    """Selection requires actual distinct observations AND the predeclared rounded node."""
    conditions, reasons = {}, []
    for name in EXPOSURE_GROUPS:
        group = exposure.get("conditions", {}).get(name, {})
        counts = {
            key: len(
                {tuple(v) if isinstance(v, list) else v for v in group.get(key, [])}
            )
            for key in ("requests", "sources", "objects")
        }
        node = plan.get("groups", {}).get(name, {}).get("node")
        eligible = (
            counts["requests"] >= 20
            and counts["sources"] >= 5
            and counts["objects"] >= 5
        )
        eligible = eligible and node is not None and step >= node
        conditions[name] = dict(
            **counts,
            node=node,
            eligible=eligible,
            unknown_observations=exposure.get("unknown", {}).get(name, 0),
        )
        if not eligible:
            reasons.append(
                f"{name}: actual support or predeclared exposure node not reached"
            )
    return dict(groups=conditions, eligible=not reasons, reasons=reasons)


def evaluation_kind(step, final=False, exposure_node=None):
    if final or step in (128, 256, 512, 1024) or step == exposure_node:
        return "full"
    if step in (0, 8):
        return "micro"
    if step in (16, 32, 64, 96) or step > 128 and step % 64 == 0:
        return "panel"
    return None


def compare_reports(paths, seed=SEED):
    """Keep both operating points and coverage limits visible; no weighted score."""
    from .model import METHOD

    reports = [json.loads(Path(path).read_text()) for path in paths]
    if not reports or any(r.get("kind") != "full" for r in reports):
        raise ValueError("candidate comparison requires complete real evaluations")
    if any(r.get("method") != METHOD for r in reports):
        raise ValueError(
            "candidate comparison requires the refined original-point method"
        )
    denominator = [
        (
            r["official_ranking"]["points"],
            r["official_ranking"]["anomaly_points"],
            r["normal201"]["1pct"]["normal_groups"]["all"]["points"],
            tuple(tuple(row) for row in r["official_population"]),
        )
        for r in reports
    ]
    if len(set(denominator)) != 1:
        raise ValueError("candidate point populations differ")
    groups = sorted(reports[0]["official"]["1pct"]["anomaly_groups"])
    if any(sorted(r["official"]["1pct"]["anomaly_groups"]) != groups for r in reports):
        raise ValueError(
            "candidate GT groups differ; reevaluate with common annotations"
        )
    target_groups = [
        key for key in groups if key.startswith(("returns/", "distance/", "height/"))
    ]
    normal_groups = sorted(reports[0]["official"]["1pct"]["normal_groups"])
    if any(
        sorted(r["official"]["1pct"]["normal_groups"]) != normal_groups for r in reports
    ):
        raise ValueError("candidate normal-region definitions differ")
    common_budget = len({r["candidate"]["update"] for r in reports}) == 1
    values, rows = [], []
    for path, report in zip(paths, reports):
        official = report["official_ranking"]
        vector = [official["AP"], -official["FPR95"]]
        intervals, recalls, false_positive = {}, {}, {}
        for point in WORKPOINTS:
            false_positive[point] = report["normal201"][point]["normal_groups"]["all"][
                "fpr"
            ]
            recalls[point] = {
                key: report["official"][point]["anomaly_groups"][key][
                    "instance_mean_point_recall"
                ]
                for key in target_groups
            }
            vector.extend(
                [
                    official[f"R_at_{point}_FPR"],
                    -false_positive[point],
                    *recalls[point].values(),
                ]
            )
            vector.extend(
                -report["official"][point]["normal_groups"][key]["fpr"]
                for key in normal_groups
                if report["official"][point]["normal_groups"][key]["points"]
            )
            intervals[point] = {}
            for key in target_groups:
                entries = [
                    report["sequences"][str(seq)][point]["anomaly_groups"].get(key)
                    for seq in VAL19
                ]
                counts = np.array(
                    [g["instance_observations"] if g else 0 for g in entries]
                )
                totals = np.array(
                    [
                        g["instance_mean_point_recall"] * g["instance_observations"]
                        if g
                        else 0
                        for g in entries
                    ]
                )
                indices = np.random.default_rng(seed).integers(
                    len(VAL19), size=(2000, len(VAL19))
                )
                support = counts[indices].sum(1)
                bootstrap = totals[indices].sum(1)[support > 0] / support[support > 0]
                intervals[point][key] = dict(
                    sequences=int((counts > 0).sum()),
                    instance_observations=int(counts.sum()),
                    interval95=np.quantile(bootstrap, [0.025, 0.975]).tolist()
                    if (counts > 0).sum() >= 2
                    else None,
                )
        values.append(vector)
        coverage = coverage_status(
            report.get("exposure", {}),
            report.get("coverage_plan", {}),
            report["candidate"]["update"],
        )
        reasons = list(coverage["reasons"])
        if not common_budget:
            reasons.append(
                "candidates have not reached a common complete-evaluation budget"
            )
        if report["candidate"]["route"] == "R" and report["candidate"]["update"] <= 128:
            reasons.append(
                "random initialization cannot be rejected by the 128-step adaptation comparison"
            )
        rows.append(
            dict(
                file=str(path),
                candidate=report["candidate"],
                AP=official["AP"],
                FPR95=official["FPR95"],
                **{
                    f"R_at_{point}_FPR": official[f"R_at_{point}_FPR"]
                    for point in WORKPOINTS
                },
                normal201_fpr=false_positive,
                instance_recalls=recalls,
                conditional_sequence_intervals=intervals,
                coverage=coverage,
                evidence_gaps=report.get("gaps", {}),
                selection_eligible=not reasons,
                retention_reasons=reasons,
            )
        )
    matrix = np.asarray(values)
    for index, row in enumerate(rows):
        observed = [
            j
            for j in range(len(rows))
            if j != index
            and np.all(matrix[j] >= matrix[index])
            and np.any(matrix[j] > matrix[index])
        ]
        row["observed_dominated_by"] = [str(paths[j]) for j in observed]
        row["dominated_by"] = [
            str(paths[j])
            for j in observed
            if row["selection_eligible"] and rows[j]["selection_eligible"]
        ]
    seed_summary = {}
    for row in rows:
        candidate = row["candidate"]
        key = f"{candidate['route']}/{candidate['mechanism']}/lambda={candidate['tail_weight']}/{candidate['update']}"
        seed_summary.setdefault(key, []).append(row)
    for key, selected in seed_summary.items():
        seeds = [r["candidate"]["seed"] for r in selected]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "a training seed must occur once per route/mechanism/risk/update"
            )
        seed_summary[key] = dict(
            seeds=seeds,
            at_least_three_seeds=len(seeds) >= 3,
            metrics={
                metric: dict(
                    mean=float(np.mean([r[metric] for r in selected])),
                    std=float(np.std([r[metric] for r in selected], ddof=1))
                    if len(seeds) > 1
                    else None,
                )
                for metric in ("AP", "FPR95", "R_at_0.1pct_FPR", "R_at_1pct_FPR")
            },
        )
    return dict(
        candidates=rows,
        seed_summary=seed_summary,
        compared_groups=target_groups,
        uncertainty_scope="sequence-cluster bootstrap conditional on fitted full-val19 thresholds; threshold-fitting uncertainty excluded; object repeats tracked only when annotated",
        selection_scope="observed full-real non-dominance at both workpoints; insufficient exposure retains candidates; missing real height/visibility annotations remain unevaluable; numerical dominance alone does not establish improvement beyond uncertainty",
    )


def evaluate(model, panels, data_root, samples, kind):
    import torch
    import psutil
    from .model import METHOD

    if kind not in {"micro", "panel", "full"}:
        raise ValueError("invalid evaluation scope")
    start, was_training = time.perf_counter(), model.training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    chosen = panels["micro"] if kind == "micro" else panels
    annotations = panels.get("annotations", {}).get("observations", {})
    keys = set(map(tuple, chosen["G"] + chosen["H"]))
    official_keys = {
        (r["sequence"], r["frame"])
        for r in panels["catalog"]
        if r["anomaly_points"] >= 5
    }
    tiny_keys = {
        (r["sequence"], r["frame"])
        for r in panels["catalog"]
        if 1 <= r["anomaly_points"] <= 4
    }
    if kind == "full":
        keys |= official_keys | tiny_keys
    real = {}
    # Each neighbor search uses four threads. Bound concurrent scans by live resources.
    workers = min(
        4,
        max(1, len(os.sched_getaffinity(0)) // 4),
        max(1, psutil.virtual_memory().available // 500_000_000),
    )
    try:
        sources = (read_real_frame(data_root, *key) for key in sorted(keys))
        for source, scan, support in prediction_inputs(sources, workers):
            sequence, frame = source.sequence_id, source.frame_id
            prediction, _ = predict(
                model, source, annotations.get(f"{sequence}/{frame}"), prepared=scan
            )
            valid = detection_targets(source, real_anomalies=True, records=True) >= 0
            prediction.normal["low_support"] = (prediction.target == 0) & support[valid]
            real[(sequence, frame)] = prediction
            if kind == "full" and (len(real) % 128 == 0 or len(real) == len(keys)):
                print(
                    json.dumps(
                        dict(evaluation="real", frames=len(real), total=len(keys))
                    ),
                    flush=True,
                )

        def combined(items):
            return np.concatenate([p.scores for p in items]), np.concatenate(
                [p.target for p in items]
            )

        g = [real[tuple(key)] for key in chosen["G"]]
        scores, labels = combined(g)
        thresholds = {
            name: operating_threshold(scores, labels, rate)
            for name, rate in WORKPOINTS.items()
        }
        high = recall_threshold(scores, labels)
        result = dict(
            method=METHOD,
            kind=kind,
            units="fraction",
            panel_thresholds=thresholds,
            panel_threshold_source="micro_G" if kind == "micro" else "G",
            G_ranking=rank_metrics(scores, labels),
            gaps=panels["gaps"],
            preparation_workers=workers,
        )
        for name in ("G", "H"):
            predictions = [real[tuple(key)] for key in chosen[name]]
            result[name] = summarize_at(predictions, thresholds)
            result[name + "_high_recall"] = (
                summarize(predictions, high) if high is not None else None
            )
        formal_thresholds, formal_high = thresholds, high
        if kind == "full":
            formal = [real[key] for key in sorted(official_keys)]
            scores, labels = combined(formal)
            result["official_ranking"] = rank_metrics(scores, labels)
            result["official_population"] = [
                [p.sequence, p.frame, len(p.target), int((p.target == 1).sum())]
                for p in formal
            ]
            formal_thresholds = result["official_ranking"]["thresholds"]
            formal_high = result["official_ranking"]["tau_95"]
            result["official"] = summarize_at(formal, formal_thresholds)
            result["official_instances"] = instance_predictions(formal, formal_thresholds)
            result["official_high_recall"] = summarize(formal, formal_high)
            result["tiny_supplement"] = summarize_at(
                [real[k] for k in sorted(tiny_keys)], formal_thresholds
            )
            # Sequence records are the cluster units for later seed/uncertainty analyses.
            result["sequences"] = {
                str(seq): summarize_at(
                    [p for p in formal if p.sequence == seq], formal_thresholds
                )
                for seq in VAL19
            }
        normal_source = STUSequence(data_root, 201)
        paired_frames = {request["frame"] for request in chosen["S"]}
        normal_frames = set(chosen["N_temporal"] + chosen["N_hard"]) | paired_frames
        if kind == "full":
            normal_frames.update(normal_source.frame_ids)
        normals, paired_originals = {}, {}
        sources = (normal_source[frame] for frame in sorted(normal_frames))
        for source, scan, support in prediction_inputs(sources, workers):
            frame = source.frame_id
            p, raw_scores = predict(
                model, source, annotations.get(f"201/{frame}"), prepared=scan
            )
            valid = detection_targets(source, records=True) >= 0
            p.normal["low_support"] = (p.target == 0) & support[valid]
            normals[frame] = p
            if frame in paired_frames:
                paired_originals[frame] = (p, raw_scores)
            if kind == "full" and (
                len(normals) % 128 == 0 or len(normals) == len(normal_frames)
            ):
                print(
                    json.dumps(
                        dict(
                            evaluation="normal201",
                            frames=len(normals),
                            total=len(normal_frames),
                        )
                    ),
                    flush=True,
                )
        for name in ("N_temporal", "N_hard"):
            selected = [normals[f] for f in chosen[name]]
            result[name] = summarize_at(selected, thresholds)
            result[name + "_high_recall"] = (
                summarize(selected, high) if high is not None else None
            )
        if kind == "full":
            result["normal201"] = summarize_at(
                list(normals.values()), formal_thresholds
            )
            result["normal201_high_recall"] = summarize(
                list(normals.values()), formal_high
            )
        # The normal-source predictions above can now be released before paired synthesis.
        del normals, real
        synthetic = FrozenDataset(samples, data_root, "validation")
        result["S"], synthetic_predictions = [], []
        for request in chosen["S"]:
            original, rendered = synthetic.pair(request["index"])
            if (
                rendered.world_identity != request["world"]
                or original.frame_id != request["frame"]
            ):
                raise ValueError("synthetic panel identity changed")
            world = synthetic.worlds[request["index"] // len(synthetic.sequence)]
            extra = {
                "instances": {
                    "60001": dict(
                        height_m=world["height_m"],
                        height_source=world["height_source"],
                        object_id=world["identity"],
                    )
                }
            }
            before, before_scores = paired_originals[original.frame_id]
            after, after_scores = predict(
                model, rendered.source, extra, rendered.inserted_mask
            )
            d = disappeared(original, rendered)
            if d != request["disappeared"]:
                raise ValueError("synthetic background disappearance changed")
            for row in after.rows:
                row["occlusion"] = (
                    "0" if d == 0 else "1-4" if d < 5 else "5-19" if d < 20 else "20+"
                )
            synthetic_predictions.append(after)
            # Retained raw file slots refer to the same return only when neither mask changes it.
            retained = ~rendered.inserted_mask & ~rendered.occluded_original_mask
            before_target = detection_targets(original, records=True)
            after_target = detection_targets(
                rendered.source, inserted=rendered.inserted_mask, records=True
            )
            before_mask = (before_target == 0) & retained[original.real_slots]
            after_mask = (after_target == 0) & retained[rendered.source.real_slots]
            if not np.array_equal(
                original.real_slots[before_mask], rendered.source.real_slots[after_mask]
            ):
                raise ValueError("retained normal-return identities do not match")
            changed = np.r_[
                original.xyzi[
                    original.observation_slots[
                        rendered.occluded_original_mask[original.observation_slots]
                    ],
                    :3,
                ],
                rendered.source.xyzi[
                    rendered.source.observation_slots[
                        rendered.inserted_mask[rendered.source.observation_slots]
                    ],
                    :3,
                ],
            ]
            affected = np.zeros(int(before_mask.sum()), bool)
            if len(changed):
                affected = (
                    cKDTree(changed).query(
                        original.xyzi[original.real_slots[before_mask], :3], k=1
                    )[0]
                    <= 2.0
                )

            def paired_at(shared):
                output = {}
                for point, threshold in shared.items():
                    paired = {}
                    for name, mask in (
                        ("retained_normal", np.ones(len(affected), bool)),
                        ("affected_normal", affected),
                    ):
                        n = int(mask.sum())
                        fp_before = int(
                            (
                                before_scores[before_mask][mask].astype(np.float64)
                                >= threshold
                            ).sum()
                        )
                        fp_after = int(
                            (
                                after_scores[after_mask][mask].astype(np.float64)
                                >= threshold
                            ).sum()
                        )
                        paired[name] = dict(
                            points=n,
                            before_false_positives=fp_before,
                            after_false_positives=fp_after,
                            before_fpr=fp_before / n if n else None,
                            after_fpr=fp_after / n if n else None,
                        )
                    output[point] = paired
                return output

            entry = dict(
                **request,
                threshold_source=result["panel_threshold_source"],
                anomaly=summarize_at([after], thresholds),
                original=summarize_at([before], thresholds),
                paired=paired_at(thresholds),
            )
            if kind == "full":
                entry["formal"] = dict(
                    threshold_source="full_val19",
                    anomaly=summarize_at([after], formal_thresholds),
                    original=summarize_at([before], formal_thresholds),
                    paired=paired_at(formal_thresholds),
                )
            result["S"].append(entry)
        result["synthetic_groups"] = summarize_at(synthetic_predictions, thresholds)
        if kind == "full":
            result["synthetic_groups_formal"] = summarize_at(
                synthetic_predictions, formal_thresholds
            )
        result["seconds"] = time.perf_counter() - start
        result["peak_cuda_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        return result
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "compare"))
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/jasongao/Data/STU")
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--panels", type=Path, default=Path("results/panels.json"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--kind", choices=("micro", "panel", "full"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reports", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_panels(
            args.data_root, args.samples, args.panels, args.annotations, args.workers
        )
        print(json.dumps(result["gaps"], ensure_ascii=False, indent=2))
    elif args.command == "compare":
        if args.reports is None or args.output is None:
            parser.error("compare requires --reports and --output")
        write_json(args.output, compare_reports(args.reports))
    else:
        import torch
        from .model import V3, METHOD, configure_runtime

        if args.checkpoint is None or args.output is None:
            parser.error("run requires --checkpoint and --output")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != METHOD:
            raise ValueError("checkpoint is not the refined original-point method")
        configure_runtime()
        model = V3(saved["mechanism"], saved["seed"]).cuda()
        model.load_state_dict(saved["model"], strict=True)
        panels = json.loads(args.panels.read_text())
        if saved["panels"] != panels:
            raise ValueError("evaluation must use the candidate's fixed panels")
        write_json(
            args.output,
            dict(
                candidate=dict(
                    route=saved["route"],
                    mechanism=saved["mechanism"],
                    seed=saved["seed"],
                    update=saved["step"],
                    tail_weight=saved["recipe"]["tail_weight"],
                ),
                exposure=saved["exposure"],
                coverage_plan=saved["coverage_plan"],
                **evaluate(model, panels, args.data_root, args.samples, args.kind),
            ),
        )


if __name__ == "__main__":
    main()
