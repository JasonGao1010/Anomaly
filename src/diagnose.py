"""Bounded, zero-update diagnosis of the fixed AJAE-NRE training trajectory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from .data import (
    FrozenWindowDataset,
    PredictionBatch,
    _stable_npz,
    normal_group_targets,
    observation_pool,
)
from .evaluate import (
    assert_unchanged,
    bits_score,
    diagnostic_bin,
    evaluation_targets,
    packed_scores,
    pooled_files,
    score_bits,
    exact_metrics,
)
from .model import AJAE, joint_voxelize
from .protocol import PROJECT_ROOT, load_protocol
from .scene import LabelMode, STUSequence
from .train import (
    FullResources,
    fixed_check,
    host_disk,
    nre_loss,
    random_state,
    restore_random_state,
    seed_all,
)


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "runs/diagnostics/nre/system"
EVALUATIONS = {
    "mid": ROOT / "runs/eval/nre/interim_14240",
    "late": ROOT / "runs/eval/nre/real",
}
VISITS = {"mid": 14240, "late": 28480}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def records(directory, row, name="evaluation_records"):
    record = row[name]
    dtype = np.uint64 if record["itemsize"] == 8 else np.float32
    with (directory / record["file"]).open("rb") as stream:
        stream.seek(record["offset"] * record["itemsize"])
        result = np.fromfile(stream, dtype=dtype, count=record["count"])
    if len(result) != record["count"]:
        raise ValueError("truncated score records")
    return result


def current_scores(directory, row, source):
    """Bind online logits to the actual source slots, independently of metric files."""
    with np.load(directory / row["prediction"]["file"], allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        frame = data["source_frame"]
        slot = (
            np.cumsum(data["source_slot_delta"], dtype=np.int32)
            if "source_slot_delta" in data
            else data["source_slot"]
        )
        scores = data["anomaly_score"]
    if (
        meta["synthetic_or_raw_sequence_id"] != row["sequence_id"]
        or meta["window_current_frame"] != source.frame_id
        or meta.get("score_kind") != "logit"
        or not np.isfinite(scores).all()
    ):
        raise ValueError("prediction source identity or numerical values differ")
    current = frame == source.frame_id
    np.testing.assert_array_equal(slot[current], source.real_slots)
    return scores[current]


class CurveObserver:
    """Retain positive-score precisions, sufficient for exact AP attribution."""

    def __init__(self):
        self.bits, self.precision, self.curves = [], [], []

    def __call__(self, bits, pos, tps, fps, positive, negative):
        use = pos > 0
        self.bits.append(bits[use])
        self.precision.append(tps[use] / (tps[use] + fps[use]))
        # All positive jumps suffice for PR; extra actual ROC nodes show its tail.
        grid = np.linspace(0, len(bits) - 1, min(256, len(bits))).astype(int)
        index = np.union1d(np.flatnonzero(use), grid)
        self.curves.append(
            np.column_stack(
                (
                    bits_score(bits[index], "logit"),
                    tps[index] / positive,
                    fps[index] / negative,
                    tps[index] / (tps[index] + fps[index]),
                )
            )
        )

    def save(self, output, model):
        bits, precision = np.concatenate(self.bits), np.concatenate(self.precision)
        order = np.argsort(bits)
        np.savez_compressed(
            output / f"precision_{model}.npz",
            bits=bits[order],
            precision=precision[order],
        )
        write_csv(
            output / f"curve_{model}.csv",
            (
                dict(zip(("threshold", "recall", "fpr", "precision"), row, strict=True))
                for row in np.concatenate(self.curves)
            ),
        )


def global_metrics(output):
    summary = {}
    for model, directory in EVALUATIONS.items():
        rows = [
            r
            for r in read_jsonl(directory / "results.jsonl")
            if r["current"]["eligible"]
        ]
        observer = CurveObserver()
        result = pooled_files(
            [directory / r["evaluation_records"]["file"] for r in rows],
            ranges=[
                (r["evaluation_records"]["offset"], r["evaluation_records"]["count"])
                for r in rows
            ],
            score_kind="logit",
            observe=observer,
        )
        expected = read_json(directory / "summary.json")["all_frames"]
        for key in ("AP", "AUROC", "FPR95"):
            if abs(result[key] - expected[key]) > 1e-9:
                raise ValueError(f"{model} {key} does not reproduce the saved result")
        assert (result["normal_count"], result["anomaly_count"], len(rows)) == (
            193792470,
            87499,
            1960,
        )
        point = result["recall_at_fpr_limit"]
        point.update(
            fn=result["anomaly_count"] - point["tp"],
            tn=result["normal_count"] - point["fp"],
        )
        observer.save(output, model)
        summary[model] = result
        emit("global_metrics", model=model, **result)
    write_json(output / "global.json", summary)
    return summary


def threshold_counts(scores, target, threshold):
    detected = scores >= threshold
    return dict(
        tp=int(np.sum(detected & (target == 1))),
        fn=int(np.sum(~detected & (target == 1))),
        fp=int(np.sum(detected & (target == 0))),
        tn=int(np.sum(~detected & (target == 0))),
    )


def attribute_sequence(task):
    data_root, sequence_id, late_rows, mid_rows, global_result, output = task
    sequence = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="val",
        sequence_id=sequence_id,
        label_mode=LabelMode.REQUIRED,
    )
    frame_profile = {
        (int(r["序列"]), int(r["帧号"])): r
        for r in read_csv(ROOT / "profiles/real/frames.csv")
    }
    windows = {
        (int(r["序列"]), int(r["当前帧"])): r
        for r in read_csv(ROOT / "profiles/real/windows.csv")
    }
    precision = {}
    for model in VISITS:
        with np.load(output / f"precision_{model}.npz") as a:
            precision[model] = (a["bits"], a["precision"])
    corrected = ROOT / "runs/train/nre/monitor_02_corrected/real"
    normal_mid = {
        r["current_frame"]: r
        for r in read_jsonl(corrected / "results.jsonl")
        if r["sequence_index"] == sequence_id and r["current"]["raw_anomaly_count"] == 0
    }
    results = []
    for row in late_rows:
        frame, counts = row["current_frame"], row["current"]
        base = dict(
            sequence=sequence_id,
            frame=frame,
            eligible=int(counts["eligible"]),
            normal_points=counts["normal_count"],
            anomaly_points=counts["anomaly_count"],
            raw_anomaly_points=counts["raw_anomaly_count"],
        )
        profile = frame_profile[sequence_id, frame]
        distance = profile["范围内异常距离P50（米）"]
        base["distance_median"] = float(distance) if distance else None
        base["stratum"] = diagnostic_bin(
            counts["anomaly_count"], base["distance_median"]
        )
        win = windows.get((sequence_id, frame))
        base["pattern"] = win["可见模式"] if win else "startup"
        base["history_scans"] = int(win["历史可见扫描数"]) if win else -1
        base["history_anomaly_points"] = int(win["历史异常点数"]) if win else None
        source = None
        paired = counts["eligible"] or frame in normal_mid
        # Existing normal-only records avoid decompressing 5,000 complete windows.
        if counts["eligible"] or 1 <= counts["anomaly_count"] <= 4 or paired:
            source = sequence.source_frame(frame)
            target = evaluation_targets(
                source.xyzi[source.real_slots, :3],
                source.labels.semantic[source.real_slots],
            )
            late = current_scores(EVALUATIONS["late"], row, source)
            values = {"late": (late, target)}
            if counts["eligible"]:
                old = mid_rows[frame]
                assert (old["check_seed"], old["frame_ids"]) == (
                    row["check_seed"],
                    row["frame_ids"],
                )
                mid = current_scores(EVALUATIONS["mid"], old, source)
                values["mid"] = (mid, target)
                for model, record in (("mid", old), ("late", row)):
                    np.testing.assert_array_equal(
                        records(EVALUATIONS[model], record),
                        packed_scores(
                            values[model][0][target >= 0],
                            target[target >= 0],
                            score_kind="logit",
                        ),
                    )
            elif frame in normal_mid:
                old = normal_mid[frame]
                assert old["check_seed"] == row["check_seed"]
                values["mid"] = (current_scores(corrected, old, source), target)
        elif counts["raw_anomaly_count"] == 0:
            late = records(EVALUATIONS["late"], row, "normal_records")
            values = {"late": (late, np.zeros(len(late), np.int8))}
        else:
            # Out-of-range-only anomaly frames remain in the frame census.
            results.append(base)
            continue
        base["paired_normal"] = int(frame in normal_mid)
        for model, (scores, target) in values.items():
            for label, field in (
                ("low", "recall_at_fpr_limit"),
                ("high", "official_high_recall"),
            ):
                c = threshold_counts(
                    scores, target, global_result[model][field]["threshold"]
                )
                base.update(
                    {f"{model}_{label}_{key}": value for key, value in c.items()}
                )
            base[f"{model}_normal_ge0"] = int(np.sum((target == 0) & (scores >= 0)))
            base[f"{model}_anomaly_ge0"] = int(np.sum((target == 1) & (scores >= 0)))
            if counts["eligible"]:
                bits, precisions = precision[model]
                query = score_bits(scores[target == 1], "logit")
                index = np.searchsorted(bits, query)
                np.testing.assert_array_equal(bits[index], query)
                base[f"{model}_ap_deficit"] = float(
                    np.sum(1 - precisions[index]) / 87499
                )
            if np.any(target == 1):
                base[f"{model}_anomaly_median"] = float(np.median(scores[target == 1]))
        results.append(base)
    emit("attributed_sequence", sequence=sequence_id, frames=len(results))
    return results


def aggregate_errors(frames, global_result):
    rows = []
    for grouping, field in (
        ("sequence", "sequence"),
        ("count_distance", "stratum"),
        ("visibility", "pattern"),
        ("history", "history_scans"),
    ):
        groups = defaultdict(list)
        for row in frames:
            if row["eligible"]:
                groups[str(row[field])].append(row)
        for group, selected in sorted(groups.items()):
            base = dict(
                grouping=grouping,
                group=group,
                sequences=len({r["sequence"] for r in selected}),
                frames=len(selected),
                anomaly_points=sum(r["anomaly_points"] for r in selected),
                normal_points=sum(r["normal_points"] for r in selected),
            )
            for model in VISITS:
                for metric in ("ap_deficit", "low_fn", "low_fp", "high_fn", "high_fp"):
                    base[f"{model}_{metric}"] = sum(
                        r[f"{model}_{metric}"] for r in selected
                    )
                    if metric == "ap_deficit":
                        base[f"{model}_deficit_pp"] = 100 * base[f"{model}_{metric}"]
                base[f"{model}_low_recall"] = (
                    1 - base[f"{model}_low_fn"] / base["anomaly_points"]
                )
            base["deficit_change_pp"] = base["late_deficit_pp"] - base["mid_deficit_pp"]
            rows.append(base)
        for model in VISITS:
            total = sum(
                r[f"{model}_ap_deficit"] for r in rows if r["grouping"] == grouping
            )
            assert abs(total - (1 - global_result[model]["AP"] / 100)) < 1e-10
    return rows


def run_scores(data_root, output, workers, *, reuse_metrics=False):
    begin = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    disk = host_disk()
    # 1.56 GB temporary sort + <2 GB probe scores + compact tables, never full features.
    if disk["SizeRemaining"] - disk["reserve_bytes"] < 5_000_000_000:
        raise OSError("insufficient host space for the bounded diagnosis")
    result = (
        read_json(output / "global.json") if reuse_metrics else global_metrics(output)
    )
    late = read_jsonl(EVALUATIONS["late"] / "results.jsonl")
    mid = read_jsonl(EVALUATIONS["mid"] / "results.jsonl")
    tasks = [
        (
            data_root,
            seq,
            [r for r in late if r["sequence_index"] == seq],
            {r["current_frame"]: r for r in mid if r["sequence_index"] == seq},
            result,
            output,
        )
        for seq in sorted({r["sequence_index"] for r in late})
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = [r for batch in executor.map(attribute_sequence, tasks) for r in batch]
    write_csv(output / "frames.csv", frames)
    errors = aggregate_errors(frames, result)
    write_csv(output / "errors.csv", errors)
    write_json(
        output / "score_check.json",
        dict(
            paired_eligible_frames=1960,
            original_slot_alignment=True,
            metric_records_equal_current_scores=True,
            deficit_sums_verified=True,
            paired_normal_frames=sum(r.get("paired_normal", 0) for r in frames),
            workers=workers,
            elapsed_seconds=time.monotonic() - begin,
            host_disk_before=disk,
            host_disk_after=host_disk(),
        ),
    )
    emit(
        "scores_complete",
        seconds=time.monotonic() - begin,
        worst=sorted(
            (r for r in errors if r["grouping"] == "sequence"),
            key=lambda r: -r["late_deficit_pp"],
        )[:4],
    )


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    result = dict(count=len(values), valid=len(valid), missing=len(values) - len(valid))
    result["mean"] = float(valid.mean()) if len(valid) else None
    result.update(
        dict(
            zip(
                ("p5", "p25", "p50", "p75", "p95"),
                np.quantile(valid, (0.05, 0.25, 0.5, 0.75, 0.95)).tolist()
                if len(valid)
                else [None] * 5,
                strict=True,
            )
        )
    )
    return result


def sparse_arrays(record):
    with np.load(record["file"], allow_pickle=False) as payload:
        arrays = {k: payload[k].copy() for k in payload.files if k != "metadata_json"}
        arrays["metadata"] = json.loads(str(payload["metadata_json"].item()))
    assert arrays["metadata"]["world_identity"] == record["world_identity"]
    return arrays


def surviving_nearest(tree, normal_slots, query, removed):
    """Reuse one raw-background tree, excluding only actually replaced returns."""
    if not len(query) or tree is None:
        return np.full(len(query), np.nan)
    result = np.full(len(query), np.nan)
    remaining = np.arange(len(query))
    k = min(8, len(normal_slots))
    while len(remaining):
        distance, index = tree.query(query[remaining], k=k, workers=1)
        distance, index = (
            distance.reshape(len(remaining), k),
            index.reshape(len(remaining), k),
        )
        distance[np.isin(normal_slots[index], removed)] = np.inf
        minimum = distance.min(1)
        good = np.isfinite(minimum)
        result[remaining[good]] = minimum[good]
        remaining = remaining[~good]
        if k == len(normal_slots):
            break
        k = min(2 * k, len(normal_slots))
    return result


def coverage_chunk(task):
    data_root, domain, source_id, arrays, frame_ids = task
    sequence = STUSequence.open(
        data_root,
        protocol=load_protocol(),
        partition="train",
        sequence_id=source_id,
        label_mode=LabelMode.REQUIRED,
    )
    rows, group_rows, union_rows = [], [], []
    for frame in frame_ids:
        raw = sequence.source_frame(int(frame))
        groups = normal_group_targets(raw.labels)
        groups[raw.zero_slot_mask] = -1
        base = np.bincount(groups[groups >= 0], minlength=20)
        normal_slots = np.flatnonzero(groups >= 0)
        needed = any(a["metadata"]["anomaly_return_counts"][frame] for a in arrays)
        tree = (
            cKDTree(raw.xyzi[normal_slots, :3])
            if needed and len(normal_slots)
            else None
        )
        common_removed = None
        for world, a in enumerate(arrays):
            start, stop = a["frame_offsets"][frame : frame + 2]
            slots = a["changed_slots"][start:stop]
            common_removed = (
                slots
                if common_removed is None
                else np.intersect1d(common_removed, slots, assume_unique=True)
            )
            xyzi = a["changed_xyzi"][start:stop]
            returned = np.any(xyzi[:, :3] != 0, axis=1)
            labels = a["changed_packed_labels"][start:stop] & 0xFFFF
            assert np.all(labels[returned] == 2) and np.all(labels[~returned] == 0)
            assert int(returned.sum()) == a["metadata"]["anomaly_return_counts"][frame]
            xyz = xyzi[returned, :3]
            distance = np.linalg.norm(xyz, axis=1)
            inside = (distance >= 2.5) & (distance <= 50)
            nearest = surviving_nearest(tree, normal_slots, xyz, slots)
            removed_groups = groups[slots]
            counts = base - np.bincount(
                removed_groups[removed_groups >= 0], minlength=20
            )
            group_rows.append((world, frame, counts))
            row = dict(
                domain=domain,
                world=world,
                source_sequence=source_id,
                frame=int(frame),
                world_identity=a["metadata"]["world_identity"],
                anomaly_points=int(inside.sum()),
                raw_anomaly_points=len(xyz),
                normal_points=int(counts.sum()),
                changed_slots=len(slots),
                occluded_slots=int((~returned).sum()),
                distance_median=float(np.median(distance[inside]))
                if inside.any()
                else None,
            )
            row["stratum"] = diagnostic_bin(
                row["anomaly_points"], row["distance_median"]
            )
            for name, values in (
                ("nearest_normal", nearest),
                ("nearest_normal_inside", nearest[inside]),
            ):
                row.update(
                    {f"{name}_{key}": value for key, value in describe(values).items()}
                )
                for edge in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                    row[f"{name}_le_{edge}"] = int(np.sum(values <= edge))
            rows.append(row)
        # A source point survives in the union unless every world replaces it.
        removed = groups[common_removed]
        union_rows.append(
            (frame, base - np.bincount(removed[removed >= 0], minlength=20))
        )
    return rows, group_rows, union_rows


def run_coverage(data_root, output, workers):
    started = time.monotonic()
    rows, normal_groups = [], []
    for domain in ("train", "validation"):
        pool, manifest = observation_pool(domain)
        arrays = [sparse_arrays(r) for r in manifest["segments"]]
        frame_count = len(arrays[0]["frame_ids"])
        tasks = [
            (data_root, domain, pool.source_sequence_id, arrays, chunk.tolist())
            for chunk in np.array_split(np.arange(frame_count), workers)
            if len(chunk)
        ]
        counts = np.zeros((len(arrays), frame_count, 20), np.int64)
        source_union = np.zeros((frame_count, 20), np.int64)
        domain_rows = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch, groups, unions in executor.map(coverage_chunk, tasks):
                domain_rows.extend(batch)
                for world, frame, c in groups:
                    counts[world, frame] = c
                for frame, c in unions:
                    source_union[frame] = c
        exposure = np.convolve(np.ones(frame_count - 4, np.int64), np.ones(5, np.int64))
        totals = (counts * exposure[None, :, None]).sum((0, 1))
        statistics = read_json(ROOT / "protocols/nre/labels.json")
        if domain == "train":
            np.testing.assert_array_equal(
                totals, statistics["normal_group_point_counts"]
            )
        for group in range(20):
            normal_groups.append(
                dict(
                    domain=domain,
                    group=group,
                    source_frames=int((source_union[:, group] > 0).sum()),
                    worlds=int(np.any(counts[:, :, group] > 0, axis=1).sum()),
                    unique_source_count=int(source_union[:, group].sum()),
                    world_frame_point_observations=int(counts[:, :, group].sum()),
                    full_window_point_exposures=int(totals[group]),
                    class_weight=(
                        statistics["class_weights"][
                            statistics["active_groups"].index(group)
                        ]
                        if group in statistics["active_groups"]
                        else None
                    ),
                )
            )
        lookup = {(r["world"], r["frame"]): r for r in domain_rows}
        for row in sorted(domain_rows, key=lambda r: (r["world"], r["frame"])):
            frame, world = row["frame"], row["world"]
            if frame < 4:
                continue
            history = [
                lookup[world, t]["raw_anomaly_points"]
                for t in range(frame - 4, frame + 1)
            ]
            row.update(
                pattern="".join(str(int(x > 0)) for x in history),
                history_anomaly_points=sum(history[:-1]),
                history_scans=sum(x > 0 for x in history[:-1]),
                window_anomaly_points=sum(history),
                dataset_index=world * (frame_count - 4) + frame - 4,
            )
            rows.append(row)
        expected = np.array(manifest["pattern_counts"])
        actual = np.bincount(
            [int(r["pattern"], 2) for r in rows if r["domain"] == domain], minlength=32
        )
        np.testing.assert_array_equal(actual, expected)
        emit(
            "coverage_pool",
            domain=domain,
            worlds=len(arrays),
            windows=len(arrays) * (frame_count - 4),
        )
    # Reuse the completed real profile without scanning real point clouds again.
    real_windows = {
        (int(r["序列"]), int(r["当前帧"])): r
        for r in read_csv(ROOT / "profiles/real/windows.csv")
    }
    real_frames = read_csv(ROOT / "profiles/real/frames.csv")
    nearby = {}
    for seq in sorted({int(r["序列"]) for r in real_frames}):
        with np.load(ROOT / f"runs/profiles/real/{seq}/anomaly.npz") as a:
            frame, inside, nearest = a["frame"], a["inside"], a["nearest_normal"]
            for t in np.unique(frame):
                use = frame == t
                row = {}
                for name, values in (
                    ("nearest_normal", nearest[use]),
                    ("nearest_normal_inside", nearest[use & inside]),
                ):
                    row.update(
                        {
                            f"{name}_{key}": value
                            for key, value in describe(values).items()
                        }
                    )
                    for edge in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                        row[f"{name}_le_{edge}"] = int(np.sum(values <= edge))
                nearby[seq, int(t)] = row
    for r in real_frames:
        seq, frame = int(r["序列"]), int(r["帧号"])
        if frame < 4:
            continue
        w = real_windows[seq, frame]
        distance = (
            float(r["范围内异常距离P50（米）"])
            if r["范围内异常距离P50（米）"]
            else None
        )
        count = int(r["范围内异常点"])
        rows.append(
            dict(
                domain="real",
                world=seq,
                source_sequence=seq,
                frame=frame,
                anomaly_points=count,
                raw_anomaly_points=int(r["全距离异常点"]),
                normal_points=int(r["正常点"]),
                distance_median=distance,
                stratum=diagnostic_bin(count, distance),
                pattern=w["可见模式"],
                history_scans=int(w["历史可见扫描数"]),
                history_anomaly_points=int(w["历史异常点数"]),
                window_anomaly_points=int(w["历史异常点数"]) + int(r["全距离异常点"]),
                **nearby.get((seq, frame), {}),
            )
        )
    write_csv(output / "coverage.csv", rows)
    write_csv(output / "normal_groups.csv", normal_groups)
    groups = []
    for domain in ("train", "validation", "real"):
        subset = [r for r in rows if r["domain"] == domain]
        for grouping, key in (
            ("visibility", "pattern"),
            ("count_distance", "stratum"),
            ("history", "history_scans"),
            ("joint", None),
        ):
            buckets = defaultdict(list)
            for r in subset:
                if grouping in ("count_distance", "joint") and r["anomaly_points"] < 5:
                    continue
                code = (
                    f"{r['stratum']}_{r['history_scans']}"
                    if key is None
                    else str(r[key])
                )
                buckets[code].append(r)
            codes = (
                [f"{x:05b}" for x in range(32)]
                if grouping == "visibility"
                else (
                    [f"{c}_{d}" for c in range(4) for d in range(4)]
                    if grouping == "count_distance"
                    else sorted(buckets)
                )
            )
            for code in codes:
                selected = buckets[code]
                groups.append(
                    dict(
                        domain=domain,
                        grouping=grouping,
                        group=code,
                        windows=len(selected),
                        worlds=len({r["world"] for r in selected}),
                        anomaly_points=sum(r["anomaly_points"] for r in selected),
                        raw_anomaly_points=sum(
                            r["raw_anomaly_points"] for r in selected
                        ),
                    )
                )
    write_csv(output / "coverage_groups.csv", groups)
    write_json(
        output / "coverage_check.json",
        dict(
            complete_formal_worlds=dict(train=32, validation=8),
            original_road_sources=dict(train=[206], validation=[201]),
            training_supervision_equals_frozen_statistics=True,
            pattern_counts_equal_manifests=True,
            real_profile_reused=True,
            nearest_normal_rule="Exact nearest surviving normal return in the same source scan; all visible distances; no new geometry grid",
            missing="No surviving normal return; no anomaly return; missing values remain empty, never imputed zero",
            workers=workers,
            seconds=time.monotonic() - started,
            host_disk=host_disk(),
        ),
    )
    emit("coverage_complete", seconds=time.monotonic() - started)


def spaced(rows, count):
    if len(rows) <= count:
        return rows
    return [rows[int(i)] for i in np.rint(np.linspace(0, len(rows) - 1, count))]


def select_probes(output):
    coverage = read_csv(output / "coverage.csv")
    frames = read_csv(output / "frames.csv")
    selected, gaps = {}, []

    def add(domain, world, row, reason, kind="basic"):
        frame = int(row["frame"])
        key = domain, world, frame
        if key in selected:
            selected[key]["reason"] += "; " + reason
            return
        selected[key] = dict(
            probe=f"{domain}_{world:03d}_{frame:06d}",
            domain=domain,
            world=world,
            frame=frame,
            kind=kind,
            reason=reason,
            sequence_id=(
                f"synthetic/v2/{domain}/{world:03d}"
                if domain in ("train", "validation")
                else f"val/{world}"
                if domain == "real"
                else "train/201"
            ),
            dataset_index=int(row["dataset_index"])
            if domain in ("train", "validation")
            else None,
            seed=23 if domain == "normal" else 23 + world,
            anomaly_points=int(row.get("anomaly_points", 0)),
            window_anomaly_points=int(row["window_anomaly_points"])
            if row.get("window_anomaly_points")
            else None,
            history_scans=int(row["history_scans"]) if row.get("history_scans") else 0,
            stratum=row.get("stratum"),
            gradient=False,
        )

    for domain, worlds in (("train", 32), ("validation", 8)):
        for world in range(worlds):
            rows = sorted(
                (
                    r
                    for r in coverage
                    if r["domain"] == domain and int(r["world"]) == world
                ),
                key=lambda r: int(r["frame"]),
            )
            normal = [r for r in rows if r["pattern"] == "00000"]
            anomaly = [r for r in rows if int(r["window_anomaly_points"]) > 0]
            eligible = [r for r in rows if int(r["anomaly_points"]) >= 5]
            choices = (
                [
                    (
                        normal[(len(normal) - 1) // 2],
                        "whole-window normal, earlier time median",
                    )
                ]
                if normal
                else []
            )
            if domain == "train" and anomaly:
                choices.append(
                    (
                        anomaly[(len(anomaly) - 1) // 2],
                        "anomaly-present window, earlier time median",
                    )
                )
            elif domain == "validation":
                choices.extend(
                    (r, "time-spread eligible window") for r in spaced(eligible, 3)
                )
            target = 2 if domain == "train" else 4
            if len(choices) < target:
                gaps.append(
                    dict(
                        domain=domain,
                        world=world,
                        normal_available=len(normal),
                        eligible_available=len(eligible),
                        anomaly_available=len(anomaly),
                    )
                )
                remaining = [r for r in rows if r not in [x[0] for x in choices]]
                choices.extend(
                    (r, "missing requested type; distinct same-world fallback")
                    for r in spaced(remaining, target - len(choices))
                )
            for r, reason in choices:
                add(domain, world, r, reason)
    for frame in np.rint(np.linspace(4, 681, 16)).astype(int):
        add("normal", 201, dict(frame=frame), "uniform full-window raw normal 201")
    for seq in sorted({int(r["sequence"]) for r in frames}):
        rows = [r for r in coverage if r["domain"] == "real" and int(r["world"]) == seq]
        eligible = [r for r in rows if int(r["anomaly_points"]) >= 5]
        normal = [r for r in rows if int(r["raw_anomaly_points"]) == 0]
        # Quarter and three-quarter times avoid concentrating base probes at sequence boundaries.
        for fraction in (0.25, 0.75):
            add(
                "real",
                seq,
                eligible[int(np.rint((len(eligible) - 1) * fraction))],
                "time-spread eligible quarter",
            )
        add(
            "real",
            seq,
            normal[(len(normal) - 1) // 2],
            "current no anomaly return, earlier time median",
        )
    lookup = {
        (int(r["world"]), int(r["frame"])): r for r in coverage if r["domain"] == "real"
    }
    for seq, times in (
        (125, [125, 130, 133, 137, 141, 145, 149, 153, 157, 161, 165, 170]),
        (142, [310, 319, 324, 330, 340, 350]),
    ):
        for frame in times:
            add(
                "real",
                seq,
                lookup[seq, frame],
                "observed continuous miss and flanks",
                "targeted",
            )
    stages = []
    for seq in sorted({int(r["sequence"]) for r in frames}):
        stage = []
        for r in [r for r in frames if int(r["sequence"]) == seq]:
            if int(r["raw_anomaly_points"]) == 0:
                stage.append(r)
            elif stage:
                stages.append(stage)
                stage = []
        if stage:
            stages.append(stage)
    stages.sort(key=lambda s: -sum(int(r["late_normal_ge0"]) for r in s))
    for stage in stages[:6]:
        r = max(stage, key=lambda r: int(r["late_normal_ge0"]))
        seq, frame = int(r["sequence"]), max(4, int(r["frame"]))
        add(
            "real",
            seq,
            lookup[seq, frame],
            f"normal stage {stage[0]['frame']}-{stage[-1]['frame']} with {sum(int(x['late_normal_ge0']) for x in stage)} FP at zero",
            "targeted",
        )
    probes = list(selected.values())
    train_a = sorted(
        (r for r in probes if r["domain"] == "train" and r["window_anomaly_points"]),
        key=lambda r: (r["history_scans"], r["window_anomaly_points"], r["world"]),
    )
    train_n = [
        r for r in probes if r["domain"] == "train" and r["window_anomaly_points"] == 0
    ]
    for r in spaced(train_a, 6) + spaced(train_n, 2):
        r["gradient"] = True
    assert len(probes) <= 193 and sum(r["gradient"] for r in probes) == 8
    write_json(
        output / "probes.json",
        dict(
            probes=probes,
            missing_types=gaps,
            rules="Selection fixed before internal forwards; basic and error-targeted sets reported separately; no input cropping",
            full_window_score_storage=True,
            feature_sample_per_window=dict(normal=256, anomaly=256, ignore=64),
        ),
    )
    write_csv(output / "probes.csv", probes)
    emit(
        "probes_fixed",
        count=len(probes),
        domains=dict(Counter(r["domain"] for r in probes)),
        missing_types=gaps,
    )
    return probes


class ProbeReader:
    def __init__(self, data_root):
        self.data_root = data_root
        self.domain = None
        self.dataset = None
        self.real = None

    def load(self, probe):
        domain = probe["domain"]
        if domain != self.domain:
            self.dataset = self.real = None
            if domain in ("train", "validation"):
                self.dataset = FrozenWindowDataset(
                    self.data_root,
                    load_protocol(),
                    pool_name=domain,
                    version="v2",
                    segment_cache_bytes=256 * 2**20,
                )
                # One domain's bounded raw scans are shared by every sparse world.
                self.dataset.source_sequence._cache_frames = (
                    449 if domain == "train" else 5
                )
            elif domain == "normal":
                self.real = STUSequence.open(
                    self.data_root,
                    protocol=load_protocol(),
                    partition="train",
                    sequence_id=201,
                    label_mode=LabelMode.REQUIRED,
                )
            self.domain = domain
        if domain in ("train", "validation"):
            window = self.dataset[probe["dataset_index"]]
        else:
            if domain == "real" and (
                self.real is None or self.real.spec.sequence_id != probe["world"]
            ):
                self.real = STUSequence.open(
                    self.data_root,
                    protocol=load_protocol(),
                    partition="val",
                    sequence_id=probe["world"],
                    label_mode=LabelMode.REQUIRED,
                )
            window = self.real.for_output(probe["frame"])
        assert (
            window.observation_sequence_id == probe["sequence_id"]
            and len(window.frame_ids) == 5
        )
        return window, joint_voxelize(window)


def feature_separation(features, labels):
    normal, anomaly = features[labels == 0], features[labels == 1]
    if not len(normal) or not len(anomaly):
        return dict(
            normal_samples=len(normal),
            anomaly_samples=len(anomaly),
            centroid_cosine=None,
            separation=None,
        )
    n, a = normal.mean(0), anomaly.mean(0)
    within = 0.5 * (
        np.square(normal - n).sum(1).mean() + np.square(anomaly - a).sum(1).mean()
    )
    return dict(
        normal_samples=len(normal),
        anomaly_samples=len(anomaly),
        centroid_cosine=float(
            np.dot(n, a) / max(np.linalg.norm(n) * np.linalg.norm(a), 1e-20)
        ),
        separation=float(np.linalg.norm(n - a) / max(np.sqrt(within), 1e-20)),
    )


class Trace:
    """Observe actual encoder parent mappings; hooks never modify model inputs."""

    def __init__(self, model, window, inputs):
        self.model, self.window, self.inputs = model, window, inputs
        self.target = torch.as_tensor(
            window.labels.anomaly_target.copy(), device="cuda"
        )
        self.classes = torch.where(self.target < 0, 2, self.target).long()
        self.current = torch.as_tensor(window.current_mask.copy(), device="cuda")
        self.inverse = inputs.point_to_voxel
        self.maps = {"voxel": self.inverse.cpu().numpy().astype(np.uint32)}
        ids = []
        for label, count in ((0, 256), (1, 256), (-1, 64)):
            candidates = np.flatnonzero(window.labels.anomaly_target == label)
            ids.extend(spaced(candidates.tolist(), count))
        self.ids = np.array(sorted(ids), np.int64)
        self.gpu_ids = torch.tensor(self.ids, device="cuda")
        self.features = dict(
            source_frame=window.points.source_frame[self.ids],
            source_slot=window.points.source_slot[self.ids],
            target=window.labels.anomaly_target[self.ids],
            current=window.current_mask[self.ids],
        )
        self.layers, self.handles, self.fusion = [], [], []
        self.offset = 0
        for name, stage in model.backbone.enc.named_children():
            self.handles.append(stage.register_forward_hook(self.encoder_hook(name)))
        self.handles.append(model.backbone.register_forward_hook(self.backbone_hook))
        self.handles.append(model.head.fusion.register_forward_hook(self.fusion_hook))

    def encoder_hook(self, name):
        def hook(module, args, point):
            if name != "enc0":
                self.inverse = point.pooling_inverse[self.inverse]
            self.record(name, point.feat, self.inverse)

        return hook

    def record(self, name, feature, inverse):
        count = len(feature)
        assert (
            len(inverse) == self.window.points.count
            and int(inverse.min()) >= 0
            and int(inverse.max()) < count
        )
        self.maps[name] = inverse.cpu().numpy().astype(np.uint32)
        members = torch.bincount(
            3 * inverse + self.classes, minlength=3 * count
        ).reshape(count, 3)
        a_units = members[:, 1] > 0
        current_a = self.current & (self.target == 1)
        all_a = self.target == 1
        row = dict(
            layer=name,
            feature_units=count,
            anomaly_units=int(a_units.sum()),
            current_anomaly_units=int(torch.unique(inverse[current_a]).numel()),
            mixed_anomaly_units=int((a_units & (members[:, 0] > 0)).sum()),
            pure_anomaly_units=int(
                (a_units & (members[:, 0] == 0) & (members[:, 2] == 0)).sum()
            ),
            all_points_mapped=True,
            features_finite=bool(torch.isfinite(feature).all()),
        )
        fraction = members.float() / members.sum(1).clamp_min(1)[:, None]
        for scope, mask in (("all_anomaly", all_a), ("current_anomaly", current_a)):
            for cls, label in enumerate(("normal", "anomaly", "ignore")):
                row[f"{scope}_{label}_member_fraction"] = (
                    float(fraction[inverse[mask], cls].mean()) if mask.any() else None
                )
        sampled = feature[inverse[self.gpu_ids]].float().cpu().numpy()
        self.features[name] = sampled
        row.update(feature_separation(sampled, self.features["target"]))
        self.layers.append(row)

    def backbone_hook(self, module, args, output):
        decoded, shallow = output
        inverse = self.inputs.point_to_voxel
        self.record("decoded", decoded.feat, inverse)
        pieces = (
            shallow[inverse[self.gpu_ids]],
            decoded.feat[inverse[self.gpu_ids]],
            self.inputs.point_features[self.gpu_ids],
        )
        self.features["fusion_input"] = torch.cat(pieces, 1).float().cpu().numpy()
        weights = self.model.head.fusion[0].weight
        for name, part, start, stop in zip(
            ("shallow", "deep", "detail"),
            pieces,
            (0, 36, 108),
            (36, 108, 117),
            strict=True,
        ):
            self.features[f"{name}_projection_norm"] = (
                F.linear(part, weights[:, start:stop]).norm(dim=1).float().cpu().numpy()
            )

    def fusion_hook(self, module, args, output):
        selected = self.ids[
            (self.ids >= self.offset) & (self.ids < self.offset + len(output))
        ]
        if len(selected):
            self.fusion.append(
                output[torch.tensor(selected - self.offset, device="cuda")]
                .float()
                .cpu()
                .numpy()
            )
        self.offset += len(output)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.features["fusion"] = np.concatenate(self.fusion)


def load_model(visit):
    payload = torch.load(
        ROOT / f"runs/train/nre/visit_{visit:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = AJAE(
        normal_groups=read_json(ROOT / "protocols/nre/labels.json")["active_groups"]
    )
    model.load_state_dict(payload["model"], strict=True)
    state = payload["state"]
    return model.cuda().eval(), state


def parameter_statistics(output):
    rows, similarities, finite = [], [], []
    for visit in (7120, 14240, 21360, 28480):
        payload = torch.load(
            ROOT / f"runs/train/nre/visit_{visit:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = payload["model"]
        bad = [
            name
            for name, value in state.items()
            if not bool(torch.isfinite(value).all())
        ]
        finite.append(
            dict(
                visit=visit,
                nonfinite=bad,
                tensors=len(state),
                batchnorm_tensors=sum(
                    name.endswith(("running_mean", "running_var")) for name in state
                ),
            )
        )
        if bad:
            raise FloatingPointError(f"nonfinite checkpoint tensors: {bad}")
        prototypes = F.normalize(state["head.prototypes"].float(), dim=2)
        cosine = (prototypes.flatten(0, 1) @ prototypes.flatten(0, 1).T).numpy()
        active = state["head.active_groups"].tolist()
        thresholds = (0.2 + 0.6 * state["head.acceptance"].sigmoid()).numpy()
        for i, group in enumerate(active):
            within = cosine[4 * i : 4 * i + 4, 4 * i : 4 * i + 4][np.triu_indices(4, 1)]
            between = np.delete(
                cosine[4 * i : 4 * i + 4], np.arange(4 * i, 4 * i + 4), axis=1
            )
            rows.append(
                dict(
                    visit=visit,
                    group=group,
                    threshold=float(thresholds[i]),
                    within_cosine_mean=float(within.mean()),
                    within_cosine_max=float(within.max()),
                    cross_cosine_max=float(between.max()),
                    cross_cosine_mean=float(between.mean()),
                )
            )
            for j, other in enumerate(active):
                similarities.append(
                    dict(
                        visit=visit,
                        group=group,
                        other_group=other,
                        cosine_mean=float(
                            cosine[4 * i : 4 * i + 4, 4 * j : 4 * j + 4].mean()
                        ),
                    )
                )
    write_csv(output / "parameters.csv", rows)
    write_csv(output / "prototype_similarity.csv", similarities)
    write_json(output / "parameter_check.json", finite)


def verify_sample(reader, probe, window, previous):
    """Check consumed returns, not merely metadata fields or tensor dimensions."""
    from .render import source_observation_identity

    row = dict(
        probe=probe["probe"],
        domain=probe["domain"],
        normal=int((window.labels.anomaly_target == 0).sum()),
        anomaly=int((window.labels.anomaly_target == 1).sum()),
        ignore=int((window.labels.anomaly_target < 0).sum()),
        unchanged_slots=0,
        proxy_returns=0,
        occluded_slots=0,
        overlap_frames=0,
    )
    if probe["domain"] not in ("train", "validation"):
        return row
    segment, _ = reader.dataset.segment_for_window(probe["dataset_index"])
    a = segment.arrays
    for item in window.frames:
        frame = item.source.frame_id
        raw = reader.dataset.source_sequence.source_frame(frame)
        start, stop = a["frame_offsets"][frame : frame + 2]
        slots = a["changed_slots"][start:stop]
        changed = np.zeros(raw.slot_count, bool)
        changed[slots] = True
        np.testing.assert_array_equal(item.source.xyzi[~changed], raw.xyzi[~changed])
        np.testing.assert_array_equal(
            item.source.labels.packed[~changed], raw.labels.packed[~changed]
        )
        returned = ~item.source.zero_slot_mask[slots]
        np.testing.assert_array_equal(
            item.source.xyzi[slots], a["changed_xyzi"][start:stop]
        )
        assert np.all(item.source.labels.anomaly_target[slots[returned]] == 1)
        assert np.all(item.source.labels.semantic_target[slots] == 255)
        assert not np.isin(slots[~returned], item.source.real_slots).any()
        raw_ignore = raw.real_slots[raw.labels.anomaly_target[raw.real_slots] < 0]
        unchanged_ignore = raw_ignore[~changed[raw_ignore]]
        assert np.isin(unchanged_ignore, item.source.real_slots).all()
        assert np.all(item.source.labels.anomaly_target[unchanged_ignore] < 0)
        key = probe["domain"], probe["world"], frame
        identity = source_observation_identity(item.source)
        if key in previous:
            assert previous[key] == identity
            row["overlap_frames"] += 1
        previous[key] = identity
        row["unchanged_slots"] += int((~changed).sum())
        row["proxy_returns"] += int(returned.sum())
        row["occluded_slots"] += int((~returned).sum())
    groups = normal_group_targets(window.labels)
    target = window.labels.anomaly_target
    assert np.array_equal(groups >= 0, target == 0)
    assert np.all(groups[(target == 0) & (window.labels.semantic_target == 255)] == 19)
    row["other_normal"] = int((groups == 19).sum())
    row["verified"] = True
    return row


def evidence_values(evidence, model):
    winning = evidence.normal_logits.argmax(1)
    groups = model.head.active_groups
    other = (groups == 19).nonzero().flatten()
    return dict(
        score=evidence.score.float().cpu().numpy(),
        base=evidence.support.float().cpu().numpy(),
        correction=evidence.correction.float().cpu().numpy(),
        max_support=evidence.normal_logits.max(1).values.float().cpu().numpy(),
        other_support=evidence.normal_logits[:, other[0]].float().cpu().numpy(),
        supported_group=groups[winning].cpu().numpy(),
        winning_prototype=evidence.prototype_choice.gather(1, winning[:, None])
        .squeeze(1)
        .cpu()
        .numpy(),
    )


def summarize_probe(
    probe, name, model, evidence, values, window, global_result, successful
):
    target = window.labels.anomaly_target
    current = window.current_mask
    official = np.full(len(target), -1, np.int8)
    official[current] = evaluation_targets(
        window.points.coordinates[current], window.labels.semantic[current]
    )
    rows, confusion, losses, usage = [], [], [], []
    for scope, mask in (
        ("all", np.ones(len(target), bool)),
        ("current", current),
        ("official_current", current & (official >= 0)),
    ):
        for label, cls in (("normal", 0), ("anomaly", 1)):
            selected = mask & (target == cls)
            row = dict(
                probe=probe["probe"],
                model=name,
                domain=probe["domain"],
                kind=probe["kind"],
                world=probe["world"],
                frame=probe["frame"],
                stratum=probe["stratum"],
                history_scans=probe["history_scans"],
                scope=scope,
                label=label,
                points=int(selected.sum()),
            )
            for key in ("score", "base", "correction", "max_support", "other_support"):
                row.update(
                    {
                        f"{key}_{k}": v
                        for k, v in describe(values[key][selected]).items()
                    }
                )
            count = int(selected.sum())
            if count:
                point_loss = np.logaddexp(
                    0, (1 - 2 * cls) * values["score"][selected].astype(np.float64)
                )
                hard = max(1, int(np.ceil(count * (0.01 if cls == 0 else 0.1))))
                row.update(
                    loss_mean=float(point_loss.mean()),
                    hard_loss_mean=float(
                        np.partition(point_loss, -hard)[-hard:].mean()
                    ),
                    correction_negative_bound=float(
                        np.mean(values["correction"][selected] <= -1.98)
                    ),
                    correction_positive_bound=float(
                        np.mean(values["correction"][selected] >= 1.98)
                    ),
                    other_supported_fraction=float(
                        np.mean(values["supported_group"][selected] == 19)
                    ),
                    score_ge0=int(np.sum(values["score"][selected] >= 0)),
                )
                for label_point, key in (
                    ("low", "recall_at_fpr_limit"),
                    ("high", "official_high_recall"),
                ):
                    threshold = global_result[name][key]["threshold"]
                    b, f = (
                        values["base"][selected] >= threshold,
                        values["score"][selected] >= threshold,
                    )
                    row.update(
                        {
                            f"{label_point}_detected": int(f.sum()),
                            f"{label_point}_base_to_final_lost": int(np.sum(b & ~f)),
                            f"{label_point}_base_to_final_gained": int(np.sum(~b & f)),
                        }
                    )
                counts = np.bincount(
                    4 * values["supported_group"][selected]
                    + values["winning_prototype"][selected],
                    minlength=80,
                ).reshape(20, 4)
                usage.extend(
                    dict(
                        probe=probe["probe"],
                        model=name,
                        scope=scope,
                        label=label,
                        group=g,
                        prototype=p,
                        points=int(counts[g, p]),
                    )
                    for g in range(20)
                    for p in range(4)
                    if counts[g, p]
                )
            rows.append(row)
        if window.labels.semantic_target is not None and scope in ("all", "current"):
            truth = normal_group_targets(window.labels)
            use = mask & (truth >= 0)
            cm = np.bincount(
                20 * truth[use] + values["supported_group"][use], minlength=400
            ).reshape(20, 20)
            confusion.extend(
                dict(
                    probe=probe["probe"],
                    model=name,
                    scope=scope,
                    truth=int(i),
                    prediction=int(j),
                    points=int(cm[i, j]),
                )
                for i, j in zip(*np.nonzero(cm), strict=True)
            )
    if probe["domain"] == "train":
        total, parts, _ = nre_loss(
            evidence,
            torch.tensor(target.copy(), device="cuda"),
            torch.tensor(normal_group_targets(window.labels), device="cuda"),
            model,
            read_json(ROOT / "protocols/nre/labels.json"),
            successful,
        )
        losses.append(
            dict(
                probe=probe["probe"],
                model=name,
                domain=probe["domain"],
                total=float(total),
                **{k: float(v) for k, v in parts.items()},
            )
        )
    return rows, confusion, losses, usage, official


def paired_changes(probe, window, values, thresholds):
    mid, late = values["mid"], values["late"]
    target, current = window.labels.anomaly_target, window.current_mask
    official = np.full(len(target), -1, np.int8)
    official[current] = evaluation_targets(
        window.points.coordinates[current], window.labels.semantic[current]
    )
    hit_mid = mid["score"] >= thresholds["mid"]["recall_at_fpr_limit"]["threshold"]
    hit_late = late["score"] >= thresholds["late"]["recall_at_fpr_limit"]["threshold"]
    partitions = dict(
        normal=target == 0,
        anomaly=target == 1,
        persistent_miss=(target == 1) & ~hit_mid & ~hit_late,
        detected_to_miss=(target == 1) & hit_mid & ~hit_late,
        miss_to_detected=(target == 1) & ~hit_mid & hit_late,
        persistent_detected=(target == 1) & hit_mid & hit_late,
    )
    rows = []
    # Subtraction in float64 separates the two score paths without extra rounding.
    delta = {
        k: late[k].astype(np.float64) - mid[k].astype(np.float64)
        for k in ("score", "base", "correction", "max_support")
    }
    error = np.max(np.abs(delta["score"] - delta["base"] - delta["correction"]))
    assert error < 5e-6
    for scope, use in (
        ("all", np.ones(len(target), bool)),
        ("official_current", official >= 0),
    ):
        for group, mask in partitions.items():
            mask = mask & use
            row = dict(
                probe=probe["probe"],
                domain=probe["domain"],
                kind=probe["kind"],
                world=probe["world"],
                frame=probe["frame"],
                scope=scope,
                group=group,
                points=int(mask.sum()),
                decomposition_max_error=float(error),
            )
            for key, values_array in delta.items():
                row.update(
                    {
                        f"delta_{key}_{k}": v
                        for k, v in describe(values_array[mask]).items()
                    }
                )
            rows.append(row)
    return rows


def run_numerics(data_root, output):
    torch.set_num_threads(1)
    probes = read_json(output / "probes.json")["probes"]
    reader = ProbeReader(data_root)
    selected = [
        next(p for p in probes if p["domain"] == domain)
        for domain in ("normal", "real")
    ]
    inputs = [(p, *reader.load(p)) for p in selected]
    checks = []
    for name, visit in VISITS.items():
        model, _ = load_model(visit)
        reference_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        for index, (probe, window, prepared) in enumerate(inputs):
            prepared = prepared.to("cuda")
            rope = [m for m in model.modules() if m.__class__.__name__ == "PointROPE"]
            for module in rope:
                module.cache.clear()
            reference = None
            for condition in (
                "cold",
                "amp_tables",
                "diagnostics",
                "chunk_32768",
                "after_alternating",
            ):
                if condition == "amp_tables":
                    for module in rope:
                        cached = list(module.cache.items())
                        with (
                            torch.no_grad(),
                            torch.autocast("cuda", dtype=torch.float16),
                        ):
                            for (dimension, device, dtype, _), tables in cached:
                                module.get_cos_sin(
                                    dimension, len(tables[0]), device, dtype
                                )
                if condition == "after_alternating":
                    other, w, p = inputs[1 - index]
                    with fixed_check(model, other["seed"]):
                        alternate = model(w, inputs=p.to("cuda")).float().cpu().numpy()
                    PredictionBatch.from_window(w, alternate, score_kind="logit").save(
                        output / "checks" / f"{name}_{probe['probe']}_alternate.npz",
                        window=w,
                    )
                model.point_chunk_size = 32768 if condition == "chunk_32768" else 65536
                with fixed_check(model, probe["seed"]):
                    trace = (
                        Trace(model, window, prepared)
                        if condition == "diagnostics"
                        else None
                    )
                    actual = model(window, inputs=prepared).float().cpu().numpy()
                    if trace is not None:
                        trace.close()
                PredictionBatch.from_window(window, actual, score_kind="logit").save(
                    output / "checks" / f"{name}_{probe['probe']}_{condition}.npz",
                    window=window,
                )
                if reference is None:
                    reference = actual
                difference = np.abs(actual - reference)
                np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
                checks.append(
                    dict(
                        model=name,
                        probe=probe["probe"],
                        condition=condition,
                        points=len(actual),
                        max_absolute_difference=float(difference.max()),
                        mean_absolute_difference=float(difference.mean()),
                        outside_tolerance=int(
                            np.sum(difference > 1e-6 + 1e-5 * np.abs(reference))
                        ),
                    )
                )
            assert_unchanged(model, reference_state)
            del prepared, reference, actual
        del model, reference_state
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(output / "numerics.csv", checks)
    emit(
        "numerics_complete",
        checks=len(checks),
        max_difference=max(r["max_absolute_difference"] for r in checks),
    )


def run_probes(data_root, output):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    resources = FullResources(emit=emit)
    before = resources()
    if (
        before["host_disk"]["SizeRemaining"] - before["host_disk"]["reserve_bytes"]
        < 4_000_000_000
    ):
        raise OSError("insufficient space for complete probe outputs")
    parameter_statistics(output)
    probes = read_json(output / "probes.json")["probes"]
    models, states, references = {}, {}, {}
    for name, visit in VISITS.items():
        models[name], states[name] = load_model(visit)
        references[name] = {
            k: v.cpu().clone() for k, v in models[name].state_dict().items()
        }
    global_result = read_json(output / "global.json")
    saved_predictions = {name: {} for name in VISITS}
    for name, directories in (
        (
            "mid",
            [
                EVALUATIONS["mid"],
                ROOT / "runs/train/nre/monitor_02_corrected/real",
                ROOT / "runs/train/nre/monitor_02_corrected/synthetic",
            ],
        ),
        ("late", [EVALUATIONS["late"], ROOT / "runs/eval/nre/synthetic"]),
    ):
        for directory in directories:
            for row in read_jsonl(directory / "results.jsonl"):
                saved_predictions[name][row["sequence_id"], row["current_frame"]] = (
                    directory,
                    row,
                )
    reader, previous = ProbeReader(data_root), {}
    result_path = output / "probe_results.jsonl"
    done = read_jsonl(result_path) if result_path.exists() else []
    completed = {r["probe"]["probe"] for r in done}
    all_started = time.monotonic()
    tail_ids = {name: Counter() for name in VISITS}
    for index, probe in enumerate(probes):
        if probe["probe"] in completed:
            continue
        started = time.monotonic()
        window, inputs = reader.load(probe)
        semantic = verify_sample(reader, probe, window, previous)
        inputs = inputs.to("cuda")
        values, results, maps = {}, {}, None
        for name, model in models.items():
            with fixed_check(model, probe["seed"]):
                trace = Trace(model, window, inputs)
                evidence = model(window, inputs=inputs, return_evidence=True)
                trace.close()
                values[name] = evidence_values(evidence, model)
                summaries, confusion, losses, usage, official = summarize_probe(
                    probe,
                    name,
                    model,
                    evidence,
                    values[name],
                    window,
                    global_result,
                    states[name]["successful_updates"],
                )
            saved = saved_predictions[name].get((probe["sequence_id"], probe["frame"]))
            difference = None
            reproduction = None
            if saved:
                directory, row = saved
                assert row["check_seed"] == probe["seed"]
                previous_batch = PredictionBatch.load(
                    directory / row["prediction"]["file"], window=window
                )
                difference = float(
                    np.max(np.abs(values[name]["score"] - previous_batch.anomaly_score))
                )
                error = np.abs(values[name]["score"] - previous_batch.anomaly_score)
                outside = error > 1e-6 + 1e-5 * np.abs(previous_batch.anomaly_score)
                reproduction = dict(
                    max_difference=float(error.max()),
                    mean_difference=float(error.mean()),
                    outside_original_tolerance=int(outside.sum()),
                    points=len(error),
                    current_max_difference=float(error[window.current_mask].max()),
                )
                for label, key in (
                    ("low", "recall_at_fpr_limit"),
                    ("high", "official_high_recall"),
                ):
                    threshold = global_result[name][key]["threshold"]
                    changed = (values[name]["score"] >= threshold) != (
                        previous_batch.anomaly_score >= threshold
                    )
                    for cls, group in ((0, "normal"), (1, "anomaly")):
                        reproduction[f"{label}_changed_current_{group}"] = int(
                            np.sum(changed & (official == cls))
                        )
                del previous_batch
            if maps is None:
                maps = trace.maps
                _stable_npz(
                    output / "mappings" / f"{probe['probe']}.npz",
                    maps,
                    compression_level=1,
                )
            else:
                for key in maps:
                    np.testing.assert_array_equal(maps[key], trace.maps[key])
            _stable_npz(
                output / "features" / name / f"{probe['probe']}.npz",
                trace.features,
                compression_level=1,
            )
            PredictionBatch.from_window(
                window, values[name]["score"], score_kind="logit"
            ).save(
                output / "predictions" / name / f"{probe['probe']}.npz", window=window
            )
            current = window.current_mask
            _stable_npz(
                output / "current" / name / f"{probe['probe']}.npz",
                dict(
                    source_slot=window.points.source_slot[current],
                    target=official[current],
                    **{k: v[current] for k, v in values[name].items()},
                ),
                compression_level=1,
            )
            if probe["domain"] == "train":
                normal = np.flatnonzero(window.labels.anomaly_target == 0)
                hard = max(1, int(np.ceil(len(normal) * 0.01)))
                ids = normal[
                    np.argpartition(values[name]["score"][normal], -hard)[-hard:]
                ]
                encoded = (
                    window.points.source_frame[ids].astype(np.int64) << 32
                ) | window.points.source_slot[ids]
                tail_ids[name].update(map(int, encoded))
                _stable_npz(
                    output / "tail" / name / f"{probe['probe']}.npz",
                    dict(source_ids=encoded),
                    compression_level=1,
                )
            results[name] = dict(
                saved_prediction_max_difference=difference,
                saved_prediction_reproduction=reproduction,
                statistics=summaries,
                confusion=confusion,
                losses=losses,
                usage=usage,
                layers=trace.layers,
                fusion_separation=feature_separation(
                    trace.features["fusion"], trace.features["target"]
                ),
            )
            del evidence, trace
        result = dict(
            probe=probe,
            semantics=semantic,
            models=results,
            paired=paired_changes(probe, window, values, global_result),
            seconds=time.monotonic() - started,
        )
        with result_path.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        del values, results, maps, inputs, window, result
        if (index + 1) % 8 == 0 or index + 1 == len(probes):
            emit(
                "probe_progress",
                completed=index + 1,
                total=len(probes),
                seconds=time.monotonic() - all_started,
            )
            resources()
    for name, model in models.items():
        assert_unchanged(model, references[name])
    write_json(
        output / "probe_check.json",
        dict(
            before=before,
            after=resources(),
            windows=len(probes),
            regular_forwards=2 * len(probes),
            parameters_buffers_unchanged=True,
            mapping_equal_between_models=True,
            zero_optimizer_updates=True,
            raw_source_cache_bound_frames=dict(train=449, validation=5),
            seconds=time.monotonic() - all_started,
        ),
    )
    summarize_probes(output)


def summarize_probes(output):
    results = read_jsonl(output / "probe_results.jsonl")
    for field, filename in (
        ("statistics", "scores.csv"),
        ("confusion", "confusion.csv"),
        ("losses", "losses.csv"),
        ("usage", "usage.csv"),
        ("layers", "layers.csv"),
    ):
        rows = []
        for result in results:
            for name in VISITS:
                rows.extend(
                    dict(probe=result["probe"]["probe"], model=name, **row)
                    if "probe" not in row
                    else row
                    for row in result["models"][name][field]
                )
        write_csv(output / filename, rows)
    write_csv(
        output / "paired.csv", (r for result in results for r in result["paired"])
    )
    write_csv(output / "semantics.csv", (result["semantics"] for result in results))
    pooled = []
    # Only diagnostic current eligible frames enter these pooled probe curves.
    for domain, kind in sorted(
        {(r["probe"]["domain"], r["probe"]["kind"]) for r in results}
    ):
        selected = [
            r["probe"]
            for r in results
            if (r["probe"]["domain"], r["probe"]["kind"]) == (domain, kind)
        ]
        for name in VISITS:
            arrays = {"base": [], "score": []}
            for probe in selected:
                with np.load(
                    output / "current" / name / f"{probe['probe']}.npz"
                ) as data:
                    target = data["target"]
                    if np.sum(target == 1) < 5:
                        continue
                    for path in arrays:
                        arrays[path].append(
                            packed_scores(
                                data[path][target >= 0],
                                target[target >= 0],
                                score_kind="logit",
                            )
                        )
            for path, blocks in arrays.items():
                ordered = np.concatenate(blocks) if blocks else np.empty(0, np.uint64)
                ordered.sort()
                metrics = exact_metrics(ordered, score_kind="logit")
                point = metrics["recall_at_fpr_limit"]
                pooled.append(
                    dict(
                        domain=domain,
                        kind=kind,
                        model=name,
                        path=path,
                        eligible_frames=len(blocks),
                        AP=metrics["AP"],
                        AUROC=metrics["AUROC"],
                        FPR95=metrics["FPR95"],
                        anomaly_points=metrics["anomaly_count"],
                        normal_points=metrics["normal_count"],
                        recall_at_one_percent=point["recall"] if point else None,
                    )
                )
    write_csv(output / "probe_metrics.csv", pooled)
    for name in VISITS:
        count = Counter()
        for path in (output / "tail" / name).glob("*.npz"):
            with np.load(path) as data:
                count.update(map(int, data["source_ids"]))
        write_csv(
            output / f"tail_repeat_{name}.csv",
            (
                dict(
                    occurrences=k,
                    distinct_original_normal_points=v,
                    point_exposures=k * v,
                )
                for k, v in sorted(Counter(count.values()).items())
            ),
        )
    emit("probe_summary_complete", windows=len(results))


def probe_details(data_root, output):
    from .render import source_observation_identity

    results = read_jsonl(output / "probe_results.jsonl")
    thresholds = read_json(output / "global.json")
    reproductions, fp_groups, fusion, condition_metrics = [], [], [], []
    old = {name: {} for name in VISITS}
    for name, directories in (
        (
            "mid",
            [
                EVALUATIONS["mid"],
                ROOT / "runs/train/nre/monitor_02_corrected/real",
                ROOT / "runs/train/nre/monitor_02_corrected/synthetic",
            ],
        ),
        ("late", [EVALUATIONS["late"], ROOT / "runs/eval/nre/synthetic"]),
    ):
        for directory in directories:
            for row in read_jsonl(directory / "results.jsonl"):
                old[name][row["sequence_id"], row["current_frame"]] = directory, row
    for result in results:
        probe = result["probe"]
        for name in VISITS:
            reproduction = result["models"][name].get("saved_prediction_reproduction")
            item = dict(
                probe=probe["probe"],
                domain=probe["domain"],
                model=name,
                **(reproduction or {}),
            )
            with np.load(output / "current" / name / f"{probe['probe']}.npz") as data:
                target, scores = data["target"], data["score"]
                saved = old[name].get((probe["sequence_id"], probe["frame"]))
                if saved and np.sum(target == 1) >= 5:
                    directory, row = saved
                    packed = records(directory, row)
                    np.testing.assert_array_equal(packed & 1, target[target >= 0])
                    previous = bits_score((packed >> 1).astype(np.uint32), "logit")
                    metrics = exact_metrics(
                        np.sort(
                            packed_scores(
                                scores[target >= 0],
                                target[target >= 0],
                                score_kind="logit",
                            )
                        ),
                        score_kind="logit",
                    )
                    item.update(
                        old_AP=row["current"]["AP"],
                        new_AP=metrics["AP"],
                        AP_change_pp=metrics["AP"] - row["current"]["AP"],
                    )
                    item["official_point_max_difference"] = float(
                        np.max(np.abs(previous - scores[target >= 0]))
                    )
                for point, threshold in (
                    ("zero", 0.0),
                    ("low", thresholds[name]["recall_at_fpr_limit"]["threshold"]),
                    ("high", thresholds[name]["official_high_recall"]["threshold"]),
                ):
                    selected = (target == 0) & (scores >= threshold)
                    counts = np.bincount(
                        data["supported_group"][selected], minlength=20
                    )
                    for group, count in enumerate(counts):
                        if count:
                            fp_groups.append(
                                dict(
                                    probe=probe["probe"],
                                    model=name,
                                    domain=probe["domain"],
                                    kind=probe["kind"],
                                    world=probe["world"],
                                    point=point,
                                    supported_group=group,
                                    fp=int(count),
                                    normal_points=int(np.sum(target == 0)),
                                )
                            )
            if reproduction is not None:
                reproductions.append(item)
            with np.load(output / "features" / name / f"{probe['probe']}.npz") as data:
                for label, cls in (("normal", 0), ("anomaly", 1)):
                    use = data["target"] == cls
                    row = dict(
                        probe=probe["probe"],
                        model=name,
                        domain=probe["domain"],
                        label=label,
                        samples=int(use.sum()),
                    )
                    for path in ("shallow", "deep", "detail"):
                        row.update(
                            {
                                f"{path}_{k}": v
                                for k, v in describe(
                                    data[f"{path}_projection_norm"][use]
                                ).items()
                            }
                        )
                    fusion.append(row)
    write_csv(output / "reproduction.csv", reproductions)
    write_csv(output / "false_positive_groups.csv", fp_groups)
    write_csv(output / "fusion.csv", fusion)
    for domain in ("train", "validation", "real"):
        probes = [
            r["probe"]
            for r in results
            if r["probe"]["domain"] == domain and r["probe"]["kind"] == "basic"
        ]
        for grouping in ("stratum", "world"):
            for group in sorted({str(p[grouping]) for p in probes}):
                selected = [p for p in probes if str(p[grouping]) == group]
                for name in VISITS:
                    arrays = []
                    for probe in selected:
                        with np.load(
                            output / "current" / name / f"{probe['probe']}.npz"
                        ) as data:
                            target = data["target"]
                            if np.sum(target == 1) >= 5:
                                arrays.append(
                                    packed_scores(
                                        data["score"][target >= 0],
                                        target[target >= 0],
                                        score_kind="logit",
                                    )
                                )
                    scores = (
                        np.concatenate(arrays) if arrays else np.empty(0, np.uint64)
                    )
                    scores.sort()
                    metrics = exact_metrics(scores, score_kind="logit")
                    point = metrics["recall_at_fpr_limit"]
                    condition_metrics.append(
                        dict(
                            domain=domain,
                            model=name,
                            grouping=grouping,
                            group=group,
                            selected_windows=len(selected),
                            eligible_windows=len(arrays),
                            worlds=len({p["world"] for p in selected}),
                            normal_points=metrics["normal_count"],
                            anomaly_points=metrics["anomaly_count"],
                            AP=metrics["AP"],
                            AUROC=metrics["AUROC"],
                            FPR95=metrics["FPR95"],
                            recall_at_one_percent=point["recall"] if point else None,
                        )
                    )
    write_csv(output / "condition_metrics.csv", condition_metrics)
    reader = ProbeReader(data_root)
    overlaps = []
    for domain in ("train", "validation"):
        probe = next(r["probe"] for r in results if r["probe"]["domain"] == domain)
        window, _ = reader.load(probe)
        adjacent = reader.dataset[probe["dataset_index"] + 1]
        left = {f.source.frame_id: f.source for f in window.frames}
        right = {f.source.frame_id: f.source for f in adjacent.frames}
        shared = sorted(left.keys() & right.keys())
        assert len(shared) == 4
        assert all(
            source_observation_identity(left[t])
            == source_observation_identity(right[t])
            for t in shared
        )
        overlaps.append(
            dict(
                domain=domain,
                world=probe["world"],
                current_frames=[probe["frame"], adjacent.current_frame_id],
                shared_source_frames=shared,
                equal=True,
            )
        )
        del window, adjacent, left, right
    dynamics, normal_errors = [], []
    movable = (
        10,
        11,
        13,
        15,
        16,
        18,
        20,
        30,
        31,
        32,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
    )
    for probe in (r["probe"] for r in results if r["probe"]["domain"] == "normal"):
        window, _ = reader.load(probe)
        current = window.current_mask
        truth = normal_group_targets(window.labels)[current]
        predictions = {}
        for name in VISITS:
            with np.load(output / "current" / name / f"{probe['probe']}.npz") as data:
                predictions[name] = {k: data[k] for k in ("target", "score")}
            data = predictions[name]
            for group in np.unique(truth[data["target"] == 0]):
                score = data["score"][(data["target"] == 0) & (truth == group)]
                normal_errors.append(
                    dict(
                        probe=probe["probe"],
                        model=name,
                        truth=int(group),
                        normal_points=len(score),
                        fp_ge0=int(np.sum(score >= 0)),
                        low_fp=int(
                            np.sum(
                                score
                                >= thresholds[name]["recall_at_fpr_limit"]["threshold"]
                            )
                        ),
                        high_fp=int(
                            np.sum(
                                score
                                >= thresholds[name]["official_high_recall"]["threshold"]
                            )
                        ),
                    )
                )
        ids = window.labels.instance
        semantic = window.labels.semantic
        valid = current & np.isin(semantic, movable) & (ids > 0)
        for instance in np.unique(ids[valid]):
            member = (ids == instance) & np.isin(semantic, movable)
            centers = []
            for frame in window.frame_ids:
                use = member & (window.points.source_frame == frame)
                if use.sum() >= 5:
                    centers.append(
                        (frame, np.median(window.points.coordinates[use], axis=0))
                    )
            if len(centers) < 2 or centers[-1][0] != window.current_frame_id:
                continue
            current_center = centers[-1][1]
            spread = max(
                float(np.linalg.norm(center - current_center)) for _, center in centers
            )
            for name in VISITS:
                data = predictions[name]
                use = member[current] & (data["target"] == 0)
                score = data["score"][use]
                dynamics.append(
                    dict(
                        probe=probe["probe"],
                        model=name,
                        frame=probe["frame"],
                        instance=int(instance),
                        scans=len(centers),
                        current_points=len(score),
                        registered_centroid_spread_m=spread,
                        fp_ge0=int(np.sum(score >= 0)),
                        fraction_ge0=float(np.mean(score >= 0)) if len(score) else None,
                        mean_score=float(np.mean(score)) if len(score) else None,
                    )
                )
    write_csv(output / "movable_instances.csv", dynamics)
    write_csv(output / "normal_class_errors.csv", normal_errors)
    write_json(
        output / "detail_notes.json",
        dict(
            adjacent_window_sources=overlaps,
            real_supported_group="Predicted normal support only; never a real fine-semantic ground truth",
            movable_instances="Published nonzero instance IDs within movable fine-semantic classes in 16 raw normal-201 probes; >=5 points in >=2 scans. Registered median-centroid spread is observation spread, not true speed or proof of motion.",
            reproduction="Original tolerance retained. Outside-tolerance saved-score differences are unresolved; historical full metrics remain unchanged. Internal diagnostics use one current path for both models.",
            confusion="Full five-frame and current-frame normal targets; inactive ground-truth groups remain rows, never remapped to a predicted class",
        ),
    )
    emit(
        "probe_details_complete",
        reproductions=len(reproductions),
        movable_instances=len(dynamics),
    )


def run_gradients(data_root, output):
    """One fixed AMP graph per training window; return gradients without an optimizer."""
    torch.set_num_threads(1)
    probes = [p for p in read_json(output / "probes.json")["probes"] if p["gradient"]]
    assert len(probes) == 8 and all(p["domain"] == "train" for p in probes)
    original_rng = random_state()
    reader = ProbeReader(data_root)
    statistics = read_json(ROOT / "protocols/nre/labels.json")
    resources = FullResources(emit=emit)
    resources()
    norms, cosines, direct, forwards = [], [], [], []
    for name, visit in VISITS.items():
        payload = torch.load(
            ROOT / f"runs/train/nre/visit_{visit:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        reference = payload["model"]
        scale = float(payload["scaler"]["scale"])
        successful = payload["state"]["successful_updates"]
        model = AJAE(normal_groups=statistics["active_groups"]).cuda()
        parameters = list(model.named_parameters())
        blocks = defaultdict(list)
        for i, (key, parameter) in enumerate(parameters):
            block = ".".join(key.split(".")[:2])
            blocks[block].append(i)
        for probe in probes:
            model.load_state_dict(reference, strict=True)
            model.train()
            seed_all(probe["seed"])
            window, inputs = reader.load(probe)
            inputs = inputs.to("cuda")
            target = torch.tensor(window.labels.anomaly_target.copy(), device="cuda")
            groups = torch.tensor(normal_group_targets(window.labels), device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                evidence = model(window, inputs=inputs, return_evidence=True)
                total, parts, observations = nre_loss(
                    evidence, target, groups, model, statistics, successful
                )
            values = evidence.score.detach().float().cpu().numpy()
            PredictionBatch.from_window(window, values, score_kind="logit").save(
                output / "gradients" / name / f"{probe['probe']}.npz", window=window
            )
            gradients = {}
            terms = dict(
                detection=1.0,
                support=0.25,
                semantic=0.5,
                tail=observations["tail_weight"],
            )
            for term_index, (term, weight) in enumerate(terms.items()):
                outputs = (evidence.score, evidence.support, evidence.normal_logits)
                grad = torch.autograd.grad(
                    parts[term] * weight * scale,
                    [p for _, p in parameters] + list(outputs),
                    retain_graph=term_index < 3,
                    allow_unused=True,
                )
                gradients[term] = [
                    None if g is None else g.detach().float() / scale
                    for g in grad[: len(parameters)]
                ]
                for label, g in zip(
                    ("score", "support", "normal_logits"),
                    grad[len(parameters) :],
                    strict=True,
                ):
                    maximum = (
                        float(g[target < 0].abs().max() / scale)
                        if g is not None and (target < 0).any()
                        else None
                    )
                    if maximum is not None and maximum != 0:
                        raise RuntimeError(
                            "ignored labels received direct loss supervision"
                        )
                    direct.append(
                        dict(
                            model=name,
                            probe=probe["probe"],
                            term=term,
                            output=label,
                            connected=g is not None,
                            ignore_points=int((target < 0).sum()),
                            ignore_direct_gradient_max=maximum,
                        )
                    )
                for block, indices in blocks.items():
                    selected = [
                        gradients[term][i]
                        for i in indices
                        if gradients[term][i] is not None
                    ]
                    finite = all(bool(torch.isfinite(g).all()) for g in selected)
                    squared = (
                        sum(float(g.double().square().sum()) for g in selected)
                        if finite
                        else None
                    )
                    norms.append(
                        dict(
                            model=name,
                            probe=probe["probe"],
                            block=block,
                            term=term,
                            weight=weight,
                            loss=float(parts[term].detach()),
                            loss_scale=scale,
                            norm=squared**0.5 if squared is not None else None,
                            finite=finite,
                            connected_tensors=len(selected),
                            unconnected_tensors=len(indices) - len(selected),
                            zero_tensors=sum(
                                not bool(torch.count_nonzero(g)) for g in selected
                            ),
                        )
                    )
            for i, left in enumerate(terms):
                for right in list(terms)[i + 1 :]:
                    for block, indices in blocks.items():
                        pairs = [
                            (gradients[left][j], gradients[right][j])
                            for j in indices
                            if gradients[left][j] is not None
                            and gradients[right][j] is not None
                        ]
                        finite = all(
                            bool(torch.isfinite(a).all() & torch.isfinite(b).all())
                            for a, b in pairs
                        )
                        na = (
                            sum(float(a.double().square().sum()) for a, b in pairs)
                            if finite
                            else 0
                        )
                        nb = (
                            sum(float(b.double().square().sum()) for a, b in pairs)
                            if finite
                            else 0
                        )
                        dot = (
                            sum(
                                float((a.double() * b.double()).sum()) for a, b in pairs
                            )
                            if finite
                            else 0
                        )
                        cosines.append(
                            dict(
                                model=name,
                                probe=probe["probe"],
                                block=block,
                                left=left,
                                right=right,
                                common_tensors=len(pairs),
                                finite=finite,
                                cosine=dot / (na * nb) ** 0.5 if na and nb else None,
                                left_common_norm=na**0.5,
                                right_common_norm=nb**0.5,
                            )
                        )
            assert all(p.grad is None for _, p in parameters)
            changed = sum(
                not torch.equal(v.detach().cpu(), reference[k])
                for k, v in model.named_buffers()
                if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
            )
            model.load_state_dict(reference, strict=True)
            assert_unchanged(model, reference)
            forwards.append(
                dict(
                    model=name,
                    probe=probe["probe"],
                    total_loss=float(total.detach()),
                    temporarily_changed_bn_tensors=changed,
                    restored=True,
                    optimizer_updates=0,
                )
            )
            del grad, gradients, evidence, total, parts, inputs, window, target, groups
            gc.collect()
            resources()
            emit("gradient_probe", model=name, probe=probe["probe"])
        del model, payload, reference, parameters
        gc.collect()
        torch.cuda.empty_cache()
    restore_random_state(original_rng)
    write_csv(output / "gradient_norms.csv", norms)
    write_csv(output / "gradient_cosines.csv", cosines)
    write_csv(output / "direct_gradients.csv", direct)
    write_csv(output / "gradient_forwards.csv", forwards)
    write_json(
        output / "gradient_check.json",
        dict(
            training_windows=8,
            forward_graphs=16,
            loss_gradients=64,
            optimizer_updates=0,
            formal_checkpoints_unchanged=True,
            temporary_bn_restored=True,
            parameter_grad_not_accumulated=True,
            random_state_restored=True,
            precision="Original training AMP float16 with FP32 head/loss; checkpoint loss scale applied then divided from returned gradients",
        ),
    )


def training_history(output):
    raw = read_jsonl(ROOT / "runs/train/nre/metrics.jsonl")
    visits = {r["step"]: r for r in raw if r["event"] == "train_step"}
    assert sorted(visits) == list(range(1, 28481))
    rows = [visits[i] for i in range(1, 28481)]
    assert sum(r["updated"] for r in rows) == 28470
    aggregates, usages = [], []
    for grouping, buckets in (
        ("visits_1000", [(s, min(s + 999, 28480)) for s in range(1, 28481, 1000)]),
        ("pass", [(1, 14240), (14241, 28480)]),
    ):
        for start, stop in buckets:
            selected = rows[start - 1 : stop]
            row = dict(
                grouping=grouping,
                first_visit=start,
                last_visit=stop,
                visits=len(selected),
                successful_updates=sum(r["updated"] for r in selected),
                overflows=sum(not r["updated"] for r in selected),
                clipped_updates=sum(
                    r["updated"] and r["grad_norm_before_clip"] > 1 for r in selected
                ),
                corrected_decoder_visits=sum(r["step"] >= 14241 for r in selected),
            )
            values = {
                "loss": [r["loss"] for r in selected],
                "gradient_norm": [
                    r["grad_norm_before_clip"] for r in selected if r["updated"]
                ],
                "scale": [r["scale_before"] for r in selected],
                "anomaly_points": [r["anomaly_count"] for r in selected],
                "correction_mean": [
                    r["nre_observations"]["correction_mean"] for r in selected
                ],
                "saturation": [
                    r["nre_observations"]["correction_near_bound_fraction"]
                    for r in selected
                ],
            }
            values.update(
                {
                    f"loss_{k}": [r["nre_loss"][k] for r in selected]
                    for k in (
                        "detection",
                        "support",
                        "semantic",
                        "tail",
                        "normal",
                        "anomaly",
                    )
                }
            )
            for key, value in values.items():
                row.update({f"{key}_{k}": v for k, v in describe(value).items()})
            aggregates.append(row)
            usage = np.sum(
                [r["nre_observations"]["prototype_usage"] for r in selected], axis=0
            )
            thresholds = np.array(
                [r["nre_observations"]["thresholds"] for r in selected]
            )
            active = read_json(ROOT / "protocols/nre/labels.json")["active_groups"]
            for i, group in enumerate(active):
                for p in range(4):
                    usages.append(
                        dict(
                            grouping=grouping,
                            first_visit=start,
                            last_visit=stop,
                            group=group,
                            prototype=p,
                            point_exposures=int(usage[i, p]),
                            threshold_mean=float(thresholds[:, i].mean()),
                        )
                    )
    write_csv(output / "training.csv", aggregates)
    write_csv(output / "training_usage.csv", usages)
    write_csv(
        output / "overflows.csv",
        (
            dict(
                visit=r["step"],
                world=r["sample"]["synthetic_sequence_index"],
                frame=r["sample"]["current_frame"],
                anomaly_points=r["anomaly_count"],
                normal_points=r["normal_count"],
                loss=r["loss"],
                scale_before=r["scale_before"],
                scale_after=r["scale_after"],
                **r["nre_loss"],
            )
            for r in rows
            if not r["updated"]
        ),
    )
    frequency = []
    for lo, hi in ((0, 0), (1, 4), (5, 19), (20, 99), (100, 499), (500, 10**12)):
        selected = [r for r in rows if lo <= r["anomaly_count"] <= hi]
        frequency.append(
            dict(
                anomaly_count_min=lo,
                anomaly_count_max=hi,
                visits=len(selected),
                worlds=len({r["sample"]["synthetic_sequence_index"] for r in selected}),
                overflows=sum(not r["updated"] for r in selected),
            )
        )
    write_csv(output / "training_frequency.csv", frequency)
    write_json(
        output / "history_check.json",
        dict(
            recorded_visits=sum(r["event"] == "train_step" for r in raw),
            effective_visits=28480,
            successful_updates=28470,
            discarded_replayed_visits=29,
            replay_rule="Keep the last actual train_step per visit; exclude failed 14241-14269 observations",
            fixed_coefficients=dict(detection=1, support=0.25, semantic=0.5, tail=0.1),
            absent_anomaly_windows=sum(r["anomaly_count"] == 0 for r in rows),
        ),
    )
    frames = [r for r in read_csv(output / "frames.csv") if r["eligible"] == "1"]
    monitor = read_json(ROOT / "protocols/observation_match_v2/monitor.json")["windows"]
    monitor_ids = {
        (r["sequence_index"], r["current_frame"])
        for r in monitor
        if r["view"] == "real" and r["scope"] == "qualified"
    }
    assert len(monitor_ids) == 152
    groups = []
    for scope in (
        "all",
        "monitor",
        "outside_monitor",
        "125_133_165",
        "125_133_165_monitor",
    ):
        selected = [
            r
            for r in frames
            if (
                scope == "all"
                or (
                    scope == "monitor"
                    and (int(r["sequence"]), int(r["frame"])) in monitor_ids
                )
                or (
                    scope == "outside_monitor"
                    and (int(r["sequence"]), int(r["frame"])) not in monitor_ids
                )
                or (
                    scope.startswith("125_")
                    and r["sequence"] == "125"
                    and 133 <= int(r["frame"]) <= 165
                    and (
                        not scope.endswith("monitor")
                        or (125, int(r["frame"])) in monitor_ids
                    )
                )
            )
        ]
        total = sum(int(r["anomaly_points"]) for r in selected)
        for sequence in ["all"] + sorted({r["sequence"] for r in selected}):
            subset = [
                r for r in selected if sequence == "all" or r["sequence"] == sequence
            ]
            row = dict(
                scope=scope,
                sequence=sequence,
                frames=len(subset),
                anomaly_points=sum(int(r["anomaly_points"]) for r in subset),
            )
            row["anomaly_share_within_scope"] = (
                row["anomaly_points"] / total if total else None
            )
            for model in VISITS:
                for key in ("ap_deficit", "low_fn", "high_fp"):
                    row[f"{model}_{key}"] = sum(
                        float(r[f"{model}_{key}"]) for r in subset
                    )
            groups.append(row)
    write_csv(output / "monitor_coverage.csv", groups)
    emit("history_complete", visits=len(rows), overflows=10)


def render_report(output):
    """Draw measured curves and paths; the authored report remains the single source."""
    import subprocess
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    for family, filename in (
        ("SimSun", "simsun.ttc"),
        ("Times New Roman", "times.ttf"),
    ):
        font_manager.fontManager.addfont(Path("/mnt/c/Windows/Fonts") / filename)
        font_manager.findfont(family, fallback_to_default=False)
    colors = {"mid": "#2366a0", "late": "#c04d28"}
    published = ROOT / "reports"
    published.mkdir(exist_ok=True)
    labels = {"mid": "14,240 节点", "late": "28,480 节点"}
    metrics = read_json(output / "global.json")
    frames = [r for r in read_csv(output / "frames.csv") if r["sequence"] == "125"]
    scores = read_csv(output / "scores.csv")
    layers = read_csv(output / "layers.csv")
    settings = {
        "font.family": ["Times New Roman", "SimSun"],
        "pdf.fonttype": 42,
        "font.size": 11,
    }
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for model in VISITS:
            curve = read_csv(output / f"curve_{model}.csv")
            values = {
                k: np.array([float(r[k]) for r in curve]) * 100
                for k in ("recall", "precision", "fpr")
            }
            axes[0].plot(
                values["recall"],
                values["precision"],
                color=colors[model],
                lw=1.4,
                label=f"{labels[model]}：AP {metrics[model]['AP']:.3f}%",
            )
            axes[1].plot(
                values["fpr"],
                values["recall"],
                color=colors[model],
                lw=1.4,
                label=labels[model],
            )
            point = metrics[model]["recall_at_fpr_limit"]
            axes[1].scatter(
                point["FPR"], point["recall"], color=colors[model], s=28, zorder=3
            )
            axes[1].annotate(
                f"{point['recall']:.3f}%",
                (point["FPR"], point["recall"]),
                xytext=(-70, -15),
                textcoords="offset points",
                color=colors[model],
            )
        axes[0].set(
            xlim=(0, 100),
            ylim=(0, 100),
            xlabel="召回率（%）",
            ylabel="精确率（%）",
            title="全部官方合格点：精确率与召回率",
        )
        axes[1].axhline(95, color="0.4", ls="--", lw=1)
        axes[1].set(
            xlim=(0, 1.01),
            ylim=(50, 100),
            xlabel="正常点误报率（%）",
            ylabel="异常点召回率（%）",
            title="低误报区间：实际分数阈值",
        )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(loc="lower left", fontsize=10)
        fig.savefig(published / "curves.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(
            3, 1, figsize=(10.5, 8), sharex=True, layout="constrained"
        )
        x = np.array([int(r["frame"]) for r in frames])
        axes[0].plot(x, [int(r["anomaly_points"]) for r in frames], color="#484848")
        axes[0].set(ylabel="当前异常点数", title="第 125 序列：连续观测与全局排序失分")
        for model in VISITS:
            for ax, field, scale in (
                (axes[1], "ap_deficit", 100),
                (axes[2], "low_fn", 1),
            ):
                values = [
                    float(r.get(f"{model}_{field}") or "nan") * scale for r in frames
                ]
                ax.plot(x, values, color=colors[model], label=labels[model], lw=1.2)
        axes[1].set(ylabel="AP 失分（百分点）")
        axes[2].set(ylabel="低误报工作点漏检数", xlabel="原始帧号")
        for ax in axes:
            ax.axvspan(133, 165, color="#ddbd69", alpha=0.22)
            ax.grid(alpha=0.2)
        axes[1].legend(loc="upper left")
        fig.savefig(published / "sequence125.pdf")
        plt.close(fig)

        cohorts = (
            ("训练", lambda r: r["domain"] == "train"),
            ("合成验证", lambda r: r["domain"] == "validation"),
            ("真实基础", lambda r: r["domain"] == "real" and r["kind"] == "basic"),
            ("125 全探针", lambda r: r["domain"] == "real" and r["world"] == "125"),
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for offset, model in enumerate(VISITS):
            means, bounds = [], []
            for _, select in cohorts:
                selected = [
                    r
                    for r in scores
                    if r["model"] == model
                    and r["scope"] == "official_current"
                    and r["label"] == "anomaly"
                    and int(r["points"])
                    and select(r)
                ]
                count = sum(int(r["points"]) for r in selected)
                means.append(
                    sum(float(r["base_mean"]) * int(r["points"]) for r in selected)
                    / count
                )
                bounds.append(
                    100
                    * sum(
                        float(r["correction_negative_bound"]) * int(r["points"])
                        for r in selected
                    )
                    / count
                )
            positions = np.arange(len(cohorts)) + (offset - 0.5) * 0.32
            axes[0, 0].bar(
                positions, means, width=0.30, color=colors[model], label=labels[model]
            )
            axes[0, 1].bar(positions, bounds, width=0.30, color=colors[model])
        axes[0, 0].set(ylabel="基础拒识分数均值", title="异常点：训练改善与真实退化")
        axes[0, 0].axhline(0, color="0.5", lw=0.7)
        axes[0, 0].legend(fontsize=10)
        axes[0, 1].set(
            ylabel="修正不高于 −1.98 的比例（%）",
            title="异常点：负向修正接近边界",
            ylim=(0, 105),
        )
        for ax in axes[0]:
            ax.set_xticks(range(len(cohorts)), [name for name, _ in cohorts])
            ax.grid(axis="y", alpha=0.2)
        pairs = [
            r
            for r in read_csv(output / "paired.csv")
            if r["domain"] == "real"
            and r["world"] == "125"
            and r["scope"] == "official_current"
            and r["group"] == "anomaly"
            and int(r["points"])
        ]
        pairs.sort(key=lambda r: int(r["frame"]))
        base = np.array([float(r["delta_base_mean"]) for r in pairs])
        correction = np.array([float(r["delta_correction_mean"]) for r in pairs])
        axes[1, 0].bar(range(len(pairs)), base, color="#688dae", label="基础拒识变化")
        axes[1, 0].bar(
            range(len(pairs)),
            correction,
            bottom=base,
            color="#da9068",
            label="有界修正变化",
        )
        axes[1, 0].set_xticks(
            range(len(pairs)), [r["frame"] for r in pairs], rotation=60
        )
        axes[1, 0].set(
            ylabel="最终节点减中期节点",
            xlabel="125 探针当前帧号",
            title="同点分数变化的前向分解",
        )
        axes[1, 0].legend(fontsize=10)
        stages = ("enc0", "enc4", "decoded")
        for cohort, prefix in (("训练", "train_"), ("125", "real_125_")):
            for model in VISITS:
                values = [
                    np.median(
                        [
                            float(r["separation"])
                            for r in layers
                            if r["model"] == model
                            and r["probe"].startswith(prefix)
                            and r["layer"] == stage
                            and r["separation"]
                        ]
                    )
                    for stage in stages
                ]
                axes[1, 1].plot(
                    range(3),
                    values,
                    marker="o",
                    color=colors[model],
                    ls="-" if cohort == "训练" else "--",
                    label=f"{cohort}，{labels[model]}",
                )
        axes[1, 1].set_xticks(range(3), ["第一编码层", "最深编码层", "最终解码层"])
        axes[1, 1].set(
            ylabel="标准化类中心距离的逐窗中位数", title="固定特征样本的可区分程度"
        )
        axes[1, 1].legend(fontsize=9)
        for ax in axes[1]:
            ax.grid(axis="y", alpha=0.2)
        fig.savefig(published / "paths.pdf")
        plt.close(fig)
    metadata = dict(
        mainfont="Times New Roman",
        sansfont="Times New Roman",
        monofont="Times New Roman",
        CJKmainfont="SimSun",
        CJKoptions=["BoldFont=SimSun", "ItalicFont=SimSun"],
        geometry=["a4paper", "margin=21mm"],
        fontsize="10pt",
        linestretch=1.12,
        colorlinks=True,
    )
    with tempfile.TemporaryDirectory(prefix="ajae-report-") as temporary:
        path = Path(temporary) / "metadata.json"
        write_json(path, metadata)
        subprocess.run(
            [
                "pandoc",
                str(ROOT / "DIAGNOSIS.md"),
                "--pdf-engine=xelatex",
                "--metadata-file",
                str(path),
                "-o",
                str(published / "diagnosis.pdf"),
            ],
            cwd=ROOT,
            check=True,
        )
    # Expose the small published PDFs beside local evidence without duplicate data.
    for name in ("curves.pdf", "sequence125.pdf", "paths.pdf", "diagnosis.pdf"):
        local = output / ("report.pdf" if name == "diagnosis.pdf" else name)
        local.unlink(missing_ok=True)
        local.symlink_to(published / name)
    emit("report_rendered", directory=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "scores",
            "errors",
            "coverage",
            "select",
            "numerics",
            "probes",
            "summarize",
            "history",
            "gradients",
            "details",
            "report",
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/jasongao/Data/STU")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "report":
        render_report(args.output)
    elif args.stage == "details":
        probe_details(args.data_root, args.output)
    elif args.stage == "gradients":
        run_gradients(args.data_root, args.output)
    elif args.stage == "history":
        training_history(args.output)
    elif args.stage == "numerics":
        run_numerics(args.data_root, args.output)
    elif args.stage == "probes":
        run_probes(args.data_root, args.output)
    elif args.stage == "summarize":
        summarize_probes(args.output)
    elif args.stage == "select":
        select_probes(args.output)
    elif args.stage == "coverage":
        run_coverage(args.data_root, args.output, args.workers)
    else:
        run_scores(
            args.data_root,
            args.output,
            args.workers,
            reuse_metrics=args.stage == "errors",
        )


if __name__ == "__main__":
    main()
