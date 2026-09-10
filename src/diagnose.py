"""Locate false positives using saved synthetic scores; never run a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .data import FrozenDataset, FrozenFrame, FramePrediction, _atomic_json, host_disk
from .evaluate import packed_scores, pooled_files
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator as Official


SCOPES = ("all", "stu_filtered")
LIMITS = (0.01, 0.001, 0.0001)
RANGES = ("<2.5", "[2.5,10)", "[10,20)", "[20,35)", "[35,50]", ">50")
FRAME_GROUPS = ("no_anomaly", "anomaly_not_eligible", "eligible")


def scope_masks(xyzi, target):
    # Reuse only STU's observation filter, never its raw-semantic anomaly mapping.
    distance = np.linalg.norm(xyzi[:, :3], axis=1)
    inside = (distance >= Official.min_eval_distance) & (distance <= Official.max_eval_distance)
    eligible = int(np.count_nonzero(inside & (target == 1))) >= Official.min_num_points_to_eval
    return ((target >= 0), (target >= 0) & inside & eligible), distance, eligible


def range_ids(distance):
    result = np.searchsorted([2.5, 10, 20, 35], distance, side="right")
    result[distance > 50] = 5
    return result


def initialize(dataset_root, data_root, run, temporary, thresholds=None):
    global DATASET, RUN, TEMPORARY, THRESHOLDS, SAMPLES, WORLDS
    DATASET = FrozenDataset(dataset_root, data_root, "validation")
    RUN, TEMPORARY, THRESHOLDS = Path(run), Path(temporary), thresholds
    SAMPLES = defaultdict(list)
    WORLDS = list(dict.fromkeys(path.parent.parent.name for path, _, _ in DATASET.samples))
    if len(WORLDS) > 32:
        raise ValueError("world recurrence bitsets support at most 32 worlds")
    for path, identity, frame in DATASET.samples:
        SAMPLES[frame].append((path, identity, WORLDS.index(path.parent.parent.name)))
    keys = [(identity, frame) for _, identity, frame in DATASET.samples]
    if len(set(keys)) != len(keys) or len(WORLDS) * len(SAMPLES) != len(keys):
        raise ValueError("diagnosis requires distinct complete world/frame identities")


def read_world(original, entry):
    path, identity, world_index = entry
    frozen = FrozenFrame.load(path, original, identity)
    path = RUN / "predictions" / "synthetic" / WORLDS[world_index] / f"{original.frame_id:06d}.npz"
    prediction = FramePrediction.load(path, frozen.source)
    # Loading already validates exact original-slot coverage and scan identity.
    scores = np.zeros(frozen.source.slot_count, np.float32)
    scores[prediction.source_slot] = prediction.anomaly_score
    return frozen, scores


def pack_frame(frame):
    original = DATASET.sequence[frame]
    rows = []
    streams = [(TEMPORARY / f"{frame:06d}_{scope}.bin").open("wb") for scope in SCOPES]
    try:
        for entry in SAMPLES[frame]:
            frozen, scores = read_world(original, entry)
            target = frozen.anomaly_target
            masks, _, eligible = scope_masks(frozen.source.xyzi, target)
            row = dict(world=WORLDS[entry[2]], world_identity=entry[1], frame=frame,
                       actual_points=len(frozen.source.real_slots), eligible=eligible)
            for scope, use, stream in zip(SCOPES, masks, streams, strict=True):
                packed_scores(scores[use], target[use], score_kind="logit").tofile(stream)
                row[scope] = dict(normal=int(np.count_nonzero(use & (target == 0))),
                                  anomaly=int(np.count_nonzero(use & (target == 1))))
            rows.append(row)
    finally:
        for stream in streams:
            stream.close()
    return rows


def recurrence(seen, false_positive):
    """Distinct world membership, so coincident slots cannot inflate recurrence."""
    return dict(
        observed=int(np.count_nonzero(seen)),
        false_positive=[int(np.count_nonzero(row)) for row in false_positive],
        observed_world_histogram=np.bincount(np.bitwise_count(seen[seen != 0]), minlength=33).tolist(),
        fp_world_histograms=[np.bincount(np.bitwise_count(row[row != 0]), minlength=33).tolist()
                             for row in false_positive],
    )


def add_group(groups, kind, values, accepted):
    # Column 0 is the normal denominator; columns 1..3 use fixed global thresholds.
    unique, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    fps = [np.bincount(inverse[row], minlength=len(unique)) for row in accepted]
    for i, value in enumerate(unique):
        groups[kind][str(value)] += np.array([counts[i], *(row[i] for row in fps)], np.int64)


def analyze_frame(frame):
    original = DATASET.sequence[frame]
    normal_source = np.flatnonzero((original.labels.semantic_target != 255) & ~original.zero_slot_mask)
    # A coordinate key is local to this source frame, not a track across time.
    _, first, inverse = np.unique(original.xyzi[normal_source, :3], axis=0,
                                  return_index=True, return_inverse=True)
    positions = np.full(original.slot_count, -1, np.int64)
    positions[normal_source] = inverse
    slot_seen = np.zeros((2, original.slot_count), np.uint32)
    slot_fp = np.zeros((2, 3, original.slot_count), np.uint32)
    position_seen = np.zeros((2, len(first)), np.uint32)
    position_fp = np.zeros((2, 3, len(first)), np.uint32)
    maximum = np.full((2, original.slot_count), -np.inf, np.float32)
    maximum_world = np.full((2, original.slot_count), -1, np.int16)
    groups = [{kind: defaultdict(lambda: np.zeros(4, np.int64))
               for kind in ("semantic", "distance", "frame_type", "world", "official_membership")}
              for _ in SCOPES]
    totals = np.zeros((2, 2, 3), np.int64)
    for entry in SAMPLES[frame]:
        frozen, scores = read_world(original, entry)
        target = frozen.anomaly_target
        masks, distance, eligible = scope_masks(frozen.source.xyzi, target)
        frame_type = 2 if eligible else int(np.any(target == 1))
        world_bit = np.uint32(1 << entry[2])
        for s, use in enumerate(masks):
            positive = scores[use & (target == 1)]
            totals[s, 1] += (positive[None, :] >= THRESHOLDS[s, :, None]).sum(axis=1)
            slots = np.flatnonzero(use & (target == 0))
            if not len(slots):
                continue
            if np.any(positions[slots] < 0) or not np.array_equal(frozen.source.xyzi[slots], original.xyzi[slots]):
                raise ValueError("normal diagnosis rows must be unchanged original background slots")
            accepted = scores[slots][None, :] >= THRESHOLDS[s, :, None]
            totals[s, 0] += accepted.sum(axis=1)
            add_group(groups[s], "semantic", original.labels.semantic[slots], accepted)
            add_group(groups[s], "distance", range_ids(distance[slots]), accepted)
            nfp = np.r_[len(slots), accepted.sum(axis=1)]
            groups[s]["frame_type"][str(frame_type)] += nfp
            groups[s]["world"][WORLDS[entry[2]]] += nfp
            inside = (distance[slots] >= 2.5) & (distance[slots] <= 50)
            membership = inside.astype(np.int8) + 2 * int(eligible)
            add_group(groups[s], "official_membership", membership, accepted)
            slot_seen[s, slots] |= world_bit
            np.bitwise_or.at(position_seen[s], positions[slots], world_bit)
            improved = slots[scores[slots] > maximum[s, slots]]
            maximum[s, improved] = scores[improved]
            maximum_world[s, improved] = entry[2]
            for k, take in enumerate(accepted):
                selected = slots[take]
                slot_fp[s, k, selected] |= world_bit
                np.bitwise_or.at(position_fp[s, k], positions[selected], world_bit)
    result = dict(frame=frame, scopes={}, candidates=[])
    for s, scope in enumerate(SCOPES):
        count = sum((v for v in groups[s]["semantic"].values()), np.zeros(4, np.int64))
        top = []
        for k in range(3):
            repetitions = np.bitwise_count(slot_fp[s, k])
            selected = np.lexsort((np.arange(len(repetitions)), -maximum[s], -repetitions.astype(np.int16)))[:5]
            top.append([dict(slot=int(i), xyz=original.xyzi[i, :3].tolist(),
                             semantic=int(original.labels.semantic[i]),
                             fp_worlds=int(repetitions[i]),
                             observed_worlds=int(np.bitwise_count(slot_seen[s, i])))
                        for i in selected if repetitions[i]])
        result["scopes"][scope] = dict(
            counts=count.tolist(), fp=totals[s, 0].tolist(), tp=totals[s, 1].tolist(),
            groups={kind: {key: value.tolist() for key, value in table.items()}
                    for kind, table in groups[s].items()},
            slots=recurrence(slot_seen[s], slot_fp[s]),
            positions=recurrence(position_seen[s], position_fp[s]), top_slots=top,
        )
        # Choose visible clusters for inspection, counting coincident 201 slots once.
        available = np.flatnonzero(slot_fp[s, 1])
        for semantic in np.unique(original.labels.semantic[available]):
            slots = available[original.labels.semantic[available] == semantic]
            _, keep = np.unique(positions[slots], return_index=True)
            slots = slots[keep]
            cells = np.floor(original.xyzi[slots, :3]).astype(np.int32)
            unique, cell_index, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
            cell = int(np.argmax(counts))
            members = slots[cell_index == cell]
            center = unique[cell] + 0.5
            anchor = int(members[np.argmin(np.linalg.norm(original.xyzi[members, :3] - center, axis=1))])
            result["candidates"].append(dict(scope=scope, semantic=int(semantic), frame=frame,
                slot=anchor, world=WORLDS[int(maximum_world[s, anchor])],
                xyz=original.xyzi[anchor, :3].tolist(), score=float(maximum[s, anchor]),
                unique_fp_positions_in_one_metre_cell=int(counts[cell])))
    return result


def group_table(table, total_fp):
    return {key: dict(normal=int(value[0]), points={f"{limit:g}": dict(
        fp=int(value[k + 1]), FPR=100 * value[k + 1] / value[0] if value[0] else None,
        fp_share=100 * value[k + 1] / total_fp[k] if total_fp[k] else None)
        for k, limit in enumerate(LIMITS)}) for key, value in table.items()}


def aggregate(rows, metrics):
    result = {}
    for scope in SCOPES:
        tables = {kind: defaultdict(lambda: np.zeros(4, np.int64)) for kind in rows[0]["scopes"][scope]["groups"]}
        source_frames = {}
        identities = {kind: dict(observed=0, false_positive=np.zeros(3, np.int64),
                        observed_world_histogram=np.zeros(33, np.int64), fp_world_histograms=np.zeros((3, 33), np.int64))
                      for kind in ("slots", "positions")}
        fp, tp = np.zeros(3, np.int64), np.zeros(3, np.int64)
        for row in rows:
            data = row["scopes"][scope]
            fp += data["fp"]
            tp += data["tp"]
            source_frames[str(row["frame"])] = np.asarray(data["counts"], np.int64)
            for kind, table in data["groups"].items():
                for key, value in table.items():
                    tables[kind][key] += value
            for kind, totals in identities.items():
                for key in totals:
                    totals[key] += data[kind][key] if key == "observed" else np.asarray(data[kind][key], np.int64)
        expected = metrics[scope]
        for k, limit in enumerate(LIMITS):
            point = expected["operating_points"][f"{limit:g}"]
            if (int(fp[k]), int(tp[k])) != (point["fp"], point["tp"]):
                raise ValueError("grouped counts disagree with complete-tie global ranking")
        tables["source_frame"] = source_frames
        for table in tables.values():
            total = sum(table.values(), np.zeros(4, np.int64))
            if not np.array_equal(total, np.r_[expected["normal_count"], fp]):
                raise ValueError("group denominator or false-positive count does not partition the scope")
        for kind, counts in identities.items():
            identities[kind] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in counts.items()}
        source_counts = np.array(list(source_frames.values()))
        concentration = []
        for k in range(3):
            descending = np.sort(source_counts[:, k + 1])[::-1]
            concentration.append(dict(source_frames_with_fp=int(np.count_nonzero(descending)),
                top_frame_fp_share={str(n): 100 * int(descending[:n].sum()) / int(fp[k]) if fp[k] else None
                                   for n in (1, 5, 10, 50)},
                unique_slots=identities["slots"]["false_positive"][k],
                unique_positions=identities["positions"]["false_positive"][k]))
        result[scope] = dict(groups={kind: group_table(table, fp) for kind, table in tables.items()},
                             identities=identities, concentration=concentration)
    return result


def inspect_cases(rows, summary, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    chinese = FontProperties(fname="/mnt/c/Windows/Fonts/simsun.ttc")
    english = FontProperties(fname="/mnt/c/Windows/Fonts/times.ttf")
    if chinese.get_name() != "SimSun" or english.get_name() != "Times New Roman":
        raise ValueError("required figure fonts are unavailable")
    fontManager.addfont(english.get_file())
    fontManager.addfont(chinese.get_file())
    matplotlib.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42,
                                "axes.unicode_minus": False})
    candidates = [case for row in rows for case in row["candidates"]]
    selected = []
    for scope in SCOPES:
        semantic = summary[scope]["groups"]["semantic"]
        leaders = sorted(semantic, key=lambda key: -semantic[key]["points"]["0.001"]["fp"])[:3]
        for key in leaders:
            choices = [case for case in candidates if case["scope"] == scope and case["semantic"] == int(key)]
            if choices:
                selected.append(max(choices, key=lambda case: (case["unique_fp_positions_in_one_metre_cell"], -case["frame"])))
    cases = []
    for index, case in enumerate(selected):
        original = DATASET.sequence[case["frame"]]
        entry = next(item for item in SAMPLES[case["frame"]] if WORLDS[item[2]] == case["world"])
        frozen, scores = read_world(original, entry)
        target = frozen.anomaly_target
        masks, _, _ = scope_masks(frozen.source.xyzi, target)
        use = masks[SCOPES.index(case["scope"])]
        threshold = metrics[case["scope"]]["operating_points"]["0.001"]["threshold"]
        center = np.asarray(case["xyz"])
        local = (np.linalg.norm(frozen.source.xyzi[:, :3] - center, axis=1) <= 6) & ~frozen.source.zero_slot_mask
        local_slots = np.flatnonzero(local)
        # Plot original geometry once per exact position; score maxima preserve visible false positives.
        xyz, representative, inverse = np.unique(frozen.source.xyzi[local, :3], axis=0, return_index=True, return_inverse=True)
        local_fp = local & use & (target == 0) & (scores >= threshold)
        fp_position = np.zeros(len(xyz), bool)
        np.logical_or.at(fp_position, inverse, local_fp[local])
        anomaly_position = target[local_slots[representative]] == 1
        ignored_position = target[local_slots[representative]] == -1
        highlighted = (original.labels.semantic[local_slots[representative]] == case["semantic"]) & ~anomaly_position & ~ignored_position
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        groups = [(np.ones(len(xyz), bool), "#cccccc", "完整局部背景", 3),
                  (highlighted, "#3182bd", "所检正常类别", 5),
                  (fp_position, "#de2d26", "正常误报", 8),
                  (anomaly_position, "#00a651", "插入异常", 12)]
        relative = xyz - center
        for ax, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2)), strict=True):
            for take, color, label, size in groups:
                ax.scatter(relative[take, a], relative[take, b], s=size, c=color, label=label, linewidths=0)
            ax.scatter([0], [0], s=65, marker="x", color="black", linewidths=1)
            ax.set_xlabel(f"{'xyz'[a]} (m)", fontproperties=english)
            ax.set_ylabel(f"{'xyz'[b]} (m)", fontproperties=english)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(english)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, prop=chinese)
        fig.suptitle("正常高分点的局部几何核对", fontproperties=chinese)
        fig.text(.5, .91, f"{case['scope']} | train/201/{case['frame']:06d} | {case['world']} | raw={case['semantic']} | slot={case['slot']}",
                 ha="center", fontproperties=english)
        fig.tight_layout(rect=(0, .07, 1, .89))
        stem = f"case_{index + 1}"
        fig.savefig(output / f"{stem}.png", dpi=160)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)
        raw, counts = np.unique(original.labels.semantic[local & (target == 0)], return_counts=True)
        anomalies = frozen.source.xyzi[target == 1, :3]
        cases.append({**case, "world_identity": entry[1], "global_threshold": threshold,
            "radius_m": 6, "normal_semantics": dict(zip(map(str, raw), map(int, counts), strict=True)),
            "normal_points": int(np.count_nonzero(local & (target == 0))),
            "false_positive_points": int(np.count_nonzero(local_fp)),
            "anomaly_points": int(np.count_nonzero(local & (target == 1))),
            "unique_positions": len(xyz), "unique_fp_positions": int(fp_position.sum()),
            "relative_xyz_min": relative.min(axis=0).tolist(), "relative_xyz_max": relative.max(axis=0).tolist(),
            "nearest_inserted_return_m": float(np.linalg.norm(anomalies - center, axis=1).min()) if len(anomalies) else None,
            "figure": f"{stem}.png", "pdf": f"{stem}.pdf"})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("results/synthetic/experiment"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    parser.add_argument("--run", type=Path, default=Path("results/v1"))
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    started = time.monotonic()
    previous = json.loads((args.run / "synthetic.json").read_text())
    binding = json.loads((args.run / "predictions" / "manifest.json").read_text())
    with (args.run / "epoch1.pt").open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = dict(checkpoint_sha256=checkpoint_digest,
                    dataset_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(), score="raw_float32_logit")
    if binding != expected:
        raise ValueError("prediction binding differs from checkpoint or final dataset")
    output = args.run / "diagnosis"
    output.mkdir(exist_ok=True)
    resources = host_disk()
    # Two record sets plus one sorted copy, with a small allowance for reports.
    peak = 24 * (previous["metrics"]["normal_count"] + previous["metrics"]["anomaly_count"]) + 100_000_000
    if resources["SizeRemaining"] - peak < resources["reserve_bytes"]:
        raise OSError("diagnostic sorting would consume the E: safety reserve")
    print(json.dumps(dict(event="start", jobs=args.jobs, peak_new_bytes=peak, host=resources)), flush=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        initargs = (args.dataset, args.data_root, args.run, temporary)
        initialize(*initargs)
        frames = sorted(SAMPLES)
        packed_rows = []
        with ProcessPoolExecutor(args.jobs, initializer=initialize, initargs=initargs) as pool:
            for i, rows in enumerate(pool.map(pack_frame, frames), 1):
                packed_rows.extend(rows)
                if i % 100 == 0 or i == len(frames):
                    print(json.dumps(dict(event="pack", source_frames=i, worlds=len(WORLDS), seconds=time.monotonic()-started,
                                          host=host_disk())), flush=True)
        metrics = {}
        for scope in SCOPES:
            paths = [Path(temporary) / f"{frame:06d}_{scope}.bin" for frame in frames]
            print(json.dumps(dict(event="sort", scope=scope, bytes=sum(p.stat().st_size for p in paths), host=host_disk())), flush=True)
            metrics[scope] = pooled_files(paths, score_kind="logit", fpr_limits=LIMITS)
            for path in paths:
                path.unlink()
            eligible = [r for r in packed_rows if scope == "all" or r["eligible"]]
            metrics[scope].update(frames=len(eligible), source_frames=len({r["frame"] for r in eligible}),
                worlds=len({r["world_identity"] for r in eligible}),
                anomaly_prevalence_percent=100 * metrics[scope]["anomaly_count"] / (metrics[scope]["normal_count"] + metrics[scope]["anomaly_count"]))
            for kind, key in (("normal", "normal_count"), ("anomaly", "anomaly_count")):
                if sum(row[scope][kind] for row in packed_rows) != metrics[scope][key]:
                    raise ValueError("world/frame counts disagree with metric pool")
            print(json.dumps(dict(event="metrics", scope=scope, metrics=metrics[scope])), flush=True)
        for key, value in previous["metrics"].items():
            if metrics["all"][key] != value:
                raise ValueError(f"original full-synthetic result did not reproduce: {key}")
        _atomic_json(output / "metrics.json", dict(binding=binding, metrics=metrics, frame_counts=packed_rows,
                                                    original_full_result_exactly_reproduced=True))
        thresholds = np.array([[metrics[scope]["operating_points"][f"{limit:g}"]["threshold"]
                                if metrics[scope]["operating_points"][f"{limit:g}"]["threshold"] is not None else np.inf
                                for limit in LIMITS] for scope in SCOPES], np.float64)
        rows = []
        with ProcessPoolExecutor(args.jobs, initializer=initialize, initargs=(*initargs, thresholds)) as pool:
            for i, row in enumerate(pool.map(analyze_frame, frames), 1):
                rows.append(row)
                if i % 100 == 0 or i == len(frames):
                    print(json.dumps(dict(event="groups", source_frames=i, seconds=time.monotonic()-started,
                                          host=host_disk())), flush=True)
        summary = aggregate(rows, metrics)
        cases = inspect_cases(rows, summary, metrics, output)
    _atomic_json(output / "groups.json", dict(binding=binding, scopes=summary, source_frames=rows, cases=cases,
        definitions=dict(distance_bins=RANGES, frame_types=FRAME_GROUPS,
            official_membership={"0": "ineligible/outside", "1": "ineligible/inside", "2": "eligible/outside", "3": "eligible/inside"},
            point_identity="train/201, source frame, original slot; independent of synthetic world",
            position_identity="train/201, source frame, exact original xyz; not a tracked surface across frames",
            recurrence="number of distinct worlds; coincident slots within one world count once",
            case_selection="top three normal semantic FP contributors per scope at 0.1% global FPR; densest one-metre cell of distinct FP positions across source frames; six-metre display only"),
        seconds=time.monotonic()-started, jobs=args.jobs, host_before=resources, host_after=host_disk()))
    print(json.dumps(dict(event="complete", seconds=time.monotonic()-started, directory=str(output))), flush=True)


if __name__ == "__main__":
    main()
