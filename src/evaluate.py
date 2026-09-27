"""SERVE pooled-point validation, paired comparisons and original-slot inference."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import NormalScans, Scans, load_manifest, make_real_manifest, read_scan, write_json
from .model import NormalHypothesis, prepare_scan, scatter_scores, to_device
from .normal import hypothesis_observation, normal_semantic_metrics
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator

STU_COMMIT = "8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2"


class PreparedScans(Scans):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if sample["targets"] is None:
            sample["targets"] = np.full(len(sample["xyzi"]), -1, np.int8)
        result = prepare_scan(sample)
        result["observation"] = hypothesis_observation(sample["xyzi"])
        if sample.get("instance") is not None:
            result["instance"] = torch.from_numpy(sample["instance"].astype(np.int64))
        return result


def evaluation_indices(manifest):
    """Use the official per-scan five-anomaly rule."""
    if any(row["eligible"] is None for row in manifest["records"]):
        raise ValueError("hidden test labels are unavailable; use export for original-slot predictions")
    return [i for i, row in enumerate(manifest["records"]) if row["eligible"]]


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # Shape-compatible historical weights still represent a different model.
    NormalHypothesis.validate_checkpoint(saved, require_fitted=True)
    model = NormalHypothesis(variant=saved["config"]["variant"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def autocast(device):
    # The paper's inference and runtime protocol both use FP32.
    return torch.autocast(device.type, enabled=False)


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine available physical memory")


def better(metrics, previous):
    if not all(np.isfinite(metrics[k]) for k in ("AP", "FPR95", "AUROC")):
        raise ValueError("nonfinite official validation metrics")
    return previous is None or (metrics["AP"], -metrics["FPR95"], metrics["AUROC"]) > (
        previous["AP"], -previous["FPR95"], previous["AUROC"])


@torch.no_grad()
def evaluate(model, manifest, device, workers=4, score_path=None, record_points=False, readout=None):
    """Call the pinned official implementation once across the complete valid set."""
    if manifest["kind"] not in ("val", "test"):
        raise ValueError("evaluation requires a held-out validation or test manifest")
    if manifest.get("labeled") is False:
        raise ValueError("hidden test labels are unavailable; use export to preserve original-slot predictions")
    indices = evaluation_indices(manifest)
    count = sum(manifest["records"][i]["normal"] + manifest["records"][i]["anomaly"] for i in indices)
    if not count:
        raise ValueError("no eligible evaluation points")
    # Budget sklearn's exact sorting, targets and cumulative sums; never use swap.
    required = 64 * count + 1_000_000_000
    if memory_available() < required:
        raise RuntimeError(f"official metrics require about {required / 1e9:.1f} GB free RAM")
    dataset = PreparedScans(manifest)
    loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 1} if workers else {}),
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    calculator = PointOODMetricsCalculator()
    # Optional diagnostic export preserves official point order and exact scores.
    scores = (np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(count,))
              if score_path is not None else np.empty(count, np.float32))
    labels = np.empty(count, np.int8)
    raw_scores = identities = None
    frames, raw_cursor = [], 0
    if record_points:
        if score_path is None:
            raise ValueError("point recording needs a persistent evaluation score path")
        score_path = Path(score_path)
        raw_count = sum(manifest["records"][i]["points"] for i in indices)
        raw_scores = np.lib.format.open_memmap(score_path.with_stem(score_path.stem + "_all"), mode="w+",
                                               dtype=np.float32, shape=(raw_count,))
        identity_path = score_path.with_stem(score_path.stem + "_points")
        identities = np.lib.format.open_memmap(identity_path, mode="r" if identity_path.exists() else "w+",
            dtype=np.dtype([("slot", "<u4"), ("target", "i1")]), shape=(raw_count,))
        if identities.shape != (raw_count,):
            raise ValueError("evaluation point population changed")
    cursor = 0
    start = time.perf_counter()
    for number, sample in enumerate(loader, 1):
        batch = to_device(sample, device)
        with autocast(device):
            prediction = model.predict(batch, readout=readout)["score"] if readout is not None else model(batch)
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite model prediction")
        prediction = prediction.cpu().numpy()
        # Recover official three-value truth from point targets only for scoring.
        truth = np.where(sample["targets"].numpy() < 0, 0, sample["targets"].numpy() + 1)
        calculator.update(sample["xyzi"][:, :3].numpy(), prediction, truth)
        selected_scores = calculator.all_scores.pop()
        selected_labels = calculator.all_labels.pop()
        stop = cursor + len(selected_scores)
        scores[cursor:stop], labels[cursor:stop] = selected_scores, selected_labels
        if record_points:
            raw_stop = raw_cursor + len(prediction)
            raw_scores[raw_cursor:raw_stop] = prediction
            for key, source in (("slot", "slots"), ("target", "targets")):
                values = sample[source].numpy()
                if identities.flags.writeable:
                    identities[raw_cursor:raw_stop][key] = values
                elif not np.array_equal(identities[raw_cursor:raw_stop][key], values):
                    raise ValueError("evaluation point identity or label changed")
            frames.append(dict(index=int(sample["index"]), start=raw_cursor, stop=raw_stop,
                               metric_start=cursor, metric_stop=stop))
            raw_cursor = raw_stop
        cursor = stop
        if number % 50 == 0 or number == len(indices):
            elapsed = time.perf_counter() - start
            print(f"validation {number}/{len(indices)} scans, {cursor}/{count} points, "
                  f"{elapsed/60:.1f} min, remaining {(len(indices)-number)*elapsed/number/60:.1f} min", flush=True)
        if number % 250 == 0 and score_path is not None:
            from .train import disk_check
            disk_check()
    if cursor != count:
        raise ValueError("official evaluation count differs from the fixed manifest")
    if score_path is not None:
        scores.flush()
    if record_points:
        if raw_cursor != raw_count:
            raise ValueError("full-return evaluation recording omitted points")
        raw_scores.flush()
        identities.flush()
        write_json(score_path.with_name(score_path.stem + "_records.json"),
            dict(manifest_sha256=manifest["sha256"], frames=frames,
                 points=raw_count, metric_points=count, identities=identity_path.name,
                 scope="The score file is the exact metric population; its *_all.npy companion preserves every actual return in the selected eligible scans."))
        del raw_scores, identities
    del batch, sample, dataset, loader
    gc.collect()
    # Integer 0/1 labels are exact in int8; sklearn still uses its own float64 sums.
    calculator.all_scores, calculator.all_labels = [scores], [labels]
    metrics = {k: float(v) for k, v in calculator.compute_metrics().items()}
    better(metrics, None)
    return dict(metrics=metrics, scans=len(indices), points=count,
                seconds=time.perf_counter() - start, manifest_sha256=manifest["sha256"],
                official_commit=STU_COMMIT)


def rank_metrics(normal_scores, anomaly_scores):
    """Independently recompute exact pooled metrics; sort normal scores in place."""
    normal_scores, anomaly_scores = np.asarray(normal_scores), np.asarray(anomaly_scores)
    if (normal_scores.ndim != 1 or anomaly_scores.ndim != 1
            or not len(normal_scores) or not len(anomaly_scores)
            or not np.isfinite(normal_scores).all() or not np.isfinite(anomaly_scores).all()):
        raise ValueError("rank verification requires finite scores from both classes")
    normal_scores.sort(kind="quicksort")
    thresholds, positives = np.unique(anomaly_scores, return_counts=True)
    thresholds, positives = thresholds[::-1], positives[::-1]
    less = np.searchsorted(normal_scores, thresholds, side="left")
    right = np.searchsorted(normal_scores, thresholds, side="right")
    tied = right - less
    false_positives = len(normal_scores) - less
    true_positives = positives.cumsum(dtype=np.int64)
    recall = true_positives.astype(np.float64) / len(anomaly_scores)
    precision = true_positives / (true_positives + false_positives)
    ap = np.sum(positives.astype(np.float64) / len(anomaly_scores) * precision)
    # AUROC is the fraction of positive/negative pairs ordered correctly, with
    # half credit for a tie; no float32 accumulation over millions of points.
    auroc = np.dot(positives.astype(np.float64), less.astype(np.float64) + .5 * tied)
    auroc /= len(anomaly_scores) * len(normal_scores)
    # sklearn removes a ROC vertex only when adjacent positive AND negative
    # increments agree. A negative-only threshold between positive groups keeps
    # the preceding positive vertex. Pure negative vertices cannot first cross 95%.
    keep = np.ones(len(thresholds), dtype=bool)
    keep[:-1] = ((less[:-1] > right[1:]) | (positives[:-1] != positives[1:])
                | (tied[:-1] != tied[1:]))
    if thresholds[0] >= normal_scores[-1]:
        keep[0] = True  # The first vertex is retained even on a straight ROC segment.
    crossing = np.flatnonzero(keep & (recall > .95))[0]
    return dict(AP=float(100 * ap), AUROC=float(100 * auroc),
                FPR95=float(100 * (false_positives[crossing] / len(normal_scores))),
                threshold=float(thresholds[crossing]))


def recompute_metrics(score_path, manifest):
    """Verify saved identities and scores, then independently check official metrics.

    This is a post-evaluation arithmetic check, never a score-selection procedure.
    Memory is bounded by one float32 copy of the metric population plus scan slices.
    """
    score_path = Path(score_path)
    records = json.loads(score_path.with_name(score_path.stem + "_records.json").read_text())
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    raw = np.load(score_path.with_stem(score_path.stem + "_all"), mmap_mode="r", allow_pickle=False)
    identities = np.load(score_path.parent / records["identities"], mmap_mode="r", allow_pickle=False)
    indices = evaluation_indices(manifest)
    counts = {key: sum(manifest["records"][i][key] for i in indices)
              for key in ("normal", "anomaly", "points")}
    metric_count = counts["normal"] + counts["anomaly"]
    if (not counts["normal"] or not counts["anomaly"]
            or records["manifest_sha256"] != manifest["sha256"]
            or records["metric_points"] != metric_count or records["points"] != counts["points"]
            or [row["index"] for row in records["frames"]] != indices
            or scores.shape != (metric_count,) or raw.shape != (counts["points"],)
            or scores.dtype != np.dtype("float32") or raw.dtype != np.dtype("float32")
            or identities.shape != raw.shape
            or identities.dtype != np.dtype([("slot", "<u4"), ("target", "i1")])):
        raise ValueError("saved metric population does not match the official manifest")
    normal = np.empty(counts["normal"], np.float32)
    anomaly = np.empty(counts["anomaly"], np.float32)
    source = Scans(manifest)
    cursor = raw_cursor = normal_cursor = anomaly_cursor = 0
    started = time.perf_counter()
    for row in records["frames"]:
        expected = manifest["records"][row["index"]]
        raw_stop, stop = raw_cursor + expected["points"], cursor + expected["normal"] + expected["anomaly"]
        if (row["start"], row["stop"], row["metric_start"], row["metric_stop"]) != (raw_cursor, raw_stop, cursor, stop):
            raise ValueError("saved frame offsets do not preserve official point order")
        points = identities[raw_cursor:raw_stop]
        targets, slots = points["target"], points["slot"]
        original = source[row["index"]]
        if not np.array_equal(slots, original["slots"]) or not np.array_equal(targets, original["targets"]):
            raise ValueError("saved point identities or labels differ from the original scan")
        del original
        if (np.any((targets < -1) | (targets > 1)) or np.any(slots[1:] <= slots[:-1])
                or (len(slots) and slots[-1] >= expected["slots"])
                or np.count_nonzero(targets == 0) != expected["normal"]
                or np.count_nonzero(targets == 1) != expected["anomaly"]):
            raise ValueError("saved point identities or labels differ from the official population")
        chosen = targets >= 0
        values = scores[cursor:stop]
        if (not np.isfinite(raw[raw_cursor:raw_stop]).all()
                or not np.array_equal(values, raw[raw_cursor:raw_stop][chosen])):
            raise ValueError("metric scores differ from recorded original-point scores")
        selected_targets = targets[chosen]
        normal[normal_cursor:normal_cursor + expected["normal"]] = values[selected_targets == 0]
        anomaly[anomaly_cursor:anomaly_cursor + expected["anomaly"]] = values[selected_targets == 1]
        normal_cursor += expected["normal"]
        anomaly_cursor += expected["anomaly"]
        cursor, raw_cursor = stop, raw_stop
    # Close mappings before sorting; only the bounded float32 class arrays remain.
    del scores, raw, identities, points, targets, slots, values
    measured = rank_metrics(normal, anomaly)
    return dict(independent_metrics=measured, points=metric_count, normal_points=normal_cursor,
                anomaly_points=anomaly_cursor, scans=len(indices), manifest_sha256=manifest["sha256"],
                seconds=time.perf_counter() - started,
                method="exact score ranks with whole ties and the official ROC vertex-removal rule")


def comparison_conditions(checkpoints, *, exploratory=False):
    """A mechanism comparison needs independently trained, matched controls."""
    if not {"semantic", "joint"}.issubset(checkpoints) or set(checkpoints) - {"semantic", "joint", "separate"}:
        raise ValueError("comparison requires independent semantic and joint checkpoints; separate is optional")
    keys = ("data_identity", "source_mapping", "target_mapping", "initial_sha256", "seed", "source_epochs",
            "target_epochs", "batch", "queries", "source_replay_fraction", "eval_every", "target_eval_every",
            "selection", "budget")
    baseline = checkpoints["semantic"]
    differences, selections = [], {}
    for method, saved in checkpoints.items():
        config = saved.get("config", {})
        if config.get("variant") != method:
            raise ValueError(f"{method} must be its independently trained variant, not an inference switch")
        if not saved.get("frozen"):
            differences.append(f"{method}: the normal training budget is incomplete")
        if config.get("synthetic_anomalies") is not False:
            differences.append(f"{method}: normal-only training is not established")
        for key in keys:
            if key not in config or key not in baseline.get("config", {}):
                differences.append(f"{method}: missing config.{key}")
            elif config[key] != baseline["config"][key]:
                differences.append(f"{method}: config.{key} differs from semantic")
        selections[method] = {}
        for stage in ("source", "target"):
            actual = saved.get("stages", {}).get(stage, {})
            reference = baseline.get("stages", {}).get(stage, {})
            for key in ("trained_frames", "trained_updates", "planned_frames", "planned_updates", "budget_complete", "selected_update"):
                if key not in actual or key not in reference:
                    differences.append(f"{method}: missing stages.{stage}.{key}")
                elif key != "selected_update" and actual[key] != reference[key]:
                    differences.append(f"{method}: actual {stage} {key} differs from semantic")
            if actual.get("budget_complete") is not True:
                differences.append(f"{method}: {stage} did not complete its planned training budget")
            selections[method][stage] = actual.get("selected_update")
    if differences and not exploratory:
        raise ValueError("unmatched scientific comparison: " + "; ".join(differences)
                         + "; --exploratory reports these differences without a matched-mechanism claim")
    return dict(matched=not differences, differences=differences, selected_updates=selections,
                selection_scope="same candidate training budget and normal selection rule; selected updates may differ",
                training_link_ablation_available="separate" in checkpoints,
                role="matched mechanism comparison" if not differences else "exploratory unmatched comparison")


def official_population(xyz, targets):
    """Use official selection itself, including its per-scan five-anomaly rule."""
    calculator = PointOODMetricsCalculator()
    truth = np.where(targets < 0, 0, targets + 1)
    calculator.update(xyz, np.arange(len(targets), dtype=np.int64), truth)
    if not calculator.all_scores:
        raise ValueError("selected comparison scan fails the official five-anomaly rule")
    return calculator.all_scores[0], calculator.all_labels[0].astype(np.int8)


def normal_threshold(scores, fpr, *, presorted=False):
    """Use only normal scores; whole ties never exceed the requested budget."""
    ordered = np.asarray(scores) if presorted else np.sort(np.asarray(scores))
    if (ordered.ndim != 1 or not len(ordered) or not np.isfinite(ordered[[0, -1]]).all()
            or not 0 <= fpr <= 1):
        raise ValueError("a threshold requires finite normal scores and an FPR in [0, 1]")
    budget = int(np.floor(float(fpr) * len(ordered)))
    return (float(ordered[len(ordered) - budget - 1]) if budget < len(ordered)
            else float(np.nextafter(np.float64(ordered[0]), -np.inf)))


def comparison_metrics(labels, predictions, *, fprs=(.01, .05), confidence=.9):
    """Pool exact ranks and paired decisions using one sortable normal copy."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("comparison needs a nonempty shared population of official normal and unknown points")
    if not 0 <= confidence <= 1 or not fprs or any(not 0 <= value <= 1 for value in fprs):
        raise ValueError("confidence and normal false-positive rates must be in [0, 1]")
    if not {"semantic", "joint"}.issubset(predictions):
        raise ValueError("comparison needs semantic and joint point predictions")
    normal = labels == 0
    unknown_indices = np.flatnonzero(~normal)
    normal_count, unknown_count = int(normal.sum()), len(unknown_indices)
    if not normal_count or not unknown_count:
        raise ValueError("comparison needs a nonempty shared population of official normal and unknown points")
    chunk_size = 1_048_576
    for name, values in predictions.items():
        for key in ("score", "raw_score", "confidence", "semantic"):
            if np.shape(values[key]) != labels.shape:
                raise ValueError(f"{name}.{key} does not preserve the shared point population")
        if not np.issubdtype(values["semantic"].dtype, np.integer):
            raise ValueError("class predictions must be integer normal IDs 0–18")
    # This subset is defined once by the independent semantic baseline, not by a
    # competing model's confidence or by whether its anomaly threshold rejected it.
    confident = predictions["semantic"]["confidence"][unknown_indices] >= confidence
    confident_count = int(confident.sum())
    methods = {}
    # Establish reference thresholds first, then release each sorted copy before
    # processing the next method. The large input arrays remain read-only maps.
    for name in ("semantic", *(key for key in predictions if key != "semantic")):
        values = predictions[name]
        for start in range(0, len(labels), chunk_size):
            stop = start + chunk_size
            for key in ("score", "raw_score", "confidence", "semantic"):
                if not np.isfinite(values[key][start:stop]).all():
                    raise ValueError(f"{name}.{key} contains nonfinite values")
            probability, classes = values["confidence"][start:stop], values["semantic"][start:stop]
            if ((probability < 0) | (probability > 1)).any():
                raise ValueError("class confidence must be a normalized class probability")
            if ((classes < 0) | (classes >= 19)).any():
                raise ValueError("class predictions must be integer normal IDs 0–18")
        ordered = values["score"][normal]
        anomaly = values["score"][unknown_indices]
        metrics = rank_metrics(ordered, anomaly)
        better(metrics, None)
        operating = []
        for requested in fprs:
            # Reject score > threshold: tied normal scores are accepted or rejected
            # together, never interpolated or split by point order.
            threshold = normal_threshold(ordered, requested, presorted=True)
            rejected = anomaly > np.float64(threshold)
            false_positives = normal_count - int(np.searchsorted(ordered, threshold, side="right"))
            recalled, confident_recalled = int(rejected.sum()), int(np.count_nonzero(rejected & confident))
            operating.append(dict(requested_fpr=float(requested), threshold=threshold, comparison="score > threshold",
                false_positives=false_positives, actual_fpr=false_positives / normal_count,
                unknown_recalled=recalled, unknown_recall=recalled / unknown_count,
                confident_unknown_recalled=confident_recalled,
                confident_unknown_recall=confident_recalled / confident_count if confident_count else None))
        methods[name] = dict(metrics=metrics, operating_points=operating)
        del ordered
        for row, baseline in zip(methods[name]["operating_points"], methods["semantic"]["operating_points"], strict=True):
            rejected = anomaly > np.float64(row["threshold"])
            base_rejected = predictions["semantic"]["score"][unknown_indices] > np.float64(baseline["threshold"])
            rescued, lost = rejected & ~base_rejected, ~rejected & base_rejected
            row.update(unknown_rescued=int(rescued.sum()), unknown_lost=int(lost.sum()),
                       confident_unknown_rescued=int(np.count_nonzero(rescued & confident)),
                       confident_unknown_lost=int(np.count_nonzero(lost & confident)),
                       normal_new_false_positives=0, normal_removed_false_positives=0,
                       equal_actual_fpr=row["false_positives"] == baseline["false_positives"])
            for prefix, count in (("unknown", unknown_count), ("confident_unknown", confident_count)):
                row[prefix + "_recovered_fraction"] = row[prefix + "_rescued"] / count if count else None
                row[prefix + "_lost_fraction"] = row[prefix + "_lost"] / count if count else None
                row[prefix + "_net_recall_change"] = ((row[prefix + "_rescued"] - row[prefix + "_lost"]) / count
                                                       if count else None)
        if name != "semantic":
            # Reuse each disk-backed chunk across every operating point.
            for start in range(0, len(labels), chunk_size):
                stop = start + chunk_size
                for row, baseline in zip(operating, methods["semantic"]["operating_points"], strict=True):
                    candidate = values["score"][start:stop] > np.float64(row["threshold"])
                    reference = predictions["semantic"]["score"][start:stop] > np.float64(baseline["threshold"])
                    row["normal_new_false_positives"] += int(np.count_nonzero(normal[start:stop] & candidate & ~reference))
                    row["normal_removed_false_positives"] += int(np.count_nonzero(normal[start:stop] & ~candidate & reference))
    return dict(methods=methods, normal_points=normal_count, unknown_points=unknown_count,
                confident_unknown_points=confident_count, baseline_confidence_threshold=float(confidence),
                confidence_definition="maximum softmax class probability of the independently trained semantic model; this is not a calibrated correctness probability",
                threshold_role="operating points on this evaluation population's normal-point curve, not deployment thresholds",
                tie_rule="each model uses its highest attainable normal false-positive count within the common requested budget; whole ties stay together; actual FPR is reported")


def instance_coverage(identities, frames, predictions, methods):
    """Give each scan-instance pair one vote after the official point mask."""
    edges = np.array([5, 10, 20, 50, 100])
    counts = np.zeros(6, np.int64)
    detected = {name: np.zeros((len(methods[name]["operating_points"]), 6), np.int64) for name in predictions}
    unassigned = 0
    for frame in frames:
        raw = identities[frame["start"]:frame["stop"]]
        points = raw[raw["official"]]
        if len(points) != frame["metric_stop"] - frame["metric_start"]:
            raise ValueError("instance identities do not match the shared official point order")
        unknown = points["target"] == 1
        unassigned += int(np.count_nonzero(unknown & (points["instance"] <= 0)))
        chosen = unknown & (points["instance"] > 0)
        _, groups, sizes = np.unique(points["instance"][chosen], return_inverse=True, return_counts=True)
        bins = np.searchsorted(edges, sizes, side="right")
        counts += np.bincount(bins, minlength=6)
        for name, values in predictions.items():
            scores = values["score"][frame["metric_start"]:frame["metric_stop"]][chosen]
            for index, operating in enumerate(methods[name]["operating_points"]):
                hits = np.bincount(groups, weights=scores > np.float64(operating["threshold"]), minlength=len(sizes))
                # Half of an odd-sized instance means ceil(size / 2) detections.
                detected[name][index] += np.bincount(bins[2 * hits >= sizes], minlength=6)
    result = dict(size_bins=["1-4", "5-9", "10-19", "20-49", "50-99", "100+"],
                  instance_counts=counts.tolist(), unassigned_anomaly_points=unassigned,
                  definition="equal weight per scan-instance pair; at least half of post-mask points score above the threshold; unassigned instance IDs are excluded",
                  methods={})
    for name, totals in detected.items():
        result["methods"][name] = [dict(requested_fpr=row["requested_fpr"], actual_fpr=row["actual_fpr"],
            detected=hits.tolist(), recall=[int(hits[i]) / int(counts[i]) if counts[i] else None for i in range(6)])
            for row, hits in zip(methods[name]["operating_points"], totals, strict=True)]
    return result


def predictive_summary(statistics):
    """Reduce additive diagnostics over queries, never over class means."""
    count = int(np.asarray(statistics.get("prediction_count", 0)).sum())
    finite = int(np.asarray(statistics.get("finite_interval_count", 0)).sum())
    if finite != count:
        raise ValueError("nonfinite predictive intervals cannot be omitted from the diagnostic population")
    result = dict(queries=count,
        nll=float(np.asarray(statistics["nll_sum"]).sum() / count) if count else None,
        coverage90=float(np.asarray(statistics["coverage90_count"]).sum() / count) if count else None,
        width90_m=float(np.asarray(statistics["width90_m_sum"]).sum() / count) if count else None,
        median_mae_m=float(np.asarray(statistics["abs_median_error_m_sum"]).sum() / count) if count else None,
        population="all fine-labeled, context-supported queries within the declared diagnostic query population")
    if "range_prediction_count" in statistics:
        result["by_range"] = []
        for index, interval in enumerate(("[2.5,10)", "[10,20)", "[20,35)", "[35,50]")):
            stratum = {key: np.asarray(statistics["range_" + key])[index] for key in
                       ("prediction_count", "finite_interval_count", "nll_sum", "coverage90_count",
                        "width90_m_sum", "abs_median_error_m_sum")}
            summary = predictive_summary(stratum)
            result["by_range"].append(dict(range_m=interval, query_share=summary["queries"] / count if count else None,
                                           **summary))
    return result


def aggregate_results(results):
    """Average independent seed-level pooled metrics with sample deviations."""
    if len(results) != 3 or {row.get("seed") for row in results} != {206, 307, 409}:
        raise ValueError("paper aggregation requires exactly one result for each seed 206, 307 and 409")
    keys = ("variant", "readout", "split", "manifest_sha256", "architecture", "score_version")
    for key in keys:
        if any(key not in row or row[key] != results[0][key] for row in results):
            raise ValueError(f"seed aggregation requires the same explicit {key}")
    metrics = {}
    for key in ("AP", "AUROC", "FPR95"):
        values = np.asarray([row["metrics"][key] for row in results], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("seed aggregation requires finite pooled metrics")
        metrics[key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
                            per_seed={str(row["seed"]): float(value) for row, value in zip(results, values, strict=True)})
    normal = [row.get("normal201") for row in results]
    if any(value is not None for value in normal):
        if any(value is None or "mean_iou_gt" not in value for value in normal):
            raise ValueError("normal mIoU must be present for all three seeds or none")
        values = np.asarray([100 * value["mean_iou_gt"] for value in normal], dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
            raise ValueError("normal mIoU must be a finite fraction in [0, 1]")
        metrics["normal_mIoU"] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
            per_seed={str(row["seed"]): float(value) for row, value in zip(results, values, strict=True)})
    return dict(**{key: results[0][key] for key in keys}, metrics=metrics, seeds=[206, 307, 409],
                definition="arithmetic mean and sample standard deviation of the three independently pooled seed metrics, in percentage points")


@torch.no_grad()
def compare(checkpoints, manifest, output, device, *, workers=4, fprs=(.01, .05), confidence=.9,
            exploratory=False, fixed_readouts=False, retain_fixed_records=True):
    """Prepare each scan once and execute models sequentially without retaining activations."""
    from .train import disk_check, runtime_snapshot
    if manifest["kind"] not in ("val", "test"):
        raise ValueError("comparison needs a held-out validation or test manifest")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("comparison output must be empty; existing point evidence is never overwritten")
    indices = evaluation_indices(manifest)
    count = sum(manifest["records"][i]["normal"] + manifest["records"][i]["anomaly"] for i in indices)
    returns = sum(manifest["records"][i]["points"] for i in indices)
    if not count:
        raise ValueError("comparison contains no eligible evaluation points")
    if not 0 <= confidence <= 1 or not fprs or any(not 0 <= value <= 1 for value in fprs):
        raise ValueError("confidence and normal false-positive rates must be in [0, 1]")
    record_type = np.dtype([("score", "<f4"), ("raw_score", "<f4"), ("confidence", "<f4"), ("semantic", "<i2")])
    identity_type = np.dtype([("frame", "<u4"), ("slot", "<u4"), ("target", "i1"), ("official", "?"), ("instance", "<i4")])
    readouts = ("appearance", "common_density", "independent_minima") if fixed_readouts else ()
    method_names = [*checkpoints, *("joint_" + name for name in readouts)]
    peak_write = count * (record_type.itemsize * len(method_names) + 1) + returns * identity_type.itemsize + 1_000_000
    resources = runtime_snapshot()
    disk_check(peak_write)
    models, saved = {}, {}
    for name, path in checkpoints.items():
        model, checkpoint = load_model(path, torch.device("cpu"))
        if model.mode != "normal_hypothesis":
            raise ValueError("mechanism comparison only accepts the current normal-evidence variants")
        models[name] = model
        saved[name] = {key: checkpoint[key] for key in ("config", "stages", "frozen", "normal201") if key in checkpoint}
        del checkpoint
    conditions = comparison_conditions(saved, exploratory=exploratory)
    weight_bytes = sum(t.numel() * t.element_size() for model in models.values()
                       for t in (*model.parameters(), *model.buffers()))
    # Weights are small relative to sparse-backbone activations. Keep them resident
    # when memory permits, but never retain outputs from more than one full scan.
    resident = device.type != "cuda" or torch.cuda.mem_get_info(device)[0] > weight_bytes + 8_000_000_000
    if resident:
        for model in models.values():
            model.to(device)
    # One sorted float32 normal copy, bounded paired chunks, and two active
    # disk-backed method records; anomaly rank work scales with anomalies only.
    anomaly_count = sum(manifest["records"][i]["anomaly"] for i in indices)
    required = 40 * count + 160 * anomaly_count + 1_000_000_000
    if memory_available() < required:
        raise RuntimeError(f"exact paired metrics need about {required / 1e9:.1f} GB free RAM after loading models")
    output.mkdir(parents=True, exist_ok=True)
    records = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=record_type, shape=(count,))
               for name in method_names}
    identities = np.lib.format.open_memmap(output / "returns.npy", mode="w+", dtype=identity_type, shape=(returns,))
    labels = np.lib.format.open_memmap(output / "labels.npy", mode="w+", dtype=np.int8, shape=(count,))
    dataset = PreparedScans(manifest)
    loader = DataLoader(dataset, batch_size=None, sampler=indices, num_workers=workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0),
                        **({"prefetch_factor": 1} if workers else {}))
    frames, cursor, raw_cursor = [], 0, 0
    started = time.perf_counter()
    inference_seconds = {name: 0. for name in models}
    for number, sample in enumerate(loader, 1):
        chosen, targets = official_population(sample["xyzi"][:, :3].numpy(), sample["targets"].numpy())
        stop, raw_stop = cursor + len(chosen), raw_cursor + len(sample["xyzi"])
        row = identities[raw_cursor:raw_stop]
        row["frame"], row["slot"], row["target"] = int(sample["index"]), sample["slots"].numpy(), sample["targets"].numpy()
        row["instance"] = sample["instance"].numpy() if "instance" in sample else -1
        row["official"] = False
        row["official"][chosen] = True
        labels[cursor:stop] = targets
        batch = to_device(sample, device)
        for name, model in models.items():
            if not resident:
                model.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with autocast(device):
                if name == "joint" and fixed_readouts:
                    outputs = model.predict_readouts(batch)
                    predictions = {name: outputs["joint"], **{"joint_" + key: outputs[key] for key in readouts}}
                else:
                    predictions = {name: model.predict(batch)}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds[name] += time.perf_counter() - tick
            for method, prediction in predictions.items():
                for key in record_type.names:
                    values = prediction[key].detach().cpu().numpy()
                    if values.shape != (len(sample["xyzi"]),) or not np.isfinite(values).all():
                        raise ValueError(f"{method}.{key} changed the point population or contains nonfinite values")
                    if key == "semantic" and (not np.issubdtype(values.dtype, np.integer) or not np.isin(values, np.arange(19)).all()):
                        raise ValueError("normal semantic predictions must be integer IDs 0–18 before recording")
                    if key != "semantic" and values.dtype != np.float32:
                        raise ValueError(f"{method}.{key} must retain the model's exact float32 output precision")
                    records[method][key][cursor:stop] = values[chosen]
            del predictions, prediction
            if name == "joint" and fixed_readouts:
                del outputs
            if not resident:
                model.to("cpu")
        frames.append(dict(index=int(sample["index"]), scan=manifest["records"][int(sample["index"])]["scan"],
                           start=raw_cursor, stop=raw_stop, metric_start=cursor, metric_stop=stop))
        cursor, raw_cursor = stop, raw_stop
        del batch, sample
        if number % 25 == 0 or number == len(indices):
            print(f"comparison {number}/{len(indices)} scans, {cursor}/{count} official points", flush=True)
            disk_check()
    if cursor != count or raw_cursor != returns:
        raise ValueError("comparison point population differs from the official manifest")
    for array in (*records.values(), identities, labels):
        array.flush()
    del model, models, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    measured = comparison_metrics(labels, records, fprs=fprs, confidence=confidence)
    measured["instance_coverage"] = instance_coverage(identities, frames, records, measured["methods"])
    omitted = []
    if not retain_fixed_records:
        for readout in readouts:
            name = "joint_" + readout
            records.pop(name)._mmap.close()
            (output / (name + ".npy")).unlink()
            omitted.append(name)
    result = dict(**measured, conditions=conditions, frames=frames, scans=len(indices), points=count,
                  seed=saved["semantic"]["config"]["seed"], split=manifest["kind"],
                  fixed_readouts=list(readouts),
                  retained_point_records=["returns.npy", "labels.npy", *(name + ".npy" for name in records)],
                  omitted_reproducible_readouts=omitted,
                  checkpoints={name: str(Path(path).resolve()) for name, path in checkpoints.items()},
                  normal201={name: row.get("normal201") for name, row in saved.items()},
                  manifest_sha256=manifest["sha256"], official_commit=STU_COMMIT, resources=resources,
                  execution=dict(weight_bytes=weight_bytes, weights_resident=resident,
                                 metric_memory_budget_bytes=required,
                                 forwards="sequential per prepared full scan; each model's activations are released before the next"),
                  inference_seconds=inference_seconds, seconds=time.perf_counter() - started,
                  inference_timing_scope="synchronized model forward only, including requested fixed readouts; use benchmark for preparation, transfer and original-slot output timing",
                  records="returns.npy stores every actual return with original slot, manifest frame index, target and official mask; filtering it by official gives exactly the common row order of labels.npy and all method *.npy files",
                  target_definition="-1 ignored, 0 normal, 1 unknown; empty original slots have no actual return and are absent")
    write_json(output / "comparison.json", result)
    return result


def normal_decision_counts(truth, predictions):
    """Paired semantic changes use the independent appearance model as reference."""
    truth = np.asarray(truth)
    if truth.ndim != 1 or not len(truth) or not np.issubdtype(truth.dtype, np.integer) or not np.isin(truth, np.arange(19)).all():
        raise ValueError("normal paired decisions require fine labels in classes 0-18")
    if not {"semantic", "joint"}.issubset(predictions):
        raise ValueError("normal paired decisions require independently trained appearance and joint models")
    for name, predicted in predictions.items():
        if (np.shape(predicted) != truth.shape or not np.issubdtype(predicted.dtype, np.integer)
                or not np.isin(predicted, np.arange(19)).all()):
            raise ValueError(f"{name} changed the normal point identities or class vocabulary")
    baseline = predictions["semantic"] == truth
    result = {}
    for name, predicted in predictions.items():
        correct = predicted == truth
        result[name] = dict(confusion=np.bincount(truth * 19 + predicted, minlength=361).reshape(19, 19),
                            corrected=int(np.count_nonzero(correct & ~baseline)),
                            introduced=int(np.count_nonzero(~correct & baseline)))
    return result


@torch.no_grad()
def compare_normal(checkpoints, records, device, *, workers=4, fixed_readouts=False, exploratory=False):
    """Pool sequence-201 confusion and independent-model corrections in scan order."""
    if not records or any(row.get("source") != "normal_stu" or row.get("scene") != "201" for row in records):
        raise ValueError("normal paired decisions use only sequence 201 development records")
    models, saved = {}, {}
    for name, path in checkpoints.items():
        model, state = load_model(path, device)
        models[name], saved[name] = model, {key: state[key] for key in ("config", "stages", "frozen") if key in state}
        del state
    conditions = comparison_conditions(saved, exploratory=exploratory)
    loader = DataLoader(NormalScans(records), batch_size=None, num_workers=workers,
                        generator=torch.Generator().manual_seed(0), pin_memory=device.type == "cuda",
                        **({"prefetch_factor": 1} if workers else {}))
    totals, frames = {}, []
    for sample in loader:
        single = sample["allowed"].sum(-1) == 1
        if not bool(single.any()):
            continue
        truth = sample["allowed"][single].long().argmax(-1).numpy()
        batch = to_device(sample, device)
        predictions = {}
        for name, model in models.items():
            with autocast(device):
                outputs = model.predict_readouts(batch) if name == "joint" and fixed_readouts else {name: model.predict(batch)}
            for readout, values in outputs.items():
                key = name if readout == name else name + "_" + readout
                predictions[key] = values["semantic"].cpu().numpy()[single.numpy()]
            del outputs
        for name, result in normal_decision_counts(truth, predictions).items():
            if name not in totals:
                totals[name] = dict(confusion=np.zeros((19, 19), np.int64), corrected=0, introduced=0)
            for key, value in result.items():
                totals[name][key] += value
        frames.append(dict(index=int(sample["index"]), points=len(truth)))
    result = {}
    for name, counts in totals.items():
        measured = normal_semantic_metrics(counts["confusion"])
        result[name] = dict(**measured, corrected=counts["corrected"], introduced=counts["introduced"],
            corrected_fraction=counts["corrected"] / measured["points"],
            introduced_fraction=counts["introduced"] / measured["points"],
            net_accuracy_change=(counts["corrected"] - counts["introduced"]) / measured["points"])
    if not result:
        raise ValueError("normal paired decisions have no fine-labeled points")
    return dict(methods=result, frames=frames, conditions=conditions,
                seed=saved["semantic"]["config"]["seed"],
                population="same fine-labeled original return identities in sequence 201; confusion pools points before averaging ground-truth-present class IoU")


@torch.no_grad()
def infer(model, scan, device, *, return_semantics=False, readout=None, record=None, partition="val"):
    if getattr(model, "mode", None) != "normal_hypothesis":
        raise ValueError("inference requires a SERVE normal perception model")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    read_started = time.perf_counter()
    io_timing = {}
    frame = read_scan(scan, io_timing=io_timing, partition=partition,
                      expected=(record["scan_sha256"], None) if record is not None else None)
    if record is not None and (len(frame.xyzi) != record["slots"] or int(frame.actual.sum()) != record["points"]):
        raise ValueError("inference point identities differ from the scan manifest")
    read_seconds = io_timing["seconds"]
    # Include decode and original-slot construction, while removing actual file I/O.
    decode_seconds = time.perf_counter() - read_started - read_seconds
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    semantic = np.full(len(frame.xyzi), -1, np.int16) if return_semantics else None
    if frame.actual.any():
        xyzi = frame.xyzi[frame.actual].copy()
        sample = prepare_scan(dict(xyzi=xyzi, slots=frame.return_slots,
                                   targets=np.full(int(frame.actual.sum()), -1, np.int8),
                                   slot_count=len(frame.xyzi), index=frame.frame_id))
        sample["observation"] = hypothesis_observation(xyzi)
        sample = to_device(sample, device)
        with autocast(device):
            if return_semantics or readout is not None:
                outputs = model.predict(sample, readout=readout) if readout is not None else model.predict(sample)
                prediction, classes = outputs["score"], outputs["semantic"]
                if classes.shape != prediction.shape or classes.is_floating_point() or bool(((classes < 0) | (classes >= 19)).any()):
                    raise ValueError("semantic outputs must follow the model's class vocabulary in original return order")
                if return_semantics:
                    semantic[frame.return_slots] = classes.to(torch.int16).cpu().numpy()
            else:
                prediction = model(sample)
        prediction = scatter_scores(prediction, sample["slots"], sample["slot_count"])
        if not torch.isfinite(prediction).all():
            raise ValueError("nonfinite inference output")
        prediction = prediction.cpu().numpy()
    else:
        prediction = np.zeros(len(frame.xyzi), np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if return_semantics:
        prediction = dict(score=prediction, semantic=semantic)
    return prediction, dict(read_seconds=read_seconds, seconds=time.perf_counter() - start + decode_seconds,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                            real_points=int(frame.actual.sum()), slots=len(frame.xyzi))


def benchmark(model, scans, device, *, warmup=50, readout=None):
    """Time every scan once after 50 warmups; disk loading stays outside timing."""
    if not scans or warmup < 0:
        raise ValueError("benchmark requires scans and a nonnegative warmup count")
    model.eval()
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("the runtime protocol requires FP32 model parameters")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(warmup):
        infer(model, scans[index % len(scans)], device, readout=readout)
    timings = [infer(model, scan, device, readout=readout)[1] for scan in scans]
    latency = np.asarray([row["seconds"] for row in timings]) * 1000
    return dict(batch_size=1, precision="FP32", warmup_scans=warmup, measured_scans=len(scans),
        scans=[str(path) for path in scans], parameters=sum(p.numel() for p in model.parameters()),
        mean_ms=float(latency.mean()), p95_ms=float(np.percentile(latency, 95)),
        mean_read_ms=float(np.mean([row["read_seconds"] for row in timings]) * 1000),
        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        timing_scope="scan preparation, host-to-device transfer, encoding, prediction, scoring and original-slot output construction; excludes disk read and global metrics",
        memory_scope="full pass peak allocated CUDA bytes after model loading, including parameters and warmup")


def summarize_experiments(directory):
    """Assemble paper tables from complete per-seed measurements, never pooled seeds."""
    from .data import NORMAL_CLASSES
    from .model import NORMAL_VARIANTS
    directory = Path(directory)
    seeds = (206, 307, 409)
    sources = []

    def read(relative):
        path = directory / relative
        value = json.loads(path.read_text())
        sources.append(str(relative))
        return value

    def summary(values, scale=1., *, counts=None):
        supplied = np.asarray(values, dtype=object)
        missing = np.equal(supplied, None)
        if counts is not None:
            counts = np.asarray(counts)
            if (counts.shape != supplied.shape or not np.isfinite(counts).all() or (counts < 0).any()
                    or not np.array_equal(missing, counts == 0)):
                raise ValueError("undefined statistics must correspond exactly to empty seed-level populations")
        elif missing.any():
            raise ValueError("a required measurement is missing")
        values = np.asarray(np.where(missing, 0., supplied), dtype=np.float64) * scale
        if values.shape[0] != 3 or not np.isfinite(values).all():
            raise ValueError("every reported statistic requires three finite seed-level measurements")
        # Missing seed populations invalidate the three-seed aggregate, not the other measurements.
        undefined = missing.any(0)
        result = dict(mean=np.where(undefined, None, values.mean(0)).tolist(),
            std=np.where(undefined, None, values.std(0, ddof=1)).tolist(),
            per_seed={str(seed): np.where(mask, None, value).tolist()
                      for seed, value, mask in zip(seeds, values, missing, strict=True)})
        if missing.any():
            result["undefined_reason"] = "At least one seed has an empty population; its value and the three-seed mean and standard deviation remain undefined. Available seeds are not averaged separately."
        return result

    def metric_summary(rows):
        return {key: summary([row[key] for row in rows]) for key in ("AP", "AUROC", "FPR95")}

    normal, validation = {}, {}
    population = None
    for variant in NORMAL_VARIANTS:
        normal[variant] = [read(f"{variant}-{seed}/normal201.json") for seed in seeds]
        for readout in (("energy", "softmax") if variant == "standard" else ("joint",)):
            method = f"standard_{readout}" if variant == "standard" else variant
            filename = f"val_{readout}.json" if variant == "standard" else "val.json"
            rows = [read(f"{variant}-{seed}/{filename}") for seed in seeds]
            for seed, row in zip(seeds, rows, strict=True):
                if (row.get("seed"), row.get("variant"), row.get("readout"), row.get("split")) != (seed, variant, readout, "val"):
                    raise ValueError(f"{method} validation does not match its seed, model and readout")
                if row.get("complete") is not True:
                    raise ValueError(f"{method} validation did not follow a complete training budget")
                current = {key: row[key] for key in ("manifest_sha256", "architecture", "score_version", "points", "scans")}
                if population is None:
                    population = current
                elif current != population:
                    raise ValueError("validation results do not share the same population and model specification")
            aggregate_results(rows)
            validation[method] = rows

    paired = [read(f"paired-{seed}/comparison.json") for seed in seeds]
    paired_normal = [read(f"paired-{seed}/normal.json") for seed in seeds]
    for seed, unknown, known in zip(seeds, paired, paired_normal, strict=True):
        for row in (unknown, known):
            conditions = row.get("conditions", {})
            if row.get("seed") != seed or conditions.get("matched") is not True or conditions.get("differences"):
                raise ValueError("paired measurements require correctly ordered seeds and matched completed training")
        if (unknown.get("manifest_sha256") != population["manifest_sha256"] or unknown.get("split") != "val"
                or unknown.get("points") != population["points"] or unknown.get("scans") != population["scans"]
                or unknown.get("baseline_confidence_threshold") != .9):
            raise ValueError("paired anomaly measurements differ from the declared validation population")

    main = {}
    missing_test = []
    test_population = None
    for method in ("standard_softmax", "standard_energy", "semantic", "cssr", "separate", "joint"):
        variant, readout = (method.split("_", 1) if method.startswith("standard_") else (method, "joint"))
        paths = [f"{variant}-{seed}/test_{readout}.json" if variant == "standard" else f"{variant}-{seed}/test.json"
                 for seed in seeds]
        missing = [path for path in paths if not (directory / path).is_file()]
        test = None
        if missing:
            missing_test.append(dict(method=method, missing_files=missing,
                reason="Three independently evaluated test seeds are not available; no test metric is inferred from validation."))
        else:
            rows = [read(path) for path in paths]
            for seed, row in zip(seeds, rows, strict=True):
                if (row.get("seed"), row.get("variant"), row.get("readout"), row.get("split")) != (seed, variant, readout, "test"):
                    raise ValueError(f"{method} test does not match its seed, model and readout")
                if row.get("complete") is not True:
                    raise ValueError("test measurements require completed training budgets")
                current = {key: row[key] for key in ("manifest_sha256", "architecture", "score_version", "points", "scans")}
                if test_population is None:
                    test_population = current
                elif current != test_population:
                    raise ValueError("test methods do not share the same evaluation population")
            aggregate_results(rows)
            test = metric_summary([row["metrics"] for row in rows])
        main[method] = dict(normal_mIoU=summary([row["mean_iou_gt"] for row in normal[variant]], 100),
                            validation=metric_summary([row["metrics"] for row in validation[method]]), test=test)

    ablations = {variant: dict(type="retrained", normal_mIoU=summary([row["mean_iou_gt"] for row in normal[variant]], 100),
        validation=metric_summary([row["metrics"] for row in validation[variant]]))
        for variant in ("joint", "separate", "target_available", "single_component", "no_nll", "no_compactness")}
    fixed = ("joint_appearance", "joint_common_density", "joint_independent_minima")
    for method in fixed:
        ablations[method] = dict(type="fixed_weights",
            normal_mIoU=summary([row["methods"][method]["mean_iou_gt"] for row in paired_normal], 100),
            validation=metric_summary([row["methods"][method]["metrics"] for row in paired]))
    for index, seed in enumerate(seeds):
        for method in ("semantic", "joint", "separate"):
            for metric in ("AP", "AUROC", "FPR95"):
                if not np.isclose(paired[index]["methods"][method]["metrics"][metric],
                                  validation[method][index]["metrics"][metric], rtol=1e-10, atol=1e-9):
                    raise ValueError(f"{method}, seed {seed}: paired and standalone validation scores disagree")
        for method in ("semantic", "joint"):
            if not np.isclose(paired_normal[index]["methods"][method]["mean_iou_gt"],
                              normal[method][index]["mean_iou_gt"], rtol=0, atol=1e-12):
                raise ValueError("paired and standalone normal semantic decisions disagree")
        known = paired_normal[index]["methods"]
        if (known["joint_appearance"]["mean_iou_gt"] != known["joint_common_density"]["mean_iou_gt"]
                or known["joint_independent_minima"]["mean_iou_gt"] != known["joint"]["mean_iou_gt"]):
            raise ValueError("fixed readouts changed semantic labels that the paper requires them to preserve")

    present = (0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18)
    truth = np.asarray(normal["joint"][0]["ground_truth_points"])
    if tuple(np.flatnonzero(truth)) != present:
        raise ValueError("sequence 201 does not contain the paper's fourteen ground-truth classes")
    for rows in normal.values():
        for row in rows:
            if not np.array_equal(row["ground_truth_points"], truth):
                raise ValueError("normal semantic runs do not share the same ground-truth point population")
    per_class = []
    for category in present:
        appearance = np.asarray([row["iou"][category] for row in normal["semantic"]])
        joint = np.asarray([row["iou"][category] for row in normal["joint"]])
        per_class.append(dict(class_id=category, name=NORMAL_CLASSES[category], appearance=summary(appearance, 100),
                              joint=summary(joint, 100), difference=summary(joint - appearance, 100)))

    fprs = (.001, .005, .01, .02, .05)
    paired_methods = tuple(paired[0]["methods"])
    confidence, instances = {}, {}
    bins = ["1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    counts = paired[0]["instance_coverage"]["instance_counts"]
    unassigned = paired[0]["instance_coverage"]["unassigned_anomaly_points"]
    for row in paired:
        if set(row["methods"]) != set(paired_methods):
            raise ValueError("paired seeds must contain the same independent models and fixed readouts")
        coverage = row["instance_coverage"]
        if (coverage["size_bins"] != bins or coverage["instance_counts"] != counts
                or coverage["unassigned_anomaly_points"] != unassigned):
            raise ValueError("paired seeds use different scan-instance populations")
    for method in paired_methods:
        for row in paired:
            for operating in (row["methods"][method]["operating_points"], row["instance_coverage"]["methods"][method]):
                if tuple(point["requested_fpr"] for point in operating) != fprs:
                    raise ValueError("paired operating-point order must be 0.1%, 0.5%, 1%, 2%, 5%")
        confidence[method], instances[method] = [], []
        for index, fpr in enumerate(fprs):
            rows = [row["methods"][method]["operating_points"][index] for row in paired]
            confidence[method].append(dict(requested_fpr=fpr, **{key: summary([row[key] for row in rows], 100,
                counts=[row["confident_unknown_points"] for row in paired] if key != "actual_fpr" else None)
                for key in ("actual_fpr", "confident_unknown_recall", "confident_unknown_recovered_fraction",
                            "confident_unknown_lost_fraction", "confident_unknown_net_recall_change")}))
            rows = [row["instance_coverage"]["methods"][method][index] for row in paired]
            instances[method].append(dict(requested_fpr=fpr, actual_fpr=summary([row["actual_fpr"] for row in rows], 100),
                                          recall=summary([row["recall"] for row in rows], 100, counts=[counts] * 3)))
    corrections = {method: {key: summary([row["methods"][method][key] for row in paired_normal], 100)
        for key in ("corrected_fraction", "introduced_fraction", "net_accuracy_change")}
        for method in paired_normal[0]["methods"]}

    predictive = {}
    for method in ("joint", "separate", "target_available"):
        rows = [row["predictive"] for row in normal[method]]
        predictive[method] = {key: summary([row[key] for row in rows], 100 if key == "coverage90" else 1,
                                         counts=[row["queries"] for row in rows])
                             for key in ("nll", "coverage90", "width90_m", "median_mae_m")}
        predictive[method]["queries_per_seed"] = {str(seed): row["queries"] for seed, row in zip(seeds, rows, strict=True)}
        if method == "joint":
            predictive[method]["by_range"] = []
            for index, interval in enumerate(("[2.5,10)", "[10,20)", "[20,35)", "[35,50]")):
                strata = [row["by_range"][index] for row in rows]
                if any(row["range_m"] != interval for row in strata):
                    raise ValueError("predictive diagnostic range order differs from the paper")
                predictive[method]["by_range"].append(dict(range_m=interval, **{key:
                    summary([row[key] for row in strata], 100 if key in ("query_share", "coverage90") else 1,
                            counts=[row["queries"] for row in rows] if key == "query_share" else
                                   [row["queries"] for row in strata])
                    for key in ("query_share", "nll", "coverage90", "width90_m", "median_mae_m")}))

    runtime = {method: [read(f"{method}-{seed}/runtime.json") for seed in seeds] for method in ("semantic", "joint")}
    reference = runtime["joint"][0]
    for rows in runtime.values():
        for row in rows:
            if any(row[key] != reference[key] for key in ("scans", "hardware", "precision", "batch_size", "warmup_scans")):
                raise ValueError("runtime measurements must share scan order, hardware, precision and warmup")
            if row["precision"] != "FP32" or row["batch_size"] != 1 or row["warmup_scans"] != 50:
                raise ValueError("runtime measurements differ from the paper's FP32, batch-one, fifty-warmup protocol")
    timing = dict(hardware=reference["hardware"], scans=len(reference["scans"]), warmup_scans=50,
        methods={method: dict(mean_ms=summary([row["mean_ms"] for row in rows]),
            peak_vram_GB=summary([row["peak_vram_bytes"] for row in rows], 1e-9)) for method, rows in runtime.items()},
        additional_ms=summary([joint["mean_ms"] - appearance["mean_ms"]
                              for joint, appearance in zip(runtime["joint"], runtime["semantic"], strict=True)]),
        timing_scope=reference["timing_scope"], memory_scope=reference["memory_scope"])
    return dict(seeds=list(seeds), sources=sources, validation_population=population, test_population=test_population,
        reduction="Within each seed, pool the prescribed points before computing metrics; then report the arithmetic mean and sample standard deviation across seeds. Reported rates and IoUs are percentage points; requested_fpr and confidence_threshold remain fractions. NLL, metres and milliseconds retain their units; GB means 10^9 bytes.",
        main=main, ablations=ablations, per_class_iou=per_class,
        confident_unknown=dict(confidence_threshold=.9,
            points_per_seed={str(seed): row["confident_unknown_points"] for seed, row in zip(seeds, paired, strict=True)},
            methods=confidence), instance_coverage=dict(size_bins=bins, instance_counts=counts,
                unassigned_anomaly_points=unassigned, methods=instances),
        normal_corrections=corrections, predictive=predictive, runtime=timing, missing_test=missing_test,
        external_references="Published LIDO, NDP and COVAL results are literature references, not measurements produced by this suite.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "test", "infer", "export", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--readout", choices=("joint", "appearance", "common_density", "independent_minima", "energy", "softmax"))
        if name == "validate":
            command.add_argument("--manifest", type=Path, default=Path("assets/val.json"))
        if name == "test":
            command.add_argument("--data", type=Path, required=True)
        if name in ("validate", "test"):
            command.add_argument("--record-points", action="store_true",
                help="retain exact metric and full-return scores with original point identities")
        if name in ("validate", "test"):
            command.add_argument("--workers", type=int, default=4)
        elif name == "export":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--split", choices=("val", "test"), default="test")
            command.add_argument("--semantic-output", action="store_true")
        else:
            if name == "infer":
                command.add_argument("--scans", type=Path, nargs="+", required=True)
                command.add_argument("--semantic-output", action="store_true",
                    help="also save *.semantic.npy with STU19 classes; empty slots are -1")
            if name == "benchmark":
                source = command.add_mutually_exclusive_group(required=True)
                source.add_argument("--scans", type=Path, nargs="+")
                source.add_argument("--manifest", type=Path, help="time every validation scan in manifest order")
                command.add_argument("--warmup", type=int, default=50)
    comparison = sub.add_parser("compare", help="paired normal-only variants on the identical official point population")
    comparison.add_argument("--semantic", type=Path, required=True, help="independently trained semantic baseline checkpoint")
    comparison.add_argument("--joint", type=Path, required=True)
    comparison.add_argument("--separate", type=Path, help="independently trained control without joint classification supervision")
    comparison.add_argument("--manifest", type=Path, required=True)
    comparison.add_argument("--split", choices=("val", "test"), default="val")
    comparison.add_argument("--output", type=Path, required=True, help="empty directory for exact paired point records and report")
    comparison.add_argument("--device", default="cuda")
    comparison.add_argument("--workers", type=int, default=4)
    comparison.add_argument("--normal-fpr", type=float, nargs="+", default=[.01, .05])
    comparison.add_argument("--confidence", type=float, default=.9)
    comparison.add_argument("--exploratory", action="store_true", help="report unmatched training conditions explicitly")
    comparison.add_argument("--fixed-readouts", action="store_true", help="also score the three fixed-weight SERVE readouts in the same forward pass")
    comparison.add_argument("--discard-fixed-records", action="store_true",
                            help="retain fixed-readout statistics and checkpoints but omit their reproducible point arrays")
    normal = sub.add_parser("compare-normal", help="paired independently trained normal decisions on sequence 201")
    for name in ("semantic", "joint", "separate"):
        normal.add_argument("--" + name, type=Path, required=name != "separate")
    normal.add_argument("--data", type=Path, help="STU root containing train/201")
    normal.add_argument("--output", type=Path, required=True)
    normal.add_argument("--device", default="cuda")
    normal.add_argument("--workers", type=int, default=4)
    normal.add_argument("--fixed-readouts", action="store_true")
    normal.add_argument("--exploratory", action="store_true")
    aggregation = sub.add_parser("aggregate", help="three-seed mean and sample standard deviation")
    aggregation.add_argument("--results", type=Path, nargs=3, required=True)
    aggregation.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("summary", help="summarize the complete paper experiment suite")
    summary.add_argument("--directory", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "summary":
        write_json(args.output, summarize_experiments(args.directory))
        return
    if args.action == "aggregate":
        write_json(args.output, aggregate_results([json.loads(path.read_text()) for path in args.results]))
        return
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.action in ("compare", "compare-normal"):
        paths = {name: getattr(args, name) for name in ("semantic", "joint", "separate") if getattr(args, name) is not None}
        if args.action == "compare":
            manifest = load_manifest(args.manifest, args.split)
            compare(paths, manifest, args.output, device, workers=args.workers, fprs=args.normal_fpr,
                    confidence=args.confidence, exploratory=args.exploratory, fixed_readouts=args.fixed_readouts,
                    retain_fixed_records=not args.discard_fixed_records)
        else:
            from .data import normal_records
            records = normal_records("201", development=True, root=args.data)
            result = compare_normal(paths, records, device, workers=args.workers,
                                    fixed_readouts=args.fixed_readouts, exploratory=args.exploratory)
            write_json(args.output, result)
        return
    model, saved = load_model(args.checkpoint, device)
    if args.action in ("validate", "test"):
        if args.action == "test":
            if not saved.get("frozen"):
                raise ValueError("test requires a selected checkpoint from a completed fixed budget")
            manifest = make_real_manifest(args.data, partition="test", workers=args.workers)
            if manifest["directory"] == saved["config"].get("val_directory"):
                raise ValueError("the validation set cannot be presented as final test data")
        else:
            manifest = load_manifest(args.manifest, "val")
        score_path = None
        if args.record_points:
            if args.output.suffix != ".json":
                parser.error("--record-points requires a .json output path so metadata cannot overwrite scores")
            from .train import disk_check
            rows = [manifest["records"][i] for i in evaluation_indices(manifest)]
            disk_check(sum(4*(row["normal"]+row["anomaly"])+9*row["points"] for row in rows)
                       + 10_000_000)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            score_path = args.output.with_suffix(".npy")
        result = evaluate(model, manifest, device, args.workers,
                          score_path=score_path, record_points=args.record_points, readout=args.readout)
        metadata = dict(version=saved["version"], complete=saved.get("frozen", False),
                        seed=saved["config"]["seed"], mode=saved.get("mode", model.mode), method=saved["config"]["variant"],
                        variant=saved["config"]["variant"], readout=args.readout or ("energy" if model.variant == "standard" else "joint"),
                        split=manifest["kind"],
                        normal201=(saved.get("normal201") if args.readout not in ("appearance", "common_density") else None),
                        architecture=saved["config"]["architecture"], score_version=saved["config"]["score_version"],
                        evaluation_role=saved["config"].get("evaluation", "recorded checkpoint evaluation"),
                        checkpoint_update=saved.get("stages", {}).get("target", {}).get("selected_update"))
        write_json(args.output, dict(**metadata, checkpoint=str(args.checkpoint.resolve()), **result))
    elif args.action in ("infer", "export"):
        from .train import disk_check
        if args.action == "export":
            manifest = load_manifest(args.manifest, args.split)
            if args.split == "test" and not saved.get("frozen"):
                raise ValueError("test export requires a selected checkpoint from a completed fixed budget")
            args.scans = [Path(row["scan"]) for row in manifest["records"]]
        disk_check(sum(p.stat().st_size // 16 * (26 if args.semantic_output else 24) for p in args.scans))
        args.output.mkdir(parents=True, exist_ok=True)
        for index, scan in enumerate(args.scans):
            prediction, timing = infer(model, scan, device, return_semantics=args.semantic_output, readout=args.readout,
                record=manifest["records"][index] if args.action == "export" else None,
                partition=args.split if args.action == "export" else "val")
            folder = args.output / scan.parent.parent.name
            folder.mkdir(exist_ok=True)
            if args.semantic_output:
                np.save(folder / (scan.stem + ".semantic.npy"), prediction["semantic"])
                prediction = prediction["score"]
            np.savetxt(folder / (scan.stem + ".txt"), prediction, fmt="%.9g")
            print(dict(scan=str(scan), **timing), flush=True)
            if (index + 1) % 100 == 0:
                disk_check()
    else:
        if args.manifest is not None:
            manifest = load_manifest(args.manifest, "val")
            args.scans = [Path(row["scan"]) for row in manifest["records"]]
        write_json(args.output, benchmark(model, args.scans, device, warmup=args.warmup, readout=args.readout))


if __name__ == "__main__":
    main()
