"""SERVE pooled-point validation, paired comparisons and original-slot inference."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Scans, load_manifest, make_real_manifest, read_scan, write_json
from .model import NormalHypothesis, prepare_scan, scatter_scores, to_device
from .normal import hypothesis_observation
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator

STU_COMMIT = "8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2"


class PreparedScans(Scans):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        result = prepare_scan(sample)
        result["observation"] = hypothesis_observation(sample["xyzi"])
        return result


def evaluation_indices(manifest):
    """Use the official per-scan five-anomaly rule."""
    return [i for i, row in enumerate(manifest["records"]) if row["eligible"]]


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # Shape-compatible historical weights still represent a different model.
    NormalHypothesis.validate_checkpoint(saved, require_calibrated=True)
    model = NormalHypothesis(variant=saved["config"]["variant"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def precision(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast(device):
    return torch.autocast(device.type, dtype=precision(device), enabled=device.type == "cuda")


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
def evaluate(model, manifest, device, workers=4, score_path=None, record_points=False):
    """Call the pinned official implementation once across the complete valid set."""
    if manifest["kind"] not in ("val", "test"):
        raise ValueError("evaluation requires a held-out validation or test manifest")
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
        identity_path = score_path.parent / "val_points.npy"
        identities = np.lib.format.open_memmap(identity_path, mode="r" if identity_path.exists() else "w+",
            dtype=np.dtype([("slot", "<u4"), ("target", "i1")]), shape=(raw_count,))
        if identities.shape != (raw_count,):
            raise ValueError("evaluation point population changed")
    cursor = 0
    start = time.perf_counter()
    for number, sample in enumerate(loader, 1):
        batch = to_device(sample, device)
        with autocast(device):
            prediction = model(batch)
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
        write_json(score_path.parent / "val.json", dict(manifest_sha256=manifest["sha256"], frames=frames,
            points=raw_count, metric_points=count, identities="val_points.npy",
            scope="Each val*.npy is the exact selected metric population; its *_all.npy companion preserves every actual return, including ignored context, in the selected eligible full scans. "))
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
    records = json.loads((score_path.parent / "val.json").read_text())
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    raw = np.load(score_path.with_stem(score_path.stem + "_all"), mmap_mode="r", allow_pickle=False)
    identities = np.load(score_path.parent / "val_points.npy", mmap_mode="r", allow_pickle=False)
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
    cursor = raw_cursor = normal_cursor = anomaly_cursor = 0
    started = time.perf_counter()
    for row in records["frames"]:
        expected = manifest["records"][row["index"]]
        raw_stop, stop = raw_cursor + expected["points"], cursor + expected["normal"] + expected["anomaly"]
        if (row["start"], row["stop"], row["metric_start"], row["metric_stop"]) != (raw_cursor, raw_stop, cursor, stop):
            raise ValueError("saved frame offsets do not preserve official point order")
        points = identities[raw_cursor:raw_stop]
        targets, slots = points["target"], points["slot"]
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
            differences.append(f"{method}: training and normal calibration are incomplete")
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


def comparison_metrics(labels, predictions, *, fprs=(.01, .05), confidence=.9):
    """Pool exact official metrics and paired decisions on one shared point order."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all() or not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("comparison needs a nonempty shared population of official normal and unknown points")
    if not 0 <= confidence <= 1 or not fprs or any(not 0 <= value <= 1 for value in fprs):
        raise ValueError("confidence and normal false-positive rates must be in [0, 1]")
    if not {"semantic", "joint"}.issubset(predictions):
        raise ValueError("comparison needs semantic and joint point predictions")
    normal, unknown = labels == 0, labels == 1
    for name, values in predictions.items():
        for key in ("score", "raw_score", "confidence", "semantic"):
            if np.shape(values[key]) != labels.shape or not np.isfinite(values[key]).all():
                raise ValueError(f"{name}.{key} does not preserve the finite shared point population")
        if ((values["confidence"] < 0) | (values["confidence"] > 1)).any():
            raise ValueError("class confidence must be a normalized class probability")
        if not np.issubdtype(values["semantic"].dtype, np.integer) or not np.isin(values["semantic"], np.arange(19)).all():
            raise ValueError("class predictions must be integer normal IDs 0–18")
    # This subset is defined once by the independent semantic baseline, not by a
    # competing model's confidence or by whether its anomaly threshold rejected it.
    confident = unknown & (predictions["semantic"]["confidence"] >= confidence)
    methods, sorted_normal = {}, {}
    for name, values in predictions.items():
        calculator = PointOODMetricsCalculator()
        calculator.all_scores, calculator.all_labels = [values["score"]], [labels]
        metrics = {key: float(value) for key, value in calculator.compute_metrics().items()}
        better(metrics, None)
        ordered = np.sort(values["score"][normal])
        sorted_normal[name] = ordered
        operating = []
        for requested in fprs:
            budget = int(np.floor(float(requested) * len(ordered)))
            # Reject score > threshold: tied normal scores are accepted or rejected
            # together, never interpolated or split by point order.
            threshold = (float(ordered[len(ordered) - budget - 1]) if budget < len(ordered)
                         else float(np.nextafter(float(values["score"].min()), -np.inf)))
            rejected = values["score"] > np.float64(threshold)
            false_positives = int(np.count_nonzero(rejected & normal))
            operating.append(dict(requested_fpr=float(requested), threshold=threshold, comparison="score > threshold",
                false_positives=false_positives, actual_fpr=false_positives / int(normal.sum()),
                unknown_recalled=int(np.count_nonzero(rejected & unknown)),
                unknown_recall=float(np.count_nonzero(rejected & unknown) / unknown.sum()),
                confident_unknown_recalled=int(np.count_nonzero(rejected & confident)),
                confident_unknown_recall=(float(np.count_nonzero(rejected & confident) / confident.sum()) if confident.any() else None)))
        methods[name] = dict(metrics=metrics, operating_points=operating)
    # Compare at equal *achieved* FPR. A score tie cannot be broken using truth or
    # input order, so choose a false-positive count attainable by every method.
    maximum = int(np.floor(max(fprs) * int(normal.sum())))
    common_counts = None
    for ordered in sorted_normal.values():
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        counts = np.r_[0, (len(ordered) - starts)[::-1]]
        counts = counts[counts <= maximum]
        common_counts = counts if common_counts is None else np.intersect1d(common_counts, counts, assume_unique=True)
    for name, values in predictions.items():
        methods[name]["individual_operating_points"] = [dict(row) for row in methods[name]["operating_points"]]
        ordered = sorted_normal[name]
        for row in methods[name]["operating_points"]:
            budget = int(np.floor(row["requested_fpr"] * len(ordered)))
            shared = int(common_counts[np.searchsorted(common_counts, budget, side="right") - 1])
            threshold = (float(ordered[len(ordered) - shared - 1]) if shared < len(ordered)
                         else float(np.nextafter(float(values["score"].min()), -np.inf)))
            rejected = values["score"] > np.float64(threshold)
            achieved = int(np.count_nonzero(rejected & normal))
            if achieved != shared:
                raise ValueError("normal score ties do not reproduce the shared false-positive count")
            row.update(threshold=threshold, false_positives=achieved, actual_fpr=achieved / len(ordered),
                       unknown_recalled=int(np.count_nonzero(rejected & unknown)),
                       unknown_recall=float(np.count_nonzero(rejected & unknown) / unknown.sum()),
                       confident_unknown_recalled=int(np.count_nonzero(rejected & confident)),
                       confident_unknown_recall=(float(np.count_nonzero(rejected & confident) / confident.sum()) if confident.any() else None))
    del sorted_normal
    for name, values in predictions.items():
        for row, baseline in zip(methods[name]["operating_points"], methods["semantic"]["operating_points"], strict=True):
            rejected = values["score"] > np.float64(row["threshold"])
            base_rejected = predictions["semantic"]["score"] > np.float64(baseline["threshold"])
            rescued, lost = rejected & ~base_rejected, ~rejected & base_rejected
            row.update(unknown_rescued=int(np.count_nonzero(rescued & unknown)),
                       unknown_lost=int(np.count_nonzero(lost & unknown)),
                       confident_unknown_rescued=int(np.count_nonzero(rescued & confident)),
                       confident_unknown_lost=int(np.count_nonzero(lost & confident)),
                       normal_new_false_positives=int(np.count_nonzero(rescued & normal)),
                       normal_removed_false_positives=int(np.count_nonzero(lost & normal)),
                       equal_actual_fpr=row["false_positives"] == baseline["false_positives"])
    return dict(methods=methods, normal_points=int(normal.sum()), unknown_points=int(unknown.sum()),
                confident_unknown_points=int(confident.sum()), baseline_confidence_threshold=float(confidence),
                confidence_definition="maximum softmax class probability of the independently trained semantic model; this is not a calibrated correctness probability",
                threshold_role="operating points on this evaluation population's normal-point curve, not deployment thresholds",
                tie_rule="whole score ties; operating_points use the greatest false-positive count jointly attainable by all methods within the requested FPR; individual_operating_points retain each method's own maximal count")


@torch.no_grad()
def compare(checkpoints, manifest, output, device, *, workers=4, fprs=(.01, .05), confidence=.9, exploratory=False):
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
    identity_type = np.dtype([("frame", "<u4"), ("slot", "<u4"), ("target", "i1"), ("official", "?")])
    peak_write = count * (record_type.itemsize * len(checkpoints) + 1) + returns * identity_type.itemsize + 1_000_000
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
    required = 112 * count + 1_000_000_000
    if memory_available() < required:
        raise RuntimeError(f"exact paired metrics need about {required / 1e9:.1f} GB free RAM after loading models")
    output.mkdir(parents=True, exist_ok=True)
    records = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=record_type, shape=(count,))
               for name in models}
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
                prediction = model.predict(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds[name] += time.perf_counter() - tick
            for key in record_type.names:
                values = prediction[key].detach().cpu().numpy()
                if values.shape != (len(sample["xyzi"]),) or not np.isfinite(values).all():
                    raise ValueError(f"{name}.{key} changed the point population or contains nonfinite values")
                if key == "semantic" and (not np.issubdtype(values.dtype, np.integer) or not np.isin(values, np.arange(19)).all()):
                    raise ValueError("normal semantic predictions must be integer IDs 0–18 before recording")
                if key != "semantic" and values.dtype != np.float32:
                    raise ValueError(f"{name}.{key} must retain the model's exact float32 output precision")
                records[name][key][cursor:stop] = values[chosen]
            del prediction
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
    result = dict(**measured, conditions=conditions, frames=frames, scans=len(indices), points=count,
                  checkpoints={name: str(Path(path).resolve()) for name, path in checkpoints.items()},
                  normal201={name: row.get("normal201") for name, row in saved.items()},
                  manifest_sha256=manifest["sha256"], official_commit=STU_COMMIT, resources=resources,
                  execution=dict(weight_bytes=weight_bytes, weights_resident=resident,
                                 forwards="sequential per prepared full scan; each model's activations are released before the next"),
                  inference_seconds=inference_seconds, seconds=time.perf_counter() - started,
                  records="returns.npy stores every actual return with original slot, manifest frame index, target and official mask; filtering it by official gives exactly the common row order of labels.npy and all method *.npy files",
                  target_definition="-1 ignored, 0 normal, 1 unknown; empty original slots have no actual return and are absent")
    write_json(output / "comparison.json", result)
    return result


@torch.no_grad()
def infer(model, scan, device, *, return_semantics=False):
    if getattr(model, "mode", None) != "normal_hypothesis":
        raise ValueError("inference requires a SERVE normal perception model")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    io_timing = {}
    frame = read_scan(scan, io_timing=io_timing)
    read_seconds = io_timing["seconds"]
    semantic = np.full(len(frame.xyzi), -1, np.int16) if return_semantics else None
    if frame.actual.any():
        xyzi = frame.xyzi[frame.actual].copy()
        sample = prepare_scan(dict(xyzi=xyzi, slots=frame.return_slots,
                                   targets=np.full(int(frame.actual.sum()), -1, np.int8),
                                   slot_count=len(frame.xyzi), index=frame.frame_id))
        sample["observation"] = hypothesis_observation(xyzi)
        sample = to_device(sample, device)
        with autocast(device):
            if return_semantics:
                outputs = model.predict(sample)
                prediction, classes = outputs["score"], outputs["semantic"]
                if classes.shape != prediction.shape or classes.is_floating_point() or bool(((classes < 0) | (classes >= 19)).any()):
                    raise ValueError("semantic outputs must follow the model's class vocabulary in original return order")
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
    return prediction, dict(read_seconds=read_seconds, seconds=time.perf_counter() - start - read_seconds,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                            real_points=int(frame.actual.sum()), slots=len(frame.xyzi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "test", "infer", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "validate":
            command.add_argument("--manifest", type=Path, default=Path("assets/val.json"))
        if name == "test":
            command.add_argument("--data", type=Path, required=True)
        if name in ("validate", "test"):
            command.add_argument("--record-points", action="store_true",
                help="retain exact metric and full-return scores with original point identities")
        if name in ("validate", "test"):
            command.add_argument("--workers", type=int, default=4)
        else:
            command.add_argument("--scans", type=Path, nargs="+", required=True)
            if name == "infer":
                command.add_argument("--semantic-output", action="store_true",
                    help="also save *.semantic.npy with STU19 classes; empty slots are -1")
            if name == "benchmark":
                command.add_argument("--warmup", type=int, default=5)
                command.add_argument("--repeats", type=int, default=20)
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
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.action == "compare":
        manifest = load_manifest(args.manifest, args.split)
        paths = {name: getattr(args, name) for name in ("semantic", "joint", "separate") if getattr(args, name) is not None}
        compare(paths, manifest, args.output, device, workers=args.workers, fprs=args.normal_fpr,
                confidence=args.confidence, exploratory=args.exploratory)
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
            from .train import disk_check
            rows = [manifest["records"][i] for i in evaluation_indices(manifest)]
            disk_check(sum(4*(row["normal"]+row["anomaly"])+9*row["points"] for row in rows)
                       + 10_000_000)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            score_path = args.output.with_suffix(".npy")
        result = evaluate(model, manifest, device, args.workers,
                          score_path=score_path, record_points=args.record_points)
        metadata = dict(version=saved["version"], complete=saved.get("frozen", False),
                        seed=saved["config"]["seed"], mode=saved["mode"], method=saved["mode"],
                        architecture=saved["config"]["architecture"], score_version=saved["config"]["score_version"],
                        evaluation_role=saved["config"].get("evaluation", "recorded checkpoint evaluation"),
                        checkpoint_update=saved.get("stages", {}).get("target", {}).get("selected_update"))
        write_json(args.output, dict(**metadata, checkpoint=str(args.checkpoint.resolve()), **result))
    elif args.action == "infer":
        from .train import disk_check
        disk_check(sum(p.stat().st_size // 16 * (26 if args.semantic_output else 24) for p in args.scans))
        args.output.mkdir(parents=True, exist_ok=True)
        for index, scan in enumerate(args.scans):
            prediction, timing = infer(model, scan, device, return_semantics=args.semantic_output)
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
        if args.repeats < 1 or args.warmup < 0:
            parser.error("invalid benchmark warmup or repeat count")
        timings = []
        for step in range(args.warmup + args.repeats):
            _, timing = infer(model, args.scans[step % len(args.scans)], device)
            if step >= args.warmup:
                timings.append(timing)
        latency = np.array([t["seconds"] for t in timings]) * 1000
        write_json(args.output, dict(batch_size=1, parameters=sum(p.numel() for p in model.parameters()),
                                    scans=[str(p) for p in args.scans], repeats=args.repeats,
                                    mean_ms=float(latency.mean()), p95_ms=float(np.percentile(latency, 95)),
                                    mean_read_ms=float(np.mean([t["read_seconds"] for t in timings]) * 1000),
                                    peak_vram_bytes=max(t["peak_vram_bytes"] for t in timings),
                                    hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"))


if __name__ == "__main__":
    main()
