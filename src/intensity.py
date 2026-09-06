"""Fixed-candidate intensity distributions and scan-local permutation diagnostics."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch

from .data import FrozenWindowDataset, PredictionBatch, _atomic_json
from .evaluate import (
    diagnostic_bin,
    evaluation_targets,
    file_hash,
    pooled_files,
    predict_window,
    save_window,
    assert_unchanged,
)
from .model import AJAE, joint_voxelize
from .protocol import PROJECT_ROOT, load_protocol
from .scene import STUSequence, LabelMode, make_source_frame
from .train import FullResources


SEEDS = (0, 1, 2)
DISTANCE_WIDTH = 2.5
QUANTILES = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)


def scan_permutation(xyzi, slots, sequence, frame, seed):
    """Shuffle all visible returns within raw-scan radial shells, without labels."""
    distance = np.linalg.norm(xyzi[slots, :3], axis=1)
    bins = np.floor(distance.astype(np.float64) / DISTANCE_WIDTH).astype(np.int64)
    identity = json.dumps([sequence, int(frame), int(seed)], separators=(",", ":"))
    entropy = np.frombuffer(hashlib.sha256(identity.encode()).digest(), dtype="<u4")
    rng = np.random.default_rng(entropy)
    permutation = np.arange(len(slots))
    for group in np.unique(bins):
        selected = np.flatnonzero(bins == group)
        permutation[selected] = rng.permutation(selected)
    return permutation


def permute_window(window, inputs, seed, cache):
    """Reuse geometry exactly and rebuild both intensity channels from one shuffle."""
    values, members = [], []
    for member in window.frames:
        source = member.source
        key = (window.observation_sequence_id, source.frame_id, seed)
        permutation = cache.pop(key, None)
        if permutation is None:
            permutation = scan_permutation(source.xyzi, source.real_slots, *key)
        cache[key] = permutation
        while len(cache) > 30:
            cache.popitem(last=False)
        intensity = source.xyzi[source.real_slots[permutation], 3]
        values.append(intensity)
        xyzi = source.xyzi.copy()
        xyzi[source.real_slots, 3] = intensity
        changed = make_source_frame(
            source.frame_id,
            xyzi,
            source.lidar_pose,
            source.labels,
            partition=source.partition,
            sequence_id=source.sequence_id,
        )
        members.append(replace(member, source=changed))
    intensity = np.concatenate(values)
    points = replace(window.points, features=intensity[:, None])
    changed = replace(window, points=points, frames=tuple(members))
    inverse = inputs.point_to_voxel.numpy()
    counts = np.bincount(inverse, minlength=len(inputs.features))
    # Match joint_voxelize's float64 sums and final float32 cast exactly.
    means = (
        np.bincount(inverse, weights=intensity, minlength=len(counts)) / counts
    ).astype(np.float32)
    features, point_features = inputs.features.clone(), inputs.point_features.clone()
    features[:, 3] = torch.from_numpy(means)
    point_features[:, 3] = torch.from_numpy(intensity)
    changed_inputs = replace(
        inputs, features=features, point_features=point_features, source_points=points
    )
    return changed, changed_inputs


def distribution(values, support):
    """Exact empirical distribution, including float32 ties and a measured grid."""
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite intensity feature")
    unique, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts)
    positions = (len(values) - 1) * np.asarray(QUANTILES)
    lo, hi = np.floor(positions).astype(np.int64), np.ceil(positions).astype(np.int64)
    lower = unique[np.searchsorted(cumulative, lo, side="right")].astype(np.float64)
    upper = unique[np.searchsorted(cumulative, hi, side="right")].astype(np.float64)
    quantiles = lower + (positions - lo) * (upper - lower)
    grid = (np.rint(unique.astype(np.float64) * 3500) / 3500).astype(np.float32)
    top = np.argsort(-counts, kind="stable")[:10]
    total = len(values)
    return dict(
        count=total,
        quantiles=dict(zip(map(str, QUANTILES), map(float, quantiles))),
        mean=float(np.dot(unique.astype(np.float64), counts) / total),
        unique_count=len(unique),
        repeated_point_fraction=float(counts[counts > 1].sum() / total),
        zero_fraction=float(counts[unique == 0].sum() / total),
        support_min_fraction=float(counts[unique == support[0]].sum() / total),
        support_max_fraction=float(counts[unique == support[1]].sum() / total),
        outside_support_fraction=float(
            counts[(unique < support[0]) | (unique > support[1])].sum() / total
        ),
        grid_1_over_3500_fraction=float(counts[unique == grid].sum() / total),
        top_values=[dict(value=float(unique[i]), count=int(counts[i])) for i in top],
    )


def fixed_samples():
    diagnostic = PROJECT_ROOT / "runs/diagnostic_v1"
    frames = [
        json.loads(line)
        for line in (diagnostic / "frames.jsonl").read_text().splitlines()
    ]
    roots = dict(
        synthetic=PROJECT_ROOT / "runs/fulltrain_v1/validation/epoch_07",
        real=PROJECT_ROOT / "runs/real_val_v1",
    )
    selected = []
    for domain, root in roots.items():
        keys = {
            (r["sequence_id"], r["current_frame"]): r
            for r in frames
            if r["domain"] == domain and r["stratum"] == "1_1"
        }
        rows = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        for row in rows:
            key = (row["sequence_id"], row["current_frame"])
            if row["view"] == domain and key in keys:
                row = dict(
                    row,
                    source_directory=str(root),
                    world_identity=keys[key]["world_identity"],
                )
                row["entity"] = (
                    f"{row['sequence_id']}/segment_{row['segment_index']:02d}"
                    if domain == "synthetic"
                    else row["sequence_id"]
                )
                selected.append(row)
    if [sum(r["view"] == d for r in selected) for d in roots] != [113, 303]:
        raise ValueError(
            "the fixed intensity diagnostic must contain 113 + 303 windows"
        )
    return selected


def load_window(row, dataset, sequences, data_root, protocol):
    if row["view"] == "synthetic":
        window = dataset[row["dataset_index"]]
    else:
        key = row["sequence_index"]
        if key not in sequences:
            sequences.clear()  # Keep only the current real sequence's bounded reader.
            sequences[key] = STUSequence.open(
                data_root,
                protocol=protocol,
                partition="val",
                sequence_id=key,
                label_mode=LabelMode.REQUIRED,
            )
        window = sequences[key].for_output(row["current_frame"])
    if (
        window.observation_sequence_id != row["sequence_id"]
        or list(window.frame_ids) != row["frame_ids"]
    ):
        raise ValueError("intensity diagnostic window differs from its fixed identity")
    return window


def training_samples(dataset):
    selected, counts, coverage = (
        [],
        dict(total=len(dataset), eligible=0, zero=0, few=0),
        {},
    )
    for index in range(len(dataset)):
        segment, start = dataset.segment_for_window(index)
        source = segment.frame(start + 4)
        target = evaluation_targets(source.xyzi[:, :3], source.labels.semantic)
        positive = int((target == 1).sum())
        if positive < 5:
            counts["zero" if positive == 0 else "few"] += 1
            continue
        counts["eligible"] += 1
        median = float(np.median(np.linalg.norm(source.xyzi[target == 1, :3], axis=1)))
        key = diagnostic_bin(positive, median)
        cell = coverage.setdefault(key, dict(frames=0, anomaly_points=0))
        cell["frames"] += 1
        cell["anomaly_points"] += positive
        if key == "1_1":
            metadata = segment.metadata
            selected.append(
                dict(
                    view="synthetic",
                    dataset_index=index,
                    current_frame=start + 4,
                    frame_ids=list(range(start, start + 5)),
                    sequence_id=metadata["synthetic_sequence_id"],
                    entity=f"{metadata['synthetic_sequence_id']}/segment_{metadata['segment_index']:02d}",
                    world_identity=metadata["world_identity"],
                    anomaly_count=positive,
                    normal_count=int((target == 0).sum()),
                    anomaly_distance_median=median,
                )
            )
    return selected, dict(
        **counts,
        strata=coverage,
        selected_windows=len(selected),
        selected_worlds=len({r["entity"] for r in selected}),
        selected_anomaly_points=sum(r["anomaly_count"] for r in selected),
    )


def feature_statistics(data_root, protocol, samples, output, resources):
    started = time.monotonic()
    calibration = torch.load(
        PROJECT_ROOT / protocol.artifacts["sensor_calibration"]["file"],
        weights_only=False,
    )
    sensor = calibration["sensor"]
    support = (sensor["intensity_min"], sensor["intensity_max"])
    datasets = {
        name: FrozenWindowDataset(data_root, protocol, pool_name=name)
        for name in ("train", "validation")
    }
    training, coverage = training_samples(datasets["train"])
    _atomic_json(output / "training.json", dict(coverage=coverage, samples=training))
    print(json.dumps({"event": "training_coverage", **coverage}), flush=True)
    sources = dict(
        training=training,
        synthetic=[r for r in samples if r["view"] == "synthetic"],
        real=[r for r in samples if r["view"] == "real"],
    )
    result = dict(training_coverage=coverage, support=support, populations={})
    for domain, rows in sources.items():
        dataset = datasets["train" if domain == "training" else "validation"]
        sequences, entities = {}, {}
        with tempfile.TemporaryDirectory(dir=output, prefix="features_") as temporary:
            paths = [Path(temporary) / f"{label}.bin" for label in range(2)]
            offsets = [0, 0]
            with paths[0].open("wb") as normal, paths[1].open("wb") as anomaly:
                streams = [normal, anomaly]
                for index, row in enumerate(rows):
                    resources()
                    window = load_window(row, dataset, sequences, data_root, protocol)
                    inputs = joint_voxelize(window)
                    current = window.current_mask
                    target = evaluation_targets(
                        window.points.coordinates[current],
                        window.labels.semantic[current],
                    )
                    raw = window.points.features[current, 0]
                    means = inputs.features.numpy()[
                        inputs.point_to_voxel.numpy()[current], 3
                    ]
                    distance = np.linalg.norm(
                        window.points.coordinates[current], axis=1
                    )
                    expected = row.get("current", row)
                    if any(
                        int((target == label).sum()) != expected[key]
                        for label, key in ((0, "normal_count"), (1, "anomaly_count"))
                    ):
                        raise ValueError(
                            "intensity feature statistics changed evaluation points"
                        )
                    entity = entities.setdefault(
                        row["entity"], dict(frames=0, ranges=[[], []])
                    )
                    entity["frames"] += 1
                    for label in range(2):
                        values = np.column_stack(
                            (
                                raw[target == label],
                                means[target == label],
                                distance[target == label],
                            )
                        ).astype(np.float32)
                        values.tofile(streams[label])
                        entity["ranges"][label].append((offsets[label], len(values)))
                        offsets[label] += len(values)
                    if (index + 1) % 50 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "features",
                                    "domain": domain,
                                    "completed": index + 1,
                                    "total": len(rows),
                                }
                            ),
                            flush=True,
                        )
            population = dict(
                frame_count=len(rows),
                entity_count=len(entities),
                pooled={},
                entities={},
            )
            for label, name in enumerate(("normal", "anomaly")):
                values = (
                    np.memmap(paths[label], np.float32, "r", shape=(offsets[label], 3))
                    if offsets[label]
                    else np.empty((0, 3), np.float32)
                )
                pooled = {
                    channel: distribution(values[:, c], support)
                    for c, channel in enumerate(("point", "voxel"))
                }
                pooled["distance_bins"] = {}
                bins = np.minimum(
                    np.floor(values[:, 2].astype(np.float64) / DISTANCE_WIDTH).astype(
                        np.int32
                    ),
                    19,
                )
                for b in range(1, 20):
                    selected = bins == b
                    pooled["distance_bins"][str(b)] = {
                        channel: distribution(values[selected, c], support)
                        for c, channel in enumerate(("point", "voxel"))
                    }
                population["pooled"][name] = pooled
                for entity, record in entities.items():
                    pieces = [
                        values[start : start + count]
                        for start, count in record["ranges"][label]
                        if count
                    ]
                    group = (
                        np.concatenate(pieces)
                        if pieces
                        else np.empty((0, 3), np.float32)
                    )
                    metrics = {
                        channel: distribution(group[:, c], support)
                        for c, channel in enumerate(("point", "voxel"))
                    }
                    group_bins = np.minimum(
                        np.floor(
                            group[:, 2].astype(np.float64) / DISTANCE_WIDTH
                        ).astype(np.int32),
                        19,
                    )
                    metrics["distance_bins"] = {
                        str(b): {
                            channel: distribution(group[group_bins == b, c], support)
                            for c, channel in enumerate(("point", "voxel"))
                        }
                        for b in range(1, 20)
                    }
                    population["entities"].setdefault(
                        entity, dict(frame_count=record["frames"])
                    )[name] = metrics
                del values
            result["populations"][domain] = population
        print(
            json.dumps({"event": "feature_statistics_completed", "domain": domain}),
            flush=True,
        )
    result["wall_seconds"] = time.monotonic() - started
    _atomic_json(output / "distributions.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/intensity_v1"))
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()
    protocol, samples = load_protocol(), fixed_samples()
    torch.set_num_threads(1)
    resources = FullResources(
        lambda event, **data: print(json.dumps(dict(event=event, **data)), flush=True)
    )
    snapshot = resources()
    predicted_points = sum(r["point_count"] for r in samples) * len(SEEDS)
    # Uncompressed score/identity arrays bound the lossless prediction footprint.
    prediction_peak = (
        predicted_points * 12
        + sum(r["evaluation_records"]["count"] for r in samples) * 8 * 4
        + 256 * 2**20
    )
    feature_peak = protocol.training_pool.total_window_count * 131072 * 12 + 256 * 2**20
    if (
        snapshot["host_disk"]["SizeRemaining"] - max(prediction_peak, feature_peak)
        < snapshot["host_disk"]["reserve_bytes"]
    ):
        raise OSError("intensity diagnostic would invade the E: reserve")
    spec = dict(
        checkpoint=protocol.data["real_anomaly_development_validation"]["checkpoint"],
        checkpoint_sha256=protocol.data["real_anomaly_development_validation"][
            "checkpoint_sha256"
        ],
        seeds=SEEDS,
        radial_width_m=DISTANCE_WIDTH,
        permutation="all visible points including ignored labels; raw-source radial floor bins; PCG64 seeded by SHA256(sequence,frame,seed); same frame reused across windows",
        reference_prevalence=json.loads(
            (PROJECT_ROOT / "runs/diagnostic_v1/spec.json").read_text()
        )["reference_prevalence"],
        feature_population="current official points; voxel mean from all five scans mapped back to each current point; feature distance subsets do not replace score evaluation",
        quantization_probe="exact float32 reconstruction on k/3500, suggested by raw train/206 frame 0; raw val/141 frame 400 inspected preliminarily but excluded from formal current-frame statistics",
        samples=[
            {
                k: r[k]
                for k in (
                    "view",
                    "sequence_id",
                    "current_frame",
                    "frame_ids",
                    "check_seed",
                    "entity",
                    "world_identity",
                )
            }
            for r in samples
        ],
        predicted_full_window_scores=predicted_points,
        prediction_peak_bytes=prediction_peak,
        feature_peak_bytes=feature_peak,
        resources=snapshot,
    )
    if not (args.output / "spec.json").exists():
        _atomic_json(args.output / "spec.json", spec)
    else:
        previous = json.loads((args.output / "spec.json").read_text())
        for key in (
            "samples",
            "radial_width_m",
            "checkpoint_sha256",
            "reference_prevalence",
            "permutation",
        ):
            if previous[key] != spec[key]:
                raise ValueError(f"existing intensity rule changed: {key}")
        if previous["seeds"] != list(SEEDS):
            raise ValueError("existing intensity seeds changed")
    if not (args.output / "distributions.json").exists():
        feature_statistics(args.data_root, protocol, samples, args.output, resources)
    if not args.statistics_only:
        result = run_permutations(
            args.data_root, protocol, samples, args.output, resources
        )
        plot_intensity(args.output, result)


def score_summary(rows, root, prevalence):
    def group(selected):
        records = [r["evaluation_records"] for r in selected]
        result = pooled_files(
            [root / r["file"] for r in records],
            ranges=[(r["offset"], r["count"]) for r in records],
            prevalence=prevalence,
        )
        high = sum(r["normal_high_count"] for r in selected)
        return dict(
            **result,
            frame_count=len(selected),
            normal_high_count=high,
            normal_fraction_ge_0_5=high / result["normal_count"],
            world_or_sequence_count=len({r["entity"] for r in selected}),
        )

    return dict(
        pooled=group(rows),
        entities={
            entity: group([r for r in rows if r["entity"] == entity])
            for entity in sorted({r["entity"] for r in rows})
        },
    )


def run_permutations(data_root, protocol, samples, output, resources):
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    started = time.monotonic()
    spec = json.loads((output / "spec.json").read_text())
    checkpoint = PROJECT_ROOT / spec["checkpoint"]
    if file_hash(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("the fixed epoch-seven checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference = payload["model"]
    model = AJAE(0.05).cuda().eval().requires_grad_(False)
    model.load_state_dict(reference, strict=True)
    del payload
    assert_unchanged(model, reference)
    dataset = FrozenWindowDataset(data_root, protocol, pool_name="validation")
    sequences, cache = {}, OrderedDict()
    result = dict(original={}, permutations={}, checks=[])
    for row in samples:
        record = row["evaluation_records"]
        with (Path(row["source_directory"]) / record["file"]).open("rb") as stream:
            stream.seek(record["offset"] * 8)
            values = np.fromfile(stream, np.uint64, count=record["count"])
        row["normal_high_count"] = int(
            np.count_nonzero(
                ((values & 1) == 0) & ((values >> 1) >= np.float32(0.5).view(np.uint32))
            )
        )
    for domain in ("synthetic", "real"):
        rows = [r for r in samples if r["view"] == domain]
        result["original"][domain] = score_summary(
            rows, Path(rows[0]["source_directory"]), spec["reference_prevalence"]
        )
    logs, completed = {}, {seed: [] for seed in SEEDS}
    for seed in SEEDS:
        directory = output / f"seed_{seed}"
        if directory.exists():
            completed[seed] = [
                json.loads(line)
                for line in (directory / "results.jsonl").read_text().splitlines()
            ]
            keys = [(r["sequence_id"], r["current_frame"]) for r in completed[seed]]
            if keys != [
                (r["sequence_id"], r["current_frame"]) for r in samples[: len(keys)]
            ]:
                raise ValueError("saved permutation rows are not the fixed prefix")
            retained = {directory / r["prediction"]["file"] for r in completed[seed]}
            for path in (directory / "predictions").rglob("*.npz"):
                if path not in retained:
                    pending = samples[len(keys)]
                    expected = (
                        directory
                        / "predictions"
                        / pending["view"]
                        / f"{pending['sequence_index']:03d}"
                        / f"frame_{pending['current_frame']:06d}.npz"
                    )
                    if path != expected:
                        raise ValueError("unexpected unregistered prediction")
                    path.unlink()  # Roll back only this run's interrupted writer tail.
            ends = {}
            for saved in completed[seed]:
                for key in ("evaluation_records", "normal_records"):
                    if key in saved:
                        record = saved[key]
                        ends[directory / record["file"]] = (
                            record["offset"] + record["count"]
                        ) * record.get("itemsize", 8)
            for path in (directory / "current").rglob("*.bin"):
                length = ends.get(path, 0)
                if path.stat().st_size < length:
                    raise ValueError("saved permutation score records are truncated")
                with path.open("r+b") as stream:
                    stream.truncate(length)
        else:
            directory.mkdir()
            _atomic_json(
                directory / "samples.json",
                dict(
                    checkpoint_sha256=spec["checkpoint_sha256"],
                    intensity_seed=seed,
                    radial_width_m=DISTANCE_WIDTH,
                    samples=spec["samples"],
                ),
            )
        logs[seed] = (directory / "results.jsonl").open("a", buffering=1)
    resumed_rows = {str(seed): len(rows) for seed, rows in completed.items()}
    checked = set()

    def prepare(row):
        window = load_window(row, dataset, sequences, data_root, protocol)
        return window, joint_voxelize(window)

    def retain(seed, row, window, inputs, scores, losses, scopes, elapsed):
        current = window.current_mask
        target = evaluation_targets(
            window.points.coordinates[current], window.labels.semantic[current]
        )
        if any(
            int((target == label).sum()) != row["current"][key]
            for label, key in ((0, "normal_count"), (1, "anomaly_count"))
        ):
            raise ValueError("permutation changed the fixed evaluation point set")
        sample = {
            k: row[k]
            for k in (
                "view",
                "sequence_index",
                "dataset_index",
                "sequence_id",
                "current_frame",
                "frame_ids",
                "check_seed",
                "scope",
                "entity",
                "world_identity",
            )
        }
        if "segment_index" in row:
            sample["segment_index"] = row["segment_index"]
        sample.update(
            intensity_seed=seed,
            radial_width_m=DISTANCE_WIDTH,
            normal_high_count=int(
                np.count_nonzero(scores[current][target == 0] >= 0.5)
            ),
        )
        return save_window(
            output / f"seed_{seed}",
            sample,
            window,
            scores,
            losses,
            scopes,
            dict(inference_seconds=elapsed, voxel_count=len(inputs.features)),
        )

    try:
        with (
            ThreadPoolExecutor(max_workers=1) as loader,
            ThreadPoolExecutor(max_workers=1) as writer,
        ):
            start_index = min(map(len, completed.values()))
            prepared = (
                loader.submit(prepare, samples[start_index])
                if start_index < len(samples)
                else None
            )
            pending = None
            for index in range(start_index, len(samples)):
                row = samples[index]
                resources()
                window, inputs = prepared.result()
                if index + 1 < len(samples):
                    prepared = loader.submit(prepare, samples[index + 1])
                else:
                    prepared = None
                if row["view"] not in checked:
                    original = PredictionBatch.load(
                        Path(row["source_directory"]) / row["prediction"]["file"],
                        window=window,
                    )
                    scores, _, _, _ = predict_window(
                        model, window, inputs.to("cuda"), row["check_seed"]
                    )
                    # Use the existing cached/direct model tolerance, not bitwise GPU equality.
                    np.testing.assert_allclose(
                        scores, original.anomaly_score, atol=1e-6, rtol=1e-5
                    )
                    result["checks"].append(
                        dict(
                            domain=row["view"],
                            frame=row["current_frame"],
                            original_max_abs_score_difference=float(
                                np.max(np.abs(scores - original.anomaly_score))
                            ),
                            point_count=window.points.count,
                        )
                    )
                    del original, scores
                for seed in SEEDS:
                    if index < len(completed[seed]):
                        continue
                    changed, cpu = permute_window(window, inputs, seed, cache)
                    if row["view"] not in checked:
                        independent = joint_voxelize(changed)
                        for name in (
                            "coordinates",
                            "grid_coord",
                            "point_to_voxel",
                            "features",
                            "point_features",
                        ):
                            if not torch.equal(
                                getattr(cpu, name), getattr(independent, name)
                            ):
                                raise ValueError(
                                    "intensity-only update differs from full voxel reconstruction"
                                )
                        del independent
                    scores, losses, scopes, elapsed = predict_window(
                        model,
                        changed,
                        cpu.to("cuda"),
                        row["check_seed"],
                        split_losses=True,
                    )
                    if pending is not None:
                        previous_seed, task = pending
                        saved = task.result()
                        logs[previous_seed].write(json.dumps(saved) + "\n")
                        completed[previous_seed].append(saved)
                    pending = (
                        seed,
                        writer.submit(
                            retain,
                            seed,
                            row,
                            changed,
                            cpu,
                            scores,
                            losses,
                            scopes,
                            elapsed,
                        ),
                    )
                    del changed, cpu, scores
                checked.add(row["view"])
                if (index + 1) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "permutations",
                                "completed_windows": index + 1,
                                "total_windows": len(samples),
                                "forward_calls": (index + 1) * len(SEEDS),
                            }
                        ),
                        flush=True,
                    )
            if pending is not None:
                seed, task = pending
                saved = task.result()
                logs[seed].write(json.dumps(saved) + "\n")
                completed[seed].append(saved)
    finally:
        for log in logs.values():
            log.close()
    assert_unchanged(model, reference)
    for seed, rows in completed.items():
        if len(rows) != len(samples):
            raise ValueError("incomplete intensity permutation")
        result["permutations"][str(seed)] = {}
        for domain in ("synthetic", "real"):
            result["permutations"][str(seed)][domain] = score_summary(
                [r for r in rows if r["view"] == domain],
                output / f"seed_{seed}",
                spec["reference_prevalence"],
            )
    result.update(
        status="completed",
        model_parameters_and_buffers_unchanged=True,
        optimizer_updates=0,
        intervention_forward_calls=len(samples) * len(SEEDS),
        original_verification_forward_calls=len(checked),
        resumed_registered_rows=resumed_rows,
        wall_seconds=time.monotonic() - started,
        final_resources=resources(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    _atomic_json(output / "summary.json", result)
    print(
        json.dumps(
            {"event": "intensity_completed", "wall_seconds": result["wall_seconds"]}
        ),
        flush=True,
    )
    return result


def plot_intensity(output, summary):
    """Render only saved statistics; the figures never rerun or select predictions."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    for family, filename in (
        ("SimSun", "simsun.ttc"),
        ("Times New Roman", "times.ttf"),
    ):
        font_manager.fontManager.addfont(Path("/mnt/c/Windows/Fonts") / filename)
        font_manager.findfont(family, fallback_to_default=False)
    distributions = json.loads((output / "distributions.json").read_text())
    conditions = [summary["original"], *summary["permutations"].values()]
    labels = ["原始", *[f"置换 {seed}" for seed in SEEDS]]
    colors = ["#333333", "#0072B2", "#D55E00", "#009E73"]
    metrics = (
        ("standardized_AP", "统一占比 AP"),
        ("AUROC", "AUROC"),
        ("recall_at_fpr_limit", "误报率不超过 1% 时的最高召回"),
        ("normal_fraction_ge_0_5", "正常点分数不低于 0.5 的比例"),
    )

    def value(result, key):
        if key == "recall_at_fpr_limit":
            return result[key]["recall"]
        return result[key] * (100 if key == "normal_fraction_ge_0_5" else 1)

    with (
        plt.rc_context(
            {"font.family": ["Times New Roman", "SimSun"], "pdf.fonttype": 42}
        ),
        PdfPages(output / "results.pdf") as pdf,
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, (key, title) in zip(axes.flat, metrics, strict=True):
            for i, condition in enumerate(conditions):
                bars = ax.bar(
                    np.arange(2) + (i - 1.5) * 0.19,
                    [value(condition[d]["pooled"], key) for d in ("synthetic", "real")],
                    width=0.18,
                    label=labels[i],
                    color=colors[i],
                )
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3, rotation=90)
            ax.set_xticks([0, 1], ["合成：113 窗", "真实：303 窗"])
            ax.set_ylabel("百分比")
            ax.set_title(title)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
            ax.grid(axis="y", alpha=0.2)
        axes[0, 0].legend(loc="upper right", fontsize=9)
        fig.suptitle(
            "固定第七轮模型的强度置换结果\n同一扫描、每 2.5 米距离段内置换；统一异常占比为 0.0451738%",
            fontsize=14,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        domains = [
            ("training", "训练合成"),
            ("synthetic", "验证合成"),
            ("real", "真实验证"),
        ]
        for column, channel in enumerate(("point", "voxel")):
            for i, (domain, name) in enumerate(domains):
                pooled = distributions["populations"][domain]["pooled"]
                for label, text, style in (
                    ("normal", "正常", "--"),
                    ("anomaly", "异常", "-"),
                ):
                    q = pooled[label][channel]["quantiles"]
                    axes[0, column].plot(
                        np.array(QUANTILES) * 100,
                        [q[str(x)] for x in QUANTILES],
                        style,
                        marker=".",
                        color=colors[i + 1],
                        label=f"{name}／{text}",
                    )
            axes[0, column].set_title(
                "逐点强度" if column == 0 else "五帧体素平均强度，按当前点加权"
            )
            axes[0, column].set_xlabel("分位点（%）")
            axes[0, column].set_ylabel("原始强度单位")
            axes[0, column].grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        for label, offset, color, text in (
            ("normal", -0.18, "#888888", "正常"),
            ("anomaly", 0.18, "#D55E00", "异常"),
        ):
            heights = [
                100
                * distributions["populations"][d]["pooled"][label]["point"][
                    "grid_1_over_3500_fraction"
                ]
                for d, _ in domains
            ]
            bars = axes[1, 0].bar(
                np.arange(3) + offset, heights, width=0.35, color=color, label=text
            )
            axes[1, 0].bar_label(bars, fmt="%.3f", fontsize=9)
        axes[1, 0].set_xticks(range(3), [name for _, name in domains])
        axes[1, 0].set_ylim(0, 115)
        axes[1, 0].set_ylabel("点数占比（%）")
        axes[1, 0].set_title("逐点强度精确落在 k/3500 网格上的比例")
        axes[1, 0].legend(fontsize=9)
        for i, (domain, name) in enumerate(domains):
            pooled = distributions["populations"][domain]["pooled"]
            for label, style in (("normal", "--"), ("anomaly", "-")):
                values = [
                    pooled[label]["distance_bins"][str(b)]["point"] for b in range(4, 8)
                ]
                medians = [
                    v["quantiles"]["0.5"] if v["count"] else np.nan for v in values
                ]
                axes[1, 1].plot(
                    np.arange(4),
                    medians,
                    style,
                    marker="o",
                    color=colors[i + 1],
                    label=name if label == "anomaly" else None,
                )
        axes[1, 1].set_xticks(
            range(4), ["[10,12.5)", "[12.5,15)", "[15,17.5)", "[17.5,20)"]
        )
        axes[1, 1].set_xlabel("逐点距离（米）；实线异常，虚线正常")
        axes[1, 1].set_ylabel("逐点强度中位数")
        axes[1, 1].set_title("相近逐点距离的特征分布；空缺表示没有异常点")
        axes[1, 1].legend(fontsize=9)
        fig.suptitle(
            "所选条件下的输入强度分布\n训练 170 窗、验证合成 113 窗、真实 303 窗；特征分层不替换原指标点集",
            fontsize=14,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 10), layout="constrained")
        for row, (domain, name) in enumerate(
            (("synthetic", "合成世界"), ("real", "真实序列"))
        ):
            entities = summary["original"][domain]["entities"]
            for column, (key, title) in enumerate([metrics[i] for i in (0, 2, 3)]):
                ax = axes[row, column]
                for seed in SEEDS:
                    delta = [
                        value(
                            summary["permutations"][str(seed)][domain]["entities"][e],
                            key,
                        )
                        - value(base, key)
                        for e, base in entities.items()
                    ]
                    ax.scatter(
                        delta,
                        np.arange(len(entities)) + (seed - 1) * 0.18,
                        s=18,
                        color=colors[seed + 1],
                        label=f"置换 {seed}",
                    )
                short = [
                    e.replace("synthetic/validation/", "").replace("segment_", "")
                    for e in entities
                ]
                ax.set_yticks(range(len(entities)), short)
                ax.invert_yaxis()
                ax.axvline(0, color="0.5", linewidth=0.8)
                ax.grid(axis="x", alpha=0.2)
                ax.set_title(f"{name}：{title}")
                ax.set_xlabel("置换减原始（百分点）")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "逐世界与逐序列的变化\n三个种子全部保留；每个实体分别统一异常占比，不能用实体均值替代合并指标",
            fontsize=14,
        )
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
