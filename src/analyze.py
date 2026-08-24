#!/usr/bin/env python3
"""Training-free geometry diagnostics for AJAE history correspondence.

The retained functions answer only two current scientific questions: whether a
historical sparse interpolation reaches generated-object evidence, and how a
wide p16 search trades correspondence recall against background candidates.
They do not assign history value labels or define a classification objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


HISTORY_ALIGNMENT_STRIDES = (16, 8, 4)
STU_VOXEL_SIZE_METRES = 0.05
P16_CANDIDATE_SPEEDS_MPS = (12.0, 15.0, 16.0, 18.0, 20.0)
P16_CANDIDATE_MARGINS_METRES = (0.0, 0.8, 1.0, 1.2, 1.6)
P16_CANDIDATE_TOP_K = (32, 64, 128, 256, 384, 448, 512, 1024)
P16_GRAPH_POLICY_IDS = ("v15_m08", "v15_m12", "v16_m10", "v15_m16")


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("distribution values must be a non-empty finite vector")
    quantiles = np.quantile(array, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "maximum": float(quantiles[6]),
    }


def _voxel_occupancy(
    coordinates: np.ndarray,
    anomaly: np.ndarray,
    stride: int,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Count generated-object and total returns in actual STU sparse cells."""

    if (
        coordinates.dtype != np.float64
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
    ):
        raise TypeError("coordinates must be float64[N,3]")
    if anomaly.dtype != np.bool_ or anomaly.shape != (coordinates.shape[0],):
        raise TypeError("anomaly must be bool[N]")
    if stride not in HISTORY_ALIGNMENT_STRIDES:
        raise ValueError("unsupported history-alignment stride")
    quantized = np.floor(coordinates / STU_VOXEL_SIZE_METRES).astype(np.int64)
    keys = np.floor_divide(quantized, stride) * stride
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    total = np.bincount(inverse, minlength=unique.shape[0])
    anomalous = np.bincount(
        inverse,
        weights=anomaly.astype(np.int64),
        minlength=unique.shape[0],
    ).astype(np.int64)
    return {
        tuple(map(int, key)): (int(anomaly_count), int(total_count))
        for key, anomaly_count, total_count in zip(
            unique, anomalous, total, strict=True
        )
    }


def _current_occupancy_statistics(
    occupancy: Mapping[tuple[int, int, int], tuple[int, int]],
) -> dict[str, object]:
    """Measure how generated-object returns share cells with normal returns."""

    anomalous = [counts for counts in occupancy.values() if counts[0] > 0]
    if not anomalous:
        raise ValueError("current occupancy contains no anomaly voxel")
    mixed = sum(
        anomaly_count < total_count for anomaly_count, total_count in anomalous
    )
    anomaly_points = sum(anomaly_count for anomaly_count, _ in anomalous)
    mixed_points = sum(
        anomaly_count
        for anomaly_count, total_count in anomalous
        if anomaly_count < total_count
    )
    pure = len(anomalous) - mixed
    return {
        "anomaly_voxels": len(anomalous),
        "pure_anomaly_voxels": pure,
        "mixed_anomaly_voxels": mixed,
        "anomaly_points": anomaly_points,
        "anomaly_points_in_mixed_voxels": mixed_points,
        "has_mixed_anomaly_voxel": mixed > 0,
        "has_no_pure_anomaly_voxel": pure == 0,
    }


def _p16_candidate_policies(age: int) -> tuple[dict[str, object], ...]:
    if age not in range(1, 5):
        raise ValueError("p16 candidate age must lie in 1..4")
    return tuple(
        {
            "id": f"v{int(speed):02d}_m{int(round(10 * margin)):02d}",
            "maximum_speed_mps": speed,
            "margin_metres": margin,
            "radius_metres": speed * 0.1 * age + margin,
        }
        for speed in P16_CANDIDATE_SPEEDS_MPS
        for margin in P16_CANDIDATE_MARGINS_METRES
    )


def _p16_candidate_query_records(
    queries: np.ndarray,
    occupancy: Mapping[tuple[int, int, int], tuple[int, int]],
    *,
    age: int,
) -> list[dict[str, object]]:
    """Describe recall and background load before any learned matching score."""

    if queries.dtype != np.float64 or queries.ndim != 2 or queries.shape[1] != 3:
        raise TypeError("p16 candidate queries must be float64[N,3]")
    if not occupancy:
        raise ValueError("p16 candidate source occupancy cannot be empty")
    keys = np.asarray(tuple(occupancy), dtype=np.int64)
    counts = np.asarray(tuple(occupancy.values()), dtype=np.int64)
    locations = keys.astype(np.float64) * STU_VOXEL_SIZE_METRES
    anomaly_only = counts[:, 0] == counts[:, 1]
    anomaly = counts[:, 0] > 0
    mixed = anomaly & ~anomaly_only
    normal_only = ~anomaly
    distance = np.linalg.norm(
        queries[:, None, :] * STU_VOXEL_SIZE_METRES - locations[None, :, :],
        axis=2,
    )
    records: list[dict[str, object]] = []
    for query_index, distances in enumerate(distance):
        if bool(anomaly.any()):
            nearest = float(np.min(distances[anomaly]))
            closer = distances < nearest - 1.0e-9
            tied = np.abs(distances - nearest) <= 1.0e-9
            best_rank: int | None = int(np.count_nonzero(closer)) + 1
            guaranteed_rank: int | None = (
                int(np.count_nonzero(closer))
                + int(np.count_nonzero(tied & normal_only))
                + 1
            )
        else:
            nearest = None
            best_rank = None
            guaranteed_rank = None
        policy_counts = {}
        for policy in _p16_candidate_policies(age):
            within = distances <= float(policy["radius_metres"]) + 1.0e-9
            policy_counts[str(policy["id"])] = {
                "total": int(np.count_nonzero(within)),
                "anomaly_only": int(np.count_nonzero(within & anomaly_only)),
                "mixed": int(np.count_nonzero(within & mixed)),
                "normal_only": int(np.count_nonzero(within & normal_only)),
            }
        records.append(
            {
                "query_index": query_index,
                "history_anomaly_visible": bool(anomaly.any()),
                "source_occupied_voxels": int(keys.shape[0]),
                "source_anomaly_voxels": int(np.count_nonzero(anomaly)),
                "nearest_anomaly_distance_metres": nearest,
                "nearest_anomaly_best_rank": best_rank,
                "nearest_anomaly_guaranteed_rank": guaranteed_rank,
                "policies": policy_counts,
            }
        )
    return records


def _p16_graph_record(
    queries: np.ndarray,
    occupancy: Mapping[tuple[int, int, int], tuple[int, int]],
    *,
    age: int,
) -> dict[str, object]:
    """Count sparse candidate edges for all occupied current p16 cells."""

    if queries.dtype != np.float64 or queries.ndim != 2 or queries.shape[1] != 3:
        raise TypeError("p16 graph queries must be float64[N,3]")
    if not occupancy:
        raise ValueError("p16 graph source occupancy cannot be empty")
    keys = np.asarray(tuple(occupancy), dtype=np.int64)
    tree = cKDTree(keys.astype(np.float64) * STU_VOXEL_SIZE_METRES)
    policies = {
        str(policy["id"]): policy
        for policy in _p16_candidate_policies(age)
        if policy["id"] in P16_GRAPH_POLICY_IDS
    }
    results = {}
    for policy_id, policy in policies.items():
        counts = np.asarray(
            tree.query_ball_point(
                queries * STU_VOXEL_SIZE_METRES,
                float(policy["radius_metres"]),
                return_length=True,
            ),
            dtype=np.int64,
        )
        results[policy_id] = {
            "radius_metres": policy["radius_metres"],
            "candidate_edges": int(counts.sum()),
            "candidate_count": _distribution(counts.astype(np.float64)),
            "zero_candidate_queries": int(np.count_nonzero(counts == 0)),
            "capped_candidate_edges": {
                str(top_k): int(np.minimum(counts, top_k).sum())
                for top_k in P16_CANDIDATE_TOP_K
            },
        }
    return {
        "current_query_voxels": int(queries.shape[0]),
        "source_occupied_voxels": int(keys.shape[0]),
        "policies": results,
    }


def _interpolation_support(
    queries: np.ndarray,
    occupancy: Mapping[tuple[int, int, int], tuple[int, int]],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce sparse trilinear support using occupancy, not learned features."""

    if queries.dtype != np.float64 or queries.ndim != 2 or queries.shape[1] != 3:
        raise TypeError("queries must be float64[N,3]")
    if stride not in HISTORY_ALIGNMENT_STRIDES:
        raise ValueError("unsupported history-alignment stride")
    scaled = queries / stride
    lower_index = np.floor(scaled).astype(np.int64)
    fraction = scaled - lower_index
    lower = lower_index * stride
    anomaly_mass = np.zeros(queries.shape[0], dtype=np.float64)
    normal_mass = np.zeros(queries.shape[0], dtype=np.float64)
    support_mass = np.zeros(queries.shape[0], dtype=np.float64)
    for corner in range(8):
        bits = np.asarray(
            ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1), dtype=np.int64
        )
        weight = np.prod(
            np.where(bits[None, :] == 1, fraction, 1.0 - fraction), axis=1
        )
        for index in np.flatnonzero(weight > 0.0):
            counts = occupancy.get(
                tuple(map(int, lower[index] + bits * stride))
            )
            if counts is None:
                continue
            anomaly_count, total_count = counts
            anomaly_fraction = anomaly_count / total_count
            support_mass[index] += weight[index]
            anomaly_mass[index] += weight[index] * anomaly_fraction
            normal_mass[index] += weight[index] * (1.0 - anomaly_fraction)
    return anomaly_mass, normal_mass, support_mass


def _support_statistics(
    anomaly_mass: np.ndarray,
    normal_mass: np.ndarray,
    support_mass: np.ndarray,
) -> dict[str, object]:
    if not (
        anomaly_mass.shape == normal_mass.shape == support_mass.shape
        and anomaly_mass.ndim == 1
    ):
        raise TypeError("support masses must be aligned vectors")
    valid = support_mass > 1.0e-12
    anomaly = anomaly_mass > 1.0e-12
    normal = normal_mass > 1.0e-12
    categories = {
        "anomaly_only": int(np.count_nonzero(valid & anomaly & ~normal)),
        "anomaly_and_normal": int(np.count_nonzero(valid & anomaly & normal)),
        "normal_only": int(np.count_nonzero(valid & ~anomaly & normal)),
        "empty": int(np.count_nonzero(~valid)),
    }
    count = int(support_mass.size)
    valid_count = int(np.count_nonzero(valid))
    anomaly_hit = categories["anomaly_only"] + categories["anomaly_and_normal"]
    purity = np.divide(
        anomaly_mass,
        support_mass,
        out=np.zeros_like(anomaly_mass),
        where=valid,
    )
    return {
        "queries": count,
        "valid_support": valid_count,
        "anomaly_hit": anomaly_hit,
        "categories": categories,
        "valid_support_rate": valid_count / count if count else 0.0,
        "anomaly_hit_rate": anomaly_hit / count if count else 0.0,
        "anomaly_only_rate": categories["anomaly_only"] / count if count else 0.0,
        "mixed_rate": categories["anomaly_and_normal"] / count if count else 0.0,
        "normal_only_rate": categories["normal_only"] / count if count else 0.0,
        "empty_rate": categories["empty"] / count if count else 0.0,
        "support_mass_sum": float(support_mass.sum()),
        "anomaly_purity_sum_valid": float(purity[valid].sum()),
        "mean_support_mass": float(support_mass.mean()) if count else 0.0,
        "mean_anomaly_purity_given_valid": (
            float(purity[valid].mean()) if valid_count else 0.0
        ),
    }


def _merge_support_statistics(
    values: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    queries = sum(int(value["queries"]) for value in values)
    valid = sum(int(value["valid_support"]) for value in values)
    categories = {
        name: sum(
            int(value["categories"][name]) for value in values  # type: ignore[index]
        )
        for name in ("anomaly_only", "anomaly_and_normal", "normal_only", "empty")
    }
    anomaly_hit = categories["anomaly_only"] + categories["anomaly_and_normal"]
    support_sum = sum(float(value["support_mass_sum"]) for value in values)
    purity_sum = sum(
        float(value["anomaly_purity_sum_valid"]) for value in values
    )
    return {
        "queries": queries,
        "valid_support": valid,
        "anomaly_hit": anomaly_hit,
        "categories": categories,
        "valid_support_rate": valid / queries if queries else 0.0,
        "anomaly_hit_rate": anomaly_hit / queries if queries else 0.0,
        "anomaly_only_rate": categories["anomaly_only"] / queries if queries else 0.0,
        "mixed_rate": categories["anomaly_and_normal"] / queries if queries else 0.0,
        "normal_only_rate": categories["normal_only"] / queries if queries else 0.0,
        "empty_rate": categories["empty"] / queries if queries else 0.0,
        "support_mass_sum": support_sum,
        "anomaly_purity_sum_valid": purity_sum,
        "mean_support_mass": support_sum / queries if queries else 0.0,
        "mean_anomaly_purity_given_valid": purity_sum / valid if valid else 0.0,
    }


def summarize_history_alignment(source: Path) -> dict[str, object]:
    """Recompute fixed-versus-Oracle summaries from retained raw records."""

    path = source.expanduser().resolve(strict=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not str(document.get("format", "")).startswith(
        "ajae-history-alignment-audit-"
    ):
        raise ValueError("source is not an AJAE history-alignment audit")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("history-alignment audit has no raw records")
    summaries: dict[str, object] = {}
    for stride in HISTORY_ALIGNMENT_STRIDES:
        fixed_parts: list[Mapping[str, object]] = []
        oracle_parts: list[Mapping[str, object]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("history-alignment record is malformed")
            scales = record.get("scales")
            if not isinstance(scales, Mapping):
                raise ValueError("history-alignment record lacks scale data")
            scale = scales.get(str(stride))
            if not isinstance(scale, Mapping):
                raise ValueError(f"history-alignment record lacks p{stride} data")
            fixed = scale.get("fixed")
            oracle = scale.get("oracle")
            if not isinstance(fixed, Mapping) or not isinstance(oracle, Mapping):
                raise ValueError("history-alignment support record is malformed")
            fixed_parts.append(fixed)
            oracle_parts.append(oracle)
        fixed_summary = _merge_support_statistics(fixed_parts)
        oracle_summary = _merge_support_statistics(oracle_parts)
        summaries[str(stride)] = {
            "fixed": fixed_summary,
            "oracle": oracle_summary,
            "anomaly_hit_rate_gain": float(oracle_summary["anomaly_hit_rate"])
            - float(fixed_summary["anomaly_hit_rate"]),
            "claim_boundary": (
                "Occupancy reachability diagnoses correspondence geometry; it is "
                "not classification accuracy or evidence that every extra history "
                "frame helps."
            ),
        }
    return {
        "format": "ajae-history-alignment-summary-v1",
        "source_format": document["format"],
        "records": len(records),
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize retained AJAE history-alignment records."
    )
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            summarize_history_alignment(arguments.source),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
