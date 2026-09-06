"""Aggregate observation profiles and export the STU19 CSV tables."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from .data import _atomic_json
from .profile import QUANTILES, describe


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "-inf" if value < 0 else "inf" if value > 0 else None
    return value


def physical_unit(meta):
    factor, key = meta["factor"], meta["metric"]
    if key == "background_intensity_variance":
        return "原始强度单位的平方"
    if "intensity" in key or factor in ("C01", "C03"):
        return "原始强度单位"
    if key in (
        "azimuth",
        "elevation",
        "azimuth_span",
        "elevation_span",
        "azimuth_gap",
        "rotation",
    ):
        return "弧度"
    if any(
        x in key for x in ("distance", "displacement", "residual", "ground_height")
    ) or key in ("x", "y", "z", "length", "width", "height", "translation"):
        return "米"
    if any(x in key for x in ("fraction", "compression")) or key in (
        "aspect",
        "linearity",
        "planarity",
        "scattering",
    ):
        return "比例或比值"
    return "类别编码" if meta["categorical"] else "数量"


def summarize(
    meta,
    x,
    c,
    w,
    *,
    sequence_weights=None,
    sequence_mean=None,
    sequence_bins=None,
    sequence_count=1,
):
    result = {}
    variants = [
        (
            "observation_equal",
            c,
            meta["total"] / meta["n"] if meta["n"] else None,
            meta["bin_counts"],
        )
    ]
    if meta["scope"] != "sequence_runs":
        variants.append(
            (
                "frame_equal",
                w,
                meta["frame_total"] / meta["valid_frames"]
                if meta["valid_frames"]
                else None,
                meta["bin_frame_weights"],
            )
        )
    if sequence_weights is not None:
        variants.append(
            (
                "sequence_equal",
                sequence_weights,
                sequence_mean / sequence_count if sequence_count else None,
                sequence_bins,
            )
        )
    for view, weights, mean, bins in variants:
        row = describe(meta, x, weights, mean=mean, bin_weights=bins)
        row.update(
            sequence_count=sequence_count,
            value_unit=physical_unit(meta),
            weighting=view,
            status="已统计"
            if meta["n"]
            else "没有适用观测"
            if not meta["denominator"]
            else "估计不可靠或缺失",
        )
        if meta["categorical"]:
            universe = set(map(int, x)) | {
                int((a + b) / 2)
                for a, b in zip(meta["bins"][:-1], meta["bins"][1:], strict=True)
                if np.isfinite(a + b)
            }
            count_map = dict(zip(map(int, x), map(int, c)))
            row["category_counts"] = {
                str(v): count_map.get(v, 0) for v in sorted(universe)
            }
            row["categories"] = {
                str(v): row["categories"].get(str(v), 0.0 if weights.sum() else None)
                for v in sorted(universe)
            }
        elif not meta["resolution"]:
            row["unique_count"] = len(x)
            row["repeated_value_fraction"] = (
                float(c[c > 1].sum() / c.sum()) if c.sum() else None
            )
            row["minimum_count"] = int(c[0]) if len(c) else 0
            row["maximum_count"] = int(c[-1]) if len(c) else 0
            top = np.argsort(-c, kind="stable")[:10]
            row["most_frequent_values"] = [
                dict(value=float(x[i]), count=int(c[i])) for i in top
            ]
        result[view] = row
    return result


def joint_tables(frames, windows, instances):
    tables = {
        name: {}
        for name in (
            "official_count_distance",
            "count_distance_history",
            "mix_count_distance",
            "mix_background",
            "motion_count_distance",
            "stage_count",
        )
    }
    object_ids = {}
    for row in instances:
        object_ids.setdefault((row["sequence"], row["frame"]), set()).add(
            row["instance"]
        )

    def record(name, key, frame):
        entry = tables[name].setdefault(
            key,
            dict(
                frames=0,
                anomaly_points=0,
                anomaly_in_range_points=0,
                sequences=set(),
                instances=set(),
            ),
        )
        entry["frames"] += 1
        entry["anomaly_points"] += frame["anomaly"]
        entry["anomaly_in_range_points"] += frame["anomaly_in_range"]
        entry["sequences"].add(frame["sequence"])
        entry["instances"].update(
            (frame["sequence"], i)
            for i in object_ids.get((frame["sequence"], frame["frame"]), ())
        )

    def cb(n):
        return int(np.searchsorted((1, 5, 20, 100, 500), n, side="right"))

    def db(frame):
        if frame["anomaly_in_range"]:
            return str(
                int(
                    np.searchsorted(
                        (10, 20, 35),
                        frame["anomaly_in_range_distance_median"],
                        side="right",
                    )
                )
            )
        return "outside_only" if frame["anomaly"] else "unseen"

    window_by_frame = {(w["sequence"], w["frame"]): w for w in windows}
    if len(window_by_frame) != len(windows):
        raise ValueError("duplicate output window identity")
    for frame in frames:
        record("stage_count", f"{frame['stage']}|{cb(frame['anomaly'])}", frame)
        window = window_by_frame.get((frame["sequence"], frame["frame"]))
        if window is None or window["scope"] != "complete_windows":
            continue
        count = cb(frame["anomaly_in_range"])
        distance = db(frame)
        record(
            "count_distance_history",
            f"{count}|{distance}|{window['history_visible_scans']}",
            frame,
        )
        movement = int(
            np.searchsorted((0.1, 0.5, 1, 2, 5), window["translation_m"], side="right")
        )
        record("motion_count_distance", f"{count}|{distance}|{movement}", frame)
        if frame["anomaly_in_range"] >= 5:
            record("official_count_distance", f"{count - 2}|{distance}", frame)
        if frame["anomaly"]:
            fraction = window["normal_mix_fraction"]
            mixed = (
                0
                if fraction == 0
                else 1
                if fraction <= 0.25
                else 2
                if fraction <= 0.75
                else 3
            )
            road = frame["neighbor_road_fraction"]
            background = (
                "no_neighbor"
                if road is None
                else "road_majority"
                if road >= 0.5
                else "other_majority"
            )
            record("mix_count_distance", f"{mixed}|{count}|{distance}", frame)
            record("mix_background", f"{mixed}|{background}", frame)
    # Empty valid cells remain visible; impossible count/distance combinations are not invented.
    keys = [f"{i}|{j}" for i in range(4) for j in range(4)]
    for key in keys:
        tables["official_count_distance"].setdefault(
            key,
            dict(
                frames=0,
                anomaly_points=0,
                anomaly_in_range_points=0,
                sequences=set(),
                instances=set(),
            ),
        )
    for count in range(6):
        distances = (
            ("unseen", "outside_only") if count == 0 else tuple(map(str, range(4)))
        )
        for d in distances:
            for h in range(5):
                tables["count_distance_history"].setdefault(
                    f"{count}|{d}|{h}",
                    dict(
                        frames=0,
                        anomaly_points=0,
                        anomaly_in_range_points=0,
                        sequences=set(),
                        instances=set(),
                    ),
                )
    for table in tables.values():
        for row in table.values():
            row["sequence_count"] = len(row["sequences"])
            row["instance_count"] = len(row["instances"])
            row["sequences"] = sorted(row["sequences"])
            row["instances"] = [list(v) for v in sorted(row["instances"])]
    return tables


def aggregate_profile(output):
    output = Path(output)
    if (output / "summary.json").exists():
        result = json.loads((output / "summary.json").read_text())
        if not (output / "statistics.csv").exists():
            write_long_table(output, result)
        return result
    started = time.monotonic()
    spec = json.loads((output / "spec.json").read_text())
    merged = {}
    result = dict(sequences={}, series={})
    frames = []
    windows = []
    instances = []
    episodes = []
    for seq in spec["sequences"]:
        directory = output / str(seq)
        coverage = json.loads((directory / "summary.json").read_text())
        result["sequences"][str(seq)] = dict(coverage=coverage, series={})
        for name, target in (
            ("frames", frames),
            ("windows", windows),
            ("instances", instances),
            ("stages", episodes),
        ):
            target.extend(read_rows(directory / f"{name}.jsonl"))
        with np.load(directory / "histograms.npz") as saved:
            for meta, x, c, w in saved_distributions(saved):
                key = meta["key"]
                assert int(c.sum()) == meta["n"]
                np.testing.assert_allclose(
                    w.sum(), meta["valid_frames"], atol=1e-9, rtol=1e-12
                )
                result["sequences"][str(seq)]["series"][key] = summarize(
                    meta, x, c, w, sequence_count=int(meta["n"] > 0)
                )
                if key not in merged:
                    combined = {
                        k: v
                        for k, v in meta.items()
                        if k
                        not in (
                            "n",
                            "denominator",
                            "frames",
                            "valid_frames",
                            "total",
                            "frame_total",
                            "minimum",
                            "maximum",
                            "bin_counts",
                            "bin_frame_weights",
                        )
                    }
                    combined.update(
                        n=0,
                        denominator=0,
                        frames=0,
                        valid_frames=0,
                        total=0.0,
                        frame_total=0.0,
                        minimum=None,
                        maximum=None,
                        bin_counts=np.zeros(len(meta["bins"]) - 1, np.int64),
                        bin_frame_weights=np.zeros(len(meta["bins"]) - 1),
                        counts=Counter(),
                        frame_weights=Counter(),
                        sequence_weights=Counter(),
                        sequence_bins=np.zeros(len(meta["bins"]) - 1),
                        sequence_mean=0.0,
                        sequence_count=0,
                    )
                    merged[key] = combined
                item = merged[key]
                for field in (
                    "n",
                    "denominator",
                    "frames",
                    "valid_frames",
                    "total",
                    "frame_total",
                ):
                    item[field] += meta[field]
                item["bin_counts"] += meta["bin_counts"]
                item["bin_frame_weights"] += meta["bin_frame_weights"]
                if meta["n"]:
                    item["minimum"] = (
                        meta["minimum"]
                        if item["minimum"] is None
                        else min(item["minimum"], meta["minimum"])
                    )
                    item["maximum"] = (
                        meta["maximum"]
                        if item["maximum"] is None
                        else max(item["maximum"], meta["maximum"])
                    )
                    item["sequence_count"] += 1
                    item["sequence_mean"] += meta["total"] / meta["n"]
                    item["sequence_bins"] += np.asarray(meta["bin_counts"]) / meta["n"]
                    item["sequence_weights"].update(
                        dict(zip(x.tolist(), (c / meta["n"]).tolist()))
                    )
                item["counts"].update(dict(zip(x.tolist(), c.tolist())))
                item["frame_weights"].update(dict(zip(x.tolist(), w.tolist())))
        print(json.dumps(dict(event="aggregated_sequence", sequence=seq)), flush=True)
    for key, item in merged.items():
        x = np.array(sorted(item["counts"]))
        c = np.array([item["counts"][v] for v in x], np.int64)
        w = np.array([item["frame_weights"][v] for v in x])
        s = np.array([item["sequence_weights"][v] for v in x])
        meta = {
            k: v
            for k, v in item.items()
            if k
            not in (
                "counts",
                "frame_weights",
                "sequence_weights",
                "sequence_mean",
                "sequence_bins",
                "sequence_count",
            )
        }
        result["series"][key] = summarize(
            meta,
            x,
            c,
            w,
            sequence_weights=s,
            sequence_mean=item["sequence_mean"],
            sequence_bins=item["sequence_bins"],
            sequence_count=item["sequence_count"],
        )
    result["joint"] = joint_tables(frames, windows, instances)
    totals = {
        key: sum(r[key] for r in frames)
        for key in (
            "slots",
            "visible",
            "zero_slots",
            "normal",
            "ignore",
            "anomaly",
            "anomaly_in_range",
            "unknown_instance_points",
        )
    }
    totals.update(
        sequences=len(spec["sequences"]),
        frames=len(frames),
        complete_windows=sum(w["scope"] == "complete_windows" for w in windows),
        startup_windows=sum(w["scope"] == "startup_windows" for w in windows),
        states=np.bincount([r["state"] for r in frames], minlength=4).tolist(),
        official_anomaly_points=sum(
            r["anomaly_in_range"] for r in frames if r["state"] == 3
        ),
        official_normal_points=sum(
            r["normal_in_range"] for r in frames if r["state"] == 3
        ),
        normal_stage_evaluation_points=sum(
            r["normal_in_range"] for r in frames if r["state"] == 0
        ),
        whole_window_unseen=sum(
            w["all_unseen"] for w in windows if w["scope"] == "complete_windows"
        ),
        instance_frames=len(instances),
        labelled_instances=len({(r["sequence"], r["instance"]) for r in instances}),
        shape_status=dict(Counter(r["shape_status"] for r in instances)),
        ground_status=dict(Counter(r["ground_status"] for r in instances)),
        static_sampled=sum(
            w["static_sampled"] for w in windows if w["scope"] == "complete_windows"
        ),
        static_matched=sum(
            w["static_matched"] for w in windows if w["scope"] == "complete_windows"
        ),
    )
    expected = dict(
        frames=8659,
        complete_windows=8583,
        startup_windows=76,
        slots=1134952448,
        visible=987807187,
        states=[5114, 465, 1120, 1960],
        official_anomaly_points=87499,
        official_normal_points=193792470,
        normal_stage_evaluation_points=366706288,
    )
    synthetic = spec.get("population") in ("train", "validation")
    if synthetic:
        expected = dict(
            frames=spec["frames"],
            complete_windows=spec["complete_windows"],
            startup_windows=0,
            whole_window_unseen=spec.get(
                "expected_whole_window_unseen",
                20 if spec["population"] == "train" else 10,
            ),
        )
        totals.update(
            worlds=len(spec["sequences"]),
            synthetic_versions=spec["synthetic_versions"],
            background_source_sequences=1,
            context_frames=4 * len(spec["sequences"]),
        )
    for key, value in expected.items():
        if totals[key] != value:
            raise ValueError(
                f"raw profile disagrees with existing evidence: {key}: {totals[key]} != {value}"
            )
    if sum(r["length"] for r in episodes) != spec["frames"]:
        raise ValueError("observation stages do not partition all frames")
    joint = result["joint"]["official_count_distance"]
    if not synthetic and (
        sum(r["frames"] for r in joint.values()) != 1956
        or sum(r["anomaly_in_range_points"] for r in joint.values()) != 87398
    ):
        raise ValueError("full-history official subset changed")
    if synthetic:
        frame_index = {(r["sequence"], r["frame"]): r for r in frames}
        legal = [frame_index[(w["sequence"], w["frame"])] for w in windows]
        if sum(
            r["frames"] for r in result["joint"]["count_distance_history"].values()
        ) != len(legal):
            raise ValueError("joint conditions do not partition legal windows")
        if sum(r["frames"] for r in joint.values()) != sum(
            r["state"] == 3 for r in legal
        ):
            raise ValueError("qualified joint cells disagree with legal current frames")
        totals["legal_current_states"] = np.bincount(
            [r["state"] for r in legal], minlength=4
        ).tolist()
        totals["legal_official_anomaly_points"] = sum(
            r["anomaly_in_range"] for r in legal if r["state"] == 3
        )
        totals["legal_official_normal_points"] = sum(
            r["normal_in_range"] for r in legal if r["state"] == 3
        )
        if (
            spec["population"] == "validation"
            and spec.get("pool_format", "ajae-synthetic-pool-manifest")
            == "ajae-synthetic-pool-manifest"
            and (
                totals["legal_current_states"][3] != 2278
                or totals["legal_official_anomaly_points"] != 1207724
                or totals["legal_official_normal_points"] != 162980036
            )
        ):
            raise ValueError(
                "synthetic validation profile differs from the saved official evaluation population"
            )
    result.update(
        status="completed",
        totals=totals,
        reconciled_existing_totals=expected,
        aggregate_seconds=time.monotonic() - started,
        model_forward_calls=0,
        parameter_updates=0,
        definitions=spec,
    )
    result = finite_json(result)
    _atomic_json(output / "summary.json", result)
    write_long_table(output, result)
    return result


def saved_distributions(saved):
    for i, meta in enumerate(json.loads(str(saved["catalog"]))):
        x, c, w = saved[f"x{i}"], saved[f"c{i}"], saved[f"w{i}"]
        yield meta, x, c, w
        if meta["metric"] == "background_intensity_std":
            # Nonnegative std -> variance is monotone; reuse every observed value and weight.
            variance = x * x
            edges = np.asarray(meta["bins"])
            derived = dict(
                meta,
                key=meta["key"].replace("_std|", "_variance|"),
                metric="background_intensity_variance",
                bins=edges * abs(edges),
                minimum=meta["minimum"] ** 2 if meta["n"] else None,
                maximum=meta["maximum"] ** 2 if meta["n"] else None,
                total=float(variance @ c),
                frame_total=float(variance @ w),
            )
            yield derived, variance, c, w


def write_long_table(output, result):
    columns = (
        "sequence_id",
        "factor_id",
        "scope",
        "label_group",
        "metric",
        "weighting",
        "unit",
        "value_unit",
        "statistic",
        "value",
        "lower",
        "upper",
        "n",
        "denominator",
        "frames",
        "valid_frames",
        "sequence_count",
        "status",
        "source_identity",
    )
    with (output / "statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        populations = [
            ("ALL", result["series"]),
            *[(seq, data["series"]) for seq, data in result["sequences"].items()],
        ]
        for seq, series in populations:
            for key, views in series.items():
                for view, data in views.items():
                    base = dict(
                        sequence_id=seq,
                        factor_id=data["factor"],
                        scope=data["scope"],
                        label_group=data["group"],
                        metric=data["metric"],
                        weighting=view,
                        unit=data["unit"],
                        value_unit=data["value_unit"],
                        n=data["n"],
                        denominator=data["denominator"],
                        frames=data["frames"],
                        valid_frames=data["valid_frames"],
                        sequence_count=data["sequence_count"],
                        status=data["status"],
                        source_identity=f"summary.json:{seq}:{key}",
                    )
                    for name in (
                        "minimum",
                        "maximum",
                        "mean",
                        "missing_fraction",
                        "empty_frame_fraction",
                    ):
                        writer.writerow(
                            dict(base, statistic=name, value=data.get(name))
                        )
                    for q, quantile in data["quantiles"].items():
                        writer.writerow(
                            dict(base, statistic=f"P{int(float(q) * 100)}", **quantile)
                        )
                    if data["categorical"]:
                        for category, count in data["category_counts"].items():
                            writer.writerow(
                                dict(
                                    base,
                                    statistic=f"category_{category}_count",
                                    value=count,
                                )
                            )
                            writer.writerow(
                                dict(
                                    base,
                                    statistic=f"category_{category}_fraction",
                                    value=data["categories"][category],
                                )
                            )
                    else:
                        for i, fraction in enumerate(data["bin_fraction"]):
                            writer.writerow(
                                dict(
                                    base,
                                    statistic=f"bin_{i}_fraction",
                                    value=fraction,
                                    lower=data["bins"][i],
                                    upper=data["bins"][i + 1],
                                )
                            )


SCOPE = {
    "all_frames": "全部原始帧",
    "complete_frames": "当前帧序号至少为4",
    "startup_frames": "启动帧0至3",
    "complete_windows": "完整五帧窗口",
    "startup_windows": "启动窗口",
    "sequence_runs": "连续观测阶段",
}
VIEW = {
    "observation_equal": "统计单位等权",
    "frame_equal": "有适用观测的帧等权",
    "sequence_equal": "有适用观测的序列等权",
}
GROUP = {
    "all": "全部",
    "normal": "正常",
    "anomaly": "异常",
    "ignore": "忽略",
    "current_anomaly": "当前异常占据体素",
    "all_current": "全部当前占据体素",
    "point_weighted": "当前异常点加权",
    "prefix": "未见回波前缀",
    "visible": "连续可见段",
    "gap": "可见间断段",
    "tail": "最终无回波尾段",
    "entire_unseen": "整条序列均未见回波",
}
UNIT = {
    "point": "原始回波点",
    "frame": "帧",
    "instance_frame": "实例×帧",
    "anomaly_point": "原始异常点",
    "anomaly_voxel": "联合异常占据体素",
    "current_anomaly_point": "当前异常点",
    "current_anomaly_voxel": "当前异常占据体素",
    "current_point": "当前点",
    "current_voxel": "当前占据体素",
    "newly_mixed_anomaly_voxel": "历史新增正常混合体素",
    "observed_run": "观测阶段",
    "sampled_current_static_point": "抽样当前静态点",
    "unique_neighbor_point_per_frame": "每帧去重的邻近正常点",
    "voxel": "联合非空体素",
}
METRIC = {
    "slots": "原始槽数",
    "visible": "实际回波数",
    "zero_slots": "零坐标槽数",
    "normal": "正常点数",
    "ignore": "忽略点数",
    "anomaly": "全距离异常点数",
    "anomaly_in_range": "范围内异常点数",
    "visible_fraction": "实际回波占原始槽比例",
    "zero_slot_fraction": "零坐标槽比例",
    "ignore_fraction": "忽略点占回波比例",
    "normal_fraction": "正常点占回波比例",
    "anomaly_fraction": "异常点占回波比例",
    "current_state": "当前四态",
    "current_unseen": "当前未见异常回波",
    "outside_only": "异常全部位于范围外",
    "one_to_four": "范围内1至4异常点",
    "eligible": "范围内至少5异常点",
    "whole_window_unseen": "整窗未见异常回波",
    "distance": "逐点距离",
    "anomaly_distance_median": "每帧全距离异常距离中位数",
    "anomaly_in_range_distance_median": "每帧范围内异常距离中位数",
    "x": "前向坐标",
    "y": "横向坐标",
    "z": "竖向坐标",
    "azimuth": "方位角",
    "elevation": "俯仰角",
    "visible_instance_count": "可见实例数",
    "length": "可见水平长边",
    "width": "可见水平短边",
    "height": "可见竖向跨度",
    "aspect": "可见水平长短边比",
    "linearity": "可见点集线性度",
    "planarity": "可见点集平面度",
    "scattering": "可见点集散乱度",
    "same_instance_neighbor_distance": "同实例最近邻距",
    "same_instance_neighbors_r0.25": "同实例0.25米邻居数",
    "azimuth_span": "可见方位角跨度",
    "elevation_span": "可见俯仰角跨度",
    "azimuth_gap": "可见方位内部最大间隙",
    "intensity": "原始逐点强度",
    "grid_match": "精确落在实测强度网格",
    "zero": "强度为零",
    "intensity_contrast": "异常强度减邻域正常均值",
    "background_intensity_std": "邻域正常强度标准差",
    "background_intensity_variance": "邻域正常强度方差",
    "raw_semantic": "全部回波语义",
    "normal_semantic": "正常点语义",
    "neighbor_semantic": "邻近正常点语义",
    "ground_height_p05": "可见点离地高度的第5百分位",
    "ground_height_median": "可见点离地高度中位数",
    "nearest_normal_distance": "最近正常点距离",
    "normal_neighbors_r0.5": "0.5米正常邻居数",
    "history_anomaly": "历史四帧异常点数",
    "history_anomaly_fraction": "历史异常点占五帧异常点比例",
    "history_visible_scans": "历史可见扫描数",
    "visibility_pattern": "五位类别可见模式",
    "stage": "连续阶段类别",
    "first_in_visible_run": "观测可见段首帧",
    "stage_length": "阶段长度",
    "translation": "最早到当前扫描平移量",
    "rotation": "最早到当前扫描旋转角",
    "sampled_static_distance": "抽样静态点同语义匹配残差",
    "static_match_fraction": "抽样静态点匹配比例",
    "current_anomaly_voxels": "当前异常占据体素数",
    "joint_anomaly_voxels": "联合异常占据体素数",
    "joint_anomaly_voxel_increment": "历史增加异常体素数",
    "current_anomaly_compression": "当前异常点数除以占据体素数",
    "joint_anomaly_compression": "五帧异常点数除以联合占据体素数",
    "all_voxel_label_presence": "全部体素三类成员存在编码",
    "anomaly_voxel_mix": "联合异常占据体素混合类别",
    "current_anomaly_point_mix": "当前异常点所在体素混合类别",
    "member_fraction": "联合异常体素内成员比例",
    "new_normal": "历史新增正常混合",
    "scan_hits": "扫描命中五位编码",
    "anomaly_history_hits": "当前异常体素历史异常扫描数",
    "mean_displacement": "联合均值相对当前成员均值的位移",
    "mean_intensity_shift": "联合强度均值减当前成员均值",
    "point_residual": "当前点至联合体素均值距离",
}
SEMANTIC = {
    0: "忽略",
    2: "异常",
    10: "汽车",
    11: "自行车",
    13: "公交车",
    15: "摩托车",
    16: "轨道车辆",
    18: "卡车",
    20: "其他车辆",
    30: "行人",
    31: "骑行者",
    32: "摩托车骑手",
    40: "道路",
    44: "停车区",
    48: "人行道",
    49: "其他地面",
    50: "建筑",
    51: "围栏",
    52: "其他结构",
    60: "车道标记",
    70: "植被",
    71: "树干",
    72: "地形",
    80: "杆体",
    81: "交通标志",
    99: "其他物体",
}


def group_name(group):
    if ":r" in group:
        label, radius = group.split(":r")
        index = int(radius)
        interval = (
            "大于50米"
            if index == 20
            else f"[{index * 2.5:g}, {(index + 1) * 2.5:g}{']' if index == 19 else ')'}米"
        )
        return f"{GROUP[label]}；{interval}"
    return GROUP.get(group, group)


def metric_name(metric):
    if metric.startswith("normal_from_scan_"):
        return f"新增正常混合含历史槽{metric[-1]}的正常点"
    return METRIC[metric]


def category_name(metric, code):
    code = int(code)
    if metric in ("visibility_pattern", "scan_hits"):
        return f"{code:05b}"
    if metric in ("raw_semantic", "normal_semantic", "neighbor_semantic"):
        return f"{code}：{SEMANTIC.get(code, '未命名语义编码')}"
    if metric == "current_state":
        return ("未见异常回波", "仅范围外异常", "范围内1至4点", "范围内至少5点")[code]
    if metric in ("anomaly_voxel_mix", "current_anomaly_point_mix"):
        return ("纯异常", "与正常混合", "与忽略混合", "同时与正常及忽略混合")[code]
    if metric == "all_voxel_label_presence":
        return (
            "+".join(
                name
                for bit, name in ((1, "正常"), (2, "异常"), (4, "忽略"))
                if code & bit
            )
            or "空体素"
        )
    if metric == "stage":
        return ("前缀", "可见段", "中断段", "尾段", "整序列未见")[code]
    if metric in ("history_visible_scans", "anomaly_history_hits"):
        return str(code)
    return "是" if code == 1 else "否" if code == 0 else str(code)


def missing_reason(data):
    if not data["denominator"]:
        return "没有适用观测；不填零"
    if data["n"] == data["denominator"]:
        return "无缺失"
    metric = data["metric"]
    if metric == "visible_instance_count":
        return "1点实例身份未确认；该帧实例数留空"
    if data["factor"] in ("B05", "B07") or metric in (
        "linearity",
        "planarity",
        "scattering",
    ):
        return "同实例不足10个不同坐标点；未将多个实例合并"
    if "same_instance" in metric:
        return "身份未确认，或同实例不足2个点无法定义最近邻距"
    if metric.startswith("ground_height"):
        return "地面点不足、平面拟合或空间支持未满足可靠性规则"
    if data["factor"] == "C04":
        return "0.5米内正常邻居少于3点"
    if data["factor"] == "E05":
        return "没有历史静态支持，或0.2米内无同语义历史匹配"
    if "compression" in metric or "fraction" in metric:
        return "相关异常点或体素数为零，比例无定义"
    if "median" in metric:
        return "当前帧在对应距离范围内没有异常回波"
    return "没有适用观测或估计不可靠；参见逐帧记录"


def get_stat(series, key, q=0.5, view="observation_equal"):
    return series[key][view]["quantiles"].get(str(q), {}).get("value")


def fraction(series, key, codes=(1,), view="observation_equal"):
    values = series[key][view]["categories"]
    return sum(values.get(str(code), 0) or 0 for code in codes)


def factor_results(result):
    t, s = result["totals"], result["series"]
    pattern = s["E02|visibility_pattern|complete_windows|all"]["observation_equal"][
        "category_counts"
    ]
    return {
        "A01": f"19条序列、{t['frames']:,}帧；首末帧与文件计数完整",
        "A02": f"原始槽{t['slots']:,}；回波{t['visible']:,}；零槽{t['zero_slots']:,}；忽略{t['ignore']:,}",
        "A03": f"当前未见异常{t['states'][0]:,}/8,659；逐序列与连续段已保存",
        "A04": f"仅范围外异常{t['states'][1]}帧；逐帧索引保留首末位置",
        "A05": f"范围内1至4点{t['states'][2]:,}帧；至少5点{t['states'][3]:,}帧",
        "A06": f"整窗无异常{t['whole_window_unseen']:,}/8,583＝{t['whole_window_unseen'] / 8583:.3%}",
        "B01": "全帧范围内点数分位数为0、0、0、3、55；完整窗口及所有零点状态保留",
        "B02": f"{t['anomaly']:,}异常回波；范围内{t['anomaly_in_range']:,}；逐点与每帧中位数分别统计",
        "B03": "16格原始复核一致；1,956帧、87,398异常点；零覆盖格保留",
        "B04": "全部异常点的原始坐标、方位和俯仰已统计；角度为弧度",
        "B05": f"{t['labelled_instances']}个发布正实例标识；{t['instance_frames']:,}条实例×帧；1,611条可估计可见跨度；1点身份未确认",
        "B06": "同实例最近邻距、0.25米邻居数及可见点集形态已统计；点数不足明确缺失",
        "B07": "1,611条实例×帧的可见角度跨度和最大内部方位间隙；不推断真实遮挡率",
        "C01": f"正常{t['normal']:,}、异常{t['anomaly']:,}、忽略{t['ignore']:,}点；精确取值计数合并分位数",
        "C02": "三类所有回波均精确落在float32(k/3500)网格；强度零值为0；极值计数与重复值单列",
        "C03": "按源扫描距离每2.5米分组至50米，超过50米另列；各组正常/异常/忽略分别统计",
        "C04": "0.5米正常邻域强度反差、标准差和方差；少于3个邻居时不估计；方差由完整精确标准差取值逐值平方后汇总",
        "D01": "正常全局语义、全部输入语义、每帧去重邻域正常语义分别统计",
        "D02": "全部异常点最近正常点距离和密度；2,202/4,729实例×帧通过局部地面代理规则",
        "E01": "8,583条完整点数向量及76条启动向量；历史数、历史比例、历史可见扫描数已保存",
        "E02": f"32种模式全部有覆盖；00000为{pattern['0']:,}窗，11111为{pattern['31']:,}窗，00001为{pattern['1']}窗",
        "E03": "18个前缀、304个可见段、285个中断段、12个尾段；所有首末截断标记已保存",
        "E04": f"五帧平移中位数{get_stat(s, 'E04|translation|complete_windows|all'):.3f}米；旋转以弧度报告，不推算速度",
        "E05": f"固定抽样{t['static_sampled']:,}当前静态点；{t['static_matched']:,}个在0.2米内同语义匹配；属于抽样代理",
        "F01": "完整输入0.05米网格；当前/联合异常体素及压缩比均已统计，零异常比值留空",
        "F02": "226,752个联合异常占据体素；纯异常96.404%；当前异常点加权另列",
        "F03": "历史新增正常混合1,633/75,412个当前异常体素；扫描来源分别保存，可同时命中多帧",
        "F04": "全部非空体素及当前异常体素的32类扫描命中、历史异常扫描数均保留",
        "F05": "全部当前成员均统计；空间分位数区间宽度至多0.1毫米，强度偏移区间宽度至多0.0001",
        "G01": "点数×距离×历史、混合×背景、混合×点数距离、运动×点数距离、阶段×点数及强度×距离均保存",
        "G02": "完整物理尺寸、真实材质、精确遮挡率、位姿真值误差、可靠时间戳未提供；不得补猜",
    }


FACTORS = [
    [
        "A01",
        "覆盖",
        "序列与扫描数量",
        "每条公开val完整读取；不按异常出现时刻裁剪",
        "原始帧/序列",
        "总帧数、首末帧号、文件完整性",
        "帧数不等于独立场景数",
    ],
    [
        "A02",
        "覆盖",
        "有效观测与零坐标槽",
        "原始槽数、xyz非全零回波数及忽略标签数分别统计",
        "原始帧/原始点",
        "逐帧分位数、各类点数、占比",
        "零槽不作为实际回波；忽略点仍保留于完整输入",
    ],
    [
        "A03",
        "状态",
        "当前帧无异常",
        "当前帧全部实际可见回波中sem=2数量为零，不限距离",
        "全部原始帧",
        "每序列计数、比例、连续长度",
        "这是无异常回波，不等于世界绝无异常物体",
    ],
    [
        "A04",
        "状态",
        "当前帧有异常但无范围内异常",
        "有实际异常回波，但2.5–50米内异常点为零",
        "全部原始帧",
        "帧数、占比、首末时刻",
        "不应记成已观测范围内漏检",
    ],
    [
        "A05",
        "状态",
        "当前帧极少点与合格状态",
        "范围内异常点数为1–4或≥5；两类分开",
        "全部帧；完整五帧区间另报",
        "帧数、占比、异常点数",
        "统计画像不能只保留官方合格帧",
    ],
    [
        "A06",
        "状态",
        "整窗无异常",
        "五帧所有实际异常回波数量之和为零",
        "t≥4完整窗口",
        "每序列计数与占比",
        "生成正常训练窗口配比的直接参考",
    ],
    [
        "B01",
        "几何",
        "异常点数分布",
        "当前帧范围内异常点数；全距离数量另列",
        "帧；实例统计另列",
        "0、1–4、5–19、20–99、100–499、≥500",
        "每帧异常总数不等于每个物体点数",
    ],
    [
        "B02",
        "几何",
        "异常距离分布",
        "逐点真实距离及每帧异常距离中位数",
        "点/帧/序列分别汇总",
        "范围内四距离组；范围外单列；分位数",
        "中位数相近不等于逐点距离分布相同",
    ],
    [
        "B03",
        "几何",
        "点数×距离联合覆盖",
        "沿用已有4×4网格，不合格状态另报",
        "t≥4官方合格帧",
        "每格帧数、异常点数、序列数及占比",
        "7格真实无数据不等于现实不可能",
    ],
    [
        "B04",
        "几何",
        "位置与方位",
        "异常点/可见实例在当前LiDAR坐标中的xyz与角度",
        "当前点/实例/序列",
        "高度、方位、俯仰的分位数及固定分箱",
        "不可将传感器高度直接当离地高度",
    ],
    [
        "B05",
        "几何",
        "可见实例数与尺寸",
        "以有效实例标签分组；统一用可见点估计尺寸",
        "实例×帧",
        "实例数；长宽高、比例、有效点数",
        "无有效实例ID标不可判定，不自动聚成一个物体",
    ],
    [
        "B06",
        "几何",
        "局部形态与采样疏密",
        "可见异常邻距、固定半径邻居数、局部形态",
        "点/实例×帧",
        "分位数、估计失败率、点数条件",
        "稀疏估计须记录样本数，不将缺测填零",
    ],
    [
        "B07",
        "几何",
        "扫描角度覆盖与缺口",
        "异常回波方位/俯仰跨度及角度间隙",
        "实例×帧/序列",
        "分位数、缺口分布",
        "未经光束校准仅报角度，不伪造beam/ring ID",
    ],
    [
        "C01",
        "强度",
        "逐点强度",
        "正常/异常/忽略分开；保留原值",
        "原始点；每帧只统计一次",
        "P5/P25/P50/P75/P95、直方图、极值",
        "子集不能推广为全部19条分布",
    ],
    [
        "C02",
        "强度",
        "离散网格、零值与边界",
        "实测float32(k/3500)关系、重复值和边界堆积",
        "原始点",
        "精确网格率、零值率、极值与计数",
        "不是官方硬件量化规格；不对输入吸附",
    ],
    [
        "C03",
        "强度",
        "强度×逐点距离",
        "按源帧自身距离分段比较强度",
        "原始点×距离组",
        "正常/异常条件分位数与覆盖",
        "均值离开离散网格不算格式错误",
    ],
    [
        "C04",
        "强度",
        "异常/邻近背景强度对比",
        "同一预定邻域中异常与正常强度差及局部方差",
        "异常点/实例×帧",
        "条件分位数、无邻居比例",
        "邻域及距离条件固定后计算",
    ],
    [
        "D01",
        "背景",
        "正常语义构成",
        "全部输入及异常邻域中的正常类别比例",
        "点/帧/序列",
        "全局与各序列频率；邻域另表",
        "标签用于统计，不进入推理特征",
    ],
    [
        "D02",
        "背景",
        "异常与地面/周边结构关系",
        "相对可靠地面高度、最近正常点距离、背景密度",
        "实例×帧",
        "分位数、估计失败率",
        "无可靠地面时不可将z冒充离地高度",
    ],
    [
        "E01",
        "时序",
        "五帧异常支持量",
        "源帧异常计数向量[a0,a1,a2,a3,a4]；历史与当前分开",
        "t≥4完整窗口",
        "计数、历史占比、按可见状态分组",
        "全距离/源帧各自评价距离两个口径分开",
    ],
    [
        "E02",
        "时序",
        "五位可见模式",
        "bi=1当第i帧有异常回波；包含00000",
        "t≥4完整窗口",
        "32种模式的计数与比例；每序列另报",
        "不是体素命中；类别可见不代表同一物体",
    ],
    [
        "E03",
        "时序",
        "接近/可见/消失阶段",
        "无回波前缀、可见段、中断段、尾段",
        "连续帧/完整序列",
        "段数与长度；首末可见；边界截断标志",
        "不能在异常事件处重新启动窗口；优先帧单位",
    ],
    [
        "E04",
        "时序",
        "自车运动与视角基线",
        "从已有位姿计算五帧相对平移/旋转",
        "t≥4完整窗口",
        "分位数、位移×点数/距离",
        "无可靠时间戳不报告实测速度",
    ],
    [
        "E05",
        "时序",
        "静态表面一致性",
        "固定匹配规则下背景表面重合残差",
        "窗口/可靠静态邻域",
        "残差分位数、匹配覆盖",
        "不是位姿真值误差，动态/遮挡可混杂",
    ],
    [
        "F01",
        "体素",
        "异常压缩与独立几何量",
        "0.05m网格异常占据数；当前与五帧分别统计",
        "体素/窗口",
        "点体素比、联合/当前异常体素数",
        "体素去重；当前无异常时比值不可定义",
    ],
    [
        "F02",
        "体素",
        "三态体素混合",
        "各体素统计正常/异常/忽略成员数及比例",
        "体素；当前异常点另加权",
        "纯异常、正常混合、忽略混合、双混合",
        "构建网格前不得按标签/距离裁点",
    ],
    [
        "F03",
        "体素",
        "历史新增正常混合",
        "当前体素无正常成员但历史带入正常成员",
        "当前异常点及其体素",
        "计数、比例、扫描来源",
        "描述成员，不等于证明历史造成漏检",
    ],
    [
        "F04",
        "体素",
        "扫描命中与异常支持",
        "体素五位命中模式；历史异常来自几帧",
        "非空体素；当前异常体素",
        "32种模式；命中次数1–5",
        "未命中≠可确认空闲；异常支持非精确对应",
    ],
    [
        "F05",
        "体素",
        "均值偏移与点内残差",
        "联合均值相对当前成员均值；点相对联合均值偏移",
        "体素/当前点",
        "位移与残差分位数",
        "同一物理格子比较，不重新选择网格原点",
    ],
    [
        "G01",
        "联合条件",
        "关键联合分布",
        "点数×距离×历史支持；混合×背景；强度×距离",
        "窗口/点/实例分别定义",
        "联合频数、支持数、缺覆盖格子",
        "先用少量可解释交叉，不拟合全高维直方图",
    ],
    [
        "G02",
        "不可观测",
        "完整物理形状/材质/准确遮挡率",
        "发布数据无真值时明确标记",
        "物体/场景",
        "只记录可用真值或带不确定性的代理量",
        "不要求为了填表虚构这些数值",
    ],
]
HISTORICAL_INTENSITY = [
    [
        "类别",
        "点数",
        "最小值",
        "P25",
        "P50",
        "P75",
        "P95",
        "P99",
        "最大值",
        "实测网格率",
        "体素均值P50",
        "来源",
    ],
    [
        "真实正常",
        31233470,
        0.000571,
        0.128571,
        0.249143,
        0.439714,
        0.768571,
        0.996,
        2.428571,
        1,
        0.255184,
        "https://github.com/JasonGao1010/Anomaly/blob/4a46a63201c319355899d168bdf185d27c5cdac1/src/intensity.py",
    ],
    [
        "真实异常",
        10546,
        0.002857,
        0.116571,
        0.284857,
        0.444286,
        0.736214,
        0.932286,
        1.085429,
        1,
        0.286357,
        "https://github.com/JasonGao1010/Anomaly/blob/4a46a63201c319355899d168bdf185d27c5cdac1/src/intensity.py",
    ],
]


class CsvTables:
    """One rectangular UTF-8 CSV per table; descriptions live in the index."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = []

    def table(self, name, title, note, headers, rows):
        path = self.directory / f"{name}.csv"
        temporary = path.with_suffix(".writing")
        count = 0
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(headers)
            for values in rows:
                if len(values) != len(headers):
                    raise ValueError(f"nonrectangular CSV row: {name}")
                writer.writerow(
                    [
                        ""
                        if value is None
                        else int(value)
                        if isinstance(value, (bool, np.bool_))
                        else value
                        for value in values
                    ]
                )
                count += 1
        temporary.replace(path)
        self.index.append([path.name, title, count, note])

    def close(self, title="19条真实验证序列全量观测画像", note=None):
        self.table(
            "index",
            title,
            note
            or "全部8,659帧；8,583完整窗口；76启动窗口。空字段表示缺测或不适用，0为实测零；五位模式按文本读取。统计单位、范围和权重以各表为准。",
            ["文件", "内容", "数据行数", "统计范围与解释"],
            self.index.copy(),
        )


def write_tables(output, result, directory):
    output = Path(output)
    series, totals = result["series"], result["totals"]
    writer = CsvTables(directory)
    notes = factor_results(result)
    factors = []
    for identifier, group, name, definition, unit, statistics, caution in FACTORS:
        status = (
            "不可直接观测，边界已记录"
            if identifier == "G02"
            else "已统计；抽样代理"
            if identifier == "E05"
            else "已统计；不足支持者留空"
            if identifier in ("B05", "B06", "B07", "C04", "D02")
            else "全量统计完成"
        )
        factors.append(
            [
                identifier,
                group,
                name,
                definition,
                unit,
                statistics,
                notes[identifier],
                status,
                caution,
                f"summary.json；{identifier}",
            ]
        )
    writer.table(
        "factors",
        "31项要素的定义、结果和完成状态",
        "原用户要素定义保留；统计结果来自全部公开真实观测。尺寸、地面和局部形态同时记录支持不足的缺失，物理属性不补猜。",
        [
            "编号",
            "要素组",
            "因素",
            "操作定义",
            "统计单位与范围",
            "应输出统计",
            "全量结果",
            "完成状态",
            "用途与限制",
            "来源",
        ],
        factors,
    )
    known, sequence_rows = [], []
    full_eligible = [0, 0, 0]
    for seq, data in result["sequences"].items():
        s, c = data["series"], data["coverage"]
        frames = read_rows(output / seq / "frames.jsonl")
        for frame in frames:
            if frame["frame"] >= 4 and frame["state"] == 3:
                full_eligible[0] += 1
                full_eligible[1] += frame["anomaly_in_range"]
                full_eligible[2] += frame["normal_in_range"]
        normal_stage = sum(r["normal_in_range"] for r in frames if r["state"] == 0)
        known.append(
            [
                int(seq),
                c["frames"],
                c["complete_windows"],
                c["states"][0],
                sum(c["states"][1:]),
                c["states"][3],
                sum(c["states"][1:3]),
                c["states"][0] / c["frames"],
                sum(c["states"][1:]) / c["frames"],
                c["states"][3] / c["frames"],
                normal_stage,
                f"{seq}/frames.jsonl",
            ]
        )
        sequence_rows.append(
            [
                int(seq),
                f"静态0至{c['last_frame']}；窗口4至{c['last_frame']}",
                c["complete_windows"],
                fraction(s, "A06|whole_window_unseen|complete_windows|all"),
                fraction(s, "E02|visibility_pattern|complete_windows|all"),
                get_stat(s, "B02|distance|all_frames|all"),
                get_stat(s, "B01|anomaly_in_range|all_frames|all"),
                get_stat(s, "B05|visible_instance_count|all_frames|all"),
                *[
                    get_stat(s, f"B05|{m}|all_frames|all")
                    for m in ("length", "width", "height")
                ],
                get_stat(s, "C01|intensity|all_frames|normal"),
                get_stat(s, "C01|intensity|all_frames|anomaly"),
                fraction(s, "C02|grid_match|all_frames|anomaly"),
                get_stat(s, "B06|same_instance_neighbor_distance|all_frames|all"),
                fraction(s, "D01|normal_semantic|all_frames|all", (40,)),
                get_stat(s, "E01|history_visible_scans|complete_windows|all"),
                get_stat(s, "E04|translation|complete_windows|all"),
                get_stat(s, "E04|rotation|complete_windows|all"),
                fraction(s, "F02|anomaly_voxel_mix|complete_windows|all", (1, 3)),
                fraction(s, "F02|anomaly_voxel_mix|complete_windows|all", (2, 3)),
                fraction(s, "F03|new_normal|complete_windows|all"),
                get_stat(s, "F01|current_anomaly_compression|complete_windows|all"),
                get_stat(s, "F01|joint_anomaly_voxel_increment|complete_windows|all"),
                get_stat(s, "F05|mean_displacement|complete_windows|current_anomaly"),
                "已完成；尺寸/地面/邻域可靠性见全量统计；无可靠时间戳"
                + ("；1点实例身份未确认" if c["unknown_instance_points"] else ""),
            ]
        )
    assert sum(r[10] for r in known) == totals["normal_stage_evaluation_points"]
    writer.table(
        "prior_sequences",
        "与既有报告复核一致的19条序列计数",
        "全部原始帧；有异常但不合格包括范围外和范围内1至4点。比例直接写入数值，未保留电子表格公式。",
        [
            "序列",
            "原始帧数",
            "完整五帧窗数",
            "当前未见异常帧",
            "当前可见异常帧",
            "官方合格帧",
            "可见但不合格帧",
            "未见异常比例",
            "可见异常比例",
            "官方合格比例",
            "无异常帧范围内正常点数",
            "来源",
        ],
        known,
    )
    writer.table(
        "sequences",
        "19条序列核心因素",
        "静态因素使用全部原始帧；窗口因素使用当前帧4至末。强度、距离按点，尺寸按有效实例×帧，混合按指定体素，其余中位数按帧；比例均为0至1的小数。",
        [
            "序列",
            "统计区间",
            "完整五帧窗数",
            "整窗未见异常比例",
            "00001比例",
            "异常逐点距离P50（米）",
            "范围内异常点数P50（含零帧）",
            "可见实例数P50",
            "可见水平长边P50（米）",
            "可见水平短边P50（米）",
            "可见竖向跨度P50（米）",
            "正常强度P50",
            "异常强度P50",
            "异常精确网格比例",
            "同实例最近邻距P50（米）",
            "道路占正常点比例",
            "历史可见扫描数P50",
            "五帧平移P50（米）",
            "五帧旋转P50（弧度）",
            "联合异常体素含正常比例",
            "联合异常体素含忽略比例",
            "当前异常体素历史新增正常比例",
            "当前异常压缩比P50",
            "历史增加异常体素数P50",
            "当前异常体素均值位移P50（米；区间中点）",
            "状态与缺失说明",
        ],
        sequence_rows,
    )
    global_rows = []

    def global_row(name, numerator, denominator, scope, definition):
        global_rows.append(
            [
                name,
                numerator,
                denominator,
                numerator / denominator if denominator else None,
                scope,
                definition,
                "summary.json",
            ]
        )

    for metric, key in (
        ("原始帧数", "frames"),
        ("完整五帧窗口", "complete_windows"),
        ("启动窗口", "startup_windows"),
        ("原始槽数", "slots"),
        ("实际回波数", "visible"),
        ("正常回波数", "normal"),
        ("忽略回波数", "ignore"),
        ("异常回波数", "anomaly"),
        ("范围内全部异常点", "anomaly_in_range"),
    ):
        global_row(metric, totals[key], None, "全部真实验证", "直接计数；分母不适用")
    for name, count in zip(
        ("当前未见异常", "仅范围外异常", "范围内1至4点", "范围内至少5点"),
        totals["states"],
    ):
        global_row(name, count, 8659, "全部原始帧", "四种互斥状态")
    global_row(
        "零坐标槽",
        totals["zero_slots"],
        totals["slots"],
        "全部原始帧",
        "零坐标槽不作为回波",
    )
    global_row(
        "忽略占回波比例",
        totals["ignore"],
        totals["visible"],
        "全部原始帧",
        "忽略点仍参加联合体素构建",
    )
    global_row(
        "整窗未见异常",
        totals["whole_window_unseen"],
        8583,
        "完整五帧窗口",
        "全距离五帧异常回波总和为零",
    )
    full_unseen = series["A03|current_unseen|complete_frames|all"]["observation_equal"][
        "category_counts"
    ]["1"]
    global_row(
        "完整区间当前未见异常",
        full_unseen,
        8583,
        "完整五帧当前帧",
        "不能替代整窗未见异常",
    )
    global_row(
        "当前可见异常帧",
        sum(totals["states"][1:]),
        8659,
        "全部原始帧",
        "包含范围外及少点帧",
    )
    global_row(
        "实际回波占槽比例",
        totals["visible"],
        totals["slots"],
        "全部原始帧",
        "每个原始槽只计一次",
    )
    global_row(
        "完整区间官方合格帧",
        full_eligible[0],
        8583,
        "完整五帧当前帧",
        "范围内至少5个异常点",
    )
    global_row(
        "完整区间合格异常点",
        full_eligible[1],
        None,
        "完整五帧合格帧",
        "保留原诊断子集分母",
    )
    global_row(
        "完整区间合格正常点",
        full_eligible[2],
        None,
        "完整五帧合格帧",
        "保留原诊断子集分母",
    )
    global_row(
        "完整区间合格异常点占比",
        full_eligible[1],
        sum(full_eligible[1:]),
        "完整五帧合格帧",
        "原诊断参考占比，未改变评价",
    )
    global_row(
        "官方合格正常点",
        totals["official_normal_points"],
        None,
        "全部官方合格帧",
        "2.5至50米含端点",
    )
    global_row(
        "官方合格异常点",
        totals["official_anomaly_points"],
        None,
        "全部官方合格帧",
        "帧内范围异常点至少5个",
    )
    global_row(
        "官方合格异常点占比",
        totals["official_anomaly_points"],
        totals["official_anomaly_points"] + totals["official_normal_points"],
        "全部官方合格帧",
        "仅用于保留原有评价点构成",
    )
    global_row(
        "无异常阶段范围内正常点",
        totals["normal_stage_evaluation_points"],
        None,
        "当前无异常回波帧",
        "原有正常阶段分母",
    )
    global_row(
        "可估计可见几何的实例×帧",
        totals["shape_status"]["reliable_visible_extent"],
        totals["instance_frames"],
        "正实例标识",
        "至少10个不同坐标点；不是完整尺寸",
    )
    global_row(
        "局部地面代理通过",
        totals["ground_status"]["local_plane_proxy"],
        totals["instance_frames"],
        "正实例标识",
        "固定平面及支持规则",
    )
    writer.table(
        "totals",
        "全量计数及明确分母",
        "比例为分子除以分母；计数无适用分母时留空。各范围互不替代。",
        ["指标", "数量或分子", "分母", "比例", "范围", "定义与限制", "来源"],
        global_rows,
    )
    joint = result["joint"]["official_count_distance"]
    joint_rows = []
    for i, count in enumerate(("5至19", "20至99", "100至499", "至少500")):
        for j, distance in enumerate(("[2.5,10)", "[10,20)", "[20,35)", "[35,50]")):
            r = joint[f"{i}|{j}"]
            joint_rows.append(
                [
                    count,
                    distance,
                    r["sequence_count"],
                    r["frames"],
                    r["anomaly_in_range_points"],
                    r["frames"] / 1956,
                    r["anomaly_in_range_points"] / 87398,
                    "有覆盖" if r["frames"] else "无覆盖",
                    f"summary.json:official_count_distance:{i}|{j}",
                ]
            )
    writer.table(
        "count_distance",
        "完整五帧合格子集的16格联合分布",
        "1,956帧、87,398异常点；空格保留。边缘合计由对应行求和，CSV不混入第二组表头。",
        [
            "当前范围内异常点数",
            "当前范围内异常距离中位数（米）",
            "序列数",
            "帧数",
            "异常点数",
            "占合格帧比例",
            "占合格异常点比例",
            "覆盖状态",
            "来源",
        ],
        joint_rows,
    )
    writer.table(
        "prior_intensity",
        "历史303个条件窗口的强度统计",
        "来自14条序列303窗，保留原报告舍入数值和分位数定义；不能替代全量19条。原始当前正常点31,233,470，异常点10,546。",
        HISTORICAL_INTENSITY[0],
        HISTORICAL_INTENSITY[1:],
    )
    write_distributions(writer, result)
    append_records(writer, output, result)
    writer.close()


def write_distributions(writer, result):
    synthetic = result["definitions"].get("population") in ("train", "validation")
    view_labels = dict(VIEW)
    if synthetic:
        view_labels["sequence_equal"] = "世界等权（同一背景）"
    populations = [
        ("全部", result["series"]),
        *[(seq, data["series"]) for seq, data in result["sequences"].items()],
    ]

    def distributions(kind):
        for seq, population in populations:
            for key, views in population.items():
                for view, d in views.items():
                    unit = (
                        "窗口"
                        if d["unit"] == "frame" and "windows" in d["scope"]
                        else UNIT[d["unit"]]
                    )
                    base = [
                        seq,
                        d["factor"],
                        metric_name(d["metric"]),
                        SCOPE[d["scope"]],
                        group_name(d["group"]),
                        unit,
                        view_labels[view],
                        d["n"],
                        d["denominator"],
                        d["frames"],
                        d["valid_frames"],
                        d["sequence_count"],
                    ]
                    source = f"summary.json:{seq}:{key}:{view}"
                    if kind == "continuous" and not d["categorical"]:
                        yield base + [
                            d["value_unit"],
                            d["missing_fraction"],
                            d["empty_frame_fraction"],
                            d["minimum"],
                            d["maximum"],
                            d["mean"],
                            *[
                                d["quantiles"].get(str(q), {}).get("value")
                                for q in QUANTILES
                            ],
                            d["resolution"],
                            missing_reason(d),
                            source,
                            *[
                                d["quantiles"].get(str(q), {}).get(bound)
                                for q in QUANTILES
                                for bound in ("lower", "upper")
                            ],
                        ]
                    elif kind == "categories" and d["categorical"]:
                        for code, count in d["category_counts"].items():
                            yield base + [
                                category_name(d["metric"], code),
                                code,
                                count,
                                d["categories"][code],
                                missing_reason(d),
                                source,
                            ]
                    elif kind == "bins" and not d["categorical"]:
                        for i, part in enumerate(d["bin_fraction"]):
                            yield base + [
                                d["bins"][i],
                                d["bins"][i + 1],
                                d["bin_counts"][i],
                                part,
                                source,
                            ]

    base_headers = [
        "世界" if synthetic else "序列",
        "因素",
        "统计量",
        "范围",
        "分组",
        "统计单位",
        "权重",
        "有效数量",
        "适用候选数量",
        "总帧或阶段组数",
        "有有效观测的帧或组数",
        "有覆盖世界数" if synthetic else "有覆盖序列数",
    ]
    writer.table(
        "continuous",
        "连续因素：分位数、覆盖与可靠性",
        "分位数由各权重的完整经验分布计算。区间宽度为0时为精确值；非零时点值展示区间中点，上下界另列。缺失率以未加权候选数为分母，三种视图不混用。",
        base_headers
        + [
            "数值单位",
            "缺失或不可靠比例",
            "无有效值帧或组比例",
            "最小值",
            "最大值",
            "均值",
            "P5",
            "P25",
            "P50",
            "P75",
            "P95",
            "分位数区间最大宽度",
            "缺失说明",
            "来源",
        ]
        + [f"P{int(q * 100)}{bound}" for q in QUANTILES for bound in ("下界", "上界")],
        distributions("continuous"),
    )
    writer.table(
        "categorical",
        "分类因素：完整类别及零覆盖",
        "类别原始计数在三种视图相同；比例按对应权重计算。五位模式由最早扫描到当前扫描排列。全部体素成员编码为正常1、异常2、忽略4的位和。",
        base_headers
        + ["类别", "类别编码", "原始数量", "对应权重比例", "缺失说明", "来源"],
        distributions("categories"),
    )
    writer.table(
        "bins",
        "连续因素固定分箱",
        "区间左闭右开；末区间包含右端。普通距离直方图[50,无穷)不能解读为严格大于50米；官方范围标记与2.5米强度距离组均将50米纳入范围内。计数为未加权数，比例按对应权重计算。",
        base_headers + ["左边界", "右边界", "原始数量", "对应权重比例", "来源"],
        distributions("bins"),
    )


def append_records(writer, output, result):
    populations = [
        ("全部", result["series"]),
        *[(seq, d["series"]) for seq, d in result["sequences"].items()],
    ]
    intensity_rows = []
    mode_rows = []
    for seq, series in populations:
        for label in ("normal", "anomaly", "ignore"):
            for view, data in series[f"C01|intensity|all_frames|{label}"].items():
                intensity_rows.append(
                    [
                        seq,
                        GROUP[label],
                        VIEW[view],
                        data["n"],
                        data["valid_frames"],
                        data["sequence_count"],
                        data["minimum"],
                        *[data["quantiles"][str(q)]["value"] for q in QUANTILES],
                        data["maximum"],
                        data["unique_count"],
                        data["repeated_value_fraction"],
                        data["minimum_count"],
                        data["maximum_count"],
                        fraction(series, f"C02|zero|all_frames|{label}", view=view),
                        fraction(
                            series, f"C02|grid_match|all_frames|{label}", view=view
                        ),
                        json.dumps(data["most_frequent_values"], ensure_ascii=False),
                    ]
                )
        for key in (
            "E02|visibility_pattern|complete_windows|all",
            "F04|scan_hits|complete_windows|all",
            "F04|scan_hits|complete_windows|current_anomaly",
        ):
            for view, data in series[key].items():
                for code in range(32):
                    mode_rows.append(
                        [
                            seq,
                            metric_name(data["metric"]),
                            group_name(data["group"]),
                            VIEW[view],
                            f"{code:05b}",
                            code.bit_count(),
                            data["category_counts"].get(str(code), 0),
                            data["n"],
                            data["categories"].get(str(code), 0),
                            data["valid_frames"],
                            data["sequence_count"],
                        ]
                    )
                for hits in range(6):
                    members = [code for code in range(32) if code.bit_count() == hits]
                    mode_rows.append(
                        [
                            seq,
                            metric_name(data["metric"]) + "：按位数汇总",
                            group_name(data["group"]),
                            VIEW[view],
                            f"合计{hits}个扫描",
                            hits,
                            sum(
                                data["category_counts"].get(str(code), 0)
                                for code in members
                            ),
                            data["n"],
                            sum(
                                data["categories"].get(str(code), 0) or 0
                                for code in members
                            ),
                            data["valid_frames"],
                            data["sequence_count"],
                        ]
                    )
    writer.table(
        "intensity",
        "全量原始强度及离散取值",
        "每个源帧只计一次，包含全部距离。重复值比例指落在出现次数至少为2的精确取值上的点比例；极值仅为样本极值。网格关系是实测格式特征，不能称为硬件规格。",
        [
            "序列",
            "类别",
            "权重",
            "点数",
            "有效帧数",
            "序列数",
            "最小值",
            "P5",
            "P25",
            "P50",
            "P75",
            "P95",
            "最大值",
            "不同取值数",
            "重复取值点比例",
            "最小值点数",
            "最大值点数",
            "零值比例",
            "精确网格比例",
            "最高频10个取值及计数",
        ],
        intensity_rows,
    )
    writer.table(
        "patterns",
        "五位可见模式和体素扫描命中",
        "最左为历史最早扫描，最右为当前扫描。类别可见模式按全距离异常回波存在性；体素命中按全部输入点。32类均保留；命中位数可用于汇总1至5次观测的频数。",
        [
            "序列",
            "统计量",
            "分组",
            "权重",
            "五位模式",
            "可见或命中位数",
            "原始数量",
            "分母",
            "比例",
            "有效窗口数",
            "序列数",
        ],
        mode_rows,
    )
    frames, windows, instances, stages = [], [], [], []
    for seq in result["sequences"]:
        for name, target in (
            ("frames", frames),
            ("windows", windows),
            ("instances", instances),
            ("stages", stages),
        ):
            target.extend(read_rows(output / seq / f"{name}.jsonl"))
    support = []
    for seq in ("全部", *result["sequences"]):
        selected = [
            w
            for w in windows
            if w["scope"] == "complete_windows"
            and (seq == "全部" or w["sequence"] == int(seq))
        ]
        for code in range(32):
            rows = [w for w in selected if w["visibility_pattern"] == f"{code:05b}"]
            vector = np.array([w["anomaly_vector"] for w in rows], np.int64).reshape(
                -1, 5
            )
            fractions = [
                w["history_anomaly_fraction"]
                for w in rows
                if w["history_anomaly_fraction"] is not None
            ]
            vector_quantiles = (
                np.quantile(vector, QUANTILES, axis=0, method="inverted_cdf")
                .T.ravel()
                .tolist()
                if rows
                else [None] * 25
            )
            history_quantiles = (
                np.quantile(fractions, QUANTILES, method="inverted_cdf").tolist()
                if fractions
                else [None] * 5
            )
            support.append(
                [
                    seq,
                    f"{code:05b}",
                    len(rows),
                    len({w["sequence"] for w in rows}),
                    int(vector[:, -1].sum()),
                    int(vector[:, :-1].sum()),
                    len(fractions),
                    1 - len(fractions) / len(rows) if rows else None,
                    *vector_quantiles,
                    *history_quantiles,
                ]
            )
    writer.table(
        "pattern_support",
        "按固定可见模式分组的五帧异常支持量",
        "完整五帧窗口；源帧全距离异常点数逐槽统计，包含实测零。各序列及全局按窗口等权；模式为空时覆盖数为0。00000没有定义历史异常点占比，不将空分母置零。",
        [
            "序列",
            "五位模式",
            "窗口数",
            "覆盖序列数",
            "当前异常点合计",
            "历史异常点合计",
            "历史占比有效窗数",
            "历史占比无定义比例",
            *[f"槽{i}异常点数P{int(q * 100)}" for i in range(5) for q in QUANTILES],
            *[f"历史异常占比P{int(q * 100)}" for q in QUANTILES],
        ],
        support,
    )
    basic = (
        "sequence",
        "frame",
        "slots",
        "visible",
        "zero_slots",
        "normal",
        "ignore",
        "anomaly",
        "anomaly_in_range",
        "normal_in_range",
        "state",
        "anomaly_distance_median",
        "anomaly_in_range_distance_median",
        "instance_count",
        "unknown_instance_points",
        "neighbor_road_fraction",
        "stage",
        "first_in_visible_run",
    )

    def frame_values():
        for row in frames:
            yield [
                *[row[k] for k in basic],
                *[
                    (row[f"{label}_intensity"] or {}).get("quantiles", {}).get(str(q))
                    for label in ("normal", "anomaly", "ignore")
                    for q in QUANTILES
                ],
            ]

    writer.table(
        "frames",
        "全部8,659帧：状态、点数及强度",
        "原始帧序号从0开始。状态0为无异常，1为仅范围外，2为范围内1至4点，3为范围内至少5点。强度分位数仅取本帧对应类别原始回波；无回波时留空。阶段是类别回波观测状态，不是物体物理出现或消失。",
        [
            "序列",
            "帧号",
            "原始槽",
            "实际回波",
            "零坐标槽",
            "正常点",
            "忽略点",
            "全距离异常点",
            "范围内异常点",
            "范围内正常点",
            "四态编码",
            "全距离异常距离P50（米）",
            "范围内异常距离P50（米）",
            "可见实例数",
            "身份未确认异常点",
            "邻近正常点中道路比例",
            "阶段编码",
            "是否可见段首帧",
            *[
                f"{GROUP[label]}强度P{int(q * 100)}"
                for label in ("normal", "anomaly", "ignore")
                for q in QUANTILES
            ],
        ],
        frame_values(),
    )
    window_keys = (
        "sequence",
        "frame",
        "scope",
        "visibility_pattern",
        "all_unseen",
        "history_anomaly",
        "history_visible_scans",
        "history_anomaly_fraction",
        "voxels",
        "current_voxels",
        "current_anomaly_voxels",
        "joint_anomaly_voxels",
        "joint_anomaly_voxel_increment",
        "current_anomaly_compression",
        "joint_anomaly_compression",
        "current_anomaly_normal_mix_voxels",
        "current_anomaly_ignore_mix_voxels",
        "history_new_normal_voxels",
        "history_new_normal_points",
        "translation_m",
        "rotation_rad",
        "normal_mix_fraction",
        "ignore_mix_fraction",
        "new_normal_fraction",
        "anomaly_mean_displacement_median",
        "static_sampled",
        "static_matched",
        "static_match_fraction",
    )

    def window_values(rows):
        for row in rows:
            yield [
                *[row[k] for k in window_keys],
                *row["anomaly_vector"],
                *row["anomaly_in_range_vector"],
            ]

    window_headers = [
        "序列",
        "当前帧",
        "范围编码",
        "可见模式",
        "整窗未见",
        "历史异常点数",
        "历史可见扫描数",
        "历史异常占比",
        "联合体素数",
        "当前占据体素数",
        "当前异常体素数",
        "联合异常体素数",
        "历史增加异常体素数",
        "当前异常压缩比",
        "联合异常压缩比",
        "当前异常体素含正常数",
        "当前异常体素含忽略数",
        "历史新增正常体素数",
        "历史新增正常影响当前异常点数",
        "最早至当前平移（米）",
        "最早至当前旋转（弧度）",
        "当前异常体素含正常比例",
        "当前异常体素含忽略比例",
        "历史新增正常比例",
        "当前异常体素均值位移P50（米）",
        "抽样静态点数",
        "静态匹配点数",
        "静态匹配比例",
        *[f"槽{i}全距离异常点数" for i in range(5)],
        *[f"槽{i}源坐标范围内异常点数" for i in range(5)],
    ]
    writer.table(
        "windows",
        "8,583个完整五帧窗口",
        "体素由所有实际输入点共同建立，再按语义事后统计。正常、异常、忽略成员均保留。历史支持是类别支持，不能解读为同一物体点的跨帧匹配。",
        window_headers,
        window_values([r for r in windows if r["scope"] == "complete_windows"]),
    )
    writer.table(
        "startup",
        "76个启动窗口单列",
        "缺少的历史槽为空，不是零点扫描；所有已到达点按原始启动路径构造体素。本页不并入完整五帧窗口分布。",
        window_headers,
        window_values([r for r in windows if r["scope"] == "startup_windows"]),
    )
    instance_keys = (
        "sequence",
        "frame",
        "instance",
        "points",
        "unique_points",
        "shape_status",
        "length",
        "width",
        "height",
        "aspect",
        "linearity",
        "planarity",
        "scattering",
        "azimuth_span",
        "elevation_span",
        "azimuth_gap",
        "ground_status",
        "ground_height_p05",
        "ground_height_median",
        "ground_support",
        "ground_rmse",
    )
    writer.table(
        "instances",
        "4,729条有效正实例标识的实例×帧记录",
        "尺寸是本帧LiDAR坐标轴下可见点的水平长短边和竖向跨度；至少10个不同坐标点。非完整物理尺寸或朝向框。正实例标识按序列隔离；另有1个零标识异常点保留于逐帧和逐点记录，未强行分组。",
        [
            "序列",
            "帧号",
            "发布实例标识",
            "点数",
            "不同坐标点数",
            "形状可靠性编码",
            "水平长边（米）",
            "水平短边（米）",
            "竖向跨度（米）",
            "长短边比",
            "线性度",
            "平面度",
            "散乱度",
            "方位跨度（弧度）",
            "俯仰跨度（弧度）",
            "最大内部方位间隙（弧度）",
            "地面代理可靠性编码",
            "可见离地高度P5（米）",
            "可见离地高度P50（米）",
            "地面支持点数",
            "地面拟合均方根残差（米）",
        ],
        ([r.get(k) for k in instance_keys] for r in instances),
    )
    stage_keys = (
        "sequence",
        "kind",
        "start",
        "end",
        "length",
        "left_censored",
        "right_censored",
    )
    writer.table(
        "stages",
        "619段连续回波观测阶段",
        "同一物体的稀疏回波可导致多次类别可见中断。左截断表示段开始于序列首帧，右截断表示段结束于序列末帧；均不能推断真实物理起止。长度单位为帧。",
        ["序列", "阶段", "首帧", "末帧", "长度（帧）", "左边界截断", "右边界截断"],
        (
            [r["sequence"], GROUP[r["kind"]], *[r[k] for k in stage_keys[2:]]]
            for r in stages
        ),
    )
    table_names = {
        "official_count_distance": "合格点数×距离",
        "count_distance_history": "点数×距离×历史可见扫描数",
        "mix_count_distance": "混合比例×点数×距离",
        "mix_background": "混合比例×邻域背景",
        "motion_count_distance": "点数×距离×平移",
        "stage_count": "阶段×全距离点数",
    }
    joint_rows = []
    for name, cells in result["joint"].items():
        for key, row in cells.items():
            joint_rows.append(
                [
                    table_names[name],
                    key,
                    row["frames"],
                    row["anomaly_points"],
                    row["anomaly_in_range_points"],
                    row["sequence_count"],
                    row["instance_count"],
                    ",".join(map(str, row["sequences"])),
                    ",".join(f"{s}:{i}" for s, i in row["instances"]),
                    "有覆盖" if row["frames"] else "无覆盖",
                ]
            )
    writer.table(
        "joints",
        "固定联合条件与共同覆盖量",
        "一般点数编码0/1/2/3/4/5对应0、1至4、5至19、20至99、100至499、至少500；合格表点数编码0至3从5至19起。距离0至3对应[2.5,10)、[10,20)、[20,35)、[35,50]；unseen和outside_only单列。混合0为0，1为(0,0.25]，2为(0.25,0.75]，3为大于0.75。平移边界0.1/0.5/1/2/5米。",
        [
            "联合条件",
            "分组编码",
            "帧数",
            "全距离异常点数",
            "范围内异常点数",
            "序列数",
            "正实例标识数",
            "序列标识",
            "序列与实例标识",
            "覆盖状态",
        ],
        joint_rows,
    )
    reliability = [
        [
            "可见形状",
            "不同坐标点至少10个",
            1611,
            4729,
            3118 / 4729,
            "不足支持者不估计可见跨度；已估计者仍不是完整物体尺寸",
        ],
        [
            "局部地面高度代理",
            "半径2米、地面至少20点；残差≤0.05米、斜率≤20度、平面空间支持合格",
            2202,
            4729,
            2527 / 4729,
            "仅对拟合通过的实例×帧给出可见点离地分位数",
        ],
        [
            "实例身份",
            "正发布标识按序列分组；零标识身份未确认",
            90738,
            90739,
            1 / 90739,
            "val/149中的1点保留原始位置和类别，不参与实例分组",
        ],
        [
            "静态表面代理",
            "每窗最多2048个当前静态点、固定种子、0.2米同语义历史最近邻",
            result["totals"]["static_matched"],
            result["totals"]["static_sampled"],
            1 - result["totals"]["static_matched"] / result["totals"]["static_sampled"],
            "抽样代理；匹配残差混合了采样与配准影响，不能称位姿真值误差",
        ],
        [
            "精确持续时间与速度",
            "未发现可靠时间戳",
            None,
            19,
            None,
            "只报告帧数、位移和旋转角",
        ],
        [
            "完整物理尺寸、真实材质、精确遮挡率",
            "发布点云不提供对应真值",
            None,
            None,
            None,
            "可见几何与强度不能补充为真实物理属性",
        ],
    ]
    writer.table(
        "reliability",
        "可靠性不足与不可观测边界",
        "缺测表示定义不适用、观测支持不足或真值缺少，不是零。阈值在原始统计执行前确定，未根据模型预测或评价成绩选择。",
        [
            "因素",
            "固定规则或缺失原因",
            "有效数",
            "候选数",
            "不可靠或缺失比例",
            "解释边界",
        ],
        reliability,
    )


def plot_profile(output, result):
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    for family, filename in (
        ("SimSun", "simsun.ttc"),
        ("Times New Roman", "times.ttf"),
    ):
        font_manager.fontManager.addfont(Path("/mnt/c/Windows/Fonts") / filename)
        font_manager.findfont(family, fallback_to_default=False)
    output = Path(output)
    sequences = list(result["sequences"])
    series = result["series"]
    colors = ("#CAD4DA", "#B39BC8", "#E2B35D", "#438DA4")

    def quantiles(key, view="observation_equal", scale=1):
        return (
            np.array(
                [series[key][view]["quantiles"][str(q)]["value"] for q in QUANTILES]
            )
            * scale
        )

    def finish(fig, pdf, title, note):
        fig.suptitle(title, fontsize=17)
        fig.text(0.035, 0.018, note, fontsize=10, va="bottom")
        fig.tight_layout(rect=(0.015, 0.075, 0.995, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

    with (
        plt.rc_context(
            {
                "font.family": ["Times New Roman", "SimSun"],
                "pdf.fonttype": 42,
                "font.size": 10,
            }
        ),
        PdfPages(output / "results.pdf") as pdf,
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.05, 1]}
        )
        states = np.array(
            [result["sequences"][seq]["coverage"]["states"] for seq in sequences]
        )
        proportions = states / states.sum(axis=1, keepdims=True) * 100
        left = np.zeros(19)
        for i, label in enumerate(
            ("未见异常回波", "仅范围外异常", "范围内1至4点", "范围内至少5点")
        ):
            axes[0].barh(
                sequences, proportions[:, i], left=left, label=label, color=colors[i]
            )
            left += proportions[:, i]
        axes[0].invert_yaxis()
        axes[0].set(
            xlabel="占本序列全部帧的比例（%）",
            ylabel="真实验证序列",
            title="全部 8,659 帧：四种互斥状态",
        )
        axes[0].legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=2, fontsize=9
        )
        modes = series["E02|visibility_pattern|complete_windows|all"][
            "observation_equal"
        ]
        codes = [f"{i:05b}" for i in range(32)]
        counts = [modes["category_counts"][str(i)] for i in range(32)]
        axes[1].barh(
            codes,
            counts,
            color=[
                colors[0] if i == 0 else colors[3] if i == 31 else colors[2]
                for i in range(32)
            ],
        )
        axes[1].invert_yaxis()
        axes[1].set_xscale("log")
        axes[1].set(xlabel="窗口数（对数轴）", title="完整五帧 8,583 窗：32 种可见模式")
        axes[1].tick_params(axis="y", labelsize=8)
        for i, count in enumerate(counts):
            axes[1].text(count * 1.04, i, str(count), va="center", fontsize=7)
        axes[1].set_xlim(0.8, 10000)
        finish(
            fig,
            pdf,
            "真实验证观测画像：覆盖与连续观测",
            "五位模式从最早扫描到当前扫描；00000 表示五帧均无异常回波。\n当前帧无异常不等于整窗无异常，也不等于物理道路上不存在异常物体。",
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        cells = result["joint"]["official_count_distance"]
        matrix = np.array(
            [[cells[f"{i}|{j}"]["frames"] for j in range(4)] for i in range(4)]
        )
        axes[0, 0].imshow(matrix, cmap="Blues")
        for i in range(4):
            for j in range(4):
                axes[0, 0].text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 300 else "black",
                )
        axes[0, 0].set(
            xticks=range(4),
            xticklabels=["[2.5,10)", "[10,20)", "[20,35)", "[35,50]"],
            yticks=range(4),
            yticklabels=["5–19", "20–99", "100–499", "≥500"],
            xlabel="本帧范围内异常点距离中位数（米）",
            ylabel="本帧范围内异常点数",
            title="合格完整五帧子集：1,956 帧，空格保留",
        )
        for i, view in enumerate(VIEW):
            for j, (label, color) in enumerate(
                (("normal", colors[0]), ("anomaly", colors[3]))
            ):
                q = quantiles(f"C01|intensity|all_frames|{label}", view)
                y = 2 * i + j
                axes[0, 1].plot(q[[0, 4]], [y, y], color=color, lw=1.5)
                axes[0, 1].plot(q[[1, 3]], [y, y], color=color, lw=7)
                axes[0, 1].plot(q[2], y, "o", color="black", ms=4)
        axes[0, 1].set(
            yticks=range(6),
            yticklabels=[
                f"{v}·{GROUP[label]}"
                for v in ("点等权", "帧等权", "序列等权")
                for label in ("normal", "anomaly")
            ],
            xlabel="原始强度",
            title="全距离强度：中位数、四分位与 5%–95% 区间",
        )
        axes[0, 1].invert_yaxis()
        for label, color in (("normal", "#8898A2"), ("anomaly", colors[3])):
            x, lo, median, hi = [], [], [], []
            for r in range(20):
                data = series[f"C03|intensity|all_frames|{label}:r{r}"][
                    "observation_equal"
                ]
                if not data["n"]:
                    continue
                x.append(r * 2.5 + 1.25)
                lo.append(data["quantiles"]["0.25"]["value"])
                median.append(data["quantiles"]["0.5"]["value"])
                hi.append(data["quantiles"]["0.75"]["value"])
            axes[1, 0].plot(x, median, ".-", label=GROUP[label], color=color)
            axes[1, 0].fill_between(x, lo, hi, color=color, alpha=0.17)
        axes[1, 0].set(
            xlabel="源扫描逐点距离（米；每 2.5 米分组）",
            ylabel="原始强度",
            title="逐点距离条件下的强度：点加权中位数和四分位",
        )
        axes[1, 0].legend()
        for i, metric in enumerate(("length", "width", "height")):
            q = quantiles(f"B05|{metric}|all_frames|all")
            axes[1, 1].plot(q[[0, 4]], [i, i], color=colors[3], lw=1.5)
            axes[1, 1].plot(q[[1, 3]], [i, i], color=colors[3], lw=7)
            axes[1, 1].plot(q[2], i, "o", color="black", ms=4)
        axes[1, 1].set(
            yticks=range(3),
            yticklabels=["可见水平长边", "可见水平短边", "可见竖向跨度"],
            xlabel="可见跨度（米）",
            title="形状支持充分的 1,611 条实例×帧",
        )
        finish(
            fig,
            pdf,
            "真实验证观测画像：少点、强度与可见几何",
            "异常点总数 90,739；4,729 条实例×帧中有 3,118 条不足 10 个不同坐标点，不估计形状。\n可见跨度依赖本帧坐标轴和回波覆盖，不是物体完整尺寸；强度分布差异本身不构成预测失败的因果证据。",
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = np.arange(19)
        normal_mix = [
            fraction(
                result["sequences"][seq]["series"],
                "F02|anomaly_voxel_mix|complete_windows|all",
                (1, 3),
            )
            * 100
            for seq in sequences
        ]
        ignore_mix = [
            fraction(
                result["sequences"][seq]["series"],
                "F02|anomaly_voxel_mix|complete_windows|all",
                (2, 3),
            )
            * 100
            for seq in sequences
        ]
        axes[0, 0].bar(x - 0.18, normal_mix, 0.36, label="含正常成员", color=colors[3])
        axes[0, 0].bar(x + 0.18, ignore_mix, 0.36, label="含忽略成员", color=colors[1])
        axes[0, 0].set(
            xticks=x,
            xticklabels=sequences,
            ylabel="占联合异常占据体素（%）",
            title="五帧共同体素的混合；两类可能同时存在",
        )
        axes[0, 0].tick_params(axis="x", rotation=90)
        axes[0, 0].legend()
        new_mix = [
            fraction(
                result["sequences"][seq]["series"],
                "F03|new_normal|complete_windows|all",
            )
            * 100
            for seq in sequences
        ]
        axes[0, 1].bar(sequences, new_mix, color=colors[2])
        axes[0, 1].set(
            ylabel="占当前异常占据体素（%）", title="当前无正常成员、历史引入正常成员"
        )
        axes[0, 1].tick_params(axis="x", rotation=90)
        for i, (label, key) in enumerate(
            (
                (
                    "均值位置偏移",
                    "F05|mean_displacement|complete_windows|current_anomaly",
                ),
                ("当前异常点残差", "F05|point_residual|complete_windows|anomaly"),
                (
                    "抽样静态匹配残差",
                    "E05|sampled_static_distance|complete_windows|all",
                ),
            )
        ):
            q = quantiles(key, scale=1000)
            axes[1, 0].plot(q[[0, 4]], [i, i], color=colors[3], lw=1.5)
            axes[1, 0].plot(q[[1, 3]], [i, i], color=colors[3], lw=7)
            axes[1, 0].plot(q[2], i, "o", color="black", ms=4)
        axes[1, 0].set(
            yticks=range(3),
            yticklabels=[
                "当前异常体素均值偏移",
                "当前异常点至联合均值",
                "抽样静态点匹配残差",
            ],
            xlabel="距离（毫米）",
            title="不同统计对象分别报告；分位数区间宽度≤0.1毫米",
        )
        kinds = ("prefix", "visible", "gap", "tail")
        for i, kind in enumerate(kinds):
            q = quantiles(f"E03|stage_length|sequence_runs|{kind}")
            axes[1, 1].plot(q[[0, 4]], [i, i], color=colors[3], lw=1.5)
            axes[1, 1].plot(q[[1, 3]], [i, i], color=colors[3], lw=7)
            axes[1, 1].plot(q[2], i, "o", color="black", ms=4)
        axes[1, 1].set(
            yticks=range(4),
            yticklabels=[
                "前缀（18段）",
                "可见（304段）",
                "中断（285段）",
                "尾段（12段）",
            ],
            xlabel="观测阶段长度（帧；对数轴）",
            title="观测阶段长度；边界截断保留",
        )
        axes[1, 1].set_xscale("log")
        finish(
            fig,
            pdf,
            "真实验证观测画像：五帧表示与观测阶段",
            "体素统计先使用完整输入建立 5 厘米网格，再按标签区分成员；序列开头 76 窗另列。\n静态表面项每窗最多抽样 2,048 点，匹配范围 0.2 米；不是真值配准误差。所有阶段均按回波观测定义。",
        )


def write_pool_tables(output, result, directory):
    """Export the same measurements with worlds and shared roads explicitly distinguished."""
    writer = CsvTables(directory)
    totals, spec = result["totals"], result["definitions"]
    rows = []
    for identifier, group, name, definition, unit, statistics, caution in FACTORS:
        keys = [k for k in result["series"] if k.startswith(identifier + "|")]
        if identifier == "A01":
            definition = "逐冻结世界读取全部帧；前四帧为上下文；只统计段内合法五帧窗口"
        rows.append(
            [
                identifier,
                group,
                name,
                definition,
                unit,
                statistics,
                "不可作真实横向比较"
                if identifier == "G02"
                else "已统计；固定抽样代理"
                if identifier == "E05"
                else "已统计；支持不足留空",
                len(keys),
                "生成器完整物理参数未混入可见观测分布"
                if identifier == "G02"
                else caution,
            ]
        )
    writer.table(
        "factors",
        "同一31项因素及可观测边界",
        "原定义见profiles/real/method.md。原始单位、帧、世界等权分开；世界不是独立道路。",
        [
            "因素",
            "组",
            "名称",
            "定义",
            "单位",
            "统计内容",
            "完成状态",
            "统计组合数",
            "边界",
        ],
        rows,
    )
    coverages = [d["coverage"] for d in result["sequences"].values()]
    headers = list(coverages[0])
    writer.table(
        "worlds",
        "各世界覆盖及边界",
        "sequence为世界内唯一目录标识；first_frame/last_frame为源扫描号；一个世界只属于一个合成版本。",
        headers,
        [
            [
                json.dumps(r.get(k), ensure_ascii=False)
                if isinstance(r.get(k), (list, dict))
                else r.get(k)
                for k in headers
            ]
            for r in coverages
        ],
    )
    version_rows = []
    for version in sorted({r["version"] for r in coverages}):
        members = [r for r in coverages if r["version"] == version]
        version_rows.append(
            [
                version,
                spec["source_sequence_id"],
                len(members),
                *[
                    sum(r[k] for r in members)
                    for k in (
                        "frames",
                        "complete_windows",
                        "whole_window_unseen",
                        "anomaly",
                    )
                ],
            ]
        )
    writer.table(
        "versions",
        "观测次数与道路来源",
        "同一池所有版本共用一条原始道路，版本数不是道路数。",
        [
            "合成版本",
            "原始道路",
            "世界数",
            "帧观测次数",
            "合法窗口",
            "整窗无异常",
            "全帧异常点",
        ],
        version_rows,
    )
    writer.table(
        "totals",
        "全帧与合法当前帧分别汇总",
        "全帧包括每个世界前四帧；legal字段只统计合法当前帧；startup_windows=0不是未保存原始上下文。",
        ["量", "值"],
        [
            [k, json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v]
            for k, v in totals.items()
        ],
    )
    write_distributions(writer, result)
    for name, title in [
        ("frames", "逐帧观测"),
        ("windows", "逐窗口观测"),
        ("instances", "逐实例观测"),
        ("stages", "连续观测阶段"),
    ]:

        def records():
            for key in spec["sequences"]:
                yield from read_rows(Path(output) / key / f"{name}.jsonl")

        headers = sorted({k for r in records() for k in r})
        writer.table(
            name,
            title,
            "sequence标识世界；frame为世界内索引，source_frame为原始扫描号。连续阶段首尾保留截断；空值不补零。",
            headers,
            (
                [
                    json.dumps(r.get(k), ensure_ascii=False)
                    if isinstance(r.get(k), (list, dict))
                    else r.get(k)
                    for k in headers
                ]
                for r in records()
            ),
        )
    writer.close(
        "冻结合成池同口径画像",
        f"背景train/{spec['source_sequence_id']}，{spec['synthetic_versions']}个版本，"
        f"{totals['worlds']}个世界，{totals['frames']}帧，{totals['complete_windows']}合法窗口。"
        "所有统计只读冻结观测；五位模式按文本读取，空字段与实测零分开。",
    )


def compare_profiles(synthetic, real, output, directory):
    """Compare completed profiles without opening real scans or using model scores."""
    populations = [synthetic["train"], synthetic["validation"], real]
    labels = ["206训练", "201合成验证", "真实开发"]
    writer = CsvTables(directory)

    def data(result, key, view="observation_equal"):
        return result["series"][key][view]

    def qtext(result, key, view="observation_equal"):
        d = data(result, key, view)
        values = [d["quantiles"].get(str(q), {}).get("value") for q in QUANTILES]
        text = "/".join("空" if v is None else f"{v:.6g}" for v in values)
        return f"P5/P25/P50/P75/P95={text}；有效{d['n']}/{d['denominator']}，{d['valid_frames']}帧，{d['sequence_count']}世界或序列"

    def cattext(result, key):
        d = data(result, key)
        return "; ".join(
            f"{category_name(d['metric'], c)}={n}/{d['n']}={n / d['n']:.6%}"
            for c, n in d["category_counts"].items()
        )

    keys = {
        "A02": ["A02|visible|all_frames|all", "A02|ignore_fraction|all_frames|all"],
        "B01": ["B01|anomaly_in_range|all_frames|all"],
        "B02": [
            "B02|distance|all_frames|all",
            "B02|anomaly_distance_median|all_frames|all",
        ],
        "B04": ["B04|x|all_frames|all", "B04|azimuth|all_frames|all"],
        "B05": [
            "B05|length|all_frames|all",
            "B05|width|all_frames|all",
            "B05|height|all_frames|all",
        ],
        "B06": ["B06|same_instance_neighbor_distance|all_frames|all"],
        "B07": ["B07|azimuth_span|all_frames|all"],
        "C01": ["C01|intensity|all_frames|normal", "C01|intensity|all_frames|anomaly"],
        "C04": [
            "C04|intensity_contrast|all_frames|all",
            "C04|background_intensity_variance|all_frames|all",
        ],
        "D02": [
            "D02|nearest_normal_distance|all_frames|all",
            "D02|ground_height_median|all_frames|all",
        ],
        "E01": [
            "E01|history_visible_scans|complete_windows|all",
            "E01|history_anomaly_fraction|complete_windows|all",
        ],
        "E03": [
            "E03|stage_length|sequence_runs|visible",
            "E03|stage_length|sequence_runs|gap",
        ],
        "E04": [
            "E04|translation|complete_windows|all",
            "E04|rotation|complete_windows|all",
        ],
        "E05": ["E05|sampled_static_distance|complete_windows|all"],
        "F01": ["F01|current_anomaly_compression|complete_windows|all"],
        "F05": ["F05|mean_displacement|complete_windows|current_anomaly"],
    }
    # The main table gives representative existing measures; full distributions retain every measure.
    common_keys = set.intersection(*(set(r["series"]) for r in populations))
    for factor, selected in keys.items():
        if any(k not in common_keys for k in selected):
            raise ValueError(
                f"comparison metric name not in authoritative profiles: {factor}: {selected}"
            )

    def overview(r, factor):
        t, s = r["totals"], r["series"]
        if factor in keys:
            return "；".join(
                f"{metric_name(k.split('|')[1])}({group_name(k.split('|')[-1])}): {qtext(r, k)}"
                for k in keys[factor]
            )
        if factor == "A01":
            return f"原始背景序列{t.get('background_source_sequences', t['sequences'])}；版本{t.get('synthetic_versions', '不适用')}；世界或序列{t['sequences']}；帧{t['frames']}；完整窗{t['complete_windows']}"
        if factor in ("A03", "A04", "A05"):
            indexes = {"A03": [0], "A04": [1], "A05": [2, 3]}[factor]
            return "；".join(
                f"四态{i}={t['states'][i]}/{t['frames']}={t['states'][i] / t['frames']:.6%}"
                for i in indexes
            )
        if factor == "A06":
            n, den = t["whole_window_unseen"], t["complete_windows"]
            members = [
                e["coverage"]
                for e in r["sequences"].values()
                if e["coverage"]["whole_window_unseen"]
            ]
            return f"{n}/{den}={n / den:.6%}；覆盖{len(members)}个世界或序列"
        if factor == "B03":
            j = r["joint"]["official_count_distance"]
            return f"合格帧{sum(x['frames'] for x in j.values())}；异常点{sum(x['anomaly_in_range_points'] for x in j.values())}；覆盖格{sum(x['frames'] > 0 for x in j.values())}/16"
        if factor == "C02":
            return "；".join(
                f"{group_name(g)}零强度{fraction(s, f'C02|zero|all_frames|{g}'):.6%}；k/3500精确匹配{fraction(s, f'C02|grid_match|all_frames|{g}'):.6%}；重复取值点{data(r, f'C01|intensity|all_frames|{g}')['repeated_value_fraction']:.6%}"
                for g in ("normal", "anomaly", "ignore")
            )
        if factor == "C03":
            return f"{sum(k.startswith('C03|') for k in s)}个原有逐点距离×类别条件；全量连续统计保留各组原始、帧、世界或序列权重"
        if factor == "D01":
            d = data(r, "D01|normal_semantic|all_frames|all")
            top = sorted(d["category_counts"].items(), key=lambda kv: -kv[1])[:5]
            return "；".join(f"{SEMANTIC.get(int(k), k)}={v}/{d['n']}" for k, v in top)
        if factor == "E02":
            d = data(r, "E02|visibility_pattern|complete_windows|all")
            c = d["category_counts"]
            return f"覆盖{sum(v > 0 for v in c.values())}/32；00000={c['0']}，11111={c['31']}，部分={d['n'] - c['0'] - c['31']}，00001={c['1']}；分母{d['n']}"
        if factor == "F02":
            return cattext(r, "F02|anomaly_voxel_mix|complete_windows|all")
        if factor == "F03":
            return cattext(r, "F03|new_normal|complete_windows|all")
        if factor == "F04":
            d = data(r, "F04|scan_hits|complete_windows|current_anomaly")
            return f"当前异常体素观测{d['n']}；扫描命中类别{sum(v > 0 for v in d['category_counts'].values())}；完整32格见分类表"
        if factor == "G01":
            j = r["joint"]["count_distance_history"]
            return f"点数×距离×历史覆盖{sum(v['frames'] > 0 for v in j.values())}/{len(j)}格；窗口{sum(v['frames'] for v in j.values())}"
        if factor == "G02":
            return "完整物理尺寸、真实材质、精确遮挡率、位姿真值误差及可靠时间未作横向估计；合成真值参数不混入"
        raise ValueError(f"unhandled original factor {factor}")

    difference = {
        "A01": ("固定背景限制", "增加世界数不能增加原始道路数"),
        "A06": ("已有覆盖但频率不匹配", "采样或训练权重；保留异常条件覆盖"),
        "B03": ("共同覆盖与空格并存，逐格判定", "采样或世界生成；见联合条件对照"),
        "G01": (
            "频率不匹配与完全缺覆盖分开",
            "采样或世界生成；先看真实非空且训练为零的格",
        ),
        "G02": ("不可观测或不可直接横向比较", "不得以不可观测属性指导确定参数"),
    }
    main = []
    for factor, group, name, definition, unit, statistics, caution in FACTORS:
        dtype, action = difference.get(
            factor,
            (
                "观测分布或可靠性差异；不作因果归因",
                "强度生成；当前不优先"
                if factor.startswith("C")
                else "生成覆盖或表示；未进行模型干预"
                if factor.startswith("F")
                else "几何、放置或观测过程；先核对可靠性"
                if factor.startswith(("B", "D"))
                else "采样或观测过程；片段边界截断单列",
            ),
        )
        coverage = (
            "全帧"
            + "/".join(str(r["totals"]["frames"]) for r in populations)
            + "；完整窗"
            + "/".join(str(r["totals"]["complete_windows"]) for r in populations)
            + "；合成等权单位为世界，真实为序列，非独立道路配对"
        )
        if factor in ("B05", "B06", "B07", "D02"):
            coverage = "；".join(
                f"{label}:实例帧{r['totals']['instance_frames']}，形状{r['totals']['shape_status']}，地面{r['totals']['ground_status']}，未知身份点{r['totals']['unknown_instance_points']}"
                for label, r in zip(labels, populations)
            )
        main.append(
            [
                factor,
                group,
                name,
                *[overview(r, factor) for r in populations],
                coverage,
                dtype,
                action,
                caution,
            ]
        )
    writer.table(
        "comparison",
        "训练—合成验证—真实开发的31项因素",
        "连续量主表按原始单位等权；每帧中位数单独列；其余权重及缺失、区间在明细中。",
        [
            "因素",
            "组",
            "因素或联合条件",
            *labels,
            "覆盖与可靠性",
            "差异类型",
            "可调整环节",
            "解释边界",
        ],
        main,
    )
    joints = []
    absent = {}
    for table in real["joint"]:
        all_cells = sorted(set.union(*(set(r["joint"][table]) for r in populations)))
        absent[table] = dict(
            cells=0, real_frames=0, real_anomaly_points=0, real_sequences=set()
        )
        for cell in all_cells:
            entries = [
                r["joint"][table].get(
                    cell,
                    dict(
                        frames=0,
                        anomaly_points=0,
                        anomaly_in_range_points=0,
                        sequence_count=0,
                        sequences=[],
                        instance_count=0,
                    ),
                )
                for r in populations
            ]
            tr, va, re = entries
            kind = (
                "训练完全缺覆盖"
                if re["frames"] and not tr["frames"]
                else "合成验证完全缺覆盖"
                if re["frames"] and not va["frames"]
                else "共同覆盖；比较频率"
                if all(e["frames"] for e in entries)
                else "三侧均无观测"
                if not any(e["frames"] for e in entries)
                else "该发布真实集合无覆盖"
            )
            if re["frames"] and not tr["frames"]:
                a = absent[table]
                a["cells"] += 1
                a["real_frames"] += re["frames"]
                a["real_anomaly_points"] += re["anomaly_points"]
                a["real_sequences"].update(re["sequences"])
            values = []
            for r, e in zip(populations, entries):
                entities = e["sequences"]
                versions = len(
                    {
                        r["sequences"][str(x)]["coverage"].get("version", str(x))
                        for x in entities
                    }
                )
                roads = 1 if entities and r is not real else len(entities)
                den = sum(x["frames"] for x in r["joint"][table].values())
                values.extend(
                    [
                        e["sequence_count"],
                        versions,
                        roads,
                        e["frames"],
                        den,
                        e["frames"] / den if den else None,
                        e["anomaly_points"],
                        e["anomaly_in_range_points"],
                        e["instance_count"],
                    ]
                )
            pieces = cell.split("|")
            counts = ("0", "1至4", "5至19", "20至99", "100至499", "至少500")
            distances = {
                str(i): v
                for i, v in enumerate(("[2.5,10)", "[10,20)", "[20,35)", "[35,50]"))
            }
            distances.update(unseen="未见异常", outside_only="仅范围外异常")
            mix = ("0", "(0,0.25]", "(0.25,0.75]", "(0.75,1]")
            if table == "official_count_distance":
                description = f"范围内{counts[int(pieces[0]) + 2]}点；距离中位数{distances[pieces[1]]}米"
            elif table in ("count_distance_history", "motion_count_distance"):
                description = f"范围内{counts[int(pieces[0])]}点；距离中位数{distances[pieces[1]]}"
                description += (
                    f"；历史可见{pieces[2]}次"
                    if table == "count_distance_history"
                    else f"；五帧平移{('[0,0.1)', '[0.1,0.5)', '[0.5,1)', '[1,2)', '[2,5)', '[5,无穷)')[int(pieces[2])]}米"
                )
            elif table == "mix_count_distance":
                description = f"混合比例{mix[int(pieces[0])]}；范围内{counts[int(pieces[1])]}点；距离中位数{distances[pieces[2]]}"
            elif table == "mix_background":
                description = f"混合比例{mix[int(pieces[0])]}；{dict(no_neighbor='无邻近正常点', road_majority='道路类占邻域至少一半', other_majority='道路类占邻域不足一半')[pieces[1]]}"
            else:
                description = f"{GROUP[pieces[0]]}；全距离{counts[int(pieces[1])]}点"
            joints.append([table, cell, description, *values, kind])
    fields = [
        "世界或序列数",
        "合成版本或真实序列数",
        "原始背景序列数（非独立道路）",
        "帧数",
        "该联合表帧分母",
        "帧比例",
        "全距离异常点",
        "范围内异常点",
        "已观测实例标识数",
    ]
    writer.table(
        "joint_comparison",
        "原有联合格的覆盖与频率",
        "分组编码与真实画像一致；零格保留；各联合表是不同切面，覆盖量不能跨表相加。",
        [
            "联合条件",
            "格编码",
            "条件说明",
            *[f"{label}_{k}" for label in labels for k in fields],
            "差异类型",
        ],
        joints,
    )
    modes = []
    key = "E02|visibility_pattern|complete_windows|all"
    for code in range(32):
        values = []
        for r in populations:
            d = data(r, key)
            members = [
                e
                for e in r["sequences"].values()
                if e["series"][key]["observation_equal"]["category_counts"].get(
                    str(code), 0
                )
            ]
            values.extend(
                [
                    d["category_counts"][str(code)],
                    d["n"],
                    d["categories"][str(code)],
                    len(members),
                ]
            )
        modes.append([f"{code:05b}", code.bit_count(), *values])
    writer.table(
        "visibility",
        "全部32种五帧异常可见模式",
        "五位文本从历史到当前；00000整窗无异常；11111持续可见；0末位与历史有1表示当前消失但历史仍可见。",
        [
            "五位文本模式",
            "可见扫描数",
            *[
                f"{label}_{k}"
                for label in labels
                for k in ["窗口数", "总窗口", "比例", "覆盖世界或序列数"]
            ],
        ],
        modes,
    )
    # Compare the full weighted CDF summaries, not an average of per-entity quantiles.
    weights = []
    for label, r in zip(labels, populations):
        for key in [
            "B02|distance|all_frames|all",
            "B02|anomaly_distance_median|all_frames|all",
            "C01|intensity|all_frames|anomaly",
        ]:
            for view, d in r["series"][key].items():
                weights.append(
                    [
                        label,
                        key,
                        "世界等权"
                        if view == "sequence_equal" and r is not real
                        else VIEW[view],
                        d["n"],
                        d["denominator"],
                        d["valid_frames"],
                        d["sequence_count"],
                        *[
                            d["quantiles"].get(str(q), {}).get("value")
                            for q in QUANTILES
                        ],
                    ]
                )
    writer.table(
        "weights",
        "逐点分布与每帧代表值分开",
        "每帧中位数的分布不是帧等权逐点分布；世界等权不代表道路等权或相互独立。",
        [
            "侧",
            "量",
            "权重",
            "有效数",
            "候选数",
            "有效帧",
            "世界或序列数",
            "P5",
            "P25",
            "P50",
            "P75",
            "P95",
        ],
        weights,
    )
    for a in absent.values():
        a["real_sequences"] = sorted(a["real_sequences"])
    comparison = dict(
        missing_training_coverage=absent,
        selected_next_candidate=(
            "evaluate_observation_matched_data"
            if populations[0]["definitions"].get("pool_format")
            == "ajae-observation-match-pool"
            else "whole_window_normal_sampling"
        ),
        recommendation_only=True,
        model_forward_calls=0,
        parameter_updates=0,
        real_raw_scans_read=0,
        real_official_ap_unchanged=0.03475451,
    )
    path = Path(output) / "comparison.json"
    if path.exists():
        if json.loads(path.read_text()) != comparison:
            raise ValueError("saved comparison differs")
    else:
        _atomic_json(path, comparison)
    writer.close(
        "三侧同口径分布对照",
        "真实侧直接复用runs/profile_v1；两侧合成观测补全同一31项因素。train/和validation/为全量明细，真实明细仍在profiles/real/。",
    )
