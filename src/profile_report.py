"""Aggregate single-scan geometry distributions and write reproducible CSV tables."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .data import _atomic_json
from .evaluate import diagnostic_bin
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


def aggregate_profile(output):
    output = Path(output)
    spec = json.loads((output / "spec.json").read_text())
    merged = {}
    result = dict(sequences={}, series={})
    frames = []
    instances = []
    for seq in spec["sequences"]:
        directory = output / str(seq)
        coverage = json.loads((directory / "summary.json").read_text())
        if coverage.get("format") != "stu-frame-profile":
            raise ValueError(
                "profile data must be regenerated with the current single-scan source"
            )
        result["sequences"][str(seq)] = dict(coverage=coverage, series={})
        for name, target in (
            ("frames", frames),
            ("instances", instances),
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
    result.update(definitions=spec, frames=frames, instances=instances)
    result["totals"] = {
        key: sum(r["coverage"][key] for r in result["sequences"].values())
        for key in (
            "frames",
            "slots",
            "visible",
            "zero_slots",
            "ignore",
            "normal",
            "anomaly",
            "anomaly_in_range",
            "unknown_instance_points",
            "official_frames",
            "official_anomaly_points",
            "official_normal_points",
            "instance_frame_count",
        )
    }
    result["totals"]["sequences"] = len(result["sequences"])
    _atomic_json(output / "summary.json", finite_json(result))
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


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(finite_json(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def write_tables(output, result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for frame in result["frames"]:
        row = {
            key: value for key, value in frame.items() if not key.endswith("_intensity")
        }
        for group in ("normal", "anomaly", "ignore"):
            distribution = frame[f"{group}_intensity"]
            for q in QUANTILES:
                row[f"{group}_intensity_p{int(q * 100):02d}"] = (
                    distribution["quantiles"][str(q)] if distribution else None
                )
        frame_rows.append(row)
    tables = {
        "frames": frame_rows,
        "instances": result["instances"],
        "sequences": [values["coverage"] for values in result["sequences"].values()],
        "totals": [result["totals"]],
    }
    continuous, categorical, bins = [], [], []
    populations = [("all", result["series"])] + [
        (seq, values["series"]) for seq, values in result["sequences"].items()
    ]
    for sequence, series in populations:
        for key, views in sorted(series.items()):
            for weighting, data in views.items():
                row = dict(
                    sequence=sequence,
                    factor=data["factor"],
                    metric=data["metric"],
                    scope=data["scope"],
                    group=data["group"],
                    unit=data["unit"],
                    weighting=weighting,
                    n=data["n"],
                    denominator=data["denominator"],
                    frames=data["frames"],
                    valid_frames=data["valid_frames"],
                    sequence_count=data["sequence_count"],
                    missing_fraction=data["missing_fraction"],
                    empty_frame_fraction=data["empty_frame_fraction"],
                    minimum=data["minimum"],
                    maximum=data["maximum"],
                    mean=data["mean"],
                )
                if data["categorical"]:
                    for category, fraction in data["categories"].items():
                        categorical.append(
                            dict(
                                row,
                                category=category,
                                fraction=fraction,
                                count=data["category_counts"][category],
                            )
                        )
                else:
                    for q in QUANTILES:
                        estimate = data["quantiles"].get(str(q), {})
                        for field in ("value", "lower", "upper"):
                            row[f"p{int(q * 100):02d}_{field}"] = estimate.get(field)
                    continuous.append(row)
                for index, fraction in enumerate(data["bin_fraction"]):
                    bins.append(
                        dict(
                            sequence=sequence,
                            factor=data["factor"],
                            metric=data["metric"],
                            scope=data["scope"],
                            group=data["group"],
                            unit=data["unit"],
                            weighting=weighting,
                            n=data["n"],
                            bin=index,
                            lower=data["bins"][index],
                            upper=data["bins"][index + 1],
                            fraction=fraction,
                        )
                    )
    tables.update(continuous=continuous, categorical=categorical, bins=bins)
    matched = {f"{i}_{j}": [] for i in range(4) for j in range(4)}
    for row in result["frames"]:
        group = diagnostic_bin(
            row["anomaly_in_range"], row["anomaly_in_range_distance_median"]
        )
        if group is not None:
            matched[group].append(row)
    tables["count_distance"] = [
        dict(
            group=group,
            count_bin=int(group[0]),
            range_bin=int(group[2]),
            sequences=len({r["sequence"] for r in rows}),
            frames=len(rows),
            anomaly_points=sum(r["anomaly_in_range"] for r in rows),
        )
        for group, rows in matched.items()
    ]
    reliability = Counter(row["ground_status"] for row in result["instances"])
    tables["reliability"] = [
        dict(ground_status=status, instance_frames=count)
        for status, count in sorted(reliability.items())
    ]
    for name, rows in tables.items():
        write_csv(directory / f"{name}.csv", rows)
    write_csv(
        directory / "index.csv",
        [
            dict(
                file=f"{name}.csv",
                rows=len(rows),
                source=str(Path(output) / "summary.json"),
            )
            for name, rows in tables.items()
        ],
    )
    totals = result["totals"]
    (directory / "method.md").write_text(
        "# 单帧真实几何统计\n\n"
        "研究对象为 STU 公开验证集的当前扫描。每个原始扫描只计一次，几何统计不读取模型或预测。\n\n"
        f"本次覆盖 {totals['sequences']} 条序列、{totals['frames']} 帧、"
        f"{totals['visible']} 个实际回波，其中异常回波 {totals['anomaly']} 个。"
        f"官方范围内异常不少于 5 点的帧共 {totals['official_frames']} 个，"
        f"对应 {totals['official_anomaly_points']} 个异常点和 {totals['official_normal_points']} 个正常点。\n\n"
        "逐帧画像保留忽略标签、范围外回波和不满足官方帧门槛的扫描；"
        "这些描述性统计不能直接作为官方指标的分母。"
        "官方评价只使用当前传感器坐标中距离为 2.5 至 50 米、语义非零的点，语义 2 为异常。\n\n"
        "点数与距离联合表包含所有符合官方门槛的原始帧，包括序列开头的帧。"
        "距离分箱为 [2.5,10)、[10,20)、[20,35)、[35,50] 米；"
        "异常点数分箱为 [5,20)、[20,100)、[100,500)、至少 500。\n\n"
        "最近正常点距离和半径 0.5 米的正常邻域使用当前扫描的真实语义标签。"
        "同实例最近邻及半径 0.25 米的邻居数使用当前帧的实例标识。"
        "这些依赖真值的量只用于诊断，不作为模型输入。\n\n"
        "实例可见尺寸和协方差形态至少要求 10 个不同坐标点。"
        "实例标识为零的异常点保留在点级统计中，但不假定其真实实例归属。"
        "尺寸描述已观测点的跨度，不等于物体完整尺寸。\n\n"
        "地面代理使用语义 40、44、48、49、60 的当前扫描回波。"
        "实例中心水平 2 米内至少 20 个支持点；稳健拟合后要求均方根残差不超过 0.05 米、"
        "斜率不超过 20 度、最小水平协方差特征值至少 0.01，且中心位于支持凸包内。"
        "不可靠高度保留缺失和具体原因，不能填零。\n\n"
        "连续分布分别保留观测等权、有效帧等权和有效序列等权结果。"
        "分位数取加权经验分布的逆函数，不平均各序列分位数。"
        "强度保留原始值；局部强度方差由完整标准差分布逐值平方后计算。"
        "CSV 为 UTF-8，空单元格表示缺失或不适用；CSV 本身不保存字体。\n\n"
        "再生成命令："
        "`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        ".venv/bin/python -m src.profile --data-root /home/jasongao/Data/STU --workers 12`。"
        "进程数应依据运行时资源重新确定。\n",
        encoding="utf-8",
    )
