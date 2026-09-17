"""Inventory normal instance identities and their raw observations in train/206."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .scene import STUSequence


# Raw STU labels, before the training map merges moving and stationary classes.
LABEL_SOURCE = (
    "https://github.com/kumuji/stu_dataset/blob/main/"
    "Mask4Former3D/conf/semantic-kitti.yaml"
)
LABELS = {
    0: "未标注", 1: "离群标签", 2: "异常", 10: "汽车", 11: "自行车",
    13: "公共汽车", 15: "摩托车", 16: "轨道车辆", 18: "卡车",
    20: "其他车辆", 30: "行人", 31: "骑自行车者", 32: "骑摩托车者",
    40: "道路", 44: "停车区域", 48: "人行道", 49: "其他地面",
    50: "建筑", 51: "围栏", 52: "其他结构", 60: "车道标线",
    70: "植被", 71: "树干", 72: "地形", 80: "杆状物",
    81: "交通标志", 99: "其他物体", 252: "运动汽车",
    253: "运动骑自行车者", 254: "运动行人", 255: "运动骑摩托车者",
    256: "运动轨道车辆", 257: "运动公共汽车", 258: "运动卡车",
    259: "运动其他车辆",
}
OBSERVATION_FIELDS = [
    "structure_id", "semantic", "instance_id", "frame", "returns_all",
    "returns_2p5_50m", "range_min_m", "range_median_m", "range_max_m",
    "range_2p5_50m_median_m", "centroid_world_x_m", "centroid_world_y_m",
    "centroid_world_z_m", "azimuth_lidar_deg", "elevation_lidar_deg",
    "azimuth_world_deg", "elevation_world_deg", "previous_observed_frame",
    "gap_frames", "centroid_displacement_m", "nn_world_median_m",
    "nn_world_p95_m",
]


def structure_id(semantic, instance):
    # STU reuses a numeric instance ID across semantic classes in the same frame.
    return f"206:{semantic}:{instance}"


def bearing(vector):
    """Sensor-to-observed-centroid bearing; this is not surface incidence."""
    x, y, z = vector
    if np.linalg.norm(vector) == 0:
        return [None, None]
    return np.rad2deg([np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))]).tolist()


def frame_statistics(frame):
    slots = frame.observation_slots
    xyz = frame.xyzi[slots, :3]
    semantic = frame.labels.semantic[slots]
    instance = frame.labels.instance[slots]
    # Keep the existing float32 range convention, without cropping the census.
    radius = np.linalg.norm(xyz, axis=1)
    in_range = (radius >= 2.5) & (radius <= 50.0)
    classes = {}
    for raw in np.unique(semantic):
        mask = semantic == raw
        classes[int(raw)] = {
            "returns_all": int(mask.sum()),
            "returns_2p5_50m": int((mask & in_range).sum()),
            "returns_with_instance": int((mask & (instance > 0)).sum()),
        }
    objects = {}
    packed = frame.labels.packed[slots]
    for value in np.unique(packed[(instance > 0) & (semantic > 2)]):
        raw, identity = int(value) & 65535, int(value) >> 16
        mask = packed == value
        points = xyz[mask].astype(np.float64)
        world = points @ frame.lidar_pose[:3, :3].T + frame.lidar_pose[:3, 3]
        ranges, valid = radius[mask], in_range[mask]
        center = world.mean(axis=0)
        azimuth, elevation = bearing(points.mean(axis=0))
        world_azimuth, world_elevation = bearing(center - frame.lidar_pose[:3, 3])
        row = dict(
            structure_id=structure_id(raw, identity), semantic=raw,
            instance_id=identity, frame=frame.frame_id, returns_all=len(points),
            returns_2p5_50m=int(valid.sum()), range_min_m=float(ranges.min()),
            range_median_m=float(np.median(ranges)), range_max_m=float(ranges.max()),
            range_2p5_50m_median_m=float(np.median(ranges[valid])) if valid.any() else None,
            centroid_world_x_m=float(center[0]), centroid_world_y_m=float(center[1]),
            centroid_world_z_m=float(center[2]), azimuth_lidar_deg=azimuth,
            elevation_lidar_deg=elevation, azimuth_world_deg=world_azimuth,
            elevation_world_deg=world_elevation,
        )
        objects[(raw, identity)] = (row, world)
    return dict(
        frame=frame.frame_id, slots=frame.slot_count, returns=len(slots),
        empty_slots=int(frame.zero_slot_mask.sum()),
        duplicate_records=frame.real_count - len(slots), classes=classes, objects=objects,
    )


def _init_worker(root):
    global _sequence
    _sequence = STUSequence(root, 206)


def _read_frame(frame):
    return frame_statistics(_sequence[frame])


def cloud_distances(first, second):
    """Unregistered bidirectional distances in the provided world frame."""
    forward = cKDTree(second).query(first, workers=1)[0]
    backward = cKDTree(first).query(second, workers=1)[0]
    # Each direction has equal standing even when return counts differ greatly.
    return np.maximum(
        np.quantile(forward, [0.5, 0.95]), np.quantile(backward, [0.5, 0.95])
    ).tolist()


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(("min", "median", "p95", "max"),
                    np.quantile(values, [0, 0.5, 0.95, 1]).tolist()))


def summarize_track(key, observations):
    raw, identity = key
    rows, clouds = zip(*observations)
    frame_ids = [row["frame"] for row in rows]
    all_points = np.concatenate(clouds)
    centers = np.array([cloud.mean(axis=0) for cloud in clouds])
    for index in range(1, len(rows)):
        current, previous = rows[index], rows[index - 1]
        median, tail = cloud_distances(clouds[index - 1], clouds[index])
        current.update(
            previous_observed_frame=previous["frame"],
            gap_frames=current["frame"] - previous["frame"],
            centroid_displacement_m=float(np.linalg.norm(centers[index] - centers[index - 1])),
            nn_world_median_m=median, nn_world_p95_m=tail,
        )
    # Temporal thirds expose long-term drift that adjacent scans alone can hide.
    blocks = []
    for indices in np.array_split(np.arange(len(rows)), 3):
        if len(indices):
            blocks.append((int(frame_ids[indices[0]]), int(frame_ids[indices[-1]]),
                           np.concatenate([clouds[i] for i in indices])))
    comparisons = []
    for first, second in zip(blocks, blocks[1:]):
        median, tail = cloud_distances(first[2], second[2])
        comparisons.append(dict(
            first_frame_range=list(first[:2]), second_frame_range=list(second[:2]),
            nn_world_median_m=median, nn_world_p95_m=tail,
        ))
    motion = "moving_label" if raw >= 252 else "nonmoving_label"
    status = "repeated_id_geometry_recorded" if len(rows) > 1 else "single_frame_only"
    return dict(
        structure_id=structure_id(raw, identity), semantic=raw, category=LABELS[raw],
        instance_id=identity, motion_label=motion, association_status=status,
        first_frame=frame_ids[0], last_frame=frame_ids[-1], observed_frames=len(rows),
        unobserved_frames_inside_span=frame_ids[-1] - frame_ids[0] + 1 - len(rows),
        max_observation_gap_frames=max(np.diff(frame_ids).tolist(), default=None),
        returns_all=sum(row["returns_all"] for row in rows),
        returns_2p5_50m=sum(row["returns_2p5_50m"] for row in rows),
        returns_per_observed_frame=quantiles([row["returns_all"] for row in rows]),
        observed_frames_with_1_to_4_returns=sum(row["returns_all"] <= 4 for row in rows),
        observed_median_range_m=quantiles([row["range_median_m"] for row in rows]),
        world_min_m=all_points.min(axis=0).tolist(), world_max_m=all_points.max(axis=0).tolist(),
        world_extent_m=np.ptp(all_points, axis=0).tolist(),
        observed_centroid_extent_m=np.ptp(centers, axis=0).tolist(),
        consecutive_observations_nn_median_m=quantiles([row["nn_world_median_m"] for row in rows[1:]]),
        temporal_thirds=comparisons,
    )


def inventory(frames, tracks):
    by_class = defaultdict(list)
    for track in tracks:
        by_class[track["semantic"]].append(track)
    result = []
    for raw, category in LABELS.items():
        found = [frame["classes"][raw] for frame in frames if raw in frame["classes"]]
        identities = by_class[raw]
        total = sum(row["returns_all"] for row in found)
        labeled = sum(row["returns_with_instance"] for row in found)
        if not found:
            unit, association = "not_observed", "本序列未观测到该类别，不能判断标注能力"
        elif raw <= 2:
            unit, association = "excluded_label", "保留原始标签计数，不作为正常结构实例"
        elif identities:
            unit = "annotated_instance"
            association = "按原始类别和非零编号关联；世界几何证据见 summary.json，未自动认证固定性"
        else:
            unit, association = "uninstanced_semantic_points", "无物体身份；固定片区尚未定义，不能按整个类别当作一个物体"
        result.append(dict(
            semantic=raw, category=category, statistical_unit=unit,
            frames_with_returns=len(found), returns_all=total,
            returns_2p5_50m=sum(row["returns_2p5_50m"] for row in found),
            returns_with_instance=labeled, returns_without_instance=total - labeled,
            nonzero_instance_ids=";".join(str(track["instance_id"]) for track in identities),
            instances=len(identities),
            repeated_instances=sum(track["observed_frames"] > 1 for track in identities),
            single_frame_instances=sum(track["observed_frames"] == 1 for track in identities),
            cross_frame_assessment=association,
        ))
    return result


def write_csv(path, fields, rows):
    # Empty numeric fields mean unobserved/undefined, never an imputed zero range.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 8) if isinstance(value, float) else value
                             for key, value in row.items()})


def census(root, output, workers):
    start = time.monotonic()
    sequence = STUSequence(root, 206)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        frames = [frame_statistics(sequence[i]) for i in sequence.frame_ids]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(root,)) as pool:
            frames = list(pool.map(_read_frame, sequence.frame_ids, chunksize=4))
    groups = defaultdict(list)
    collisions = set()
    for frame in frames:
        classes_by_id = defaultdict(list)
        for key, observation in frame["objects"].items():
            groups[key].append(observation)
            classes_by_id[key[1]].append(key[0])
        for identity, categories in classes_by_id.items():
            if len(categories) > 1:
                collisions.add((identity, tuple(sorted(categories))))
    tracks = [summarize_track(key, groups[key]) for key in sorted(groups)]
    categories = inventory(frames, tracks)
    observations = []
    for key in sorted(groups):
        observed = {row["frame"]: row for row, cloud in groups[key]}
        for frame in sequence.frame_ids:
            observations.append(observed.get(frame, dict(
                structure_id=structure_id(*key), semantic=key[0], instance_id=key[1],
                frame=frame, returns_all=0, returns_2p5_50m=0,
            )))
    summary = dict(
        source="STU/train/206", label_source=LABEL_SOURCE, frames=len(sequence),
        stored_slots=sum(frame["slots"] for frame in frames),
        empty_slots=sum(frame["empty_slots"] for frame in frames),
        duplicate_records=sum(frame["duplicate_records"] for frame in frames),
        independent_returns=sum(frame["returns"] for frame in frames),
        instance_identities=len(tracks), repeated_identities=sum(t["observed_frames"] > 1 for t in tracks),
        observed_instance_frames=sum(t["observed_frames"] for t in tracks),
        observation_rows=len(observations),
        reused_numeric_ids=[dict(instance_id=i, simultaneous_raw_classes=list(c)) for i, c in sorted(collisions)],
        definitions={
            "identity": "序列、原始语义类别和非零实例编号共同组成；不合并运动/非运动类别，不把编号 0 当作物体。",
            "independent_return": "使用 SourceFrame.observation_slots：有限且 XYZ 非全零的原始独立文件槽；不按坐标去重。本序列无已知重复射线块。",
            "range": "原始雷达坐标欧氏距离；总回波不裁剪，另列 2.5≤距离≤50 米的回波数，范围计算沿用现有 float32 约定。保留 1—4 回波观测。",
            "zero_rows": "每个已发现身份列出全部 449 帧；计数 0 只表示没有该标注身份的有效回波，不证明物体离场、被遮挡或不存在。其距离、方向和几何字段留空，不插值。",
            "world": "p_world = p_lidar @ R.T + t；[R,t] = inv(Tr) @ pose_camera @ Tr，与现有 STUSequence 一致。世界轴为初始雷达参考轴，不是地理方位。",
            "centroid": "本帧该身份全部有效回波的均值；是被观测表面的中心，随可见部位变化，不是完整物体中心或真实运动轨迹。",
            "bearing": "雷达原点指向观测回波中心；方位角 atan2(y,x)，仰角 atan2(z,hypot(x,y))，单位度。分别在本帧雷达轴和世界参考轴表达，不是表面入射角。",
            "geometry": "相继两次有观测的帧，不限于相邻帧；直接用已给位姿变换后的所有实例点，不再配准。双向最近点距离分别求中位数/95分位，再取两方向较大值。缺帧跨度另列，距离不是速度。",
            "temporal_thirds": "按有观测帧的时间顺序等分三段，段内汇集全部点，比较相继两段的双向最近点距离。检查长期世界几何，不设合成匹配或身份通过阈值。",
            "association": "重复编号和局部世界几何只能支持身份一致性；稀疏采样、部位变化、姿态变化、位姿误差和真实运动都能改变距离。输出证据，不把几何阈值当人工身份真值。",
            "uninstanced": "道路等连续类别及未标实例的杆状物、树干、标志等只做类别覆盖清点；没有定义固定世界片区，不生成其物体曲线。原始标签 0/1/2 单列，不修改现有训练目标。",
        },
        tracks=tracks,
        execution=dict(workers=workers, wall_seconds=time.monotonic() - start),
    )
    # Reconcile every recorded return with its label inventory and object table.
    if sum(row["returns_all"] for row in categories) != summary["independent_returns"]:
        raise ValueError("class inventory lost observed returns")
    if sum(row["returns_all"] for row in observations) != sum(t["returns_all"] for t in tracks):
        raise ValueError("observation table lost instance returns")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "identity.csv", list(categories[0]), categories)
    write_csv(output / "observations.csv", OBSERVATION_FIELDS, observations)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/normal206"))
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    summary = census(args.data_root, args.output, args.workers)
    print(json.dumps({key: summary[key] for key in (
        "source", "frames", "independent_returns", "instance_identities",
        "observed_instance_frames", "observation_rows", "execution",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
