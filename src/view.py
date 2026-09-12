"""Render labelled scan previews and world maps with the recorded vehicle trajectory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby, product
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import FrozenFrame, host_disk
from .protocol import PROJECT_ROOT, load_protocol
from .scene import STUSequence


def load_frame(path, data_root, protocol):
    """Read one complete scan, reconstructing a frozen delta when necessary."""
    path = Path(path).expanduser().resolve(strict=True)
    if not path.stem.isdecimal():
        raise ValueError("frame filenames must be numeric")
    frame_id = int(path.stem)
    if path.suffix == ".npz" and path.parent.name == "frames":
        manifest = json.loads((path.parent.parent / "manifest.json").read_text())
        if frame_id not in {row["frame"] for row in manifest["frames"]}:
            raise ValueError("frame is absent from its world manifest")
        sequence = STUSequence.open(
            data_root, protocol=protocol, partition="train",
            sequence_id=manifest["source_sequence"], label_mode="required",
        )
        return FrozenFrame.load(path, sequence[frame_id], manifest["world_identity"])
    if path.suffix == ".bin" and path.parent.name == "velodyne":
        partition = path.parents[2].name
        if partition not in {"train", "val"}:
            raise ValueError("only normal sources and public development scans are allowed")
        sequence = STUSequence.open(
            path.parents[3], protocol=protocol, partition=partition,
            sequence_id=int(path.parents[1].name), label_mode="required",
        )
        return sequence[frame_id]
    raise ValueError("expected a frozen frames/<frame>.npz or STU velodyne/<frame>.bin")


def labelled_points(sample):
    """Return every real XYZI slot with its actual detection supervision target."""
    if isinstance(sample, FrozenFrame):
        source, target = sample.source, sample.anomaly_target
    else:
        source = sample
        if source.labels is None:
            raise ValueError("ground-truth visualization requires labels")
        if source.partition == "train":
            if source.labels.semantic_target is None:
                raise ValueError("normal-source targets require the training class map")
            # Raw semantic 2 in a normal source is not a synthetic positive.
            target = np.where(source.labels.semantic_target != 255, 0, -1)
        elif source.partition == "val":
            target = source.labels.anomaly_target
        else:
            raise ValueError("hidden test data is outside the visualization scope")
    slots = source.real_slots
    # Empty LiDAR slots must be removed before translating to the camera origin.
    return source.xyzi[slots], target[slots]


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    horizontal_fov_degrees: float
    offset_lidar_m: tuple[float, float, float]
    point_radius_px: int

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.width, self.height)):
            raise ValueError("camera dimensions must be positive integers")
        if not 0 < self.horizontal_fov_degrees < 180:
            raise ValueError("horizontal field of view must be between 0 and 180 degrees")
        offset = np.asarray(self.offset_lidar_m)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("camera offset must contain three finite metre values")
        if type(self.point_radius_px) is not int or not 0 <= self.point_radius_px < min(self.width, self.height):
            raise ValueError("point radius must be a nonnegative integer smaller than the image")

    @property
    def focal_px(self):
        return self.width / (2 * np.tan(np.deg2rad(self.horizontal_fov_degrees) / 2))

    @property
    def vertical_fov_degrees(self):
        return float(np.rad2deg(2 * np.arctan(self.height / (2 * self.focal_px))))

    @property
    def lidar_to_camera(self):
        # LiDAR: forward/left/up. Camera: right/down/forward; no world-pose rotation.
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        matrix[:3, 3] = -matrix[:3, :3] @ np.asarray(self.offset_lidar_m)
        return matrix

    def project(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("projection requires finite xyz[N,3] actual returns")
        transform = self.lidar_to_camera
        camera = xyz @ transform[:3, :3].T + transform[:3, 3]
        centre = np.array([self.width / 2, self.height / 2])
        # Clip the viewing pyramid before division, including the camera-plane singularity.
        limits = camera[:, 2, None] * (centre / self.focal_px)
        indices = np.flatnonzero((camera[:, 2] > 0)
                                 & np.all(np.abs(camera[:, :2]) <= limits, axis=1))
        # Rectilinear perspective preserves 3D line collinearity; square pixels use fx = fy.
        uv = self.focal_px * (camera[indices, :2] / camera[indices, 2, None]) + centre
        inside = np.all((uv >= 0) & (uv < [self.width, self.height]), axis=1)
        return indices[inside], uv[inside]

    def rasterize(self, xyz, point_colors, background):
        """Average all overlapping point colours; never discard a point by depth."""
        point_colors = np.asarray(point_colors, dtype=np.float64)
        if point_colors.shape != (len(xyz), 3) or not np.isfinite(point_colors).all():
            raise ValueError("each input point requires a finite RGB colour")
        indices, uv = self.project(xyz)
        pixel = np.floor(uv).astype(np.int64)
        total = np.zeros((self.height * self.width, 3), np.float64)
        contributions = np.zeros(self.height * self.width, np.int64)
        radius = self.point_radius_px
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x, y = pixel[:, 0] + dx, pixel[:, 1] + dy
                inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                destination = y[inside] * self.width + x[inside]
                # Repeated indices must accumulate, including coincident source returns.
                np.add.at(total, destination, point_colors[indices[inside]])
                np.add.at(contributions, destination, 1)
        occupied = contributions > 0
        rgb = np.empty((self.height * self.width, 3), np.uint8)
        rgb[:] = background
        rgb[occupied] = np.rint(total[occupied] / contributions[occupied, None]).astype(np.uint8)
        return rgb.reshape(self.height, self.width, 3), indices, contributions.reshape(self.height, self.width)


def intensity_colors(intensity, target, colors, background, settings):
    """Use one fixed monotonic intensity transfer for every frame and label."""
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.shape != target.shape or not np.isfinite(intensity).all():
        raise ValueError("every point requires its original finite return intensity")
    half = settings["half_saturation"]
    minimum = settings["minimum_contrast"]
    if not np.isfinite(half) or half <= 0 or not 0 < minimum < 1:
        raise ValueError("intensity half-saturation must be positive; minimum contrast must be in (0,1)")
    positive = np.maximum(intensity, 0)
    # No per-frame/label normalisation or clipping at 1: raw STU intensities can exceed 1.
    contrast = minimum + (1 - minimum) * positive / (positive + half)
    palette = np.asarray([colors[-1], colors[0], colors[1]], dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    return background + contrast[:, None] * (palette[target + 1] - background)


@lru_cache(maxsize=4)
def _fonts(size):
    from matplotlib.ft2font import FT2Font
    from .profile_report import _geometry_fonts

    paths = [font.get_file() for font in _geometry_fonts()]
    fonts = [ImageFont.truetype(path, size) for path in paths]
    faces = [FT2Font(path) for path in paths]
    names = [font.getname()[0] for font in fonts]
    if names != ["SimSun", "Times New Roman"]:
        raise ValueError("rendered text must use actual SimSun and Times New Roman fonts")
    return fonts, faces, names


def _text_writer(draw, size):
    fonts, faces, names = _fonts(size)

    def font_index(character):
        return 0 if "\u2e80" <= character <= "\u9fff" or "\uff00" <= character <= "\uffef" else 1

    def write(x, baseline, text, color):
        for index, characters in groupby(text, font_index):
            span = "".join(characters)
            if any(faces[index].get_char_index(ord(c)) == 0 for c in span):
                raise ValueError(f"required font lacks a glyph in {span!r}")
            draw.text((x, baseline), span, font=fonts[index], fill=color, anchor="ls")
            x += fonts[index].getlength(span)
        return x

    return write, names


def save_view(sample, frame_path, output, settings, *, preview=None):
    if settings["projection"] != "rectilinear_perspective":
        raise ValueError("this camera uses the rectilinear perspective projection")
    camera = Camera(**settings["camera"])
    xyzi, target = labelled_points(sample)
    colors = {int(label): tuple(rgb) for label, rgb in settings["label_colors"].items()}
    if set(colors) != {-1, 0, 1} or not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("detection targets must be ignore=-1, normal=0, anomaly=1")
    background = tuple(settings["background_rgb"])
    for rgb in [background, *colors.values()]:
        if len(rgb) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in rgb):
            raise ValueError("colors must be three integer RGB channels in [0,255]")
    point_colors = intensity_colors(xyzi[:, 3], target, colors, background, settings["intensity"])
    rgb, projected, contributions = camera.rasterize(xyzi[:, :3], point_colors, background)
    counts = {
        scope: {str(label): int(np.count_nonzero(values == label)) for label in (-1, 0, 1)}
        for scope, values in (("scan", target), ("in_image", target[projected]),
                              ("drawn", target[projected]))
    }
    source = sample.source if isinstance(sample, FrozenFrame) else sample
    frame_path = Path(frame_path).resolve()
    world = frame_path.parent.parent.name if isinstance(sample, FrozenFrame) else "original"
    displayed_world = (f"{preview['world']} (original)"
                       if preview is not None and preview["role"] == "original" else world)
    image = Image.new("RGB", (camera.width, camera.height + 112), background)
    image.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(image)
    write, fonts = _text_writer(draw, 24)
    for x, label, name in ((24, 0, "正常"), (244, 1, "异常"), (464, -1, "忽略")):
        draw.rectangle((x, camera.height + 13, x + 16, camera.height + 29), fill=colors[label])
        write(x + 28, camera.height + 31, f"{name} ({label})", colors[label])
    foreground = (222, 226, 232)
    write(760, camera.height + 31,
          f"全量点：整帧 {len(xyzi)} / 画内 {len(projected)} / 绘制 {len(projected)}", foreground)
    write(24, camera.height + 66,
          f"来源 {source.partition}/{source.sequence_id}   世界 {displayed_world}   帧 {source.frame_id:06d}   异常点：整帧 {counts['scan']['1']} / 绘制 {counts['drawn']['1']}", foreground)
    offset = ", ".join(f"{value:g}" for value in camera.offset_lidar_m)
    note = ""
    if preview is not None:
        if preview["role"] == "original":
            paired = preview["paired_with"]
            note = f"原始背景 · 对应 {paired} · " if paired else "原始背景 · 三档均无可见预览 · "
        else:
            note = f"{preview['role']} · 异常回波中位距离 {preview['range_median_m']:.2f} m · "
    write(24, camera.height + 101,
          note + f"广角透视 · 水平视场 {camera.horizontal_fov_degrees:g}° · 固定相机位置 ({offset}) m · 强回波更亮 · 全量融合", foreground)
    metadata = dict(
        input=str(frame_path), partition=source.partition, sequence=source.sequence_id,
        frame=source.frame_id, world=world,
        world_identity=sample.world_identity if isinstance(sample, FrozenFrame) else None,
        projection=settings["projection"], camera=settings["camera"],
        focal_px=camera.focal_px, principal_point_px=[camera.width / 2, camera.height / 2],
        vertical_centerline_fov_degrees=camera.vertical_fov_degrees,
        lidar_to_camera=camera.lidar_to_camera.tolist(),
        coordinate_convention="LiDAR x forward, y left, z up; camera x right, y down, z forward",
        pose_basis="assumed virtual driver position, not measured camera calibration",
        truth="frozen insertion and valid normal class map; public val uses released binary labels",
        label_colors=settings["label_colors"], background_rgb=list(background), counts=counts,
        occupied_pixels=int(np.count_nonzero(contributions)), fonts=fonts,
        overlap_pixels=int(np.count_nonzero(contributions > 1)),
        point_pixel_contributions=int(contributions.sum()),
        point_sampling="none; every nonzero XYZ source slot, including repeated coordinates",
        overlap="equal contribution of all point footprints; no depth rejection or overpainting",
        intensity=dict(settings=settings["intensity"], source="unaltered XYZI channel 3",
                       mapping="contrast = minimum + (1 - minimum) * max(I,0) / (max(I,0) + half_saturation)",
                       negative_returns=int(np.count_nonzero(xyzi[:, 3] < 0))),
        camera_image_size=[camera.width, camera.height], image_size=list(image.size), footer_height=112,
    )
    if preview is not None:
        metadata["preview"] = preview
    _save_jpeg(image, output, metadata)
    return metadata


def _save_jpeg(image, output, metadata):
    output = Path(output)
    if output.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("output filename must end in .jpg or .jpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[270] = json.dumps(metadata, ensure_ascii=True)
    image.save(output, format="JPEG", quality=95, subsampling=0, exif=exif)


def select_preview(rows, directory, identity, camera, bounds):
    """Choose the closest band-midpoint scan with actual foreground in the fixed view."""
    centre = sum(bounds) / 2
    candidates = sorted(rows, key=lambda row: (abs(row["range"] - centre), row["frame"]))
    for row in candidates:
        path = directory / "frames" / f"{row['frame']:06d}.npz"
        with np.load(path, allow_pickle=False) as delta:
            if (delta["world_identity"].item() != identity
                    or delta["source_identity"].item() != row["source_identity"]):
                raise ValueError("preview observations refer to a different frozen scan")
            inserted = delta["inserted_slot"]
            slots = delta["source_slot"]
            index = np.searchsorted(slots, inserted)
            if np.any(index >= len(slots)) or not np.array_equal(slots[index], inserted):
                raise ValueError("inserted returns must be present in the frozen delta")
            xyz = delta["xyzi"][index, :3]
        if len(camera.project(xyz)[0]):
            return row, None
    return None, "outside_front_view" if candidates else "no_range_observation"


def _missing_view(output, settings, preview):
    camera = Camera(**settings["camera"])
    image = Image.new("RGB", (camera.width, camera.height + 112), tuple(settings["background_rgb"]))
    write, fonts = _text_writer(ImageDraw.Draw(image), 32)
    reason = ("该距离档没有实际异常回波观测" if preview["reason"] == "no_range_observation"
              else "该距离档的异常回波均在固定前视画面之外")
    low, high = preview["distance_bounds_m"]
    for y, text in ((120, f"{preview['role']} · 缺少可见预览"),
                    (185, f"{preview['split']} / {preview['world']}"),
                    (250, f"异常回波中位距离档：{low:g}—{high:g} m"),
                    (315, reason), (380, "此图为缺失说明，不是点云渲染图。")):
        write(40, y, text, (222, 226, 232))
    _save_jpeg(image, output, dict(input=None, counts=None, preview=preview,
        camera=settings["camera"], projection=settings["projection"],
        image_size=list(image.size), fonts=fonts))


def _initialize_views(data_root, protocol_path, settings, bins):
    global _view_sources, _view_settings, _view_bins
    protocol = load_protocol(protocol_path)
    _view_sources = {split: STUSequence.open(data_root, protocol=protocol, partition="train",
        sequence_id=sequence, label_mode="required") for split, sequence in (("train", 206), ("validation", 201))}
    _view_settings, _view_bins = settings, bins
    _fonts(24)
    _fonts(32)


def _world_views(job):
    split, directory, identity, rows, output = job
    source = _view_sources[split]
    settings = _view_settings
    camera = Camera(**settings["camera"])
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["world_identity"] != identity or manifest["source_sequence"] != source.spec.sequence_id:
        raise ValueError("world preview must use its unchanged normal source")
    frames = {r["frame"] for r in manifest["frames"]}
    if any(row["frame"] not in frames for values in rows.values() for row in values):
        raise ValueError("preview frame is absent from the frozen world")
    chosen, missing, paired_source = {}, {}, None
    common = dict(split=split, world=directory.name, world_identity=identity,
        selection="closest band midpoint with >=1 actual anomaly in fixed front view; earliest frame breaks ties")
    for index, role in enumerate(("near", "middle", "far")):
        bounds = _view_bins[index:index + 2]
        row, reason = select_preview(rows[role], directory, identity, camera, bounds)
        preview = dict(common, role=role, distance_bounds_m=bounds, candidate_frames=len(rows[role]))
        path = output / f"{role}.jpg"
        if row is None:
            missing[role] = reason
            _missing_view(path, settings, dict(preview, status="missing", reason=reason))
            continue
        frame = row["frame"]
        frame_path = directory / "frames" / f"{frame:06d}.npz"
        original = source[frame]
        if not chosen:
            paired_source = original
        sample = FrozenFrame.load(frame_path, original, identity)
        metadata = save_view(sample, frame_path, path, settings,
            preview=dict(preview, status="rendered", range_median_m=row["range"]))
        if metadata["counts"]["drawn"]["1"] < 1:
            raise ValueError("selected preview must contain actual foreground in the image")
        chosen[role] = row
    # A single original is paired with near, then middle/far when near is missing.
    paired = next(iter(chosen), None)
    if paired is not None:
        frame = chosen[paired]["frame"]
    else:
        available = [r for values in rows.values() for r in values]
        frame = min(available, key=lambda r: (r["range"], r["frame"]))["frame"] if available else min(frames)
    original_path = source.sequence_dir / "velodyne" / f"{frame:06d}.bin"
    save_view(paired_source if paired_source is not None else source[frame], original_path,
        output / "original.jpg", settings, preview=dict(common, role="original", status="rendered", paired_with=paired,
            selection="same source frame as near, else middle/far; if all missing, shortest observed range or first source frame"))
    return dict(split=split, world=directory.name, selected_frames={k:r["frame"] for k,r in chosen.items()},
        missing=missing, original_frame=frame, original_paired_with=paired,
        bytes=sum((output / f"{role}.jpg").stat().st_size for role in ("near", "middle", "far", "original")))


def batch_views(directory, data_root, config, protocol_path, output=None, workers=16):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "stu-frozen-dataset" or manifest.get("status") != "frozen":
        raise ValueError("batch previews require the current frozen world manifest")
    bins = config["research_coverage"]["distance_bins_m"]
    if bins != [2.5, 10, 35, 50]:
        raise ValueError("preview bands must use the current declared distance definitions")
    members = {w["world_identity"]:(split, w) for split, data in manifest["splits"].items() for w in data["worlds"]}
    rows = defaultdict(lambda: defaultdict(list))
    observations = Path(config["research_coverage"]["output"]) / "observations.csv"
    with observations.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row["world_identity"]
            if identity not in members or not row["range"] or int(row["in_range_rays"]) < 1:
                continue
            split, world = members[identity]
            if row["split"] != split or row["world"] != Path(world["path"]).name:
                raise ValueError("coverage observations and current world names disagree")
            distance = float(row["range"])
            role = ("near" if 2.5 <= distance < 10 else "middle" if 10 <= distance < 35
                    else "far" if 35 <= distance <= 50 else None)
            if role:
                rows[identity][role].append(dict(frame=int(row["frame"]), range=distance,
                    source_identity=row["source_identity"]))
    jobs = []
    for identity, (split, world) in members.items():
        path = directory / world["path"]
        destination = Path(output) / split / path.name if output is not None else path
        destination.mkdir(parents=True, exist_ok=True)
        jobs.append((split, path, identity,
            {role:rows[identity][role] for role in ("near", "middle", "far")}, destination))
    if workers < 1 or workers > len(os.sched_getaffinity(0)):
        raise ValueError("workers must fit the available CPU allocation")
    disk = host_disk()
    # Include all JPEGs and concurrent writers without relying on ext4 free space.
    planned_bytes = len(jobs) * 4 * 3_000_000 + workers * 8_000_000
    if disk["SizeRemaining"] - planned_bytes < disk["reserve_bytes"]:
        raise OSError("preview batch would enter the host E: 10 GB reserve")
    started = checked = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
            initializer=_initialize_views, initargs=(str(data_root), str(protocol_path), config["visualization"], bins)) as pool:
        futures = [pool.submit(_world_views, job) for job in jobs]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                if time.monotonic() - checked >= 30:
                    host_disk()
                    checked = time.monotonic()
                if len(results) % 30 == 0 or len(results) == len(jobs):
                    print(json.dumps(dict(completed_worlds=len(results), total_worlds=len(jobs),
                        seconds=round(time.monotonic()-started, 2))), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return dict(worlds=len(results), images=4*len(results), bytes=sum(r["bytes"] for r in results),
        missing={split:dict(Counter(f"{role}:{reason}" for r in results if r["split"] == split
            for role, reason in r["missing"].items())) for split in ("train", "validation")},
        original_pairing=dict(Counter(r["original_paired_with"] or "none" for r in results)),
        seconds=time.monotonic()-started, workers=workers, host_before=disk, host_after=host_disk())


def map_pixels(xy, bounds, size):
    """Orthographic world XY projection: equal metre scale, world +Y upwards."""
    bounds = np.asarray(bounds, float)
    size = np.asarray(size, int)
    scale = np.min(size / (bounds[1] - bounds[0]))
    pixels = (np.asarray(xy)[..., :2] - bounds.mean(0)) * scale
    return pixels * [1, -1] + size / 2


def map_object(record):
    """Use the saved physical extent only to choose a window containing the object."""
    obj = record["world"]["objects"][0]
    geometry = record["generation"]["geometry"]
    corners = np.array(list(product(*zip(geometry["lower_local_m"], geometry["upper_local_m"]))))
    rotation, centre = np.asarray(obj["rotation_world_from_local"]), np.asarray(obj["translation_world_m"])
    return corners @ rotation.T + centre, centre


def map_returns(directory, manifest, source):
    """Recover all actual inserted returns; never replace a missing return with geometry."""
    points = []
    for row in manifest["frames"]:
        if not row["count"]:
            continue
        frame = row["frame"]
        with np.load(directory / "frames" / f"{frame:06d}.npz", allow_pickle=False) as delta:
            inserted, slots = delta["inserted_slot"], delta["source_slot"]
            indices = np.searchsorted(slots, inserted)
            if (delta["world_identity"].item() != manifest["world_identity"] or len(inserted) != row["count"]
                    or np.any(indices >= len(slots)) or not np.array_equal(slots[indices], inserted)):
                raise ValueError("map visibility and actual frozen inserted returns disagree")
            xyzi = delta["xyzi"][indices]
        pose = source.lidar_pose(frame)
        world = xyzi[:, :3].astype(float) @ pose[:3, :3].T + pose[:3, 3]
        points.append(np.column_stack([world, xyzi[:, 3]]))
    return np.concatenate(points) if points else np.empty((0, 4))


def _map_background(job):
    split, frames, bounds, shape = job
    source, settings = _view_sources[split], _view_settings
    count = np.zeros(shape[0] * shape[1], np.uint32)
    total = np.zeros((len(count), 3), np.float64)
    colors = {int(k): tuple(v) for k, v in settings["label_colors"].items()}
    real = included = 0
    for frame in frames:
        scan = source[int(frame)]
        xyzi, target = labelled_points(scan)
        # Use raw sensor XYZ and the calibrated pose exactly once; coordinates is already transformed.
        world = xyzi[:, :3].astype(float) @ scan.lidar_pose[:3, :3].T + scan.lidar_pose[:3, 3]
        pixel = np.floor((world[:, :2] - bounds[0]) / .1).astype(np.int64)
        valid = np.all((pixel >= 0) & (pixel < shape), axis=1)
        ids = pixel[valid, 1] * shape[0] + pixel[valid, 0]
        rgb = intensity_colors(xyzi[valid, 3], target[valid], colors,
                               settings["background_rgb"], settings["intensity"])
        # Every in-bounds return contributes, including coincident slots and overlapping scans.
        np.add.at(count, ids, 1)
        np.add.at(total, ids, rgb)
        real += len(xyzi)
        included += len(ids)
    return split, count, total, real, included


def _map_arrow(draw, a, b, color, width=4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    if length < 1:
        return
    direction = (b - a) / length
    side = np.array([-direction[1], direction[0]])
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    base = b - direction * min(14, length * .6)
    draw.polygon([tuple(b), tuple(base + 6 * side), tuple(base - 6 * side)], fill=color)


def _map_panel(base, raster_bounds, bounds, size, trajectory, points, centre, closest, visible):
    # Expand world limits to match panel aspect instead of stretching the path or object.
    scale = np.min(np.asarray(size) / np.ptp(bounds, axis=0))
    bounds = bounds.mean(0) + np.array([-1, 1])[:, None] * np.asarray(size) / scale / 2
    crop = ((bounds[0, 0] - raster_bounds[0, 0]) / .1,
            (raster_bounds[1, 1] - bounds[1, 1]) / .1,
            (bounds[1, 0] - raster_bounds[0, 0]) / .1,
            (raster_bounds[1, 1] - bounds[0, 1]) / .1)
    panel = base.transform(size, Image.Transform.EXTENT, crop, resample=Image.Resampling.BILINEAR,
                           fillcolor=tuple(_view_settings["background_rgb"]))
    draw = ImageDraw.Draw(panel)
    write, _ = _text_writer(draw, 20)
    positions = map_pixels(trajectory, bounds, size)
    colors = np.where(np.asarray(visible)[:, None], [255, 218, 45], [194, 116, 255])
    for axis in (0, 1):
        step = 10 ** np.floor(np.log10(np.ptp(bounds, axis=0)[axis] / 5))
        if np.ptp(bounds, axis=0)[axis] / step > 10:
            step *= 5
        for value in np.arange(np.ceil(bounds[0, axis] / step) * step, bounds[1, axis], step):
            if abs(value) < 1e-9:
                value = 0.
            endpoints = bounds.copy()
            endpoints[:, axis] = value
            line = map_pixels(endpoints, bounds, size)
            draw.line([tuple(p) for p in line], fill=(39, 48, 61), width=1)
            if axis == 0:
                write(min(line[0, 0] + 3, size[0] - 44), size[1] - 6, f"{value:g}", (175, 185, 200))
            else:
                write(5, line[0, 1] - 4, f"{value:g}", (175, 185, 200))
    # A recorded origin owns the half-segments on either side; no unseen frame is inferred.
    for i, p in enumerate(positions):
        a = (positions[i-1] + p) / 2 if i else p
        b = (p + positions[i+1]) / 2 if i+1 < len(positions) else p
        draw.line([tuple(a), tuple(p), tuple(b)], fill=tuple(colors[i]), width=4)
    travelled = np.r_[0., np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))]
    for distance in np.linspace(0, travelled[-1], 10)[1:-1]:
        i = min(int(np.searchsorted(travelled, distance)), len(positions) - 2)
        direction = positions[i + 1] - positions[i]
        norm = np.linalg.norm(direction)
        if norm > .001:
            _map_arrow(draw, positions[i], positions[i] + direction / norm * 26, tuple(colors[i]))
    for i, name in ((0, "起点"), (-1, "终点")):
        x, y = positions[i]
        marker = tuple(colors[i])
        if 0 <= x < size[0] and 0 <= y < size[1]:
            draw.ellipse((x-7, y-7, x+7, y+7), fill=marker, outline="white", width=2)
            label_x = x+11 if x+165 < size[0] else x-155
            write(max(label_x, 5), max(y-10, 52), f"{name} {i if i == 0 else len(positions)-1:06d}", marker)
    pixel = np.floor(map_pixels(points, bounds, size)).astype(np.int64)
    valid = np.all((pixel >= 0) & (pixel < size), axis=1)
    ids = pixel[valid, 1] * size[0] + pixel[valid, 0]
    settings = _view_settings
    palette = {int(k): v for k, v in settings["label_colors"].items()}
    rgb = intensity_colors(points[valid, 3], np.ones(valid.sum(), np.int8), palette,
                           settings["background_rgb"], settings["intensity"])
    count = np.zeros(size[0] * size[1], np.uint32)
    total = np.zeros((len(count), 3))
    np.add.at(count, ids, 1)
    np.add.at(total, ids, rgb)
    rendered = np.array(panel).reshape(-1, 3)
    occupied = count > 0
    rendered[occupied] = np.rint(total[occupied] / count[occupied, None]).astype(np.uint8)
    panel = Image.fromarray(rendered.reshape(size[1], size[0], 3))
    draw = ImageDraw.Draw(panel)
    write, _ = _text_writer(draw, 20)
    x, y = map_pixels(centre, bounds, size)
    write(min(max(x+18, 5), size[0]-140), max(y-18, 25), "物体回波", (255, 137, 113))
    px, py = positions[closest]
    if 0 <= px < size[0] and 0 <= py < size[1]:
        draw.ellipse((px-6, py-6, px+6, py+6), outline=(239, 242, 246), width=2)
    write(size[0]-132, size[1]-30, "世界 X (m)", (211, 220, 234))
    write(10, 28, "世界 Y (m)", (211, 220, 234))
    return panel, bounds.tolist()


def _world_map(job):
    split, directory, identity, output = job
    record = json.loads((directory / "world.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    source = _view_sources[split]
    if (manifest["world_identity"] != identity or record["world"]["source_sequence_id"] != source.spec.sequence_id
            or [r["frame"] for r in manifest["frames"]] != list(source.frame_ids)):
        raise ValueError("map source, world and full frame order must agree")
    trajectory = np.array([source.lidar_pose(i)[:2, 3] for i in source.frame_ids])
    visible = np.array([r["count"] > 0 for r in manifest["frames"]])
    points = map_returns(directory, manifest, source)
    corners, centre = map_object(record)
    distance = np.linalg.norm(trajectory - centre[:2], axis=1)
    closest = int(np.argmin(distance))
    base, raster_bounds, background_counts = _map_bases[split]
    content = np.concatenate([trajectory, corners[:, :2]])
    full_bounds = np.array([content.min(0)-12, content.max(0)+12])
    local = np.vstack([corners[:, :2], trajectory[closest]])
    local_bounds = np.array([local.min(0)-5, local.max(0)+5])
    image = Image.new("RGB", (1920, 1240), tuple(_view_settings["background_rgb"]))
    draw = ImageDraw.Draw(image)
    write, fonts = _text_writer(draw, 26)
    write(45, 45, f"俯视地图 · {split} / {directory.name}", (235, 239, 245))
    write(45, 95, "完整行驶轨迹", (211, 220, 234))
    write(1300, 95, "物体附近放大", (211, 220, 234))
    full, full_limits = _map_panel(base, raster_bounds, full_bounds, (1200, 990), trajectory, points, centre, closest, visible)
    zoom, zoom_limits = _map_panel(base, raster_bounds, local_bounds, (610, 610), trajectory, points, centre, closest, visible)
    image.paste(full, (40, 115))
    image.paste(zoom, (1280, 115))
    path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    for y, text in ((778, f"正常来源：{source.spec.sequence_id} · 全部 {len(trajectory)} 帧"),
                    (820, f"扫到物体：{int(visible.sum())} 帧"),
                    (862, f"平面路径长度：{path_length:.2f} m"),
                    (904, f"物体 (X, Y)：({centre[0]:.2f}, {centre[1]:.2f}) m"),
                    (946, f"最近位姿帧：{closest:06d}"),
                    (988, f"至物体原点平面距离：{distance[closest]:.2f} m"),
                    (1046, "红色点：实际扫描到的物体回波"),
                    (1088, "黄色：扫到物体；紫色：未扫到")):
        write(1285, y, text, (211, 220, 234))
    write(45, 1150, "背景：原始正常序列全部回波的俯视投影；蓝色为正常，绿色为忽略，明暗取决于原始强度。", (182, 195, 213))
    write(45, 1192, "路径采用 LiDAR 原点位姿；物体点来自各帧实际插入回波，至少一个回波即标黄。上下方向对应世界 Y，不表示地理北。", (182, 195, 213))
    metadata = dict(projection="orthographic_world_xy", world=directory.name, world_identity=identity,
        source_sequence=source.spec.sequence_id, source_frames=len(trajectory),
        source_pose=str(source.sequence_dir / "poses.txt"), calibration=str(source.sequence_dir / "calib.txt"),
        object_definition=str(directory / "world.json"), object_origin_world_m=centre.tolist(),
        object_bounds_world_m=corners.tolist(), trajectory="all calibrated LiDAR origins in source frame order",
        closest_frame=closest, closest_origin_xy_distance_m=float(distance[closest]), path_length_xy_m=path_length,
        visibility=dict(visible_frames=np.flatnonzero(visible).tolist(), visible_count=int(visible.sum()),
            invisible_count=int((~visible).sum()),
            definition="frozen frame manifest count > 0; at least one actual inserted return in the complete scan",
            filters="no official range/minimum-point filter; independent of the front-view camera",
            colors=dict(visible=[255, 218, 45], invisible=[194, 116, 255]),
            segment_rule="each recorded origin owns adjacent half-segments; midpoint transitions are display interpolation"),
        full_bounds_xy_m=full_limits, zoom_bounds_xy_m=zoom_limits,
        background=dict(**background_counts, grid_m=.1, bounds_xy_m=raster_bounds.tolist(),
            source="all original scans before insertion; every actual return within map bounds contributes",
            aggregation="mean intensity-shaded RGB; no height, depth or point-count rejection"),
        object_display="actual frozen inserted returns from all source frames, transformed once to world XY; no box or substitute points",
        object_return_occurrences=len(points),
        fonts=fonts, image_size=list(image.size))
    _save_jpeg(image, output / "map.jpg", metadata)
    return dict(world=directory.name, split=split, bytes=(output / "map.jpg").stat().st_size)


def batch_maps(directory, data_root, config, protocol_path, output=None, workers=8):
    global _map_bases
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "stu-frozen-dataset" or manifest.get("status") != "frozen":
        raise ValueError("maps require the current frozen world manifest")
    if not 1 <= workers <= len(os.sched_getaffinity(0)):
        raise ValueError("workers must fit the available CPU allocation")
    disk, started = host_disk(), time.monotonic()
    worlds = sum(len(s["worlds"]) for s in manifest["splits"].values())
    if disk["SizeRemaining"] - worlds * 4_000_000 < disk["reserve_bytes"]:
        raise OSError("world maps would enter the host E: reserve")
    _initialize_views(data_root, protocol_path, config["visualization"], None)
    background_jobs, jobs, accumulators = [], [], {}
    for split, data in manifest["splits"].items():
        source = _view_sources[split]
        if data["source_sequence"] != source.spec.sequence_id:
            raise ValueError("map split must keep its declared normal source")
        content = [np.array([source.lidar_pose(i)[:2, 3] for i in source.frame_ids])]
        for world in data["worlds"]:
            path = directory / world["path"]
            content.append(map_object(json.loads((path / "world.json").read_text()))[0][:, :2])
            destination = Path(output) / split / path.name if output is not None else path
            jobs.append((split, path, world["world_identity"], destination))
        content = np.concatenate(content)
        # Pad for both equal-aspect overview limits and local object windows.
        middle = (content.min(0) + content.max(0)) / 2
        half = max(np.ptp(content, axis=0)) * .65 + 30
        bounds = np.array([np.floor((middle-half)*10)/10, np.ceil((middle+half)*10)/10])
        shape = np.rint(np.ptp(bounds, axis=0) / .1).astype(int)
        accumulators[split] = [np.zeros(np.prod(shape), np.uint32), np.zeros((np.prod(shape), 3)), bounds, shape, 0, 0]
        for frames in np.array_split(source.frame_ids, min(workers, len(source))):
            background_jobs.append((split, frames.tolist(), bounds, shape))
    _map_bases = {}
    checked = time.monotonic()
    chunks = len(background_jobs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(_map_background, job) for job in background_jobs}
        for future in as_completed(futures.copy()):
            split, count, total, real, included = future.result()
            acc = accumulators[split]
            acc[0] += count
            acc[1] += total
            acc[4] += real
            acc[5] += included
            futures.remove(future)
            if time.monotonic() - checked > 30:
                host_disk()
                checked = time.monotonic()
            print(json.dumps(dict(event="map_background", source=split, completed_chunks=chunks-len(futures))), flush=True)
    for split, (count, total, bounds, shape, real, included) in accumulators.items():
        occupied = count > 0
        rgb = np.empty((len(count), 3), np.uint8)
        rgb[:] = config["visualization"]["background_rgb"]
        rgb[occupied] = np.rint(total[occupied] / count[occupied, None]).astype(np.uint8)
        _map_bases[split] = (Image.fromarray(rgb.reshape(shape[1], shape[0], 3)[::-1]), bounds,
                             dict(actual_returns=real, projected_returns=included))
    del accumulators, acc, count, total, future
    results, checked = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        for future in as_completed([pool.submit(_world_map, job) for job in jobs]):
            results.append(future.result())
            if time.monotonic() - checked > 30:
                host_disk()
                checked = time.monotonic()
            if len(results) % 30 == 0 or len(results) == worlds:
                print(json.dumps(dict(event="world_maps", worlds=len(results), total=worlds,
                                      seconds=round(time.monotonic()-started, 2))), flush=True)
    return dict(worlds=len(results), images=len(results), bytes=sum(r["bytes"] for r in results),
                seconds=time.monotonic()-started, workers=workers, host_before=disk, host_after=host_disk())


def main():
    parser = argparse.ArgumentParser(description="绘制单帧、每世界四张前视预览或俯视轨迹地图。")
    parser.add_argument("frame", type=Path, nargs="?", help="合成 frames/<帧>.npz 或原始 velodyne/<帧>.bin")
    parser.add_argument("--dataset", type=Path, help="为清单内每世界生成远、中、近和原始背景四张图")
    parser.add_argument("--maps", action="store_true", help="与 --dataset 合用，为每世界生成俯视轨迹与物体位置 map.jpg")
    parser.add_argument("--workers", type=int, default=min(16, len(os.sched_getaffinity(0))))
    parser.add_argument("--output", type=Path,
                        help="单帧输出 JPG（合成帧默认同世界目录，原始帧须指定）；批量输出目录默认各世界原目录")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "protocol/data.json")
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"),
                        help="还原合成差量所需的原始 STU 根目录")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = load_protocol(args.config.parent / config["base_protocol"])
    if (args.frame is None) == (args.dataset is None):
        parser.error("provide either one frame or --dataset")
    if args.maps and args.dataset is None:
        parser.error("--maps requires --dataset")
    if args.dataset is not None:
        render = batch_maps if args.maps else batch_views
        result = render(args.dataset, args.data_root, config, protocol.path, args.output, args.workers)
        print(json.dumps(result, ensure_ascii=False))
        return
    sample = load_frame(args.frame, args.data_root, protocol)
    source = sample.source if isinstance(sample, FrozenFrame) else sample
    if args.output is None and not isinstance(sample, FrozenFrame):
        parser.error("original scans require --output; never write previews into the raw dataset")
    output = args.output or args.frame.parent.parent / f"frame_{source.frame_id:06d}.jpg"
    metadata = save_view(sample, args.frame, output, config["visualization"])
    print(json.dumps(dict(output=str(output.resolve()), counts=metadata["counts"],
                          camera_image_size=metadata["camera_image_size"],
                          image_size=metadata["image_size"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
