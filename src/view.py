"""Project one labelled STU scan through a fixed virtual equidistant camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import groupby
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import FrozenFrame
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
    """Keep original point identities and the actual detection supervision rules."""
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
    return source.xyzi[slots, :3], target[slots]


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
        return self.width / np.deg2rad(self.horizontal_fov_degrees)

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
        transverse = np.hypot(camera[:, 0], camera[:, 1])
        theta = np.arctan2(transverse, camera[:, 2])
        # Equidistant fisheye: image radius = f * theta, not f * tan(theta).
        scale = np.divide(self.focal_px * theta, transverse,
                          out=np.zeros_like(theta), where=transverse > 0)
        uv = camera[:, :2] * scale[:, None] + [self.width / 2, self.height / 2]
        visible = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < self.width)
                   & (uv[:, 1] >= 0) & (uv[:, 1] < self.height))
        indices = np.flatnonzero(visible)
        return indices, uv[indices], np.linalg.norm(camera[indices], axis=1)

    def rasterize(self, xyz):
        """Resolve finite point footprints by range, independently of labels."""
        indices, uv, ranges = self.project(xyz)
        pixel = np.floor(uv).astype(np.int64)
        linear = pixel[:, 1] * self.width + pixel[:, 0]
        order = np.lexsort((indices, ranges, linear))
        # Same-size footprints make farther points at an identical centre redundant.
        first = np.r_[True, np.diff(linear[order]) != 0] if len(order) else np.zeros(0, bool)
        kept = order[first]
        pixel, ranges, point_ids = pixel[kept], ranges[kept], indices[kept]
        depth = np.full(self.height * self.width, np.inf)
        owner = np.full(self.height * self.width, -1, np.int64)
        radius = self.point_radius_px
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x, y = pixel[:, 0] + dx, pixel[:, 1] + dy
                inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                destination = y[inside] * self.width + x[inside]
                distance, ids = ranges[inside], point_ids[inside]
                # Each offset has unique destinations. Equal ranges use source-slot order.
                nearer = ((distance < depth[destination])
                          | ((distance == depth[destination]) & (ids < owner[destination])))
                destination = destination[nearer]
                depth[destination], owner[destination] = distance[nearer], ids[nearer]
        return owner.reshape(self.height, self.width), indices


def _text_writer(draw, size):
    from matplotlib.ft2font import FT2Font
    from .profile_report import _geometry_fonts

    paths = [font.get_file() for font in _geometry_fonts()]
    fonts = [ImageFont.truetype(path, size) for path in paths]
    faces = [FT2Font(path) for path in paths]
    names = [font.getname()[0] for font in fonts]
    if names != ["SimSun", "Times New Roman"]:
        raise ValueError("rendered text must use actual SimSun and Times New Roman fonts")

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


def save_view(sample, frame_path, output, settings):
    if settings["projection"] != "equidistant_fisheye":
        raise ValueError("this camera uses the equidistant fisheye projection")
    camera = Camera(**settings["camera"])
    xyz, target = labelled_points(sample)
    owner, projected = camera.rasterize(xyz)
    colors = {int(label): tuple(rgb) for label, rgb in settings["label_colors"].items()}
    if set(colors) != {-1, 0, 1} or not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("detection targets must be ignore=-1, normal=0, anomaly=1")
    background = tuple(settings["background_rgb"])
    for rgb in [background, *colors.values()]:
        if len(rgb) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in rgb):
            raise ValueError("colors must be three integer RGB channels in [0,255]")
    rgb = np.empty((camera.height, camera.width, 3), np.uint8)
    rgb[:] = background
    occupied = owner >= 0
    pixel_labels = target[owner[occupied]]
    palette = np.array([colors[-1], colors[0], colors[1]], np.uint8)
    rgb[occupied] = palette[pixel_labels + 1]
    displayed = np.unique(owner[occupied])
    counts = {
        scope: {str(label): int(np.count_nonzero(values == label)) for label in (-1, 0, 1)}
        for scope, values in (("scan", target), ("in_image", target[projected]),
                              ("displayed", target[displayed]))
    }
    source = sample.source if isinstance(sample, FrozenFrame) else sample
    frame_path = Path(frame_path).resolve()
    world = frame_path.parent.parent.name if isinstance(sample, FrozenFrame) else "original"
    image = Image.new("RGB", (camera.width, camera.height + 112), background)
    image.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(image)
    write, fonts = _text_writer(draw, 24)
    for x, label, name in ((24, 0, "正常"), (244, 1, "异常"), (464, -1, "忽略")):
        draw.rectangle((x, camera.height + 13, x + 16, camera.height + 29), fill=colors[label])
        write(x + 28, camera.height + 31, f"{name} ({label})", colors[label])
    foreground = (222, 226, 232)
    write(760, camera.height + 31,
          f"异常点：整帧 {counts['scan']['1']} / 画内 {counts['in_image']['1']} / 显示 {counts['displayed']['1']}", foreground)
    write(24, camera.height + 66,
          f"来源 {source.partition}/{source.sequence_id}   世界 {world}   帧 {source.frame_id:06d}", foreground)
    offset = ", ".join(f"{value:g}" for value in camera.offset_lidar_m)
    write(24, camera.height + 101,
          f"等距鱼眼 · 水平视场 {camera.horizontal_fov_degrees:g}° · 固定相机位置 ({offset}) m · 点云真值投影", foreground)
    metadata = dict(
        input=str(frame_path), partition=source.partition, sequence=source.sequence_id,
        frame=source.frame_id, world=world,
        world_identity=sample.world_identity if isinstance(sample, FrozenFrame) else None,
        projection=settings["projection"], camera=settings["camera"],
        focal_px=camera.focal_px, principal_point_px=[camera.width / 2, camera.height / 2],
        vertical_centerline_fov_degrees=camera.horizontal_fov_degrees * camera.height / camera.width,
        lidar_to_camera=camera.lidar_to_camera.tolist(),
        coordinate_convention="LiDAR x forward, y left, z up; camera x right, y down, z forward",
        pose_basis="assumed virtual driver position, not measured camera calibration",
        truth="frozen insertion and valid normal class map; public val uses released binary labels",
        label_colors=settings["label_colors"], background_rgb=list(background), counts=counts,
        occupied_pixels=int(occupied.sum()), fonts=fonts,
        visibility="nearest observed point footprint; unseen surfaces are unknown",
        camera_image_size=[camera.width, camera.height], image_size=list(image.size), footer_height=112,
    )
    output = Path(output)
    if output.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("output filename must end in .jpg or .jpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[270] = json.dumps(metadata, ensure_ascii=True)
    image.save(output, format="JPEG", quality=95, subsampling=0, exif=exif)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="将一帧带真值的点云投影为固定120°广角 JPG。")
    parser.add_argument("frame", type=Path, help="合成 frames/<帧>.npz 或原始 velodyne/<帧>.bin")
    parser.add_argument("--output", type=Path, help="输出 JPG；默认写入 results/view/")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "protocol/data.json")
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasongao/Data/STU"),
                        help="还原合成差量所需的原始 STU 根目录")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = load_protocol(args.config.parent / config["base_protocol"])
    sample = load_frame(args.frame, args.data_root, protocol)
    source = sample.source if isinstance(sample, FrozenFrame) else sample
    world = args.frame.parent.parent.name if isinstance(sample, FrozenFrame) else "original"
    output = args.output or PROJECT_ROOT / "results/view" / f"{source.partition}_{source.sequence_id}_{world}_{source.frame_id:06d}.jpg"
    metadata = save_view(sample, args.frame, output, config["visualization"])
    print(json.dumps(dict(output=str(output.resolve()), counts=metadata["counts"],
                          camera_image_size=metadata["camera_image_size"],
                          image_size=metadata["image_size"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
