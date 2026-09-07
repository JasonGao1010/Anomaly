"""Stream complete, truth-colored STU worlds to PLY and virtual front-view video."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import av
from numba import njit
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import FrozenSyntheticSegment, FrozenWindowDataset
from .protocol import load_protocol
from .scene import LabelMode, SourceFrame, STUSequence
from .train import host_disk


COLORS = np.array(((0, 128, 255), (160, 160, 160), (255, 0, 0)), dtype=np.uint8)
VERTEX_DTYPE = np.dtype(
    [(name, "<f4") for name in ("x", "y", "z")]
    + [(name, "u1") for name in ("red", "green", "blue")]
    + [("label", "<u4"), ("frame", "<u2"), ("slot", "<u4")]
)
WIDTH, HEIGHT, FPS = 1280, 720, 10
HFOV_DEG, PITCH_DEG, POINT_RADIUS = 90.0, 3.0, 2
CAMERA_CENTER = np.array((0.0, 0.0, -0.5), dtype=np.float64)
FONT_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")
_THREAD_FONTS = threading.local()


def camera_coordinates(xyz: np.ndarray) -> np.ndarray:
    """LiDAR: forward/left/up; virtual camera: right/down/forward, pitched down."""
    pitch = np.deg2rad(PITCH_DEG)
    rotation = np.array(
        (
            (0, -1, 0),
            (-np.sin(pitch), 0, -np.cos(pitch)),
            (np.cos(pitch), 0, -np.sin(pitch)),
        ),
        dtype=np.float64,
    )
    return (xyz.astype(np.float64) - CAMERA_CENTER) @ rotation.T


@njit(nogil=True, cache=False)
def visible_points(camera, width, height, focal, radius):
    """Label-blind z-buffer; equal-depth ties retain the first original slot."""
    depth = np.full((height, width), np.inf, dtype=np.float64)
    point = np.full((height, width), -1, dtype=np.int32)
    for i in range(len(camera)):
        x, y, z = camera[i]
        if z < 0.25:
            continue
        u = focal * x / z + (width - 1) / 2
        v = focal * y / z + (height - 1) / 2
        if u < -radius or u >= width + radius or v < -radius or v >= height + radius:
            continue
        col, row = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
        for dy in range(-radius, radius + 1):
            yy = row + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = col + dx
                if xx < 0 or xx >= width or dx * dx + dy * dy > radius * radius:
                    continue
                if z < depth[yy, xx]:
                    depth[yy, xx] = z
                    point[yy, xx] = i
    return point


def front_image(frame: SourceFrame, title: str, frame_count: int) -> np.ndarray:
    slots = frame.real_slots
    xyz = frame.xyzi[slots, :3]
    focal = WIDTH / (2 * np.tan(np.deg2rad(HFOV_DEG) / 2))
    index = visible_points(camera_coordinates(xyz), WIDTH, HEIGHT, focal, POINT_RADIUS)
    target = frame.labels.anomaly_target[slots]
    rgb = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[:] = (14, 18, 24)
    valid = index >= 0
    rgb[valid] = COLORS[target[index[valid]] + 1]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if not hasattr(_THREAD_FONTS, "pair"):
        _THREAD_FONTS.pair = tuple(
            ImageFont.truetype(str(FONT_PATH), size) for size in (22, 19)
        )
    font, small = _THREAD_FONTS.pair
    draw.rectangle((0, 0, WIDTH, 42), fill=(14, 18, 24))
    draw.text((18, 8), title, font=font, fill=(240, 240, 240))
    status = f"Frame {frame.frame_id:03d} / {frame_count - 1:03d}"
    draw.text((WIDTH - 205, 9), status, font=small, fill=(220, 224, 232))
    draw.rectangle((0, HEIGHT - 67, WIDTH, HEIGHT), fill=(14, 18, 24))
    for x, text, color in (
        (18, "Anomaly", COLORS[2]),
        (172, "Normal", COLORS[1]),
        (318, "Ignore", COLORS[0]),
    ):
        draw.rectangle(
            (x, HEIGHT - 52, x + 13, HEIGHT - 39), fill=tuple(map(int, color))
        )
        draw.text((x + 23, HEIGHT - 57), text, font=small, fill=(225, 225, 225))
    draw.text(
        (18, HEIGHT - 29),
        "Virtual front view | Current scan only | Playback: 10 frames/s",
        font=small,
        fill=(180, 190, 205),
    )
    draw.text(
        (WIDTH - 330, HEIGHT - 56),
        f"Anomaly returns (360 deg): {int((target == 1).sum())}",
        font=small,
        fill=(225, 225, 225),
    )
    return np.asarray(image)


def ply_records(frame: SourceFrame, anchor_from_world: np.ndarray) -> np.ndarray:
    """Store each return once, aligned by inv(T_world_anchor) @ T_world_scan."""
    slots = frame.real_slots
    transform = anchor_from_world @ frame.lidar_pose
    xyz = frame.xyzi[slots, :3].astype(np.float64) @ transform[:3, :3].T
    xyz += transform[:3, 3]
    records = np.empty(len(slots), dtype=VERTEX_DTYPE)
    for i, name in enumerate(("x", "y", "z")):
        records[name] = xyz[:, i]
    colors = COLORS[frame.labels.anomaly_target[slots] + 1]
    for i, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, i]
    records["label"] = frame.labels.packed[slots]
    records["frame"] = frame.frame_id
    records["slot"] = slots
    if not np.isfinite(xyz).all() or frame.frame_id > np.iinfo(np.uint16).max:
        raise ValueError("PLY coordinates or frame identity cannot be represented")
    return records


@dataclass
class World:
    name: str
    title: str
    source: STUSequence
    segment: FrozenSyntheticSegment | None = None

    @property
    def frames(self):
        return self.source.frame_ids if self.segment is None else self.segment.frame_ids

    def frame(self, frame_id):
        return (
            self.source.source_frame(frame_id)
            if self.segment is None
            else self.segment.frame(frame_id)
        )

    @property
    def point_upper_bound(self):
        if self.segment is not None:
            return sum(self.segment.metadata["visible_point_counts"])
        return sum(
            path.stat().st_size // 16
            for path in (self.source.sequence_dir / "velodyne").glob("*.bin")
        )


def selected_worlds(data_root, train_world, validation_world, real_sequence):
    protocol = load_protocol()
    worlds = []
    for domain, index in (("train", train_world), ("validation", validation_world)):
        dataset = FrozenWindowDataset(
            data_root, protocol, pool_name=domain, version="v2"
        )
        source = dataset.source_sequence
        segment, _ = dataset.segment_for_window(index * len(source.window_starts))
        name = f"{domain}{source.spec.sequence_id}_w{index:03d}"
        source._cache_frames = 1
        worlds.append(
            World(
                name,
                f"V2 {domain} | {source.spec.sequence_id} | World {index:03d}",
                source,
                segment,
            )
        )
    source = STUSequence.open(
        data_root,
        protocol=protocol,
        partition="val",
        sequence_id=real_sequence,
        label_mode=LabelMode.REQUIRED,
    )
    source._cache_frames = 1
    worlds.append(
        World(
            f"real{real_sequence}",
            f"Real validation | Sequence {real_sequence}",
            source,
        )
    )
    return worlds


def export_world(world, output, stop):
    started = time.monotonic()
    ply = output / "ply" / f"{world.name}.ply"
    video = output / "video" / f"{world.name}.mp4"
    temporary_ply, temporary_video = (
        ply.with_suffix(".ply.tmp"),
        video.with_suffix(".mp4.tmp"),
    )
    identity = (
        world.segment.metadata["world_identity"]
        if world.segment is not None
        else f"STU/val/{world.source.spec.sequence_id}"
    )
    prefix = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment source {world.name}\ncomment world {identity}\n"
        f"comment frames {world.frames[0]} through {world.frames[-1]} inclusive\n"
        "comment coordinates first_frame_lidar_meters x_forward y_left z_up\n"
        "comment every_return_once no_downsampling no_distance_or_label_filter\n"
        "comment truth normal=160,160,160 anomaly=255,0,0 ignore=0,128,255\n"
        "comment label uint32_packed_semantic_low16_instance_high16\n"
        "comment frame source_frame_id slot original_file_slot\nelement vertex "
    ).encode("ascii")
    suffix = (
        "\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uint label\nproperty ushort frame\nproperty uint slot\nend_header\n"
    ).encode("ascii")
    anchor = np.linalg.inv(world.source.lidar_pose(world.frames[0]))
    counts = np.zeros(3, dtype=np.int64)
    frame_records, written, encoded = [], 0, 0
    try:
        with (
            temporary_ply.open("xb") as stream,
            av.open(str(temporary_video), mode="w", format="mp4") as container,
        ):
            stream.write(prefix + b"00000000000000000000" + suffix)
            container.metadata["title"] = world.title
            container.metadata["comment"] = (
                "Ground truth point cloud; virtual camera, not calibrated RGB. "
                "10 fps is playback rate; source timestamps are unavailable."
            )
            encoder = container.add_stream("libx264", rate=FPS)
            encoder.width, encoder.height, encoder.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            encoder.codec_context.thread_count = 3
            encoder.options = {
                "crf": "18",
                "preset": "medium",
                "maxrate": "5000k",
                "bufsize": "10000k",
            }
            for frame_id in world.frames:
                if stop.is_set():
                    raise InterruptedError(
                        "visualization stopped to preserve resource limits"
                    )
                frame = world.frame(frame_id)
                records = ply_records(frame, anchor)
                records.tofile(stream)
                written += len(records)
                target = frame.labels.anomaly_target[frame.real_slots]
                current_counts = np.bincount(target + 1, minlength=3)
                counts += current_counts
                frame_records.append(
                    {
                        "frame": frame_id,
                        "points": len(records),
                        "ignore_normal_anomaly": current_counts.tolist(),
                    }
                )
                picture = av.VideoFrame.from_ndarray(
                    front_image(frame, world.title, len(world.frames)), format="rgb24"
                )
                picture.pts = encoded
                for packet in encoder.encode(picture):
                    container.mux(packet)
                encoded += 1
                if encoded % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "world": world.name,
                                "frames": encoded,
                                "total_frames": len(world.frames),
                                "points": written,
                            }
                        ),
                        flush=True,
                    )
            for packet in encoder.encode():
                container.mux(packet)
            stream.seek(len(prefix))
            stream.write(f"{written:020d}".encode("ascii"))
        expected_bytes = (
            len(prefix) + 20 + len(suffix) + written * VERTEX_DTYPE.itemsize
        )
        if temporary_ply.stat().st_size != expected_bytes or encoded != len(
            world.frames
        ):
            raise AssertionError("incomplete PLY or video")
        if world.segment is not None and written != world.point_upper_bound:
            raise AssertionError("exported points differ from the frozen world")
        # Decode every frame, including delayed encoder packets and the final scan.
        with av.open(str(temporary_video)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
            if decoded != encoded:
                raise AssertionError(
                    "video frame count differs from source observations"
                )
        temporary_ply.replace(ply)
        temporary_video.replace(video)
    except BaseException:
        stop.set()
        temporary_ply.unlink(missing_ok=True)
        temporary_video.unlink(missing_ok=True)
        raise
    return {
        "name": world.name,
        "world_identity": identity,
        "frames": frame_records,
        "point_count": written,
        "ignore_normal_anomaly": counts.tolist(),
        "ply": str(ply),
        "ply_bytes": ply.stat().st_size,
        "video": str(video),
        "video_bytes": video.stat().st_size,
        "decoded_frames": decoded,
        "duration_seconds": encoded / FPS,
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/visuals"))
    parser.add_argument("--train-world", type=int, default=1)
    parser.add_argument("--validation-world", type=int, default=2)
    parser.add_argument("--real-sequence", type=int, default=125)
    parser.add_argument("--budget-bytes", type=int, default=5_000_000_000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"output must be empty: {args.output}")
    if ImageFont.truetype(str(FONT_PATH), 22).getname()[0] != "Times New Roman":
        raise RuntimeError("Times New Roman font is unavailable")
    worlds = selected_worlds(
        args.data_root, args.train_world, args.validation_world, args.real_sequence
    )
    # PLY is bounded by return counts; 5 Mbit/s VBV plus mux/buffer allowance bounds video.
    projected_bytes = sum(
        world.point_upper_bound * VERTEX_DTYPE.itemsize
        + len(world.frames) / FPS * 5_000_000 / 8
        + 5_000_000
        for world in worlds
    )
    disk = host_disk()
    if projected_bytes > min(
        args.budget_bytes - 1_000_000, disk["SizeRemaining"] - disk["reserve_bytes"]
    ):
        raise OSError(
            f"full exports exceed available budget: {projected_bytes:,.0f} bytes"
        )
    print(
        json.dumps({"projected_peak_bytes": projected_bytes, "host_disk": disk}),
        flush=True,
    )
    for kind in ("ply", "video"):
        (args.output / kind).mkdir(parents=True, exist_ok=True)
    # Compile once before the three independent worlds use the same CPU rasterizer.
    visible_points(
        np.zeros((0, 3), dtype=np.float64), WIDTH, HEIGHT, 640.0, POINT_RADIUS
    )
    stop = threading.Event()
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {
            executor.submit(export_world, world, args.output, stop) for world in worlds
        }
        try:
            while pending:
                done, pending = wait(pending, timeout=25)
                results.extend(future.result() for future in done)
                volume = host_disk()
                size = sum(
                    p.stat().st_size for p in args.output.rglob("*") if p.is_file()
                )
                if (
                    size > args.budget_bytes - 10_000_000
                    or volume["SizeRemaining"] < volume["reserve_bytes"] + 500_000_000
                ):
                    raise OSError("visualization is approaching its storage limit")
                print(
                    json.dumps(
                        {
                            "output_bytes": size,
                            "host_free_bytes": volume["SizeRemaining"],
                            "remaining_worlds": len(pending),
                        }
                    ),
                    flush=True,
                )
        except BaseException:
            stop.set()
            raise
    report = {
        "budget_bytes": args.budget_bytes,
        "media_bytes": size,
        "host_disk": volume,
        "resolution": [WIDTH, HEIGHT],
        "playback_fps": FPS,
        "time_basis": "frame order; no source timestamps available",
        "camera_center_in_lidar_m": CAMERA_CENTER.tolist(),
        "camera_pitch_down_deg": PITCH_DEG,
        "camera_horizontal_fov_deg": HFOV_DEG,
        "point_radius_pixels": POINT_RADIUS,
        "video_observations": "current scan only; label-blind depth buffer",
        "font": {"path": str(FONT_PATH), "family": "Times New Roman"},
        "worlds": sorted(results, key=lambda item: item["name"]),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "summary": str(args.output / "summary.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
