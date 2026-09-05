from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    DataProtocolError,
    FrozenSyntheticSegment,
    FrozenWindowDataset,
    PredictionBatch,
    WindowPartition,
    _atomic_json,
    _prediction_content_hash,
    _stable_npz,
    build_pool_manifest,
    generation_identity,
    load_pool_manifest,
    save_sparse_segment,
)
from src.protocol import (
    AJAEProtocol,
    FrameSpan,
    ProtocolError,
    SCHEMA_VERSION,
    SequenceSpec,
    load_protocol,
)
from src.render import (
    MaterialSpec,
    ObjectSpec,
    RayGrid,
    RenderError,
    SensorCalibration,
    ShapeSpec,
    WorldGenerationReport,
    WorldSpec,
    render_segment_world,
    world_content_identity,
)
from src.scene import PointLabels, STUSequence, assemble_window, make_source_frame
from src.qualify import QualificationError, _write_window_ply, qualify_data


ROOT = Path(__file__).resolve().parents[1]


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def _source(
    frame_id: int,
    *,
    sequence_id: int = 206,
    pose_seed: int | None = None,
    semantics: tuple[int, ...] = (40, 0, 48),
) -> object:
    xyzi = np.asarray(
        (
            (10.0 + frame_id, 0.25, 0.5, 0.2),
            (11.0 + frame_id, -0.5, 0.25, 0.4),
            (12.0 + frame_id, 1.0, -0.25, 0.6),
        ),
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float64)
    if pose_seed is not None:
        pose[:3, :3] = _rotation(pose_seed)
    pose[:3, 3] = (0.3 * frame_id, -0.1 * frame_id, 0.05 * frame_id)
    semantic = np.asarray(semantics, dtype=np.uint16)
    packed = semantic.astype(np.uint32)
    target = np.where(semantic == 0, 255, 8).astype(np.uint8)
    return make_source_frame(
        frame_id,
        xyzi,
        pose,
        PointLabels(packed, semantic, np.zeros(3, dtype=np.uint16), target),
        partition="train",
        sequence_id=sequence_id,
    )


def _window(order: tuple[int, ...] = (0, 1, 2, 3, 4)) -> object:
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 5))
    by_id = {frame: _source(frame, pose_seed=frame + 1) for frame in range(5)}
    return assemble_window(
        spec,
        0,
        tuple(range(5)),
        tuple(by_id[frame] for frame in order),
        observation_sequence_id="synthetic/train/000",
    )


def _single_ray_sources(sequence_id: int = 206) -> tuple[object, ...]:
    packed = np.asarray((40,), dtype=np.uint32)
    labels = PointLabels(
        packed,
        packed.astype(np.uint16),
        np.zeros(1, dtype=np.uint16),
        np.asarray((8,), dtype=np.uint8),
    )
    return tuple(
        make_source_frame(
            frame,
            np.asarray(((5.0, 0.0, 0.0, 0.2),), dtype=np.float32),
            np.eye(4, dtype=np.float64),
            labels,
            partition="train",
            sequence_id=sequence_id,
        )
        for frame in range(5)
    )


def _rendered_fixture(sequence_id: int = 206) -> tuple[object, tuple[object, ...]]:
    sources = _single_ray_sources(sequence_id)
    shape = ShapeSpec(
        ((0.5, 0.5, 0.5),),
        ((0.0, 0.0, 0.0),),
        ((1.0, 1.0),),
        (0.0,),
        ("union",),
    )
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    world = WorldSpec(
        34100000,
        sequence_id,
        (
            ObjectSpec(
                1,
                "anomaly-proxy",
                shape,
                MaterialSpec(0.5, 0.1),
                (3.0, 0.0, 0.0),
                rotation,
            ),
        ),
    )
    report = WorldGenerationReport(
        world.seed,
        sequence_id,
        "anomaly_only",
        0,
        1,
        world.seed,
    )
    grid = RayGrid(
        np.asarray(((1.0, 0.0, 0.0),)),
        np.asarray((0.0,)),
        np.asarray((0.0,)),
        beam_count=1,
    )
    segment = render_segment_world(
        world,
        report,
        sources,
        grid,
        SensorCalibration.constant(0.4),
        renderer_identity="a" * 64,
    )
    return segment, sources


def test_schema34_freezes_data_roles_and_counts() -> None:
    protocol = load_protocol()
    assert protocol.schema_version == SCHEMA_VERSION == 34
    assert protocol.status["state"] == "frozen"
    assert protocol.status["data_pool_frozen"] is True
    assert protocol.status["training_allowed"] is True
    assert protocol.status["validation_tuning_allowed"] is True
    assert protocol.training_sequence.span == FrameSpan(0, 449)
    assert protocol.validation_sequence.span == FrameSpan(0, 682)
    assert len(protocol.validation_sequence.legal_window_starts()) == 678
    assert protocol.training_pool.world_count == 128
    assert protocol.training_pool.total_window_count == 3080
    assert protocol.validation_pool.world_count == 92
    assert protocol.validation_pool.total_window_count == 2360
    assert protocol.status["real_anomaly_access_allowed"] is False
    for key in (
        "train_pool_manifest",
        "validation_pool_manifest",
        "qualification",
    ):
        record = protocol.artifacts[key]
        path = ROOT / str(record["file"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_segments_cover_sources_without_overlap_and_windows_do_not_cross() -> None:
    protocol = load_protocol()
    for pool, stop in (
        (protocol.training_pool, 449),
        (protocol.validation_pool, 682),
    ):
        covered = tuple(
            frame
            for segment in pool.segments
            for frame in range(segment.start, segment.stop)
        )
        assert covered == tuple(range(stop))
        for index, segment in enumerate(pool.segments):
            for start in pool.window_starts(index):
                assert segment.start <= start
                assert start + 4 < segment.stop


def test_protocol_rejects_more_than_one_anomaly_proxy_per_segment() -> None:
    document = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    document["synthetic_pools"]["anomaly_objects_per_segment"] = 2
    with pytest.raises(ProtocolError, match="world-before-window"):
        AJAEProtocol(document, path=ROOT / "protocol.json")


def test_formal_segments_each_contain_exactly_one_anomaly_proxy() -> None:
    protocol = load_protocol()
    assert protocol.synthetic_pools["anomaly_objects_per_segment"] == 1
    for pool in (protocol.training_pool, protocol.validation_pool):
        manifest = json.loads(protocol.pool_manifest_path(pool.name).read_text())
        assert len(manifest["segments"]) == pool.world_count
        for record in manifest["segments"]:
            with np.load(ROOT / record["file"], allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                assert len(metadata["world"]["objects"]) == 1
                assert metadata["world_generation_report"]["anomaly_count"] == 1
                assert np.all(data["changed_object_ids"] == 1)


def test_segment_renderer_rejects_multiple_anomaly_proxies() -> None:
    segment, sources = _rendered_fixture()
    first = segment.world.objects[0]
    multiple = replace(
        segment.world,
        objects=(
            first,
            replace(first, object_id=2, translation_world_m=(8.0, 0.0, 0.0)),
        ),
    )
    with pytest.raises(RenderError, match="exactly one anomaly proxy"):
        render_segment_world(
            multiple,
            replace(segment.report, anomaly_count=2),
            sources,
            None,
            None,
            renderer_identity="a" * 64,
        )


def test_formal_seeds_are_unique_and_predeclared() -> None:
    protocol = load_protocol()
    train = {
        protocol.training_pool.world_seed(sequence, segment)
        for sequence in range(8)
        for segment in range(16)
    }
    validation = {
        protocol.validation_pool.world_seed(sequence, segment)
        for sequence in range(4)
        for segment in range(23)
    }
    assert len(train) == 128
    assert len(validation) == 92
    assert train.isdisjoint(validation)


def test_world_content_identity_does_not_count_a_seed_change_as_diversity() -> None:
    segment, _ = _rendered_fixture()
    same_physics = WorldSpec(
        segment.world.seed + 1,
        segment.world.source_sequence_id,
        segment.world.objects,
        segment.world.tie_tolerance_m,
    )
    assert segment.world.identity != same_physics.identity
    assert world_content_identity(segment.world) == world_content_identity(same_physics)


def test_window_order_is_canonical_after_shuffled_input() -> None:
    window = _window((2, 4, 0, 3, 1))
    assert tuple(item.source.frame_id for item in window.frames) == (0, 1, 2, 3, 4)
    assert tuple(np.unique(window.points.source_frame)) == (0, 1, 2, 3, 4)


def test_current_frame_coordinates_are_bitwise_raw_xyz() -> None:
    window = _window((4, 2, 0, 3, 1))
    source = window.current_frame.source
    assert np.array_equal(
        window.points.coordinates[window.current_mask],
        source.xyzi[source.real_slots, :3],
    )
    assert np.array_equal(window.current_frame.source_to_current, np.eye(4))


def test_history_uses_current_from_world_times_world_from_source() -> None:
    window = _window()
    source = window.frames[1].source
    transform = (
        np.linalg.inv(window.current_frame.source.lidar_pose) @ source.lidar_pose
    )
    expected = (
        source.xyzi[source.real_slots, :3].astype(np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    ).astype(np.float32)
    mask = window.points.source_frame == source.frame_id
    np.testing.assert_allclose(
        window.points.coordinates[mask], expected, atol=1e-6, rtol=1e-5
    )


def test_point_identity_label_and_intensity_rows_remain_aligned() -> None:
    window = _window()
    identities = set()
    for index, (frame, slot, feature, packed) in enumerate(
        zip(
            window.points.source_frame,
            window.points.source_slot,
            window.points.features,
            window.labels.packed,
            strict=True,
        )
    ):
        identity = (window.observation_sequence_id, int(frame), int(slot))
        point_id = window.point_id(index)
        assert (
            point_id.observation_sequence_id,
            point_id.frame_id,
            point_id.source_slot,
        ) == identity
        assert identity not in identities
        identities.add(identity)
        source = window.frame_for_id(int(frame)).source
        assert feature[0] == source.xyzi[int(slot), 3]
        assert packed == source.labels.packed[int(slot)]
    assert len(identities) == window.points.count


def test_all_window_valid_labels_supervise_and_current_mask_only_selects_online() -> (
    None
):
    window = _window()
    expected_current = window.points.source_frame == window.current_frame_id
    assert np.array_equal(window.current_mask, expected_current)
    assert np.array_equal(window.supervision_mask, window.labels.semantic != 0)
    assert np.count_nonzero(window.supervision_mask & ~window.current_mask) > 0
    assert set(np.unique(window.labels.anomaly_target)) == {-1, 0}


def test_prediction_batch_requires_and_persists_all_point_scores(
    tmp_path: Path,
) -> None:
    window = _window()
    scores = np.arange(window.points.count, dtype=np.float32)
    batch = PredictionBatch.from_window(window, scores)
    assert np.array_equal(batch.anomaly_score, scores)
    assert np.array_equal(batch.online_mask, window.current_mask)
    first = tmp_path / "prediction_a.npz"
    second = tmp_path / "prediction_b.npz"
    first_record = batch.save(first, window=window)
    second_record = batch.save(second, window=window)
    assert first_record["file_sha256"] == second_record["file_sha256"]
    restored = PredictionBatch.load(
        first,
        window=window,
        expected_sha256=str(first_record["file_sha256"]),
    )
    assert restored.observation_sequence_id == window.observation_sequence_id
    assert restored.window_current_frame == window.current_frame_id
    assert np.array_equal(restored.source_frame, batch.source_frame)
    assert np.array_equal(restored.source_slot, batch.source_slot)
    assert np.array_equal(restored.anomaly_score, scores)
    with pytest.raises(Exception, match="every point"):
        PredictionBatch.from_window(window, scores[:-1])
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        batch.save(first, window=window)
    assert first.read_bytes() == before
    with pytest.raises(TypeError, match="window"):
        batch.save(first)
    with pytest.raises(TypeError, match="window"):
        PredictionBatch.load(first)


@pytest.mark.parametrize(
    "invalid",
    ("missing_history", "missing_current", "reordered", "wrong_sequence", "wrong_slot"),
)
def test_prediction_save_rejects_incomplete_or_mismatched_window(
    tmp_path: Path, invalid: str
) -> None:
    window = _window()
    indices = np.arange(window.points.count)
    if invalid == "missing_history":
        indices = indices[1:]
    elif invalid == "missing_current":
        indices = indices[:-1]
    elif invalid == "reordered":
        indices = indices[::-1]
    slots = window.points.source_slot[indices].copy()
    if invalid == "wrong_slot":
        slots[0] += 100
    batch = PredictionBatch(
        "synthetic/train/001"
        if invalid == "wrong_sequence"
        else window.observation_sequence_id,
        window.current_frame_id,
        window.points.source_frame[indices],
        slots,
        np.zeros(indices.size, dtype=np.float32),
    )
    path = tmp_path / "invalid.npz"
    with pytest.raises(DataProtocolError, match="every input window point"):
        batch.save(path, window=window)
    assert not path.exists()


def test_prediction_rejects_duplicate_points_and_nonfinite_scores() -> None:
    window = _window()
    indices = np.arange(window.points.count)
    indices[-1] = indices[-2]
    with pytest.raises(DataProtocolError, match="duplicated"):
        PredictionBatch(
            window.observation_sequence_id,
            window.current_frame_id,
            window.points.source_frame[indices],
            window.points.source_slot[indices],
            np.zeros(indices.size, dtype=np.float32),
        )
    for value in (np.nan, np.inf):
        scores = np.full(window.points.count, value, dtype=np.float32)
        with pytest.raises(DataProtocolError, match="finite"):
            PredictionBatch.from_window(window, scores)


def test_prediction_load_checks_actual_window_even_with_valid_file_hash(
    tmp_path: Path,
) -> None:
    window = _window()
    path = tmp_path / "prediction.npz"
    batch = PredictionBatch.from_window(
        window, np.zeros(window.points.count, dtype=np.float32)
    )
    record = batch.save(path, window=window)
    wrong_sequence = replace(window, observation_sequence_id="synthetic/train/001")
    later = assemble_window(
        SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 6)),
        1,
        (1, 2, 3, 4, 5),
        tuple(_source(t) for t in range(1, 6)),
        observation_sequence_id=window.observation_sequence_id,
    )
    for wrong_window in (wrong_sequence, later):
        with pytest.raises(DataProtocolError, match="every input window point"):
            PredictionBatch.load(
                path, window=wrong_window, expected_sha256=record["file_sha256"]
            )
    # A valid self-hash cannot make a truncated prediction complete.
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: payload[name][:-1]
            for name in ("source_frame", "source_slot", "anomaly_score")
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    metadata.pop("content_hash")
    metadata["point_count"] -= 1
    metadata["content_hash"] = _prediction_content_hash(metadata, arrays)
    shortened = tmp_path / "shortened.npz"
    _stable_npz(
        shortened, {**arrays, "metadata_json": np.asarray(json.dumps(metadata))}
    )
    with pytest.raises(DataProtocolError, match="every input window point"):
        PredictionBatch.load(shortened, window=window)


def test_rechecks_cannot_overwrite_frozen_or_existing_evidence(tmp_path: Path) -> None:
    protocol = load_protocol()
    frozen = ROOT / protocol.artifacts["qualification"]["file"]
    before = frozen.read_bytes()
    alias = tmp_path / "alias.json"
    alias.symlink_to(frozen)
    for path in (frozen, alias):
        with pytest.raises(QualificationError, match="cannot replace frozen"):
            qualify_data(tmp_path, protocol, output_path=path)
    assert frozen.read_bytes() == before
    report = tmp_path / "recheck.json"
    _atomic_json(report, {"original": True})
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        qualify_data(tmp_path, protocol, output_path=report)
    with pytest.raises(FileExistsError):
        _atomic_json(report, {"replacement": True})
    assert report.read_bytes() == before
    for pool in ("train", "validation"):
        with pytest.raises(DataProtocolError, match="read-only"):
            build_pool_manifest(protocol, pool)
    assert not list(tmp_path.glob("*.tmp"))


def _linked_frozen_protocol(tmp_path: Path) -> AJAEProtocol:
    """Use small isolated links; damage tests replace a link, never its target."""

    protocol = load_protocol()
    path = tmp_path / "protocol.json"
    path.write_bytes((ROOT / "protocol.json").read_bytes())
    paths = [protocol.artifacts["qualification"]["file"]]
    for pool in (protocol.training_pool, protocol.validation_pool):
        manifest_path = protocol.pool_manifest_path(pool.name)
        paths.append(manifest_path.relative_to(ROOT).as_posix())
        manifest = json.loads(manifest_path.read_text())
        paths.extend(record["file"] for record in manifest["segments"])
    for relative in paths:
        link = tmp_path / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(ROOT / relative)
    return load_protocol(path)


@pytest.mark.parametrize(
    "damaged",
    (
        "artifacts/data/qualification.json",
        "artifacts/data/train_manifest.json",
        "artifacts/data/validation_manifest.json",
        "artifacts/data/train/sequence_007/segment_15.npz",
        "artifacts/data/validation/sequence_003/segment_22.npz",
    ),
)
def test_training_constructor_verifies_all_frozen_files_before_loading_data(
    tmp_path: Path,
    damaged: str,
) -> None:
    protocol = _linked_frozen_protocol(tmp_path)
    path = tmp_path / damaged
    path.unlink()
    path.write_bytes(b"damaged fixture")
    with pytest.raises(DataProtocolError, match="bytes differ|segment file differs"):
        FrozenWindowDataset(tmp_path / "no_raw_data", protocol, pool_name="train")


def test_consumer_changes_do_not_redefine_frozen_generation_provenance() -> None:
    protocol = load_protocol()
    evidence = json.loads(
        (ROOT / protocol.artifacts["qualification"]["file"]).read_text()
    )
    for pool in (protocol.training_pool, protocol.validation_pool):
        manifest = load_pool_manifest(protocol, pool)
        assert manifest["generation_identity"] == generation_identity(
            protocol,
            pool,
            source_files_sha256=evidence["source_files_sha256"],
        )


def test_ply_preserves_all_coordinates_and_truth_colors(tmp_path: Path) -> None:
    window = _window()
    path = tmp_path / "window.ply"
    size = _write_window_ply(window, path)
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert f"element vertex {window.points.count}".encode() in header
    points = np.frombuffer(body, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    assert np.array_equal(points["xyz"], window.points.coordinates)
    assert np.all(points["rgb"][window.labels.anomaly_target == 0] == (160, 160, 160))
    assert np.all(points["rgb"][window.labels.anomaly_target == -1] == (0, 128, 255))
    assert size == len(raw) == _write_window_ply(window, path)
    segment, sources = _rendered_fixture()
    spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 5))
    anomaly_window = assemble_window(
        spec, 0, tuple(range(5)), tuple(f.source for f in segment.rendered_frames)
    )
    anomaly_path = tmp_path / "anomaly.ply"
    _write_window_ply(anomaly_window, anomaly_path)
    anomaly_body = anomaly_path.read_bytes().split(b"end_header\n", 1)[1]
    anomaly_points = np.frombuffer(
        anomaly_body, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)]
    )
    assert np.all(anomaly_points["rgb"] == (255, 0, 0))


def test_window_partition_maps_each_output_to_past_four_plus_current() -> None:
    sequence = object.__new__(STUSequence)
    sequence.window_starts = tuple(range(20))
    sequence.window = lambda start: tuple(range(start, start + 5))
    partition = WindowPartition(sequence, 8, 12)
    assert len(partition) == 5
    assert list(partition) == [
        (4, 5, 6, 7, 8),
        (5, 6, 7, 8, 9),
        (6, 7, 8, 9, 10),
        (7, 8, 9, 10, 11),
        (8, 9, 10, 11, 12),
    ]
    with pytest.raises(TypeError, match="WindowPartition"):
        iter(sequence).__next__()


def test_rendered_segment_reuses_each_rendered_frame_across_windows() -> None:
    segment, _ = _rendered_fixture()
    assert len(segment.rendered_frames) == 5
    assert len(segment.windows) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            segment.windows[0].rendered_frames,
            segment.rendered_frames,
            strict=True,
        )
    )
    assert all(
        frame.inserted_mask.tolist() == [True] for frame in segment.rendered_frames
    )


def test_sparse_segment_round_trip_preserves_points_labels_and_window(
    tmp_path: Path,
) -> None:
    segment, sources = _rendered_fixture()
    path = tmp_path / "segment.npz"
    record = save_sparse_segment(
        path,
        segment,
        sources,
        pool_name="train",
        synthetic_sequence_id="synthetic/train/000",
        synthetic_sequence_index=0,
        segment_index=0,
    )
    sequence = object.__new__(STUSequence)
    sequence.spec = SequenceSpec("train", 206, "fixture", True, FrameSpan(0, 5))
    by_id = {item.frame_id: item for item in sources}
    sequence.source_frame = lambda frame_id: by_id[frame_id]
    frozen = FrozenSyntheticSegment(path, sequence, str(record["file_sha256"]))
    for expected in segment.rendered_frames:
        actual = frozen.frame(expected.frame_id)
        assert np.array_equal(actual.xyzi, expected.source.xyzi)
        assert np.array_equal(actual.labels.packed, expected.packed_labels)
    window = frozen.window(0)
    assert window.observation_sequence_id == "synthetic/train/000"
    assert window.points.count == 5
    assert np.all(window.labels.anomaly_target == 1)
    assert np.array_equal(
        window.points.coordinates[window.current_mask],
        segment.rendered_frames[-1].source.xyzi[:, :3],
    )
