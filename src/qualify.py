#!/usr/bin/env python3
"""Run the lightweight schema-31 mechanical qualifications.

The default command uses only deterministic synthetic fixtures. It does not
open STU data, render a training bank, load the released STU network, train a
model, or turn historical schema-30 records into schema-31 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

try:
    from .model import (
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        GroupedKnnUpsample,
        GroupedRadiusKNN,
        GroupedVoxelPool,
        JointWindowPointTransformer,
        assigned_stu_evidence,
    )
    from .protocol import (
        DEFAULT_PROTOCOL_PATH,
        SCHEMA_VERSION,
        AJAEProtocol,
        ExperimentCondition,
        FrameSpan,
        SequenceSpec,
        load_development_worlds,
        load_protocol,
    )
    from .scene import (
        SceneDataError,
        SceneWindow,
        _grant_sealed_sequence_access,
        _require_sealed_sequence_access,
        assemble_window,
        make_source_frame,
    )
except ImportError:  # Direct script execution.
    from model import (
        MASK_DIM,
        NUM_NORMAL_CLASSES,
        GroupedKnnUpsample,
        GroupedRadiusKNN,
        GroupedVoxelPool,
        JointWindowPointTransformer,
        assigned_stu_evidence,
    )
    from protocol import (
        DEFAULT_PROTOCOL_PATH,
        SCHEMA_VERSION,
        AJAEProtocol,
        ExperimentCondition,
        FrameSpan,
        SequenceSpec,
        load_development_worlds,
        load_protocol,
    )
    from scene import (
        SceneDataError,
        SceneWindow,
        _grant_sealed_sequence_access,
        _require_sealed_sequence_access,
        assemble_window,
        make_source_frame,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "ajae"
DEFAULT_DEVELOPMENT_PATH = PROJECT_ROOT / "dev.json"
AUDIT_NAME = "Window Densification Audit"
AUDIT_FORMAT = "ajae-schema31-qualification-v1"


class QualificationError(ValueError):
    """Report a failed qualification or malformed retained record."""


@dataclass(frozen=True, slots=True)
class RetainedArtifact:
    """One historical record that keeps only its original scientific scope."""

    capability: str
    claim: str
    filename: str


RETAINED_ARTIFACTS = (
    RetainedArtifact(
        "renderer",
        "schema7_geometry_and_coverage",
        "e20a_v3_schema7_geometry_coverage.npz",
    ),
    RetainedArtifact(
        "renderer", "support_placement_and_collision", "e21_v4_support_pool.npz"
    ),
    RetainedArtifact(
        "renderer", "support_placement_and_collision", "e23_observed_collision.npz"
    ),
    RetainedArtifact(
        "renderer", "support_placement_and_collision", "e24_v2_pair_collision.npz"
    ),
    RetainedArtifact(
        "renderer",
        "ray_mapping_and_range_image_round_trip",
        "e11_v3_qualification.npz",
    ),
    RetainedArtifact(
        "renderer",
        "ray_mapping_and_range_image_round_trip",
        "e13_count_round_trip.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e29_return_sampling.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e30_normal_returns.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e31_proxy_returns.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e32_background_occlusion.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e33_foreground_occlusion.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e34_empty_rays.npz",
    ),
    RetainedArtifact(
        "renderer",
        "first_return_occlusion_intensity_and_empty_ray_rendering",
        "e35_intensity.npz",
    ),
    RetainedArtifact(
        "renderer",
        "shared_renderer_for_normal_control_and_proxy",
        "e36_v2_shared_path.npz",
    ),
    RetainedArtifact(
        "stu",
        "official_STU_weight_and_point_evidence_restoration",
        "e50_stu_features.npz",
    ),
    RetainedArtifact(
        "stu",
        "official_STU_weight_and_point_evidence_restoration",
        "e51_inverse_mapping.npz",
    ),
    RetainedArtifact(
        "stu",
        "official_STU_weight_and_point_evidence_restoration",
        "e53_query_assignment.npz",
    ),
    RetainedArtifact(
        "stu",
        "official_STU_weight_and_point_evidence_restoration",
        "e54_evidence_reliability.npz",
    ),
    RetainedArtifact(
        "point_identity", "point_identity", "e37_world_frame_consistency.npz"
    ),
    RetainedArtifact(
        "point_identity",
        "E61_raw_normal_safety_frame_ray_identity_not_old_scores",
        "e61_safety_identities.npz",
    ),
    RetainedArtifact(
        "official_evaluator",
        "official_evaluator_numerical_equivalence",
        "e62_evaluator_equivalence.npz",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(array: np.ndarray, name: str) -> Mapping[str, object]:
    if array.size != 1:
        raise QualificationError(f"{name} must contain one JSON value")
    try:
        value = json.loads(str(array.reshape(-1)[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise QualificationError(f"{name} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise QualificationError(f"{name} must decode to an object")
    return value


def _artifact_pass_record(path: Path) -> tuple[bool, str | None]:
    """Read only the stored decision field; no large scientific array is loaded."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in ("metadata_json", "summary_json"):
                if name in archive.files:
                    record = _json_value(np.asarray(archive[name]), name)
                    return record.get("passed") is True, str(record.get("experiment"))
            if "passed" in archive.files:
                value = np.asarray(archive["passed"])
                return value.size == 1 and bool(value.reshape(-1)[0]), None
    except (OSError, ValueError) as error:
        raise QualificationError(f"cannot read retained record {path}") from error
    raise QualificationError(f"retained record {path} has no decision field")


def retained_evidence_audit(
    protocol: AJAEProtocol,
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    *,
    hash_files: bool = False,
) -> dict[str, object]:
    """Bind valid historical records without widening their schema-30 claims."""

    root = Path(runs_root).expanduser().resolve(strict=True)
    historical = protocol.historical_evidence
    if not isinstance(historical, Mapping):
        raise QualificationError("protocol historical_evidence must be a mapping")
    _require(
        int(historical["evidence_source_schema"]) == 30,
        "retained evidence must declare schema 30 as its source",
    )
    declared = frozenset(
        str(value) for value in historical["continues_with_original_scope"]
    )
    records: list[dict[str, object]] = []
    for artifact in RETAINED_ARTIFACTS:
        _require(
            artifact.claim in declared,
            f"protocol does not retain the original scope {artifact.claim}",
        )
        path = root / artifact.filename
        _require(path.is_file(), f"missing retained record {path}")
        passed, experiment = _artifact_pass_record(path)
        _require(passed, f"retained record {path} does not contain a pass decision")
        record: dict[str, object] = {
            "capability": artifact.capability,
            "claim": artifact.claim,
            "file": artifact.filename,
            "bytes": path.stat().st_size,
            "recorded_experiment": experiment,
            "recorded_pass": True,
            "interpretation": "schema30_original_scope_only",
        }
        if hash_files:
            record["sha256"] = _sha256(path)
        records.append(record)
    return {
        "source_schema": 30,
        "record_count": len(records),
        "stored_decision_fields_verified": True,
        "claim_scope_widened": False,
        "records": records,
    }


def _yaw_pose(angle: float, translation: Sequence[float]) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def _scene_fixture() -> tuple[SceneWindow, SceneWindow]:
    spec = SequenceSpec(
        "train", 206, "schema31 qualification fixture", False, FrameSpan(0, 5)
    )
    frames = []
    for frame_id in range(5):
        xyzi = np.asarray(
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.00 + 0.03 * frame_id, 0.10, 0.25, 0.20),
                (1.25, -0.15 + 0.02 * frame_id, 0.40, 0.35),
                (0.0, 0.0, 0.0, 0.0),
                (1.70, 0.20, 0.55 + 0.01 * frame_id, 0.55),
                (2.10, -0.25, 0.65, 0.70),
            ),
            dtype=np.float32,
        )
        pose = _yaw_pose(
            0.035 * (frame_id - 2),
            (0.4 * frame_id, 0.07 * frame_id**2, 0.01 * frame_id),
        )
        frames.append(
            make_source_frame(
                frame_id,
                xyzi,
                pose,
                partition="train",
                sequence_id=206,
            )
        )
    frame_ids = (0, 1, 2, 3, 4)
    first = assemble_window(spec, 0, frame_ids, tuple(frames))
    permutation = (3, 0, 4, 1, 2)
    second = assemble_window(
        spec, 0, frame_ids, tuple(frames[index] for index in permutation)
    )
    return first, second


def _point_rows(window: SceneWindow) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    return {
        (int(frame), int(slot)): (window.points.coordinates[index], int(ray))
        for index, (frame, slot, ray) in enumerate(
            zip(
                window.points.source_frame,
                window.points.source_slot,
                window.points.source_ray,
                strict=True,
            )
        )
    }


def symmetric_coordinate_audit() -> dict[str, object]:
    """Check order, global-rigid, repeatability, and proper-rotation semantics."""

    first, second = _scene_fixture()
    _require(
        np.array_equal(
            first.reference_pose.world_from_window,
            second.reference_pose.world_from_window,
        ),
        "symmetric reference pose changed under scan permutation",
    )
    left = _point_rows(first)
    right = _point_rows(second)
    _require(
        left.keys() == right.keys(), "point identities changed under scan permutation"
    )
    maximum_error = max(
        float(np.max(np.abs(left[key][0] - right[key][0]))) for key in left
    )
    _require(maximum_error == 0.0, "symmetric point coordinates changed by scan order")
    _require(
        all(left[key][1] == right[key][1] for key in left),
        "canonical rays changed under scan permutation",
    )
    source_frames = tuple(item.source for item in first.frames)
    repeated = assemble_window(
        first.spec, first.window_start, first.frame_ids, source_frames
    )
    repeated_rows = _point_rows(repeated)
    _require(
        np.array_equal(
            first.reference_pose.world_from_window,
            repeated.reference_pose.world_from_window,
        )
        and all(
            np.array_equal(left[key][0], repeated_rows[key][0])
            for key in left
        ),
        "repeated symmetric-window construction is not deterministic",
    )
    global_transform = _yaw_pose(0.41, (7.0, -3.0, 1.25))
    transformed_sources = tuple(
        make_source_frame(
            source.frame_id,
            source.xyzi,
            global_transform @ source.lidar_pose,
            partition=source.partition,
            sequence_id=source.sequence_id,
        )
        for source in source_frames
    )
    transformed = assemble_window(
        first.spec, first.window_start, first.frame_ids, transformed_sources
    )
    transformed_rows = _point_rows(transformed)
    rigid_coordinate_error = max(
        float(np.max(np.abs(left[key][0] - transformed_rows[key][0])))
        for key in left
    )
    rigid_pose_error = float(
        np.max(
            np.abs(
                transformed.reference_pose.world_from_window
                - global_transform @ first.reference_pose.world_from_window
            )
        )
    )
    _require(
        rigid_coordinate_error <= 1.0e-6 and rigid_pose_error <= 1.0e-10,
        "symmetric coordinates changed under one global rigid transform",
    )
    transform = first.reference_pose.world_from_window
    inverse_error = float(
        np.max(np.abs(transform @ first.reference_pose.window_from_world - np.eye(4)))
    )
    orthogonality_error = float(
        np.max(
            np.abs(
                first.reference_pose.rotation.T @ first.reference_pose.rotation
                - np.eye(3)
            )
        )
    )
    return {
        "passed": True,
        "points": len(left),
        "maximum_permutation_coordinate_error": maximum_error,
        "maximum_global_rigid_coordinate_error": rigid_coordinate_error,
        "global_rigid_reference_pose_error": rigid_pose_error,
        "repeated_construction_exact": True,
        "inverse_error": inverse_error,
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant": float(np.linalg.det(first.reference_pose.rotation)),
    }


def full_point_recovery_audit() -> dict[str, object]:
    """Recover every visible return to its original frame and file slot."""

    window, _ = _scene_fixture()
    tokens = (
        window.points.source_frame.astype(np.int64) + 1
    ) * 1000 + window.points.source_slot.astype(np.int64)
    recovered = 0
    for frame_id in window.frame_ids:
        frame = window.frame_for_id(frame_id).source
        restored = window.restore_source_frame(frame_id, tokens)
        expected = np.zeros(frame.slot_count, dtype=np.int64)
        expected[frame.real_slots] = (frame_id + 1) * 1000 + frame.real_slots
        _require(
            np.array_equal(restored, expected),
            f"frame {frame_id} was not restored losslessly",
        )
        recovered += int(np.count_nonzero(restored))
        for index in np.flatnonzero(window.points.source_frame == frame_id):
            identity = window.points.point_id(int(index))
            _require(
                identity.frame_id == frame_id
                and identity.ray.beam_id * 1024 + identity.ray.azimuth_column
                == int(window.points.source_ray[index]),
                "PointId does not reproduce frame-ray identity",
            )
    _require(
        recovered == window.points.count, "visible return count changed on recovery"
    )
    return {
        "passed": True,
        "input_visible_returns": window.points.count,
        "recovered_visible_returns": recovered,
        "source_frames": len(window.frame_ids),
    }


def grouped_voxel_audits() -> tuple[dict[str, object], ...]:
    """Expose the exact grouping and density inputs used by production pooling."""

    torch.manual_seed(3102)
    pool = GroupedVoxelPool(hidden_dim=16, voxel_size=1.0)
    features = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 31.0
    coordinates = torch.tensor(((0.10, 0.10, 0.10), (0.12, 0.11, 0.10)))
    groups = torch.tensor((0, 1), dtype=torch.long)
    joint = pool(features, coordinates, groups, grouping_mode="joint")
    isolated = pool(features, coordinates, groups, grouping_mode="per_scan")
    _require(
        joint.population.tolist() == [2],
        "joint grouping did not merge coincident cross-scan points",
    )
    _require(
        isolated.population.tolist() == [1, 1]
        and isolated.scan_group.tolist() == [0, 1],
        "scan-isolated grouping merged different scans",
    )

    projection_inputs: list[Tensor] = []

    def capture_projection(_module: object, arguments: tuple[Tensor, ...]) -> None:
        projection_inputs.append(arguments[0].detach().clone())

    handle = pool.projection[0].register_forward_pre_hook(capture_projection)
    try:
        pool(features[:1], coordinates[:1], groups[:1], grouping_mode="joint")
        pool(features, coordinates, groups, grouping_mode="joint")
    finally:
        handle.remove()
    _require(len(projection_inputs) == 2, "density projection was not observed twice")
    single_density = float(projection_inputs[0][0, -1])
    joint_density = float(projection_inputs[1][0, -1])
    _require(
        np.isclose(single_density, np.log1p(1.0), atol=1.0e-7)
        and np.isclose(joint_density, np.log1p(2.0), atol=1.0e-7)
        and joint_density > single_density,
        "pooling did not expose log1p population growth",
    )
    return (
        {
            "passed": True,
            "joint_voxels": int(joint.population.numel()),
            "joint_population": joint.population.tolist(),
        },
        {
            "passed": True,
            "isolated_voxels": int(isolated.population.numel()),
            "isolated_population": isolated.population.tolist(),
        },
        {
            "passed": True,
            "single_scan_density": single_density,
            "joint_density": joint_density,
        },
    )


def no_order_feature_audit(protocol: AJAEProtocol) -> dict[str, object]:
    """Bind the model to point content, space, and a grouping boundary only."""

    parameters = tuple(
        inspect.signature(JointWindowPointTransformer.forward).parameters
    )
    expected = (
        "self",
        "coordinates",
        "stu_features",
        "normal_evidence",
        "reliability_assign",
        "reliability_noobj",
        "intensity",
        "scan_group",
        "grouping_mode",
    )
    _require(
        parameters == expected, "model forward interface contains an undeclared input"
    )
    model = protocol.model
    _require(
        tuple(model["input_features"])
        == (
            "stu_point_feature_128d",
            "normal_evidence_19d",
            "assignment_reliability",
            "no_object_reliability",
            "intensity",
        ),
        "protocol model inputs differ from the five content features",
    )
    fixture = _tiny_model(3101)
    allowed_prefixes = (
        "input_projection.",
        "pyramid.",
        "decoder_fusions.",
        "high_resolution_fusion.",
        "anomaly_head.",
    )
    state_names = tuple(fixture.state_dict())
    _require(
        all(name.startswith(allowed_prefixes) for name in state_names),
        "model state contains a component outside the joint spatial hierarchy",
    )
    return {
        "passed": True,
        "forward_parameters": list(parameters[1:]),
        "state_tensor_count": len(state_names),
    }


def _model_inputs(seed: int = 3103) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    groups = torch.arange(5, dtype=torch.long).repeat_interleave(4)
    local = torch.tensor(
        (
            (0.00, 0.00, 0.00),
            (0.11, 0.03, 0.02),
            (0.22, -0.02, 0.06),
            (0.34, 0.04, 0.09),
        ),
        dtype=torch.float32,
    )
    offsets = torch.stack(
        (
            groups.to(torch.float32) * 0.015,
            groups.to(torch.float32) * -0.012,
            groups.to(torch.float32) * 0.008,
        ),
        dim=1,
    )
    coordinates = local.repeat(5, 1) + offsets
    count = coordinates.shape[0]
    return (
        coordinates,
        torch.randn((count, MASK_DIM), generator=generator),
        torch.randn((count, NUM_NORMAL_CLASSES), generator=generator),
        torch.rand(count, generator=generator),
        torch.rand(count, generator=generator),
        torch.rand(count, generator=generator),
        groups,
    )


def _tiny_model(seed: int = 3104) -> JointWindowPointTransformer:
    torch.manual_seed(seed)
    return JointWindowPointTransformer(
        hidden_dim=16,
        voxel_sizes=(0.20, 0.45, 0.90),
        neighbor_radii=(0.32, 0.64, 1.28, 2.56),
        neighbor_k=(6, 6, 6, 6),
        heads=4,
        attention_chunk_size=64,
    )


def scan_permutation_audit() -> dict[str, object]:
    """Check B2/B3 equivariance when complete scan blocks change row order."""

    model = _tiny_model().eval()
    inputs = _model_inputs()
    scan_order = (3, 0, 4, 1, 2)
    permutation = torch.cat(
        tuple(
            torch.nonzero(inputs[-1] == group, as_tuple=False).flatten()
            for group in scan_order
        )
    )
    relabel = torch.empty(5, dtype=torch.long)
    for new_group, old_group in enumerate(scan_order):
        relabel[old_group] = new_group
    permuted_inputs = tuple(value[permutation] for value in inputs[:-1])
    permuted_groups = relabel[inputs[-1][permutation]]
    errors: dict[str, float] = {}
    with torch.no_grad():
        for grouping_mode in ("per_scan", "joint"):
            expected = model(*inputs, grouping_mode=grouping_mode)
            observed = model(
                *permuted_inputs,
                permuted_groups,
                grouping_mode=grouping_mode,
            )
            errors[grouping_mode] = float(
                torch.max(torch.abs(observed - expected[permutation]))
            )
    error = max(errors.values())
    _require(
        error <= 2.0e-6,
        "B2/B3 model output is not scan-permutation equivariant",
    )
    return {
        "passed": True,
        "points": int(expected.numel()),
        "maximum_logit_error": error,
        "maximum_logit_error_by_grouping_mode": errors,
    }


def cross_scan_operator_audit() -> dict[str, object]:
    """Prove that production neighborhoods and interpolation cross scan groups."""

    coordinates = torch.tensor(
        ((0.00, 0.0, 0.0), (0.08, 0.0, 0.0), (2.0, 0.0, 0.0)),
        dtype=torch.float32,
    )
    groups = torch.tensor((0, 1, 0), dtype=torch.long)
    neighborhood = GroupedRadiusKNN(radius=0.20, k=3, workers=1)
    joint_neighbor, joint_valid, joint_count = neighborhood(
        coordinates, groups, grouping_mode="joint"
    )
    isolated_neighbor, isolated_valid, isolated_count = neighborhood(
        coordinates, groups, grouping_mode="per_scan"
    )
    joint_rows = joint_neighbor[0, joint_valid[0]].tolist()
    isolated_rows = isolated_neighbor[0, isolated_valid[0]].tolist()
    _require(
        1 in joint_rows and 1 not in isolated_rows,
        "joint neighborhood did not use the nearby point from another scan",
    )
    _require(
        int(joint_count[0]) == 2 and int(isolated_count[0]) == 1,
        "uncapped neighborhood counts do not reflect scan grouping",
    )

    upsample = GroupedKnnUpsample(k=1, workers=1)
    source_features = torch.tensor(((2.0,), (9.0,)), dtype=torch.float32)
    source_coordinates = torch.tensor(
        ((3.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=torch.float32
    )
    source_groups = torch.tensor((0, 1), dtype=torch.long)
    target_coordinates = torch.tensor(((0.01, 0.0, 0.0),), dtype=torch.float32)
    target_groups = torch.tensor((0,), dtype=torch.long)
    joint_value = upsample(
        source_features,
        source_coordinates,
        source_groups,
        target_coordinates,
        target_groups,
        grouping_mode="joint",
    )
    isolated_value = upsample(
        source_features,
        source_coordinates,
        source_groups,
        target_coordinates,
        target_groups,
        grouping_mode="per_scan",
    )
    _require(
        torch.equal(joint_value, torch.tensor(((9.0,),)))
        and torch.equal(isolated_value, torch.tensor(((2.0,),))),
        "joint interpolation did not select the nearer cross-scan source",
    )
    return {
        "passed": True,
        "joint_neighbor_rows_for_point_0": joint_rows,
        "isolated_neighbor_rows_for_point_0": isolated_rows,
        "joint_upsample": float(joint_value[0, 0]),
        "isolated_upsample": float(isolated_value[0, 0]),
    }


def joint_gradient_audit() -> dict[str, object]:
    """Backpropagate every point loss through all four joint spatial levels."""

    model = _tiny_model(3105).train()
    raw_inputs = _model_inputs(3106)
    inputs = tuple(value.detach().clone() for value in raw_inputs[:-1])
    stu_features = inputs[1].requires_grad_(True)
    model_inputs = (
        inputs[0],
        stu_features,
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        raw_inputs[-1],
    )
    hierarchy_outputs: list[Tensor] = []

    def retain_output(_module: object, _arguments: object, output: Tensor) -> None:
        output.retain_grad()
        hierarchy_outputs.append(output)

    handles = [
        block.register_forward_hook(retain_output) for block in model.pyramid.blocks
    ]
    try:
        logits = model(*model_inputs, grouping_mode="joint")
        # Softplus gives every output a strictly positive local loss derivative.
        torch.nn.functional.softplus(logits).sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    _require(logits.shape == (20,), "joint model did not score every input point")
    _require(stu_features.grad is not None, "input point evidence received no gradient")
    row_gradient = stu_features.grad.abs().sum(dim=1)
    _require(
        bool(torch.all(torch.isfinite(row_gradient) & (row_gradient > 0.0))),
        "at least one input return has no finite path to the loss",
    )
    _require(len(hierarchy_outputs) == 4, "joint hierarchy did not expose four levels")
    level_gradient = [
        0.0 if value.grad is None else float(value.grad.abs().sum())
        for value in hierarchy_outputs
    ]
    _require(
        all(np.isfinite(value) and value > 0.0 for value in level_gradient),
        "at least one joint hierarchy level received no gradient",
    )
    return {
        "passed": True,
        "scored_points": int(logits.numel()),
        "minimum_point_gradient_l1": float(row_gradient.min()),
        "hierarchy_gradient_l1": level_gradient,
    }


def occurrence_fusion_audit() -> dict[str, object]:
    """Compare production post-sigmoid occurrence means with an independent sum."""

    try:
        from .evaluate import WindowScoreFusion
    except ImportError:
        from evaluate import WindowScoreFusion

    fusion = WindowScoreFusion()
    expected: dict[tuple[int, int], list[float]] = {}
    world_identity = "schema31-fusion-fixture"
    for occurrence_count in range(1, 6):
        # Frame c-1 occurs in exactly the consecutive starts 0 through c-1.
        source_frame = occurrence_count - 1
        source_ray = 2000 + occurrence_count
        values = []
        for occurrence in range(occurrence_count):
            logit = -1.2 + 0.35 * occurrence_count + 0.17 * occurrence
            fusion.add_synthetic(
                world_identity=world_identity,
                window_start=occurrence,
                source_frame=np.asarray((source_frame,), dtype=np.int32),
                source_ray=np.asarray((source_ray,), dtype=np.int32),
                source_slot=np.asarray((occurrence_count,), dtype=np.int32),
                logits=np.asarray((logit,), dtype=np.float64),
            )
            values.append(float(1.0 / (1.0 + np.exp(-logit))))
        expected[(source_frame, source_ray)] = values

    # An identical frame-ray in another world must remain a distinct point.
    fusion.add_synthetic(
        world_identity="another-fixture-world",
        window_start=0,
        source_frame=np.asarray((0,), dtype=np.int32),
        source_ray=np.asarray((2001,), dtype=np.int32),
        source_slot=np.asarray((1,), dtype=np.int32),
        logits=np.asarray((8.0,), dtype=np.float64),
    )
    result = fusion.finalize_synthetic(world_identity)
    observed = {
        (int(frame), int(ray)): (float(probability), int(count))
        for frame, ray, probability, count in zip(
            result.source_frame,
            result.source_ray,
            result.probability,
            result.occurrence_count,
            strict=True,
        )
    }
    _require(observed.keys() == expected.keys(), "fusion changed point identities")
    errors = []
    for identity, probabilities in expected.items():
        probability, count = observed[identity]
        independent = float(
            np.sum(probabilities, dtype=np.float64) / len(probabilities)
        )
        errors.append(abs(probability - independent))
        _require(count == len(probabilities), "fusion occurrence count is incorrect")
    maximum_error = max(errors)
    _require(
        maximum_error <= 2.0e-7,
        "fusion differs from independent probability mean",
    )
    counts = sorted(count for _, count in observed.values())
    _require(
        counts == [1, 2, 3, 4, 5],
        "fusion did not expose occurrence strata 1 through 5",
    )
    return {
        "passed": True,
        "points": len(observed),
        "occurrence_counts": counts,
        "maximum_probability_error": maximum_error,
        "namespace_isolated": True,
    }


def stu_evidence_primitive_audit(seed: int = 3107) -> dict[str, object]:
    """Independently recompute the assigned-query evidence on a small fixture."""

    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((100, 20), generator=generator)
    masks = torch.randn((7, 100), generator=generator)
    observed = assigned_stu_evidence(logits, masks)
    class_probability = logits.softmax(dim=1)
    normal_probability = class_probability[:, :NUM_NORMAL_CLASSES]
    mask_probability = masks.sigmoid()
    strength = mask_probability * normal_probability.max(dim=1).values[None, :]
    query = strength.argmax(dim=1)
    rows = torch.arange(masks.shape[0])
    expected_evidence = mask_probability[rows, query, None] * normal_probability[query]
    expected_assign = strength[rows, query]
    expected_no_object = class_probability[query, NUM_NORMAL_CLASSES]
    expected_score = 1.0 - (mask_probability @ normal_probability).max(dim=1).values
    errors = (
        float(torch.max(torch.abs(observed.normal_evidence - expected_evidence))),
        float(torch.max(torch.abs(observed.reliability_assign - expected_assign))),
        float(torch.max(torch.abs(observed.reliability_noobj - expected_no_object))),
        float(torch.max(torch.abs(observed.maxlogit_score - expected_score))),
    )
    _require(torch.equal(observed.assigned_query, query), "assigned STU query differs")
    _require(
        max(errors) == 0.0,
        "assigned STU evidence differs from independent recomputation",
    )
    return {
        "passed": True,
        "voxels": int(masks.shape[0]),
        "maximum_absolute_error": max(errors),
        "scope": "synthetic_assignment_primitive_only",
    }


def _independent_point_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    cumulative_positive = np.cumsum(ranked_labels, dtype=np.int64)
    cumulative_negative = np.cumsum(~ranked_labels, dtype=np.int64)
    precision = cumulative_positive / np.arange(1, labels.size + 1)
    average_precision = float(precision[ranked_labels].sum() / positive_count)
    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    pairwise = (positive_scores[:, None] > negative_scores[None, :]).mean() + 0.5 * (
        positive_scores[:, None] == negative_scores[None, :]
    ).mean()
    candidates = np.flatnonzero(cumulative_positive / positive_count > 0.95)
    false_positive_rate = (
        0.0
        if candidates.size == 0
        else float(cumulative_negative[int(candidates[0])] / negative_count)
    )
    return {
        "AP": 100.0 * average_precision,
        "AUROC": 100.0 * float(pairwise),
        "FPR95": 100.0 * false_positive_rate,
    }


def evaluator_primitive_audit(seed: int = 3108) -> dict[str, object]:
    """Compare production point metrics against independent rank calculations."""

    try:
        from .evaluate import _point_metrics
    except ImportError:
        from evaluate import _point_metrics

    generator = np.random.default_rng(seed)
    labels = np.zeros(53, dtype=np.bool_)
    labels[generator.choice(labels.size, 21, replace=False)] = True
    scores = generator.normal(size=labels.size) + labels.astype(np.float64) * 0.35
    scores += np.arange(labels.size, dtype=np.float64) * 1.0e-8
    production = _point_metrics(labels, scores)
    independent = _independent_point_metrics(labels, scores)
    errors = {
        name: abs(float(production[name]) - independent[name])
        for name in ("AP", "AUROC", "FPR95")
    }
    _require(
        max(errors.values()) <= 1.0e-10,
        "point metrics differ from rank recomputation",
    )
    return {
        "passed": True,
        "points": int(labels.size),
        "anomaly_points": int(labels.sum()),
        "maximum_absolute_error": max(errors.values()),
        "scope": "lightweight_formula_regression_plus_retained_official_record",
    }


def sealed_access_audit(protocol: AJAEProtocol) -> dict[str, object]:
    """Check denial, protocol binding, and partition binding before file access."""

    denied = 0
    for partition, sequence_id in (("val", 125), ("test", 100)):
        try:
            _require_sealed_sequence_access(
                protocol, partition, None, sequence_id=sequence_id
            )
        except SceneDataError:
            denied += 1
        else:
            raise QualificationError(f"unsealed {partition} access was accepted")
        access = _grant_sealed_sequence_access(
            protocol, partition=partition, condition=ExperimentCondition.B3.value
        )
        _require_sealed_sequence_access(
            protocol, partition, access, sequence_id=sequence_id
        )
        other_partition = "test" if partition == "val" else "val"
        try:
            _require_sealed_sequence_access(
                protocol, other_partition, access, sequence_id=sequence_id
            )
        except SceneDataError:
            denied += 1
        else:
            raise QualificationError("sealed capability crossed its partition boundary")
    _require_sealed_sequence_access(protocol, "train", None, sequence_id=206)
    return {
        "passed": True,
        "denied_invalid_accesses": denied,
        "accepted_bound_capabilities": 2,
        "train_requires_capability": False,
    }


def window_proxy_evidence_status(
    protocol: AJAEProtocol,
    development_path: Path | str = DEFAULT_DEVELOPMENT_PATH,
) -> dict[str, object]:
    """Report scientific R02 evidence separately from synthetic mechanics."""

    path = Path(development_path).expanduser()
    if not path.is_file():
        return {
            "status": "pending",
            "schema31_bank_loaded": False,
            "reason": "development record does not exist",
        }
    try:
        worlds = load_development_worlds(path, protocol=protocol)
    except (OSError, ValueError) as error:
        return {
            "status": "pending",
            "schema31_bank_loaded": False,
            "reason": str(error),
        }
    clip_count = len(worlds.clips)
    window_count = len(worlds.windows)
    validation = dict(worlds.validation)
    matching = any(
        value is True and "match" in name.lower() for name, value in validation.items()
    )
    shortcut = any(
        value is True and "shortcut" in name.lower()
        for name, value in validation.items()
    )
    qualified = worlds.validated and window_count > 0 and matching and shortcut
    return {
        "status": "qualified_record_available" if qualified else "pending",
        "schema31_bank_loaded": True,
        "development_status": worlds.status,
        "clip_records": clip_count,
        "window_records": window_count,
        "matching_recorded": matching,
        "shortcut_audit_recorded": shortcut,
        "reason": (
            "validated schema31 window-bank record contains both decisions"
            if qualified
            else "a validated nonempty schema31 window bank with matching and shortcut decisions is required"
        ),
    }


def run_window_densification_audit(protocol: AJAEProtocol) -> dict[str, object]:
    """Execute the ten result-blind schema-31 densification mechanics."""

    voxel_joint, voxel_isolated, density = grouped_voxel_audits()
    checks = (
        (
            "symmetric_coordinates_scan_permutation_invariant",
            symmetric_coordinate_audit(),
        ),
        ("all_five_scan_points_losslessly_restored", full_point_recovery_audit()),
        ("joint_grouping_merges_cross_scan_voxels", voxel_joint),
        ("scan_isolated_grouping_keeps_voxels_separate", voxel_isolated),
        ("joint_population_changes_density_feature", density),
        ("model_interface_has_no_scan_order_input", no_order_feature_audit(protocol)),
        ("scan_permutation_output_equivariance", scan_permutation_audit()),
        (
            "joint_neighborhood_and_upsample_cross_scans",
            cross_scan_operator_audit(),
        ),
        (
            "all_points_backpropagate_through_joint_hierarchy",
            joint_gradient_audit(),
        ),
        (
            "occurrence_fusion_matches_independent_probability_mean",
            occurrence_fusion_audit(),
        ),
    )
    _require(len(checks) == 10, "densification audit must contain exactly ten checks")
    _require(
        all(bool(result["passed"]) for _, result in checks), "an audit check failed"
    )
    return {
        "name": AUDIT_NAME,
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "check_count": len(checks),
        "checks": {name: result for name, result in checks},
        "scope": "deterministic_synthetic_mechanical_qualification_only",
    }


def run_schema31_qualification(
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
    *,
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    development_path: Path | str = DEFAULT_DEVELOPMENT_PATH,
    inspect_retained: bool = True,
    hash_retained_files: bool = False,
) -> dict[str, object]:
    """Run mechanics and keep scientific evidence categories explicit."""

    protocol = load_protocol(protocol_path)
    _require(protocol.schema_version == SCHEMA_VERSION, "schema-31 protocol required")
    mechanical = run_window_densification_audit(protocol)
    retained: dict[str, object] | None = None
    if inspect_retained:
        retained = retained_evidence_audit(
            protocol, runs_root, hash_files=hash_retained_files
        )
    return {
        "format": AUDIT_FORMAT,
        "protocol_schema": SCHEMA_VERSION,
        "mechanical": mechanical,
        "retained_schema30_evidence": retained,
        "lightweight_primitives": {
            "stu_assigned_evidence": stu_evidence_primitive_audit(),
            "official_point_metrics": evaluator_primitive_audit(),
            "sealed_data_access": sealed_access_audit(protocol),
        },
        "window_proxy_science": window_proxy_evidence_status(
            protocol, development_path
        ),
        "formal_training_qualified": False,
        "performance_claim_available": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run result-blind schema-31 window densification qualifications."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-retained-evidence",
        action="store_true",
        help="skip reading the historical pass records",
    )
    parser.add_argument(
        "--hash-retained-files",
        action="store_true",
        help="also hash historical files; this reads about 200 MiB",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_schema31_qualification(
        args.protocol,
        runs_root=args.runs_root,
        development_path=args.development,
        inspect_retained=not args.skip_retained_evidence,
        hash_retained_files=args.hash_retained_files,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
