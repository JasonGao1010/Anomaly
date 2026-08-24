#!/usr/bin/env python3
"""Focused semantic tests for the active AJAE Oracle temporal experiment."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.train as training
from src.analyze import (
    _current_occupancy_statistics,
    _interpolation_support,
    _p16_candidate_query_records,
    _p16_graph_record,
    _support_statistics,
    _voxel_occupancy,
    summarize_history_alignment,
)
from src.evaluate import (
    EvaluationError,
    _point_metrics,
    _range_mask,
    balanced_binary_cross_entropy,
    mechanism_point_metrics,
    normal_alarm_threshold,
)
from src.protocol import ProtocolError, load_protocol
from src.static import (
    HistoryCandidate,
    HistoryMatchMass,
    HistoryPointPrediction,
    HistorySamplingOffsets,
    StaticModelError,
    WindowDetectorPrototype,
    _CausalWindowResidual,
    model_state_sha256,
    oracle_temporal_loss,
    proposal_match_null_loss,
)
from src.train import (
    ORACLE_TEMPORAL_PROTOCOL_FORMAT,
    TRAJECTORY_CONTROLS,
    OracleTemporalExperiment,
    TrainingError,
    TrajectoryPlan,
    _oracle_current_targets,
    _trajectory_profile_name,
)


ROOT = Path(__file__).resolve().parent


def test_persisted_paths_are_project_relative() -> None:
    for path in (
        ROOT / "oracle.json",
        ROOT / "results" / "oracle_source.json",
        Path(training.sys.executable),
    ):
        serialized = training._project_relative_path(path)
        assert not Path(serialized).is_absolute()
        assert (ROOT / serialized).resolve() == path.resolve()


def test_protocol_describes_only_the_current_temporal_question() -> None:
    protocol = load_protocol()
    document = protocol.plain_document()
    assert document["schema_version"] == 28
    assert document["task"]["history_lengths"] == [0, 1, 2, 4]
    assert document["task"]["window_startup_rule"] == {
        "available_history_only": (
            "At a sequence boundary, use only causal scans that already exist; "
            "never borrow a scan from another sequence or the future."
        ),
        "history_padding_forbidden": True,
        "frame_repetition_forbidden": True,
        "general_complete_window_minimum_current_frame": 4,
        "normal_201_complete_window_minimum_current_frame": 8,
        "complete_window_reason": (
            "A five-scan causal window generally starts at current frame 4. Only "
            "normal 201 starts at current frame 8 because its source frames 0 "
            "through 3 contain exact internal duplicates and must not appear as "
            "current input or history."
        ),
    }
    assert protocol.complete_window_frame_ids("train", 206, 4) == (0, 1, 2, 3, 4)
    assert protocol.complete_window_frame_ids("train", 201, 8) == (4, 5, 6, 7, 8)
    with pytest.raises(ProtocolError, match="current_frame >= 8"):
        protocol.complete_window_frame_ids("train", 201, 7)
    labels = document["label_semantics"]["binary_anomaly_training"]
    assert labels["raw_semantic_0"] == "ignore"
    assert labels["raw_semantic_2"] == "anomaly"
    assert labels["other_nonzero_semantics"] == "normal"
    assert labels["synthetic_members"].startswith("anomaly")
    assert document["model"]["name"] == "current-anchored factorized causal window encoder"
    assert document["model"]["temporal_scales"] == ["p16", "p8", "p4"]
    correspondence = document["model"]["history_correspondence"]
    assert set(correspondence) == {
        "clean_select",
        "proposal_oracle_candidates",
        "truth_use",
    }
    assert "exactly one truth-selected h_mix real candidate" in correspondence["clean_select"]
    assert "h_mix is never duplicated" in correspondence["clean_select"]
    assert document["training"]["stage_b"]["history_lengths"] == [1, 2, 4]
    assert set(document["training"]["stage_b"]) == {
        "purpose",
        "history_lengths",
        "independent_arms",
        "state_isolation",
        "classification_objective",
        "window_loss",
        "normal_safety",
        "magnitude_control",
        "direct_match",
        "direct_null",
        "direct_weights",
        "state_rule",
    }
    assert set(document["training"]["stage_b"]["independent_arms"]) == {
        "clean_select",
        "proposal_direct",
        "proposal_classification",
    }
    assert "Proposal arms receive no Clean Select gradient" in document["training"]["stage_b"]["state_isolation"]
    assert "query-age pairs" in document["training"]["stage_b"]["direct_match"]
    assert "structurally certain" in document["training"]["stage_b"]["direct_null"]
    assert "normalizes independently within each history age" in document["model"][
        "factorized_temporal_update"
    ]
    assert document["training"]["future_matcher"]["implemented"] is False


def test_protocol_rejects_a_retired_objective_field(tmp_path: Path) -> None:
    document = json.loads((ROOT / "split.json").read_text(encoding="utf-8"))
    document["training"]["stage_b"]["retired_auxiliary_objective"] = {
        "weight": 1.0
    }
    changed = tmp_path / "split.json"
    changed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(changed)


def test_binary_targets_keep_synthetic_raw_zero_returns() -> None:
    raw_semantic = np.asarray((0, 0, 2, 1, 52, 99), dtype=np.uint16)
    rendered = SimpleNamespace(
        synthetic_members=np.asarray((False, True, False, False, False, False)),
        counterfactual=SimpleNamespace(
            members=SimpleNamespace(current_slice=slice(0, raw_semantic.size)),
            current=SimpleNamespace(
                labels=SimpleNamespace(semantic=raw_semantic),
                real_slots=np.arange(raw_semantic.size),
            ),
        ),
    )
    anomaly, valid = _oracle_current_targets(rendered, torch.device("cpu"))
    assert anomaly.tolist() == [False, True, True, False, False, False]
    assert valid.tolist() == [False, True, True, True, True, True]


def test_oracle_source_has_one_plan_per_frozen_window() -> None:
    source = json.loads(
        (ROOT / "results" / "oracle_source.json").read_text(encoding="utf-8")
    )
    assert source["format"] == "ajae-oracle-mechanism-source-v3"
    assert source["scope"]["protocol_schema"] == 28
    selection = source["selection"]
    train = selection["train_frames"]
    validation = selection["validation_frames"]
    assert len(train) == len(set(train)) == 96
    assert len(validation) == len(set(validation)) == 64
    assert len(selection["train_orders"]) == 3
    assert all(sorted(order) == sorted(train) for order in selection["train_orders"])
    plans = selection["trajectory_plans"]
    assert set(plans) == {
        *(f"train:0:{frame}" for frame in train),
        *(f"validation:0:{frame}" for frame in validation),
    }
    allowed = {
        "seed",
        "angular_scale_rad",
        "radial_speed_mps",
        "anchor_mode",
        "current_anomaly_points",
        "trajectory_profile",
    }
    assert all(set(plan) == allowed for plan in plans.values())
    active = source["active_schema28_use"]
    active_train = active["active_train_frames"]
    active_validation = active["active_validation_frames"]
    assert len(active_train) == len(set(active_train)) == 24
    assert len(active_validation) == len(set(active_validation)) == 16
    assert set(active_train) < set(train)
    assert set(active_validation) < set(validation)
    assert active["preflight_only_validation_frames"] == [28]
    assert 28 in validation and 28 not in active_validation
    assert active["preflight_frames_are_excluded_from_metrics"] is True
    assert min((*active_train, *active_validation)) >= 8
    assert active["source_pool_is_inherited_96_64"] is True
    assert "scientifically valid" in active["normal_206_frame_4_status"]
    assert train[0] == 4


def test_trajectory_plan_accepts_only_the_eight_frozen_control_profiles() -> None:
    assert len(TRAJECTORY_CONTROLS) == 8
    assert len(set(TRAJECTORY_CONTROLS)) == 8
    profiles = set()
    for index, controls in enumerate(TRAJECTORY_CONTROLS):
        profile = _trajectory_profile_name(controls)
        plan = TrajectoryPlan(
            seed=index,
            angular_scale_rad=controls.angular_scale_rad,
            radial_speed_mps=controls.radial_speed_mps,
            anchor_mode=controls.anchor_mode,
            current_anomaly_points=1,
            trajectory_profile=profile,
        )
        assert plan.trajectory_profile == profile
        profiles.add(profile)
    assert len(profiles) == 8

    with pytest.raises(TrainingError, match="unknown trajectory controls"):
        TrajectoryPlan(
            seed=0,
            angular_scale_rad=0.012,
            radial_speed_mps=15.0,
            anchor_mode="uniform_ground",
            current_anomaly_points=1,
            trajectory_profile="unsupported",
        )
    controls = TRAJECTORY_CONTROLS[0]
    with pytest.raises(TrainingError, match="inconsistent trajectory controls"):
        TrajectoryPlan(
            seed=0,
            angular_scale_rad=controls.angular_scale_rad,
            radial_speed_mps=controls.radial_speed_mps,
            anchor_mode=controls.anchor_mode,
            current_anomaly_points=1,
            trajectory_profile="wrong-profile",
        )


def test_training_entry_loads_the_schema28_source_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = ROOT / "results" / "oracle_source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_record = manifest["sources"]["calibration"]
    calibration_path = (ROOT / calibration_record["path"]).resolve(strict=True)
    declared_calibration_hash = calibration_record["sha256"]
    real_sha256 = training._sha256_file

    def bounded_sha256(path: Path) -> str:
        resolved = Path(path).expanduser().resolve(strict=True)
        if resolved == calibration_path:
            return declared_calibration_hash
        return real_sha256(resolved)

    monkeypatch.setattr(training, "_sha256_file", bounded_sha256)
    data_root = tmp_path / "stu"
    for sequence_id in (206, 201):
        (data_root / "train" / str(sequence_id)).mkdir(parents=True)

    def bounded_source_identity(path: Path) -> dict[str, object]:
        sequence_id = Path(path).name
        source_key = {
            "206": "raw_train_206",
            "201": "raw_validation_201",
        }[sequence_id]
        identity = manifest["sources"][source_key]["identity"]
        return {
            "file_count": identity["file_count"],
            "bytes": identity["total_bytes"],
            "manifest_sha256": identity["content_manifest_sha256"],
        }

    monkeypatch.setattr(training, "_source_tree_identity", bounded_source_identity)
    experiment = object.__new__(OracleTemporalExperiment)
    experiment.manifest_path = manifest_path.resolve(strict=True)
    experiment.protocol = load_protocol()
    experiment.experiment_protocol_path = (ROOT / "oracle.json").resolve(strict=True)
    experiment.experiment_protocol = experiment._load_experiment_protocol()
    experiment.data_root = data_root
    source = experiment._load_source()

    assert len(source["train_frames"]) == 96
    assert len(source["validation_frames"]) == 64
    assert len(source["train_orders"]) == 3
    assert all(
        set(order) == set(source["train_frames"])
        for order in source["train_orders"]
    )
    assert len(source["plans"]) == 160
    assert all(isinstance(plan, TrajectoryPlan) for plan in source["plans"].values())
    assert source["calibration_path"] == calibration_path
    assert source["split_path"] == (ROOT / "split.json").resolve(strict=True)


def test_training_entry_loads_the_current_oracle_protocol_without_gpu() -> None:
    experiment = object.__new__(OracleTemporalExperiment)
    experiment.experiment_protocol_path = (ROOT / "oracle.json").resolve(strict=True)
    protocol = experiment._load_experiment_protocol()
    assert protocol["format"] == ORACLE_TEMPORAL_PROTOCOL_FORMAT
    assert protocol["source_scope"]["manifest"] == "results/oracle_source.json"
    assert protocol["trajectory"]["history_lengths"] == [1, 2, 4]
    assert protocol["micro_screen"]["conditions"][0] == "Current"
    assert len(protocol["micro_screen"]["train_frames"]) == 24
    assert len(protocol["micro_screen"]["validation_frames"]) == 16
    assert "combined_manifest_sha256" not in protocol["micro_screen"]
    assert len(protocol["gradient_audit"]["train_frames"]) == 8
    assert protocol["gradient_audit"] == {
        "report_format": training.STAGE0_AUDIT_FORMAT,
        "audit_mode": "zero_temporal_optimizer_updates_after_stage_a",
        "train_frames": [9, 72, 118, 214, 267, 329, 377, 436],
        "frames_sha256": "137b4af9c6210f4f52658cda8dab71fb807ef3f40ff5550e3ba67f0ff77d1b38",
        "history_length": 4,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "gradient_api": "torch.autograd.grad",
        "classification_and_direct_use_independent_forwards": True,
        "classification_terms": ["L_win", "L_safe", "0.1*L_mag"],
        "direct_terms": ["L_match", "L_null"],
        "joint_ratio": (
            "per-window L2 norm of grad(L_match+L_null) divided by L2 norm "
            "of grad(L_win+L_safe+0.1*L_mag)"
        ),
        "overall_median_ratio_minimum": 0.1,
        "overall_median_ratio_maximum": 10.0,
        "per_scale_median_ratio_maximum": 20.0,
        "undefined_zero_over_zero_json_value": None,
        "temporal_point_delta_zero_over_zero_is_expected_at_initialization": True,
        "branches": [
            "temporal_p16",
            "temporal_p8",
            "temporal_p4",
            "temporal_point_delta",
        ],
        "lambda_dir": 1.0,
        "restore_formal_parameters_rng_mode_requires_grad_and_grad_buffers_in_finally": True,
        "automatic_weight_change": False,
    }
    preflight = protocol["candidate_order_preflight"]
    assert preflight["required_before_temporal_training"] is True
    assert preflight["evaluation_states"] == [
        "formal_temporal_initial",
        "deterministic_nonzero_model_copy",
    ]
    assert preflight["repetitions_per_order"] == 2
    assert preflight["repeat_noise_multiplier"] == 4.0
    assert preflight["nonzero_copy"] == {
        "seed": 20260824,
        "standard_deviation": 0.01,
        "perturbation": (
            "independent CPU-generated Gaussian perturbation added only to "
            "copied temporal parameters"
        ),
        "formal_model_mutated": False,
        "optimizer_steps": 0,
    }
    assert protocol["micro_screen"]["stage_a_passes"] == 1
    assert protocol["micro_screen"]["stage_b_passes"] == 1
    continue_rule = protocol["micro_screen"]["continue_rule"]
    assert continue_rule["moving_normal_sentinel_minimum_points"] == 100
    assert continue_rule["moving_normal_sentinel_minimum_windows"] == 2
    assert protocol["direct_supervision"]["direct_softmax_scope"].startswith(
        "independently within each history age"
    )
    assert protocol["next_96_64"]["implemented_in_this_protocol"] is False
    assert protocol["runtime_budget"]["final_result_write_reserve_seconds"] == 10
    assert protocol["runtime_budget"]["accumulate_recovered_wall_seconds"] is True
    assert protocol["runtime_budget"]["projection_after_windows"] == 8
    assert protocol["recovery"]["optimizer_retention_scope"].startswith(
        "All three optimizer and RNG states are stored after every active stage-B window"
    )
    assert protocol["cache"] == {
        "persistent_render_or_feature_cache": False,
        "deterministic_rerender_from_frozen_plan": True,
        "window_local_fp32_materialization": True,
        "release_derived_window_after_use": True,
        "resume_recomputes_next_window_from_source": True,
        "precision": "FP32",
        "sparse_coordinate_manager_is_never_serialized": True,
    }


def test_runtime_limit_accumulates_prior_recoverable_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = object.__new__(OracleTemporalExperiment)
    experiment._run_wall_started = 100.0
    experiment._run_wall_prior_seconds = 200.0
    experiment.experiment_protocol = {
        "runtime_budget": {"maximum_wall_seconds": 300.0}
    }
    monkeypatch.setattr(training.time, "perf_counter", lambda: 199.0)
    assert experiment._observed_wall_seconds() == pytest.approx(299.0)
    experiment._enforce_hard_runtime_limit()
    monkeypatch.setattr(training.time, "perf_counter", lambda: 201.0)
    with pytest.raises(TrainingError, match="300-second hard wall"):
        experiment._enforce_hard_runtime_limit()


def test_live_runtime_projection_uses_recovery_component_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = object.__new__(OracleTemporalExperiment)
    experiment.train_frames = tuple(range(24))
    experiment.validation_frames = tuple(range(16))
    experiment.experiment_protocol = {
        "runtime_budget": {
            "maximum_wall_seconds": 300.0,
            "projection_after_windows": 8,
            "window_quantile": 0.9,
            "include_first_window_in_projection": True,
            "fixed_projection_overhead_seconds": 60.0,
            "projection_contingency_multiplier": 1.2,
        }
    }
    monkeypatch.setattr(experiment, "_observed_wall_seconds", lambda: 100.0)
    rows = [
        {
            "preparation_seconds": 2.0,
            "arm_update_seconds": {"A": 2.0, "B": 3.0, "C": 3.0},
            "checkpoint_seconds": None,
            "window_seconds_including_checkpoint": None,
        }
        for _ in range(8)
    ]
    with pytest.raises(TrainingError, match="runtime projection exceeds 300 seconds"):
        experiment._enforce_live_runtime_projection(
            stage="stage_b", completed_position=8, timing_rows=rows
        )


def test_failure_runtime_checkpoint_does_not_advance_model_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = object.__new__(OracleTemporalExperiment)
    experiment.output = tmp_path
    experiment._run_wall_started = 100.0
    experiment._run_wall_prior_seconds = 12.0
    experiment.progress = {
        "format": "progress",
        "stage": "stage_a_complete",
        "temporal_state": {"weight": torch.tensor([3.0])},
        "data": {"runtime_elapsed_wall_seconds": 12.0},
    }
    training._save_checkpoint(tmp_path / "progress.pt", experiment.progress)
    monkeypatch.setattr(training.time, "perf_counter", lambda: 105.0)
    experiment._checkpoint_failure_runtime()
    saved = torch.load(tmp_path / "progress.pt", weights_only=True)
    assert saved["stage"] == "stage_a_complete"
    assert torch.equal(saved["temporal_state"]["weight"], torch.tensor([3.0]))
    assert saved["data"]["runtime_elapsed_wall_seconds"] == pytest.approx(17.0)


def test_repeat_noise_relative_permutation_gate() -> None:
    comparison = training._repeat_noise_relative_comparison(
        torch.tensor([0.0]),
        torch.tensor([0.5e-6]),
        torch.tensor([3.0e-6]),
        torch.tensor([2.5e-6]),
        absolute_tolerance=1.0e-6,
        relative_tolerance=0.0,
        repeat_noise_multiplier=4.0,
    )
    assert comparison["base_tolerance"] == pytest.approx(1.0e-6)
    assert comparison["repeat_noise"] == pytest.approx(0.5e-6)
    assert comparison["allowed_swap_difference"] == pytest.approx(3.0e-6)
    assert comparison["passed"] is True

    excessive_swap = training._repeat_noise_relative_comparison(
        torch.tensor([0.0]),
        torch.tensor([0.5e-6]),
        torch.tensor([3.1e-6]),
        torch.tensor([2.6e-6]),
        absolute_tolerance=1.0e-6,
        relative_tolerance=0.0,
        repeat_noise_multiplier=4.0,
    )
    assert excessive_swap["passed"] is False

    unstable_repeat = training._repeat_noise_relative_comparison(
        torch.tensor([0.0]),
        torch.tensor([2.0e-6]),
        torch.tensor([0.0]),
        torch.tensor([2.0e-6]),
        absolute_tolerance=1.0e-6,
        relative_tolerance=0.0,
        repeat_noise_multiplier=4.0,
    )
    assert unstable_repeat["repeat_stable"] is False
    assert unstable_repeat["passed"] is False


def test_write_json_converts_nested_nonfinite_values_to_null(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.json"
    training._write_json(
        path,
        {
            "finite": 1.25,
            "nested": [
                float("nan"),
                float("inf"),
                -float("inf"),
                np.float32("nan"),
                np.float64("inf"),
            ],
        },
    )
    text = path.read_text(encoding="utf-8")

    def reject_constant(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {token}")

    document = json.loads(text, parse_constant=reject_constant)
    assert document == {
        "finite": 1.25,
        "nested": [None, None, None, None, None],
    }
    assert "NaN" not in text and "Infinity" not in text


def test_stage0_audit_contains_no_optimizer_step() -> None:
    for method in (
        OracleTemporalExperiment._loss_gradient_audit,
        OracleTemporalExperiment._stage0_invariance_checks,
    ):
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
        assert "torch.optim" not in source
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]


def test_stage0_audit_restores_formal_state_on_exception(tmp_path: Path) -> None:
    class TinyAuditModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_p16 = torch.nn.Linear(1, 1, bias=False)
            self.temporal_p8 = torch.nn.Linear(1, 1, bias=False)
            self.temporal_p4 = torch.nn.Linear(1, 1, bias=False)
            self.temporal_point_delta = torch.nn.Linear(1, 1, bias=False)
            self.point_anomaly_head = torch.nn.Linear(1, 1)

        def temporal_parameters(self):
            for module in (
                self.temporal_p16,
                self.temporal_p8,
                self.temporal_p4,
                self.temporal_point_delta,
            ):
                yield from module.parameters()

    model = TinyAuditModel()
    model.train()
    next(model.point_anomaly_head.parameters()).grad = torch.ones(1, 1)
    experiment = object.__new__(OracleTemporalExperiment)
    experiment.model = model
    experiment.training = object()
    experiment.output = tmp_path
    experiment.experiment_protocol = {
        "gradient_audit": {"train_frames": [1]}
    }
    experiment._enforce_hard_runtime_limit = lambda: None
    experiment._rng_snapshot = lambda: {"torch": torch.get_rng_state().clone()}
    experiment._restore_rng = lambda state: torch.set_rng_state(state["torch"])
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    flags_before = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    rng_before = torch.get_rng_state().clone()

    class ForcedAuditError(RuntimeError):
        pass

    def fail_after_mutation(*_args, **_kwargs):
        with torch.no_grad():
            next(model.temporal_parameters()).add_(1.0)
            model.point_anomaly_head.weight.add_(1.0)
        model.eval()
        torch.rand(())
        raise ForcedAuditError("forced audit failure")

    experiment._compile_history_features = fail_after_mutation
    with pytest.raises(ForcedAuditError, match="forced audit failure"):
        experiment._loss_gradient_audit()

    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == flags_before[name]
        expected = gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected)
    assert not (tmp_path / "stage0.json").exists()


def test_history_alignment_support_distinguishes_object_normal_and_empty() -> None:
    coordinates = np.asarray(
        (
            (0.01, 0.01, 0.01),
            (0.02, 0.01, 0.01),
            (0.21, 0.01, 0.01),
            (0.41, 0.01, 0.01),
            (-0.01, 0.01, 0.01),
        ),
        dtype=np.float64,
    )
    anomaly = np.asarray((True, False, False, True, False), dtype=np.bool_)
    occupancy = _voxel_occupancy(coordinates, anomaly, stride=4)
    assert occupancy[(-4, 0, 0)] == (0, 1)
    queries = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (8.0, 0.0, 0.0),
            (12.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    statistics = _support_statistics(
        *_interpolation_support(queries, occupancy, stride=4)
    )
    assert statistics["categories"] == {
        "anomaly_only": 1,
        "anomaly_and_normal": 2,
        "normal_only": 1,
        "empty": 1,
    }
    assert statistics["anomaly_hit_rate"] == pytest.approx(3 / 5)
    assert statistics["valid_support_rate"] == pytest.approx(4 / 5)
    assert _current_occupancy_statistics(occupancy) == {
        "anomaly_voxels": 2,
        "pure_anomaly_voxels": 1,
        "mixed_anomaly_voxels": 1,
        "anomaly_points": 2,
        "anomaly_points_in_mixed_voxels": 1,
        "has_mixed_anomaly_voxel": True,
        "has_no_pure_anomaly_voxel": False,
    }


def test_p16_search_records_recall_background_and_null() -> None:
    occupancy = {
        (0, 0, 0): (1, 1),
        (16, 0, 0): (0, 2),
        (32, 0, 0): (1, 2),
        (48, 0, 0): (0, 1),
    }
    query = np.asarray(((16.0, 0.0, 0.0),), dtype=np.float64)
    record = _p16_candidate_query_records(query, occupancy, age=1)[0]
    assert record["nearest_anomaly_distance_metres"] == pytest.approx(0.8)
    assert record["nearest_anomaly_best_rank"] == 2
    assert record["nearest_anomaly_guaranteed_rank"] == 2
    assert record["policies"]["v12_m00"] == {
        "total": 3,
        "anomaly_only": 1,
        "mixed": 1,
        "normal_only": 1,
    }
    null = _p16_candidate_query_records(
        query,
        {(0, 0, 0): (0, 1), (16, 0, 0): (0, 2)},
        age=1,
    )[0]
    assert null["history_anomaly_visible"] is False
    assert null["nearest_anomaly_distance_metres"] is None
    graph = _p16_graph_record(query, occupancy, age=1)
    assert graph["policies"]["v15_m08"]["candidate_edges"] == 4
    assert graph["policies"]["v15_m08"]["capped_candidate_edges"]["32"] == 4


def test_retained_alignment_records_can_be_resummarized(tmp_path: Path) -> None:
    statistic = _support_statistics(
        np.asarray((1.0, 0.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((1.0, 1.0)),
    )
    record = {
        "scales": {
            str(stride): {"fixed": statistic, "oracle": statistic}
            for stride in (16, 8, 4)
        }
    }
    source = tmp_path / "alignment.json"
    source.write_text(
        json.dumps(
            {
                "format": "ajae-history-alignment-audit-v2",
                "records": [record],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_history_alignment(source)
    assert summary["records"] == 1
    assert summary["summaries"]["4"]["anomaly_hit_rate_gain"] == 0.0


def test_window_residual_is_initially_identity_but_receives_gradient() -> None:
    torch.manual_seed(4)
    module = _CausalWindowResidual(8)
    current = torch.randn(7, 8)
    history = [
        (torch.randn(7, 8), torch.tensor([1, 1, 0, 1, 0, 1, 1]).bool(), 1),
        (torch.randn(7, 8), torch.tensor([1, 0, 1, 1, 1, 0, 1]).bool(), 2),
    ]
    output, context, support = module(current, history)
    assert torch.equal(output, current)
    assert context.shape == current.shape
    assert bool(support.all())
    output.square().mean().backward()
    assert module.output.weight.grad is not None
    assert float(module.output.weight.grad.abs().sum()) > 0.0


def _prediction_with_match_mass(
    output: torch.Tensor,
    current: torch.Tensor,
    support: torch.Tensor,
    mass: HistoryMatchMass,
) -> HistoryPointPrediction:
    return HistoryPointPrediction(
        logits=output[:, 0],
        correction=output[:, 0] - current[:, 0],
        point_history_support=support,
        history_coverage=torch.zeros(3),
        scale_residuals=(output - current, output - current, output - current),
        match_mass_by_scale=(mass, mass, mass),
    )


def test_clean_select_cache_has_one_real_slot_and_neutral_null_prior() -> None:
    prototype = object.__new__(WindowDetectorPrototype)
    torch.nn.Module.__init__(prototype)
    prototype.register_parameter(
        "_test_device_parameter", torch.nn.Parameter(torch.zeros(()))
    )
    record = {
        "feature": torch.zeros(3, 1),
        "valid": torch.ones(3, dtype=torch.bool),
        "age": 1,
    }
    payload = {
        "candidates": {
            "oracle_select": {
                scale: [record] for scale in ("p16", "p8", "p4")
            }
        }
    }
    candidates = prototype._cached_candidates(
        payload,
        "oracle_select",
        1,
        "actual",
        "static_object",
        "oracle_select",
    )
    assert all(len(candidates[scale]) == 1 for scale in ("p16", "p8", "p4"))

    module = _CausalWindowResidual(1)
    current = torch.zeros(3, 1)
    _, _, support, mass = module(
        current, candidates["p16"], return_match_mass=True
    )
    assert bool(support.all())
    # One zero-score evidence slot and one zero-score null split mass equally.
    torch.testing.assert_close(mass.null, torch.full((3,), 0.5))
    assert torch.count_nonzero(mass.same_object) == 0
    assert torch.count_nonzero(mass.target_weight) == 0


def test_proposal_match_null_loss_uses_age_local_truth_mass() -> None:
    module = _CausalWindowResidual(4)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    current = torch.zeros(3, 4)
    target_weight = torch.tensor((0.25, 0.75, 1.0))
    static = HistoryCandidate(
        feature=torch.zeros(3, 4),
        valid=torch.ones(3, dtype=torch.bool),
        age=1,
        same_object=torch.tensor((True, True, False)),
        target_weight=target_weight,
    )
    moving = HistoryCandidate(
        feature=torch.zeros(3, 4),
        valid=torch.ones(3, dtype=torch.bool),
        age=1,
        same_object=torch.tensor((False, True, False)),
        target_weight=target_weight,
    )
    output, _, support, mass = module(
        current, (static, moving), return_match_mass=True
    )
    torch.testing.assert_close(
        mass.same_object, torch.tensor((1.0 / 3.0, 2.0 / 3.0, 0.0))
    )
    torch.testing.assert_close(mass.null, torch.full((3,), 1.0 / 3.0))
    assert mass.has_same_object.tolist() == [True, True, False]

    loss = proposal_match_null_loss(
        _prediction_with_match_mass(output, current, support, mass)
    )
    expected_match = -(
        0.25 * torch.log(torch.tensor(1.0 / 3.0))
        + 0.75 * torch.log(torch.tensor(2.0 / 3.0))
    )
    expected_null = -torch.log(torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(loss.match, expected_match)
    torch.testing.assert_close(loss.null, expected_null)
    torch.testing.assert_close(loss.total, expected_match + expected_null)
    assert loss.match_queries_by_scale == (2, 2, 2)
    assert loss.null_queries_by_scale == (1, 1, 1)
    assert loss.structural_null_queries_by_scale == (0, 0, 0)


def test_age_local_null_remains_learnable_when_another_age_has_a_match() -> None:
    module = _CausalWindowResidual(4)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    current = torch.zeros(2, 4)
    target_weight = torch.ones(2)
    history = (
        HistoryCandidate(
            torch.zeros(2, 4),
            torch.tensor((True, False)),
            1,
            torch.zeros(2, dtype=torch.bool),
            target_weight,
        ),
        HistoryCandidate(
            torch.zeros(2, 4),
            torch.tensor((True, False)),
            1,
            torch.zeros(2, dtype=torch.bool),
            target_weight,
        ),
        HistoryCandidate(
            torch.zeros(2, 4),
            torch.ones(2, dtype=torch.bool),
            2,
            torch.ones(2, dtype=torch.bool),
            target_weight,
        ),
        HistoryCandidate(
            torch.zeros(2, 4),
            torch.ones(2, dtype=torch.bool),
            2,
            torch.zeros(2, dtype=torch.bool),
            target_weight,
        ),
    )
    output, _, support, mass = module(current, history, return_match_mass=True)
    assert mass.has_same_object.tolist() == [True, True]
    assert mass.direct_has_same_object.tolist() == [[False, True], [False, True]]
    assert mass.direct_real_valid.tolist() == [[True, True], [False, True]]

    loss = proposal_match_null_loss(
        _prediction_with_match_mass(output, current, support, mass)
    )
    expected = -torch.log(torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(loss.match, expected)
    torch.testing.assert_close(loss.null, expected)
    assert loss.match_queries_by_scale == (2, 2, 2)
    assert loss.null_queries_by_scale == (1, 1, 1)
    assert loss.structural_null_queries_by_scale == (1, 1, 1)


def test_match_truth_changes_only_loss_metadata_not_default_classification() -> None:
    torch.manual_seed(31)
    module = _CausalWindowResidual(8)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    current = torch.randn(4, 8)
    feature = torch.randn(4, 8)
    valid = torch.ones(4, dtype=torch.bool)
    first = HistoryCandidate(
        feature,
        valid,
        1,
        torch.tensor((True, False, False, True)),
        torch.tensor((0.2, 0.3, 0.4, 0.5)),
    )
    second = HistoryCandidate(
        feature,
        valid,
        1,
        torch.tensor((False, True, True, False)),
        torch.tensor((0.9, 0.8, 0.7, 0.6)),
    )
    default_first = module(current, (first,))
    default_second = module(current, (second,))
    assert len(default_first) == len(default_second) == 3
    for left, right in zip(default_first, default_second, strict=True):
        assert torch.equal(left, right)

    *_, first_mass = module(current, (first,), return_match_mass=True)
    *_, second_mass = module(current, (second,), return_match_mass=True)
    assert not torch.equal(first_mass.has_same_object, second_mass.has_same_object)
    assert not torch.equal(first_mass.target_weight, second_mass.target_weight)


def test_explicit_null_is_zero_value_and_not_real_support() -> None:
    torch.manual_seed(23)
    module = _CausalWindowResidual(8)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_()
    current = torch.randn(4, 8)
    history = [(torch.zeros(4, 8), torch.zeros(4, dtype=torch.bool), 2)]
    output, context, support = module(current, history)
    assert torch.equal(output, current)
    assert torch.count_nonzero(context) == 0
    assert not bool(support.any())


@pytest.mark.parametrize("channels", (64, 256))
def test_history_candidate_order_preserves_output_and_gradient(channels: int) -> None:
    torch.manual_seed(channels)
    module = _CausalWindowResidual(channels)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    assert torch.count_nonzero(module.output.weight) > 0
    assert torch.count_nonzero(module.query.weight) > 0
    assert torch.count_nonzero(module.key.weight) > 0
    current = torch.randn(5, channels)
    target_weight = torch.tensor((0.2, 0.4, 0.6, 0.8, 1.0))
    static = HistoryCandidate(
        feature=torch.randn(5, channels),
        valid=torch.ones(5, dtype=torch.bool),
        age=1,
        same_object=torch.tensor((True, False, False, True, False)),
        target_weight=target_weight,
    )
    moving = HistoryCandidate(
        feature=torch.randn(5, channels),
        valid=torch.tensor((1, 1, 0, 1, 1), dtype=torch.bool),
        age=1,
        same_object=torch.tensor((False, True, False, True, False)),
        target_weight=target_weight,
    )

    def evaluate(history: tuple[HistoryCandidate, ...]):
        module.zero_grad(set_to_none=True)
        output, context, support, mass = module(
            current, history, return_match_mass=True
        )
        direct = proposal_match_null_loss(
            _prediction_with_match_mass(output, current, support, mass)
        )
        (output.square().mean() + context.square().mean() + direct.total).backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        return {
            "output": output.detach(),
            "context": context.detach(),
            "support": support.detach(),
            "same_object": mass.same_object.detach(),
            "null": mass.null.detach(),
            "direct_same_object": mass.direct_same_object.detach(),
            "direct_null": mass.direct_null.detach(),
            "direct_has_same_object": mass.direct_has_same_object.detach(),
            "direct_real_valid": mass.direct_real_valid.detach(),
            "direct_loss": direct.total.detach(),
            "gradients": gradients,
        }

    first = evaluate((static, moving))
    repeated = evaluate((static, moving))
    swapped = evaluate((moving, static))
    swapped_repeated = evaluate((moving, static))
    for name in (
        "output",
        "context",
        "same_object",
        "null",
        "direct_same_object",
        "direct_null",
        "direct_loss",
    ):
        torch.testing.assert_close(
            swapped[name], first[name], rtol=1.0e-6, atol=1.0e-6
        )
        comparison = training._repeat_noise_relative_comparison(
            first[name],
            repeated[name],
            swapped[name],
            swapped_repeated[name],
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-5,
            repeat_noise_multiplier=4.0,
        )
        assert comparison["passed"] is True
    for name in ("support", "direct_has_same_object", "direct_real_valid"):
        assert torch.equal(swapped[name], first[name])
    assert set(first["gradients"]) == set(swapped["gradients"])
    for name in first["gradients"]:
        torch.testing.assert_close(
            swapped["gradients"][name],
            first["gradients"][name],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        comparison = training._repeat_noise_relative_comparison(
            first["gradients"][name],
            repeated["gradients"][name],
            swapped["gradients"][name],
            swapped_repeated["gradients"][name],
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-5,
            repeat_noise_multiplier=4.0,
        )
        assert comparison["passed"] is True


def test_oracle_mix_does_not_renormalize_a_missing_component() -> None:
    static = torch.tensor(((1.0,), (2.0,), (3.0,), (4.0,)))
    moving = torch.tensor(((10.0,), (20.0,), (30.0,), (40.0,)))
    fraction = torch.tensor((0.0, 1.0, 0.25, 1.0))
    mixed, valid = WindowDetectorPrototype._oracle_mix(
        static,
        torch.ones(4, dtype=torch.bool),
        moving,
        torch.tensor((True, True, True, False)),
        fraction,
        torch.ones(4, dtype=torch.bool),
    )
    torch.testing.assert_close(mixed[:, 0], torch.tensor((1.0, 20.0, 9.75, 0.0)))
    assert valid.tolist() == [True, True, True, False]


def test_history_sampling_controls_change_only_query_offsets() -> None:
    coordinates = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
    offsets = np.zeros((5, 2, 3), dtype=np.float32)
    offsets[1:] = np.asarray(((0.4, -0.2, 0.1), (-0.7, 0.3, 0.0)))
    membership = np.asarray((True, False), dtype=np.bool_)
    oracle = HistorySamplingOffsets(coordinates, offsets, membership)
    sham = oracle.sham()
    assert np.array_equal(sham.current_coordinates, coordinates)
    assert np.array_equal(sham.query_offsets, -offsets)
    assert np.array_equal(
        np.linalg.norm(sham.query_offsets, axis=2),
        np.linalg.norm(oracle.query_offsets, axis=2),
    )
    assert np.array_equal(sham.object_membership, membership)
    fixed = oracle.fixed_like()
    assert not np.count_nonzero(fixed.query_offsets)
    assert np.array_equal(fixed.object_membership, membership)


def test_oracle_temporal_loss_freezes_current_and_balances_magnitude() -> None:
    window = torch.tensor((1.2, -0.8, 0.3, -1.7), requires_grad=True)
    current = torch.tensor((0.4, -0.2, 0.1, -1.0), requires_grad=True)
    anomaly = torch.tensor((True, True, False, False))
    valid = torch.ones(4, dtype=torch.bool)
    loss = oracle_temporal_loss(window, current, anomaly, valid)
    torch.testing.assert_close(
        loss.total,
        loss.window_bce + loss.normal_safety + 0.1 * loss.magnitude,
    )
    expected_magnitude = 0.5 * (loss.anomaly_magnitude + loss.normal_magnitude)
    torch.testing.assert_close(loss.magnitude, expected_magnitude)
    loss.total.backward()
    assert window.grad is not None and float(window.grad.abs().sum()) > 0.0
    assert current.grad is None


def test_point_metrics_and_class_balance_have_known_values() -> None:
    labels = np.asarray((False, True, False, True), dtype=np.bool_)
    logits = np.asarray((-2.0, 2.0, -1.0, 1.0), dtype=np.float64)
    metrics = mechanism_point_metrics(labels, logits)
    assert metrics["AP"] == 100.0
    assert metrics["AUROC"] == 100.0
    assert metrics["balanced_BCE"] == pytest.approx(
        balanced_binary_cross_entropy(labels, logits)
    )
    official = _point_metrics(labels, np.asarray((0.1, 0.9, 0.2, 0.8)))
    assert official["AP"] == 100.0
    assert official["AUROC"] == 100.0
    with pytest.raises(EvaluationError):
        balanced_binary_cross_entropy(np.zeros(3, dtype=np.bool_), np.zeros(3))


def test_normal_threshold_and_range_boundaries_are_exact() -> None:
    values = np.asarray((0.1, 0.4, 0.2, 0.3), dtype=np.float32)
    assert normal_alarm_threshold(values, 0.25) == pytest.approx(0.3)
    protocol = SimpleNamespace(
        evaluation=SimpleNamespace(minimum_range_m=1.0, maximum_range_m=3.0)
    )
    source = SimpleNamespace(
        xyzi=np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0), (3.1, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
    )
    assert _range_mask(protocol, source).tolist() == [True, True, False]


def test_model_state_hash_binds_name_dtype_shape_and_bytes() -> None:
    state = {
        "b": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        "a": torch.tensor([3, 4], dtype=torch.int64),
    }
    digest = model_state_sha256(state)
    assert digest == "043451f790f46bc913e1440925755db0805c6faab769017403bf7efddce419d6"
    changed = dict(state)
    changed["b"] = changed["b"].reshape(2, 1)
    assert model_state_sha256(changed) != digest
    with pytest.raises(StaticModelError):
        model_state_sha256({"bad": object()})


def test_schema28_manifest_payload_and_raw_tree_identities_are_bound() -> None:
    source = (ROOT / "results" / "oracle_source.json").read_bytes()
    file_digest = hashlib.sha256(source).hexdigest()
    assert len(file_digest) == 64
    document = json.loads(source)
    payload = dict(document)
    declared_payload_hash = payload.pop("payload_sha256")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert declared_payload_hash == hashlib.sha256(encoded).hexdigest()
    assert document["format"] == "ajae-oracle-mechanism-source-v3"
    assert document["scope"]["protocol_schema"] == 28
    assert document["scope"]["parameter_updates"] == 0
    sources = document["sources"]
    train_identity = sources["raw_train_206"]["identity"]
    validation_identity = sources["raw_validation_201"]["identity"]
    assert sources["raw_train_206"]["root"] == "train/206"
    assert train_identity["file_count"] == 900
    assert train_identity["total_bytes"] == 1_177_133_182
    assert train_identity["content_manifest_sha256"] == (
        "3c2c39428430f98f0a1cc3053338295ac8ebefea7f4d07816559ee2cb3952930"
    )
    assert train_identity["semantic_target_255_nonzero_normal_negative_points"] == 1_589_676
    assert sources["raw_validation_201"]["root"] == "train/201"
    assert validation_identity["file_count"] == 1366
    assert validation_identity["total_bytes"] == 1_804_296_734
    assert validation_identity["content_manifest_sha256"] == (
        "9c332978e2328fea719c03ad0f74fd92a5278ac44a97445627bb832ea7dccd5e"
    )
    assert validation_identity["semantic_target_255_nonzero_normal_negative_points"] == 76_735
