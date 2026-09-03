#!/usr/bin/env python3
"""Frozen STU point evidence and the AJAE joint-window point model."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STU_REPOSITORY = (
    PROJECT_ROOT.parent / "DynaCAN-deps" / "stu_dataset" / "Mask4Former3D"
)
NUM_NORMAL_CLASSES = 19
NUM_QUERIES = 100
MASK_DIM = 128
STU_VOXEL_SIZE_METRES = 0.05
STU_MODEL_STATE_FORMAT = "ajae-stu-normal-model-state-v2"
STU_MODEL_STATE_CONVERSION_RULE = "extract_exact_model_prefix_strip_once_v1"
KDTREE_WORKERS = max(1, min(24, os.cpu_count() or 1))

# Frozen by the controlled model.* extraction from the official STU release.
STU_CHECKPOINT_BYTES = 476_261_075
STU_CHECKPOINT_SHA256 = (
    "743b10d39c4076d98533bf1e84d389ad2703016904d31146e48919618b07b67a"
)
STU_MODEL_STATE_BYTES = 158_806_731
STU_MODEL_STATE_SHA256 = (
    "bd62c2ace0fd13911e2ba81f4969ca6633e73ec5270ffc0b1bd61840b05f924d"
)
STU_MODEL_STATE_TENSOR_SHA256 = (
    "0be4805592a3d064b21655c6c6eeeb7227322c9670873345be52747b0a24d1fb"
)


class ModelError(ValueError):
    """Report an invalid model input, checkpoint, or output."""


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        count = int((~torch.isfinite(value)).sum().item())
        raise ModelError(f"{name} contains {count} non-finite value(s)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in canonical key order."""

    if any(not isinstance(name, str) for name in state):
        raise ModelError("STU model state must use string tensor names")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise ModelError("STU model state must map strings to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def stu_model_state_path(checkpoint: Path | str) -> Path:
    """Resolve the restricted-loader tensor file actually consumed by AJAE."""

    source = Path(checkpoint).expanduser().resolve(strict=True)
    return source.with_name(f"{source.stem}.model_state.pt").resolve(strict=True)


def stu_source_manifest(repository: Path | str) -> dict[str, object]:
    """Hash every official STU Python and YAML source consumed by AJAE."""

    root = Path(repository).expanduser().resolve(strict=True)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "files": records,
    }


def _validated_stu_weights(
    checkpoint: Path | str,
) -> tuple[Mapping[str, Tensor], dict[str, object]]:
    """Validate both official bytes and the safely converted tensor payload."""

    source = Path(checkpoint).expanduser().resolve(strict=True)
    converted = stu_model_state_path(source)
    source_sha256 = _sha256(source)
    converted_sha256 = _sha256(converted)
    if source.stat().st_size != STU_CHECKPOINT_BYTES or (
        source_sha256 != STU_CHECKPOINT_SHA256
    ):
        raise ModelError("STU checkpoint is not the frozen official release")
    if converted.stat().st_size != STU_MODEL_STATE_BYTES or (
        converted_sha256 != STU_MODEL_STATE_SHA256
    ):
        raise ModelError("converted STU checkpoint is not the frozen extraction")
    try:
        payload = torch.load(converted, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelError("converted STU checkpoint cannot be safely loaded") from error

    required = {
        "format",
        "conversion_rule",
        "source_checkpoint_bytes",
        "source_checkpoint_sha256",
        "tensor_sha256",
        "state_dict",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ModelError("converted STU checkpoint has an invalid format")
    if payload["format"] != STU_MODEL_STATE_FORMAT:
        raise ModelError("converted STU checkpoint version is unsupported")
    if payload["conversion_rule"] != STU_MODEL_STATE_CONVERSION_RULE:
        raise ModelError("converted STU checkpoint uses an unsupported extraction")
    if payload["source_checkpoint_bytes"] != STU_CHECKPOINT_BYTES:
        raise ModelError("converted state refers to a different checkpoint size")
    if payload["source_checkpoint_sha256"] != STU_CHECKPOINT_SHA256:
        raise ModelError("converted state does not match the STU checkpoint")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ModelError("converted STU state_dict is empty")
    tensor_sha256 = model_state_sha256(state)
    if payload["tensor_sha256"] != STU_MODEL_STATE_TENSOR_SHA256:
        raise ModelError("converted STU checkpoint declares the wrong tensors")
    if tensor_sha256 != STU_MODEL_STATE_TENSOR_SHA256:
        raise ModelError("converted STU tensor content has the wrong identity")
    if _sha256(converted) != converted_sha256:
        raise ModelError("converted STU checkpoint changed while it was loaded")
    return state, {
        "checkpoint_bytes": source.stat().st_size,
        "checkpoint_sha256": source_sha256,
        "model_state_bytes": converted.stat().st_size,
        "model_state_sha256": converted_sha256,
        "model_state_tensor_sha256": tensor_sha256,
    }


def stu_weight_identity(checkpoint: Path | str) -> dict[str, object]:
    """Return the validated identities of both STU weight artifacts."""

    _, identity = _validated_stu_weights(checkpoint)
    return identity


def _official_modules(repository: Path | str) -> tuple[Any, Any, Any]:
    """Import the unmodified official STU package and its runtime dependencies."""

    resolved = Path(repository).expanduser().resolve(strict=True)
    if not (resolved / "models" / "mask4former.py").is_file():
        raise FileNotFoundError("STU repository lacks models/mask4former.py")
    loaded = sys.modules.get("models")
    if loaded is not None:
        location = Path(loaded.__file__).resolve()
        if resolved not in location.parents:
            raise ModelError("another top-level models package is already loaded")
    elif str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    try:
        models = importlib.import_module("models")
        me = importlib.import_module("MinkowskiEngine")
        omega = importlib.import_module("omegaconf")
    except ImportError as error:
        raise RuntimeError(
            "AJAE requires the official STU environment with MinkowskiEngine, "
            "Hydra, OmegaConf, and PyTorch3D"
        ) from error
    return models, me, omega


def _build_official_model(repository: Path | str) -> nn.Module:
    """Construct the exact released STU Mask4Former-3D module."""

    models, _, omega = _official_modules(repository)
    backbone = omega.OmegaConf.create(
        {
            "_target_": "models.Res16UNet34C",
            "config": {
                "dialations": [1, 1, 1, 1],
                "conv1_kernel_size": 5,
                "bn_momentum": 0.02,
            },
            "in_channels": 2,
            "out_channels": NUM_NORMAL_CLASSES,
        }
    )
    return models.Mask4Former3D(
        backbone=backbone,
        num_queries=NUM_QUERIES,
        num_heads=8,
        num_decoders=3,
        num_levels=4,
        sample_sizes=[4000, 8000, 16000, 32000],
        mask_dim=MASK_DIM,
        dim_feedforward=1024,
        num_labels=NUM_NORMAL_CLASSES,
    )


def load_stu_weights(model: nn.Module, checkpoint: Path | str) -> None:
    """Load the restricted tensor extraction after binding it to official bytes."""

    state, _ = _validated_stu_weights(checkpoint)
    model.load_state_dict(state, strict=True)


@dataclass(frozen=True, slots=True)
class STUVoxelEvidence:
    """Assigned STU evidence on the sparse voxel rows."""

    assigned_query: Tensor
    normal_evidence: Tensor
    reliability_assign: Tensor
    reliability_noobj: Tensor
    maxlogit_score: Tensor


def assigned_stu_evidence(
    pred_logits: Tensor, pred_masks: Tensor
) -> STUVoxelEvidence:
    """Assign one official query to each voxel and retain the official B0 score."""

    if pred_logits.ndim != 2 or pred_logits.shape != (
        NUM_QUERIES,
        NUM_NORMAL_CLASSES + 1,
    ):
        raise ModelError("STU pred_logits must be [100,20]")
    if pred_masks.ndim != 2 or pred_masks.shape[1] != NUM_QUERIES:
        raise ModelError("STU pred_masks must be [V,100]")
    if pred_masks.shape[0] == 0:
        raise ModelError("STU pred_masks must contain at least one voxel")
    if pred_logits.device != pred_masks.device:
        raise ModelError("STU query logits and masks must share a device")
    _finite("STU pred_logits", pred_logits)
    _finite("STU pred_masks", pred_masks)

    class_probability = pred_logits.softmax(dim=-1)
    normal_probability = class_probability[:, :NUM_NORMAL_CLASSES]
    mask_probability = pred_masks.sigmoid()
    query_normal_confidence = normal_probability.max(dim=1).values
    assignment_strength = mask_probability * query_normal_confidence[None, :]
    # torch.argmax returns the smallest query index for an exact tie.
    assigned_query = assignment_strength.argmax(dim=1)
    row = torch.arange(pred_masks.shape[0], device=pred_masks.device)
    assigned_mask = mask_probability[row, assigned_query]
    normal_evidence = assigned_mask[:, None] * normal_probability[assigned_query]
    reliability_assign = assignment_strength[row, assigned_query]
    reliability_noobj = class_probability[assigned_query, NUM_NORMAL_CLASSES]

    # This all-query aggregation is retained only as the released STU B0 baseline.
    official_confidence = mask_probability @ normal_probability
    maxlogit_score = 1.0 - official_confidence.max(dim=1).values
    for name, value in (
        ("STU assigned normal evidence", normal_evidence),
        ("STU assignment reliability", reliability_assign),
        ("STU no-object reliability", reliability_noobj),
        ("STU official MaxLogit score", maxlogit_score),
    ):
        _finite(name, value)
    return STUVoxelEvidence(
        assigned_query=assigned_query,
        normal_evidence=normal_evidence,
        reliability_assign=reliability_assign,
        reliability_noobj=reliability_noobj,
        maxlogit_score=maxlogit_score,
    )


@dataclass(frozen=True, slots=True)
class STUPointEncoding:
    """Frozen STU evidence restored to the supplied real-return order."""

    point_features: Tensor
    assigned_query: Tensor
    normal_evidence: Tensor
    reliability_assign: Tensor
    reliability_noobj: Tensor
    maxlogit_score: Tensor
    inverse_map: Tensor
    real_slots: Tensor
    input_identity: str

    def __post_init__(self) -> None:
        count = self.inverse_map.numel()
        if self.point_features.shape != (count, MASK_DIM):
            raise ModelError("STU point features must be [N,128]")
        if self.assigned_query.dtype != torch.long or self.assigned_query.shape != (
            count,
        ):
            raise ModelError("STU assigned query must be int64[N]")
        if bool(torch.any((self.assigned_query < 0) | (self.assigned_query >= NUM_QUERIES))):
            raise ModelError("STU assigned query is out of range")
        if self.normal_evidence.shape != (count, NUM_NORMAL_CLASSES):
            raise ModelError("STU normal evidence must be [N,19]")
        for name, value in (
            ("STU assignment reliability", self.reliability_assign),
            ("STU no-object reliability", self.reliability_noobj),
            ("STU MaxLogit score", self.maxlogit_score),
        ):
            if value.shape != (count,):
                raise ModelError(f"{name} must be [N]")
        if self.inverse_map.dtype != torch.long or self.inverse_map.shape != (count,):
            raise ModelError("STU inverse map must be int64[N]")
        if self.real_slots.dtype != torch.long or self.real_slots.shape != (count,):
            raise ModelError("STU real slots must be int64[N]")
        if (
            not isinstance(self.input_identity, str)
            or len(self.input_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.input_identity
            )
        ):
            raise ModelError("STU input identity must be a lowercase SHA-256 digest")
        if count == 0:
            raise ModelError("STU cannot encode an empty real-return set")
        if int(self.inverse_map.min()) < 0:
            raise ModelError("STU inverse map contains a negative row")
        devices = {
            self.point_features.device,
            self.assigned_query.device,
            self.normal_evidence.device,
            self.reliability_assign.device,
            self.reliability_noobj.device,
            self.maxlogit_score.device,
            self.inverse_map.device,
            self.real_slots.device,
        }
        if len(devices) != 1:
            raise ModelError("all STU point outputs must share a device")
        for name, value in (
            ("STU point features", self.point_features),
            ("STU normal evidence", self.normal_evidence),
            ("STU assignment reliability", self.reliability_assign),
            ("STU no-object reliability", self.reliability_noobj),
            ("STU MaxLogit score", self.maxlogit_score),
        ):
            _finite(name, value)
            if value.requires_grad:
                raise ModelError(f"{name} must be frozen")


def _numpy(value: np.ndarray | Tensor, *, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def stu_input_identity(
    coordinates: np.ndarray | Tensor,
    features: np.ndarray | Tensor,
    real_slots: np.ndarray | Tensor | None = None,
) -> str:
    """Bind frozen STU outputs to the exact effective single-scan inputs."""

    coordinates_np = np.ascontiguousarray(_numpy(coordinates, dtype=np.float64))
    features_np = np.ascontiguousarray(_numpy(features, dtype=np.float32))
    if coordinates_np.ndim != 2 or coordinates_np.shape[1] != 3:
        raise ModelError("STU identity coordinates must be [S,3]")
    if features_np.shape != (coordinates_np.shape[0], 2):
        raise ModelError("STU identity features must be [S,2]")
    slots_np = (
        np.arange(coordinates_np.shape[0], dtype=np.int64)
        if real_slots is None
        else np.ascontiguousarray(_numpy(real_slots, dtype=np.int64))
    )
    if (
        slots_np.ndim != 1
        or slots_np.size == 0
        or np.any(slots_np < 0)
        or np.any(slots_np >= coordinates_np.shape[0])
        or np.unique(slots_np).size != slots_np.size
    ):
        raise ModelError("STU identity real_slots must select distinct input rows")
    if not (
        np.isfinite(coordinates_np[slots_np]).all()
        and np.isfinite(features_np[slots_np]).all()
    ):
        raise ModelError("STU identity inputs contain non-finite values")

    digest = hashlib.sha256(b"AJAE-schema31-frozen-STU-input\0")
    for name, value in (
        (b"coordinates", coordinates_np[slots_np]),
        (b"features", features_np[slots_np]),
        (b"real_slots", slots_np),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenSTUPointEncoder(nn.Module):
    """Run official single-frame STU and restore frozen point-level evidence."""

    def __init__(
        self,
        *,
        checkpoint: Path | str = PROJECT_ROOT / "weights" / "59p6pq_ens1.ckpt",
        official_repository: Path | str = DEFAULT_STU_REPOSITORY,
        voxel_size: float = STU_VOXEL_SIZE_METRES,
    ) -> None:
        super().__init__()
        if not math.isclose(voxel_size, STU_VOXEL_SIZE_METRES, abs_tol=1e-12):
            raise ModelError("official STU voxel size must be exactly 0.05 m")
        self.official_repository = (
            Path(official_repository).expanduser().resolve(strict=True)
        )
        self.voxel_size = float(voxel_size)
        self.stu = _build_official_model(self.official_repository)
        load_stu_weights(self.stu, checkpoint)
        for parameter in self.stu.parameters():
            parameter.requires_grad_(False)
        self.stu.eval()

    @classmethod
    def from_protocol(
        cls,
        protocol: Any,
        *,
        project_root: Path | str = PROJECT_ROOT,
    ) -> FrozenSTUPointEncoder:
        if not hasattr(protocol, "stu") or not hasattr(protocol, "checkpoint_path"):
            raise TypeError("protocol must expose stu and checkpoint_path")
        stu = protocol.stu
        if int(stu["point_feature_dim"]) != MASK_DIM:
            raise ModelError("protocol STU point feature width must be 128")
        if int(stu["normal_evidence_dim"]) != NUM_NORMAL_CLASSES:
            raise ModelError("protocol STU normal evidence width must be 19")
        if (
            int(stu["assignment_reliability_dim"]) != 1
            or int(stu["no_object_reliability_dim"]) != 1
            or not bool(stu["frozen"])
            or stu["source"] != "STU_official_Mask4Former3D"
            or stu["b0_score"] != "official_STU_MaxLogit"
            or int(stu["checkpoint_bytes"]) != STU_CHECKPOINT_BYTES
            or stu["checkpoint_sha256"] != STU_CHECKPOINT_SHA256
        ):
            raise ModelError("protocol does not identify the frozen official STU")
        if not math.isclose(
            float(stu["voxel_size_m"]), STU_VOXEL_SIZE_METRES, abs_tol=1.0e-12
        ):
            raise ModelError("protocol STU voxel size must be exactly 0.05 m")
        root = Path(project_root).expanduser().resolve()
        repository = root / str(stu["repository"])
        checkpoint_path = getattr(protocol, "checkpoint_path", None)
        checkpoint = (
            checkpoint_path(root)
            if callable(checkpoint_path)
            else root / str(stu["checkpoint"])
        )
        return cls(
            checkpoint=checkpoint,
            official_repository=repository,
            voxel_size=float(stu["voxel_size_m"]),
        )

    @property
    def device(self) -> torch.device:
        return next(self.stu.parameters()).device

    def train(self, mode: bool = True) -> FrozenSTUPointEncoder:
        super().train(mode)
        self.stu.eval()
        return self

    @staticmethod
    def _single_prediction(value: Any, name: str) -> Tensor:
        if isinstance(value, (list, tuple)):
            if len(value) != 1 or not isinstance(value[0], Tensor):
                raise ModelError(f"STU {name} must contain one frame")
            return value[0]
        if isinstance(value, Tensor) and value.ndim == 3 and value.shape[0] == 1:
            return value[0]
        if isinstance(value, Tensor) and value.ndim == 2:
            return value
        raise ModelError(f"STU {name} has an unsupported single-frame format")

    def forward(
        self,
        coordinates: np.ndarray | Tensor,
        features: np.ndarray | Tensor,
        real_slots: np.ndarray | Tensor | None = None,
    ) -> STUPointEncoding:
        coordinates_np = _numpy(coordinates, dtype=np.float64)
        features_np = _numpy(features, dtype=np.float32)
        if coordinates_np.ndim != 2 or coordinates_np.shape[1] != 3:
            raise ModelError("STU coordinates must be [S,3]")
        if features_np.shape != (coordinates_np.shape[0], 2):
            raise ModelError("STU features must be [S,2]: intensity and distance")
        if not np.isfinite(coordinates_np).all() or not np.isfinite(features_np).all():
            raise ModelError("STU input contains non-finite values")

        if real_slots is None:
            slots_np = np.arange(coordinates_np.shape[0], dtype=np.int64)
        else:
            slots_np = _numpy(real_slots, dtype=np.int64)
            if slots_np.ndim != 1:
                raise ModelError("real_slots must be one-dimensional")
        if slots_np.size == 0:
            raise ModelError("STU needs at least one real return")
        if np.any(slots_np < 0) or np.any(slots_np >= coordinates_np.shape[0]):
            raise ModelError("real_slots contains an out-of-range source slot")
        if np.unique(slots_np).size != slots_np.size:
            raise ModelError("real_slots must identify distinct source returns")

        # Quantize only visible returns; inverse retains their supplied order.
        point_coordinates = coordinates_np[slots_np]
        point_features = features_np[slots_np]
        _, me, _ = _official_modules(self.official_repository)
        sparse_coordinates, sparse_features, unique, inverse = me.utils.sparse_quantize(
            coordinates=point_coordinates,
            features=point_features,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        if not isinstance(sparse_features, Tensor):
            sparse_features = torch.from_numpy(np.asarray(sparse_features))
        collated_coordinates, collated_features = me.utils.sparse_collate(
            [sparse_coordinates], [sparse_features.float()]
        )
        sparse = me.SparseTensor(
            coordinates=collated_coordinates,
            features=collated_features,
            device=self.device,
        )
        unique_np = _numpy(unique, dtype=np.int64)
        raw_coordinates = torch.from_numpy(
            np.column_stack(
                (
                    point_coordinates[unique_np],
                    np.zeros(unique_np.size, dtype=np.float64),
                )
            )
        ).float()
        raw_coordinates = raw_coordinates.to(self.device)

        captured: list[Any] = []

        def capture_point_features(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            captured.append(output)

        hook = self.stu.point_features_head.register_forward_hook(
            capture_point_features
        )
        try:
            self.stu.eval()
            with torch.no_grad():
                output = self.stu(sparse, raw_coordinates=raw_coordinates, is_eval=True)
        finally:
            hook.remove()

        if len(captured) != 1 or not hasattr(captured[0], "F"):
            raise ModelError(
                "STU point_features_head hook did not capture one sparse map"
            )
        if not isinstance(output, Mapping) or not {
            "pred_logits",
            "pred_masks",
        }.issubset(output):
            raise ModelError("STU forward output lacks query logits or masks")
        sparse_point_features = captured[0].F
        if (
            sparse_point_features.ndim != 2
            or sparse_point_features.shape[1] != MASK_DIM
        ):
            raise ModelError("STU point_features_head did not produce 128 channels")
        logits = self._single_prediction(output["pred_logits"], "pred_logits")
        masks = self._single_prediction(output["pred_masks"], "pred_masks")
        if masks.shape[0] != sparse_point_features.shape[0]:
            raise ModelError("STU masks and hooked sparse features do not align")
        sparse_evidence = assigned_stu_evidence(logits, masks)

        inverse_map = torch.as_tensor(inverse, dtype=torch.long, device=self.device)
        if inverse_map.shape != (slots_np.size,):
            raise ModelError("STU inverse map does not restore every real return")
        if int(inverse_map.max()) >= sparse_point_features.shape[0]:
            raise ModelError("STU inverse map exceeds the hooked sparse feature map")
        slots = torch.as_tensor(slots_np, dtype=torch.long, device=self.device)
        return STUPointEncoding(
            point_features=sparse_point_features[inverse_map].detach(),
            assigned_query=sparse_evidence.assigned_query[inverse_map].detach(),
            normal_evidence=sparse_evidence.normal_evidence[inverse_map].detach(),
            reliability_assign=sparse_evidence.reliability_assign[
                inverse_map
            ].detach(),
            reliability_noobj=sparse_evidence.reliability_noobj[
                inverse_map
            ].detach(),
            maxlogit_score=sparse_evidence.maxlogit_score[inverse_map].detach(),
            inverse_map=inverse_map,
            real_slots=slots,
            input_identity=stu_input_identity(
                coordinates_np, features_np, slots_np
            ),
        )


GROUPING_MODES = frozenset({"single", "per_scan", "joint"})


def _grouping_operation(
    scan_group: Tensor, count: int, device: torch.device, grouping_mode: str
) -> str:
    if grouping_mode not in GROUPING_MODES:
        raise ModelError("grouping_mode must be single, per_scan, or joint")
    if scan_group.dtype != torch.long or scan_group.shape != (count,):
        raise ModelError("scan_group must be int64[N]")
    if scan_group.device != device:
        raise ModelError("scan_group must share the point-tensor device")
    if bool(torch.any(scan_group < 0)):
        raise ModelError("scan_group must contain non-negative group labels")
    if grouping_mode == "single":
        if torch.unique(scan_group).numel() != 1:
            raise ModelError("single grouping requires exactly one scan group")
        return "joint"
    return grouping_mode


def _stable_candidate_order(
    global_index: np.ndarray,
    distance: np.ndarray,
    points: np.ndarray,
    tie_breaker: Tensor | None,
) -> np.ndarray:
    """Order a cutoff tie by intrinsic values, never by input row number."""

    columns = [
        np.asarray(distance, dtype=np.float64),
        points[global_index, 0],
        points[global_index, 1],
        points[global_index, 2],
    ]
    if tie_breaker is not None:
        index = torch.as_tensor(
            global_index, dtype=torch.long, device=tie_breaker.device
        )
        values = (
            tie_breaker.index_select(0, index)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        if values.ndim == 1:
            values = values[:, None]
        columns.extend(values[:, column] for column in range(values.shape[1]))
    # np.lexsort uses the last key as primary, hence the reversed tuple.
    return np.lexsort(tuple(reversed(columns)))


def _canonical_pool_order(
    keys: Tensor, coordinates: Tensor, features: Tensor
) -> Tensor:
    """Canonicalize every reduction sequence under input-row permutations."""

    key_values = keys.detach().cpu().numpy()
    point_values = coordinates.detach().to(device="cpu", dtype=torch.float64).numpy()
    columns = [
        *(key_values[:, column] for column in range(key_values.shape[1])),
        *(point_values[:, column] for column in range(point_values.shape[1])),
    ]
    order = np.lexsort(tuple(reversed(columns)))

    # Exact coincident returns are uncommon; use content only inside such ties.
    ordered_keys = key_values[order]
    ordered_points = point_values[order]
    same = np.zeros(order.size, dtype=np.bool_)
    same[1:] = np.all(ordered_keys[1:] == ordered_keys[:-1], axis=1) & np.all(
        ordered_points[1:] == ordered_points[:-1], axis=1
    )
    boundaries = np.flatnonzero(~same)
    stops = np.append(boundaries[1:], order.size)
    repeated = stops - boundaries > 1
    for start, stop in zip(boundaries[repeated], stops[repeated], strict=True):
        rows = order[start:stop]
        index = torch.as_tensor(rows, dtype=torch.long, device=features.device)
        values = (
            features.index_select(0, index)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        intrinsic = np.lexsort(
            tuple(reversed([values[:, column] for column in range(values.shape[1])]))
        )
        order[start:stop] = rows[intrinsic]
    return torch.as_tensor(order, dtype=torch.long, device=features.device)


@dataclass(frozen=True, slots=True)
class VoxelLevel:
    """One spatial pyramid level and its child-to-voxel assignment."""

    coordinates: Tensor
    scan_group: Tensor
    features: Tensor
    inverse_map: Tensor
    population: Tensor


class GroupedRadiusKNN(nn.Module):
    """Build one spatial radius neighborhood, optionally isolated by scan."""

    def __init__(self, radius: float, k: int, *, workers: int | None = None) -> None:
        super().__init__()
        selected_workers = KDTREE_WORKERS if workers is None else int(workers)
        if radius <= 0.0 or k <= 0 or selected_workers < 1:
            raise ModelError("neighbor radius, K, and workers must be positive")
        self.radius = float(radius)
        self.k = int(k)
        self.workers = selected_workers

    def forward(
        self,
        coordinates: Tensor,
        scan_group: Tensor,
        *,
        grouping_mode: str,
        tie_breaker: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = coordinates.shape[0]
        if coordinates.ndim != 2 or coordinates.shape != (count, 3) or count == 0:
            raise ModelError("neighbor coordinates must be non-empty [N,3]")
        operation = _grouping_operation(
            scan_group, count, coordinates.device, grouping_mode
        )
        if tie_breaker is not None:
            if (
                tie_breaker.ndim not in {1, 2}
                or tie_breaker.shape[0] != count
                or tie_breaker.device != coordinates.device
            ):
                raise ModelError("neighbor tie_breaker must be [N] or [N,D]")
            _finite("neighbor tie breaker", tie_breaker)
        _finite("neighbor coordinates", coordinates)

        points = (
            coordinates.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        groups = scan_group.detach().cpu().numpy().astype(np.int64, copy=False)
        neighbor = np.repeat(np.arange(count, dtype=np.int64)[:, None], self.k, axis=1)
        valid = np.zeros((count, self.k), dtype=np.bool_)
        uncapped_count = np.zeros(count, dtype=np.int64)
        partitions = (
            (np.arange(count, dtype=np.int64),)
            if operation == "joint"
            else tuple(np.flatnonzero(groups == group) for group in np.unique(groups))
        )
        strict_radius = np.nextafter(self.radius, -np.inf)

        for source_index in partitions:
            source_points = points[source_index]
            tree = cKDTree(source_points)
            local_count = np.asarray(
                tree.query_ball_point(
                    source_points,
                    strict_radius,
                    workers=self.workers,
                    return_length=True,
                ),
                dtype=np.int64,
            )
            uncapped_count[source_index] = local_count
            width = min(self.k + 1, source_index.size)
            distance, local_index = tree.query(
                source_points,
                k=width,
                distance_upper_bound=self.radius,
                workers=self.workers,
            )
            if width == 1:
                distance = distance[:, None]
                local_index = local_index[:, None]
            distance = np.asarray(distance, dtype=np.float64)
            local_index = np.asarray(local_index, dtype=np.int64)
            raw_valid = (
                np.isfinite(distance)
                & (distance < self.radius)
                & (local_index < source_index.size)
            )
            take = min(self.k, width)
            tied = np.zeros(source_index.size, dtype=np.bool_)
            if take > 1:
                tied |= np.any(
                    raw_valid[:, : take - 1]
                    & raw_valid[:, 1:take]
                    & np.isclose(
                        distance[:, : take - 1],
                        distance[:, 1:take],
                        rtol=0.0,
                        atol=1.0e-12,
                    ),
                    axis=1,
                )
            boundary_tie = np.zeros(source_index.size, dtype=np.bool_)
            if source_index.size > self.k:
                boundary_tie = (
                    raw_valid[:, self.k - 1]
                    & raw_valid[:, self.k]
                    & np.isclose(
                        distance[:, self.k - 1],
                        distance[:, self.k],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
                tied |= boundary_tie

            fast = ~tied
            if np.any(fast):
                row = source_index[fast]
                local = local_index[fast, :take]
                present = raw_valid[fast, :take]
                safe = np.minimum(local, source_index.size - 1)
                local_row, column = np.nonzero(present)
                neighbor[row[local_row], column] = source_index[
                    safe[local_row, column]
                ]
                valid[row[local_row], column] = True

            for local_row in np.flatnonzero(tied):
                present = raw_valid[local_row, :take]
                candidates = local_index[local_row, :take][present]
                candidate_distance = distance[local_row, :take][present]
                if boundary_tie[local_row]:
                    boundary = distance[local_row, self.k - 1]
                    candidates = np.asarray(
                        tree.query_ball_point(
                            source_points[local_row], np.nextafter(boundary, np.inf)
                        ),
                        dtype=np.int64,
                    )
                    candidate_distance = np.linalg.norm(
                        source_points[candidates] - source_points[local_row], axis=1
                    )
                    inside = candidate_distance < self.radius
                    candidates = candidates[inside]
                    candidate_distance = candidate_distance[inside]
                global_index = source_index[candidates]
                order = _stable_candidate_order(
                    global_index,
                    candidate_distance,
                    points,
                    tie_breaker,
                )[: self.k]
                selected = global_index[order]
                row = source_index[local_row]
                neighbor[row, : selected.size] = selected
                valid[row, : selected.size] = True

        expected = np.minimum(uncapped_count, self.k)
        if not np.array_equal(valid.sum(axis=1), expected):
            raise ModelError("radius neighborhood count disagrees with top-K selection")
        return (
            torch.as_tensor(neighbor, dtype=torch.long, device=coordinates.device),
            torch.as_tensor(valid, dtype=torch.bool, device=coordinates.device),
            torch.as_tensor(
                uncapped_count, dtype=torch.long, device=coordinates.device
            ),
        )


class PointInputProjection(nn.Module):
    """Project frozen STU content and symmetric-window spatial coordinates."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.content = nn.Sequential(
            nn.Linear(MASK_DIM + NUM_NORMAL_CLASSES + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coordinates: Tensor,
        stu_features: Tensor,
        normal_evidence: Tensor,
        reliability_assign: Tensor,
        reliability_noobj: Tensor,
        intensity: Tensor,
    ) -> Tensor:
        if reliability_assign.ndim == 1:
            reliability_assign = reliability_assign[:, None]
        if reliability_noobj.ndim == 1:
            reliability_noobj = reliability_noobj[:, None]
        if intensity.ndim == 1:
            intensity = intensity[:, None]
        content = torch.cat(
            (
                stu_features,
                normal_evidence,
                reliability_assign,
                reliability_noobj,
                intensity,
            ),
            dim=1,
        )
        return self.norm(
            self.content(content)
            + self.position(coordinates.to(dtype=content.dtype))
        )


class JointPointBlock(nn.Module):
    """Aggregate one grouped spatial neighborhood with explicit density."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        radius: float,
        neighbors: int,
        *,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ModelError("hidden dimension must be divisible by attention heads")
        if chunk_size <= 0:
            raise ModelError("attention chunk size must be positive")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.radius = float(radius)
        self.chunk_size = int(chunk_size)
        self.neighborhood = GroupedRadiusKNN(radius, neighbors)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, heads),
        )
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_count_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _message(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        coordinates: Tensor,
        neighbor: Tensor,
        valid: Tensor,
    ) -> Tensor:
        def chunk_message(
            local_query: Tensor,
            all_key: Tensor,
            all_value: Tensor,
            all_coordinates: Tensor,
            index: Tensor,
            local_valid: Tensor,
            local_coordinates: Tensor,
        ) -> Tensor:
            local_key = all_key[index]
            local_value = all_value[index]
            score = (local_query * local_key).sum(dim=-1) / math.sqrt(
                self.head_dim
            )
            displacement = (
                all_coordinates[index] - local_coordinates[:, None]
            ) / self.radius
            score = score + self.relative_bias(displacement)
            present = local_valid.any(dim=1)
            score = score.masked_fill(~local_valid[..., None], -torch.inf)
            score = torch.where(
                present[:, None, None], score, torch.zeros_like(score)
            )
            weight = F.softmax(score, dim=1) * local_valid[..., None]
            return (weight[..., None] * local_value).sum(dim=1).reshape(
                local_query.shape[0], self.hidden_dim
            )

        messages: list[Tensor] = []
        for start in range(0, query.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, query.shape[0])
            arguments = (
                query[start:stop, None],
                key,
                value,
                coordinates,
                neighbor[start:stop],
                valid[start:stop],
                coordinates[start:stop],
            )
            if self.training and torch.is_grad_enabled():
                # Recompute only local attention tensors during backward.
                message = checkpoint(
                    chunk_message,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                message = chunk_message(*arguments)
            messages.append(message)
        return torch.cat(messages, dim=0)

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        scan_group: Tensor,
        *,
        grouping_mode: str,
    ) -> Tensor:
        normalized = self.norm1(features)
        count = features.shape[0]
        query = self.query(normalized).view(count, self.heads, self.head_dim)
        key = self.key(normalized).view(count, self.heads, self.head_dim)
        value = self.value(normalized).view(count, self.heads, self.head_dim)
        neighbor, valid, uncapped_count = self.neighborhood(
            coordinates,
            scan_group,
            grouping_mode=grouping_mode,
            tie_breaker=normalized,
        )
        message = self._message(query, key, value, coordinates, neighbor, valid)
        density = torch.log1p(uncapped_count.to(dtype=features.dtype))[:, None]
        updated = (
            features
            + self.message_projection(message)
            + self.neighbor_count_projection(density)
        )
        return updated + self.ffn(self.norm2(updated))


class GroupedVoxelPool(nn.Module):
    """Pool joint xyz voxels or scan-isolated group-plus-xyz voxels."""

    def __init__(self, hidden_dim: int, voxel_size: float) -> None:
        super().__init__()
        if voxel_size <= 0.0:
            raise ModelError("voxel size must be positive")
        self.voxel_size = float(voxel_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        scan_group: Tensor,
        *,
        grouping_mode: str,
    ) -> VoxelLevel:
        operation = _grouping_operation(
            scan_group, features.shape[0], features.device, grouping_mode
        )
        quantized = torch.floor(coordinates / self.voxel_size).to(dtype=torch.long)
        keys = (
            torch.cat((scan_group[:, None], quantized), dim=1)
            if operation == "per_scan"
            else quantized
        )
        unique, inverse = torch.unique(
            keys, dim=0, sorted=True, return_inverse=True
        )
        voxel_count = unique.shape[0]
        population = torch.bincount(inverse, minlength=voxel_count)
        order = _canonical_pool_order(keys, coordinates, features)
        mean = torch.segment_reduce(
            features.index_select(0, order), "mean", lengths=population
        )
        maximum = torch.segment_reduce(
            features.index_select(0, order), "max", lengths=population
        )
        pooled_coordinates = torch.segment_reduce(
            coordinates.index_select(0, order), "mean", lengths=population
        )
        pooled_group = (
            unique[:, 0]
            if operation == "per_scan"
            else torch.zeros(voxel_count, dtype=torch.long, device=features.device)
        )
        density = torch.log1p(population.to(dtype=features.dtype))[:, None]
        pooled_features = self.projection(
            torch.cat((mean, maximum, density), dim=1)
        )
        return VoxelLevel(
            coordinates=pooled_coordinates,
            scan_group=pooled_group,
            features=pooled_features,
            inverse_map=inverse,
            population=population,
        )


class JointVoxelPyramid(nn.Module):
    """Encode raw returns and three grouped spatial voxel levels."""

    def __init__(
        self,
        hidden_dim: int,
        voxel_sizes: Sequence[float],
        neighbor_radii: Sequence[float],
        neighbor_k: Sequence[int],
        *,
        heads: int,
        attention_chunk_size: int,
    ) -> None:
        super().__init__()
        if len(voxel_sizes) != 3:
            raise ModelError("AJAE requires exactly three voxel sizes for L1-L3")
        if len(neighbor_radii) != 4 or len(neighbor_k) != 4:
            raise ModelError("AJAE requires exactly four spatial levels L0-L3")
        if any(right <= left for left, right in zip(voxel_sizes, voxel_sizes[1:])):
            raise ModelError("voxel sizes must increase from fine to coarse")
        if any(
            right <= left for left, right in zip(neighbor_radii, neighbor_radii[1:])
        ):
            raise ModelError("neighbor radii must increase from L0 through L3")
        self.pools = nn.ModuleList(
            GroupedVoxelPool(hidden_dim, size) for size in voxel_sizes
        )
        self.blocks = nn.ModuleList(
            JointPointBlock(
                hidden_dim,
                heads,
                radius,
                neighbors,
                chunk_size=attention_chunk_size,
            )
            for radius, neighbors in zip(
                neighbor_radii, neighbor_k, strict=True
            )
        )

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        scan_group: Tensor,
        *,
        grouping_mode: str,
    ) -> tuple[VoxelLevel, ...]:
        features = self.blocks[0](
            features,
            coordinates,
            scan_group,
            grouping_mode=grouping_mode,
        )
        levels = [
            VoxelLevel(
                coordinates=coordinates,
                scan_group=scan_group,
                features=features,
                inverse_map=torch.arange(
                    features.shape[0], dtype=torch.long, device=features.device
                ),
                population=torch.ones(
                    features.shape[0], dtype=torch.long, device=features.device
                ),
            )
        ]
        for pool, block in zip(self.pools, self.blocks[1:], strict=True):
            level = pool(
                features,
                coordinates,
                scan_group,
                grouping_mode=grouping_mode,
            )
            attended = block(
                level.features,
                level.coordinates,
                level.scan_group,
                grouping_mode=grouping_mode,
            )
            level = VoxelLevel(
                coordinates=level.coordinates,
                scan_group=level.scan_group,
                features=attended,
                inverse_map=level.inverse_map,
                population=level.population,
            )
            levels.append(level)
            features = attended
            coordinates = level.coordinates
            scan_group = level.scan_group
        return tuple(levels)


class GroupedKnnUpsample(nn.Module):
    """Interpolate from joint or scan-isolated coarse spatial nodes."""

    def __init__(self, k: int = 3, *, workers: int | None = None) -> None:
        super().__init__()
        selected_workers = KDTREE_WORKERS if workers is None else int(workers)
        if k <= 0 or selected_workers < 1:
            raise ModelError("upsampling K and workers must be positive")
        self.k = int(k)
        self.workers = selected_workers

    def forward(
        self,
        source_features: Tensor,
        source_coordinates: Tensor,
        source_group: Tensor,
        target_coordinates: Tensor,
        target_group: Tensor,
        *,
        grouping_mode: str,
    ) -> Tensor:
        source_count = source_coordinates.shape[0]
        target_count = target_coordinates.shape[0]
        if (
            source_coordinates.shape != (source_count, 3)
            or target_coordinates.shape != (target_count, 3)
            or source_features.ndim != 2
            or source_features.shape[0] != source_count
            or source_count == 0
            or target_count == 0
        ):
            raise ModelError("upsampling tensors have invalid non-empty point shapes")
        if not (
            source_features.device
            == source_coordinates.device
            == source_group.device
            == target_coordinates.device
            == target_group.device
        ):
            raise ModelError("upsampling tensors must share one device")
        source_operation = _grouping_operation(
            source_group, source_count, source_features.device, grouping_mode
        )
        target_operation = _grouping_operation(
            target_group, target_count, source_features.device, grouping_mode
        )
        if source_operation != target_operation:
            raise AssertionError("source and target grouping operations differ")
        operation = source_operation
        _finite("upsampling source features", source_features)
        _finite("upsampling source coordinates", source_coordinates)
        _finite("upsampling target coordinates", target_coordinates)

        source_points = (
            source_coordinates.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        target_points = (
            target_coordinates.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        source_groups = source_group.detach().cpu().numpy().astype(np.int64, copy=False)
        target_groups = target_group.detach().cpu().numpy().astype(np.int64, copy=False)
        if operation == "joint":
            partitions = ((
                np.arange(source_count, dtype=np.int64),
                np.arange(target_count, dtype=np.int64),
            ),)
        else:
            partitions_list: list[tuple[np.ndarray, np.ndarray]] = []
            for group in np.unique(target_groups):
                source_index = np.flatnonzero(source_groups == group)
                target_index = np.flatnonzero(target_groups == group)
                if source_index.size == 0:
                    raise ModelError("upsampling target group has no source nodes")
                partitions_list.append((source_index, target_index))
            partitions = tuple(partitions_list)

        output = source_features.new_zeros((target_count, source_features.shape[1]))
        for source_index, target_index in partitions:
            tree = cKDTree(source_points[source_index])
            width = min(self.k, source_index.size)
            query_width = min(self.k + 1, source_index.size)
            distance, local_index = tree.query(
                target_points[target_index],
                k=query_width,
                workers=self.workers,
            )
            if query_width == 1:
                distance = distance[:, None]
                local_index = local_index[:, None]
            distance = np.asarray(distance, dtype=np.float64)
            local_index = np.asarray(local_index, dtype=np.int64)
            selected_local = local_index[:, :width].copy()
            selected_distance = distance[:, :width].copy()
            tied = np.zeros(target_index.size, dtype=np.bool_)
            if width > 1:
                tied |= np.any(
                    np.isclose(
                        selected_distance[:, :-1],
                        selected_distance[:, 1:],
                        rtol=0.0,
                        atol=1.0e-12,
                    ),
                    axis=1,
                )
            boundary_tie = np.zeros(target_index.size, dtype=np.bool_)
            if source_index.size > self.k:
                boundary_tie = np.isclose(
                    distance[:, self.k - 1],
                    distance[:, self.k],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                tied |= boundary_tie
            for local_row in np.flatnonzero(tied):
                candidates = selected_local[local_row]
                candidate_distance = selected_distance[local_row]
                if boundary_tie[local_row]:
                    boundary = distance[local_row, self.k - 1]
                    candidates = np.asarray(
                        tree.query_ball_point(
                            target_points[target_index[local_row]],
                            np.nextafter(boundary, np.inf),
                        ),
                        dtype=np.int64,
                    )
                    candidate_distance = np.linalg.norm(
                        source_points[source_index[candidates]]
                        - target_points[target_index[local_row]],
                        axis=1,
                    )
                global_index = source_index[candidates]
                order = _stable_candidate_order(
                    global_index,
                    candidate_distance,
                    source_points,
                    source_features,
                )[:width]
                selected_local[local_row] = candidates[order]
                selected_distance[local_row] = candidate_distance[order]

            zero = selected_distance <= 1.0e-12
            weight = np.empty_like(selected_distance)
            rows_with_zero = zero.any(axis=1)
            if np.any(rows_with_zero):
                weight[rows_with_zero] = zero[rows_with_zero] / zero[
                    rows_with_zero
                ].sum(axis=1, keepdims=True)
            if np.any(~rows_with_zero):
                inverse_distance = 1.0 / selected_distance[~rows_with_zero]
                weight[~rows_with_zero] = inverse_distance / inverse_distance.sum(
                    axis=1, keepdims=True
                )
            neighbor = source_index[selected_local]
            neighbor_tensor = torch.as_tensor(
                neighbor, dtype=torch.long, device=source_features.device
            )
            weight_tensor = torch.as_tensor(
                weight, dtype=source_features.dtype, device=source_features.device
            )
            interpolated = (
                source_features[neighbor_tensor] * weight_tensor[..., None]
            ).sum(dim=1)
            output[
                torch.as_tensor(
                    target_index, dtype=torch.long, device=source_features.device
                )
            ] = interpolated
        return output


class PointAnomalyHead(nn.Module):
    """Map decoded high-resolution point features to one anomaly logit."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(1)


class JointWindowPointTransformer(nn.Module):
    """Four-level point model shared exactly by single, B2, and B3 paths."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        voxel_sizes: Sequence[float] = (0.1, 0.2, 0.4),
        neighbor_radii: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
        neighbor_k: Sequence[int] = (12, 16, 24, 32),
        heads: int = 4,
        upsample_k: int = 3,
        attention_chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        if hidden_dim < 16 or hidden_dim % heads:
            raise ModelError("hidden dimension must be at least 16 and divide heads")
        if upsample_k != 3:
            raise ModelError("AJAE requires grouped spatial 3-NN upsampling")
        self.input_projection = PointInputProjection(hidden_dim)
        self.pyramid = JointVoxelPyramid(
            hidden_dim,
            tuple(float(value) for value in voxel_sizes),
            tuple(float(value) for value in neighbor_radii),
            tuple(int(value) for value in neighbor_k),
            heads=heads,
            attention_chunk_size=attention_chunk_size,
        )
        self.upsample = GroupedKnnUpsample(upsample_k)
        self.decoder_fusions = nn.ModuleList(
            self._fusion(hidden_dim) for _ in range(len(voxel_sizes) - 1)
        )
        self.high_resolution_fusion = self._fusion(hidden_dim)
        self.anomaly_head = PointAnomalyHead(hidden_dim)

    @staticmethod
    def _fusion(hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    @classmethod
    def from_protocol(cls, protocol: Any) -> JointWindowPointTransformer:
        if not hasattr(protocol, "model"):
            raise TypeError("protocol must expose a model mapping")
        model = protocol.model
        expected_features = (
            "stu_point_feature_128d",
            "normal_evidence_19d",
            "assignment_reliability",
            "no_object_reliability",
            "intensity",
        )
        expected_forbidden = {
            "source_frame",
            "window_member_index",
            "relative_time",
            "absolute_time",
            "time_embedding",
            "reversible_time_encoding",
        }
        grouping = model.get("grouping_modes")
        radius = model.get("radius_neighbors")
        if int(model["input_dim"]) != MASK_DIM + NUM_NORMAL_CLASSES + 3:
            raise ModelError("protocol point input dimension must be 150")
        if tuple(model["input_features"]) != expected_features:
            raise ModelError("protocol point input components do not match AJAE")
        if not isinstance(grouping, Mapping) or dict(grouping) != {
            "B1": "single",
            "B2": "per_scan",
            "B3": "joint",
        }:
            raise ModelError("protocol grouping modes do not define B1, B2, and B3")
        if not isinstance(radius, Mapping):
            raise ModelError("protocol radius-neighbor geometry is missing")
        if (
            int(model["levels"]) != 4
            or len(model["voxel_sizes_m"]) != 3
            or tuple(model["voxel_feature"])
            != ("mean", "max", "log1p_population")
            or model["neighborhood_feature"]
            != "log1p_uncapped_count_of_all_points_strictly_inside_radius_before_top_K_selection"
            or not expected_forbidden.issubset(model["forbidden_features"])
            or model["B2_B3_shared_class_and_parameterization"] is not True
            or model["output"]
            != "one_anomaly_logit_for_every_visible_input_return"
            or model["scan_permutation_equivariant"] is not True
        ):
            raise ModelError("protocol does not define the joint-window model contract")
        return cls(
            hidden_dim=int(model["hidden_dim"]),
            voxel_sizes=tuple(float(value) for value in model["voxel_sizes_m"]),
            neighbor_radii=tuple(float(value) for value in radius["radii_m"]),
            neighbor_k=tuple(int(value) for value in radius["maximum_neighbors"]),
            heads=int(model["heads"]),
            upsample_k=int(model["upsample_neighbors"]),
        )

    @staticmethod
    def _validate_inputs(
        coordinates: Tensor,
        stu_features: Tensor,
        normal_evidence: Tensor,
        reliability_assign: Tensor,
        reliability_noobj: Tensor,
        intensity: Tensor,
        scan_group: Tensor,
        grouping_mode: str,
    ) -> None:
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ModelError("symmetric window coordinates must be [N,3]")
        count = coordinates.shape[0]
        if count == 0:
            raise ModelError("AJAE cannot score an empty window")
        if stu_features.shape != (count, MASK_DIM):
            raise ModelError("STU point features must be [N,128]")
        if normal_evidence.shape != (count, NUM_NORMAL_CLASSES):
            raise ModelError("normal evidence must be [N,19]")
        if reliability_assign.shape not in {(count,), (count, 1)}:
            raise ModelError("assignment reliability must be [N] or [N,1]")
        if reliability_noobj.shape not in {(count,), (count, 1)}:
            raise ModelError("no-object reliability must be [N] or [N,1]")
        if intensity.shape not in {(count,), (count, 1)}:
            raise ModelError("point intensity must be [N] or [N,1]")
        tensors = (
            coordinates,
            stu_features,
            normal_evidence,
            reliability_assign,
            reliability_noobj,
            intensity,
        )
        if any(not torch.is_floating_point(value) for value in tensors):
            raise ModelError("AJAE coordinate and content tensors must be floating point")
        if len({value.device for value in (*tensors, scan_group)}) != 1:
            raise ModelError("all AJAE point tensors must share a device")
        if len(
            {
                value.dtype
                for value in (
                    stu_features,
                    normal_evidence,
                    reliability_assign,
                    reliability_noobj,
                    intensity,
                )
            }
        ) != 1:
            raise ModelError("all AJAE content tensors must share a floating dtype")
        for name, value in (
            ("coordinates", coordinates),
            ("STU features", stu_features),
            ("normal evidence", normal_evidence),
            ("assignment reliability", reliability_assign),
            ("no-object reliability", reliability_noobj),
            ("intensity", intensity),
        ):
            _finite(name, value)
        _grouping_operation(scan_group, count, coordinates.device, grouping_mode)

    def forward(
        self,
        coordinates: Tensor,
        stu_features: Tensor,
        normal_evidence: Tensor,
        reliability_assign: Tensor,
        reliability_noobj: Tensor,
        intensity: Tensor,
        scan_group: Tensor,
        *,
        grouping_mode: str,
    ) -> Tensor:
        self._validate_inputs(
            coordinates,
            stu_features,
            normal_evidence,
            reliability_assign,
            reliability_noobj,
            intensity,
            scan_group,
            grouping_mode,
        )
        point_features = self.input_projection(
            coordinates,
            stu_features,
            normal_evidence,
            reliability_assign,
            reliability_noobj,
            intensity,
        )
        coordinates = coordinates.to(dtype=point_features.dtype)
        levels = self.pyramid(
            point_features,
            coordinates,
            scan_group,
            grouping_mode=grouping_mode,
        )

        decoded = levels[-1].features
        source = levels[-1]
        for fusion, target in zip(
            self.decoder_fusions, reversed(levels[1:-1]), strict=True
        ):
            decoded = self.upsample(
                decoded,
                source.coordinates,
                source.scan_group,
                target.coordinates,
                target.scan_group,
                grouping_mode=grouping_mode,
            )
            decoded = fusion(torch.cat((decoded, target.features), dim=1))
            source = target

        decoded = self.upsample(
            decoded,
            source.coordinates,
            source.scan_group,
            levels[0].coordinates,
            levels[0].scan_group,
            grouping_mode=grouping_mode,
        )
        decoded = self.high_resolution_fusion(
            torch.cat((decoded, levels[0].features), dim=1)
        )
        logits = self.anomaly_head(decoded)
        _finite("AJAE point logits", logits)
        return logits
