#!/usr/bin/env python3
"""Frozen official STU inference for single and dense pseudo-scans."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


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
    normal_class: Tensor


def assigned_stu_evidence(pred_logits: Tensor, pred_masks: Tensor) -> STUVoxelEvidence:
    """Assign one official query to each voxel and retain official MaxLogit."""

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

    # The released STU score aggregates all queries before taking MaxLogit.
    official_confidence = mask_probability @ normal_probability
    maxlogit_score = 1.0 - official_confidence.max(dim=1).values
    normal_class = official_confidence.argmax(dim=1)
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
        normal_class=normal_class,
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
    normal_class: Tensor
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
        if bool(
            torch.any((self.assigned_query < 0) | (self.assigned_query >= NUM_QUERIES))
        ):
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
        if self.normal_class.dtype != torch.long or self.normal_class.shape != (count,):
            raise ModelError("STU normal class must be int64[N]")
        if bool(
            torch.any(
                (self.normal_class < 0) | (self.normal_class >= NUM_NORMAL_CLASSES)
            )
        ):
            raise ModelError("STU normal class is out of range")
        if self.inverse_map.dtype != torch.long or self.inverse_map.shape != (count,):
            raise ModelError("STU inverse map must be int64[N]")
        if self.real_slots.dtype != torch.long or self.real_slots.shape != (count,):
            raise ModelError("STU real slots must be int64[N]")
        if (
            not isinstance(self.input_identity, str)
            or len(self.input_identity) != 64
            or any(
                character not in "0123456789abcdef" for character in self.input_identity
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
            self.normal_class.device,
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

    digest = hashlib.sha256(b"AJAE-schema33-frozen-STU-input\0")
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
            or stu["score"] != "official_STU_MaxLogit"
            or int(stu["checkpoint_bytes"]) != STU_CHECKPOINT_BYTES
            or stu["checkpoint_sha256"] != STU_CHECKPOINT_SHA256
            or stu["model_state_tensor_sha256"] != STU_MODEL_STATE_TENSOR_SHA256
        ):
            raise ModelError("protocol does not identify the frozen official STU")
        source_identity = stu_source_manifest(
            Path(project_root).expanduser().resolve() / str(stu["repository"])
        )["manifest_sha256"]
        if stu["source_manifest_sha256"] != source_identity:
            raise ModelError("protocol STU source manifest differs from the repository")
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
            reliability_assign=sparse_evidence.reliability_assign[inverse_map].detach(),
            reliability_noobj=sparse_evidence.reliability_noobj[inverse_map].detach(),
            maxlogit_score=sparse_evidence.maxlogit_score[inverse_map].detach(),
            normal_class=sparse_evidence.normal_class[inverse_map].detach(),
            inverse_map=inverse_map,
            real_slots=slots,
            input_identity=stu_input_identity(coordinates_np, features_np, slots_np),
        )
