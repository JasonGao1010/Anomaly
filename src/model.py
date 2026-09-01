#!/usr/bin/env python3
"""Frozen STU point evidence and the AJAE spatiotemporal point model."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STU_REPOSITORY = (
    PROJECT_ROOT.parent / "DynaCAN-deps" / "stu_dataset" / "Mask4Former3D"
)
NUM_NORMAL_CLASSES = 19
NUM_QUERIES = 100
MASK_DIM = 128
STU_VOXEL_SIZE_METRES = 0.05
RELATIVE_TIMES = (-2, -1, 0, 1, 2)
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
        if int(stu["query_count"]) != NUM_QUERIES or not bool(stu["frozen"]):
            raise ModelError("protocol must select the frozen 100-query STU")
        if tuple(stu["input_channels"]) != (
            "intensity",
            "official_STU_distance",
        ) or not bool(stu["full_forward_is_eval"]):
            raise ModelError(
                "protocol must retain the official STU input and eval path"
            )
        root = Path(project_root).expanduser().resolve()
        repository = root / str(stu["repository"])
        return cls(
            checkpoint=protocol.checkpoint_path(root),
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
        )


@dataclass(frozen=True, slots=True)
class VoxelLevel:
    """One time-preserving voxel level and its child-to-voxel assignment."""

    coordinates: Tensor
    relative_times: Tensor
    features: Tensor
    inverse_map: Tensor


def temporal_radius_knn(
    coordinates: Tensor,
    relative_times: Tensor,
    delta: int,
    radius: float,
    k: int,
    *,
    workers: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Find an independent radius-K neighborhood for one exact time difference."""

    count = coordinates.shape[0]
    if coordinates.ndim != 2 or coordinates.shape != (count, 3) or count == 0:
        raise ModelError("neighbor coordinates must be non-empty [N,3]")
    if relative_times.dtype != torch.long or relative_times.shape != (count,):
        raise ModelError("neighbor relative times must be int64[N]")
    if delta not in RELATIVE_TIMES:
        raise ModelError("neighbor time difference must be one of -2,-1,0,1,2")
    selected_workers = KDTREE_WORKERS if workers is None else int(workers)
    if radius <= 0 or k <= 0 or selected_workers < 1:
        raise ModelError("neighbor radius and K must be positive")
    if relative_times.device != coordinates.device:
        raise ModelError("neighbor coordinates and times must share a device")
    _finite("neighbor coordinates", coordinates)

    points = coordinates.detach().float().cpu().numpy().astype(np.float64, copy=False)
    times = relative_times.detach().cpu().numpy().astype(np.int64, copy=False)
    neighbor = np.repeat(np.arange(count, dtype=np.int64)[:, None], k, axis=1)
    valid = np.zeros((count, k), dtype=np.bool_)
    for query_time in RELATIVE_TIMES:
        query_index = np.flatnonzero(times == query_time)
        source_index = np.flatnonzero(times == query_time + delta)
        if query_index.size == 0 or source_index.size == 0:
            continue
        # Ask for one extra row so a distance tie at the K boundary can be
        # resolved by the stable source-row identity instead of KD-tree order.
        width = min(k + 1, int(source_index.size))
        tree = cKDTree(points[source_index])
        distance, local_index = tree.query(
            points[query_index],
            k=width,
            distance_upper_bound=float(radius),
            workers=selected_workers,
        )
        if width == 1:
            distance = distance[:, None]
            local_index = local_index[:, None]
        raw_valid = (
            np.isfinite(distance)
            & (distance < float(radius))
            & (local_index < source_index.size)
        )
        tied = np.zeros(query_index.size, dtype=np.bool_)
        if width > 1:
            tied = np.any(
                raw_valid[:, :-1]
                & raw_valid[:, 1:]
                & np.isclose(
                    distance[:, :-1], distance[:, 1:], rtol=0.0, atol=1.0e-12
                ),
                axis=1,
            )
        # Strictly ordered rows already have the exact lexicographic result.
        # Vectorizing them avoids one Python iteration per LiDAR return.
        fast = ~tied
        take = min(k, width)
        if np.any(fast):
            fast_local = local_index[fast, :take]
            fast_valid = raw_valid[fast, :take]
            safe_local = np.minimum(fast_local, source_index.size - 1)
            target_rows = query_index[fast]
            local_row, local_column = np.nonzero(fast_valid)
            neighbor[target_rows[local_row], local_column] = source_index[
                safe_local[local_row, local_column]
            ]
            valid[target_rows[local_row], local_column] = True
        for row in np.flatnonzero(tied):
            point = points[query_index[row]]
            row_distance = np.asarray(distance[row], dtype=np.float64)
            row_local = np.asarray(local_index[row], dtype=np.int64)
            row_valid = raw_valid[row]
            row_distance = row_distance[row_valid]
            row_local = row_local[row_valid]
            if row_local.size > k and np.isclose(
                row_distance[k - 1], row_distance[k], rtol=0.0, atol=1.0e-12
            ):
                tied_local = np.asarray(
                    tree.query_ball_point(
                        point, np.nextafter(row_distance[k - 1], np.inf)
                    ),
                    dtype=np.int64,
                )
                row_local = tied_local
                row_distance = np.linalg.norm(
                    points[source_index[tied_local]] - point, axis=1
                )
                inside = row_distance < float(radius)
                row_local = row_local[inside]
                row_distance = row_distance[inside]
            global_index = source_index[row_local]
            selected_order = np.lexsort((global_index, row_distance))[:k]
            selected = global_index[selected_order]
            count_selected = selected.size
            neighbor[query_index[row], :count_selected] = selected
            valid[query_index[row], :count_selected] = True
    return (
        torch.as_tensor(neighbor, dtype=torch.long, device=coordinates.device),
        torch.as_tensor(valid, dtype=torch.bool, device=coordinates.device),
    )


class PointInputProjection(nn.Module):
    """Project STU evidence and add center-frame geometry and time embeddings."""

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
        self.time = nn.Embedding(len(RELATIVE_TIMES), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coordinates: Tensor,
        relative_times: Tensor,
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
        dtype = content.dtype
        time_index = relative_times.to(dtype=torch.long) - RELATIVE_TIMES[0]
        return self.norm(
            self.content(content)
            + self.position(coordinates.to(dtype=dtype))
            + self.time(time_index)
        )


class TemporalPointBlock(nn.Module):
    """Independent temporal branches with rejectable cross-frame evidence."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        radii: Sequence[float],
        neighbors: Sequence[int],
        *,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ModelError("hidden dimension must be divisible by attention heads")
        if len(radii) != len(RELATIVE_TIMES) or len(neighbors) != len(
            RELATIVE_TIMES
        ):
            raise ModelError("each level requires five temporal radius-K branches")
        if (
            any(float(radius) <= 0.0 for radius in radii)
            or any(int(k) <= 0 for k in neighbors)
            or chunk_size <= 0
        ):
            raise ModelError("attention geometry and chunk size must be positive")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.radii = tuple(float(radius) for radius in radii)
        self.neighbor_counts = tuple(int(k) for k in neighbors)
        self.chunk_size = int(chunk_size)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, heads),
        )
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def neighbors(
        self, coordinates: Tensor, relative_times: Tensor, delta: int
    ) -> tuple[Tensor, Tensor]:
        branch = RELATIVE_TIMES.index(delta)
        return temporal_radius_knn(
            coordinates,
            relative_times,
            delta,
            self.radii[branch],
            self.neighbor_counts[branch],
        )

    def _message(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        coordinates: Tensor,
        neighbor: Tensor,
        valid: Tensor,
        *,
        radius: float,
        delta: int,
    ) -> Tensor:
        messages: list[Tensor] = []
        count = query.shape[0]
        for start in range(0, count, self.chunk_size):
            stop = min(start + self.chunk_size, count)
            index = neighbor[start:stop]
            local_valid = valid[start:stop]
            local_key = key[index]
            local_value = value[index]
            local_query = query[start:stop, None]
            score = (local_query * local_key).sum(dim=-1) / math.sqrt(self.head_dim)
            relative_position = (
                coordinates[index] - coordinates[start:stop, None]
            ) / radius
            delta_channel = relative_position.new_full(
                (*relative_position.shape[:-1], 1), float(delta) / 2.0
            )
            score = score + self.relative_bias(
                torch.cat((relative_position, delta_channel), dim=-1)
            )
            present = local_valid.any(dim=1)
            score = score.masked_fill(~local_valid[..., None], -torch.inf)
            # Avoid an undefined all-masked softmax, then zero invalid weights.
            score = torch.where(present[:, None, None], score, torch.zeros_like(score))
            weight = F.softmax(score, dim=1) * local_valid[..., None]
            message = (weight[..., None] * local_value).sum(dim=1)
            messages.append(message.reshape(stop - start, self.hidden_dim))
        return torch.cat(messages, dim=0)

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        relative_times: Tensor,
        *,
        cross_frame_enabled: bool,
    ) -> Tensor:
        if not isinstance(cross_frame_enabled, bool):
            raise TypeError("cross_frame_enabled must be boolean")
        normalized = self.norm1(features)
        count = features.shape[0]
        query = self.query(normalized).view(count, self.heads, self.head_dim)
        key = self.key(normalized).view(count, self.heads, self.head_dim)
        value = self.value(normalized).view(count, self.heads, self.head_dim)
        same_neighbor, same_valid = self.neighbors(coordinates, relative_times, 0)
        same_message = self._message(
            query,
            key,
            value,
            coordinates,
            same_neighbor,
            same_valid,
            radius=self.radii[RELATIVE_TIMES.index(0)],
            delta=0,
        )
        updated = features + self.message_projection(same_message)

        if cross_frame_enabled:
            for delta in RELATIVE_TIMES:
                if delta == 0:
                    continue
                branch = RELATIVE_TIMES.index(delta)
                neighbor, valid = self.neighbors(
                    coordinates, relative_times, delta
                )
                message = self._message(
                    query,
                    key,
                    value,
                    coordinates,
                    neighbor,
                    valid,
                    radius=self.radii[branch],
                    delta=delta,
                )
                present = valid.any(dim=1)
                delta_feature = features.new_full((count, 1), float(delta) / 2.0)
                gate = torch.sigmoid(
                    self.cross_gate(
                        torch.cat((features, message, delta_feature), dim=1)
                    )
                ).squeeze(1)
                gate = torch.where(present, gate, torch.zeros_like(gate))
                updated = updated + gate[:, None] * self.message_projection(message)
        return updated + self.ffn(self.norm2(updated))


class VoxelPool(nn.Module):
    """Pool each time slice independently with concatenated mean and max."""

    def __init__(self, hidden_dim: int, voxel_size: float) -> None:
        super().__init__()
        if voxel_size <= 0:
            raise ModelError("voxel size must be positive")
        self.voxel_size = float(voxel_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, features: Tensor, coordinates: Tensor, relative_times: Tensor
    ) -> VoxelLevel:
        quantized = torch.floor(coordinates / self.voxel_size).to(dtype=torch.long)
        # Time is part of the key, so coincident points from different frames never merge.
        keys = torch.cat((relative_times[:, None].to(torch.long), quantized), dim=1)
        unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        count = unique.shape[0]
        width = features.shape[1]
        index = inverse[:, None].expand(-1, width)
        totals = features.new_zeros((count, width))
        totals.scatter_add_(0, index, features)
        population = features.new_zeros((count, 1))
        population.scatter_add_(
            0, inverse[:, None], features.new_ones((features.shape[0], 1))
        )
        mean = totals / population
        maximum = features.new_full((count, width), -torch.inf)
        maximum.scatter_reduce_(0, index, features, reduce="amax", include_self=True)
        coordinate_totals = coordinates.new_zeros((count, 3))
        coordinate_totals.scatter_add_(0, inverse[:, None].expand(-1, 3), coordinates)
        pooled_coordinates = coordinate_totals / population.to(coordinates.dtype)
        return VoxelLevel(
            coordinates=pooled_coordinates,
            relative_times=unique[:, 0],
            features=self.projection(torch.cat((mean, maximum), dim=1)),
            inverse_map=inverse,
        )


class VoxelPyramid(nn.Module):
    """Encode fixed L0 raw points and three time-preserving voxel levels."""

    def __init__(
        self,
        hidden_dim: int,
        voxel_sizes: Sequence[float],
        neighbor_radii: Sequence[Sequence[float]],
        neighbor_k: Sequence[Sequence[int]],
        *,
        heads: int,
        attention_chunk_size: int,
    ) -> None:
        super().__init__()
        if len(voxel_sizes) != 3:
            raise ModelError("AJAE requires exactly three voxel sizes for L1-L3")
        if len(neighbor_radii) != 4 or len(neighbor_k) != 4:
            raise ModelError("AJAE requires exactly four attention levels L0-L3")
        if any(len(row) != 5 for row in neighbor_radii) or any(
            len(row) != 5 for row in neighbor_k
        ):
            raise ModelError("each attention level requires five temporal branches")
        if any(right <= left for left, right in zip(voxel_sizes, voxel_sizes[1:])):
            raise ModelError("voxel sizes must increase from fine to coarse")
        for branch in range(len(RELATIVE_TIMES)):
            radii = tuple(float(level[branch]) for level in neighbor_radii)
            if any(right <= left for left, right in zip(radii, radii[1:])):
                raise ModelError("attention radii must increase from L0 through L3")
        self.pools = nn.ModuleList(VoxelPool(hidden_dim, size) for size in voxel_sizes)
        self.blocks = nn.ModuleList(
            TemporalPointBlock(
                hidden_dim,
                heads,
                radii,
                counts,
                chunk_size=attention_chunk_size,
            )
            for radii, counts in zip(neighbor_radii, neighbor_k, strict=True)
        )

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        relative_times: Tensor,
        *,
        cross_frame_enabled: bool,
    ) -> tuple[VoxelLevel, ...]:
        features = self.blocks[0](
            features,
            coordinates,
            relative_times,
            cross_frame_enabled=cross_frame_enabled,
        )
        levels = [
            VoxelLevel(
                coordinates=coordinates,
                relative_times=relative_times,
                features=features,
                inverse_map=torch.arange(
                    features.shape[0], dtype=torch.long, device=features.device
                ),
            )
        ]
        for pool, block in zip(self.pools, self.blocks[1:], strict=True):
            level = pool(features, coordinates, relative_times)
            attended = block(
                level.features,
                level.coordinates,
                level.relative_times,
                cross_frame_enabled=cross_frame_enabled,
            )
            level = VoxelLevel(
                coordinates=level.coordinates,
                relative_times=level.relative_times,
                features=attended,
                inverse_map=level.inverse_map,
            )
            levels.append(level)
            features = attended
            coordinates = level.coordinates
            relative_times = level.relative_times
        return tuple(levels)


class KnnUpsample(nn.Module):
    """Interpolate geometrically from coarse nodes without crossing time slices."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        if k <= 0:
            raise ModelError("upsampling K must be positive")
        self.k = int(k)

    def forward(
        self,
        source_features: Tensor,
        source_coordinates: Tensor,
        source_times: Tensor,
        target_coordinates: Tensor,
        target_times: Tensor,
    ) -> Tensor:
        output = source_features.new_zeros(
            (target_coordinates.shape[0], source_features.shape[1])
        )
        for relative_time in torch.unique(target_times, sorted=True).tolist():
            source_index = torch.nonzero(
                source_times == relative_time, as_tuple=False
            ).flatten()
            target_index = torch.nonzero(
                target_times == relative_time, as_tuple=False
            ).flatten()
            if source_index.numel() == 0:
                raise ModelError(
                    "kNN upsampling found a target frame without source nodes"
                )
            source_points = (
                source_coordinates[source_index]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            target_points = (
                target_coordinates[target_index]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            width = min(self.k, source_points.shape[0])
            distance, local_index = cKDTree(source_points).query(
                target_points, k=width, workers=KDTREE_WORKERS
            )
            if width == 1:
                distance = distance[:, None]
                local_index = local_index[:, None]
            weight = 1.0 / np.maximum(distance, 1e-8)
            weight /= weight.sum(axis=1, keepdims=True)
            neighbor = source_index[
                torch.as_tensor(
                    local_index, dtype=torch.long, device=source_index.device
                )
            ]
            weight_tensor = torch.as_tensor(
                weight, dtype=source_features.dtype, device=source_features.device
            )
            interpolated = (source_features[neighbor] * weight_tensor[..., None]).sum(
                dim=1
            )
            output[target_index] = interpolated
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


class AJAEPointTransformer(nn.Module):
    """Fixed four-level AJAE point model for single- or five-frame conditions."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        voxel_sizes: Sequence[float] = (0.1, 0.2, 0.4),
        neighbor_radii: Sequence[Sequence[float]] = (
            (0.45, 0.35, 0.25, 0.35, 0.45),
            (0.90, 0.70, 0.50, 0.70, 0.90),
            (1.80, 1.40, 1.00, 1.40, 1.80),
            (3.60, 2.80, 2.00, 2.80, 3.60),
        ),
        neighbor_k: Sequence[Sequence[int]] = (
            (6, 8, 12, 8, 6),
            (8, 12, 16, 12, 8),
            (12, 16, 24, 16, 12),
            (16, 24, 32, 24, 16),
        ),
        heads: int = 4,
        upsample_k: int = 3,
        attention_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if hidden_dim < 16 or hidden_dim % heads:
            raise ModelError("hidden dimension must be at least 16 and divide heads")
        if upsample_k != 3:
            raise ModelError("AJAE requires same-frame 3-NN upsampling")
        self.input_projection = PointInputProjection(hidden_dim)
        self.pyramid = VoxelPyramid(
            hidden_dim,
            tuple(float(value) for value in voxel_sizes),
            tuple(tuple(float(value) for value in row) for row in neighbor_radii),
            tuple(tuple(int(value) for value in row) for row in neighbor_k),
            heads=heads,
            attention_chunk_size=attention_chunk_size,
        )
        self.upsample = KnnUpsample(upsample_k)
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
    def from_protocol(cls, protocol: Any) -> AJAEPointTransformer:
        if not hasattr(protocol, "model"):
            raise TypeError("protocol must expose a model mapping")
        model = protocol.model
        expected_input = MASK_DIM + NUM_NORMAL_CLASSES + 3
        if int(model["input_dim"]) != expected_input:
            raise ModelError("protocol point input dimension must be 150")
        if (
            int(model["stu_feature_dim"]) != MASK_DIM
            or int(model["normal_evidence_dim"]) != NUM_NORMAL_CLASSES
            or int(model["input_intensity_dim"]) != 1
        ):
            raise ModelError("protocol point input components do not match AJAE")
        if model["pooling"] != "per_time_mean_max":
            raise ModelError("AJAE has one authoritative mean-max pooling path")
        if model["upsample"] != "same_time_3NN_with_high_resolution_skip":
            raise ModelError("AJAE has one authoritative same-frame kNN decoder")
        if int(model["levels"]) != 4 or len(model["voxel_sizes_m"]) != 3:
            raise ModelError("protocol must define fixed L0 plus voxel levels L1-L3")
        radii = tuple(
            tuple(float(value) for value in row)
            for row in model["attention_radii_m"]
        )
        neighbors = tuple(
            tuple(int(value) for value in row) for row in model["neighbors"]
        )
        if len(radii) != 4 or len(neighbors) != 4 or any(
            len(row) != 5 for row in (*radii, *neighbors)
        ):
            raise ModelError("protocol attention geometry must be 4x5")
        return cls(
            hidden_dim=int(model["hidden_dim"]),
            voxel_sizes=tuple(model["voxel_sizes_m"]),
            neighbor_radii=radii,
            neighbor_k=neighbors,
            heads=int(model["heads"]),
            upsample_k=int(model["upsample_neighbors"]),
        )

    @staticmethod
    def _validate_inputs(
        coordinates: Tensor,
        relative_times: Tensor,
        stu_features: Tensor,
        normal_evidence: Tensor,
        reliability_assign: Tensor,
        reliability_noobj: Tensor,
        intensity: Tensor,
    ) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ModelError("center-frame coordinates must be [N,3]")
        count = coordinates.shape[0]
        if count == 0:
            raise ModelError("AJAE cannot score an empty window")
        if relative_times.shape != (count,):
            raise ModelError("relative times must be [N]")
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
        devices = {
            coordinates.device,
            relative_times.device,
            stu_features.device,
            normal_evidence.device,
            reliability_assign.device,
            reliability_noobj.device,
            intensity.device,
        }
        if len(devices) != 1:
            raise ModelError("all AJAE point tensors must share a device")
        for name, value in (
            ("coordinates", coordinates),
            ("relative_times", relative_times),
            ("STU features", stu_features),
            ("normal evidence", normal_evidence),
            ("assignment reliability", reliability_assign),
            ("no-object reliability", reliability_noobj),
            ("intensity", intensity),
        ):
            _finite(name, value)
        rounded = relative_times.round()
        if not bool(torch.equal(relative_times, rounded)):
            raise ModelError("relative times must be exact integer positions")
        integer_times = rounded.to(dtype=torch.long)
        observed = tuple(torch.unique(integer_times, sorted=True).tolist())
        if observed not in {(0,), RELATIVE_TIMES}:
            raise ModelError("AJAE input must be single-frame q=0 or complete q=-2..2")
        return integer_times

    def forward(
        self,
        coordinates: Tensor,
        relative_times: Tensor,
        stu_features: Tensor,
        normal_evidence: Tensor,
        reliability_assign: Tensor,
        reliability_noobj: Tensor,
        intensity: Tensor,
        *,
        cross_frame_enabled: bool = True,
    ) -> Tensor:
        integer_times = self._validate_inputs(
            coordinates,
            relative_times,
            stu_features,
            normal_evidence,
            reliability_assign,
            reliability_noobj,
            intensity,
        )
        point_features = self.input_projection(
            coordinates,
            integer_times,
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
            integer_times,
            cross_frame_enabled=cross_frame_enabled,
        )

        decoded = levels[-1].features
        source = levels[-1]
        for fusion, target in zip(
            self.decoder_fusions, reversed(levels[1:-1]), strict=True
        ):
            decoded = self.upsample(
                decoded,
                source.coordinates,
                source.relative_times,
                target.coordinates,
                target.relative_times,
            )
            decoded = fusion(torch.cat((decoded, target.features), dim=1))
            source = target

        decoded = self.upsample(
            decoded,
            source.coordinates,
            source.relative_times,
            levels[0].coordinates,
            levels[0].relative_times,
        )
        decoded = self.high_resolution_fusion(
            torch.cat((decoded, levels[0].features), dim=1)
        )
        logits = self.anomaly_head(decoded)
        _finite("AJAE point logits", logits)
        return logits
