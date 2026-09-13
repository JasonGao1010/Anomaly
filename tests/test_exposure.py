from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.exposure as exposure
from src.exposure import CELLS, SPARSE, summarize


def test_exposure_respects_class_means_warmup_and_unknown_geometry():
    config = dict(training=dict(batch_frames=2, warmup_steps=0, ramp_steps=2),
                  loss=dict(keep_weight=2.))
    base = dict(world="a", frame=1, source_identity="source1", parent="p1", regions=["r1", "r2"],
                flags={name: True for name in CELLS}, geometry_measured=True,
                normal=8, anomaly=2, keep=4, sparse_sides=dict(normal=2, anomaly=1, keep=1))
    rows = []
    for step in range(2):
        a, b = deepcopy(base), deepcopy(base)
        for r in (a, b):
            r.update(step=step, keep_active=step > 0)
        b.update(world="b", parent="p2", frame=2, source_identity="source2", anomaly=6,
                 sparse_sides=None, geometry_measured=False)
        b["flags"][SPARSE] = None
        rows.extend((a, b))
    result = summarize(rows, config, steps=2)
    sparse = result[SPARSE]
    assert rows[0]["coefficient_mass"][SPARSE]["keep"] == 0
    assert rows[1]["coefficient_mass"].get(SPARSE) is None
    assert sparse["coefficient_mass"] == pytest.approx(dict(normal=.125, anomaly=.125, keep=.125))
    assert sparse["active_keep_queries"] == 1
    assert sparse["matched_sides_both_queried"] == 2
    assert sparse["source_frames"] == 1
    assert result[CELLS[0]]["coefficient_mass"] == pytest.approx(dict(normal=1., anomaly=1., keep=1.))


def test_query_exposure_uses_original_slots_and_checks_source_identity(monkeypatch):
    source = SimpleNamespace(real_slots=np.array([1, 3, 5, 7]), xyzi=np.ones((8, 4)))
    frozen, original = SimpleNamespace(world_identity="world", source=source), SimpleNamespace(frame_id=3)
    queries = dict(query=torch.tensor([0, 1, 2, 3]), detection_index=torch.tensor([0, 2, 3]),
                   target=torch.tensor([0, 1, 0]), keep_slot=np.array([1, 7]), near_pairs=1)
    frames = SimpleNamespace(queries=lambda index, draw: (frozen, original, queries))
    row = dict(source_identity="source", range=8., in_range_rays=3, anomaly_rays=3, changed_native_rays=1)
    world = dict(world="name", parent="parent", regions=["0/0", "1/1"], height_m=.1,
                 relations={k: False for k in ("compact", "elongated", "sheet", "multi_branch", "multiple_contact")})
    contrast = dict(normal_queries=1, central_anomaly_queries=1, normal_source_slots=[7], central_anomaly_slots=[5])
    config = dict(training=dict(batch_frames=2), coverage=dict(low_height_m=.2,
                  background_change=dict(weak_max_changed_rays=4),
                  normal_contrasts=dict(minimum_normal_queries=1, minimum_central_anomaly_queries=1)))
    for name, value in dict(_frames=frames, _records={("world", 3): row}, _worlds={"world": world},
                            _geometry={("world", "source"): dict(contrasts=dict(sparse=contrast), changed_normal_source_slots=[1])},
                            _config=config).items():
        monkeypatch.setattr(exposure, name, value, raising=False)
    monkeypatch.setattr(exposure, "source_identity", lambda source: "source")
    measured = exposure._measure((11, 4, True))
    assert measured["sparse_sides"] == dict(normal=1, anomaly=1, keep=1)
    assert measured["sparse_detection_index"] == dict(normal=[2], anomaly=[1])
    assert measured["changed_normal"] == dict(normal=1, keep=1)
    assert measured["flags"][SPARSE] is True
    row["source_identity"] = "changed_source"
    with pytest.raises(ValueError, match="source identities"):
        exposure._measure((11, 4, True))
