"""Check original source observations and exact reviewed normal annotations."""

import numpy as np
import pytest


@pytest.fixture
def normal_sample(monkeypatch):
    import torch
    from src import data

    monkeypatch.setattr("src.model.voxelize", lambda points: dict(xyzi=torch.from_numpy(points.copy())))

    def sample(allowed, *, queries=4096, seed=206, key=0, augment=False):
        points = np.zeros((len(allowed), 4), dtype=np.float32)
        points[:, 0] = np.linspace(5, 25, len(points))
        points[:, 1], points[:, 3] = .3, .5
        monkeypatch.setattr(data, "read_normal_record", lambda record:
            dict(xyzi=points.copy(), allowed=allowed.copy(), slots=np.arange(len(points)), slot_count=len(points)))
        return data.NormalScans([{}], queries=queries, seed=seed, augment=augment)[key]

    return sample


def test_background_split_preserves_full_context_and_normal_only_labels(tmp_path, monkeypatch):
    import json
    from src import nuscenes
    from src.data import read_nuscenes, point_targets, load_manifest

    meta = tmp_path / "v1.0-trainval"
    meta.mkdir()
    (meta / "log.json").write_text(json.dumps([dict(token=s, location="test") for s in ("train", "val")]))
    mapping = [dict(raw=i, name=str(i), target=int(i == 24)) for i in range(32)]
    raw = np.array([[2.5, 0, 0, 255, 0], [50, 0, 0, 10, 1], [50.1, 0, 0, 5, 2],
                    [5, 0, 0, 80, 3], [6, 0, 0, 90, 4], [0, 0, 0, 0, 5]], np.float32)
    records = {}
    for split in ("train", "val"):
        scan, label = tmp_path / f"{split}.bin", tmp_path / f"{split}.label"
        raw.tofile(scan)
        np.array([24, 24, 24, 10, 11, 24], np.uint8).tofile(label)
        records[split] = [dict(source="nuscenes", scene=split, log_token=split, token=split,
            sample_token=split, timestamp=0, frame=0, scan=str(scan), label=str(label),
            group="normal_nuscenes", subset=split, pose=np.eye(4).tolist())]
    monkeypatch.setattr(nuscenes, "sources", lambda root: (records, mapping))
    output = tmp_path / "background"
    nuscenes.build(tmp_path, output, 1)
    assert {p.name for p in output.iterdir()} == {"train.json", "val.json"}
    for split in ("train", "val"):
        manifest = load_manifest(output / f"{split}.json", split)
        assert manifest["summary"]["normal"] == 2
        assert manifest["summary"]["ignored_in_range"] == 2
        assert manifest["summary"]["outside_range"] == 1
        assert manifest["summary"]["empty_slots"] == 1
        assert not manifest["records"][0]["eligible"]
        sample = read_nuscenes(manifest["records"][0], mapping)
        np.testing.assert_array_equal(sample.xyzi[sample.actual, :3], raw[:5, :3])
        np.testing.assert_array_equal(point_targets(sample)[sample.actual], [0, 0, -1, -1, -1])
    with pytest.raises(ValueError, match="empty output"):
        nuscenes.build(tmp_path, output, 1)


def test_reviewed_normals_are_point_specific(tmp_path):
    from src.data import read_nuscenes

    raw = np.array([[5., 0., 0., 100., 0.], [6., 0., 0., 90., 1.],
                    [7., 0., 0., 80., 2.]], np.float32)
    scan, label = tmp_path / 'scan.bin', tmp_path / 'label.bin'
    raw.tofile(scan)
    np.array([0, 0, 1], np.uint8).tofile(label)
    mapping = [dict(raw=0, name='static.manmade', target=0),
               dict(raw=1, name='flat.driveable_surface', target=1)]
    record = dict(scan=str(scan), label=str(label), token='native', frame=0, normal_slots=[0])
    np.testing.assert_array_equal(read_nuscenes(record, mapping).labels, [1, 0, 1])
    for slots in ([2], [0, 0], [-1], [3]):
        with pytest.raises(ValueError, match='supplemental normal'):
            read_nuscenes(dict(record, normal_slots=slots), mapping)


def test_queries_balance_distinct_allowed_sets_including_rare_singletons(normal_sample):
    allowed = np.zeros((20008, 19), dtype=bool)
    allowed[:10000, 8:12] = True
    allowed[10000:20000, 14:16] = True
    allowed[20000:, 9] = True
    sample = normal_sample(allowed, queries=18)
    chosen = sample["queries"].numpy()
    # Different coarse-set cardinalities must not multiply their sampling quota.
    np.testing.assert_array_equal(np.bincount(np.searchsorted([10000, 20000], chosen, side="right")), [6, 6, 6])
    assert len(chosen) == len(np.unique(chosen)) == 18
    np.testing.assert_array_equal(sample["allowed"], allowed)
    assert len(sample["xyzi"]) == len(allowed)


@pytest.mark.parametrize("queries", [4096, 30000])
def test_queries_fill_budget_once_and_keep_all_rare_points_when_quota_permits(normal_sample, queries):
    allowed = np.zeros((20040, 19), dtype=bool)
    allowed[:10000, 8:12] = True
    allowed[10000:20000, 14:16] = True
    allowed[20000:20008, 9] = True
    sample = normal_sample(allowed, queries=queries)
    chosen = sample["queries"].numpy()
    assert len(chosen) == len(np.unique(chosen)) == min(queries, 20008)
    assert np.isin(np.arange(20000, 20008), chosen).all()
    assert np.all(np.diff(chosen) > 0) and np.all(chosen < 20008)
    assert len(sample["xyzi"]) == 20040  # Ignored points remain model input.


def test_empty_supervision_preserves_input_and_has_no_queries(normal_sample):
    sample = normal_sample(np.zeros((20, 19), dtype=bool))
    assert len(sample["xyzi"]) == 20 and len(sample["queries"]) == 0


def test_query_and_rotation_streams_are_paired_and_change_per_visit(normal_sample):
    import torch
    allowed = np.zeros((300, 19), dtype=bool)
    allowed[:150, 8] = True
    allowed[150:, 14:16] = True
    first = normal_sample(allowed, queries=64, key=(1, 0, 7), augment=True)
    paired = normal_sample(allowed, queries=64, key=(1, 0, 7), augment=True)
    repeated = normal_sample(allowed, queries=64, key=(1, 0, 8), augment=True)
    different_seed = normal_sample(allowed, queries=64, key=(1, 0, 7), seed=307, augment=True)
    for key in ("xyzi", "queries", "allowed", "slots"):
        torch.testing.assert_close(first[key], paired[key], atol=0, rtol=0)
    for other in (repeated, different_seed):
        assert not torch.equal(first["xyzi"], other["xyzi"])
        assert not torch.equal(first["queries"], other["queries"])
        torch.testing.assert_close(first["allowed"], other["allowed"], atol=0, rtol=0)
        torch.testing.assert_close(first["xyzi"][:, :3].norm(dim=1), other["xyzi"][:, :3].norm(dim=1))
    integer = normal_sample(allowed, queries=64, key=0)
    explicit = normal_sample(allowed, queries=64, key=(0, 0, 0))
    torch.testing.assert_close(integer["queries"], explicit["queries"], atol=0, rtol=0)
    with pytest.raises(ValueError):
        normal_sample(allowed, key=(1, 0))
