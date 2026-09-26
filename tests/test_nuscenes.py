"""Check original source observations and exact reviewed normal annotations."""

import numpy as np


def test_background_split_preserves_full_context_and_normal_only_labels(tmp_path, monkeypatch):
    import json
    import pytest
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
    import pytest
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
