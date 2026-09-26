"""Paired mechanism comparisons retain official populations and exact decisions."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.evaluate import comparison_conditions, comparison_metrics, official_population
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


def test_voxel_sort_preserves_exact_point_order_means_and_origin_grid():
    from src.model import voxelize, GRID_SIZE
    rng = np.random.default_rng(107)
    cells = rng.integers(-5, 6, (400, 3))
    cells[1::3] = cells[::3][:len(cells[1::3])]
    cells[2::3] = cells[::3][:len(cells[2::3])]
    points = np.c_[(cells + rng.uniform(.05, .95, cells.shape)) * GRID_SIZE,
                   np.resize([1e20, 1., -1e20], len(cells))].astype(np.float32)
    # Large cancelling values expose changes in within-cell reduction order.
    samples = (points, points[:1], np.tile(points[:1], (7, 1)),
               np.array([[np.nextafter(np.float32(.05), np.float32(0)), -1., 2., .3],
                         [.05, -1., 2., .4], [-.05, -1., 2., .5]], np.float32))
    for xyzi in samples:
        grid = np.floor(xyzi[:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
        unique, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        pointer = np.r_[0, np.cumsum(counts)].astype(np.int64)
        mean = (np.add.reduceat(xyzi[order].astype(np.float64), pointer[:-1], axis=0)
                / counts[:, None]).astype(np.float32)
        expected = dict(xyzi=xyzi, grid=unique - (unique.min(axis=0) // 16) * 16,
                        voxel_xyzi=mean, inverse=inverse, order=order, pointer=pointer,
                        offset=((xyzi[:, :3].astype(np.float64) - (grid + .5) * GRID_SIZE)
                                / GRID_SIZE).astype(np.float32))
        actual = voxelize(xyzi)
        for key, value in expected.items():
            assert actual[key].numpy().dtype == value.dtype
            np.testing.assert_array_equal(actual[key].numpy(), value, err_msg=key)


def test_normal_semantics_keeps_unseen_training_classes_and_false_positives():
    from src.normal import normal_semantic_metrics
    matrix = torch.zeros(19, 19, dtype=torch.long)
    matrix[0, 0], matrix[1, 1], matrix[9, 0] = 128, 128, 128
    measured = normal_semantic_metrics(matrix)
    assert measured["iou"][0] == .5 and measured["iou"][1] == 1
    assert measured["iou"][9] == 0 and measured["mean_iou_gt"] == .5


def test_feature_evidence_initialization_preserves_linear_predictions_in_float32():
    from src.normal import FeatureEvidence
    with torch.random.fork_rng():
        torch.manual_seed(913)
        model = FeatureEvidence()
        features = torch.randn(11, 252, dtype=torch.float64)
        with torch.no_grad():
            model.location.copy_(torch.randn(252))
            model.whitener.copy_(torch.randn(252, 252) / np.sqrt(252))
    # Force multiple chunks without changing the independent linear calculation.
    model.point_chunk = 4
    encoded = (features.float()-model.location) @ model.whitener
    expected_logits = encoded @ model.semantic.weight.T + model.semantic.bias
    expected_score = (encoded @ model.anomaly.weight.T + model.anomaly.bias).squeeze(-1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.components(features)
        score = model(features, torch.zeros(11, 2))
        logits = model.normal_logits(features)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert result["score"].dtype == result["logits"].dtype == torch.float32
    torch.testing.assert_close(result["logits"], expected_logits)
    torch.testing.assert_close(result["score"], expected_score)
    torch.testing.assert_close(score, expected_score)
    torch.testing.assert_close(logits, expected_logits)
    linear = FeatureEvidence(residual=False)
    linear.load_state_dict({key: value for key, value in model.state_dict().items()
                            if not key.startswith("residual.")})
    assert linear.residual is None
    assert not any(name.startswith("residual.") for name, _ in linear.named_parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        linear_result = linear.components(features)
    torch.testing.assert_close(linear_result["logits"], result["logits"])
    torch.testing.assert_close(linear_result["score"], result["score"])


def test_feature_evidence_both_supervisions_update_shared_residual_not_statistics():
    from src.normal import FeatureEvidence
    with torch.random.fork_rng():
        torch.manual_seed(914)
        model = FeatureEvidence()
        features = torch.randn(13, 252)
    result = model.components(features)
    semantic_loss = torch.nn.functional.cross_entropy(result["logits"], torch.arange(13) % 19)
    anomaly_loss = torch.nn.functional.binary_cross_entropy_with_logits(result["score"],
        (torch.arange(13) % 2).float())
    shared = model.residual[-1].weight
    normal_gradient = torch.autograd.grad(semantic_loss, shared, retain_graph=True)[0]
    anomaly_gradient = torch.autograd.grad(anomaly_loss, shared, retain_graph=True)[0]
    assert torch.isfinite(normal_gradient).all() and normal_gradient.abs().sum() > 0
    assert torch.isfinite(anomaly_gradient).all() and anomaly_gradient.abs().sum() > 0
    (semantic_loss+anomaly_loss).backward()
    torch.testing.assert_close(shared.grad, normal_gradient+anomaly_gradient)
    parameters = dict(model.named_parameters())
    for name in ("location", "whitener"):
        value = getattr(model, name)
        assert name not in parameters and not value.requires_grad and value.grad is None


def test_semantic_competition_uses_joint_class_probability_and_binary_semantic_gradients():
    from src.normal import FeatureEvidence
    with torch.random.fork_rng():
        torch.manual_seed(919)
        model = FeatureEvidence(semantic_competition=True)
        features = torch.randn(7, 252)
    result = model.components(features)
    unknown = model.anomaly(model.encode(features))
    joint = torch.cat((result["logits"], unknown), dim=-1).softmax(-1)[:, -1]
    torch.testing.assert_close(result["score"].sigmoid(), joint)
    binary = torch.nn.functional.binary_cross_entropy_with_logits(result["score"],
        torch.tensor([0., 1., 0., 1., 0., 1., 0.]))
    gradients = torch.autograd.grad(binary, (model.anomaly.weight, model.semantic.weight, model.residual[-1].weight))
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_evidence_binary_weights_preserve_class_mass_and_equalize_only_anomaly_views():
    from src.train import evidence_binary_weights
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.tensor([-1, -1, 0, 1, 0, 1, 1, 1])
    counts = torch.tensor([1, 3])
    point = evidence_binary_weights(targets, views, counts, "point")
    old = (len(targets)/(2*torch.bincount(targets).float()))[targets]
    assert torch.equal(point, old)
    weighted = evidence_binary_weights(targets, views, counts, "view")
    assert torch.equal(weighted[targets == 0], point[targets == 0])
    for target in (0, 1):
        assert float(weighted[targets == target].sum()/len(targets)) == pytest.approx(.5)
    for view in (0, 1):
        selected = (targets == 1) & (views == view)
        assert float(weighted[selected].sum()/len(targets)) == pytest.approx(.25)
    with pytest.raises(ValueError, match="recorded positive anomaly count"):
        evidence_binary_weights(targets, views, torch.tensor([2, 2]), "view")
    with pytest.raises(ValueError, match="recorded positive anomaly count"):
        evidence_binary_weights(targets, views, torch.tensor([1, 3, 0]), "point")
    control = torch.tensor([False, True, True, False, False, False, False, False])
    controlled = evidence_binary_weights(targets, views, counts, "view", control)
    assert torch.equal(controlled[targets == 1], weighted[targets == 1])
    assert float(controlled[control].sum()/len(targets)) == pytest.approx(.25)
    assert float(controlled[(targets == 0) & ~control].sum()/len(targets)) == pytest.approx(.25)
    with pytest.raises(ValueError, match="genuine inserted normal"):
        evidence_binary_weights(targets, views, counts, "view", targets == 1)


def test_evidence_refinement_updates_only_anomaly_head_and_preserves_semantics():
    from src.normal import FeatureEvidence
    from src.train import evidence_refinement, evidence_tail_loss
    with torch.random.fork_rng():
        torch.manual_seed(916)
        original, model = FeatureEvidence(), FeatureEvidence()
        with torch.no_grad():
            original.residual[-1].weight.normal_(0, .03)
        features = torch.randn(9, 252)
    evidence_refinement(model, original.state_dict())
    frozen = {key: value.clone() for key, value in model.state_dict().items() if not key.startswith("anomaly.")}
    before = model.components(features)
    assert torch.equal(before["logits"], original.normal_logits(features))
    assert [name for name, p in model.named_parameters() if p.requires_grad] == ["anomaly.weight", "anomaly.bias"]
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(before["score"], targets.float())
    loss += .1 * evidence_tail_loss(before["score"], targets, torch.ones(9), before["score"][:3])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.03)
    loss.backward()
    assert model.anomaly.weight.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("anomaly."))
    optimizer.step()
    for key, value in frozen.items():
        assert torch.equal(value, model.state_dict()[key])
    after = model.components(features)
    assert torch.equal(before["logits"], after["logits"])
    assert not torch.equal(before["score"], after["score"])


def test_normal_control_mass_moves_only_normal_coefficients_and_preserves_all_points():
    from src.train import evidence_binary_weights
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.tensor([-1, -1, 0, 1, 0, 1, 1, 1])
    counts = torch.tensor([1, 3])
    control = torch.tensor([True, False, False, False, False, False, False, False])
    default = evidence_binary_weights(targets, views, counts, "view", control)
    explicit = evidence_binary_weights(targets, views, counts, "view", control, .25)
    assert torch.equal(default, explicit)
    adjusted = evidence_binary_weights(targets, views, counts, "view", control, .05)
    assert torch.equal(adjusted[targets == 1], default[targets == 1])
    assert (adjusted > 0).all()
    assert float(adjusted[control].sum()/len(targets)) == pytest.approx(.05)
    assert float(adjusted[(targets == 0) & ~control].sum()/len(targets)) == pytest.approx(.45)
    assert float(adjusted[targets == 1].sum()/len(targets)) == pytest.approx(.5)
    for invalid in (0, .5, -.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strictly between"):
            evidence_binary_weights(targets, views, counts, "view", control, invalid)


def test_sqrt_normal_weights_conserve_mass_without_changing_anomalies_or_controls():
    from src.train import evidence_binary_weights
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.tensor([-1, -1, -1, -1, -1, -1, 0, 1, 1, 1])
    semantic = torch.tensor([0, 1, 1, 1, 1, 18, -1, -1, -1, -1])
    control = torch.tensor([False, False, False, False, False, True, False, False, False, False])
    counts = torch.tensor([1, 3])
    for selected_control in (None, control):
        point = evidence_binary_weights(targets, views, counts, "view", selected_control, .05)
        weighted = evidence_binary_weights(targets, views, counts, "view", selected_control, .05,
                                           semantic, "sqrt_class")
        ordinary = (targets == 0) if selected_control is None else (targets == 0)&~selected_control
        assert torch.equal(weighted[~ordinary], point[~ordinary])
        mass = .5 if selected_control is None else .45
        assert float(weighted[ordinary].sum()/len(targets)) == pytest.approx(mass)
        category_mass = torch.bincount(semantic[ordinary], weights=weighted[ordinary]/len(targets), minlength=19)
        roots = torch.bincount(semantic[ordinary], minlength=19).float().sqrt()
        torch.testing.assert_close(category_mass, mass*roots/roots.sum())
        assert (category_mass[roots == 0] == 0).all()
        assert (weighted > 0).all()


def test_evidence_tail_weights_reproduce_global_view_ranking_across_point_batches():
    from src.train import evidence_binary_weights, evidence_tail_loss
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.tensor([-1, -1, 0, 1, 0, 1, 1, 1])
    counts = torch.tensor([1, 3])
    control = torch.tensor([True, False, False, False, False, False, False, False])
    weights = evidence_binary_weights(targets, views, counts, "view", control)
    scores = torch.tensor([-.2, 1., .1, -.5, .4, .8, 1.2, 2.], requires_grad=True)
    hard = torch.tensor([1., 2.], requires_grad=True)
    # Unequal point batches expose accidental normalization by batch positive mass.
    batches = (torch.tensor([0, 1, 4]), torch.tensor([2, 3, 5, 6, 7]))
    actual = sum(len(at)/len(scores) * evidence_tail_loss(scores[at], targets[at], weights[at], hard) for at in batches)
    pair = torch.nn.functional.softplus(1 + hard[None] - scores[targets == 1, None]).mean(1)
    q = 1/(len(counts)*counts[views[targets == 1]].float())
    expected = (q * pair).sum()
    torch.testing.assert_close(actual, expected)
    ag = torch.autograd.grad(actual, (scores, hard), retain_graph=True)
    eg = torch.autograd.grad(expected, (scores, hard))
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e)
    assert (ag[0][targets == 1] < 0).all() and (ag[1] > 0).all()
    empty = evidence_tail_loss(scores[:4], targets[:4], weights[:4], hard)
    assert empty.item() == 0 and torch.isfinite(empty)


def test_target_cache_view_excludes_every_auxiliary_point(tmp_path):
    from src.train import target_normal_cache
    path = tmp_path / "source.npy"
    np.save(path, np.array([1, 1, 1, 0, 0], np.uint8))
    frames = [dict(index=0,begin=0,end=3,source="normal_stu",scene="206"),
              dict(index=1,begin=3,end=5,source="nuscenes",scene="source")]
    cache = dict(count=5,capacity=5,frames=frames,paths=dict(source=str(path)))
    selected = target_normal_cache(cache,"206")
    assert selected["count"] == 3 and selected["frames"] == frames[:1]
    assert cache["count"] == 5
    with pytest.raises(ValueError,match="original STU prefix"):
        target_normal_cache(cache,"201")


def test_auxiliary_ranking_lowers_normal_and_raises_anomaly_scores():
    from src.train import auxiliary_loss
    # One supported class isolates ranking from semantic competition.
    energy = torch.tensor([[4., torch.inf], [1., torch.inf]], requires_grad=True)
    scorer = SimpleNamespace(classes=2, temperature=torch.tensor(2.),
                             class_energy=lambda features, **kwargs: features)
    data = dict(features=energy, frame=torch.tensor([0, 1]), semantic=torch.tensor([0, -1]))
    positive, negative = torch.tensor([1]), torch.tensor([0])
    loss, ranking, normal, missing = auxiliary_loss(scorer, data, positive, negative)
    loss.backward()
    assert torch.isfinite(energy.grad).all()
    assert energy.grad[0, 0] > 0 and energy.grad[1, 0] < 0
    assert normal == 0 and missing == 0 and ranking > 0
    # Anomalies have no normal category; their semantic value must never enter CE.
    data["semantic"][positive] = 999
    changed, _, _, _ = auxiliary_loss(scorer, data, positive, negative)
    torch.testing.assert_close(changed, loss)


@pytest.mark.parametrize("all_missing", [False, True])
def test_auxiliary_semantics_skips_only_unavailable_true_class_references(all_missing):
    from src.train import auxiliary_loss
    energy = torch.tensor([[4., torch.inf], [2., 7.], [1., 3.], [5., 6.]], requires_grad=True)
    if all_missing:
        with torch.no_grad():
            energy[1, 1] = torch.inf
    scorer = SimpleNamespace(classes=2, temperature=torch.tensor(2.),
                             class_energy=lambda features, **kwargs: features)
    labels = torch.tensor([1, 1 if all_missing else 0, -1, -1])
    data = dict(features=energy, frame=torch.arange(4), semantic=labels)
    loss, _, normal, missing = auxiliary_loss(scorer, data, torch.tensor([2, 3]), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(energy.grad).all()
    assert missing == (2 if all_missing else 1)
    expected = 0. if all_missing else np.log1p(np.exp(-2.5))
    assert float(normal.detach()) == pytest.approx(expected)
    assert energy.grad[0, 0] > 0  # Missing fine-class support does not erase ranking supervision.


def test_auxiliary_views_use_count_strata_and_keep_source_splits_separate(tmp_path, monkeypatch):
    import src.train as training
    from src.train import auxiliary_records
    folder = tmp_path / "train" / "object"
    (folder / "frames").mkdir(parents=True)
    counts = (0, 1, 4, 2, 3, 1, 4, 5, 20, 6, 10, 21, 100, 50, 75, 101, 200, 120, 400)
    frames = [dict(frame=i, in_range=count, range=30-i) for i, count in enumerate(counts)]
    world = dict(world_identity="object", source_sequence=206, frames=frames)
    world_path = folder / "manifest.json"
    world_path.write_text(json.dumps(world))
    # The zero-anomaly frame and the synthetic validation tree deliberately do not exist.
    for row in frames[1:]:
        np.savez(folder / "frames" / f"{row['frame']:06d}.npz", fixture=np.array([row["frame"]]))
    empty = tmp_path / "train" / "empty"
    empty.mkdir()
    (empty / "manifest.json").write_text(json.dumps(dict(world_identity="empty", source_sequence=206,
        frames=[dict(frame=0, in_range=0, range=0)])))
    pool = dict(format="stu-frozen-dataset", splits=dict(
        train=dict(source_sequence=206, samples=len(frames)+1, worlds=[
            dict(path="train/object", world_identity="object"), dict(path="train/empty", world_identity="empty")]),
        validation=dict(source_sequence=201, worlds=[dict(path="missing-validation", world_identity="unused")])))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(pool))
    hashed = []
    original_sha = training.file_sha256

    def record_hash(filename):
        hashed.append(str(filename))
        return original_sha(filename)

    monkeypatch.setattr(training, "file_sha256", record_hash)
    selected = auxiliary_records(path, samples=12)
    assert [r["available"] for r in selected["count_strata"]] == [6, 4, 4, 4]
    assert [r["selected"] for r in selected["count_strata"]] == [3, 3, 3, 3]
    assert len(selected["records"]) == len({r["frame"] for r in selected["records"]}) == 12
    assert set(hashed) == {str(path), *(r["delta"] for r in selected["records"])}
    assert auxiliary_records(path, samples=12) == selected
    redistributed = auxiliary_records(path, samples=17)
    assert [r["selected"] for r in redistributed["count_strata"]] == [5, 4, 4, 4]
    assert selected["positive_frames"] == len(frames)-1 > 8
    assert selected["zero_anomaly_frames_excluded"] == 2
    assert selected["source_sequence"] == 206
    world["source_sequence"] = 201
    world_path.write_text(json.dumps(world))
    with pytest.raises(ValueError, match="source identities"):
        auxiliary_records(path)
    pool["splits"]["train"]["source_sequence"] = 201
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="train input must use STU206"):
        auxiliary_records(path)
    pool["splits"]["train"].update(source_sequence=206, samples=1,
        worlds=[dict(path="train/empty", world_identity="empty")])
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="no in-range anomaly observations"):
        auxiliary_records(path)
    # The independently authorized development split must use 201, never train206.
    validation = tmp_path / "validation" / "object"
    (validation / "frames").mkdir(parents=True)
    (validation / "manifest.json").write_text(json.dumps(dict(world_identity="dev", source_sequence=201,
        frames=[dict(frame=0, in_range=4, range=8)])))
    np.savez(validation / "frames" / "000000.npz", fixture=np.array([0]))
    pool["splits"]["validation"] = dict(source_sequence=201, samples=1,
        worlds=[dict(path="validation/object", world_identity="dev")])
    path.write_text(json.dumps(pool))
    development = auxiliary_records(path, split="validation", samples=80)
    assert development["source_sequence"] == development["seed"] == 201
    assert development["selected_frames"] == 1 and development["records"][0]["world"] == "dev"
    pool["splits"]["validation"]["worlds"][0]["path"] = "train/object"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError):
        auxiliary_records(path, split="validation")


@pytest.mark.parametrize("sequence", [206, 201])
def test_auxiliary_scan_preserves_training_and_full_evaluation_points(tmp_path, monkeypatch, sequence):
    import src.data as data
    from src.train import AuxiliaryScans, file_sha256

    xyzi = np.zeros((524, 4), np.float32)
    xyzi[:520, 0] = 10
    xyzi[521:, 0] = [10, 100, 10]
    packed = np.full(524, 40, np.uint32)
    packed[520], packed[521], packed[523] = 0, 1, 0
    original = data.Frame(0, xyzi, np.eye(4), packed, sequence_id=sequence)
    monkeypatch.setattr(data, "STUSequence", lambda root: [original])
    scan_path, label_path = tmp_path / "source.bin", tmp_path / "source.label"
    xyzi.tofile(scan_path)
    packed.tofile(label_path)

    def normal_records(source, *, development):
        assert source == "201" and development
        return [dict(frame=0, scan=str(scan_path), label=str(label_path), pose=np.eye(4).tolist())]

    monkeypatch.setattr(data, "normal_records", normal_records)
    delta = tmp_path / "frame.npz"
    np.savez(delta, format="stu-frozen-frame", source_identity=data.legacy_source_identity(original),
        world_identity="object", source_slot=np.array([520], np.int32),
        inserted_slot=np.array([520], np.int32), occluded_slot=np.array([], np.int32),
        xyzi=np.array([[10, 0, 1, .5]], np.float32),
        packed_labels=np.array([(60001 << 16) | 2], np.uint32))
    records = [dict(frame=0, delta=str(delta), delta_sha256=file_sha256(delta),
                    world="object", recorded_anomalies=1)]
    manifest = dict(records=records, source_sequence=sequence, split="train" if sequence == 206 else "validation")
    sample = AuxiliaryScans(manifest)[0]
    assert len(sample["slots"]) == 513 and sample["slots"][-1] == 520
    assert sample["targets"].tolist() == [0]*512 + [1]
    assert sample["semantic"].tolist() == [8]*512 + [-1]
    complete = AuxiliaryScans(manifest, full=True)[0]
    assert complete["slots"].tolist() == list(range(522))
    assert complete["targets"].tolist() == [0]*520 + [1, 0]
    assert complete["semantic"].tolist() == [8]*520 + [-1, -1]
    np.testing.assert_allclose(complete["conditions"][:520, 0].numpy(), np.log(10))
    assert (complete["conditions"][:, 1] == 0).all()
    records[0]["recorded_anomalies"] = 2
    with pytest.raises(ValueError, match="anomaly count disagrees"):
        AuxiliaryScans(manifest)[0]


def point_records(scores, confidence):
    records = np.zeros(len(scores), dtype=[("score", "f4"), ("raw_score", "f4"),
                                         ("confidence", "f4"), ("semantic", "i2")])
    records["score"], records["raw_score"], records["confidence"] = scores, scores, confidence
    records["semantic"] = np.arange(len(scores)) % 19
    return records


def test_normal_control_scan_keeps_all_insertions_and_only_samples_original_background(tmp_path, monkeypatch):
    import src.data as data
    from src.train import AuxiliaryScans, file_sha256
    # More than 512 inserted points must survive the background sampling limit.
    xyzi = np.zeros((1121, 4), np.float32)
    xyzi[:520, 0] = xyzi[-1, 0] = 10
    packed = np.zeros(1121, np.uint32)
    packed[:520], packed[-1] = 40, 1
    original = data.Frame(0, xyzi, np.eye(4), packed, sequence_id=206)
    monkeypatch.setattr(data, "STUSequence", lambda root: [original])
    delta = tmp_path / "normal.npz"
    slots = np.arange(520, 1120, dtype=np.int32)
    np.savez(delta, format="stu-normal-control-frame", normal_semantic=np.uint16(10),
        source_identity=data.legacy_source_identity(original), world_identity="normal-object",
        source_slot=slots, inserted_slot=slots, occluded_slot=np.array([], np.int32),
        xyzi=np.tile(np.array([[10, 0, 1, .5]], np.float32), (600, 1)),
        packed_labels=np.full(600, 10, np.uint32))
    record = dict(frame=0, delta=str(delta), delta_sha256=file_sha256(delta), world="normal-object",
                  recorded_anomalies=0, inserted_normal_points=600, role="normal_control")
    manifest = dict(records=[record], source_sequence=206, split="train", role="normal_control")
    sample = AuxiliaryScans(manifest)[0]
    assert len(sample["slots"]) == 1112 and sample["inserted"].sum() == 600
    assert (sample["targets"] == 0).all()
    assert (sample["semantic"][sample["inserted"]] == 0).all()
    assert (sample["semantic"][~sample["inserted"]] == 8).all()
    complete = AuxiliaryScans(manifest, full=True)[0]
    assert complete["slots"].tolist() == list(range(1121))
    assert complete["inserted"].sum() == 600 and complete["semantic"][-1] == -1
    record["inserted_normal_points"] = 599
    with pytest.raises(ValueError, match="inserted normal count"):
        AuxiliaryScans(manifest)[0]


@pytest.mark.parametrize("sequence", [206, 201])
def test_normal_control_cache_preserves_insertion_identity_without_anomaly_points(tmp_path, monkeypatch, sequence):
    import src.train as train
    import src.model as model
    sample = dict(queries=torch.arange(3), targets=torch.zeros(3, dtype=torch.int8),
        semantic=torch.tensor([8, 0, 0]), conditions=torch.zeros(3, 2),
        slots=torch.arange(3), inserted=torch.tensor([False, True, True]), frame=0, index=0)
    class Perception(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seg_head = torch.nn.Linear(72, 19)
        def encode(self, batch, indices):
            return dict(features=torch.zeros(len(indices), 252))
    monkeypatch.setattr(model, "FrozenPerception", Perception)
    monkeypatch.setattr(train, "AuxiliaryScans", lambda manifest, full: [sample])
    monkeypatch.setattr(train, "disk_check", lambda size: None)
    manifest = dict(source_sequence=sequence, role="normal_control", sha256="normal-controls",
                    records=[dict(recorded_anomalies=0, inserted_normal_points=2)])
    result = train.auxiliary_cache(manifest, tmp_path, torch.device("cpu"), workers=0)
    if sequence == 201:
        metadata, path = result, result["frames"][0]
    else:
        metadata = json.loads((tmp_path/"auxiliary.json").read_text())
        path = tmp_path/"auxiliary.npz"
    assert metadata["role"] == "normal_control" and metadata["source_sequence"] == sequence
    assert metadata["anomalies"] == 0 and metadata["normals"] == 3 and metadata["inserted_points"] == 2
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["inserted"], sample["inserted"].numpy())
        assert (saved["targets"] == 0).all()


def test_synthetic_development_preserves_background_and_separates_few_point_frames(tmp_path):
    from src.train import auxiliary_development
    class Scores(torch.nn.Module):
        def forward(self, features, conditions, predicted=None):
            return features[:, 0]
    normal = np.array([.1, .2, .3, .5, .8, .9], np.float32)
    anomaly = np.array([.3, .4, .7, .8, 1.], np.float32)
    samples = ((normal, anomaly), (normal+.1, anomaly[:4]-.2))
    paths = []
    for i, (n, a) in enumerate(samples):
        score = np.r_[n, a]
        path = tmp_path / f"{i}.npz"
        np.savez(path, features=score[:, None], targets=np.r_[np.zeros(len(n)), np.ones(len(a))],
            conditions=np.zeros((len(score), 2), np.float32), predicted=np.zeros(len(score), np.int64))
        paths.append(str(path))
    result = auxiliary_development(Scores(), dict(frames=paths), torch.device("cpu"))
    expected = PointOODMetricsCalculator()
    expected.all_scores, expected.all_labels = [np.r_[normal, anomaly]], [np.r_[np.zeros(6), np.ones(5)]]
    for key, value in expected.compute_metrics().items():
        assert result["metrics"][key] == pytest.approx(value, abs=2e-6)
    assert result["frames"] == 2 and result["eligible_frames"] == 1
    assert result["normal_points"] == 12 and result["anomaly_points"] == 9
    assert result["all_positive_frame_metrics"]["AP"] != result["metrics"]["AP"]


def test_control_development_pools_all_normals_without_changing_original_population(tmp_path):
    from src.train import auxiliary_development
    from src.evaluate import rank_metrics
    class Scores(torch.nn.Module):
        def forward(self, features, conditions):
            return features[:, 0]
    path, control_path = tmp_path / "anomaly.npz", tmp_path / "control.npz"
    normal, anomaly, control = np.array([0., 1.]), np.arange(2., 7.), np.array([3., 7., 0.])
    score = np.r_[normal, anomaly]
    np.savez(path, features=score[:, None], targets=np.r_[np.zeros(2), np.ones(5)],
        conditions=np.zeros((7, 2)), predicted=np.zeros(7))
    np.savez(control_path, features=control[:, None], targets=np.zeros(3),
        conditions=np.zeros((3, 2)), predicted=np.zeros(3), inserted=np.array([True, True, False]))
    cache = dict(frames=[str(path)])
    control_cache = dict(frames=[str(control_path)], source_sequence=201, role="normal_control")
    original = auxiliary_development(Scores(), cache, torch.device("cpu"))
    expanded = auxiliary_development(Scores(), cache, torch.device("cpu"), controls=control_cache)
    assert expanded["original60_metrics"] == original["metrics"]
    assert expanded["metrics"] == rank_metrics(np.r_[normal, control], anomaly)
    assert expanded["expanded_normal_points"] == 5
    assert expanded["normal_controls"]["FPR95"] == pytest.approx(200/3)
    assert expanded["normal_controls"]["inserted_FPR95"] == 100
    assert expanded["metrics"]["AP"] < original["metrics"]["AP"]


def test_official_population_preserves_raw_slot_order_and_boundary_rules():
    distances = np.array([2.5, 50, 2.49, 50.01, 5, 5, 10, 10, 10, 10, 10, 10], dtype=np.float32)
    xyz = np.c_[distances, np.zeros((len(distances), 2), np.float32)]
    targets = np.array([0, 0, 0, 0, -1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    slots = np.array([40, 3, 97, 31, 2, 66, 13, 29, 75, 99, 1, 53])
    selected, labels = official_population(xyz, targets)
    assert selected.tolist() == [0, 1, 5, 6, 7, 8, 9, 10, 11]
    assert slots[selected].tolist() == [40, 3, 66, 13, 29, 75, 99, 1, 53]
    np.testing.assert_array_equal(labels, targets[selected])
    with pytest.raises(ValueError, match="five-anomaly"):
        official_population(xyz[:10], targets[:10])


def test_paired_rescue_counts_and_complete_ties_use_official_scores():
    labels = np.r_[np.zeros(10, np.int8), np.ones(5, np.int8)]
    confidence = [.95] * 10 + [.95, .99, .7, .95, .95]
    baseline = point_records([0, 0, 0, 0, .1, .1, .2, .2, .3, .3, .15, .25, .35, .45, .05], confidence)
    joint = point_records([0, 0, .1, .1, .2, .2, .2, .2, .4, .4, .5, .15, .6, .55, .45], [.8] * 15)
    result = comparison_metrics(labels, dict(semantic=baseline, joint=joint), fprs=(.1, .2, .4), confidence=.9)
    assert result["confident_unknown_points"] == 4
    rows = result["methods"]["joint"]["operating_points"]
    assert rows[0]["actual_fpr"] == 0
    assert rows[0]["unknown_recalled"] == 4
    assert rows[0]["confident_unknown_recalled"] == 3
    assert rows[0]["confident_unknown_rescued"] == 2
    assert rows[0]["unknown_lost"] == 0
    assert rows[1]["actual_fpr"] == .2
    assert rows[1]["confident_unknown_rescued"] == 2
    assert rows[1]["confident_unknown_lost"] == 1
    assert rows[1]["equal_actual_fpr"]
    assert rows[2]["actual_fpr"] == .2
    assert result["methods"]["semantic"]["individual_operating_points"][2]["actual_fpr"] == .4
    assert result["methods"]["semantic"]["operating_points"][2]["actual_fpr"] == .2
    for name, records in (("semantic", baseline), ("joint", joint)):
        reference = PointOODMetricsCalculator()
        reference.all_labels, reference.all_scores = [labels], [records["score"]]
        assert result["methods"][name]["metrics"] == reference.compute_metrics()
    # Every method shares one identity array; a common reorder cannot alter any
    # paired statistic, including confidence-selected rescue and loss counts.
    order = np.random.default_rng(7).permutation(len(labels))
    reordered = comparison_metrics(labels[order], dict(semantic=baseline[order], joint=joint[order]),
                                   fprs=(.1, .2, .4), confidence=.9)
    assert result == reordered


def test_requested_fpr_endpoints_and_missing_confident_points_are_explicit():
    labels = np.array([0, 0, 1, 1], np.int8)
    values = point_records([.1, .1, .05, .2], [.5] * 4)
    result = comparison_metrics(labels, dict(semantic=values, joint=values), fprs=(0., 1.))
    rows = result["methods"]["joint"]["operating_points"]
    assert [row["actual_fpr"] for row in rows] == [0., 1.]
    assert all(row["confident_unknown_recall"] is None for row in rows)
    invalid = values.copy()
    invalid["confidence"][0] = 1.1
    with pytest.raises(ValueError, match="confidence"):
        comparison_metrics(labels, dict(semantic=invalid, joint=values))


def matched_checkpoint(variant):
    config = dict(variant=variant, synthetic_anomalies=False, data_identity={"target": "same206"},
                  source_mapping={"24": [8, 9]}, target_mapping={"40": 8}, initial_sha256="official",
                  seed=206, source_epochs=1, target_epochs=8, batch=2, queries=4096,
                  source_replay_fraction=.2, eval_every=2000, target_eval_every=225,
                  selection="normal semantic quality first", budget={"source": {"visits": 100}, "target": {"visits": 200}})
    stage = dict(trained_frames=100, trained_updates=50, planned_frames=100, planned_updates=50,
                 budget_complete=True, selected_update=40)
    return dict(config=config, frozen=True, stages=dict(source=deepcopy(stage), target=deepcopy(stage)))


def test_independent_variants_must_have_matched_actual_training_exposure():
    checkpoints = {variant: matched_checkpoint(variant) for variant in ("semantic", "joint", "separate")}
    checkpoints["joint"]["stages"]["target"]["selected_update"] = 30
    result = comparison_conditions(checkpoints)
    assert result["matched"] and result["training_link_ablation_available"]
    checkpoints["joint"]["stages"]["target"].update(trained_frames=90, trained_updates=45, budget_complete=False)
    with pytest.raises(ValueError, match="trained_frames"):
        comparison_conditions(checkpoints)
    result = comparison_conditions(checkpoints, exploratory=True)
    assert not result["matched"] and "exploratory" in result["role"]
    checkpoints["semantic"]["config"]["variant"] = "joint"
    with pytest.raises(ValueError, match="independently trained"):
        comparison_conditions(checkpoints, exploratory=True)


def test_compare_records_one_shared_preparation_and_exact_point_identity(tmp_path, monkeypatch):
    import src.evaluate as evaluation
    import src.train as training
    from torch import nn

    calls = []
    targets = torch.tensor([-1] + [0] * 6 + [1] * 5, dtype=torch.int8)
    slots = torch.tensor([33, 1, 4, 8, 12, 17, 19, 21, 23, 27, 28, 30])

    class Prepared:
        def __getitem__(self, index):
            calls.append(index)
            xyzi = torch.zeros(12, 4)
            xyzi[:, 0] = 10
            return dict(xyzi=xyzi, slots=slots, targets=targets, index=index, slot_count=40)

    class Model(nn.Module):
        mode = "normal_hypothesis"

        def __init__(self, variant):
            super().__init__()
            self.variant = variant
            self.anchor = nn.Parameter(torch.zeros(()))

        def predict(self, sample):
            score = torch.arange(12, dtype=torch.float32) + int(sample["index"]) * 20
            if self.variant == "joint":
                score = score + .5
            return dict(score=score, raw_score=score + 1, confidence=torch.full((12,), .95),
                        semantic=torch.arange(12) % 19)

    monkeypatch.setattr(evaluation, "PreparedScans", lambda *args, **kwargs: Prepared())
    monkeypatch.setattr(evaluation, "load_model", lambda path, device: (Model(path), matched_checkpoint(path)))
    monkeypatch.setattr(evaluation, "memory_available", lambda: 10 ** 15)
    monkeypatch.setattr(training, "disk_check", lambda *args: None)
    monkeypatch.setattr(training, "runtime_snapshot", lambda: {"role": "unit fixture"})
    manifest = dict(kind="val", version="unit-fixture", sha256="unit-fixture", records=[
        dict(eligible=True, normal=6, anomaly=5, points=12, scan=f"unit-fixture/{i}.bin") for i in range(2)])
    result = evaluation.compare(dict(semantic="semantic", joint="joint"), manifest, tmp_path,
                                torch.device("cpu"), workers=0)
    assert calls == [0, 1]
    identities = np.load(tmp_path / "returns.npy")
    labels = np.load(tmp_path / "labels.npy")
    selected = identities[identities["official"]]
    np.testing.assert_array_equal(selected["target"], labels)
    np.testing.assert_array_equal(selected["slot"], np.tile(slots.numpy()[1:], 2))
    np.testing.assert_array_equal(selected["frame"], np.repeat([0, 1], 11))
    for name, offset in (("semantic", 0), ("joint", .5)):
        values = np.load(tmp_path / f"{name}.npy")
        expected = np.r_[np.arange(1, 12), np.arange(1, 12) + 20] + offset
        np.testing.assert_array_equal(values["score"], expected)
        np.testing.assert_array_equal(values["raw_score"], expected + 1)
    assert result["points"] == 22 and result["scans"] == 2
    assert result["conditions"]["matched"]


def test_observation_diagnostics_matches_independent_student_mixture_quantiles():
    from scipy.optimize import brentq
    from scipy.special import entr, logsumexp, softmax
    from scipy.stats import t
    from src.normal import observation_diagnostics

    rng = np.random.default_rng(410)
    means = np.log(rng.uniform(7., 18., (7, 19, 3)))
    scales = rng.uniform(.07, .3, (7, 19, 3))
    weights = rng.dirichlet([1., 2., 3.], (7, 19))
    beliefs = rng.normal(size=(7, 19))
    appearance = rng.dirichlet([1., 2., 3., 4.], (7, 19))
    means[0, 0], scales[0, 0], weights[0, 0] = np.log([8., 10., 15.]), [.1, .25, .4], [.2, .5, .3]
    # Coincident components reduce exactly to one t distribution, regardless of priors.
    means[2, 2], scales[2, 2], weights[2, 2] = np.log(12.), .15, [.1, .7, .2]
    appearance[2, 2] = [0., 0., 1., 0.]
    distances = np.array([11., 45., 12., 15., 10., 7., 20.])
    values = np.log(distances)
    indices = torch.tensor([7, 2, 4, 1, 0, 8, 6])
    full_values = torch.zeros(10, dtype=torch.float64)
    full_values[indices] = torch.from_numpy(values)
    allowed = torch.zeros(7, 19, dtype=torch.bool)
    for row, category in ((0, 0), (1, 0), (2, 2), (3, 3)):
        allowed[row, category] = True
    allowed[4, [4, 5]] = True
    allowed[5, [6, 7]] = True
    log_density = t.logpdf(values[:, None, None], 3, loc=means, scale=scales)
    prediction = {key: torch.from_numpy(value) for key, value in
                  dict(mean=means, scale=scales, weight=np.log(weights), log_prob=log_density, belief=beliefs).items()}
    prediction["supported"] = torch.tensor([True, True, True, False, True, False, False])
    observation = dict(log_distance=full_values)
    actual = observation_diagnostics(prediction, observation, indices, allowed, torch.from_numpy(appearance))

    expected = {key: np.zeros_like(value, dtype=float) for key, value in actual.items() if isinstance(value, np.ndarray)}
    for row, category in ((0, 0), (1, 0), (2, 2), (3, 3)):
        expected["query_count"][category] += 1
        expected["appearance_mode_mass"][category] += appearance[row, category]
        expected["appearance_mode_entropy_sum"][category] += entr(appearance[row, category]).sum()
        if not prediction["supported"][row]:
            continue
        mu, sigma, prior = means[row, category], scales[row, category], weights[row, category]

        def cdf(value):
            return np.dot(prior, t.cdf(value, 3, loc=mu, scale=sigma))

        quantiles = np.exp([brentq(lambda value: cdf(value) - probability, -50., 50., xtol=1e-13, rtol=1e-14)
                            for probability in (.05, .5, .95)])
        posterior = softmax(np.log(prior) + log_density[row, category])
        expected["prediction_count"][category] += 1
        expected["nll_sum"][category] -= logsumexp(np.log(prior) + log_density[row, category])
        expected["coverage90_count"][category] += .05 <= cdf(values[row]) <= .95
        expected["width90_m_sum"][category] += quantiles[2] - quantiles[0]
        expected["abs_median_error_m_sum"][category] += abs(quantiles[1] - distances[row])
        expected["finite_interval_count"][category] += 1
        expected["geometry_mode_mass"][category] += posterior
        expected["geometry_prior_mass"][category] += prior
        expected["geometry_scale_sum"][category] += sigma
        expected["geometry_mode_entropy_sum"][category] += entr(posterior).sum()
        expected["mean_pair_separation_sum"][category] += np.abs(mu[:, None] - mu[None, :])[np.triu_indices(3, 1)].sum()
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=2e-10, atol=2e-10, err_msg=key)
    assert actual["coarse_query_count"] == 1 and actual["unsupported_query_count"] == 2
    coarse_prior = softmax(beliefs[4, [4, 5]])
    coarse_nll = -logsumexp(np.log(coarse_prior)[:, None] + np.log(weights[4, [4, 5]]) + log_density[4, [4, 5]])
    assert actual["coarse_nll_sum"] == pytest.approx(coarse_nll, abs=1e-12)
    np.testing.assert_allclose(actual["geometry_mode_mass"].sum(1), actual["prediction_count"], atol=1e-12)
    np.testing.assert_allclose(actual["geometry_mode_mass"][2], weights[2, 2], atol=1e-12)
    assert actual["mean_pair_separation_sum"][2] == 0
    assert actual["coverage90_count"][0] == 1 and actual["coverage90_count"][2] == 1

    semantic = observation_diagnostics(None, observation, indices, allowed, torch.from_numpy(appearance))
    for key in ("query_count", "appearance_mode_mass", "appearance_mode_entropy_sum"):
        np.testing.assert_allclose(semantic[key], actual[key], atol=1e-12)
    for key, value in semantic.items():
        if key not in ("query_count", "appearance_mode_mass", "appearance_mode_entropy_sum"):
            assert np.all(np.asarray(value) == 0), key


def test_feature_support_matches_independent_conditional_gaussian_mixture():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from src.normal import FeatureSupport

    rng = np.random.default_rng(206)
    scorer = FeatureSupport(features=3, classes=3, modes=3)
    features = rng.normal(size=(7, 3)).astype(np.float32)
    conditions = np.column_stack((np.linspace(1., 4., 7), rng.normal(size=7))).astype(np.float32)
    location = np.array([.3, -.4, .7, -.8])
    scale = np.array([1.2, .7, 1.6, .4])
    coefficients = rng.normal(scale=.2, size=(3, 3, 4))
    log_variance_coefficients = np.array([[.45, -.35, .3], [0., 0., 0.], [-.4, .55, -.2]])
    centers = rng.normal(scale=.5, size=(3, 3, 4))
    scorer.location.copy_(torch.from_numpy(location))
    scorer.scale.copy_(torch.from_numpy(scale))
    scorer.range_location.fill_(2.1)
    scorer.range_scale.fill_(.8)
    scorer.coefficients.copy_(torch.from_numpy(coefficients))
    scorer.log_variance_coefficients.copy_(torch.from_numpy(log_variance_coefficients))
    scorer.centers.copy_(torch.from_numpy(centers))
    expected = np.full((len(features), 3), np.inf)
    values = (np.column_stack((features, conditions[:, 1])).astype(np.float64) - location) / scale
    distance = (conditions[:, 0].astype(np.float64) - 2.1) / .8
    basis = np.column_stack((np.ones(len(features)), distance, distance ** 2))
    for category, weights in ((0, [.2, .8, 0.]), (2, [.15, .35, .5])):
        # Off-diagonal covariance detects a transposed precision factor.
        factor = np.tril(rng.normal(scale=.3, size=(4, 4)))
        np.fill_diagonal(factor, [.7, 1.1, 1.4, .9])
        covariance = factor @ factor.T
        precision = np.linalg.solve(factor, np.eye(4)).T
        log_weights = np.full(3, -np.inf)
        active = np.asarray(weights) > 0
        log_weights[active] = np.log(np.asarray(weights)[active])
        scorer.present[category] = True
        scorer.precision_cholesky[category].copy_(torch.from_numpy(precision))
        scorer.log_volume[category] = np.log(np.diag(factor)).sum()
        scorer.log_weights[category].copy_(torch.from_numpy(log_weights))
        for row in range(len(features)):
            means = basis[row] @ coefficients[category] + centers[category]
            log_variance = 4 * np.tanh(basis[row] @ log_variance_coefficients[category] / 4)
            # SciPy includes the changing covariance volume as well as residual scaling.
            conditional_covariance = np.exp(log_variance) * covariance
            components = [multivariate_normal.logpdf(values[row], mean=mean, cov=conditional_covariance)
                          for mean in means]
            expected[row, category] = -logsumexp(log_weights + components)
    actual = scorer.class_energy(torch.from_numpy(features), torch.from_numpy(conditions))
    assert torch.isinf(actual[:, 1]).all()
    np.testing.assert_allclose(actual.numpy()[:, [0, 2]], expected[:, [0, 2]], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(scorer(torch.from_numpy(features), torch.from_numpy(conditions)).numpy(),
                               expected.min(1), rtol=3e-6, atol=3e-6)


def test_feature_support_rejects_unfitted_mismatched_and_nonfinite_inputs():
    from src.normal import FeatureSupport

    scorer = FeatureSupport(features=3, classes=2, modes=1)
    features, conditions = torch.zeros(2, 3), torch.zeros(2, 2)
    with pytest.raises(ValueError, match="not been fitted"):
        scorer(features, conditions)
    scorer.present[0] = True
    scorer.log_weights[0, 0] = 0
    for wrong_features, wrong_conditions in ((torch.zeros(2, 4), conditions),
                                               (features, torch.zeros(3, 2)),
                                               (features, torch.zeros(2, 3))):
        with pytest.raises(ValueError, match="same points"):
            scorer(wrong_features, wrong_conditions)
    conditions[0, 1] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scorer(features, conditions)


def test_feature_support_can_distinguish_joint_values_at_identical_class_confidence():
    from src.normal import FeatureSupport

    scorer = FeatureSupport(features=3, classes=2, modes=1)
    scorer.present[0] = True
    scorer.log_weights[0, 0] = 0
    features = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 0., 0.]])
    conditions = torch.tensor([[2., 0.], [2., 0.], [2., 2.]])
    logits = torch.full((3, 16), -5.)
    logits[:, 3] = 5.
    confidence = logits.softmax(-1).amax(-1)
    assert torch.equal(confidence, confidence[0].expand_as(confidence))
    assert confidence[0] > .999
    # This arithmetic fixture verifies score dependence, not anomaly performance.
    scores = scorer(features, conditions)
    torch.testing.assert_close(scores[1:] - scores[0], torch.tensor([4.5, 2.]))


def test_support_conditions_preserves_raw_query_identity_and_repeated_indices():
    from src.normal import support_conditions

    xyz = np.array([[1 + i * .3, (i % 3) * .4, (i % 2) * .2] for i in range(13)], np.float32)
    xyzi = np.column_stack((xyz, np.arange(len(xyz), dtype=np.float32)))
    distances = np.linalg.norm(xyz.astype(np.float64), axis=1)
    pairwise = np.linalg.norm(xyz.astype(np.float64)[:, None] - xyz.astype(np.float64)[None], axis=-1)
    eighth_neighbor = np.sort(pairwise, axis=1)[:, 8]
    expected = np.column_stack((np.log(distances), np.log(eighth_neighbor))).astype(np.float32)
    actual = support_conditions(xyzi)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    indices = np.array([9, 2, 9, 0, 11])
    # Neighbors come from the complete raw scan, even for fewer than nine queries.
    np.testing.assert_array_equal(support_conditions(xyzi, indices), actual[indices])
    order = np.random.default_rng(43).permutation(len(xyzi))
    np.testing.assert_array_equal(support_conditions(xyzi[order], np.argsort(order)[indices]), actual[indices])
    coincident = support_conditions(np.repeat(xyzi[:1], 9, axis=0))
    np.testing.assert_allclose(coincident[:, 1], np.float32(np.log(.001)), rtol=0, atol=0)
    with pytest.raises(ValueError, match="too few or nonfinite"):
        support_conditions(xyzi[:8])


def test_instance_support_matches_independent_full_feature_nearest_neighbors():
    from src.normal import InstanceSupport

    # Artificial numerical fixture; no anomaly-detection performance is inferred.
    rng = np.random.default_rng(206)
    model = InstanceSupport(memory_size=6)
    memory = rng.normal(scale=.3, size=(6, 252)).astype(np.float32)
    features = rng.normal(scale=.3, size=(5, 252)).astype(np.float32)
    location = rng.normal(scale=.1, size=252).astype(np.float32)
    whitener = np.eye(252, dtype=np.float32) + np.triu(
        rng.normal(scale=.005, size=(252, 252)).astype(np.float32), 1)
    transform = np.eye(252, dtype=np.float32) + rng.normal(scale=.002, size=(252, 252)).astype(np.float32)
    with torch.no_grad():
        for name, value in (("memory", memory), ("location", location),
                            ("whitener", whitener), ("transform", transform)):
            getattr(model, name).copy_(torch.from_numpy(value))
        model.memory_allowed[[0, 1], 0] = True
        model.memory_allowed[2, 7] = True
        model.memory_allowed[3, 18] = True
        model.memory_allowed[4, [0, 7]] = True
        model.temperature.fill_(7.)
    encoded = (features.astype(float) - location) @ whitener.astype(float) @ transform.astype(float)
    bank = (memory.astype(float) - location) @ whitener.astype(float) @ transform.astype(float)
    distance = np.square(encoded[:, None] - bank[None]).sum(-1)
    expected = np.full((len(features), 19), np.inf)
    for category, indices in ((0, [0, 1]), (7, [2]), (18, [3])):
        expected[:, category] = distance[:, indices].min(1)
    f, g = torch.from_numpy(features), torch.zeros(len(features), 2)
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.class_energy(f, g)
        score = model(f, g)
        torch.testing.assert_close(model.raw_score(f, g + 100), score, rtol=0, atol=0)
    assert actual.dtype == score.dtype == torch.float32
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(score.numpy(), distance[:, :5].min(1), rtol=2e-6, atol=2e-5)
    with pytest.raises(ValueError, match="shape"):
        model.class_energy(f[:, :-1])
    with pytest.raises(ValueError, match="same points"):
        model.raw_score(f, g[:1])


def test_instance_support_requires_one_whole_instance_and_preserves_coarse_labels():
    from src.normal import InstanceSupport

    model = InstanceSupport(memory_size=4)
    with torch.no_grad():
        model.memory[0, 36] = 10
        model.memory[1, [0, 180]] = 10
        model.memory_allowed[:2, 0] = True
    query = torch.zeros(1, 252)
    # Layerwise nearest anchors could invent an exact hybrid match; no complete
    # normal instance matches all three levels, so the real minimum is 100.
    torch.testing.assert_close(model.raw_score(query), torch.tensor([100.]), rtol=0, atol=0)
    assert model.class_energy(query)[0, 0] == 100
    with torch.no_grad():
        model.memory_allowed[2, [0, 7]] = True
    assert model.raw_score(query).item() == 0
    energy = model.class_energy(query)
    assert energy[0, 0] == 100 and torch.isinf(energy[0, 7])
    with torch.no_grad():
        model.memory_allowed[:2] = False
    assert torch.isinf(model.class_energy(query)).all()
    assert model.raw_score(query).item() == 0
    model.memory_allowed.zero_()
    with pytest.raises(ValueError, match="trusted normal"):
        model.raw_score(query)


def test_instance_support_excludes_same_scene_or_near_target_frames_only():
    from src.normal import InstanceSupport

    model = InstanceSupport(memory_size=5, temporal_window=16)
    with torch.no_grad():
        model.memory[:, 0] = torch.arange(5)
        model.memory_allowed[:2, 0] = True
        model.memory_allowed[2:, 7] = True
        model.memory_source.copy_(torch.tensor([0, 1, 1, 1, 0]))
        model.memory_frame.copy_(torch.tensor([5, 5, 20, 21, 6]))
    query = torch.zeros(4, 252)
    source, frame = torch.tensor([0, 1, 0, 1]), torch.tensor([5, 5, 6, 6])
    energy = model.class_energy(query, source=source, frame=frame)
    # source 0 uses scene equality; source 1 excludes |frame difference| < 16,
    # including points across a 16-frame block edge. Other sources stay eligible.
    torch.testing.assert_close(energy[:, 0], torch.tensor([1., 0., 0., 0.]), rtol=0, atol=0)
    torch.testing.assert_close(energy[:, 7], torch.tensor([4., 9., 4., 16.]), rtol=0, atol=0)
    torch.testing.assert_close(model.raw_score(query, source=source, frame=frame),
                               energy.amin(1), rtol=0, atol=0)
    # Development supplies no training identities: equal integer IDs in distinct
    # caches must not be interpreted as the same actual scan.
    assert model.raw_score(query).eq(0).all()
    with pytest.raises(ValueError, match="together"):
        model.class_energy(query, source=source)
    with torch.no_grad():
        model.memory_source.fill_(1)
        model.memory_frame.fill_(5)
    assert torch.isinf(model.raw_score(query[:1], source=torch.tensor([1]), frame=torch.tensor([5]))).all()


def test_instance_support_spectral_projection_preserves_all_whitened_directions():
    from src.normal import InstanceSupport

    rng = np.random.default_rng(84)
    left = np.linalg.qr(rng.normal(size=(252, 252)))[0]
    right = np.linalg.qr(rng.normal(size=(252, 252)))[0]
    matrix = ((left * np.linspace(.1, 3., 252)) @ right.T).astype(np.float32)
    model = InstanceSupport(memory_size=1)
    with torch.no_grad():
        model.transform.copy_(torch.from_numpy(matrix))
    assert model.project_metric() is model
    u, singular, vh = np.linalg.svd(matrix.astype(float), full_matrices=False)
    expected = (u * singular.clip(.5, 2)) @ vh
    actual = model.transform.detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-6)
    projected = np.linalg.svd(actual.astype(float), compute_uv=False)
    assert projected.min() >= .5 - 1e-5 and projected.max() <= 2 + 1e-5
    delta = rng.normal(size=(32, 252))
    ratio = np.square(delta @ actual).sum(1) / np.square(delta).sum(1)
    assert np.all((ratio >= .25 - 1e-5) & (ratio <= 4 + 1e-5))


def test_instance_support_training_cache_reload_and_chunks_preserve_the_same_metric():
    from src.normal import InstanceSupport

    rng = np.random.default_rng(827)
    model = InstanceSupport(memory_size=6)
    with torch.no_grad():
        model.memory.copy_(torch.from_numpy(rng.normal(scale=.1, size=(6, 252)).astype(np.float32)))
        model.memory_allowed[:3, 0] = True
        model.memory_allowed[3:, 7] = True
    query = torch.from_numpy(rng.normal(scale=.1, size=(7, 252)).astype(np.float32))
    model.eval().freeze_metric()
    with torch.no_grad():
        before = model.raw_score(query).clone()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    energy = model.class_energy(query)[:, [0, 7]]
    target = torch.arange(len(query)) % 2
    loss = (torch.nn.functional.cross_entropy(-energy / model.temperature, target)
            + .01 * energy.gather(1, target[:, None]).mean())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(model.transform.grad).all() and model.transform.grad.abs().sum() > 0
    assert not model.memory.requires_grad
    optimizer.step()
    model.project_metric().eval().freeze_metric()
    with torch.no_grad():
        fitted = model.raw_score(query)
        assert not torch.allclose(fitted, before)
        full = model.class_energy(query)
        for chunk in (1, 3, 4096):
            model.point_chunk = chunk
            torch.testing.assert_close(model.raw_score(query), fitted, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(model.class_energy(query), full, rtol=2e-5, atol=2e-5)
        order = torch.arange(len(query) - 1, -1, -1)
        torch.testing.assert_close(model.raw_score(query[order]), fitted[order], rtol=2e-5, atol=2e-5)
    state = deepcopy(model.state_dict())
    assert [name for name, value in state.items() if value.shape == (6, 252)] == ["memory"]
    restored = InstanceSupport(memory_size=6).eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored.raw_score(query), fitted, rtol=2e-5, atol=2e-5)
    # Reload into an already cached model must not keep transformed old memory.
    state["transform"] *= 1.2
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored.raw_score(query), fitted * 1.2 ** 2, rtol=2e-5, atol=2e-5)
        restored.memory[0].add_(.25)
        white = (query.double() - restored.location.double()) @ restored.whitener.double()
        bank = (restored.memory.double() - restored.location.double()) @ restored.whitener.double()
        matrix = restored.transform.double()
        expected = ((white @ matrix)[:, None] - (bank @ matrix)[None]).square().sum(-1).amin(1)
        torch.testing.assert_close(restored.raw_score(query).double(), expected, rtol=2e-5, atol=2e-5)


def test_score_calibration_uses_frozen_groups_and_preserves_unbounded_tail_order():
    from src.normal import ScoreCalibration

    calibration = ScoreCalibration()
    scores = torch.cat((torch.linspace(0, 1, 2048), torch.linspace(10, 20, 2048), torch.ones(7)))
    predicted = torch.cat((torch.zeros(2048), torch.ones(2048), torch.full((7,), 2))).long()
    conditions = torch.zeros(len(scores), 2)
    report = calibration.fit(scores, predicted, conditions)
    assert report["class_counts"][:3] == [2048, 2048, 7]
    assert report["fallback_classes"] == list(range(2, 16))
    assert calibration(scores, None, conditions) is scores
    calibration.enabled.fill_(True)
    probes = torch.tensor([30., 40., 50.])
    reference = calibration(probes, torch.zeros(3, dtype=torch.long), conditions[:3])
    assert torch.isfinite(reference).all() and (reference.diff() > 0).all()
    assert reference[0] > -np.log(.001)
    same_score = torch.full((3,), .5)
    rescaled = calibration(same_score, torch.tensor([0, 1, 2]), conditions[:3])
    assert rescaled[0] != rescaled[1]
    global_reference = calibration(same_score, torch.full((3,), 15), conditions[:3])
    assert rescaled[2] == global_reference[2]
    torch.testing.assert_close(reference, calibration(
        probes, torch.zeros(3, dtype=torch.long), torch.tensor([[100., -100.]]).repeat(3, 1)),
        rtol=0, atol=0)
    with pytest.raises(ValueError, match="16-class"):
        calibration(probes, torch.tensor([0, 1, 18]), conditions[:3])


def test_score_calibration_ties_fallback_and_state_roundtrip():
    import io
    from src.normal import ScoreCalibration

    calibration = ScoreCalibration()
    scores = torch.cat((torch.zeros(2048), torch.arange(2048).float().div(8).floor()))
    predicted = torch.cat((torch.zeros(2048), torch.ones(2048))).long()
    conditions = torch.zeros(len(scores), 2)
    report = calibration.fit(scores, predicted, conditions)
    assert 0 in report["fallback_classes"] and 1 not in report["fallback_classes"]
    assert calibration.lengths[0] < 11
    calibration.enabled.fill_(True)
    probes = torch.tensor([-1., 0., 1., 300., 600.])
    classes = torch.tensor([0, 0, 1, 1, 15])
    expected = calibration(probes, classes, conditions[:5])
    checkpoint = io.BytesIO()
    torch.save(calibration.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = ScoreCalibration()
    state = torch.load(checkpoint, weights_only=True)
    assert set(state) == {"probabilities", "knots", "levels", "lengths", "counts", "groups",
                          "minimum_points", "fitted", "enabled"}
    restored.load_state_dict(state, strict=True)
    ScoreCalibration(range_bandwidth=0.).load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(probes, classes, conditions[:5]), expected, rtol=0, atol=0)
    if torch.cuda.is_available():
        actual = restored.cuda()(probes.cuda(), classes.cuda(), conditions[:5].cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_range_calibration_matches_weighted_quantiles_and_is_continuous():
    from src.normal import ScoreCalibration

    rng = np.random.default_rng(73)
    distance = np.linspace(np.log(2.5), np.log(50), 16384)
    scores = 6 * distance + rng.normal(0, .5, len(distance))
    conditions = torch.from_numpy(np.column_stack((distance, np.zeros(len(distance)))))
    classes = torch.zeros(len(distance), dtype=torch.long)
    calibration = ScoreCalibration(range_bandwidth=.5)
    report = calibration.fit(scores, classes, conditions)
    assert report["range_fallback_anchors"][1] == 0
    calibration.enabled.fill_(True)

    # Independent empirical inverse CDF verifies the scientific q99 threshold.
    order = np.argsort(scores)
    quantiles = []
    for anchor in calibration.range_anchors.numpy():
        weights = np.exp(-.5 * ((distance[order] - anchor) / .5) ** 2)
        indices = np.searchsorted(np.cumsum(weights), calibration.probabilities.numpy() * weights.sum())
        quantiles.append(scores[order[indices]])
    quantiles = np.asarray(quantiles, dtype=np.float32)
    np.testing.assert_array_equal(calibration.range_knots[1].numpy(), quantiles)
    anchor = calibration.range_anchors.numpy()
    probe_range = anchor[9] * .65 + anchor[10] * .35
    threshold = quantiles[9, 8] * .65 + quantiles[10, 8] * .35
    at_threshold = calibration(torch.tensor([threshold]), torch.zeros(1, dtype=torch.long),
                               torch.tensor([[probe_range, 0.]]))
    assert float(at_threshold[0]) == pytest.approx(-np.log(.01), abs=2e-5)

    probes = torch.linspace(-10, 50, 128)
    context = torch.tensor([[probe_range, 0.]]).repeat(len(probes), 1)
    output = calibration(probes, classes[:len(probes)], context)
    assert (output.diff() > 0).all() and output[-1] > -np.log(.001)
    shifted_spacing = context.clone()
    shifted_spacing[:, 1] = float("nan")
    torch.testing.assert_close(calibration(probes, classes[:len(probes)], shifted_spacing), output, rtol=0, atol=0)
    for position in anchor[1:-1]:
        near = torch.tensor([[position - 1e-6, 0.], [position, 0.], [position + 1e-6, 0.]])
        values = calibration(torch.full((3,), 16.), classes[:3], near)
        assert float((values - values[1]).abs().max()) < 1e-4
    endpoints = torch.tensor([[anchor[0], 0.], [anchor[-1], 0.]])
    assert calibration(torch.full((2,), 16.), classes[:2], endpoints).diff().abs().item() > .1
    invalid = context.clone()
    invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="log range"):
        calibration(probes, classes[:len(probes)], invalid)
    with pytest.raises(ValueError, match="log range"):
        calibration.fit(probes, classes[:len(probes)], invalid)


def test_range_calibration_effective_sample_fallback_and_rare_class_global():
    from src.normal import ScoreCalibration

    first = np.linspace(np.log(2.5), np.log(50), 3072)
    second = np.linspace(np.log(2.5), np.log(50), 12288)
    distance = np.concatenate((first, second, first[:7]))
    scores = np.concatenate((np.linspace(0, 1, len(first)), 8 + 2 * second, np.zeros(7)))
    classes = torch.tensor([0] * len(first) + [1] * len(second) + [2] * 7)
    conditions = torch.from_numpy(np.column_stack((distance, np.zeros(len(distance)))))
    calibration = ScoreCalibration(range_bandwidth=.25)
    report = calibration.fit(scores, classes, conditions)
    assert report["range_fallback_anchors"][1] == 16
    assert calibration.range_groups_enabled[1]
    torch.testing.assert_close(calibration.range_knots[1], calibration.knots[1].expand(16, -1), rtol=0, atol=0)
    assert calibration.groups[3] == 0 and not calibration.range_groups_enabled[0]
    calibration.enabled.fill_(True)
    probes = torch.full((2,), 12.)
    context = torch.tensor([[np.log(10), 0.], [np.log(20), 0.]])
    rare = calibration(probes, torch.full((2,), 2), context)
    absent = calibration(probes, torch.full((2,), 15), context)
    torch.testing.assert_close(rare, absent, rtol=0, atol=0)
    original = ScoreCalibration()
    original.fit(scores, classes, conditions)
    original.enabled.fill_(True)
    torch.testing.assert_close(rare, original(probes, torch.full((2,), 2), context), rtol=0, atol=0)
    assert rare[0] == rare[1]
    supported = calibration(probes, torch.ones(2, dtype=torch.long), context)
    assert calibration.range_groups_enabled[2] and supported.diff().abs().item() > .1

    # Loading an older fitted state must not silently change its scoring rule.
    state = deepcopy(calibration.state_dict())
    state["range_groups_enabled"][0] = True
    state["range_knots"][0] = state["range_knots"][2]
    historical = ScoreCalibration(range_bandwidth=.25)
    historical.load_state_dict(state, strict=True)
    torch.testing.assert_close(historical(probes, torch.full((2,), 2), context), supported, rtol=0, atol=0)


def test_range_calibration_atoms_keep_class_cdf_and_local_atoms_fall_back():
    from src.normal import ScoreCalibration

    size = 16384
    continuous = np.linspace(0, 100, size)
    local_atom = (continuous > 28) & (continuous < 48)
    continuous[local_atom] = 38
    scores = np.concatenate((np.repeat([0., 1.], 4096), continuous))
    distance = np.full(len(scores), np.log(50))
    distance[8192:][local_atom] = np.log(2.5)
    conditions = np.column_stack((distance, np.zeros(len(scores))))
    classes = torch.tensor([0] * 8192 + [1] * size)
    calibration = ScoreCalibration(range_bandwidth=.25)
    calibration.fit(scores, classes, conditions)
    assert not calibration.range_groups_enabled[1]
    assert calibration.lengths[1] == 3  # Linear empirical median lies between the two atoms.
    assert calibration.range_groups_enabled[2]
    # The local atom has >2048 effective points; duplicate quantiles cause fallback.
    weights = np.exp(-.5 * ((distance[8192:] - np.log(2.5)) / .25) ** 2)
    assert weights.sum() ** 2 / np.square(weights).sum() > 2048
    torch.testing.assert_close(calibration.range_knots[2, 0], calibration.knots[2], rtol=0, atol=0)
    assert not torch.equal(calibration.range_knots[2, -1], calibration.knots[2])
    calibration.enabled.fill_(True)
    original = ScoreCalibration()
    original.fit(scores, classes, conditions)
    original.enabled.fill_(True)
    probes = torch.tensor([-1., .5, 3.])
    context = torch.tensor([[np.log(2.5), 0.], [np.log(20), 0.], [np.log(50), 0.]])
    torch.testing.assert_close(calibration(probes, torch.zeros(3, dtype=torch.long), context),
                               original(probes, torch.zeros(3, dtype=torch.long), context), rtol=0, atol=0)
    restored = ScoreCalibration(range_bandwidth=1.)
    restored.load_state_dict(calibration.state_dict(), strict=True)
    assert float(restored.range_bandwidth) == .25
    torch.testing.assert_close(restored(probes, torch.ones(3, dtype=torch.long), context),
                               calibration(probes, torch.ones(3, dtype=torch.long), context), rtol=0, atol=0)


def test_instance_support_calibrates_only_the_score_using_supplied_official_prediction():
    from src.normal import InstanceSupport

    model = InstanceSupport(memory_size=1)
    model.memory_allowed[0, 18] = True
    features, conditions = torch.zeros(3, 252), torch.zeros(3, 2)
    features[:, 0] = 10
    energy = model.class_energy(features)
    raw = model.raw_score(features)
    torch.testing.assert_close(energy.amin(-1), raw, rtol=0, atol=0)
    scores = torch.cat((torch.linspace(0, 200, 2048), torch.linspace(300, 500, 2048)))
    official_classes = torch.cat((torch.zeros(2048), torch.ones(2048))).long()
    model.calibration.fit(scores, official_classes, torch.zeros(len(scores), 2))
    model.calibration.enabled.fill_(True)
    # A 19-class nearest explanation cannot choose an official 16-class group.
    assert (energy.argmin(1) == 18).all()
    with pytest.raises(ValueError, match="16-class"):
        model(features, conditions)
    supplied = torch.tensor([0, 1, 0])
    expected = model.calibration(raw, supplied, conditions)
    torch.testing.assert_close(model(features, conditions, supplied), expected, rtol=0, atol=0)
    assert expected[0] != expected[1]


def test_support_queries_and_cache_preserve_coarse_normal_candidate_sets(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src import data
    from src.train import SupportScans, support_cache

    points = np.zeros((12, 4), dtype=np.float32)
    points[:, 0] = np.arange(4, 16)
    points[:, 3] = .2
    points[[0, 11], 0] = [2., 55.]
    allowed = np.zeros((len(points), 19), dtype=bool)
    allowed[1, 0], allowed[4, 3], allowed[7, 12] = True, True, True
    allowed[2, [5, 6]] = True
    allowed[3, [8, 9, 10, 11]] = True
    allowed[6, [14, 15]] = True
    slots = np.array([1, 4, 6, 8, 12, 15, 17, 20, 25, 27, 30, 31])
    # read_normal_record already clears out-of-range and untrusted label sets;
    # every returned row still represents an actual return in the input context.
    raw = dict(xyzi=points, allowed=allowed, slots=slots, slot_count=32)
    monkeypatch.setattr(data, "read_normal_record", lambda record: raw)
    records = [dict(source="nuscenes", scene="normal-fixture")]
    expected = np.flatnonzero(allowed.any(1))
    semantic = np.array([0, -1, -1, 3, -1, 12], dtype=np.int16)
    for development in (False, True):
        sample = SupportScans(records, development=development)[0]
        np.testing.assert_array_equal(sample["queries"].numpy(), expected)
        np.testing.assert_array_equal(sample["slots"].numpy(), slots[expected])
        np.testing.assert_array_equal(sample["allowed"].numpy(), allowed[expected])
        np.testing.assert_array_equal(sample["semantic"].numpy(), semantic)
        assert sample["allowed"].dtype == torch.bool
        assert len(sample["inverse"]) == len(points)
        assert np.all(np.any(points[expected, :3] != 0, axis=1))
        assert np.all((np.linalg.norm(points[expected, :3], axis=1) >= 2.5)
                      & (np.linalg.norm(points[expected, :3], axis=1) <= 50))

    def encode(sample, indices):
        return dict(features=indices[:, None].float().expand(-1, 252))

    model = SimpleNamespace(perception=SimpleNamespace(encode=encode))
    info = support_cache(model, records, tmp_path, torch.device("cpu"), workers=0)
    assert info["labels"] == "normal_candidate_sets_v1" and info["count"] == len(expected)
    np.testing.assert_array_equal(np.load(info["paths"]["allowed"])[:info["count"]], allowed[expected])
    np.testing.assert_array_equal(np.load(info["paths"]["semantic"])[:info["count"]], semantic)
    np.testing.assert_array_equal(np.load(info["paths"]["slot"])[:info["count"]], slots[expected])
    np.testing.assert_array_equal(np.load(info["paths"]["features"])[:info["count"], 0], expected)
    with pytest.raises(ValueError, match="will not overwrite"):
        support_cache(model, records, tmp_path, torch.device("cpu"), workers=0)
