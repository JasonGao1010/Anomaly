import numpy as np

from src.attribute import cell_keys, loss_weights, pair_errors, recall_loss_weights, surface_groups


def test_ap_case_allocation_matches_point_pair_enumeration_with_ties():
    scores = np.array([-2., 0., 0., 1., 3., 3., 4.])
    labels = np.array([1, 0, 1, 0, 1, 0, 0])
    values, rank = np.unique(scores, return_inverse=True)
    positive = np.bincount(rank[labels == 1], minlength=len(values))
    negative = np.bincount(rank[labels == 0], minlength=len(values))
    denominator, ploss, nloss = loss_weights(positive, negative)
    p, n = np.flatnonzero(labels), np.flatnonzero(labels == 0)
    direct = np.array([[100 / len(p) / (scores >= scores[i]).sum() if scores[j] >= scores[i] else 0.
                       for j in n] for i in p])
    np.testing.assert_allclose(ploss[rank[p]], direct.sum(1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(nloss[rank[n]], direct.sum(0), rtol=0, atol=1e-12)
    ap = np.mean([((scores >= scores[i]) & (labels == 1)).sum() / (scores >= scores[i]).sum() for i in p])
    np.testing.assert_allclose(direct.sum(), 100 * (1 - ap), rtol=0, atol=1e-12)
    # A single-object histogram must assign only that object's part of each normal loss.
    h = np.bincount(rank[p[:2]], minlength=len(values))
    np.testing.assert_allclose((np.cumsum(h / denominator) * 100 / len(p))[rank[n]], direct[:2].sum(0))
    bands=recall_loss_weights(positive,negative)
    np.testing.assert_allclose(bands.sum(0)[positive>0],ploss[positive>0],rtol=0,atol=1e-12)
    np.testing.assert_allclose((bands*positive).sum(),direct.sum(),rtol=0,atol=1e-12)
    p_error,n_error=pair_errors(positive,negative)
    comparisons=(scores[n][None,:]>scores[p][:,None])+.5*(scores[n][None,:]==scores[p][:,None])
    np.testing.assert_allclose(p_error[rank[p]],comparisons.mean(1),rtol=0,atol=1e-12)
    np.testing.assert_allclose(n_error[rank[n]],comparisons.mean(0),rtol=0,atol=1e-12)


def test_surface_groups_merge_repeated_cells_without_crossing_semantics():
    xyz = np.array([[0., 0., 0.], [.49, .49, .49], [.6, .6, .6], [5., 0., 0.], [0., 0., 0.]])
    keys = cell_keys(xyz, np.array([40, 40, 40, 40, 50]))
    unique = np.unique(keys)
    groups = surface_groups(unique)[np.searchsorted(unique, keys)]
    assert groups[0] == groups[1] == groups[2]
    assert groups[0] != groups[3] and groups[0] != groups[4]
    orientation=np.zeros(len(unique),np.int8)
    orientation[np.searchsorted(unique,keys[2])]=1
    separated=surface_groups(unique,orientation)[np.searchsorted(unique,keys)]
    assert separated[0]!=separated[2]
