"""Paired, diagnostic-only serialization comparison; no training source changes."""
import csv
import gc
import json
import resource
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.data import file_sha256, load_manifest
from src.evaluate import PreparedScans, autocast, load_model, memory_available
from src.model import GRID_SIZE, to_device
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator

OUT = Path(__file__).with_suffix('')
CHECKPOINT = ROOT / 'results/train/r2/0/base/best.pt'
OLD_THRESHOLD = -7.96875
# The smallest symmetric binary cube covering Ouster's 270 m representable range.
# This rule is fixed before inference, without using scan extrema or AP to select it.
DEPTH = 14
ORIGIN = np.full(3, -(2 ** (DEPTH - 1)), np.int64)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Ordering:
    def __init__(self, model):
        self.fixed = False
        self.reference = {}
        self.handles = []
        self.original_depths = defaultdict(set)
        for level in range(1, 5):
            down = model.backbone.enc._modules[f'enc{level}']._modules['down']
            self.handles.append(down.register_forward_hook(self.hook(level)))

    def hook(self, level):
        def observe(module, args, point):
            # Compare every pooling membership and physical position on every scan.
            # enc3.pool is the last feature tensor before the ordering intervention.
            fields = ['grid_coord', 'coord', 'pooling_inverse']
            if level == 3:
                fields.append('feat')
            for field in fields:
                key = (level, field)
                if self.fixed:
                    if not torch.equal(point[field], self.reference[key]):
                        raise ValueError(f'Non-ordering difference: {key}')
                else:
                    self.reference[key] = point[field].detach().clone()
            if level not in (3, 4):
                return
            if not self.fixed:
                self.original_depths[level].add(int(point.serialized_depth))
                return
            physical = point.grid_coord
            # Recover sensor-origin cell indices at this level. The original shift
            # is divisible by 16, so no voxel or pooling membership changes.
            coding = physical + self.delta[level]
            depth = DEPTH - level
            if coding.min() < 0 or coding.max() >= 2 ** depth:
                raise ValueError('Full input exceeds the predeclared encoding cube')
            try:
                point.grid_coord = coding
                point.serialization(order=point.order, depth=depth, shuffle_orders=False)
            finally:
                # Attention position encoding and all spatial operations see the
                # original physical grid, never the diagnostic coding coordinates.
                point.grid_coord = physical
        return observe

    def set_scan(self, shift, device):
        self.delta = {level: torch.as_tensor((shift - ORIGIN) // 2 ** level, device=device)
                      for level in (3, 4)}
        self.reference.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reference.clear()


def joint_counts(a, b, labels):
    # Lossless IEEE-754 score-pair keys; no bins, rounding, or point subsampling.
    keys = (a.view(np.uint32).astype(np.uint64) << np.uint64(32)) | b.view(np.uint32)
    unique, inverse = np.unique(keys, return_inverse=True)
    result = np.empty(len(unique), dtype=[('key', '<u8'), ('normal', '<u4'), ('anomaly', '<u4')])
    result['key'] = unique
    for label, name in enumerate(('normal', 'anomaly')):
        result[name] = np.bincount(inverse[labels == label], minlength=len(unique))
    return result


def unpack(joint):
    return ((joint['key'] >> np.uint64(32)).astype(np.uint32).view(np.float32),
            joint['key'].astype(np.uint32).view(np.float32))


def add_histogram(histogram, scores, joint):
    unique, inverse = np.unique(scores, return_inverse=True)
    counts = [np.bincount(inverse, weights=joint[name], minlength=len(unique)).astype(np.int64)
              for name in ('normal', 'anomaly')]
    for score, n, a in zip(unique, *counts):
        target = histogram[float(score)]
        target[0] += int(n)
        target[1] += int(a)


def histogram():
    return defaultdict(lambda: [0, 0])


def histogram_arrays(hist):
    scores = np.asarray(sorted(hist), np.float32)
    counts = np.asarray([hist[float(s)] for s in scores], np.int64)
    # Two labels at each exact score, with their exact integer multiplicities.
    return np.repeat(scores, 2), np.tile(np.array([0, 1], np.int8), len(scores)), counts.ravel()


def metrics(hist):
    scores, labels, counts = histogram_arrays(hist)
    valid = counts > 0
    scores, labels, counts = scores[valid], labels[valid], counts[valid]
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=counts)
    idx = np.flatnonzero(tpr > .95)[0]  # Match the official strict inequality.
    return dict(AP=float(100 * average_precision_score(labels, scores, sample_weight=counts)),
                AUROC=float(100 * auc(fpr, tpr)), FPR95=float(100 * fpr[idx]),
                threshold=float(thresholds[idx]))


def official_metrics(hist):
    scores, labels, counts = histogram_arrays(hist)
    required = 64 * int(counts.sum()) + 1_000_000_000
    def training_near_validation():
        if not Path('/proc/1970191').exists():
            return False
        logs = list((ROOT / 'results/train/r2/0').glob('*/log.jsonl'))
        latest = max(logs, key=lambda p: p.stat().st_mtime)
        with latest.open('rb') as stream:
            stream.seek(max(0, latest.stat().st_size - 4096))
            row = json.loads(stream.read().splitlines()[-1])
        return row['batches'] - row['batch'] < 500
    while memory_available() < required or training_near_validation():
        print('Waiting for a safe RAM window away from background validation', flush=True)
        time.sleep(5)
    # Exact expansion preserves every score/label multiplicity; only point order
    # changes, which cannot affect pooled AP or ROC. Call the pinned class unchanged.
    calculator = PointOODMetricsCalculator()
    calculator.all_scores = [np.repeat(scores, counts)]
    calculator.all_labels = [np.repeat(labels, counts)]
    result = {k: float(v) for k, v in calculator.compute_metrics().items()}
    expected = metrics(hist)
    for key in expected:
        assert abs(result[key] - expected[key]) < 1e-10, (key, result, expected)
    del calculator
    gc.collect()
    return result


def transitions(joint, thresholds):
    a, b = unpack(joint)
    pa, pb = a >= thresholds[0], b >= thresholds[1]
    n, p = joint['normal'].astype(np.int64), joint['anomaly'].astype(np.int64)
    result = dict(A_FN=int(p[~pa].sum()), B_FN=int(p[~pb].sum()),
                  A_FP=int(n[pa].sum()), B_FP=int(n[pb].sum()),
                  recovered=int(p[~pa & pb].sum()), lost=int(p[pa & ~pb].sum()),
                  fp_removed=int(n[pa & ~pb].sum()), fp_added=int(n[~pa & pb].sum()))
    assert result['A_FN'] - result['B_FN'] == result['recovered'] - result['lost']
    assert result['B_FP'] - result['A_FP'] == result['fp_added'] - result['fp_removed']
    return result


def analyze():
    run = json.loads((OUT / 'run.json').read_text())
    records = run['records']
    pooled = [histogram(), histogram()]
    sequences = defaultdict(lambda: [histogram(), histogram()])
    frames = []
    with np.load(OUT / 'pairs.npz') as archive:
        for row in records:
            joint = archive[str(row['index'])]
            scores = unpack(joint)
            local = [histogram(), histogram()]
            for side in (0, 1):
                add_histogram(pooled[side], scores[side], joint)
                add_histogram(sequences[row['sequence']][side], scores[side], joint)
                add_histogram(local[side], scores[side], joint)
            frames.append(dict(index=row['index'], sequence=row['sequence'], frame=row['frame'],
                               normal=int(joint['normal'].sum()), anomaly=int(joint['anomaly'].sum()),
                               A_AP=metrics(local[0])['AP'], B_AP=metrics(local[1])['AP']))
    result = dict(A=metrics(pooled[0]), B=metrics(pooled[1]))
    write_json('metrics.json', result)
    print('Exact full-set metrics', json.dumps(result), flush=True)
    totals = {}
    definitions = {'old': [OLD_THRESHOLD] * 2,
                   'own95': [result['A']['threshold'], result['B']['threshold']]}
    with np.load(OUT / 'pairs.npz') as archive:
        for row in frames:
            joint = archive[str(row['index'])]
            for name, thresholds in definitions.items():
                changes = transitions(joint, thresholds)
                for key, value in changes.items():
                    row[f'{name}_{key}'] = value
                    totals[f'{name}_{key}'] = totals.get(f'{name}_{key}', 0) + value
    sequence_rows = []
    for sequence, pair in sorted(sequences.items()):
        subset = [r for r in frames if r['sequence'] == sequence]
        row = dict(sequence=sequence, frames=len(subset),
                   normal=sum(r['normal'] for r in subset), anomaly=sum(r['anomaly'] for r in subset))
        for name, hist in zip(('A', 'B'), pair):
            row.update({f'{name}_{k}': v for k, v in metrics(hist).items()})
        row.update({k: sum(r[k] for r in subset) for k in totals})
        sequence_rows.append(row)
    # Descriptive failure criteria declared before inference; these are not new
    # official metrics or evidence of temporal causality.
    failure_sets = {}
    for setting in definitions:
        selected = {}
        for method in ('A', 'B'):
            severe, dips = [], []
            for row in frames:
                if row['anomaly'] >= 50 and row[f'{setting}_{method}_FN'] / row['anomaly'] >= .9:
                    severe.append([row['sequence'], row['frame']])
            for seq in sequences:
                seq_rows = sorted((r for r in frames if r['sequence'] == seq), key=lambda r: r['frame'])
                for left, mid, right in zip(seq_rows, seq_rows[1:], seq_rows[2:]):
                    if right['frame'] != mid['frame'] + 1 or left['frame'] != mid['frame'] - 1:
                        continue
                    if min(r['anomaly'] for r in (left, mid, right)) < 50:
                        continue
                    miss = [r[f'{setting}_{method}_FN'] / r['anomaly'] for r in (left, mid, right)]
                    if miss[1] - max(miss[0], miss[2]) >= .5:
                        dips.append([seq, mid['frame']])
            selected[method] = dict(severe_miss_frames=severe, isolated_recall_dips=dips)
        selected['good_to_bad'] = [[r['sequence'], r['frame']] for r in frames
            if r['anomaly'] >= 50 and r[f'{setting}_A_FN'] / r['anomaly'] <= .1
            and r[f'{setting}_B_FN'] / r['anomaly'] >= .5]
        selected['large_fp_increase'] = [[r['sequence'], r['frame']] for r in frames
            if r[f'{setting}_B_FP'] - r[f'{setting}_A_FP'] >= 1000
            and r[f'{setting}_B_FP'] >= 2 * r[f'{setting}_A_FP']]
        failure_sets[setting] = selected
    write_csv('frames.csv', frames)
    write_csv('sequences.csv', sequence_rows)
    write_json('errors.json', dict(totals=totals, definitions=definitions, failure_sets=failure_sets))
    print('Calling unchanged official calculator on all point multiplicities', flush=True)
    result['official'] = {name: official_metrics(hist) for name, hist in zip(('A', 'B'), pooled)}
    result['official_matches_exact_counts'] = True
    result['normal'] = sum(r['normal'] for r in frames)
    result['anomaly'] = sum(r['anomaly'] for r in frames)
    result['scans'] = len(frames)
    result['sequences'] = len(sequences)
    result['historical'] = json.loads((CHECKPOINT.parent / 'epoch8.json').read_text())['validation']['metrics']
    result['A_minus_historical'] = {k: result['A'][k] - result['historical'][k] for k in result['A']}
    result['finished_at'] = datetime.now().astimezone().isoformat()
    result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json('metrics.json', result)
    print('COMPLETE', json.dumps(result), flush=True)


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'pairs.npz').exists():
        raise FileExistsError('Do not overwrite completed diagnostic predictions')
    manifest = load_manifest(ROOT / 'assets/val.json', 'val')
    indices = [i for i, row in enumerate(manifest['records']) if row['eligible']]
    assert len(indices) == 1960
    identity = dict(started_at=datetime.now().astimezone().isoformat(), checkpoint=str(CHECKPOINT),
                    checkpoint_sha256=file_sha256(CHECKPOINT), manifest_sha256=manifest['sha256'],
                    protocol='AJAE-V4-F240-R2', method='base', epoch=8, scans=len(indices),
                    origin_grid_units=ORIGIN.tolist(), origin_meters=(ORIGIN * GRID_SIZE).tolist(),
                    cube_upper_exclusive_meters=(-ORIGIN * GRID_SIZE).tolist(), depth_5cm=DEPTH,
                    depth_enc3=DEPTH-3, depth_enc4=DEPTH-4,
                    precision='bfloat16 autocast; sparse operations float32',
                    origin_basis='Ouster OS1 maximum representable range 270 m; smallest symmetric binary cube',
                    range_reference='https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_operations/sensor-operations.html',
                    rejected_range_before_validation_metrics={'cube': [-204.8, 204.8],
                        'reason': '200 m paper description is not a hard coordinate bound',
                        'out_of_range_frames': [[138,431], [141,488], [142,381]],
                        'raw_global_min_meters': [-219.02696228027344,-170.88568115234375,-70.51082611083984],
                        'raw_global_max_meters': [232.8235321044922,184.8860626220703,78.7309799194336]},
                    scope='Only enc3/enc4 serialization coordinates and depth; original positional grids restored',
                    error_definitions={'dense_frame_min_anomalies': 50, 'severe_miss_fraction': .9,
                        'isolated_recall_drop_pp_against_both_neighbors': 50,
                        'good_to_bad_recall': [.9, .5], 'large_fp_increase': '>=1000 added and >=2x A'},
                    records=[])
    write_json('run.json', identity)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.2, device)
    model, saved = load_model(CHECKPOINT, device)
    assert saved['mode'] == 'base' and saved['epoch'] == 8
    del saved
    ordering = Ordering(model)
    loader = DataLoader(PreparedScans(manifest), batch_size=None, sampler=indices, num_workers=2,
                        pin_memory=True, prefetch_factor=1, generator=torch.Generator().manual_seed(0))
    seen = set()
    minima, maxima = np.full(3, np.inf), np.full(3, -np.inf)
    probes, checks = {}, []
    start = time.perf_counter()
    with zipfile.ZipFile(OUT / 'pairs.npz', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for number, sample in enumerate(loader, 1):
            idx = int(sample['index'])
            row = manifest['records'][idx]
            xyz = sample['xyzi'][:, :3].numpy()
            grid = np.floor(xyz.astype(np.float64) / GRID_SIZE).astype(np.int64)
            if (grid < ORIGIN).any() or (grid >= -ORIGIN).any():
                raise ValueError(f'Input outside fixed cube: {row}')
            shift = grid.min(0) // 16 * 16
            minima, maxima = np.minimum(minima, xyz.min(0)), np.maximum(maxima, xyz.max(0))
            batch = to_device(sample, device)
            ordering.set_scan(shift, device)
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(device)
            with torch.inference_mode(), autocast(device):
                ordering.fixed = False
                a = model(batch).cpu().numpy()
                ordering.fixed = True
                b = model(batch).cpu().numpy()
                if row['sequence'] not in seen:
                    ordering.fixed = False
                    repeated = model(batch).cpu().numpy()
                    assert np.array_equal(a, repeated), 'A changed after B'
                    checks.append(dict(sequence=row['sequence'], frame=row['frame'], repeat_A_exact=True))
                    seen.add(row['sequence'])
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
            assert np.isfinite(a).all() and np.isfinite(b).all()
            truth = np.where(sample['targets'].numpy() < 0, 0, sample['targets'].numpy() + 1)
            calculator = PointOODMetricsCalculator()
            calculator.update(xyz, a, truth)
            calculator.update(xyz, b, truth)
            sa, sb = calculator.all_scores
            labels, labels_b = calculator.all_labels
            assert np.array_equal(labels, labels_b)
            assert (int((labels == 0).sum()), int(labels.sum())) == (row['normal'], row['anomaly'])
            joint = joint_counts(sa, sb, labels)
            assert int(joint['normal'].sum() + joint['anomaly'].sum()) == len(labels)
            with archive.open(f'{idx}.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, joint, allow_pickle=False)
            identity['records'].append(dict(index=idx, sequence=row['sequence'], frame=row['frame'],
                full_input_points=len(a), shift=shift.tolist(), joint_score_pairs=len(joint)))
            if row['sequence'] == 125 and row['frame'] in (145, 146, 147):
                probes[f'{row["frame"]}_A'] = a.copy()
                probes[f'{row["frame"]}_B'] = b.copy()
                probes[f'{row["frame"]}_slots'] = sample['slots'].numpy().copy()
            if number % 100 == 0 or number == len(indices):
                elapsed = time.perf_counter() - start
                print(json.dumps(dict(scans=number, total=len(indices), seconds=elapsed,
                    remaining_seconds=elapsed / number * (len(indices) - number),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device),
                    archive_bytes=(OUT / 'pairs.npz').stat().st_size)), flush=True)
            del batch, calculator, grid
    identity.update(inference_seconds=time.perf_counter() - start,
                    input_min_meters=minima.tolist(), input_max_meters=maxima.tolist(),
                    original_depths={k: sorted(v) for k, v in ordering.original_depths.items()},
                    repeat_checks=checks, all_pooling_and_physical_positions_equal=True,
                    all_enc3_pre_attention_features_equal=True, random_states_unchanged=True,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
    assert file_sha256(CHECKPOINT) == identity['checkpoint_sha256']
    write_json('run.json', identity)
    np.savez_compressed(OUT / 'probes.npz', **probes)
    ordering.close()
    del model, ordering, loader, sample
    gc.collect()
    torch.cuda.empty_cache()
    analyze()


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text

    chinese_path = '/mnt/c/Windows/Fonts/simsun.ttc'
    english_path = '/mnt/c/Windows/Fonts/times.ttf'
    for path in (chinese_path, english_path):
        font_manager.fontManager.addfont(path)
    chinese = font_manager.FontProperties(fname=chinese_path)
    english = font_manager.FontProperties(fname=english_path)
    assert FT2Font(chinese_path).family_name == 'SimSun'
    assert FT2Font(english_path).family_name == 'Times New Roman'
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11,
                         'pdf.fonttype': 42, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    with (OUT / 'sequences.csv').open() as stream:
        sequences = list(csv.DictReader(stream))
    with (OUT / 'frames.csv').open() as stream:
        frames = list(csv.DictReader(stream))
    cmap = ['#245f94', '#d46a38']
    fonts_used = set()

    def finish(fig, pdf, name):
        fig.canvas.draw()
        for item in fig.findobj(match=Text):
            value = item.get_text()
            if not value.strip():
                continue
            path = font_manager.findfont(item.get_fontproperties(), fallback_to_default=False)
            face = FT2Font(path)
            expected = 'SimSun' if any('\u4e00' <= c <= '\u9fff' for c in value) else 'Times New Roman'
            assert face.family_name == expected, (value, path)
            assert all(c.isspace() or ord(c) in face.get_charmap() for c in value), value
            fonts_used.add((face.family_name, path))
        fig.savefig(OUT / f'{name}.png', dpi=170)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(OUT / 'comparison.pdf') as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        x = np.arange(len(sequences))
        for side, method in enumerate(('A', 'B')):
            label = ('原排序', '固定排序')[side]
            axes[0].bar(x + (side - .5) * .36, [float(r[f'{method}_AP']) for r in sequences],
                        width=.36, color=cmap[side], label=label)
            rate = [100 * int(r[f'own95_{method}_FP']) / int(r['normal']) for r in sequences]
            axes[1].bar(x + (side - .5) * .36, rate, width=.36, color=cmap[side], label=label)
        axes[0].set_title('完整验证集的逐序列比较', fontproperties=chinese, fontsize=18)
        axes[0].set_ylabel('平均精确率（百分比）', fontproperties=chinese)
        axes[1].set_ylabel('误报率（百分比）', fontproperties=chinese)
        axes[1].set_title('按各自全局百分之九十五召回阈值统计', fontproperties=chinese)
        for ax in axes:
            ax.set_xticks(x, [r['sequence'] for r in sequences])
            ax.set_xlabel('序列', fontproperties=chinese)
            ax.legend(prop=chinese, frameon=False)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        finish(fig, pdf, 'sequences')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        chosen = [r for r in frames if int(r['anomaly']) >= 50]
        for ax, setting, title in zip(axes, ('old', 'own95'), ('沿用历史全局阈值', '各自全局百分之九十五召回阈值')):
            ra = [100 * (1 - int(r[f'{setting}_A_FN']) / int(r['anomaly'])) for r in chosen]
            rb = [100 * (1 - int(r[f'{setting}_B_FN']) / int(r['anomaly'])) for r in chosen]
            ax.scatter(ra, rb, s=18, alpha=.32, color=cmap[0], edgecolors='none')
            ax.plot([0, 100], [0, 100], color='#777777', linestyle='--', linewidth=1)
            ax.set(xlim=(-2, 102), ylim=(-2, 102), aspect='equal')
            ax.set_xlabel('原排序召回率（百分比）', fontproperties=chinese)
            ax.set_ylabel('固定排序召回率（百分比）', fontproperties=chinese)
            ax.set_title(title, fontproperties=chinese, fontsize=15)
            ax.grid(alpha=.2)
        fig.suptitle('异常点不少于五十个的全部验证帧', fontproperties=chinese, fontsize=18)
        finish(fig, pdf, 'frames')

        chosen = sorted((r for r in frames if int(r['sequence']) == 125), key=lambda r: int(r['frame']))
        x = np.arange(min(int(r['frame']) for r in chosen), max(int(r['frame']) for r in chosen) + 1)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        for ax, setting, title in zip(axes, ('old', 'own95'), ('沿用历史全局阈值', '各自全局百分之九十五召回阈值')):
            for side, method in enumerate(('A', 'B')):
                values = {int(r['frame']): 100 * (1 - int(r[f'{setting}_{method}_FN']) / int(r['anomaly']))
                          for r in chosen}
                ax.plot(x, [values.get(int(f), np.nan) for f in x], color=cmap[side],
                        linewidth=1.1, label=('原排序', '固定排序')[side])
            ax.axvline(146, color='#777777', linewidth=.7, linestyle=':')
            ax.set_title(title, fontproperties=chinese)
            ax.set_xlabel('帧号', fontproperties=chinese)
            ax.set_ylabel('异常点召回率（百分比）', fontproperties=chinese)
            ax.set_ylim(-3, 103)
            ax.grid(alpha=.2)
            ax.legend(prop=chinese, frameon=False)
        fig.suptitle('125', fontproperties=english, fontsize=18)
        finish(fig, pdf, '125')
    write_json('fonts.json', {'verified_actual_render_fonts': sorted(fonts_used)})


if __name__ == '__main__':
    if sys.argv[1:] == ['analyze']:
        analyze()
    elif not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ['plot']:
        plot()
    else:
        raise SystemExit('usage: order.py [analyze|plot]')
