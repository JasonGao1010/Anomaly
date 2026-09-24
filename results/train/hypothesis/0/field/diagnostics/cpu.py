"""Bounded CPU failure analysis using the unchanged model and data reader.

Run from the repository root with PYTHONPATH=. and CUDA_VISIBLE_DEVICES=''.
The score-independent panel is diagnostic, not a replacement validation set.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from src.data import file_sha256, load_manifest, write_json
from src.evaluate import PreparedScans, evaluation_indices, load_model, memory_available


def select_panel(records, categories):
    cells = defaultdict(list)
    originals = {}
    for i, row in enumerate(records):
        if row['group'] == 'normal_nuscenes':
            originals[row['token']] = i
        elif row['anomaly']:
            band = int(np.searchsorted([10, 20, 30, 40], row['point_range_median'], side='right'))
            cells[('eligible' if row['eligible'] else 'sparse', row['instance'], band)].append(i)
        elif row['group'] == 'control_nuscenes' and row['inserted_points']:
            band = int(np.argmax(row['inserted_point_histogram']))
            cells[('control', categories[row['donor']], band)].append(i)
    chosen = []
    for key, indices in sorted(cells.items()):
        # Hash ordering uses identities only; diversify scenes before repeating them.
        indices.sort(key=lambda i: hashlib.sha256(
            f"cpu-panel-1250:{records[i]['token']}:{records[i].get('delta', '')}".encode()).hexdigest())
        scenes, tokens, selected = set(), set(), []
        quota = 1 if key[0] == 'sparse' else 2
        for distinct_scene in (True, False):
            for i in indices:
                row = records[i]
                if len(selected) == quota:
                    break
                if row['token'] in tokens or (distinct_scene and row['scene'] in scenes):
                    continue
                selected.append(i)
                scenes.add(row['scene'])
                tokens.add(row['token'])
        chosen.extend(selected)
    pairs = {}
    for i in chosen.copy():
        if records[i]['group'] == 'control_nuscenes':
            pairs[i] = originals[records[i]['token']]
            chosen.append(pairs[i])
    return sorted(set(chosen)), pairs


def counts(target, scores, threshold):
    positive, negative = target == 1, target == 0
    predicted = scores >= threshold
    return dict(positive=int(positive.sum()), negative=int(negative.sum()),
                tp=int((positive & predicted).sum()), fp=int((negative & predicted).sum()))


def quantiles(values):
    return list(map(float, np.quantile(values, [.01, .05, .5, .95, .99]))) if len(values) else None


def init_worker(checkpoint, manifest, update, threads):
    global DATA, MODEL
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    MODEL, saved = load_model(checkpoint, torch.device('cpu'))
    if saved['successful_updates'] != update:
        raise ValueError('Checkpoint changed before a worker loaded it')
    DATA = PreparedScans(manifest, normal=True, normal_reference=False)


def infer(task):
    index, repeat = task
    if memory_available() < 6_000_000_000:
        raise RuntimeError('Insufficient spare RAM; protect the ongoing training')
    sample = DATA[index]
    with torch.no_grad():
        prediction = MODEL(sample).numpy()
        difference = float(np.max(abs(prediction - MODEL(sample).numpy()))) if repeat else None
    return (prediction, sample['targets'].numpy(), sample['slots'].numpy(),
            np.linalg.norm(sample['xyzi'][:, :3].numpy(), axis=1), difference,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def group_counts(points, frames, manifest, threshold):
    groups = defaultdict(Counter)
    ranges = ((2.5, 10), (10, 20), (20, 30), (30, 40), (40, 50.00001))
    for frame in frames:
        p = points[frame['start']:frame['stop']]
        target, score, role = p['target'], p['score'], frame['role']
        masks = [('role', role, np.ones(len(p), bool))]
        for semantic in np.unique(p['semantic'][target == 0]):
            if semantic < 0:
                continue
            name = manifest['mapping'][int(semantic)]['name']
            normal = (p['semantic'] == semantic) & (target == 0)
            masks.append((role + '/normal_semantic', name, normal))
            for lo, hi in ranges:
                masks.append((role + '/normal_semantic_range', f'{name}/{lo:g}-{min(hi,50):g}',
                              normal & (p['range'] >= lo) & (p['range'] < hi)))
        for lo, hi in ranges:
            masks.append((role + '/range', f'{lo:g}-{min(hi,50):g}', (p['range'] >= lo) & (p['range'] < hi)))
        if frame['positive']:
            masks.extend([(role + '/family', frame['family'], target == 1),
                          (role + '/donor', frame['instance'], target == 1),
                          (role + '/point_count', str(int(np.searchsorted([5, 10, 20, 50, 100], frame['positive'], side='right'))), target == 1)])
        if role == 'control_nuscenes':
            masks.append(('control/inserted_category', frame['control_category'], p['inserted'] & (target == 0)))
        masks.append((role + '/scene', frame['scene'], target >= 0))
        for dimension, group, mask in masks:
            values = counts(target[mask], score[mask], threshold)
            values['frames'] = int(values['positive'] + values['negative'] > 0)
            groups[dimension, group].update(values)
    return [dict(dimension=key[0], group=key[1], **values) for key, values in sorted(groups.items())]


def mechanisms(args, manifest, epoch):
    """Case interventions distinguish BN statistics and contaminated context.

    The context intervention uses labels only as an oracle diagnostic; its scores
    are never deployable predictions or replacements for the official metrics.
    """
    analysis = json.loads((args.output / 'analysis.json').read_text())
    if file_sha256(args.checkpoint) != analysis['checkpoint_sha256']:
        raise ValueError('Mechanism analysis requires the exact panel checkpoint')
    points = np.load(args.output / 'points.npy', mmap_mode='r')
    by_index = {f['index']: f for f in analysis['frames']}
    failed = [f['index'] for f in analysis['frames'] if f['role'] == 'eligible' and f['tp'] < f['positive']]
    dense = sorted((f for f in analysis['frames'] if f['role'] == 'eligible' and f['tp'] == f['positive']),
                   key=lambda f: -f['positive'])[:2]
    controls = []
    for category in sorted({f['control_category'] for f in analysis['frames'] if f['role'] == 'control_nuscenes'}):
        candidates = [f for f in analysis['frames'] if f['control_category'] == category]
        controls.append(max(candidates, key=lambda f: f['inserted_fp'])['index'])
    pairs = {p['control']: p['original'] for p in analysis['paired_controls']}
    indices = sorted(set(failed + [f['index'] for f in dense] + controls + [pairs[i] for i in controls]))
    model, saved = load_model(args.checkpoint, torch.device('cpu'))
    del saved
    data = PreparedScans(manifest, normal=True, normal_reference=False)
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    def restore():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
        model.eval()
    output, start = [], time.perf_counter()
    for number, i in enumerate(indices, 1):
        sample, frame = data[i], by_index[i]
        target = sample['targets'].numpy()
        saved_points = points[frame['start']:frame['stop']]
        if not np.array_equal(saved_points['slot'], sample['slots'].numpy()):
            raise ValueError('Case point identities changed')
        density = []
        hook = model.head.register_forward_pre_hook(lambda module, values: density.append(values[0][:, -3:].detach().clone()))
        with torch.no_grad():
            baseline = model(sample).numpy()
        hook.remove()
        marginal = torch.cat(density).numpy()
        with torch.no_grad():
            for module in bn:
                module.train()
            current = model(sample).numpy()
            restore()
        variants = {'baseline': baseline, 'current_scan_bn': current}
        if i in failed or frame['positive']:
            for grid in sample['observation']['grids'].values():
                group = grid['group']
                contaminated = torch.bincount(group[sample['targets'] == 1], minlength=len(grid['cells'])) > 0
                flagged = torch.cat((contaminated, torch.tensor([False])))
                neighbors = grid['neighbors']
                grid['neighbors'] = torch.where(flagged[neighbors], len(grid['cells']), neighbors)
            with torch.no_grad():
                variants['oracle_exclude_anomaly_neighbors'] = model(sample).numpy()
        strata = []
        for name, mask in [('normal', target == 0), ('anomaly', target == 1),
                           ('inserted_normal', saved_points['inserted'] & (target == 0)),
                           ('normal_ge20', (target == 0) & (saved_points['range'] >= 20)),
                           ('missed_anomaly', (target == 1) & (baseline < analysis['threshold']))]:
            if not mask.any():
                continue
            strata.append(dict(name=name, points=int(mask.sum()),
                marginal_quantiles=[quantiles(marginal[mask, k]) for k in range(3)],
                variants={key: dict(**counts(target[mask], score[mask], analysis['threshold']),
                    quantiles=quantiles(score[mask]), change_from_baseline=quantiles(score[mask]-baseline[mask]))
                    for key, score in variants.items()}))
        output.append(dict(index=i, scene=frame['scene'], role=frame['role'], family=frame['family'],
                           baseline_max_abs_vs_panel=float(np.max(abs(baseline-saved_points['score']))), strata=strata))
        print(f'MECHANISM {number}/{len(indices)} | {time.perf_counter()-start:.1f}s', flush=True)
    if any(not torch.equal(value, buffers[name]) for name, value in model.named_buffers()):
        raise ValueError('Read-only case analysis failed to restore BN buffers')
    write_json(args.output / 'mechanism.json', dict(update=epoch['successful_updates'],
        checkpoint_sha256=analysis['checkpoint_sha256'], manifest_sha256=manifest['sha256'],
        threshold=analysis['threshold'], seconds=time.perf_counter()-start, batchnorm_layers=len(bn),
        scope='Error-selected mechanism cases, not population estimates. Fixed weights, full scans, FP32, CPU. '
              'BN-only intervention enables current-scan statistics while dropout stays off; buffers are restored. '
              'Oracle context intervention removes neighbors containing labeled anomaly points at all three scales; '
              'the backbone and queried points stay unchanged. It is unavailable without labels and is not a proposed inference rule.',
        cases=output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/data/sequence/val.json'))
    parser.add_argument('--epoch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mechanisms', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error('workers and threads must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This diagnostic requires an empty CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    manifest = load_manifest(args.manifest, 'val')
    epoch = json.loads(args.epoch.read_text())
    if epoch['validation']['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Validation population differs from the recorded checkpoint evaluation')
    if args.mechanisms:
        mechanisms(args, manifest, epoch)
        return
    threshold = epoch['validation']['metrics']['threshold']
    catalog = json.loads(Path('results/data/objects/catalog.json').read_text())
    family = {row['instance']: row['object_group'] for row in catalog['selected']}
    surfaces = json.loads(Path('results/data/sequence/sequences.json').read_text())['surfaces']
    categories = {key: row['category'] for key, row in surfaces.items()}
    del surfaces, catalog
    records = manifest['records']
    indices, pairs = select_panel(records, categories)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(args.checkpoint)
    model, saved = load_model(args.checkpoint, torch.device('cpu'))
    if saved['successful_updates'] != epoch['successful_updates']:
        raise ValueError('Checkpoint step differs from the requested diagnostic epoch')
    del saved, model
    gc.collect()
    dtype = np.dtype([('score', '<f4'), ('range', '<f4'), ('slot', '<u4'),
                      ('target', 'i1'), ('semantic', 'i1'), ('inserted', '?')])
    output_path = args.output / 'points.npy'
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite an existing diagnostic score population')
    points = np.lib.format.open_memmap(output_path, mode='w+', dtype=dtype,
                                      shape=(sum(records[i]['points'] for i in indices),))
    selected_roles = Counter('eligible' if records[i]['eligible'] else
                            'sparse' if records[i]['anomaly'] else records[i]['group'] for i in indices)
    print('PANEL', dict(selected_roles), 'scans', len(indices), 'bytes', points.nbytes, flush=True)
    frames, offset, repeated, worker_peak = [], 0, None, 0.
    start = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=init_worker,
                               initargs=(args.checkpoint, manifest, epoch['successful_updates'], args.threads))
    predictions = pool.map(infer, [(i, number == 0) for number, i in enumerate(indices)])
    for number, i in enumerate(indices, 1):
        if memory_available() < 6_000_000_000:
            raise RuntimeError('Insufficient spare RAM; protect the ongoing training')
        row = records[i]
        prediction, target, slots, distance, difference, peak = next(predictions)
        worker_peak = max(worker_peak, peak)
        if difference is not None:
            repeated = difference
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite CPU predictions')
        semantic = np.fromfile(row['label'], dtype=np.uint8)[slots].astype(np.int8)
        inserted = np.zeros(len(slots), bool)
        if row.get('delta'):
            with np.load(row['delta'], allow_pickle=False) as delta:
                # A replaced slot no longer has its original background semantics.
                changed = np.isin(slots, delta['slots'])
                valid_insertions = delta['slots'][delta['labels'] == (2 if row['anomaly'] else 1)]
                inserted = np.isin(slots, valid_insertions)
                semantic[changed] = -1
            expected = row['anomaly'] if row['anomaly'] else row['inserted_points']
            if int((inserted & (target >= 0)).sum()) != expected:
                raise ValueError('Inserted point identities do not match manifest supervision')
        stop = offset + len(target)
        chunk = points[offset:stop]
        for name, values in (('score', prediction), ('range', distance), ('slot', slots),
                             ('target', target), ('semantic', semantic), ('inserted', inserted)):
            chunk[name] = values
        role = 'eligible' if row['eligible'] else 'sparse' if row['anomaly'] else row['group']
        frame = dict(index=i, start=offset, stop=stop, role=role, scene=row['scene'], token=row['token'],
                     instance=row.get('instance'), family=family.get(row.get('instance')),
                     control_category=categories.get(row.get('donor')) if role == 'control_nuscenes' else None,
                     **counts(target, prediction, threshold),
                     normal_quantiles=quantiles(prediction[target == 0]),
                     anomaly_quantiles=quantiles(prediction[target == 1]),
                     inserted_normal=int((inserted & (target == 0)).sum()),
                     inserted_fp=int((inserted & (target == 0) & (prediction >= threshold)).sum()))
        frames.append(frame)
        offset = stop
        if number % 20 == 0 or number == len(indices):
            points.flush()
            print(f'CPU {number}/{len(indices)} scans | {time.perf_counter()-start:.1f}s | '
                  f'worker RSS peak {worker_peak:.0f} MiB', flush=True)
    pool.shutdown()
    points.flush()
    group_rows = group_counts(points, frames, manifest, threshold)
    with (args.output / 'groups.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dimension', 'group', 'positive', 'negative', 'tp', 'fp', 'frames'])
        writer.writeheader()
        writer.writerows(group_rows)
    by_index = {row['index']: row for row in frames}
    paired = []
    for control, original in pairs.items():
        f, g = by_index[control], by_index[original]
        a, b = points[f['start']:f['stop']], points[g['start']:g['stop']]
        slots, ia, ib = np.intersect1d(a['slot'], b['slot'], return_indices=True)
        unchanged = (a['semantic'][ia] >= 0) & (a['target'][ia] == 0) & (b['target'][ib] == 0)
        sa, sb = a['score'][ia[unchanged]], b['score'][ib[unchanged]]
        paired.append(dict(control=control, original=original, category=f['control_category'],
                           unchanged_points=len(sa), control_fp=int((sa >= threshold).sum()),
                           original_fp=int((sb >= threshold).sum()),
                           newly_flagged=int(((sa >= threshold) & (sb < threshold)).sum()),
                           no_longer_flagged=int(((sa < threshold) & (sb >= threshold)).sum()),
                           score_change_quantiles=quantiles(sa - sb),
                           inserted_normal=f['inserted_normal'], inserted_fp=f['inserted_fp']))
    full = [records[i] for i in evaluation_indices(manifest)]
    population = dict(scans=len(full), unique_tokens=len({r['token'] for r in full}),
                      normal=sum(r['normal'] for r in full), anomaly=sum(r['anomaly'] for r in full))
    population['positive_by_family'] = dict(Counter())
    population['positive_by_donor'] = dict(Counter())
    for r in full:
        for key, name in (('positive_by_family', family[r['instance']]), ('positive_by_donor', r['instance'])):
            population[key][name] = population[key].get(name, 0) + r['anomaly']
    population['positive_by_range_10m'] = np.sum([r['point_histogram'] for r in full], axis=0).tolist()
    write_json(args.output / 'analysis.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        update=epoch['successful_updates'], manifest_sha256=manifest['sha256'], threshold=threshold,
        scope='Exploratory stratified CPU panel; not full-validation metrics or an unbiased population estimate. '
              'No model, label, threshold, training state, or official result was modified. '
              'CPU and GPU field models both use FP32; cross-device roundoff was not measured.',
        selection='Two distinct-token scans per eligible donor x 10m median-range bin; one per sparse donor x bin; '
                  'two per inserted-normal category x dominant 10m range bin; prefer distinct scenes; '
                  'identity-hash ordering; each selected control paired to its original scan.',
        quantile_levels=[.01, .05, .5, .95, .99], point_count_edges=[5, 10, 20, 50, 100],
        population=population, selected_roles=dict(selected_roles), frames=frames, paired_controls=paired,
        cpu_repeat_max_abs=repeated, seconds=time.perf_counter()-start,
        workers=args.workers, threads_per_worker=args.threads, allowed_cpus=sorted(os.sched_getaffinity(0)),
        peak_worker_rss_mib=worker_peak,
        peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        point_scores='points.npy: all actual returns in frames/start:stop order, including ignored context'))
    print('DONE', args.output / 'analysis.json', flush=True)


if __name__ == '__main__':
    main()
