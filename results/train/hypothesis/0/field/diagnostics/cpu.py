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
import subprocess
import time

import numpy as np
import torch

from src.data import file_sha256, load_manifest, write_json
from src.evaluate import PreparedScans, evaluation_indices, load_model, memory_available
from vendor.stu.compute_point_level_ood import PointOODMetricsCalculator


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


def disk_available():
    """The host volume, not the virtual ext4 size, limits persistent recordings."""
    result = subprocess.run([
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-Command',
        'Get-Volume -DriveLetter E | Select-Object SizeRemaining | ConvertTo-Json -Compress'],
        check=True, capture_output=True, text=True, timeout=20)
    return int(json.loads(result.stdout)['SizeRemaining'])


def init_full_worker(checkpoint, manifest, update, threads, identity):
    global RECORDER, IDENTITY
    from layers import Recorder
    os.nice(19)
    if file_sha256(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('The preserved checkpoint changed')
    init_worker(checkpoint, manifest, update, threads)
    RECORDER, IDENTITY = Recorder(MODEL), identity


def infer_full(task):
    """Write one bounded frame locally; return no feature arrays through IPC."""
    import h5py
    index, path, verify = task
    if memory_available() < 6_000_000_000:
        raise RuntimeError('Insufficient spare RAM; protect the ongoing training')
    path = Path(path)
    temporary = path.with_suffix('.partial')
    if temporary.exists():
        temporary.unlink()  # Only a failed recording owned by this exact frame.
    start = time.perf_counter()
    sample = DATA[index]
    with torch.no_grad():
        baseline = MODEL(sample).numpy().copy() if verify else None
        RECORDER.start(sample, DATA.records[index], temporary)
        prediction = MODEL(sample)
        summary = RECORDER.finish(prediction)
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite CPU predictions')
    difference = float(np.max(abs(prediction.numpy() - baseline))) if verify else None
    if verify and difference != 0:
        raise ValueError(f'Recording changed same-thread CPU predictions: {difference}')
    with h5py.File(temporary, 'a') as handle:
        for key, value in IDENTITY.items():
            handle.attrs[key] = value
        handle.attrs['index'] = index
        handle.attrs['recording_max_abs_difference'] = difference if verify else np.nan
        handle.attrs['complete'] = True
    temporary.rename(path)
    summary['path'] = str(path)
    return dict(index=index, bytes=path.stat().st_size,
                seconds=time.perf_counter()-start, recording_max_abs_difference=difference,
                worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                recording=summary)


def aggregate_full(args, manifest, epoch, indices, identity, progress):
    import h5py
    records = manifest['records']
    count = sum(records[i]['normal'] + records[i]['anomaly'] for i in indices)
    if memory_available() < 64 * count + 1_000_000_000:
        raise RuntimeError('Insufficient RAM for exact official pooled sorting')
    scores, labels = np.empty(count, np.float32), np.empty(count, np.int8)
    catalog = json.loads(Path('results/data/objects/catalog.json').read_text())
    families = {row['instance']: row['object_group'] for row in catalog['selected']}
    threshold = epoch['validation']['metrics']['threshold']
    cursor, frames, groups = 0, [], defaultdict(Counter)
    layer_values, module_calls = {}, Counter()
    calculator = PointOODMetricsCalculator()
    dtype = np.dtype([('score', '<f4'), ('range', '<f4'), ('slot', '<u4'),
                      ('target', 'i1'), ('semantic', 'i1'), ('inserted', '?')])
    for number, i in enumerate(indices, 1):
        record = records[i]
        with h5py.File(args.output / 'frames' / f'{i:05d}.h5', 'r') as handle:
            if (not handle.attrs.get('complete', False) or handle.attrs.get('index') != i
                    or any(handle.attrs[k] != v for k, v in identity.items())):
                raise ValueError(f'Incomplete or mismatched frame {i}')
            xyz = handle['points/xyzi'][:, :3]
            prediction = handle['points/scores'][:]
            target, slots = handle['points/targets'][:], handle['points/slots'][:]
            names = handle['coverage/modules'].asstr()[:]
            module_calls.update(dict(zip(names, map(int, handle['coverage/calls'][:] ))))
            statistics, calls = handle['statistics/channels'][:], handle['statistics/calls'][:]
            seen = set()
            for call in calls:
                begin, end = int(call['start']), int(call['stop'])
                if (begin, end) in seen:
                    continue  # Repeated invocations reference an aggregate, not new values.
                seen.add((begin, end))
                name = names[int(call['module'])]
                values = statistics[begin:end]
                summary = layer_values.setdefault(name, dict(values=0, nonfinite=0, minimum=None, maximum=None))
                summary['values'] += int(values['count'].sum())
                summary['nonfinite'] += int((values['count'] - values['finite']).sum())
                finite = values['finite'] > 0
                if finite.any():
                    lo, hi = float(values['min'][finite].min()), float(values['max'][finite].max())
                    summary['minimum'] = lo if summary['minimum'] is None else min(lo, summary['minimum'])
                    summary['maximum'] = hi if summary['maximum'] is None else max(hi, summary['maximum'])
        if len(target) != record['points'] or int((target == 1).sum()) != record['anomaly'] or int((target == 0).sum()) != record['normal']:
            raise ValueError(f'Point population changed in frame {i}')
        # Use the same official mask and >=5 anomaly rule as the GPU evaluation.
        calculator.update(xyz, prediction, np.where(target < 0, 0, target + 1))
        selected_scores, selected_labels = calculator.all_scores.pop(), calculator.all_labels.pop()
        stop = cursor + len(selected_scores)
        scores[cursor:stop], labels[cursor:stop] = selected_scores, selected_labels
        p = np.empty(len(target), dtype=dtype)
        p['score'], p['range'], p['slot'], p['target'] = prediction, np.linalg.norm(xyz, axis=1), slots, target
        p['semantic'] = np.fromfile(record['label'], np.uint8)[slots]
        p['inserted'] = False
        with np.load(record['delta'], allow_pickle=False) as delta:
            p['semantic'][np.isin(slots, delta['slots'])] = -1
            p['inserted'] = np.isin(slots, delta['slots'][delta['labels'] == 2])
        frame = dict(index=i, start=0, stop=len(p), role='eligible', scene=record['scene'],
                     token=record['token'], instance=record['instance'], family=families[record['instance']],
                     metric_start=cursor, metric_stop=stop,
                     **counts(target, prediction, threshold),
                     normal_quantiles=quantiles(prediction[target == 0]),
                     anomaly_quantiles=quantiles(prediction[target == 1]))
        frames.append(frame)
        for row in group_counts(p, [frame], manifest, threshold):
            key = row.pop('dimension'), row.pop('group')
            groups[key].update(row)
        cursor = stop
        if number % 200 == 0 or number == len(indices):
            print(f'AGGREGATE {number}/{len(indices)} | {cursor}/{count} metric points', flush=True)
    if cursor != count or int(labels.sum()) != sum(records[i]['anomaly'] for i in indices):
        raise ValueError('Official pooled evaluation population changed')
    calculator.all_scores, calculator.all_labels = [scores], [labels]
    metrics = {key: float(value) for key, value in calculator.compute_metrics().items()}
    with (args.output / 'groups.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dimension', 'group', 'positive', 'negative', 'tp', 'fp', 'frames'])
        writer.writeheader()
        writer.writerows(dict(dimension=key[0], group=key[1], **values) for key, values in sorted(groups.items()))
    progress['status'] = 'complete'
    result = dict(**identity, status='complete', scans=len(indices), points=count,
                  actual_points=sum(records[i]['points'] for i in indices),
                  normal=int((labels == 0).sum()), anomaly=int(labels.sum()), metrics=metrics,
                  gpu_metrics=epoch['validation']['metrics'],
                  cpu_minus_gpu={key: value - epoch['validation']['metrics'][key] for key, value in metrics.items()},
                  strata_threshold=threshold, frames=frames, execution=progress,
                  layer_numerics=layer_values, module_calls=dict(module_calls),
                  scope='Complete 1796 official-eligible nuScenes frames, unchanged FP32 model and official pooled metrics. '
                        'Layer recording is passive. Full raw activations are NOT retained: see per-frame coverage. '
                        'GPU midpoint per-point scores were not retained; CPU/GPU equality is not established by aggregate agreement.')
    write_json(args.output / 'analysis.json', result)
    print('COMPLETE', json.dumps(metrics), flush=True)


def full_run(args, manifest, epoch):
    import h5py
    indices = evaluation_indices(manifest)
    if len(indices) != epoch['validation']['scans']:
        raise ValueError('Official frame population differs from the midpoint evaluation')
    # Source checks are limited to the unchanged inference and label-reading path.
    configuration = json.loads((args.epoch.parent / 'config.json').read_text())['configuration']
    for path, digest in configuration['code']['files'].items():
        if path in ('src/data.py', 'src/model.py', 'src/normal.py', 'src/evaluate.py') or path.startswith(('vendor/litept/', 'vendor/stu/')):
            if file_sha256(path) != digest:
                raise ValueError(f'Inference source changed since training: {path}')
    identity = dict(checkpoint_sha256=file_sha256(args.checkpoint), manifest_sha256=manifest['sha256'],
                    recording_sha256=file_sha256(Path(__file__).with_name('layers.py')),
                    update=epoch['successful_updates'], threads=args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'frames').mkdir(exist_ok=True)
    jobs, completed = [], []
    for i in indices:
        path = args.output / 'frames' / f'{i:05d}.h5'
        if path.exists():
            with h5py.File(path, 'r') as handle:
                if (not handle.attrs.get('complete', False) or handle.attrs.get('index') != i
                        or any(handle.attrs.get(k) != v for k, v in identity.items())):
                    raise ValueError(f'Refusing to reuse incomplete or mismatched frame {path}')
            completed.append(dict(index=i, bytes=path.stat().st_size, resumed=True))
        else:
            jobs.append((i, str(path), not completed and not jobs))
    free = disk_available()
    # The declared budget bounds recordings; 2 GB covers concurrent training writes,
    # final aggregation, and a few in-flight frames before a capacity stop.
    stored = sum(row['bytes'] for row in completed)
    if free < 10_000_000_000 + 2_000_000_000 + max(0, args.storage_gb * 1e9 - stored):
        raise RuntimeError('Recording budget would invade the 10 GB host-volume reserve')
    start = time.perf_counter()
    progress = dict(**identity, status='running', total=len(indices), completed=len(completed),
                    workers=args.workers, allowed_cpus=sorted(os.sched_getaffinity(0)),
                    recording_budget_bytes=int(args.storage_gb * 1e9), host_free_at_start=free,
                    frames=completed)
    def report():
        progress.update(completed=len(completed), stored_bytes=sum(row['bytes'] for row in completed),
                        seconds=time.perf_counter()-start, frames=completed)
        write_json(args.output / 'progress.json', progress)
        print(f'FULL {len(completed)}/{len(indices)} | {progress["seconds"]:.0f}s | '
              f'{progress["stored_bytes"]/1e9:.2f} GB | spare RAM {memory_available()/1e9:.1f} GB', flush=True)
    report()
    # executor.map on 3.13 eagerly queues the iterable; explicitly bound submissions.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=init_full_worker,
            initargs=(args.checkpoint, manifest, epoch['successful_updates'], args.threads, identity)) as pool:
        pending, cursor = [], 0
        while cursor < len(jobs) or pending:
            while cursor < len(jobs) and len(pending) < args.workers:
                pending.append(pool.submit(infer_full, jobs[cursor]))
                cursor += 1
            completed.append(pending.pop(0).result())
            if len(completed) % 10 == 0 or not pending and cursor == len(jobs):
                report()
                if progress['stored_bytes'] > args.storage_gb * 1e9:
                    raise RuntimeError('Recording budget exceeded; completed frames are preserved')
                if disk_available() < 12_000_000_000:
                    raise RuntimeError('Host volume approaching safety reserve; completed frames are preserved')
    progress['status'] = 'aggregating'
    report()
    aggregate_full(args, manifest, epoch, indices, identity, progress)
    progress['status'] = 'complete'
    report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/data/sequence/val.json'))
    parser.add_argument('--epoch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mechanisms', action='store_true')
    parser.add_argument('--full', action='store_true', help='Recompute every official-eligible frame with layer records')
    parser.add_argument('--storage-gb', type=float, default=60)
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
    if args.full:
        full_run(args, manifest, epoch)
        return
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
