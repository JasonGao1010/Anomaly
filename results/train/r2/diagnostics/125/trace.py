import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/jasongao/Study/AJAE-v4')
sys.path.insert(0, str(ROOT))
from src.data import Scans, load_manifest, file_sha256
from src.evaluate import load_model, autocast
from src.model import prepare_scan, to_device, GRID_SIZE

OUT = ROOT / 'results/train/r2/diagnostics/125'
FRAMES = (145, 146, 147)


def array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


class Trace:
    def __init__(self, model, sample, selected, shift):
        self.ids = torch.as_tensor(selected, device=sample['xyzi'].device)
        self.ancestors = {0: sample['inverse'][self.ids]}
        self.shift = shift
        self.features = {}
        self.chunks = {}
        self.cursors = {}
        self.patches = {}
        self.levels = {}
        self.handles = []
        self.point_feature('input_xyzi', sample['xyzi'], self.ids)
        self.point_feature('input_offset', sample['offset'], self.ids)
        self.handles.append(model.detail.register_forward_hook(self.chunk('detail')))
        self.handles.append(model.adapter.register_forward_pre_hook(
            lambda module, args: self.point_feature('voxel_pool', args[0], self.ancestors[0])))
        self.handles.append(model.adapter.register_forward_hook(
            lambda module, args, value: self.point_feature('adapter', value, self.ancestors[0])))
        self.handles.append(model.backbone.embedding.register_forward_hook(self.point('embedding', 0)))
        for level in range(5):
            stage = model.backbone.enc._modules[f'enc{level}']
            if level:
                self.handles.append(stage._modules['down'].register_forward_hook(self.down(level)))
            else:
                self.handles.append(stage.register_forward_pre_hook(
                    lambda module, args: self.point_feature('encoder_input', args[0].feat, self.ancestors[0])))
            for name, block in stage.named_children():
                if not name.startswith('block'):
                    continue
                key = f'enc{level}.{name}'
                self.handles.append(block.register_forward_hook(self.point(key, level)))
                if block.enable_attn:
                    self.handles.append(block.attn.register_forward_pre_hook(self.attention(key, level)))
                    self.handles.append(block.attn.register_forward_hook(self.point(key + '.attn_output', level)))
            self.handles.append(stage.register_forward_hook(self.point(f'enc{level}', level)))
        for level in reversed(range(4)):
            stage = model.backbone.dec._modules[f'dec{level}']
            self.handles.append(stage.register_forward_hook(self.point(f'dec{level}', level)))
        self.handles.append(model.context.register_forward_hook(
            lambda module, args, value: self.point_feature('context', value, self.ancestors[0])))
        self.handles.append(model.sampling.register_forward_hook(self.chunk('sampling')))
        self.handles.append(model.head[0].register_forward_pre_hook(self.chunk('head_input', pre=True)))
        self.handles.append(model.head[0].register_forward_hook(self.chunk('head_linear')))
        self.handles.append(model.head[1].register_forward_hook(self.chunk('head_gelu')))
        self.handles.append(model.head[2].register_forward_hook(self.chunk('score')))

    def point_feature(self, name, value, ids):
        self.features[name] = array(value[ids])

    def chunk(self, name, pre=False):
        def capture(module, args, value=None):
            value = args[0] if pre else value
            start = self.cursors.get(name, 0)
            chosen = self.ids[(self.ids >= start) & (self.ids < start + len(value))] - start
            self.chunks.setdefault(name, []).append(array(value[chosen]))
            self.cursors[name] = start + len(value)
        return capture

    def point(self, name, level):
        def capture(module, args, point):
            self.point_feature(name, point.feat, self.ancestors[level])
        return capture

    def down(self, level):
        def capture(module, args, point):
            self.ancestors[level] = point.pooling_inverse[self.ancestors[level - 1]]
            self.point_feature(f'enc{level}.pool', point.feat, self.ancestors[level])
            self.levels[level] = {
                'voxels': len(point.feat),
                'grid': point.grid_coord[self.ancestors[level]].cpu().numpy().copy() + self.shift // (2 ** level),
                'coord': array(point.coord[self.ancestors[level]]),
            }
        return capture

    def attention(self, name, level):
        def capture(module, args):
            point = args[0]
            ids = self.ancestors[level]
            self.point_feature(name + '.attn_input', point.feat, ids)
            pad, unpad, _ = point.get_padding_and_inverse(module.patch_size)
            order = point.serialized_order[module.order_index][pad]
            groups = (unpad[point.serialized_inverse[module.order_index]][ids] // module.patch_size).cpu().numpy()
            grid = point.grid_coord.cpu().numpy().astype(np.int64) + self.shift // (2 ** level)
            order = order.cpu().numpy()
            members = {int(group): grid[order[int(group) * module.patch_size:(int(group) + 1) * module.patch_size]].copy()
                       for group in np.unique(groups)}
            self.patches[name] = {'groups': groups, 'members': members, 'order_index': module.order_index,
                                  'level': level, 'serialized_depth': int(point.serialized_depth),
                                  'patch_size': module.patch_size}
        return capture

    def finish(self):
        for handle in self.handles:
            handle.remove()
        for name, chunks in self.chunks.items():
            self.features[name] = np.concatenate(chunks)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.2, device)
    checkpoint = ROOT / 'results/train/r2/0/base/best.pt'
    model, saved = load_model(checkpoint, device)
    assert saved['mode'] == 'base'
    manifest = load_manifest(ROOT / 'assets/val.json', 'val')
    dataset = Scans(manifest)
    baseline = np.load(OUT / 'scores.npz')
    matching = np.load(OUT / 'matches.npz')
    traces, arrays, records = {}, {}, []
    start = time.perf_counter()
    for fid in FRAMES:
        index = next(i for i, row in enumerate(manifest['records']) if row['sequence'] == 125 and row['frame'] == fid)
        sample = dataset[index]
        selected = np.flatnonzero(sample['targets'] == 1)
        grid = np.floor(sample['xyzi'][:, :3].astype(np.float64) / GRID_SIZE).astype(np.int64)
        shift = (grid.min(0) // 16) * 16
        batch = to_device(prepare_scan(sample), device)
        trace = Trace(model, batch, selected, shift)
        with torch.inference_mode(), autocast(device):
            scores = model(batch).cpu().numpy()
        trace.finish()
        previous = baseline[str(fid)][sample['slots']]
        delta = np.abs(scores - previous)
        if not np.array_equal(scores, previous):
            delta = np.abs(scores - previous)
            print('observation_difference', json.dumps({'frame': fid, 'count': int(np.count_nonzero(delta)),
                  'max': float(delta.max()), 'anomaly_max': float(delta[selected].max()),
                  'FN': int(np.sum(scores[selected] < -7.96875))}), flush=True)
            with torch.inference_mode(), autocast(device):
                plain = model(batch).cpu().numpy()
            print('without_hooks_difference', json.dumps({'previous_max': float(np.max(np.abs(plain - previous))),
                  'observed_max': float(np.max(np.abs(plain - scores))),
                  'observed_count': int(np.count_nonzero(plain != scores))}), flush=True)
            assert np.array_equal(scores, plain), 'observation changed predictions'
        for name, value in trace.features.items():
            arrays[f'{fid}:{name}'] = value
        arrays[f'{fid}:full_scores'] = scores
        arrays[f'{fid}:ids'] = selected
        arrays[f'{fid}:xyz'] = sample['xyzi'][selected, :3]
        for level, row in trace.levels.items():
            arrays[f'{fid}:level{level}:grid'] = row['grid']
            arrays[f'{fid}:level{level}:coord'] = row['coord']
        for name, patch in trace.patches.items():
            arrays[f'{fid}:{name}:groups'] = patch['groups']
            for group, member in patch['members'].items():
                arrays[f'{fid}:{name}:members{group}'] = member
        record = {'frame': fid, 'raw_return_points': len(scores), 'anomaly_points': len(selected),
                  'grid_shift_5cm_units': shift.tolist(), 'voxels': len(batch['grid']),
                  'coarse_voxels': {level: row['voxels'] for level, row in trace.levels.items()},
                  'observed_prediction_exactly_matches_unobserved': True,
                  'archived_score_difference': {'changed_points': int(np.count_nonzero(delta)),
                      'max_abs': float(delta.max()), 'anomaly_max_abs': float(delta[selected].max()),
                      'archived_FN': int(np.sum(previous[selected] < -7.96875)),
                      'current_FN': int(np.sum(scores[selected] < -7.96875))},
                  'serialized_depth': {name: p['serialized_depth'] for name, p in trace.patches.items()}}
        records.append(record)
        traces[fid] = trace
        print(json.dumps(record), flush=True)
        del batch
    # Diagnostic intervention: alter only serialization coordinates, then restore
    # physical grids before positional encoding and sparse operations consume them.
    common_shift = np.min([r['grid_shift_5cm_units'] for r in records], axis=0)
    controls = []
    for fid, record in zip(FRAMES, records):
        index = next(i for i, row in enumerate(manifest['records']) if row['sequence'] == 125 and row['frame'] == fid)
        sample = dataset[index]
        selected = arrays[f'{fid}:ids']
        shift = np.asarray(record['grid_shift_5cm_units'])
        batch = to_device(prepare_scan(sample), device)
        handles = []
        for level in (3, 4):
            def change_origin(module, args, point, level=level):
                original_grid = point.grid_coord
                delta = torch.as_tensor((shift - common_shift) // (2 ** level), device=original_grid.device)
                point.grid_coord = original_grid + delta
                point.serialization(order=point.order, depth=point.serialized_depth, shuffle_orders=False)
                point.grid_coord = original_grid
            handles.append(model.backbone.enc._modules[f'enc{level}']._modules['down'].register_forward_hook(change_origin))
        trace = Trace(model, batch, selected, shift)
        with torch.inference_mode(), autocast(device):
            scores = model(batch).cpu().numpy()
        trace.finish()
        for handle in handles:
            handle.remove()
        early = ('input_xyzi', 'input_offset', 'detail', 'voxel_pool', 'embedding', 'adapter',
                 'encoder_input', 'enc0', 'enc1', 'enc2', 'enc3.pool')
        unchanged = {name: bool(np.array_equal(trace.features[name], traces[fid].features[name])) for name in early}
        assert all(unchanged.values()), unchanged
        if np.array_equal(shift, common_shift):
            assert np.array_equal(scores, arrays[f'{fid}:full_scores']), 'zero-change control differs'
        previous = arrays[f'{fid}:full_scores']
        normal = sample['targets'] == 0
        row = {'frame': fid, 'common_shift_5cm_units': common_shift.tolist(),
               'serialization_origin_delta_5cm_units': (shift - common_shift).tolist(),
               'early_features_unchanged': unchanged,
               'original_FN': int(np.sum(previous[selected] < -7.96875)),
               'changed_FN': int(np.sum(scores[selected] < -7.96875)),
               'original_FP': int(np.sum(previous[normal] >= -7.96875)),
               'changed_FP': int(np.sum(scores[normal] >= -7.96875)),
               'original_anomaly_score_median': float(np.median(previous[selected])),
               'changed_anomaly_score_median': float(np.median(scores[selected])),
               'all_score_max_abs_change': float(np.max(np.abs(scores - previous)))}
        controls.append(row)
        arrays[f'{fid}:control_scores'] = scores
        for name in ('enc3', 'enc4', 'context', 'sampling', 'score'):
            arrays[f'{fid}:control:{name}'] = trace.features[name]
        print('control', json.dumps(row), flush=True)
        del batch
    rows, neighborhoods = [], []
    for a, b in ((145, 146), (146, 147), (145, 147)):
        ia = np.searchsorted(arrays[f'{a}:ids'], matching[f'{a}_{b}_a'])
        ib = np.searchsorted(arrays[f'{b}:ids'], matching[f'{a}_{b}_b'])
        for name in traces[a].features:
            fa = np.atleast_2d(traces[a].features[name][ia]).astype(np.float64)
            fb = np.atleast_2d(traces[b].features[name][ib]).astype(np.float64)
            delta = np.linalg.norm(fa - fb, axis=1)
            scale = np.sqrt(.5 * (np.mean(fa ** 2) + np.mean(fb ** 2)))
            na, nb = np.linalg.norm(fa, axis=1), np.linalg.norm(fb, axis=1)
            cosine = np.sum(fa * fb, axis=1) / np.maximum(na * nb, 1e-12)
            rows.append({'frames': f'{a}-{b}', 'layer': name, 'pairs': len(ia),
                         'relative_rms_difference': float(np.sqrt(np.mean((fa - fb) ** 2)) / max(scale, 1e-12)),
                         'median_cosine_similarity': float(np.median(cosine)),
                         'rms_feature_a': float(np.sqrt(np.mean(fa ** 2))),
                         'rms_feature_b': float(np.sqrt(np.mean(fb ** 2))),
                         'median_vector_difference': float(np.median(delta))})
        for name, pa in traces[a].patches.items():
            pb = traces[b].patches[name]
            sa = {k: set(map(tuple, v)) for k, v in pa['members'].items()}
            sb = {k: set(map(tuple, v)) for k, v in pb['members'].items()}
            overlaps = [len(sa[int(ga)] & sb[int(gb)]) / len(sa[int(ga)] | sb[int(gb)])
                        for ga, gb in zip(pa['groups'][ia], pb['groups'][ib])]
            level = pa['level']
            same = np.all(traces[a].levels[level]['grid'][ia] == traces[b].levels[level]['grid'][ib], axis=1)
            neighborhoods.append({'frames': f'{a}-{b}', 'layer': name,
                                  'same_physical_coarse_cell_pct': 100 * float(np.mean(same)),
                                  'patch_jaccard_quantiles': np.quantile(overlaps, [0, .1, .5, .9, 1]).tolist()})
    with (OUT / 'layers.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT / 'trace.npz', **arrays)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
              'matching': 'same-class mutual spatial nearest neighbors <= 5 cm; source-return IDs only locate matched points within each frame',
              'feature_difference': 'RMS(fa-fb)/sqrt((mean(fa^2)+mean(fb^2))/2)',
              'neighborhood_jaccard': 'intersection/union of unique physical coarse-grid coordinates in the actual padded attention groups',
              'frames': records, 'neighborhoods': neighborhoods, 'controls': controls,
              'control_description': 'Only deep-stage serialization uses a common integer coordinate origin. XYZ, intensity, voxel membership, physical grids and PointROPE coordinates remain unchanged.',
              'numerical_caveat': 'Fresh-process inference differs slightly from archived scores; observer hooks match the unobserved model exactly. All interventions compare against the same-process baseline; official stored scores are unchanged.',
              'seconds': time.perf_counter() - start,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(device)}
    (OUT / 'trace.json').write_text(json.dumps(report, indent=2) + '\n')
    print('layers', json.dumps([r for r in rows if r['layer'] in ('detail', 'voxel_pool', 'embedding', 'encoder_input', 'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec3', 'dec2', 'dec1', 'dec0', 'context', 'sampling', 'head_linear', 'score')]), flush=True)
    print('neighborhoods', json.dumps(neighborhoods), flush=True)
    print('complete', report['seconds'], report['peak_cuda_bytes'], flush=True)


if __name__ == '__main__':
    main()
