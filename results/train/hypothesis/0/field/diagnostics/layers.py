"""Read-only, identity-preserving activation recording for the field segmentor.

All evaluated points retain the complete 163-dimensional classifier input. Other
layers retain full-population channel statistics and exact vectors for every
anomaly plus a score-independent, range-stratified normal sample. Hooks never
replace outputs, weights, buffers, or functional implementations.
"""

from collections import Counter
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch


STAT_DTYPE = np.dtype([(key, '<u8' if key in ('count', 'finite') else
                       '<f4' if key in ('min', 'max') else '<f8')
                      for key in ('count', 'finite', 'min', 'max', 'mean', 'std')])
CALL_DTYPE = np.dtype([('module', '<i4'), ('call', '<i4'), ('output', '<i4'),
                      ('count', '<i8'), ('start', '<i8'), ('stop', '<i8')])
VECTOR_DTYPE = np.dtype([('capture', '<i4'), ('point', '<i8'), ('row', '<i8'),
                        ('start', '<i8'), ('stop', '<i8')])


def array(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def output_tensors(value):
    """Point metadata is not an activation; summarize its current features once."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict) and 'feat' in value:
        return [value['feat']]
    if hasattr(value, 'features') and isinstance(value.features, torch.Tensor):
        return [value.features]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in output_tensors(item)]
    return []


def channel_statistics(tensor):
    value = array(tensor)
    # A score vector has one scalar channel, not one channel per scan point.
    flat = value.reshape(-1, value.shape[-1] if value.ndim >= 2 else 1)
    result = np.empty(flat.shape[1], dtype=STAT_DTYPE)
    result['count'] = len(flat)
    finite = np.isfinite(flat)
    result['finite'] = finite.sum(0)
    if finite.all():
        result['min'], result['max'] = flat.min(0), flat.max(0)
        result['mean'] = flat.mean(0, dtype=np.float64)
        result['std'] = flat.std(0, dtype=np.float64)
    else:
        # Nonfinite values are counted explicitly, never silently imputed as zero.
        values = np.where(finite, flat, np.nan).astype(np.float64)
        with np.errstate(invalid='ignore'):
            result['min'], result['max'] = np.nanmin(values, 0), np.nanmax(values, 0)
            result['mean'], result['std'] = np.nanmean(values, 0), np.nanstd(values, 0)
    return result


class Recorder:
    def __init__(self, model):
        if model.mode != 'field' or next(model.parameters()).device.type != 'cpu':
            raise ValueError('Recording requires the unchanged CPU field model')
        self.model = model
        self.modules = dict(model.named_modules())
        self.names = list(self.modules)
        self.ids = {name: index for index, name in enumerate(self.names)}
        self.file = None
        self.active = False
        self.handles = []
        self.boundaries = {'Block', 'GridPooling', 'GridUnpooling', 'PointROPEAttention',
                           'Embedding', 'Conditional', 'Compatibility', 'NormalField', 'Segmentor'}
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(
                lambda mod, inputs, output, key=name: self._after(key, mod, inputs, output)))
        for name, callback in (('detail', self._detail), ('normal.encoder', self._normal_encoder),
                               ('point_detail', self._query), ('head', self._head)):
            self.handles.append(self.modules[name].register_forward_pre_hook(callback))

    def _write(self, name, value):
        value = array(value)
        self.logical_bytes += value.nbytes
        if self.logical_bytes > 1_000_000_000:
            raise RuntimeError('A frame would exceed the one-GB uncompressed recording bound')
        parent, _, leaf = name.rpartition('/')
        group = self.file.require_group(parent) if parent else self.file
        options = dict(compression='lzf', shuffle=True) if value.ndim and value.size else {}
        return group.create_dataset(leaf, data=value, **options)

    def start(self, sample, record, output_path):
        if self.file is not None:
            raise RuntimeError('Finish or close the preceding frame first')
        if self.model.training:
            raise ValueError('The recorder does not change model mode; call eval first')
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, 'x')
        self.active = True
        self.file.attrs['complete'] = False
        self.file.attrs['record'] = json.dumps(record, separators=(',', ':'))
        self.file.attrs['schema'] = 'field-activation-recording-1'
        self.sample = sample
        self.logical_bytes = 0
        self.calls, self.summaries, self.summary_calls, self.summary_shapes = Counter(), [], [], []
        self.summary_outputs, self.channel_cursor, self.accumulators = [], 0, {}
        self.vectors, self.vector_rows, self.vector_calls, self.vector_cursor = [], [], [], 0
        self.detail_cursor = self.normal_cursor = self.query_cursor = self.metric_cursor = 0
        self.query_indices = self.detail_indices = self.normal_indices = np.empty(0, np.int64)
        self.n = len(sample['xyzi'])
        target, xyz, slots = array(sample['targets']), array(sample['xyzi'])[:, :3], array(sample['slots'])
        distance = np.linalg.norm(xyz, axis=1)
        self.valid = (target >= 0) & (distance >= 2.5) & (distance <= 50.)
        selected = [np.flatnonzero(target == 1)]
        boundaries = [2.5, 10., 20., 30., 40., np.nextafter(50., np.inf)]
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            eligible = np.flatnonzero((target == 0) & (distance >= lower) & (distance < upper))
            eligible = eligible[np.argsort(slots[eligible], kind='stable')]
            if len(eligible):
                selected.append(eligible[np.linspace(0, len(eligible) - 1, min(16, len(eligible)), dtype=int)])
        self.selected = np.unique(np.concatenate(selected))
        self.ancestors = [array(sample['inverse']).copy()]
        for key in ('xyzi', 'slots', 'targets', 'inverse', 'grid', 'voxel_xyzi', 'offset'):
            self._write(('points/' if key in ('xyzi', 'slots', 'targets') else 'input/') + key, sample[key])
        self._write('points/range', distance)
        self._write('head/valid_rows', np.flatnonzero(self.valid))
        self._write('selected/point_index', self.selected)
        self._write('selected/raw_slot', slots[self.selected])
        self.file['selected'].attrs['rule'] = ('All actual points with target=1; up to 16 target=0 points '
            'per [2.5,10),[10,20),[20,30),[30,40),[40,50]m bin, equally spaced in raw-slot order; '
            'independent of scores. A feature may be shared by mixed-label points; no voxel labels are assigned.')
        self._write('mapping/level0/coord', sample['voxel_xyzi'][:, :3])
        self.file['mapping/level0/grid'] = self.file['input/grid']
        self.file['mapping/level0/point_ancestor'] = self.file['input/inverse']
        for size, grid in sample['observation']['grids'].items():
            for key in ('cells', 'group', 'neighbors', 'position', 'basis'):
                self._write(f'angular/{size}/{key}', grid[key])
            self.file[f'angular/{size}'].attrs['missing_block'] = len(grid['cells'])
            self.file[f'angular/{size}'].attrs['learned_empty_block'] = len(grid['cells']) + 1
        count = int(self.valid.sum())
        self.head = self.file.create_dataset('head/input', shape=(count, 163), dtype='<f4',
            chunks=(min(1024, max(count, 1)), 163), compression='lzf', shuffle=True)
        self.probability = self.file.create_dataset('head/log_probability', shape=(self.n, 3, 4), dtype='<f4',
            chunks=(min(1024, self.n), 3, 4), compression='lzf', shuffle=True)
        self.logical_bytes += count * 163 * 4 + self.n * 3 * 4 * 4
        self.file['head'].attrs['head_row_identity'] = 'head/valid_rows indexes points/slots and points/xyzi'
        self.file['head'].attrs['probability_row_identity'] = 'Every input actual point in points/slots order'
        self.file['head'].attrs['head_columns'] = 'state[0:64]; scale1[64:96]; scale2[96:128]; scale4[128:160]; marginal_log_density[160:163]'
        self.file['head'].attrs['probability_axes'] = 'actual_point, scale=(1,2,4), hypothesis=(0,1,2,3)'

    def _detail(self, module, inputs):
        if self.active:
            count = len(inputs[0])
            self.detail_indices = np.arange(self.detail_cursor, self.detail_cursor + count)
            self.detail_cursor += count

    def _normal_encoder(self, module, inputs):
        if self.active:
            count = len(inputs[0])
            self.normal_indices = np.arange(self.normal_cursor, self.normal_cursor + count)
            self.normal_cursor += count

    def _query(self, module, inputs):
        if self.active:
            count = len(inputs[0])
            self.query_indices = np.arange(self.query_cursor, self.query_cursor + count)
            self.query_cursor += count

    def _head(self, module, inputs):
        if self.active:
            value = array(inputs[0])
            if value.shape != (len(self.query_indices), 163):
                raise ValueError('Classifier input no longer has the expected point identity')
            selected = self.valid[self.query_indices]
            count = int(selected.sum())
            self.head[self.metric_cursor:self.metric_cursor + count] = value[selected]
            self.metric_cursor += count

    def _selected_rows(self, point_indices):
        # Input rows need not be unique (coarse voxels); original point IDs remain explicit.
        mask = np.isin(self.selected, point_indices)
        chosen = self.selected[mask]
        return chosen, np.searchsorted(point_indices, chosen)

    def _vectors(self, name, tensor, points, rows, call, output=0):
        if not len(points):
            return
        if tensor.dtype != torch.float32:
            raise ValueError(f'Exact selected activation unexpectedly changed dtype: {name}: {tensor.dtype}')
        value = array(tensor)[rows].copy()
        flat = value.reshape(len(rows), -1)
        capture = len(self.vector_calls)
        self.vector_calls.append(dict(module=name, call=call, output=output, shape=list(value.shape),
                                      row_identity='row in the unselected module output; point is input actual index'))
        begin = self.vector_cursor + np.arange(len(rows)) * flat.shape[1]
        mapping = np.empty(len(rows), dtype=VECTOR_DTYPE)
        mapping['capture'], mapping['point'], mapping['row'] = capture, points, rows
        mapping['start'], mapping['stop'] = begin, begin + flat.shape[1]
        self.vector_rows.append(mapping)
        self.vectors.append(flat.reshape(-1))
        self.vector_cursor += flat.size
        if self.logical_bytes + self.vector_cursor * 4 > 1_000_000_000:
            raise RuntimeError('Selected vectors would exceed the one-GB per-frame bound')

    def _statistics(self, name, tensor, call, output):
        statistics = channel_statistics(tensor)
        key = (self.ids[name], output, len(statistics))
        count = int(statistics['count'][0])
        if key not in self.accumulators:
            start = self.channel_cursor
            stop = start + len(statistics)
            self.accumulators[key] = (statistics,
                np.where(statistics['finite'] > 0, statistics['std'] ** 2 * statistics['finite'], 0.), start, stop)
            self.summaries.append(statistics)
            self.channel_cursor = stop
        else:
            previous, second_moment, start, stop = self.accumulators[key]
            first, second = previous['finite'].astype(np.float64), statistics['finite'].astype(np.float64)
            total = first + second
            both = (first > 0) & (second > 0)
            delta = np.where(both, statistics['mean'] - previous['mean'], 0.)
            # Parallel Welford combination preserves within-call and between-call variance.
            second_moment += np.where(second > 0, statistics['std'] ** 2 * second, 0.)
            second_moment += delta ** 2 * first * second / np.maximum(total, 1.)
            previous['mean'] = np.where(first == 0, statistics['mean'],
                previous['mean'] + delta * second / np.maximum(total, 1.))
            previous['std'] = np.where(total > 0, np.sqrt(second_moment / np.maximum(total, 1.)), np.nan)
            previous['min'] = np.fmin(previous['min'], statistics['min'])
            previous['max'] = np.fmax(previous['max'], statistics['max'])
            previous['count'] += statistics['count']
            previous['finite'] += statistics['finite']
        _, _, start, stop = self.accumulators[key]
        self.summary_calls.append((self.ids[name], call, output, count, start, stop))
        self.summary_shapes.append(json.dumps(list(tensor.shape), separators=(',', ':')))

    def _after(self, name, module, inputs, output):
        if not self.active:
            return
        self.calls[name] += 1
        call = self.calls[name] - 1
        kind = type(module).__name__
        numerical = not list(module.children()) or kind in self.boundaries or name in ('detail', 'adapter', 'head')
        tensors = output_tensors(output)
        if numerical:
            for number, tensor in enumerate(tensors):
                if not tensor.is_floating_point() or not tensor.numel():
                    continue
                self._statistics(name, tensor, call, number)
        if kind == 'GridPooling':
            level = int(re.search(r'\.enc(\d+)\.', name).group(1))
            inverse = array(output['pooling_inverse']).copy()
            if len(self.ancestors) != level:
                raise ValueError('Unexpected encoder pooling order')
            self.ancestors.append(inverse[self.ancestors[-1]])
            for key, value in (('parent_to_child', inverse), ('point_ancestor', self.ancestors[-1]),
                               ('coord', output['coord']), ('grid', output['grid_coord'])):
                self._write(f'mapping/level{level}/{key}', value)
            if 'serialized_order' in output:
                for key in ('serialized_order', 'serialized_inverse'):
                    self._write(f'mapping/level{level}/{key}', output[key])
        if kind == 'PointROPEAttention':
            level = int(re.search(r'\.enc(\d+)\.', name).group(1))
            location = f'mapping/level{level}'
            if location + '/pad' not in self.file:
                for key in ('pad', 'unpad'):
                    self._write(location + '/' + key, output[key])
        if name == 'normal':
            for scale, field in output.items():
                groups = array(self.sample['observation']['grids'][scale]['group'])[self.selected]
                blocks, inverse = np.unique(groups, return_inverse=True)
                self._write(f'normal_selected/{scale}/block_row', blocks)
                self._write(f'normal_selected/{scale}/selected_point_to_block', inverse)
                for key, value in field.items():
                    self._write(f'normal_selected/{scale}/{key}', array(value)[blocks])
                self.file[f'normal_selected/{scale}'].attrs['scope'] = 'Only angular blocks containing selected/point_index; not all field blocks'
                self.file[f'normal_selected/{scale}'].attrs['axes'] = 'selected_block, hypothesis(4), kernel(8) where applicable; kernel/hypothesis IDs carry no fixed semantic names'
                # The field is a scientific boundary with no Tensor-valued container output.
                for key, value in field.items():
                    self._statistics(name, value, call, len(self.summary_outputs))
                    self.summary_outputs.append(f'{scale}/{key}')
        if name == 'compatibility':
            if not np.array_equal(array(inputs[3]), self.query_indices):
                raise ValueError('Compatibility indices differ from the recorded actual-point chunk')
            self.probability[self.query_indices[0]:self.query_indices[-1] + 1] = array(output[1])

        points = rows = None
        if name == 'detail' or name.startswith('detail.'):
            points, rows = self._selected_rows(self.detail_indices)
        elif name == 'normal.encoder' or name.startswith('normal.encoder.'):
            points, rows = self._selected_rows(self.normal_indices)
        elif name.startswith(('point_detail', 'point_position', 'compatibility', 'head')):
            points, rows = self._selected_rows(self.query_indices)
        elif name == 'adapter' or name.startswith('adapter.'):
            points, rows = self.selected, self.ancestors[0][self.selected]
        elif name == 'conditional' or name.startswith('conditional.'):
            level = int(name.split('.')[2]) if name.startswith('conditional.projections.') else 0
            points = self.selected
            rows = self.ancestors[min(level, 4) if level < 5 else 0][self.selected]
        elif kind in ('Block', 'GridPooling', 'GridUnpooling', 'Embedding'):
            match = re.search(r'\.(?:enc|dec)(\d+)\.', name)
            level = int(match.group(1)) if match else 0
            points, rows = self.selected, self.ancestors[level][self.selected]
        if points is not None and numerical:
            for number, tensor in enumerate(tensors):
                self._vectors(name, tensor, points, rows, call, number)

    def finish(self, scores):
        if self.file is None:
            raise RuntimeError('No active frame')
        if self.detail_cursor != self.n or self.normal_cursor != self.n or self.query_cursor != self.n:
            raise ValueError('Full-return module calls did not cover every actual point')
        if self.metric_cursor != int(self.valid.sum()) or scores.shape != (self.n,):
            raise ValueError('Full evaluation population was not preserved')
        if len(self.ancestors) != 5:
            raise ValueError('Missing backbone stage identity')
        self._write('points/scores', scores)
        self._write('statistics/channels', np.concatenate(self.summaries))
        self._write('statistics/calls', np.asarray(self.summary_calls, dtype=CALL_DTYPE))
        string = h5py.string_dtype('utf-8')
        self.file.create_dataset('statistics/shapes', data=np.asarray(self.summary_shapes, dtype=object), dtype=string)
        self.file.create_dataset('statistics/field_output_names', data=np.asarray(self.summary_outputs, dtype=object), dtype=string)
        self._write('selected/values', np.concatenate(self.vectors))
        self._write('selected/rows', np.concatenate(self.vector_rows))
        self.file.create_dataset('selected/calls', data=np.asarray([json.dumps(value, separators=(',', ':'))
            for value in self.vector_calls], dtype=object), dtype=string)
        self.file.create_dataset('coverage/modules', data=np.asarray(self.names, dtype=object), dtype=string)
        self.file.create_dataset('coverage/types', data=np.asarray([type(self.modules[name]).__name__
            for name in self.names], dtype=object), dtype=string)
        self._write('coverage/calls', np.asarray([self.calls[name] for name in self.names], dtype=np.int64))
        self.file['coverage'].attrs['statistics_scope'] = ('Every actual floating leaf output and scientific boundary; '
            'duplicate sequential/container outputs omitted. Every invocation retains module/name/output/shape and row count; '
            'statistics aggregate all invocations of the same module/output in this frame with FP64 parallel Welford variance. '
            'Channels are the final tensor axis, except 1D per-point scores are one channel; all preceding axes are reduced. '
            'Repeated call rows intentionally reference the same aggregate channel slice. Statistics are not class-conditional. '
            'Point and sparse-tensor outputs summarize feature tensors, not metadata. FP32 min/max preserve original extrema exactly.')
        self.file['coverage'].attrs['not_directly_recorded'] = json.dumps([
            'Functional SDPA attention matrices, internal softmax and every elementary operation',
            'NormalField functional Q/K/V projections; MultiheadAttention.forward is intentionally bypassed',
            'Sparse-convolution per-offset F.linear contributions and index_add intermediate accumulators',
            'Ray analytic integrals/masses and intermediate mu,tau,log_h; final actual per-hypothesis log densities are complete',
            'Compatibility and Conditional functional attention weights and residual additions except their module boundaries',
            'Full raw activations for every point at every layer; exact selected vectors and all-population statistics are saved'])
        self.file['coverage'].attrs['expected_uncalled'] = ('NormalField MultiheadAttention.forward; backbone attention.softmax; '
            'backbone.forward and backbone.enc.forward (Segmentor calls stages directly); ModuleList/ModuleDict containers')
        self.file.attrs['logical_array_bytes'] = self.logical_bytes
        self.file.attrs['complete'] = True
        self.file.flush()
        self.file.close()
        self.file = None
        self.active = False
        result = dict(path=str(self.path), bytes=self.path.stat().st_size,
            logical_bytes=self.logical_bytes, points=self.n, metric_points=int(self.valid.sum()),
            selected_points=len(self.selected), called_modules=len(self.calls),
            recorded_output_calls=len(self.summary_calls), recorded_selected_calls=len(self.vector_calls))
        self.summaries = self.vectors = self.vector_rows = []
        self.sample = None
        return result

    def close(self):
        self.active = False
        if self.file is not None:
            self.file.close()
            self.file = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
