#!/usr/bin/env python3
"""Build an offline sector sphere for JEV transformer checkpoints.

Requires NumPy. Uses PyTorch's weights_only loader when PyTorch is installed;
otherwise supports the float32 ZIP checkpoint format used by this project.
Does not import or execute the model architecture or run inference.
"""
import argparse
import base64
import collections
import io
import json
import math
import pickle
from pathlib import Path
import sys
import zipfile

import numpy as np


class FloatCheckpointReader(pickle.Unpickler):
    """Read tensor descriptors; reject every unrecognized pickle global."""
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('torch', 'FloatStorage'):
            return 'f4'
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda storage, offset, size, stride, *_: (storage, offset, size, stride)
        raise ValueError(f'Unsupported checkpoint object {module}.{name}. Use a Python environment with PyTorch installed.')

    def persistent_load(self, descriptor):
        if len(descriptor) != 5 or descriptor[0:2] != ('storage', 'f4'):
            raise ValueError('The NumPy-only loader supports float32 tensor storage only.')
        return descriptor[2], descriptor[4]


def numpy_load(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1:
            raise ValueError('Expected one data.pkl in a PyTorch ZIP checkpoint.')
        prefix = names[0][:-len('data.pkl')]
        state = FloatCheckpointReader(io.BytesIO(archive.read(names[0]))).load()
        byteorder = archive.read(prefix + 'byteorder').decode() if prefix + 'byteorder' in archive.namelist() else 'little'
        if byteorder not in ('little', 'big'):
            raise ValueError('Unknown checkpoint byte order.')
        dtype = np.dtype('<f4' if byteorder == 'little' else '>f4')
        weights = {}
        for key, descriptor in state['model'].items():
            (storage, count), offset, shape, strides = descriptor
            raw = archive.read(prefix + 'data/' + str(storage))
            if len(raw) != count * 4 or len(shape) != len(strides):
                raise ValueError(f'Invalid storage for {key}')
            if offset < 0 or any(s < 0 for s in shape + strides):
                raise ValueError(f'Unsupported shape or stride for {key}')
            end = offset + sum((s - 1) * step for s, step in zip(shape, strides)) + 1 if math.prod(shape) else offset
            if end > count:
                raise ValueError(f'Tensor {key} extends beyond its storage.')
            weights[key] = np.ndarray(shape, dtype=dtype, buffer=raw, offset=offset * 4,
                                      strides=tuple(s * 4 for s in strides)).astype(np.float32, copy=True)
        return state['config'], weights


def load_checkpoint(path):
    try:
        import torch
    except ImportError:
        return numpy_load(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    weights = {key: value.detach().float().cpu().numpy() for key, value in state['model'].items()}
    return state['config'], weights


def block_parts(weights, layer, width):
    prefix = f'blocks.{layer}.'
    specs = [('attention.qkv.weight', (3 * width, width)),
             ('attention.output.weight', (width, width)),
             ('mlp.0.weight', (4 * width, width)),
             ('mlp.2.weight', (width, 4 * width))]
    for name, shape in specs:
        a = weights[prefix + name]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'{prefix + name}: expected finite weights of shape {shape}, got {a.shape}')
    qkv = weights[prefix + 'attention.qkv.weight']
    return [
        ('Query', prefix + 'attention.qkv.weight', qkv[:width], 0),
        ('Key', prefix + 'attention.qkv.weight', qkv[width:2 * width], width),
        ('Value', prefix + 'attention.qkv.weight', qkv[2 * width:], 2 * width),
        ('Attention output', prefix + 'attention.output.weight', weights[prefix + 'attention.output.weight'], 0),
        ('Feed-forward expansion', prefix + 'mlp.0.weight', weights[prefix + 'mlp.0.weight'], 0),
        ('Feed-forward contraction', prefix + 'mlp.2.weight', weights[prefix + 'mlp.2.weight'], 0),
    ]


def b64(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


def build_data(config, weights, source, points_per_layer=6000, seed=2026):
    layers, width = int(config['layers']), int(config['width'])
    if layers < 1 or width < 1:
        raise ValueError('Layer count and width must be positive.')
    blocks = [block_parts(weights, i, width) for i in range(layers)]
    total = sum(a.size for _, _, a, _ in blocks[0])
    indices = []
    # One shared set of coordinates across blocks. Include every block's
    # per-matrix min/max, so sampling cannot hide its most extreme spikes.
    for j, (_, _, a, _) in enumerate(blocks[0]):
        if a.size > np.iinfo(np.uint32).max:
            raise ValueError('A matrix is too large for this viewer index format.')
        n = min(a.size, max(1, round(points_per_layer * a.size / total)))
        sampled = np.random.default_rng(seed + j).choice(a.size, n, replace=False)
        extrema = [index for block in blocks for index in (np.argmin(block[j][2]), np.argmax(block[j][2]))]
        indices.append(np.unique(np.concatenate((sampled, extrema))).astype('<u4'))
    result = {'config': {'layers': layers, 'width': width, 'heads': int(config.get('heads', 1))},
              'layers': [], 'source': source, 'parameters': sum(a.size for a in weights.values()),
              'scale': 0.0, 'sampleCount': 0, 'seed': seed}
    for block in blocks:
        parts, offset = [], 0
        arrays = []
        for j, (label, key, a, row_offset) in enumerate(block):
            flat = a.reshape(-1)
            idx = indices[j]
            parts.append({'label': label, 'key': key, 'shape': list(a.shape), 'rowOffset': row_offset,
                          'start': offset, 'total': a.size, 'indices': b64(idx, '<u4'),
                          'values': b64(flat[idx], '<f4')})
            offset += a.size
            result['sampleCount'] += len(idx)
            arrays.append(flat)
        full = np.concatenate(arrays).astype(np.float64)
        result['scale'] = max(result['scale'], float(np.max(np.abs(full))))
        result['layers'].append({'parts': parts, 'count': total, 'mean': float(full.mean()),
                                 'std': float(full.std()), 'min': float(full.min()), 'max': float(full.max())})
    return result


def write_viewer(data, output):
    template = Path(__file__).with_name('viewer-template.html').read_text(encoding='utf-8')
    if template.count('__WEIGHTS__') != 1:
        raise ValueError('Viewer template is missing its unique data placeholder.')
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__WEIGHTS__', payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='JEV .pt checkpoint containing config and model')
    parser.add_argument('-o', '--output', type=Path, default=Path('weight-sphere.html'))
    parser.add_argument('--points-per-layer', type=int, default=6000,
                        help='Approximate sample budget per block, plus extrema (default: 6000)')
    parser.add_argument('--seed', type=int, default=2026, help='Deterministic sampling seed')
    args = parser.parse_args()
    if args.points_per_layer < 6:
        parser.error('--points-per-layer must be at least 6')
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error('Output must not overwrite the checkpoint.')
    if args.output.suffix.lower() != '.html':
        parser.error('Output must have an .html extension.')
    try:
        config, weights = load_checkpoint(args.checkpoint)
        data = build_data(config, weights, args.checkpoint.name, args.points_per_layer, args.seed)
        write_viewer(data, args.output)
    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Could not generate viewer: {exc}\n')
    print(f"Wrote {args.output.resolve()}\n{config['layers']} sectors; {data['sampleCount']:,} displayed weights. Open the HTML in Firefox.")


if __name__ == '__main__':
    main()
