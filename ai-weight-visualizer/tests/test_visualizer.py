import base64
import unittest
import numpy as np
from visualize_weights import build_data


def fixture(layers=5, width=4, zero=False):
    rng = np.random.default_rng(17)
    weights = {}
    for layer in range(layers):
        for name, shape in [('attention.qkv.weight', (3 * width, width)),
                            ('attention.output.weight', (width, width)),
                            ('mlp.0.weight', (4 * width, width)),
                            ('mlp.2.weight', (width, 4 * width))]:
            weights[f'blocks.{layer}.{name}'] = (np.zeros(shape) if zero else rng.normal(size=shape)).astype(np.float32)
    return {'layers': layers, 'width': width, 'heads': 2}, weights


class VisualizerTests(unittest.TestCase):
    def test_sixteen_sectors_and_extrema(self):
        config, weights = fixture(16)
        data = build_data(config, weights, 'test.pt', 30)
        self.assertEqual(len(data['layers']), 16)
        self.assertEqual(data['parameters'], sum(a.size for a in weights.values()))
        for layer in data['layers']:
            all_values = []
            for part in layer['parts']:
                values = np.frombuffer(base64.b64decode(part['values']), dtype='<f4')
                idx = np.frombuffer(base64.b64decode(part['indices']), dtype='<u4')
                matrix = weights[part['key']][part['rowOffset']:part['rowOffset'] + part['shape'][0]]
                np.testing.assert_array_equal(values, matrix.ravel()[idx])
                self.assertEqual(values.min(), matrix.min())
                self.assertEqual(values.max(), matrix.max())
                all_values.extend(values)
            self.assertEqual(min(all_values), layer['min'])
            self.assertEqual(max(all_values), layer['max'])

    def test_shared_coordinates_and_determinism(self):
        config, weights = fixture()
        a = build_data(config, weights, 'test.pt', 30)
        self.assertEqual(a, build_data(config, weights, 'test.pt', 30))
        for part in range(6):
            self.assertEqual(len({layer['parts'][part]['indices'] for layer in a['layers']}), 1)

    def test_zero_weights(self):
        config, weights = fixture(zero=True)
        data = build_data(config, weights, 'zero.pt')
        self.assertEqual(data['scale'], 0)
        self.assertTrue(all(l['std'] == 0 for l in data['layers']))

    def test_reject_wrong_shape_or_nonfinite(self):
        config, weights = fixture()
        key = 'blocks.0.attention.output.weight'
        weights[key][0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_data(config, weights, 'bad.pt')
        weights[key] = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            build_data(config, weights, 'bad.pt')


if __name__ == '__main__':
    unittest.main()
