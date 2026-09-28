"""Public interface rejection and device-context tests with synthetic tensors."""

import copy
import types
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from tools.load_kernel import CheckedAttention


class Tensor:
    def __init__(self, shape, device=1):
        self.shape, self.ndim = shape, len(shape)
        self.device = types.SimpleNamespace(type='cuda', index=device)
        self.dtype, self.is_cuda, self.requires_grad = 'bf16', True, False
        self.contiguous = True

    def is_contiguous(self): return self.contiguous


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.active_device = 0
        @contextmanager
        def device(index):
            previous, self.active_device = self.active_device, index
            try: yield
            finally: self.active_device = previous
        self.torch = types.SimpleNamespace(Tensor=Tensor, bfloat16='bf16', is_grad_enabled=lambda: True,
                                           cuda=types.SimpleNamespace(device=device))
        self.module = types.SimpleNamespace(__name__='fake', unified_prefill=Mock(), unified_decode=Mock())
        self.kernel = CheckedAttention(self.module, {'evaluation_spec':
            {'query_heads': 32, 'kv_heads': 8, 'head_dim': 128}}, 1)
        self.q = Tensor((1, 32, 128, 128))
        self.k = Tensor((1, 8, 128, 128))
        self.v = Tensor((1, 8, 128, 128))

    def test_device_guard_and_unchanged_arguments(self):
        def launch(*args):
            self.assertEqual(self.active_device, 1)
            self.assertEqual(args, (self.q, self.k, self.v, 32, 8))
            return 'output'
        self.module.unified_prefill.side_effect = launch
        with patch.dict('sys.modules', {'torch': self.torch}):
            self.assertEqual(self.kernel.unified_prefill(self.q, self.k, self.v, 32, 8), 'output')
        self.assertEqual(self.active_device, 0)

    def test_invalid_inputs_never_reach_cuda(self):
        cases = ('dtype_q', 'dtype_k', 'dtype_v', 'cpu', 'dimension', 'layout', 'device',
                 'other_device', 'heads', 'kv_shape', 'batch', 'empty', 'sequence', 'grad', 'not_tensor')
        for case in cases:
            with self.subTest(case=case), patch.dict('sys.modules', {'torch': self.torch}):
                q, k, v = copy.deepcopy((self.q, self.k, self.v))
                if case.startswith('dtype_'):
                    {'q':q, 'k':k, 'v':v}[case[-1]].dtype = 'fp32'
                elif case == 'cpu': k.is_cuda = False
                elif case == 'dimension': q.ndim = 3
                elif case == 'layout': v.contiguous = False
                elif case == 'device': v.device.index = 0
                elif case == 'other_device':
                    for t in (q, k, v): t.device.index = 0
                elif case == 'heads': q.shape = (1, 16, 128, 128)
                elif case == 'kv_shape': v.shape = (1, 8, 64, 128)
                elif case == 'batch': q.shape = (4, 32, 128, 128)
                elif case == 'empty': q.shape = (1, 32, 0, 128)
                elif case == 'sequence': q.shape = (1, 32, 64, 128)
                elif case == 'grad': q.requires_grad = True
                elif case == 'not_tensor': q = None
                with self.assertRaises((ValueError, TypeError)):
                    self.kernel.unified_prefill(q, k, v, 32, 8)
                self.module.unified_prefill.assert_not_called()

    def test_decode_and_explicit_head_arguments(self):
        with patch.dict('sys.modules', {'torch': self.torch}):
            with self.assertRaises(ValueError):
                self.kernel.unified_decode(self.q, self.k, self.v, 32, 8)
            self.q.shape = (1, 32, 1, 128)
            with self.assertRaises(ValueError):
                self.kernel.unified_decode(self.q, self.k, self.v, 16, 8)
            self.kernel.unified_decode(self.q, self.k, self.v, 32, 8)
            self.module.unified_decode.assert_called_once()
