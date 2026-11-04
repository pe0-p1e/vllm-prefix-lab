"""Opt-in real CUDA injection checks; normal discovery does not import vLLM."""
import os
import unittest


@unittest.skipUnless(os.environ.get('CPU_KV_GPU_TEST') == '1', 'requires explicit GPU test')
class InjectionTests(unittest.TestCase):
    def test_live_blocks_and_persistent_host_buffers(self):
        import json
        import pathlib
        import tempfile
        from types import SimpleNamespace
        import torch
        from src.synthetic_cpu_connector import SyntheticCPUConnector, LoadMetadata, LoadRequest
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory() as tmp:
            c = object.__new__(SyntheticCPUConnector)
            c.hf_config = SimpleNamespace(num_key_value_heads=8, head_dim=128)
            c.expected_layers = 2
            c.block_size = 16
            c.forced_hit_tokens = 32
            c.pinned = True
            c._cpu_layers = {}
            c._counter = 0
            c.output_dir = pathlib.Path(tmp)
            caches = {str(i): torch.zeros((7, 8, 16, 256), device='cuda', dtype=torch.bfloat16)
                      for i in range(2)}
            c.register_kv_caches(caches)
            c.bind_connector_metadata(LoadMetadata([LoadRequest('test', [1, 5], 32)]))
            c.start_load_kv(None)
            pointers = {k: v.data_ptr() for k, v in c._cpu_layers.items()}
            for v in caches.values():
                self.assertTrue(torch.all(v[1] == .125).item())
                self.assertTrue(torch.all(v[5] == .125).item())
                self.assertTrue(torch.all(v[0] == 0).item())
                v.zero_()
            c.start_load_kv(None)
            self.assertEqual(pointers, {k: v.data_ptr() for k, v in c._cpu_layers.items()})
            for v in caches.values():
                self.assertTrue(torch.all(v[5] == .125).item())
            records = [json.loads(p.read_text()) for p in pathlib.Path(tmp).glob('kvload*')]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r['bytes'] == 32*2*2*8*128*2 for r in records))
            c.bind_connector_metadata(LoadMetadata([LoadRequest('bad', [1, 7], 32)]))
            with self.assertRaises(RuntimeError):
                c.start_load_kv(None)
