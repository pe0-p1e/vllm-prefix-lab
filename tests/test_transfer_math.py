import unittest
from scripts.bench_cpu_kv_transfer import kv_bytes, percentile, profile_model


class TestTransferMath(unittest.TestCase):
    def test_size(self):
        self.assertEqual(kv_bytes(16, 2, 4, 8, 2), 4096)

    def test_empty(self):
        self.assertEqual(kv_bytes(0, 28, 8, 128, 2), 0)

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], .95), 4.8)

    def test_config(self):
        class Config:
            num_hidden_layers = 28
            num_attention_heads = 16
            num_key_value_heads = 8
            hidden_size = 2048
            head_dim = 128
        self.assertEqual(profile_model(Config()), (28, 8, 128))


if __name__ == "__main__":
    unittest.main()
