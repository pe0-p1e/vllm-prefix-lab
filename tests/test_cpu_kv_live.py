import unittest
from scripts.run_cpu_kv_matrix import metric_values
from scripts.summarize_cpu_kv_live import quantile
from src.token_ids import make_ids


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch)%251+2 for ch in text]


class AuditTests(unittest.TestCase):
    def test_metric_source_filter(self):
        raw='''# HELP x
vllm:request_prefill_time_seconds_sum{model_name="a"} 0.125
vllm:request_prefill_kv_computed_tokens_sum{model_name="a"} 4096
vllm:prompt_tokens_cached_by_source_total{source="host"} 2048
vllm:prompt_tokens_cached_by_source_total{source="device"} 999
'''
        m=metric_values(raw)
        self.assertEqual(m["vllm:prompt_tokens_cached_by_source_total"],2048)
        self.assertAlmostEqual(m["vllm:request_prefill_time_seconds_sum"],.125)
        self.assertEqual(m["vllm:request_prefill_kv_computed_tokens_sum"],4096)

    def test_token_exact_lengths_and_distinct_ids(self):
        tok=FakeTokenizer()
        for n in (2048,8192,32768):
            a=make_ids(tok,n,100)
            b=make_ids(tok,n,101)
            self.assertEqual(len(a),n)
            self.assertEqual(len(b),n)
            self.assertNotEqual(a,b)

    def test_quantile(self):
        self.assertAlmostEqual(quantile([1,2,3,4,5],.95),4.8)


if __name__=="__main__":
    unittest.main()
