import unittest
from src.prefix_lab import align_hit, build_pair


class FakeTokenizer:
    vocab_size = 512
    all_special_ids = [0, 1]
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 250 + 2 for c in text]


class TestPair(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_alignment(self):
        self.assertEqual(align_hit(8192, 0.5, 16), 4096)
        self.assertEqual(align_hit(2048, 0.333, 16), 672)
        self.assertRaises(ValueError, align_hit, 8192, -0.1, 16)

    def test_pair(self):
        for hit in (0, 256, 512, 1024):
            primer, query = build_pair(self.tok, 1024, hit, 12)
            self.assertEqual(len(query), 1024)
            if hit == 0:
                self.assertIsNone(primer)
            elif hit == 1024:
                self.assertEqual(primer, query)
            else:
                self.assertEqual(primer[:hit], query[:hit])
                self.assertNotEqual(primer[hit], query[hit])


if __name__ == "__main__":
    unittest.main()
