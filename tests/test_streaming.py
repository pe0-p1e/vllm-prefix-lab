"""Regression tests for max_tokens=1 streaming completions (no GPU needed)."""
import json
import unittest
from src.prefix_lab import one_token


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


class FakeOpener:
    def __init__(self, lines):
        self.lines = lines

    def open(self, request, timeout=None):
        assert json.loads(request.data)["max_tokens"] == 1
        return FakeResponse(self.lines)


class TestStreaming(unittest.TestCase):
    def test_terminal_chunk_holds_only_token(self):
        lines = [b'data: {"choices":[{"text":"x","finish_reason":"length"}]}\n',
                 b'data: [DONE]\n']
        self.assertGreaterEqual(one_token(FakeOpener(lines),
                                           "http://localhost", "prefix-lab",
                                           [4, 5, 6]), 0)

    def test_empty_text_terminal_chunk(self):
        lines = [b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n',
                 b'data: [DONE]\n']
        self.assertGreaterEqual(one_token(FakeOpener(lines),
                                           "http://localhost", "prefix-lab",
                                           [4, 5, 6]), 0)

    def test_nonterminal_token(self):
        lines = [b'data: {"choices":[{"text":"a","finish_reason":null}]}\n',
                 b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n']
        self.assertGreaterEqual(one_token(FakeOpener(lines),
                                           "http://localhost", "prefix-lab",
                                           [4, 5, 6]), 0)

    def test_no_completion_is_error(self):
        lines = [b'data: {"choices":[]}\n', b'data: [DONE]\n']
        with self.assertRaisesRegex(RuntimeError, "No completion choice"):
            one_token(FakeOpener(lines), "http://localhost",
                      "prefix-lab", [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
