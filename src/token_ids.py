"""Exact-length token ID list generation. No tokenizer of prompt strings on wire."""
from __future__ import annotations
def make_ids(tokenizer, length: int, tag: int) -> list[int]:
    if length < 64:
        raise ValueError("length too short")
    prefix = tokenizer.encode(f"CPU-KV-BENCH-{tag:09d} ",
                              add_special_tokens=False)
    filler = tokenizer.encode(" Synthetic host KV performance test. ",
                              add_special_tokens=False)
    if not filler or len(prefix) >= length:
        raise ValueError("Tokenizer incompatible with length")
    return prefix + (filler * ((length - len(prefix) + len(filler)-1)
                               // len(filler)))[:length-len(prefix)]
