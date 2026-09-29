"""Rebuild Llama-3 and Qwen2 tokenizers offline from llama.cpp vocab-only GGUF files.

Hugging Face is not required: llama.cpp ships vocab-only GGUF files (tokens + BPE merges)
together with reference test vectors (`*.gguf.inp` / `*.gguf.out`). We rebuild a
`tokenizers` byte-level BPE with the model's pre-tokenizer regex and check it against
those test vectors. Both tokenizers reproduce the llama.cpp reference ids exactly.

Usage: python analysis/build_tokenizers.py   (after analysis/fetch_data.sh)
"""
import gguf
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, normalizers

from common import TOK_DIR

# Pre-tokenizer split patterns (from the models' tokenizer.json files).
PATS = {
    "llama-bpe": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
SPECS = [  # gguf file, pre-tokenizer, ignore_merges (Llama-3 uses ignore_merges=True)
    ("ggml-vocab-llama-bpe.gguf", "llama-bpe", True),
    ("ggml-vocab-qwen2.gguf", "qwen2", False),
]


def _strs(reader, key):
    f = reader.fields[key]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def _ints(reader, key):
    f = reader.fields[key]
    return [int(f.parts[i][0]) for i in f.data]


def build(path, pre, ignore_merges):
    r = gguf.GGUFReader(str(path))
    toks = _strs(r, "tokenizer.ggml.tokens")
    types = _ints(r, "tokenizer.ggml.token_type")
    merges = [tuple(m.split(" ", 1)) for m in _strs(r, "tokenizer.ggml.merges")]
    tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(toks)}, merges=merges,
                              ignore_merges=ignore_merges, byte_fallback=False))
    if pre == "qwen2":
        tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATS[pre]), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([t for t, ty in zip(toks, types) if ty == 3])  # CONTROL tokens
    return tk


def check(tk, gguf_path):
    inp = open(str(gguf_path) + ".inp", encoding="utf-8").read().split("\n__ggml_vocab_test__\n")
    out = open(str(gguf_path) + ".out", encoding="utf-8").read().strip("\n").split("\n")
    ok = sum(tk.encode(s, add_special_tokens=False).ids == [int(x) for x in o.split()]
             for s, o in zip(inp, out))
    return ok, len(out)


if __name__ == "__main__":
    for fname, pre, im in SPECS:
        tk = build(TOK_DIR / fname, pre, im)
        out = TOK_DIR / (fname.replace("ggml-vocab-", "").replace(".gguf", "") + ".tokenizer.json")
        tk.save(str(out))
        ok, n = check(tk, TOK_DIR / fname)
        print(f"{out.name}: vocab={tk.get_vocab_size()} llama.cpp test vectors {ok}/{n} exact")
        assert ok == n, "tokenizer does not reproduce llama.cpp reference ids"
