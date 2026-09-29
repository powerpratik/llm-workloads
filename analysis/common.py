"""Shared paths, tokenizer access and small statistics helpers for the analysis scripts."""
import csv
import gzip
import json
import os
import pathlib
import re
from functools import lru_cache

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("LLMW_DATA", REPO / "data"))
RAW = DATA / "raw"
TOK_DIR = DATA / "tok"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Llama-3 chat template adds <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|> per message.
CHAT_MSG_OVERHEAD = 5


@lru_cache(maxsize=None)
def tokenizer(name="llama3"):
    from tokenizers import Tokenizer
    fname = {"llama3": "llama-bpe.tokenizer.json", "qwen2": "qwen2.tokenizer.json"}[name]
    return Tokenizer.from_file(str(TOK_DIR / fname))


def ntok(texts, name="llama3"):
    """Token counts (no special tokens) for a list of strings."""
    if isinstance(texts, str):
        texts = [texts]
    tk = tokenizer(name)
    return [len(e.ids) for e in tk.encode_batch([t or "" for t in texts], add_special_tokens=False)]


def encode(text, name="llama3"):
    return tokenizer(name).encode(text or "", add_special_tokens=False)


def encode_ids(texts, name="llama3"):
    tk = tokenizer(name)
    return [e.ids for e in tk.encode_batch([t or "" for t in texts], add_special_tokens=False)]


def summarize(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return dict(n=0, mean=None, p10=None, p50=None, p90=None, p99=None, max=None)
    p10, p50, p90, p99 = np.percentile(a, [10, 50, 90, 99])
    return dict(n=int(a.size), mean=float(a.mean()), p10=float(p10), p50=float(p50),
                p90=float(p90), p99=float(p99), max=float(a.max()))


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows, path, fields=None):
    rows = list(rows)
    if not rows:
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# A compact English stop-list used to decide whether a copied n-gram carries information.
STOPWORDS = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just let me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours yourself yourselves
also may might must shall us one two first new get got use used using make made like well way yes ok okay
let's i'm it's that's there's here's don't doesn't can't won't isn't aren't wasn't weren't i'll we'll you'll
need needs want wait hmm so now think thinking maybe perhaps actually really right see look looks""".split())

_WORD = re.compile(r"[A-Za-z0-9_]+")


def informative(text):
    """True if a decoded n-gram contains a digit or a non-stopword alphanumeric word of length >= 3."""
    for w in _WORD.findall(text):
        if any(c.isdigit() for c in w):
            return True
        if len(w) >= 3 and w.lower() not in STOPWORDS:
            return True
    return False
