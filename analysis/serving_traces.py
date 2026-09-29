"""M4 - Cross-request KV reuse in production serving traces (prefix caching / cache eviction).

Traces (all public, content-free):
  Qwen-Bailian (Alibaba, 2025/26): to-C chat, to-B API, thinking, coder  - 16-token block hashes, sessions
  Mooncake (Kimi, FAST'25):         conversation, tool&agent, synthetic  - 512-token block hashes
  Azure LLM inference 2023:         code, conversation                   - lengths only
  BurstGPT (Azure OpenAI):          ChatGPT/GPT-4 x conversation/API     - lengths only

Prefix-cache semantics: a request can reuse the longest *leading* run of its blocks that is
resident (block hashes are prefix-chained; verified: runs never resume after a miss in the
Mooncake/thinking/coder traces). Within a request, ancestors are touched after descendants,
so LRU never evicts a prefix before its extensions (as in vLLM/SGLang).
Outputs: results/serving_traces.csv, results/serving_mrc.csv, results/serving_reuse_time.csv
"""
import csv
import heapq
import json
import sys
from collections import OrderedDict, defaultdict

import numpy as np

from common import RAW, RESULTS, summarize, write_csv

T = RAW / "traces"
TRACES = [
    ("qwen:to-C chat", T / "qwen_bailian/qwen_traceA_blksz_16.jsonl", 16),
    ("qwen:to-B API", T / "qwen_bailian/qwen_traceB_blksz_16.jsonl", 16),
    ("qwen:thinking", T / "qwen_bailian/qwen_thinking_blksz_16.jsonl", 16),
    ("qwen:coder", T / "qwen_bailian/qwen_coder_blksz_16.jsonl", 16),
    ("mooncake:conversation", T / "mooncake/conversation_trace.jsonl", 512),
    ("mooncake:tool&agent", T / "mooncake/toolagent_trace.jsonl", 512),
    ("mooncake:synthetic", T / "mooncake/synthetic_trace.jsonl", 512),
]
CAPS_TOKENS = [2 ** k for k in range(12, 27)]  # 4K .. 64M tokens of KV


def load(path):
    reqs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            h = r["hash_ids"]
            reqs.append(dict(t=float(r["timestamp"]) / (1000.0 if "chat_id" not in r else 1.0),
                             inp=int(r["input_length"]), out=int(r["output_length"]),
                             h=json.loads(h) if isinstance(h, str) else h,
                             chat=int(r.get("chat_id", -1)), parent=int(r.get("parent_chat_id", -1)),
                             turn=int(r.get("turn", 1)), type=r.get("type", "-")))
    reqs.sort(key=lambda r: r["t"])
    return reqs


def sessions(reqs):
    root = {}
    by_id = {r["chat"]: r for r in reqs}
    for r in reqs:
        c, seen = r["chat"], 0
        while by_id.get(c, {}).get("parent", -1) not in (-1, None) and seen < 10000:
            c = by_id[c]["parent"]; seen += 1
        root[r["chat"]] = c
    return root


class Fenwick:
    def __init__(self, n):
        self.n, self.t = n, np.zeros(n + 1, dtype=np.int64)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v; i += i & -i

    def sum(self, i):  # prefix sum [0, i]
        i += 1; s = 0
        while i > 0:
            s += self.t[i]; i -= i & -i
        return s


def infinite_and_lru(name, reqs, bs, root):
    """One pass: infinite-cache leading-run hits (same- vs cross-session), reuse times, and LRU stack
    distances of leading-run candidates (Mattson) -> full LRU miss-ratio curve."""
    total_acc = sum(len(r["h"]) for r in reqs)
    fw = Fenwick(total_acc + 1)
    last_time, last_sess, last_ts = {}, {}, {}
    clock = 0
    hit_tok = same_tok = cross_tok = in_tok = 0
    nonprefix = 0
    reuse_same, reuse_cross = [], []
    # per request: the list of stack distances of its leading run (distances are non-decreasing)
    dist_hist = defaultdict(int)  # capacity-independent: count tokens by 'required capacity' in blocks
    for r in reqs:
        h, sess = r["h"], root.get(r["chat"], r["chat"])
        in_tok += r["inp"]
        run, miss = 0, False
        prev_d = 0
        for i, b in enumerate(h):
            lt = last_time.get(b)
            if lt is None:
                miss = True
                continue
            if miss:
                nonprefix += 1
                continue
            run += 1
            d = fw.sum(clock - 1) - fw.sum(lt)  # distinct blocks referenced since b's last reference
            d = max(d, prev_d)                   # leading run hits only if all ancestors hit
            prev_d = d
            tok = min(bs, r["inp"] - i * bs) if i == len(h) - 1 else bs
            dist_hist[d + 1] += tok              # needs capacity >= d+1 blocks
            if last_sess.get(b) == sess:
                same_tok += tok; reuse_same.append(r["t"] - last_ts[b])
            else:
                cross_tok += tok; reuse_cross.append(r["t"] - last_ts[b])
            hit_tok += tok
        # touch: descendants first, ancestors last (ancestors most recent)
        for b in reversed(h):
            lt = last_time.get(b)
            if lt is not None:
                fw.add(lt, -1)
            fw.add(clock, 1)
            last_time[b] = clock; clock += 1
            last_sess[b] = sess; last_ts[b] = r["t"]
    caps_blocks = sorted(dist_hist)
    cum = np.cumsum([dist_hist[c] for c in caps_blocks])
    mrc = []
    for cap_tok in CAPS_TOKENS:
        cap_b = cap_tok // bs
        k = np.searchsorted(caps_blocks, cap_b, side="right")
        mrc.append(dict(trace=name, policy="LRU", capacity_tokens=cap_tok,
                        hit_ratio=float(cum[k - 1] / in_tok) if k > 0 else 0.0))
    has_sess = bool(root)
    stats = dict(ideal_hit=hit_tok / in_tok,
                 same_session_share=same_tok / max(1, hit_tok) if has_sess else None,
                 cross_session_share=cross_tok / max(1, hit_tok) if has_sess else None, nonprefix_repeat_blocks=nonprefix / total_acc,
                 unique_blocks=len(last_time), footprint_tokens=len(last_time) * bs)
    return stats, mrc, reuse_same, reuse_cross


def simulate(reqs, bs, cap_tok, policy):
    """Explicit simulation for FIFO / LFU / OPT(Belady) with prefix (leading-run) semantics."""
    cap = max(1, cap_tok // bs)
    hit_tok = in_tok = 0
    if policy == "OPT":
        seq = [b for r in reqs for b in r["h"]]
        nxt, pos_next = [0] * len(seq), {}
        for i in range(len(seq) - 1, -1, -1):
            nxt[i] = pos_next.get(seq[i], 10 ** 12); pos_next[seq[i]] = i
    cache = OrderedDict() if policy == "FIFO" else {}
    freq, heap, cur_key, pos = defaultdict(int), [], {}, 0
    tick = 0
    for r in reqs:
        h = r["h"]; in_tok += r["inp"]
        for i, b in enumerate(h):
            if b in cache:
                hit_tok += min(bs, r["inp"] - i * bs) if i == len(h) - 1 else bs
            else:
                break
        order = list(range(len(h) - 1, -1, -1))  # descendants first, ancestors last
        req_pos = {b: pos + i for i, b in enumerate(h)}
        for i in order:
            b = h[i]; tick += 1
            if policy == "FIFO":
                if b not in cache:
                    cache[b] = True
            elif policy == "LFU":
                freq[b] += 1
                key = (freq[b], tick); cur_key[b] = key; cache[b] = True
                heapq.heappush(heap, (key, b))
            elif policy == "OPT":
                key = -nxt[req_pos[b]]; cur_key[b] = key; cache[b] = True
                heapq.heappush(heap, (key, tick, b))
            while len(cache) > cap:
                if policy == "FIFO":
                    cache.popitem(last=False)
                else:
                    while True:
                        item = heapq.heappop(heap)
                        k, v = (item[0], item[1]) if policy == "LFU" else (item[0], item[2])
                        if v in cache and cur_key.get(v) == k:
                            del cache[v]; break
        pos += len(h)
    return hit_tok / in_tok


def length_rows():
    out = []
    for name, f in [("azure2023:code", T / "azure2023/AzureLLMInferenceTrace_code.csv"),
                    ("azure2023:conversation", T / "azure2023/AzureLLMInferenceTrace_conv.csv")]:
        rows = list(csv.DictReader(open(f)))
        i = [int(r["ContextTokens"]) for r in rows]; o = [int(r["GeneratedTokens"]) for r in rows]
        out.append(trace_len_row(name, i, o))
    groups = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(T / "burstgpt/BurstGPT_1.csv")):
        if int(r["Response tokens"]) == 0:
            continue
        g = groups[f"burstgpt:{r['Model']}:{r['Log Type'].replace(' log', '')}"]
        g[0].append(int(r["Request tokens"])); g[1].append(int(r["Response tokens"]))
    for k, (i, o) in sorted(groups.items()):
        out.append(trace_len_row(k, i, o))
    return out


def trace_len_row(name, i, o, extra=None):
    si, so = summarize(i), summarize(o)
    shares = np.asarray(o) / np.maximum(1, np.asarray(i) + np.asarray(o))
    row = dict(trace=name, requests=len(i), in_p10=si["p10"], in_p50=si["p50"], in_p90=si["p90"], in_p99=si["p99"],
               in_mean=si["mean"], out_p10=so["p10"], out_p50=so["p50"], out_p90=so["p90"], out_p99=so["p99"],
               out_mean=so["mean"], decode_share_p50=float(np.median(shares)),
               decode_token_share=float(np.sum(o) / max(1, np.sum(i) + np.sum(o))))
    row.update(extra or {})
    return row


def merge():
    """Merge per-trace partial outputs (from parallel runs) into *_all.csv files and delete the partials."""
    for kind, key in [("serving_traces", ("trace",)), ("serving_mrc", ("trace", "policy", "capacity_tokens")),
                      ("serving_reuse_time", ("trace", "reuse"))]:
        seen, out = set(), []
        parts = sorted(p for p in RESULTS.glob(f"{kind}_*.csv") if not p.name.endswith("_all.csv"))
        for p in parts:
            for r in csv.DictReader(open(p)):
                k = tuple(r[x] for x in key)
                if k not in seen:
                    seen.add(k); out.append(r)
        if out:
            write_csv(out, RESULTS / f"{kind}_all.csv")
            for p in parts:
                p.unlink()
        print(kind, len(out), "rows")


if __name__ == "__main__":
    if sys.argv[1:] == ["--merge"]:
        merge(); sys.exit()
    only = sys.argv[1:]  # optional subset of trace names (run several in parallel, then --merge)
    rows, mrcs, rt_rows = length_rows(), [], []
    for name, path, bs in TRACES:
        if only and name not in only:
            continue
        reqs = load(path)
        root = sessions(reqs) if reqs[0]["chat"] >= 0 else {}
        dur = reqs[-1]["t"] - reqs[0]["t"]
        extra = dict(block_size=bs, duration_s=dur, req_per_s=len(reqs) / max(1e-9, dur),
                     multi_turn_req_share=float(np.mean([r["turn"] > 1 for r in reqs])) if root else None,
                     sessions=len(set(root.values())) if root else None,
                     turns_p90=float(np.percentile([r["turn"] for r in reqs], 90)) if root else None,
                     turns_max=int(max(r["turn"] for r in reqs)) if root else None,
                     type_mix=json.dumps({t: round(sum(r["type"] == t for r in reqs) / len(reqs), 3)
                                          for t in sorted({r["type"] for r in reqs})}))
        stats, mrc, rs, rc = infinite_and_lru(name, reqs, bs, root)
        extra.update(stats)
        rows.append(trace_len_row(name, [r["inp"] for r in reqs], [r["out"] for r in reqs], extra))
        mrcs += mrc
        for label, arr in ([("same-session", rs), ("cross-session", rc)] if root else [("any", rs + rc)]):
            if arr:
                s = summarize(arr)
                rt_rows.append(dict(trace=name, reuse=label, hits=len(arr), p10_s=s["p10"], p50_s=s["p50"],
                                    p90_s=s["p90"], p99_s=s["p99"]))
        # explicit policy comparison at capacities where LRU reaches ~50% and ~90% of the ideal hit ratio
        ideal = stats["ideal_hit"]
        pts = sorted({next((m["capacity_tokens"] for m in mrc if m["hit_ratio"] >= f * ideal), CAPS_TOKENS[-1])
                      for f in (0.5, 0.9)})
        for cap in pts:
            for pol in ["FIFO", "LFU", "OPT"]:
                mrcs.append(dict(trace=name, policy=pol, capacity_tokens=cap, hit_ratio=simulate(reqs, bs, cap, pol)))
        print(f"{name:24s} req={len(reqs):7d} ideal_hit={ideal:.3f} same_sess={stats['same_session_share']} "
              f"footprint={stats['footprint_tokens'] / 1e6:.1f}M tok  pts={pts}", flush=True)
        for m in [m for m in mrcs if m["trace"] == name and m["capacity_tokens"] in pts]:
            print("    ", m["policy"], m["capacity_tokens"], round(m["hit_ratio"], 3), flush=True)
    suffix = "_" + "_".join(o.replace(":", "-").replace("&", "").replace(" ", "") for o in only) if only else "_all"
    write_csv(rows, RESULTS / f"serving_traces{suffix}.csv")
    write_csv(mrcs, RESULTS / f"serving_mrc{suffix}.csv")
    write_csv(rt_rows, RESULTS / f"serving_reuse_time{suffix}.csv")
