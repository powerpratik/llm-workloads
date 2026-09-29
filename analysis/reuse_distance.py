"""M2 - Model-free dependency profiles: lexical reuse distance of decoded tokens.

Classic cache characterisation describes a reference stream by its reuse (stack) distance
distribution, from which the miss-ratio curve of LRU follows directly (Mattson et al., 1970).
We apply the same idea to generation: whenever the decoder emits an *informative* token
n-gram (n=4, containing a digit or a non-stopword) that already occurred earlier in the
sequence (prompt or earlier output), the model must have "read" that earlier occurrence
(or recomputed it). The distance to the nearest earlier occurrence is a LOWER BOUND on how
far back that dependency reaches. From the distribution we report:
  copy_rate          fraction of informative output n-grams that re-occur (verbatim dependency)
  src_prompt         fraction of copy events whose nearest source lies in the prompt
  MR(W)              fraction of copy events whose nearest source is > W tokens back
                     == miss ratio of a pure sliding window of W tokens (StreamingLLM minus sinks)
  src_initial        fraction of copy events whose nearest source lies in the INITIAL prompt region
                     (everything before the first decoded token: system prompt, task, document). Since
                     the nearest occurrence is the latest one, these items were never restated later:
                     they must survive in the cache from the very start (persistent state).
Outputs: results/reuse_distance.csv, results/reuse_distance_cdf.json
Limitations: lexical proxy only (paraphrase and computed values are invisible); nearest
occurrence under-estimates the reach of aggregation-type dependencies (all occurrences needed).
"""
import glob
import json
import re

import numpy as np

from common import RAW, RESULTS, informative, read_jsonl, tokenizer, write_csv, write_json

N = 4
WINDOWS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DS, REPOS = RAW / "datasets", RAW / "repos"

_info_cache = {}


def is_info(ids):
    key = tuple(ids)
    v = _info_cache.get(key)
    if v is None:
        v = informative(tokenizer().decode(list(ids)))
        if len(_info_cache) < 2_000_000:
            _info_cache[key] = v
    return v


def events(segments):
    """segments: list of (token_ids, is_output). Returns, for each informative output n-gram with an
    earlier occurrence, (distance, src_in_prompt, src_in_initial_region), plus the
    number of informative output n-grams."""
    seq, is_out = [], []
    for ids, o in segments:
        seq.extend(ids)
        is_out.extend([o] * len(ids))
    init = next((i for i, o in enumerate(is_out) if o), len(seq))
    last = {}
    ev, n_info = [], 0
    for t in range(N - 1, len(seq)):
        g = tuple(seq[t - N + 1:t + 1])
        if is_out[t] and is_info(g):
            n_info += 1
            s = last.get(g)
            if s is not None:
                ev.append((t - s, not is_out[s], s < init))
        last[g] = t
    return ev, n_info


def profile(name, archetype, samples, note=""):
    tk = tokenizer()
    dists, src_p, src_i, n_info, n_out = [], [], [], 0, 0
    for segs in samples:
        enc = [(tk.encode(txt or "", add_special_tokens=False).ids, o) for txt, o in segs]
        ev, ni = events(enc)
        n_info += ni
        n_out += sum(len(ids) for ids, o in enc if o)
        for d, p, i in ev:
            dists.append(d); src_p.append(p); src_i.append(i)
    d = np.asarray(dists)
    row = dict(workload=name, archetype=archetype, samples=len(samples), out_tokens=n_out, informative_ngrams=n_info,
               copy_events=len(d), copy_rate=len(d) / max(1, n_info),
               src_prompt=float(np.mean(src_p)) if len(d) else None,
               src_initial=float(np.mean(src_i)) if len(d) else None,
               dist_p50=float(np.median(d)) if len(d) else None,
               dist_p90=float(np.percentile(d, 90)) if len(d) else None, note=note)
    for w in WINDOWS:
        row[f"MR@{w}"] = float(np.mean(d > w)) if len(d) else None
    print(f"{name:45s} n={len(samples):5d} copy_rate={row['copy_rate']:.2f} src_prompt={row['src_prompt'] or 0:.2f} "
          f"p50={row['dist_p50'] or 0:7.0f} p90={row['dist_p90'] or 0:8.0f} MR@512={row['MR@512'] or 0:.2f} "
          f"MR@4096={row['MR@4096'] or 0:.2f} src_initial={row['src_initial'] or 0:.2f}")
    qs = np.unique(np.round(np.logspace(0, 5.3, 80)).astype(int))
    cdf = [float(np.mean(d <= q)) for q in qs] if len(d) else []
    return row, dict(archetype=archetype, x=qs.tolist(), cdf=cdf)


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return json.dumps(content)


# ------------------------------------------------------------------------------ workloads
def w_gsm8k():
    data = read_jsonl(DS / "gsm8k/test.jsonl")
    return [[(d["question"], False), (re.sub(r"<<.*?>>", "", d["answer"].split("####")[0]), True)] for d in data]


def w_math500():
    return [[(d["problem"], False), (d["solution"], True)] for d in read_jsonl(DS / "math500/test.jsonl")]


def w_humaneval():
    return [[(d["prompt"], False), (d["canonical_solution"], True)] for d in read_jsonl(DS / "humaneval/HumanEval.jsonl.gz")]


def w_alpaca(model="Meta-Llama-3-8B-Instruct"):
    data = json.load(open(REPOS / f"alpaca_eval/results/{model}/model_outputs.json"))
    return [[(d["instruction"], False), (d["output"], True)] for d in data]


def w_arena_thinking(model, category="hard_prompt"):
    q = {d["uid"]: d for d in read_jsonl(DS / "arenahard/v2.0_question.jsonl")}
    out = []
    for r in read_jsonl(REPOS / f"arenahard/data/arena-hard-v2.0/model_answer/{model}.jsonl"):
        if r["uid"] in q and q[r["uid"]]["category"] == category:
            c = r["messages"][-1]["content"]
            out.append([(q[r["uid"]]["prompt"], False), (c.get("thought", ""), True), (c.get("answer", ""), True)])
    return out


def w_cceval(lang="python"):
    data = read_jsonl(DS / f"cceval/{lang}/line_completion_rg1_bm25.jsonl")
    return [[("\n".join(c["retrieved_chunk"] for c in d["crossfile_context"]["list"]) + "\n" + d["prompt"], False),
             (d["groundtruth"], True)] for d in data]


def w_leval(task):
    f = glob.glob(str(REPOS / f"leval/LEval-data/*/{task}.jsonl"))[0]
    out = []
    for d in read_jsonl(f):
        for ins, o in zip(d["instructions"], d["outputs"]):
            out.append([(d["input"], False), (ins, False), (str(o), True)])
    return out


def w_locomo():
    data = json.load(open(DS / "locomo/locomo10.json"))
    out = []
    for d in data:
        c = d["conversation"]
        sess = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        text = "\n".join(f"[{c[s + '_date_time']}]\n" + "\n".join(f"{t['speaker']}: {t['text']}" for t in c[s]) for s in sess)
        for q in d["qa"]:
            if "answer" in q:
                out.append([(text, False), (q["question"], False), (str(q["answer"]), True)])
    return out


def w_mtbench101():
    out = []
    for d in read_jsonl(DS / "mtbench101/mtbench101.jsonl"):
        segs = []
        for i, t in enumerate(d["history"]):
            segs.append((t["user"], False))
            segs.append((t["bot"], i == len(d["history"]) - 1))  # only the final response is "decoded" here
        out.append(segs)
    return out


def w_swebench(sub):
    out = []
    for f in sorted(glob.glob(str(RAW / f"swebench_trajs/{sub}/*.traj.json"))):
        msgs = json.load(open(f))["messages"]
        out.append([(text_of(m.get("content")), m["role"] == "assistant") for m in msgs])
    return out


def w_taubench(name):
    out = []
    for t in json.load(open(REPOS / f"taubench/historical_trajectories/{name}.json")):
        out.append([(text_of(m.get("content")) or json.dumps(m.get("tool_calls")), m["role"] == "assistant")
                    for m in t["traj"]])
    return out


WORKLOADS = [
    ("gsm8k (ref CoT)", "short_gen", w_gsm8k, ""),
    ("math500 (ref solution)", "short_gen", w_math500, ""),
    ("humaneval (canonical)", "short_gen", w_humaneval, ""),
    ("alpacaeval: Llama-3-8B-Instruct", "short_gen", w_alpaca, ""),
    ("arenahard-v2 hard: deepseek-r1 (thinking)", "long_reason", lambda: w_arena_thinking("deepseek-r1"), ""),
    ("arenahard-v2 hard: qwq-32b (thinking)", "long_reason", lambda: w_arena_thinking("qwq-32b"), ""),
    ("arenahard-v2 creative: qwq-32b (thinking)", "long_gen", lambda: w_arena_thinking("qwq-32b", "creative_writing"), ""),
    ("crosscodeeval python (bm25 ctx)", "lc_code_struct", w_cceval, ""),
    ("leval: legal_contract_qa", "sparse_retrieval", lambda: w_leval("legal_contract_qa"), "reference answers"),
    ("leval: financial_qa", "sparse_retrieval", lambda: w_leval("financial_qa"), "reference answers"),
    ("leval: multidoc_qa", "sparse_retrieval", lambda: w_leval("multidoc_qa"), "reference answers"),
    ("leval: gov_report_summ", "aggregation", lambda: w_leval("gov_report_summ"), "reference summaries"),
    ("leval: meeting_summ", "aggregation", lambda: w_leval("meeting_summ"), "reference summaries"),
    ("leval: paper_assistant", "aggregation", lambda: w_leval("paper_assistant"), "reference outputs"),
    ("locomo qa", "shared_context_multiturn", w_locomo, "gold answers"),
    ("mtbench101 final turn", "shared_context_multiturn", w_mtbench101, "gold responses"),
    ("swebench: claude-sonnet-4 (mini-swe-agent)", "agentic",
     lambda: w_swebench("20250726_mini-v1.0.0_claude-sonnet-4-20250514"), "all assistant turns"),
    ("swebench: qwen3-coder-480b (mini-swe-agent)", "agentic",
     lambda: w_swebench("20250802_mini-v1.0.0_qwen3-coder-480b-a35b-instruct"), "all assistant turns"),
    ("swebench: gpt-5 (mini-swe-agent)", "agentic", lambda: w_swebench("20250807_mini-v1.7.0_gpt-5"),
     "visible output only"),
    ("tau-bench retail: sonnet-3.5", "agentic", lambda: w_taubench("sonnet-35-new-retail"), "all assistant turns"),
    ("tau-bench airline: gpt-4o", "agentic", lambda: w_taubench("gpt-4o-airline"), "all assistant turns"),
]

if __name__ == "__main__":
    rows, cdfs = [], {}
    for name, arch, fn, note in WORKLOADS:
        r, c = profile(name, arch, fn(), note)
        rows.append(r)
        cdfs[name] = c
    write_csv(rows, RESULTS / "reuse_distance.csv")
    write_json(cdfs, RESULTS / "reuse_distance_cdf.json")
