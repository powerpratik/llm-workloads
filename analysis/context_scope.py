"""M5 - Evidence scope, dispersion and multi-query overlap in long contexts; context redundancy.

Scope (after Goldman et al., 2024: "how much information is needed") is measured model-free:
the context is cut into C-token chunks and we greedily pick the fewest chunks that cover the
content words of the reference answer that occur in the context (a lexical oracle of the
smallest KV budget that still holds the answer's sources).
  k80            chunks needed to cover 80% of the matchable answer words
  scope80        k80 / #chunks (fraction of the context an oracle must keep)
  spread80       (last - first selected chunk + 1) / #chunks (dispersion of the needed chunks)
  union_scope    fraction of chunks needed by >= 1 of the queries asked about the same context
  pair_jaccard   mean Jaccard overlap of the chunk sets of two queries on the same context
LoCoMo ships gold evidence turns, so we also report exact evidence distance / depth / hops.
Redundancy: gzip ratio and repeated-4-gram rate of each context type.
Outputs: results/context_scope.csv, results/locomo_evidence.csv, results/context_redundancy.csv
"""
import glob
import json
import re
import zlib
from collections import defaultdict
from itertools import combinations

import numpy as np

from common import RAW, RESULTS, STOPWORDS, encode, ntok, read_jsonl, summarize, write_csv

C = 128  # chunk size in tokens (typical eviction budgets are 128-2048 tokens)
DS, REPOS = RAW / "datasets", RAW / "repos"
WORD = re.compile(r"[A-Za-z0-9]+")


def content_words(text):
    return {w.lower() for w in WORD.findall(text) if (len(w) >= 3 and w.lower() not in STOPWORDS) or w.isdigit()}


def chunk_words(context):
    enc = encode(context)
    offs = enc.offsets
    chunks = []
    for s in range(0, len(offs), C):
        a, b = offs[s][0], offs[min(s + C, len(offs)) - 1][1]
        chunks.append(content_words(context[a:b]))
    return chunks


def cover(chunks, target, frac):
    need = {w for w in target if any(w in c for c in chunks)}
    if len(need) < 2:
        return None
    goal, covered, chosen = frac * len(need), set(), []
    while len(covered) < goal:
        best = max(range(len(chunks)), key=lambda i: len((chunks[i] & need) - covered))
        gain = (chunks[best] & need) - covered
        if not gain:
            break
        covered |= gain
        chosen.append(best)
    return chosen


def scope_rows(task, archetype, docs):
    """docs: list of (context, [reference answers])."""
    k80s, sc80, sp80, unions, jacc, nq = [], [], [], [], [], []
    for ctx, refs in docs:
        chunks = chunk_words(ctx)
        sets = []
        for ref in refs:
            ch = cover(chunks, content_words(ref), 0.8)
            if ch is None:
                continue
            k80s.append(len(ch)); sc80.append(len(ch) / len(chunks))
            sp80.append((max(ch) - min(ch) + 1) / len(chunks))
            sets.append(set(ch))
        if len(sets) >= 2:
            unions.append(len(set().union(*sets)) / len(chunks))
            jacc.append(float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])))
        nq.append(len(sets))
    k, s, p = summarize(k80s), summarize(sc80), summarize(sp80)
    row = dict(task=task, archetype=archetype, contexts=len(docs), queries=len(k80s),
               queries_per_context=float(np.mean(nq)), k80_p50=k["p50"], k80_p90=k["p90"],
               scope80_p50=s["p50"], scope80_p90=s["p90"], spread80_p50=p["p50"], spread80_p90=p["p90"],
               union_scope_p50=float(np.median(unions)) if unions else None,
               pair_jaccard_mean=float(np.mean(jacc)) if jacc else None)
    print(f"{task:28s} q={len(k80s):5d} k80_p50={k['p50']} scope80_p50={s['p50'] or 0:.3f} "
          f"spread_p50={p['p50'] or 0:.2f} union={row['union_scope_p50'] or 0:.2f} jacc={row['pair_jaccard_mean'] or 0:.2f}")
    return row


LEVAL_ARCH = {"gov_report_summ": "aggregation", "meeting_summ": "aggregation", "news_summ": "aggregation",
              "patent_summ": "aggregation", "review_summ": "aggregation", "tv_show_summ": "aggregation",
              "paper_assistant": "aggregation", "codeU": "lc_code_struct", "gsm100": "many_shot_icl"}
MC = {"quality", "coursera", "tpo", "sci_fi", "codeU", "gsm100", "topic_retrieval_longchat"}


def leval_docs(task):
    f = glob.glob(str(REPOS / f"leval/LEval-data/*/{task}.jsonl"))[0]
    return [(d["input"], [str(o) for o in d["outputs"]]) for d in read_jsonl(f)]


def quality_docs():
    out = []
    for a in read_jsonl(DS / "quality/QuALITY.v1.0.1.htmlstripped.dev"):
        refs = [q["options"][q["gold_label"] - 1] for q in a["questions"] if "gold_label" in q]
        out.append((a["article"], refs))
    return out


def locomo_evidence():
    data = json.load(open(DS / "locomo/locomo10.json"))
    names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    rows, per_cat = [], defaultdict(list)
    for d in data:
        c = d["conversation"]
        sess = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        pos, cur = {}, 0
        for s in sess:
            cur += ntok(f"[{c[s + '_date_time']}]")[0]
            lens = ntok([f"{t['speaker']}: {t['text']}" for t in c[s]])
            for t, n in zip(c[s], lens):
                pos[t["dia_id"]] = (cur, cur + n)
                cur += n + 1
        total, turns = cur, len(pos)
        ev_sets = []
        for q in d["qa"]:
            ev = [e.strip() for e in q.get("evidence", []) if e.strip() in pos]
            if not ev:
                continue
            ev_sets.append(set(ev))
            starts = [pos[e][0] for e in ev]
            per_cat[q["category"]].append(dict(hops=len(ev), dist_far=total - min(starts), dist_near=total - max(starts),
                                               depth=min(starts) / total, span=(max(starts) - min(starts)) / total))
        allq = set().union(*ev_sets)
        jac = [len(a & b) / len(a | b) for a, b in combinations(ev_sets, 2)]
        rows.append(dict(conversation=d["sample_id"], tokens=total, turns=turns, questions=len(ev_sets),
                         evidence_turn_coverage=len(allq) / turns, pair_jaccard_mean=float(np.mean(jac))))
    cat_rows = []
    for k in sorted(per_cat):
        v = per_cat[k]
        cat_rows.append(dict(category=k, name=names.get(k, str(k)), questions=len(v),
                             hops_mean=float(np.mean([x["hops"] for x in v])),
                             multi_evidence_share=float(np.mean([x["hops"] > 1 for x in v])),
                             dist_far_p50=float(np.median([x["dist_far"] for x in v])),
                             dist_far_p90=float(np.percentile([x["dist_far"] for x in v], 90)),
                             dist_near_p50=float(np.median([x["dist_near"] for x in v])),
                             depth_p50=float(np.median([x["depth"] for x in v])),
                             span_p50=float(np.median([x["span"] for x in v]))))
    return rows, cat_rows


def redundancy_rows():
    items = []
    essays = "\n".join(open(f).read() for f in sorted(glob.glob(str(REPOS / "niah/needlehaystack/PaulGrahamEssays/*.txt"))))
    items.append(("NIAH haystack (Paul Graham essays)", "sparse_retrieval", [essays[i:i + 60000] for i in range(0, 600000, 60000)]))
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    items.append(("passkey filler (repeated sentences)", "sparse_retrieval", [filler * 700]))
    rng = np.random.default_rng(0)
    import uuid
    kv = [json.dumps({str(uuid.UUID(int=int(rng.integers(2 ** 63)) << 64 | int(rng.integers(2 ** 63)))):
                      str(uuid.UUID(int=int(rng.integers(2 ** 63)) << 64 | int(rng.integers(2 ** 63)))) for _ in range(700)})]
    items.append(("JSON key-value store (random UUIDs)", "sparse_retrieval", kv))
    for task in ["narrative_qa", "legal_contract_qa", "financial_qa", "meeting_summ", "gov_report_summ", "multidoc_qa",
                 "codeU", "gsm100", "topic_retrieval_longchat"]:
        docs = [c for c, _ in leval_docs(task)][:30]
        items.append((f"L-Eval {task}", LEVAL_ARCH.get(task, "sparse_retrieval"), docs))
    items.append(("QuALITY articles", "sparse_retrieval", [c for c, _ in quality_docs()][:30]))
    cc = read_jsonl(DS / "cceval/python/line_completion_rg1_bm25.jsonl")[:300]
    items.append(("CrossCodeEval python (retrieved ctx + file)", "lc_code_struct",
                  ["\n".join(x["retrieved_chunk"] for x in d["crossfile_context"]["list"]) + "\n" + d["prompt"] for d in cc]))
    lc = json.load(open(DS / "locomo/locomo10.json"))
    items.append(("LoCoMo conversations", "shared_context_multiturn",
                  ["\n".join(f"{t['speaker']}: {t['text']}" for k, v in d["conversation"].items()
                             if re.fullmatch(r"session_\d+", k) for t in v) for d in lc]))
    obs = []
    for f in sorted(glob.glob(str(RAW / "swebench_trajs/20250726_mini-v1.0.0_claude-sonnet-4-20250514/*.traj.json")))[:60]:
        m = json.load(open(f))["messages"]
        last_asst = max(i for i, x in enumerate(m) if x["role"] == "assistant")  # later messages never reach the model
        obs.append("\n".join((x["content"] if isinstance(x["content"], str) else
                              "\n".join(c.get("text", "") for c in x["content"])) for x in m[2:last_asst]
                             if x["role"] == "user"))
    items.append(("SWE-bench agent observations", "agentic", obs))
    tau = json.load(open(REPOS / "taubench/historical_trajectories/gpt-4o-retail.json"))[:60]
    items.append(("tau-bench retail trajectories", "agentic",
                  ["\n".join(str(m.get("content") or m.get("tool_calls")) for m in t["traj"]) for t in tau]))
    rows = []
    for name, arch, texts in items:
        texts = [t for t in texts if t and len(t) > 2000]
        gz = [len(t.encode()) / len(zlib.compress(t.encode(), 9)) for t in texts]
        rep = []
        for t in texts[:40]:
            ids = encode(t[:400000]).ids
            grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
            rep.append(1 - len(set(grams)) / max(1, len(grams)))
        rows.append(dict(context=name, archetype=arch, samples=len(texts), gzip_ratio_p50=float(np.median(gz)),
                         repeated_4gram_rate_p50=float(np.median(rep))))
        print(f"{name:45s} gzip={np.median(gz):5.2f} rep4={np.median(rep):.3f}")
    return rows


if __name__ == "__main__":
    rows = []
    for task in ["narrative_qa", "natural_question", "legal_contract_qa", "financial_qa", "scientific_qa", "multidoc_qa",
                 "paper_assistant", "meeting_summ", "review_summ", "gov_report_summ", "news_summ", "patent_summ",
                 "tv_show_summ"]:
        rows.append(scope_rows(f"L-Eval {task}", LEVAL_ARCH.get(task, "sparse_retrieval"), leval_docs(task)))
    rows.append(scope_rows("QuALITY dev (gold option)", "sparse_retrieval", quality_docs()))
    write_csv(rows, RESULTS / "context_scope.csv")
    conv, cats = locomo_evidence()
    write_csv(conv, RESULTS / "locomo_conversations.csv")
    write_csv(cats, RESULTS / "locomo_evidence.csv")
    for r in cats:
        print("LoCoMo", r)
    write_csv(redundancy_rows(), RESULTS / "context_redundancy.csv")
