"""Which workload archetypes does the KV-eviction literature evaluate on?

Input: taxonomy/sources/methods_classic.jsonl and methods_recent.jsonl (78 records of KV-cache
eviction / compression methods and evaluation studies, compiled from official repositories and
paper abstracts, with confidence tags). Each record's `evaluated_benchmarks` strings are mapped to
archetypes with the explicit keyword rules below (auditable per method in the output CSV).

Outputs: results/literature_coverage.csv (per method) and results/literature_coverage_summary.csv
(share of methods evaluating each archetype, by first-appearance year), figures/fig7_literature_coverage.png
"""
import json
import re

import numpy as np

from common import FIGURES, REPO, RESULTS, write_csv

SRC = REPO / "taxonomy" / "sources"
ARCH = ["compact", "short_gen", "long_reason", "long_gen", "sparse_retrieval", "multi_hop", "aggregation",
        "many_shot_icl", "lc_code_struct", "shared_context_multiturn", "agentic", "streaming_lm"]

# (regex, archetypes). Matched case-insensitively against each benchmark string of a method.
RULES = [
    (r"longbench[- ]?v2", ["sparse_retrieval", "multi_hop", "aggregation", "many_shot_icl", "lc_code_struct"]),
    (r"longbench(?![- ]?v2)(?!-write)", ["sparse_retrieval", "multi_hop", "aggregation"]),
    (r"longbench.*\b(16|15|all|trec|triviaqa|samsum|few-?shot|fsl)\b|\btrec\b|few-?shot demos|many[-_ ]?shot|\bicl\b",
     ["many_shot_icl"]),
    (r"longbench.*\b(16|15|all|lcc|repobench|code|cc)\b|\blcc\b|repobench|code completion|repoqa|code\.debug|codeu|crosscodeeval",
     ["lc_code_struct"]),
    (r"\bruler\b", ["sparse_retrieval", "multi_hop", "aggregation"]),
    (r"infinitebench|∞bench|infinite-?bench", ["sparse_retrieval", "multi_hop", "aggregation", "lc_code_struct"]),
    (r"needle|haystack|\bniah\b|passkey|pass-?key|retrieve\.|passageretrieval|nolima|needlebench",
     ["sparse_retrieval"]),
    (r"qasper|narrativeqa|squad|multifieldqa|l-?eval|loogle|zeroscrolls|quality|\bnq\b|natural questions|coqa|en\.qa",
     ["sparse_retrieval"]),
    (r"hotpot|2wiki|musique|multi-?hop|variable tracking|\bvt\b|babilong|reason-in-a-haystack|multi-doc|\bmqa\b",
     ["multi_hop"]),
    (r"summar(?!.*(xsum|cnn))|govreport|qmsum|multinews|multi-news|squality|\bcwe\b|\bfwe\b|math\.find|\bmf\b",
     ["aggregation"]),
    (r"xsum|cnn|sharegpt", ["short_gen"]),
    (r"piqa|copa|winogrande|openbookqa|hellaswag|\brte\b|boolq|mathqa|\bmmlu\b|\barc\b|commonsenseqa|truthfulqa|csqa|"
     r"lm-eval", ["compact"]),
    (r"gsm8k|gsm-?8k|humaneval|mbpp|\bbbh\b|alpacaeval|ifeval|\bdrop\b|folio|strategyqa|reclor|logiqa|minerva|gaokao|"
     r"mmlu-?pro|instruction following|math-?500|\bmath\b", ["short_gen"]),
    (r"aime|livecodebench|gpqa|\bamc\b|hle\b|longreason", ["long_reason"]),
    (r"longgenbench|longwriter|long-?form|story generation|long story|longproc|hellobench|long-context generation|"
     r"long generation", ["long_gen"]),
    (r"scbench|locomo|longmemeval|multi-?turn|multi-if|prefeval|dailydialog|grocery|rock-paper|realtalk|mt-bench|"
     r"conversation|dialogue", ["shared_context_multiturn"]),
    (r"swe-?bench|bfcl|openhands|browsecomp|agent|tau-?bench|τ-bench|webarena|terminal-bench", ["agentic"]),
    (r"perplexity|\bpg-?19\b|wikitext|\bc4\b|\bptb\b|language model|streaming|streameval|arc-?stream|openwebtext",
     ["streaming_lm"]),
]
# Records whose benchmark field is too vague for keyword matching; lists taken from the evaluation-study notes.
OVERRIDE = {"KVFundaBench / ShotKV": ["MMLU", "CommonsenseQA", "GSM8K", "HumanEval", "JailBreakV",
                                      "LongGenBench-GSM8K (long-context generation)"]}
REASONING_MODELS = re.compile(r"r1|qwq|distill|qwen3|thinking|reasoning|nemotron|gpt-oss|acereason", re.I)
SKIP = re.compile(r"demo only|not a scored|download script|unverified\)?$|observation analys", re.I)
STUDY = re.compile(r"analysis|KVFundaBench|Rethinking|Pitfalls|Hold Onto", re.I)


def load():
    seen, out = {}, []
    for f in ["methods_recent.jsonl", "methods_classic.jsonl"]:  # recent first: it has the fuller records
        for r in map(json.loads, open(SRC / f)):
            key = re.sub(r"[^a-z0-9]", "", r["name"].lower().split("(")[0])
            if key in seen:
                continue
            seen[key] = True
            out.append(r)
    return out


def map_archetypes(benchmarks, models=()):
    hits = set()
    for b in benchmarks:
        if SKIP.search(b):
            continue
        if re.search(r"longgenbench", b, re.I):  # K questions answered in one long response
            hits.add("long_gen")
            continue
        if re.search(r"stream", b, re.I):  # streaming perplexity / streaming QA over a concatenated stream
            hits.add("streaming_lm")
            continue
        for rx, archs in RULES:
            if re.search(rx, b, re.I):
                hits |= set(archs)
    # MATH-500 / GSM8K decoded by thinking models count as long-horizon reasoning
    if re.search(r"math-?500|gsm8k", " ".join(benchmarks), re.I) and REASONING_MODELS.search(" ".join(models)):
        hits.add("long_reason")
    return hits


if __name__ == "__main__":
    rows = []
    for r in load():
        if r["name"].startswith("Continuum"):  # serving-level (whole-request KV pinning), not token-level eviction
            continue
        b = OVERRIDE.get(r["name"]) or [x for x in r.get("evaluated_benchmarks", []) if x and x.lower() != "unknown"]
        if not b:
            continue
        a = map_archetypes(b, r.get("models", []) or [])
        rows.append(dict(name=r["name"], year=int(r["year"]) if str(r.get("year", "")).isdigit() else None,
                         venue=r.get("venue"), kind="study" if STUDY.search(r["name"]) else "method",
                         phase=r.get("phase"), archetypes=";".join(a2 for a2 in ARCH if a2 in a),
                         n_archetypes=len(a), confidence=str(r.get("confidence"))[:40],
                         benchmarks=" | ".join(b)[:400]))
    write_csv(rows, RESULTS / "literature_coverage.csv")
    cohorts = [("2023-2024", lambda y: y is not None and y <= 2024), ("2025-2026", lambda y: y is not None and y >= 2025)]
    summ = []
    for name, sel in cohorts:
        rs = [r for r in rows if r["kind"] == "method" and sel(r["year"])]
        row = {"cohort": name, "methods": len(rs)}
        for a in ARCH:
            row[a] = sum(a in r["archetypes"].split(";") for r in rs) / max(1, len(rs))
        row["median_archetypes_per_method"] = float(np.median([r["n_archetypes"] for r in rs])) if rs else None
        summ.append(row)
        print(name, len(rs), {a: round(row[a], 2) for a in ARCH}, "median #arch", row["median_archetypes_per_method"])
    write_csv(summ, RESULTS / "literature_coverage_summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import SLOTS, SURFACE, INK2, save  # shared style
    labels = {"compact": "compact", "short_gen": "short-gen", "long_reason": "long-reason", "long_gen": "long-gen",
              "sparse_retrieval": "sparse retrieval", "multi_hop": "multi-hop", "aggregation": "aggregation",
              "many_shot_icl": "many-shot ICL", "lc_code_struct": "long-ctx code", "shared_context_multiturn":
              "multi-turn / shared ctx", "agentic": "agentic", "streaming_lm": "streaming LM"}
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(ARCH))
    w = 0.38
    for k, row in enumerate(summ):
        vals = [100 * row[a] for a in ARCH]
        ax.bar(x + (k - 0.5) * w, vals, width=w - 0.04, color=SLOTS[k], label=f"{row['cohort']} ({row['methods']} methods)",
               zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in ARCH], rotation=30, ha="right")
    ax.set_ylabel("% of eviction/compression methods\nevaluating the archetype")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("What KV-eviction papers test on (benchmarks mapped to workload archetypes)", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig7_literature_coverage.png")
