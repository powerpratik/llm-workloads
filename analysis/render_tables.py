"""Render the measurement CSVs in results/ as markdown tables (docs/tables.md).

The narrative in docs/characterization.md quotes these tables; regenerate after re-running the
analysis scripts so text and numbers stay in sync.
"""
import csv

from common import REPO, RESULTS


def rows(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def n(x, nd=0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if nd == 0:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def pct(x):
    try:
        return f"{100 * float(x):.0f}%"
    except (TypeError, ValueError):
        return "–"


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def t_lengths():
    body = []
    for r in rows("length_profiles.csv"):
        body.append([r["workload"], r["archetype"], n(r["n"]), n(r["in_p50"]), n(r["in_p90"]), n(r["out_p50"]),
                     n(r["out_p90"]), pct(r["decode_share_p50"]), r["out_source"]])
    return table(["workload", "archetype", "n", "prompt p50", "prompt p90", "output p50", "output p90",
                  "decode share p50", "output measured from"], body)


def t_reuse():
    body = []
    for r in rows("reuse_distance.csv"):
        body.append([r["workload"], r["archetype"], n(r["samples"]), pct(r["copy_rate"]), pct(r["src_prompt"]),
                     n(r["dist_p50"]), n(r["dist_p90"]), pct(r["MR@256"]), pct(r["MR@512"]), pct(r["MR@1024"]),
                     pct(r["MR@4096"]), pct(r["MR@16384"])])
    return table(["workload", "archetype", "samples", "copy rate", "source in prompt", "distance p50", "distance p90",
                  "MR@256", "MR@512", "MR@1K", "MR@4K", "MR@16K"], body)


def t_scope():
    body = []
    for r in rows("context_scope.csv"):
        body.append([r["task"], r["archetype"], n(r["queries"]), n(r["queries_per_context"], 1), n(r["k80_p50"], 1),
                     pct(r["scope80_p50"]), pct(r["spread80_p50"]), pct(r["union_scope_p50"]),
                     n(r["pair_jaccard_mean"], 2)])
    return table(["task", "archetype", "queries", "queries / context", "k80 chunks p50", "scope80 p50",
                  "spread80 p50", "union scope p50", "pairwise Jaccard"], body)


def t_locomo():
    body = []
    for r in rows("locomo_evidence.csv"):
        body.append([f"{r['category']} ({r['name']})", n(r["questions"]), n(r["hops_mean"], 2),
                     pct(r["multi_evidence_share"]), n(r["dist_far_p50"]), n(r["dist_far_p90"]), n(r["dist_near_p50"]),
                     n(r["depth_p50"], 2)])
    return table(["LoCoMo category", "questions", "evidence turns (mean)", "multi-evidence share",
                  "farthest evidence p50 (tokens back)", "farthest p90", "nearest evidence p50", "depth p50"], body)


def t_redundancy():
    body = [[r["context"], r["archetype"], n(r["samples"]), n(r["gzip_ratio_p50"], 2), pct(r["repeated_4gram_rate_p50"])]
            for r in rows("context_redundancy.csv")]
    return table(["context", "archetype", "samples", "gzip ratio p50", "repeated 4-grams p50"], body)


def t_agents():
    body = []
    for r in rows("agent_summary.csv"):
        body.append([r["workload"], n(r["n"]), n(r["steps_p50"]), n(r["steps_p90"]), n(r["persistent_prefix_p50"]),
                     n(r["peak_context_p50"]), n(r["peak_context_p90"]), n(r["peak_context_max"]),
                     pct(r["obs_share_p50"]), pct(r["decode_share_p50"]), pct(r["persistent_share_p50"]),
                     n(r["reuse_saving_x_p50"], 1) + "x"])
    return table(["workload", "trajectories", "steps p50", "steps p90", "persistent prefix p50", "peak context p50",
                  "peak p90", "peak max", "observation share", "model-output share", "persistent share",
                  "prefill saved by prefix reuse"], body)


def t_traces():
    body = []
    for r in rows("serving_traces_all.csv"):
        body.append([r["trace"], n(r["requests"]), n(r["in_p50"]), n(r["in_p90"]), n(r["out_p50"]), n(r["out_p90"]),
                     pct(r["decode_token_share"]), pct(r.get("multi_turn_req_share")), pct(r.get("ideal_hit")),
                     pct(r.get("same_session_share")), pct(r.get("nonprefix_repeat_blocks"))])
    return table(["trace", "requests", "input p50", "input p90", "output p50", "output p90", "decode share of tokens",
                  "follow-up turns", "ideal prefix hit", "hits from same session", "non-prefix repeats"], body)


def t_policies():
    m = rows("serving_mrc_all.csv")
    body = []
    for trace in dict.fromkeys(r["trace"] for r in m):
        caps = sorted({int(r["capacity_tokens"]) for r in m if r["trace"] == trace and r["policy"] == "OPT"})
        for c in caps:
            get = {r["policy"]: float(r["hit_ratio"]) for r in m if r["trace"] == trace and int(r["capacity_tokens"]) == c}
            best = max(get, key=get.get)
            body.append([trace, f"{c:,}"] + [f"{get[p]:.3f}" + (" *" if p == best else "") for p in
                                             ["LRU", "FIFO", "LFU", "OPT"]])
    return table(["trace", "cache capacity (tokens)", "LRU", "FIFO", "LFU", "OPT (Belady)"], body)


def t_reuse_time():
    body = [[r["trace"], r["reuse"], n(r["hits"]), n(r["p10_s"], 1), n(r["p50_s"], 1), n(r["p90_s"], 1), n(r["p99_s"], 1)]
            for r in rows("serving_reuse_time_all.csv")]
    return table(["trace", "reuse source", "block hits", "p10 (s)", "p50 (s)", "p90 (s)", "p99 (s)"], body)


if __name__ == "__main__":
    parts = [
        "# Measurement tables (auto-generated by analysis/render_tables.py)",
        "Token counts use the Llama-3 tokenizer. MR@W = share of verbatim (copy) dependencies whose nearest source is more "
        "than W tokens back. See docs/characterization.md for definitions and discussion.",
        "## T1. Prefill / decode length profiles (results/length_profiles.csv)", t_lengths(),
        "## T2. Lexical dependency (reuse) distance of decoded tokens (results/reuse_distance.csv)", t_reuse(),
        "## T3. Evidence scope, dispersion and multi-query overlap, 128-token chunks (results/context_scope.csv)",
        t_scope(),
        "## T4. LoCoMo gold-evidence geometry (results/locomo_evidence.csv)", t_locomo(),
        "## T5. Context redundancy (results/context_redundancy.csv)", t_redundancy(),
        "## T6. Agent trajectories: KV growth (results/agent_summary.csv)", t_agents(),
        "## T7. Production serving traces (results/serving_traces_all.csv)", t_traces(),
        "## T8. Prefix-cache hit ratio by eviction policy (results/serving_mrc_all.csv; * = best)", t_policies(),
        "## T9. Time between a block's reuse (results/serving_reuse_time_all.csv)", t_reuse_time(),
    ]
    out = REPO / "docs" / "tables.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n")
    print("wrote", out)
