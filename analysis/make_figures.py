"""Render the figures in figures/ from the CSV/JSON files in results/.

Style follows a validated categorical palette (fixed slot order; <= 3 hues on scatter plots,
<= 4 lines per panel), hairline solid grids, 2px lines, >= 8px markers with a surface ring,
selective direct labels and a legend whenever there are >= 2 series. Every figure has a
table twin in results/.
"""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAYPT = "#a9a79f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def rows(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


# --------------------------------------------------------------------- fig 1: workload map
REGIME = {  # archetype -> cache-growth regime (colour group)
    "compact": 0, "short_gen": 0, "long_reason": 2, "long_gen": 2, "sparse_retrieval": 1, "multi_hop": 1,
    "aggregation": 1, "many_shot_icl": 1, "lc_code_struct": 1, "shared_context_multiturn": 3, "agentic": 3,
    "streaming_lm": 2}
REGIME_STYLE = [("short prompt, short output", GRAYPT), ("prefill-dominated", SLOTS[0]),
                ("decode-dominated", SLOTS[1]), ("accumulating (multi-turn / agentic)", SLOTS[2])]
# workload -> (label, dx, dy, ha) ; only these points get a direct label
LABEL = {"gsm8k-8shot-cot": ("GSM8K 8-shot CoT", 6, -11, "left"), "math500": ("MATH-500", 6, -10, "left"),
         "humaneval": ("HumanEval", 6, -11, "left"),
         "alpacaeval:Meta-Llama-3-8B-Instruct": ("AlpacaEval (6 models)", 4, 8, "left"),
         "arenahard-v2:hard_prompt:qwq-32b": ("Arena-Hard v2, QwQ-32B thinking", 6, 4, "left"),
         "arenahard-v2:hard_prompt:deepseek-r1": ("Arena-Hard v2, R1 thinking", 6, -10, "left"),
         "longbench-write(required)": ("LongBench-Write (required length)", -6, 6, "right"),
         "leval:narrative_qa": ("L-Eval NarrativeQA", 6, 4, "left"),
         "leval:meeting_summ": ("L-Eval meeting summ.", 6, -11, "left"), "quality-dev": ("QuALITY (MC)", 6, 4, "left"),
         "locomo-qa": ("LoCoMo QA", -6, -11, "right"), "crosscodeeval:python": ("CrossCodeEval", 6, 4, "left"),
         "mtbench101-final-turn": ("MT-Bench-101", 6, 4, "left"), "bbh-3shot-cot": ("BBH 3-shot CoT", -6, 6, "right"),
         "swebench-verified:claude-sonnet-4-20250514": ("SWE-bench agent (Sonnet 4)", 6, 4, "left"),
         "tau-bench:sonnet-35-new-retail": ("tau-bench retail", 6, 4, "left")}
TRACE = {  # trace -> (regime group, label, dx, dy, ha); None label = unlabeled
    "azure2023:code": (1, "Azure'23 code", 6, -11, "left"),
    "azure2023:conversation": (3, "Azure'23 conv.", 6, -11, "left"),
    "burstgpt:ChatGPT:API": (0, "BurstGPT API (GPT-3.5)", -6, -11, "right"),
    "burstgpt:ChatGPT:Conversation": (3, "BurstGPT chat (GPT-3.5 / GPT-4)", -6, 6, "right"),
    "burstgpt:GPT-4:API": (0, "BurstGPT API (GPT-4)", -6, 6, "right"),
    "burstgpt:GPT-4:Conversation": (3, None, 0, 0, "left"),
    "qwen:to-B API": (0, "Qwen to-B API", 6, -11, "left"),
    "qwen:to-C chat": (3, "Qwen to-C chat", -6, 6, "right"),
    "qwen:thinking": (2, "Qwen thinking", 6, 4, "left"),
    "qwen:coder": (1, "Qwen coder", 6, 4, "left"),
    "mooncake:conversation": (3, "Mooncake conv.", 6, -11, "left"),
    "mooncake:tool&agent": (3, "Mooncake tool&agent", 6, 4, "left"),
    "mooncake:synthetic": (1, "Mooncake synthetic", 6, 4, "left"),
}


def _label(ax, spec, x, y):
    if spec and spec[0]:
        text, dx, dy, ha = spec
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.2, color=INK2, ha=ha)


def fig_workload_map():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True)
    ax = axes[0]
    pts = []
    for r in rows("length_profiles.csv"):
        if not r["out_p50"] or r["workload"].startswith("niah"):
            continue
        g = REGIME.get(r["archetype"], 0)
        pts.append((r["workload"], g, fnum(r["in_p10"]), fnum(r["in_p50"]), fnum(r["in_p90"]),
                    fnum(r["out_p10"]), fnum(r["out_p50"]), fnum(r["out_p90"])))
    for r in rows("agent_summary.csv"):  # agents: tokens prefilled vs. decoded over a whole trajectory
        pts.append((r["workload"], 3, None, fnum(r["peak_context_p50"]) - fnum(r["decoded_total_p50"]),
                    None, None, fnum(r["decoded_total_p50"]), None))
    for w, g, i10, i50, i90, o10, o50, o90 in pts:
        c = REGIME_STYLE[g][1]
        if i10 is not None:
            ax.plot([max(i10, 1), max(i90, 1)], [max(o50, 1)] * 2, color=c, lw=1, alpha=0.45, zorder=2)
            ax.plot([i50, i50], [max(o10, 1), max(o90, 1)], color=c, lw=1, alpha=0.45, zorder=2)
        ax.scatter([i50], [max(o50, 1)], s=46, color=c, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        _label(ax, LABEL.get(w), i50, max(o50, 1))
    ax.set_title("(a) Benchmarks: median prompt vs. median decoded tokens\n"
                 "(whiskers p10-p90; agents: tokens prefilled vs. decoded per trajectory)",
                 fontsize=9, loc="left", color=INK)
    ax = axes[1]
    for r in rows("serving_traces_all.csv"):
        g, *spec = TRACE[r["trace"]]
        c = REGIME_STYLE[g][1]
        i10, i50, i90 = fnum(r["in_p10"]), fnum(r["in_p50"]), fnum(r["in_p90"])
        o10, o50, o90 = fnum(r["out_p10"]), fnum(r["out_p50"]), fnum(r["out_p90"])
        ax.plot([max(i10, 1), i90], [max(o50, 1)] * 2, color=c, lw=1, alpha=0.45)
        ax.plot([i50, i50], [max(o10, 1), max(o90, 1)], color=c, lw=1, alpha=0.45)
        ax.scatter([i50], [max(o50, 1)], s=46, color=c, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        _label(ax, spec, i50, max(o50, 1))
    ax.set_title("(b) Production traces (same axes; per-request input vs. output)", fontsize=9, loc="left", color=INK)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(8, 3e5); ax.set_ylim(0.8, 3e4)
        ax.set_xlabel("prefill tokens (prompt / context)")
    axes[0].set_ylabel("decoded tokens (output)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for l, c in REGIME_STYLE]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig1_workload_map.png")


# --------------------------------------------------------------------- fig 2: reuse-distance miss ratio
def fig_reuse():
    data = {r["workload"]: r for r in rows("reuse_distance.csv")}
    ws = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    panels = [
        ("(a) Short-context generation", ["gsm8k (ref CoT)", "math500 (ref solution)", "humaneval (canonical)",
                                          "alpacaeval: Llama-3-8B-Instruct"],
         ["GSM8K", "MATH-500", "HumanEval", "AlpacaEval"]),
        ("(b) Decode-dominated (thinking models)", ["arenahard-v2 hard: deepseek-r1 (thinking)",
                                                    "arenahard-v2 hard: qwq-32b (thinking)",
                                                    "arenahard-v2 creative: qwq-32b (thinking)"],
         ["R1 hard prompts", "QwQ hard prompts", "QwQ creative writing"]),
        ("(c) Long-context QA / summarization", ["leval: financial_qa", "leval: multidoc_qa", "leval: meeting_summ",
                                                 "locomo qa"],
         ["financial QA", "multi-doc QA", "meeting summ.", "LoCoMo QA"]),
        ("(d) Agentic trajectories", ["swebench: claude-sonnet-4 (mini-swe-agent)",
                                      "swebench: qwen3-coder-480b (mini-swe-agent)", "swebench: gpt-5 (mini-swe-agent)",
                                      "tau-bench retail: sonnet-3.5"],
         ["SWE-bench Sonnet 4", "SWE-bench Qwen3-Coder", "SWE-bench GPT-5", "tau-bench retail"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    for ax, (title, names, labels) in zip(axes, panels):
        for k, (n, lab) in enumerate(zip(names, labels)):
            r = data[n]
            y = [fnum(r[f"MR@{w}"]) for w in ws]
            ax.plot(ws, y, color=SLOTS[k], label=lab)
            ax.scatter([ws[-1]], [y[-1]], s=30, color=SLOTS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024, 4096, 16384])
        ax.set_xticklabels(["64", "256", "1K", "4K", "16K"])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("sliding-window size W (tokens)")
        ax.legend(fontsize=7.2, loc="lower left" if title.startswith("(c)") else "upper right")
    axes[0].set_ylabel("share of copy dependencies\nfarther back than W")
    fig.tight_layout()
    save(fig, "fig2_reuse_distance.png")


# --------------------------------------------------------------------- fig 3: agent context growth
def fig_agents():
    growth = json.load(open(RESULTS / "agent_growth.json"))
    panels = [("(a) SWE-bench Verified, mini-SWE-agent", [
        ("swebench-verified:claude-sonnet-4-20250514", "Claude Sonnet 4"),
        ("swebench-verified:qwen3-coder-480b-a35b-instruct", "Qwen3-Coder-480B"),
        ("swebench-verified:gpt-5", "GPT-5 (visible tokens)")]),
        ("(b) tau-bench (policy + tool schemas persist)", [
            ("tau-bench:sonnet-35-new-retail", "retail, Sonnet 3.5"),
            ("tau-bench:gpt-4o-airline", "airline, GPT-4o")])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (title, series) in zip(axes, panels):
        for k, (w, lab) in enumerate(series):
            curves = growth[w]
            L = int(np.percentile([len(c) for c in curves], 90))
            steps = np.arange(1, L + 1)
            med, lo, hi = [], [], []
            for s in range(L):
                v = [c[s] for c in curves if len(c) > s]
                med.append(np.median(v)); lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
            ax.fill_between(steps, lo, hi, color=SLOTS[k], alpha=0.10, linewidth=0)
            ax.plot(steps, med, color=SLOTS[k], label=f"{lab} (median, p10-p90 band)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("LLM call (agent step)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("prompt tokens at this call")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_agent_context_growth.png")


# --------------------------------------------------------------------- fig 4: scope / multi-query
def fig_scope():
    rs = [r for r in rows("context_scope.csv") if fnum(r["union_scope_p50"]) and float(r["queries_per_context"]) >= 2]
    rs.sort(key=lambda r: fnum(r["union_scope_p50"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    y = np.arange(len(rs))
    one = [fnum(r["scope80_p50"]) for r in rs]
    uni = [fnum(r["union_scope_p50"]) for r in rs]
    for i in y:
        ax.plot([one[i], uni[i]], [i, i], color=AXIS, lw=2, zorder=1)
    ax.scatter(one, y, s=46, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.6, zorder=3,
               label="one query: chunks an oracle must keep (median)")
    ax.scatter(uni, y, s=46, color=SLOTS[1], edgecolor=SURFACE, linewidth=1.6, zorder=3,
               label="all queries on the same context: union (median)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['task']}  (~{float(r['queries_per_context']):.0f} q/ctx)" for r in rs], fontsize=8)
    ax.set_xlabel("fraction of 128-token chunks of the context needed (lexical oracle, 80% answer coverage)")
    ax.set_xlim(0, max(uni) * 1.1)
    ax.legend(fontsize=7.8, loc="lower right")
    ax.set_title("Query-aware compression for one question discards what the next question needs",
                 fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig4_multiquery_scope.png")


# --------------------------------------------------------------------- fig 5: serving MRCs
def fig_mrc():
    m = rows("serving_mrc_all.csv")
    panels = [("(a) Alibaba Qwen-Bailian (16-token blocks)", ["qwen:to-C chat", "qwen:to-B API", "qwen:thinking",
                                                            "qwen:coder"]),
              ("(b) Mooncake / Kimi (512-token blocks)", ["mooncake:conversation", "mooncake:tool&agent",
                                                          "mooncake:synthetic"])]
    ideal = {r["trace"]: fnum(r["ideal_hit"]) for r in rows("serving_traces_all.csv") if r.get("ideal_hit")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, names) in zip(axes, panels):
        for k, n in enumerate(names):
            pts = sorted((int(r["capacity_tokens"]), fnum(r["hit_ratio"])) for r in m
                         if r["trace"] == n and r["policy"] == "LRU")
            if not pts:
                continue
            x, yv = zip(*pts)
            ax.plot(x, yv, color=SLOTS[k], label=f"{n.split(':', 1)[1]}  (LRU; ideal {ideal.get(n, 0):.2f})")
            opt = sorted((int(r["capacity_tokens"]), fnum(r["hit_ratio"])) for r in m
                         if r["trace"] == n and r["policy"] == "OPT")
            if opt:
                ax.scatter(*zip(*opt), s=40, marker="D", color=SLOTS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("prefix-cache capacity (tokens of KV)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7.4, loc="upper left")
        ax.set_xticks([2 ** k for k in range(12, 27, 2)])
        ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M"])
    axes[0].set_ylabel("share of input tokens served\nfrom the prefix cache")
    fig.text(0.5, -0.03, "Lines: LRU hit-ratio curve (1 - miss ratio), prefix-contiguous hits only. Diamonds: Belady OPT "
             "at the two capacities where LRU reaches 50% / 90% of the ideal (infinite-cache) hit ratio.",
             ha="center", fontsize=7.8, color=INK2)
    fig.tight_layout()
    save(fig, "fig5_prefix_cache_mrc.png")


# --------------------------------------------------------------------- fig 6: suite coverage heatmap
def fig_suite_coverage():
    rs = rows("suite_coverage.csv")
    rs.sort(key=lambda r: -float(r["coverage"]))
    arch = ["compact", "short_gen", "long_reason", "long_gen", "sparse_retrieval", "multi_hop", "aggregation",
            "many_shot_icl", "lc_code_struct", "shared_context_multiturn", "agentic", "streaming_lm"]
    alab = ["compact", "short-gen", "long-reason", "long-gen", "sparse retr.", "multi-hop", "aggregation", "many-shot ICL",
            "long-ctx code", "multi-turn", "agentic", "streaming LM"]
    stress = ["growth=decode_dominated", "growth=accumulating", "query=deferred", "fidelity=exact_high_entropy",
              "dependency=persistent"]
    slab = ["decode-dominated", "accumulating", "query deferred", "exact high-entropy", "persistent instr."]
    cols = arch + stress
    M = np.array([[min(int(r[c]), 8) for c in cols] for r in rs], dtype=float)
    ramp = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(ramp); norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(11.5, 0.30 * len(rs) + 1.8))
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i, r in enumerate(rs):
        for j, c in enumerate(cols):
            v = int(r[c])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color="#ffffff" if min(v, 8) >= 4 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(alab + slab, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"{r['suite'][:46]}  ({float(r['coverage']):.0%})" for r in rs], fontsize=7.2)
    ax.axvline(len(arch) - 0.5, color=INK2, lw=1)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Benchmark suites x workload archetypes (cell = #tasks; right block = key eviction stressors;"
                 " % = weighted coverage)", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig6_suite_coverage.png")


if __name__ == "__main__":
    fig_suite_coverage()
    fig_workload_map()
    fig_reuse()
    fig_agents()
    fig_scope()
    fig_mrc()
