"""Map a benchmark suite onto the KV-stress taxonomy, report coverage gaps, and recommend tasks.

Examples
  python tools/coverage.py --list-suites
  python tools/coverage.py --suite "RULER"
  python tools/coverage.py --suite "SnapKV-era standard (LongBench EN-16 + NIAH)" --recommend 6
  python tools/coverage.py --tasks ruler:niah-single-3,scbench:retr-kv,aime --recommend 5
  python tools/coverage.py --custom my_tasks.yaml          # new benchmarks described with facet values
  python tools/coverage.py --matrix                        # all suites -> results/suite_coverage.csv + docs table
  python tools/coverage.py --recommend 14                  # a covering core suite built from scratch

Custom YAML format (one entry per task; see taxonomy/taxonomy.yaml for allowed values):
  - id: mybench:task1
    archetype: sparse_retrieval
    growth: prefill_dominated
    dependency: [point]
    query: deferred
    fidelity: exact_high_entropy
    composition: [structured]
    reuse: [session]
"""
import argparse
import csv
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAX = yaml.safe_load(open(ROOT / "taxonomy/taxonomy.yaml"))
CAT = yaml.safe_load(open(ROOT / "taxonomy/benchmarks.yaml"))
TASKS = {t["id"]: t for t in CAT["tasks"]}
ARCH = [a["id"] for a in TAX["archetypes"]]
ARCH_NAME = {a["id"]: a["name"] for a in TAX["archetypes"]}
FACETS = {k: [v["id"] for v in f["values"]] for k, f in TAX["facets"].items()}

# What a complete eviction-policy test suite must exercise. Weight = importance for coverage scoring.
REQUIREMENTS = (
    [("archetype", a, 3.0) for a in ARCH]
    + [("growth", v, 2.0) for v in FACETS["growth"]]
    + [("dependency", v, 2.0) for v in FACETS["dependency"]]
    + [("query", v, 2.0) for v in FACETS["query"]]
    + [("fidelity", v, 1.0) for v in FACETS["fidelity"]]
    + [("composition", v, 0.5) for v in FACETS["composition"]]
    + [("position", v, 0.5) for v in ["swept", "early", "uniform"]]
    + [("reuse", v, 1.0) for v in ["shared_prefix", "session", "agentic_loop"]]
)
SUITE_PRIORITY = ["RULER", "LongBench", "HELMET", "SCBench", "InfiniteBench", "LoCoMo", "L-Eval", "SWE-bench",
                  "tau-bench", "AIME", "MATH-500", "GSM8K", "LongBench-Write", "LongGenBench", "LongMemEval"]


def covered_by(t, use_alt=False):
    """Set of (facet, value) pairs a task covers."""
    c = {("archetype", t["archetype"])}
    c |= {("archetype", s) for s in t.get("secondary", [])}
    c |= {("growth", t["growth"]), ("query", t["query"]), ("fidelity", t["fidelity"]), ("position", t["position"])}
    c |= {("dependency", d) for d in t["dependency"]}
    c |= {("composition", d) for d in t["composition"]}
    c |= {("reuse", r) for r in t["reuse"]}
    if use_alt and t.get("alt"):
        c.add(("query", t["alt"]["query"]))
        c |= {("reuse", r) for r in t["alt"].get("reuse", [])}
    return c


def coverage(task_list, use_alt=False):
    got = set().union(*[covered_by(t, use_alt) for t in task_list]) if task_list else set()
    total = sum(w for *_, w in REQUIREMENTS)
    score = sum(w for f, v, w in REQUIREMENTS if (f, v) in got)
    missing = [(f, v) for f, v, _ in REQUIREMENTS if (f, v) not in got]
    return got, score / total, missing


def resolve(names):
    out = []
    for n in names:
        n = n.strip()
        if not n:
            continue
        if n in TASKS:
            out.append(TASKS[n])
        elif n.endswith(":"):
            out += [t for i, t in TASKS.items() if i.startswith(n)]
        else:
            sys.exit(f"unknown task id: {n} (see taxonomy/benchmarks.yaml)")
    return out


def load_custom(path):
    items = yaml.safe_load(open(path))
    out = []
    for t in items:
        t = dict(t)
        t.setdefault("secondary", []); t.setdefault("composition", []); t.setdefault("reuse", ["none"])
        t.setdefault("position", "uniform"); t.setdefault("flags", []); t.setdefault("domains", [])
        assert t["archetype"] in ARCH, f"{t['id']}: unknown archetype {t['archetype']}"
        for k in ("growth", "query", "fidelity", "position"):
            assert t[k] in FACETS[k], f"{t['id']}: {k}={t[k]} not in {FACETS[k]}"
        for k in ("dependency", "composition", "reuse"):
            for v in t[k]:
                assert v in FACETS[k], f"{t['id']}: {k}={v} not in {FACETS[k]}"
        out.append(t)
    return out


def recommend(seed, k, use_alt=False, pool=None):
    """Greedy weighted set cover: add up to k catalogue tasks that cover the most missing requirement weight."""
    pool = pool or list(TASKS.values())
    weights = {(f, v): w for f, v, w in REQUIREMENTS}
    have = set().union(*[covered_by(t, use_alt) for t in seed]) if seed else set()
    chosen = []
    conf_rank = {"[verified]": 0, "[likely]": 1, "[unverified]": 2}

    def prio(t):
        suite = t.get("suite", "")
        p = next((i for i, s in enumerate(SUITE_PRIORITY) if s.lower() in suite.lower()), len(SUITE_PRIORITY))
        return (conf_rank.get(t.get("confidence"), 3), p)

    for _ in range(k):
        best, gain = None, 0.0
        for t in pool:
            if t in seed or t in chosen:
                continue
            g = sum(weights.get(x, 0) for x in covered_by(t, use_alt) - have)
            if g > gain + 1e-9 or (abs(g - gain) < 1e-9 and best is not None and g > 0 and prio(t) < prio(best)):
                best, gain = t, g
        if best is None or gain <= 0:
            break
        chosen.append(best)
        have |= covered_by(best, use_alt)
    return chosen


def report(name, task_list, use_alt=False, k_rec=0):
    got, score, missing = coverage(task_list, use_alt)
    print(f"\n## Coverage report: {name}  ({len(task_list)} tasks{', alt protocols counted' if use_alt else ''})")
    print(f"weighted coverage score: {score:.0%}\n")
    print("| archetype | tasks |")
    print("|---|---|")
    for a in ARCH:
        n = sum(1 for t in task_list if t["archetype"] == a or a in t.get("secondary", []))
        print(f"| {ARCH_NAME[a]} | {n if n else '**none**'} |")
    print("\n| facet | covered values | missing values |")
    print("|---|---|---|")
    for f in ["growth", "dependency", "query", "fidelity", "composition", "position", "reuse"]:
        vals = [v for (ff, v) in sorted(got) if ff == f]
        miss = [v for (ff, v) in missing if ff == f]
        print(f"| {f} | {', '.join(vals) or '-'} | {', '.join(miss) or '-'} |")
    flags = {}
    for t in task_list:
        for x in t.get("flags", []):
            flags[x] = flags.get(x, 0) + 1
    if flags:
        print("\nvalidity flags present: " + ", ".join(f"{k} ({v} tasks)" for k, v in sorted(flags.items())))
    alts = [t["id"] for t in task_list if t.get("alt")]
    if alts and not use_alt:
        print(f"\n{len(alts)} task(s) can also be run in shared-context mode (compress once, ask all questions), "
              "which adds query=deferred coverage; rerun with --alt to count it.")
    if k_rec:
        rec = recommend(task_list, k_rec, use_alt)
        print(f"\n### Suggested additions (greedy, up to {k_rec})")
        print("| task id | archetype | adds |")
        print("|---|---|---|")
        have = got
        for t in rec:
            new = sorted(covered_by(t, use_alt) - have)
            have = have | covered_by(t, use_alt)
            print(f"| {t['id']} | {t['archetype']} | {', '.join(f'{f}={v}' for f, v in new)} |")
        _, s2, _ = coverage(task_list + rec, use_alt)
        print(f"\ncoverage after additions: {s2:.0%}")


def matrix(use_alt=False):
    """Suite x archetype task counts and suite x key-stressor flags; written as CSV and a markdown table."""
    stressors = [("growth", "decode_dominated"), ("growth", "accumulating"), ("query", "deferred"),
                 ("query", "evolving"), ("fidelity", "exact_high_entropy"), ("dependency", "persistent"),
                 ("dependency", "global"), ("dependency", "self"), ("reuse", "session")]
    rows, exact = [], {}
    for name, ids in CAT["suites"].items():
        tl = [TASKS[i] for i in ids]
        got, score, _ = coverage(tl, use_alt)
        exact[name] = score  # format percentages from the unrounded score (no double rounding)
        r = {"suite": name, "tasks": len(tl), "coverage": round(score, 5)}
        for a in ARCH:
            r[a] = sum(1 for t in tl if t["archetype"] == a or a in t.get("secondary", []))
        for f, v in stressors:
            r[f"{f}={v}"] = sum(1 for t in tl if (f, v) in covered_by(t, use_alt))
        rows.append(r)
    out = ROOT / "results" / "suite_coverage.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    short = {"compact": "CMP", "short_gen": "SGEN", "long_reason": "LRSN", "long_gen": "LGEN",
             "sparse_retrieval": "RETR", "multi_hop": "MHOP", "aggregation": "AGG", "many_shot_icl": "ICL",
             "lc_code_struct": "CODE", "shared_context_multiturn": "MTURN", "agentic": "AGENT", "streaming_lm": "STRM"}
    md = ["| suite | n | score | " + " | ".join(short[a] for a in ARCH) + " | decode-dom. | accum. | q-deferred | "
          "exact-HE | persistent |", "|---|---|---|" + "---|" * (len(ARCH) + 5)]
    for r in sorted(rows, key=lambda r: -exact[r["suite"]]):
        cells = [str(r[a]) if r[a] else "·" for a in ARCH]
        st = [r["growth=decode_dominated"], r["growth=accumulating"], r["query=deferred"],
              r["fidelity=exact_high_entropy"], r["dependency=persistent"]]
        md.append(f"| {r['suite']} | {r['tasks']} | {exact[r['suite']]:.0%} | " + " | ".join(cells) + " | "
                  + " | ".join(str(x) if x else "·" for x in st) + " |")
    (ROOT / "results" / "suite_coverage.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out} and results/suite_coverage.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite")
    ap.add_argument("--tasks", help="comma-separated task ids (prefix 'suite:' selects all tasks of a suite)")
    ap.add_argument("--custom", help="YAML file describing tasks not in the catalogue")
    ap.add_argument("--recommend", type=int, default=0, help="suggest up to N catalogue tasks that close gaps")
    ap.add_argument("--alt", action="store_true", help="count alternative (shared-context) protocols")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--list-suites", action="store_true")
    a = ap.parse_args()
    if a.list_suites:
        for n, ids in CAT["suites"].items():
            print(f"{n}  ({len(ids)} tasks)")
        sys.exit()
    if a.matrix:
        matrix(a.alt)
        sys.exit()
    tl, name = [], "custom selection"
    if a.suite:
        if a.suite not in CAT["suites"]:
            sys.exit(f"unknown suite; try --list-suites")
        tl += [TASKS[i] for i in CAT["suites"][a.suite]]
        name = a.suite
    if a.tasks:
        tl += resolve(a.tasks.split(","))
    if a.custom:
        tl += load_custom(a.custom)
    if not tl and not a.recommend:
        ap.print_help()
        sys.exit()
    report(name if tl else "empty suite (build from scratch)", tl, a.alt, a.recommend)
