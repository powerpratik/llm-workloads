"""Build taxonomy/benchmarks.yaml: every benchmark task with its KV-stress signature.

Inputs
  sources/longctx_tasks.jsonl  task-level facts for 29 long-context / long-output suites, compiled from
                               official repositories (fields: lengths, metric, query position, evidence
                               structure, sources, confidence tag)
  EXTRA_TASKS (below)          short-context, reasoning, chat, agentic, streaming and KV-evaluation tasks
Facet values are assigned by the explicit rules below (suite/category defaults + per-task overrides),
so every mapping decision is auditable and easy to change. Values must exist in taxonomy.yaml.

  python taxonomy/build_catalog.py        -> writes taxonomy/benchmarks.yaml and validates it
"""
import json
import pathlib
import re

import yaml

HERE = pathlib.Path(__file__).resolve().parent
TAX = yaml.safe_load(open(HERE / "taxonomy.yaml"))

SUITE_SHORT = {
    "LongBench (v1)": "longbench", "LongBench v2": "longbench-v2", "RULER": "ruler",
    "InfiniteBench (∞Bench)": "infinitebench", "HELMET": "helmet", "L-Eval": "leval", "ZeroSCROLLS": "zeroscrolls",
    "LooGLE": "loogle", "BABILong": "babilong", "Michelangelo (Latent Structure Queries)": "michelangelo",
    "LV-Eval": "lv-eval", "NoCha": "nocha", "Loong": "loong", "NeedleBench (OpenCompass, v2)": "needlebench",
    "Counting-Stars": "counting-stars", "LongICLBench": "longiclbench", "NoLiMa": "nolima", "LongProc": "longproc",
    "Needle-in-a-Haystack (gkamradt)": "niah", "Passkey retrieval (Mohtashami & Jaggi 2023)": "passkey",
    "SCBench (SharedContextBench)": "scbench", "OpenAI MRCR": "mrcr", "Fiction.LiveBench": "fiction-livebench",
    "LongMemEval": "longmemeval", "LoCoMo": "locomo", "LongBench-Write (LongWriter)": "longbench-write",
    "LongGenBench (Liu et al. 2024)": "longgenbench-liu", "LongGenBench (Wu et al. 2024)": "longgenbench-wu",
    "HelloBench": "hellobench",
}


def slug(s):
    s = s.replace("∞Bench", "InfiniteBench")
    s = re.sub(r"\(.*?\)", "", s).strip() or s
    s = s.split(":")[0] if s.lower().startswith("niah v2") else s
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ----------------------------------------------------------------------------------------------------
# Signature shorthands:  A=archetype, D=dependency (first = primary), G=growth, Q=query, F=fidelity,
# C=composition, P=position, R=reuse, M=domains, X=validity flags, alt=alternative protocol
def sig(A, D, F, C, M, G="prefill_dominated", Q="known_last", P="uniform", R=("none",), X=(), **kw):
    d = dict(archetype=A, dependency=list(D), growth=G, query=Q, fidelity=F, composition=list(C),
             position=P, reuse=list(R), domains=list(M), flags=list(X))
    d.update(kw)
    return d


SR, MH, AG, ICL, CODE = "sparse_retrieval", "multi_hop", "aggregation", "many_shot_icl", "lc_code_struct"
MT, AGENT, SG, LR, LG, CMP, STR = ("shared_context_multiturn", "agentic", "short_gen", "long_reason", "long_gen",
                                   "compact", "streaming_lm")
SHARED_ALT = {"query": "deferred", "reuse": ["session"],
              "how": "prefill the context once and ask every question of that context against the same "
                     "(compressed) KV cache instead of re-prefilling per question"}

CATEGORY_RULES = {  # (suite short name, suite_category) -> signature
    ("longbench", "Single-doc QA"): sig(SR, ["point"], "semantic", ["prose"], ["reading"]),
    ("longbench", "Multi-doc QA"): sig(MH, ["multi", "point"], "semantic", ["multidoc"], ["retrieval_rag", "reading"],
                                        X=["parametric_leakage"]),
    ("longbench", "Summarization"): sig(AG, ["global"], "semantic", ["prose"], ["summarization"], X=["fuzzy_metric"]),
    ("longbench", "Few-shot learning"): sig(ICL, ["global", "point"], "exact_low_entropy", ["demonstrations"], ["icl"]),
    ("longbench", "Synthetic"): sig(SR, ["point"], "exact_low_entropy", ["multidoc"], ["reading"], X=["short_answer"]),
    ("longbench", "Code completion"): sig(CODE, ["local", "point"], "exact_high_entropy", ["code"], ["code"], P="recent"),
    ("infinitebench", None): sig(SR, ["point"], "semantic", ["prose"], ["reading"]),
    ("longbench-v2", None): sig(SR, ["point", "multi"], "exact_low_entropy", ["prose"], ["reading"], X=["short_answer"]),
    ("longbench-v2", "II. Multi-Document QA"): sig(MH, ["multi", "global"], "exact_low_entropy", ["multidoc"], ["reading"],
                                                   X=["short_answer"]),
    ("ruler", "Retrieval (NIAH)"): sig(SR, ["point"], "exact_high_entropy", ["prose", "synthetic_filler"], ["reading"],
                                       P="swept"),
    ("ruler", "Multi-hop tracing"): sig(MH, ["multi"], "exact_high_entropy", ["synthetic_filler"], ["reasoning"], P="swept"),
    ("ruler", "Aggregation"): sig(AG, ["global"], "exact_low_entropy", ["synthetic_filler"], ["reasoning"]),
    ("ruler", "Question answering"): sig(SR, ["point"], "semantic", ["multidoc"], ["retrieval_rag"], P="swept",
                                         X=["parametric_leakage"]),
    ("helmet", "Synthetic recall"): sig(SR, ["point"], "exact_high_entropy", ["synthetic_filler"], ["reading"], P="swept"),
    ("helmet", "Retrieval-augmented generation"): sig(SR, ["point"], "semantic", ["multidoc"], ["retrieval_rag"],
                                                      X=["parametric_leakage"]),
    ("helmet", "Passage re-ranking"): sig(AG, ["global"], "exact_high_entropy", ["multidoc"], ["retrieval_rag"]),
    ("helmet", "Generation with citations"): sig(MH, ["multi", "persistent"], "semantic", ["multidoc", "instructions"],
                                                 ["retrieval_rag"], Q="known_first"),
    ("helmet", "Long-document QA"): sig(MH, ["multi", "point"], "semantic", ["prose"], ["reading"]),
    ("helmet", "Summarization"): sig(AG, ["global"], "semantic", ["prose"], ["summarization"], X=["fuzzy_metric"]),
    ("helmet", "Many-shot in-context learning"): sig(ICL, ["global", "point"], "exact_low_entropy", ["demonstrations"], ["icl"]),
    ("leval", "Closed-ended"): sig(SR, ["point"], "exact_low_entropy", ["prose"], ["reading"], X=["short_answer"],
                                   alt=SHARED_ALT),
    ("leval", "Open-ended"): sig(SR, ["point"], "semantic", ["prose"], ["reading"], alt=SHARED_ALT),
    ("zeroscrolls", "Summarization"): sig(AG, ["global"], "semantic", ["prose"], ["summarization"], X=["fuzzy_metric"]),
    ("zeroscrolls", "Query-based summarization"): sig(AG, ["global", "point"], "semantic", ["prose"], ["summarization"]),
    ("zeroscrolls", "QA"): sig(SR, ["point"], "semantic", ["prose"], ["reading"]),
    ("zeroscrolls", "Multiple-choice QA"): sig(SR, ["point", "global"], "exact_low_entropy", ["prose"], ["reading"]),
    ("zeroscrolls", "Aggregation"): sig(AG, ["global"], "exact_high_entropy", ["prose"], ["summarization"]),
    ("loogle", "Long dependency"): sig(MH, ["multi", "global"], "semantic", ["prose"], ["reading"]),
    ("loogle", "Short dependency"): sig(SR, ["point"], "semantic", ["prose"], ["reading"]),
    ("babilong", None): sig(SR, ["point"], "exact_low_entropy", ["prose", "synthetic_filler"], ["reasoning"], P="swept"),
    ("lv-eval", "Single-hop QA"): sig(SR, ["point"], "semantic", ["multidoc"], ["reading"]),
    ("lv-eval", "Multi-hop QA"): sig(MH, ["multi"], "semantic", ["multidoc"], ["reading"]),
    ("loong", None): sig(MH, ["multi"], "semantic", ["multidoc"], ["reading"]),
    ("needlebench", None): sig(SR, ["point"], "exact_high_entropy", ["prose"], ["reading"], P="swept"),
    ("counting-stars", None): sig(AG, ["global", "multi"], "exact_high_entropy", ["prose", "synthetic_filler"],
                                  ["reading"], P="swept"),
    ("longiclbench", None): sig(ICL, ["global", "point"], "exact_low_entropy", ["demonstrations"], ["icl"]),
    ("nolima", None): sig(SR, ["point"], "exact_low_entropy", ["prose"], ["reading"], P="swept"),
    ("longproc", None): sig(LR, ["self", "point"], "exact_high_entropy", ["instructions"], ["reasoning"],
                            G="decode_dominated", Q="evolving"),
    ("niah", None): sig(SR, ["point"], "exact_high_entropy", ["prose"], ["reading"], P="swept"),
    ("passkey", None): sig(SR, ["point"], "exact_high_entropy", ["synthetic_filler"], ["reading"], P="swept"),
    ("scbench", None): sig(MT, ["point"], "exact_high_entropy", ["structured"], ["dialogue_memory"], G="accumulating",
                           Q="deferred", R=["session", "shared_prefix"]),
    ("mrcr", None): sig(SR, ["point"], "exact_high_entropy", ["dialogue"], ["dialogue_memory"], G="mixed_long",
                        secondary=[MT]),
    ("fiction-livebench", None): sig(MH, ["multi", "global"], "semantic", ["prose"], ["reading"]),
    ("longmemeval", None): sig(MT, ["point"], "semantic", ["dialogue"], ["dialogue_memory"], alt=SHARED_ALT),
    ("locomo", None): sig(MT, ["point"], "semantic", ["dialogue"], ["dialogue_memory"], alt=SHARED_ALT),
    ("longbench-write", None): sig(LG, ["self", "persistent"], "open", ["generated_trace"], ["writing"],
                                   G="decode_dominated", Q="evolving", X=["fuzzy_metric"]),
    ("longgenbench-liu", None): sig(LR, ["point", "self"], "exact_high_entropy", ["demonstrations"], ["math", "reasoning"],
                                    G="mixed_long", Q="evolving"),
    ("longgenbench-wu", None): sig(LG, ["persistent", "self"], "open", ["instructions", "generated_trace"], ["writing"],
                                   G="decode_dominated", Q="evolving"),
    ("hellobench", None): sig(LG, ["self"], "open", ["generated_trace"], ["writing"], G="decode_dominated", Q="evolving",
                              X=["fuzzy_metric"]),
    ("michelangelo", None): sig(AG, ["global", "multi"], "exact_low_entropy", ["synthetic_filler"], ["reasoning"]),
    ("nocha", None): sig(MH, ["multi", "global"], "exact_low_entropy", ["prose"], ["reading"], alt=SHARED_ALT),
}

LV2 = dict(fidelity="exact_low_entropy", flags=["short_answer"])
TASK_OVERRIDES = {  # id -> partial signature (merged over the category rule)
    "longbench:multifieldqa-zh": dict(domains=["reading", "multilingual"]),
    "longbench:dureader": dict(archetype=SR, dependency=["point", "multi"], domains=["retrieval_rag", "multilingual"]),
    "longbench:qmsum": dict(dependency=["global", "point"], composition=["dialogue"]),
    "longbench:multinews": dict(composition=["multidoc"]),
    "longbench:vcsum": dict(composition=["dialogue"], domains=["summarization", "multilingual"]),
    "longbench:triviaqa": dict(dependency=["point", "local"], fidelity="semantic", position="recent",
                               flags=["parametric_leakage"]),
    "longbench:samsum": dict(dependency=["point", "global"], fidelity="semantic", position="recent"),
    "longbench:lsht": dict(domains=["icl", "multilingual"]),
    "longbench:passagecount": dict(archetype=AG, dependency=["global"]),
    "longbench:passageretrieval-zh": dict(domains=["reading", "multilingual"]),
    "longbench:repobench-p": dict(dependency=["point", "local"], composition=["code", "multidoc"]),
    "longbench-v2:single-doc-qa-academic": dict(archetype=MH, dependency=["multi"], **LV2),
    "longbench-v2:single-doc-qa-literary": dict(archetype=AG, dependency=["global"], **LV2),
    "longbench-v2:single-doc-qa-legal": dict(dependency=["point", "multi"], **LV2),
    "longbench-v2:single-doc-qa-financial": dict(dependency=["point", "multi"], **LV2),
    "longbench-v2:single-doc-qa-governmental": dict(dependency=["point", "multi"], **LV2),
    "longbench-v2:single-doc-qa-detective": dict(archetype=MH, dependency=["multi", "global"], **LV2),
    "longbench-v2:single-doc-qa-event-ordering": dict(archetype=AG, dependency=["global"], **LV2),
    "longbench-v2:long-icl-user-guide-qa": dict(dependency=["point", "multi"], domains=["icl", "reading"], **LV2),
    "longbench-v2:long-icl-new-language-translation": dict(archetype=ICL, dependency=["global", "multi"],
                                                           composition=["demonstrations", "prose"],
                                                           domains=["icl", "multilingual"], **LV2),
    "longbench-v2:long-icl-many-shot-learning": dict(archetype=ICL, dependency=["global", "point"],
                                                     composition=["demonstrations"], domains=["icl"], **LV2),
    "longbench-v2:long-dialogue-history-agent-history-qa": dict(archetype=MH, dependency=["multi"],
                                                                composition=["dialogue"], domains=["dialogue_memory"], **LV2),
    "longbench-v2:long-dialogue-history-dialogue-history-qa": dict(dependency=["point"], composition=["dialogue"],
                                                                   domains=["dialogue_memory"], **LV2),
    "longbench-v2:code-repository-code-repo-qa": dict(archetype=CODE, dependency=["point", "multi"],
                                                      composition=["code"], domains=["code"], **LV2),
    "longbench-v2:structured-data-table-qa": dict(archetype=CODE, dependency=["point", "global"],
                                                  composition=["structured"], domains=["structured"], **LV2),
    "longbench-v2:structured-data-knowledge-graph-reasoning": dict(archetype=CODE, dependency=["multi"],
                                                                   composition=["structured"], domains=["structured"], **LV2),
    "ruler:niah-single-1": dict(composition=["synthetic_filler"]),
    "ruler:niah-single-2": dict(composition=["prose"]),
    "ruler:niah-single-3": dict(composition=["prose"]),
    "ruler:niah-multikey-2": dict(composition=["structured", "synthetic_filler"]),
    "ruler:niah-multikey-3": dict(composition=["structured", "synthetic_filler"]),
    "ruler:niah-multivalue": dict(archetype=MH, dependency=["multi"]),
    "ruler:niah-multiquery": dict(archetype=MH, dependency=["multi"]),
    "ruler:qa-2": dict(archetype=MH, dependency=["multi", "point"]),
    "infinitebench:en-sum": dict(archetype=AG, dependency=["global"], fidelity="semantic", composition=["prose"],
                                 domains=["summarization"], growth="mixed_long", flags=["fuzzy_metric"]),
    "infinitebench:en-qa": dict(archetype=MH, dependency=["multi", "global"], fidelity="semantic", composition=["prose"],
                                domains=["reading"]),
    "infinitebench:en-mc": dict(archetype=MH, dependency=["multi", "global"], fidelity="exact_low_entropy",
                                composition=["prose"], domains=["reading"], flags=["short_answer"]),
    "infinitebench:en-dia": dict(archetype=SR, dependency=["point", "global"], fidelity="exact_low_entropy",
                                 composition=["dialogue"], domains=["reading"]),
    "infinitebench:zh-qa": dict(archetype=MH, dependency=["multi", "global"], fidelity="semantic", composition=["prose"],
                                domains=["reading", "multilingual"]),
    "infinitebench:code-debug": dict(archetype=CODE, dependency=["point"], fidelity="exact_low_entropy",
                                     composition=["code"], domains=["code"], flags=["short_answer"]),
    "infinitebench:code-run": dict(archetype=CODE, dependency=["multi"], fidelity="exact_high_entropy",
                                   composition=["code"], domains=["code"]),
    "infinitebench:math-calc": dict(archetype=AG, dependency=["global", "self"], fidelity="exact_high_entropy",
                                    composition=["structured"], domains=["math"], growth="mixed_long"),
    "infinitebench:math-find": dict(archetype=AG, dependency=["global"], fidelity="exact_high_entropy",
                                    composition=["structured"], domains=["math"], flags=["short_answer"]),
    "infinitebench:retrieve-passkey": dict(archetype=SR, dependency=["point"], fidelity="exact_high_entropy",
                                           composition=["synthetic_filler"], position="swept"),
    "infinitebench:retrieve-number": dict(archetype=SR, dependency=["point"], fidelity="exact_high_entropy",
                                          composition=["synthetic_filler"], position="swept"),
    "infinitebench:retrieve-kv": dict(archetype=SR, dependency=["point"], fidelity="exact_high_entropy",
                                      composition=["structured"], domains=["structured"], position="swept"),
    "helmet:json-kv": dict(composition=["structured"], domains=["structured"]),
    "helmet:ruler-mv": dict(archetype=MH, dependency=["multi"]),
    "helmet:hotpotqa": dict(archetype=MH, dependency=["multi", "point"]),
    "helmet:alce-qampari": dict(dependency=["multi", "global", "persistent"]),
    "helmet:infinitebench-qa": dict(fidelity="semantic"),
    "helmet:infinitebench-mc": dict(fidelity="exact_low_entropy", flags=["short_answer"]),
    "helmet:infinitebench-sum": dict(growth="mixed_long"),
    "helmet:multi-lexsum": dict(composition=["multidoc"]),
    "leval:toefl": dict(domains=["reading", "understanding"]),
    "leval:gsm": dict(archetype=ICL, dependency=["global", "self"], fidelity="exact_high_entropy",
                      composition=["demonstrations"], domains=["math", "icl"], flags=[], alt=None),
    "leval:quality": dict(dependency=["point", "global"]),
    "leval:topicret": dict(composition=["dialogue"], position="early", fidelity="semantic"),
    "leval:sfiction": dict(flags=["short_answer"], notes_extra="counterfactual to world knowledge: resists parametric leakage"),
    "leval:codeu": dict(archetype=CODE, dependency=["multi"], fidelity="exact_high_entropy", composition=["code"],
                        domains=["code"]),
    "leval:multidoc2dial": dict(composition=["multidoc", "dialogue"]),
    "leval:cuad": dict(fidelity="exact_high_entropy", domains=["reading"]),
    "leval:nq": dict(flags=["parametric_leakage"]),
    "leval:multi-news": dict(archetype=AG, dependency=["global"], composition=["multidoc"], domains=["summarization"]),
    "leval:govreport": dict(archetype=AG, dependency=["global"], domains=["summarization"]),
    "leval:bigpatent": dict(archetype=AG, dependency=["global"], domains=["summarization"]),
    "leval:summscreen": dict(archetype=AG, dependency=["global"], composition=["dialogue"], domains=["summarization"]),
    "leval:openreview": dict(archetype=AG, dependency=["global"], domains=["summarization", "writing"]),
    "leval:qmsum": dict(archetype=AG, dependency=["global", "point"], composition=["dialogue"], domains=["summarization"]),
    "leval:space": dict(archetype=AG, dependency=["global"], composition=["multidoc"], domains=["summarization"]),
    "zeroscrolls:summscreenfd": dict(composition=["dialogue"]),
    "zeroscrolls:qmsum": dict(composition=["dialogue"]),
    "zeroscrolls:musique": dict(archetype=MH, dependency=["multi"], composition=["multidoc"],
                                flags=["parametric_leakage"]),
    "zeroscrolls:spacedigest": dict(composition=["multidoc"]),
    "zeroscrolls:booksumsort": dict(fidelity="exact_high_entropy"),
    "loogle:longdep-summarization": dict(archetype=AG, dependency=["global"], domains=["summarization"]),
    "loogle:shortdep-cloze": dict(fidelity="exact_high_entropy", composition=["dialogue"]),
    "loogle:longdep-qa-timeline-reorder": dict(archetype=AG, dependency=["global"]),
    "loogle:longdep-qa-computation": dict(archetype=AG, dependency=["multi", "global"], fidelity="exact_high_entropy",
                                          domains=["reading", "math"]),
    "michelangelo:mrcr": dict(archetype=SR, dependency=["point"], fidelity="exact_high_entropy", composition=["dialogue"],
                              domains=["dialogue_memory"], growth="mixed_long", secondary=[MT]),
    "michelangelo:idk": dict(archetype=SR, dependency=["global", "point"], composition=["prose"], domains=["reading"],
                             flags=["short_answer"]),
    "michelangelo:latent-list": dict(composition=["code"], domains=["code", "reasoning"]),
    "lv-eval:cmrc-mixup": dict(domains=["reading", "multilingual"]),
    "lv-eval:multifieldqa-zh-mixup": dict(domains=["reading", "multilingual"]),
    "lv-eval:factrecall-en": dict(fidelity="exact_high_entropy"),
    "lv-eval:factrecall-zh": dict(fidelity="exact_high_entropy", domains=["reading", "multilingual"]),
    "lv-eval:dureader-mixup": dict(domains=["reading", "multilingual"]),
    "lv-eval:lic-mixup": dict(composition=["prose"], domains=["reading", "multilingual"]),
    "loong:spotlight-locating": dict(archetype=SR, dependency=["point"]),
    "loong:clustering": dict(archetype=AG, dependency=["global"], fidelity="exact_high_entropy"),
    "needlebench:multi-needle-retrieval": dict(archetype=MH, dependency=["multi"]),
    "needlebench:multi-needle-reasoning": dict(archetype=MH, dependency=["multi"]),
    "needlebench:ancestral-trace-challenge": dict(archetype=MH, dependency=["multi", "global"], composition=["structured"],
                                                  position="uniform"),
    "longproc:html-to-tsv": dict(archetype=CODE, dependency=["global", "point"], composition=["structured"],
                                 domains=["structured"], growth="mixed_long", query="known_last"),
    "longproc:pseudocode-to-code": dict(archetype=SG, dependency=["point", "self"], composition=["code"], domains=["code"],
                                        growth="compact"),
    "longproc:path-traversal": dict(dependency=["point", "self", "multi"], composition=["structured"]),
    "longproc:theory-of-mind-tracking": dict(dependency=["self", "multi"], composition=["prose"]),
    "longproc:countdown": dict(dependency=["self"], composition=["instructions"], domains=["math", "reasoning"]),
    "longproc:travel-planning": dict(dependency=["self", "persistent"], composition=["structured", "instructions"]),
    "niah:niah-v2-tasks": dict(dependency=["point", "multi"]),
    "scbench:retr-kv": dict(dependency=["point"], composition=["structured"]),
    "scbench:retr-prefix-suffix": dict(dependency=["point"], composition=["structured"]),
    "scbench:retr-multihop": dict(dependency=["multi"], composition=["synthetic_filler"]),
    "scbench:code-repoqa": dict(dependency=["point"], composition=["code"], domains=["code", "dialogue_memory"]),
    "scbench:en-qa": dict(dependency=["point", "multi"], fidelity="semantic", composition=["prose"]),
    "scbench:zh-qa": dict(dependency=["point", "multi"], fidelity="semantic", composition=["prose"],
                          domains=["dialogue_memory", "multilingual"]),
    "scbench:en-multichoice": dict(dependency=["point", "multi"], fidelity="exact_low_entropy", composition=["prose"]),
    "scbench:math-find": dict(dependency=["global"], composition=["structured"]),
    "scbench:icl-manyshot": dict(dependency=["global"], fidelity="exact_low_entropy", composition=["demonstrations"],
                                 domains=["icl", "dialogue_memory"]),
    "scbench:en-sum": dict(dependency=["global"], fidelity="semantic", composition=["prose"]),
    "scbench:mix-sum-niah": dict(dependency=["global", "point"], fidelity="semantic", composition=["prose"]),
    "scbench:mix-repoqa-kv": dict(dependency=["point"], composition=["code", "structured"]),
    "longmemeval:single-session-preference": dict(notes_extra="implicit (non-literal) evidence"),
    "longmemeval:multi-session": dict(dependency=["multi", "global"]),
    "longmemeval:knowledge-update": dict(dependency=["multi"], notes_extra="must prefer the updated over the stale value"),
    "longmemeval:temporal-reasoning": dict(dependency=["multi"]),
    "longmemeval:abstention": dict(dependency=["global"], notes_extra="absence of evidence must be detected"),
    "locomo:qa-multi-hop": dict(dependency=["multi"]),
    "locomo:qa-temporal": dict(dependency=["point", "multi"]),
    "locomo:qa-open-domain-commonsense": dict(dependency=["multi", "parametric"]),
    "locomo:qa-adversarial": dict(dependency=["global"], notes_extra="unanswerable / speaker-swapped questions"),
    "locomo:event-summarization": dict(archetype=AG, dependency=["global"], domains=["summarization", "dialogue_memory"],
                                       alt=None),
    "locomo:multimodal-dialogue-generation": dict(dependency=["persistent", "point"], fidelity="open", alt=None),
    "longbench-write:longwrite-ruler": dict(dependency=["self", "persistent"]),
    "hellobench:summarization": dict(archetype=AG, dependency=["global", "self"], growth="mixed_long",
                                     composition=["prose"], domains=["summarization", "writing"]),
    "hellobench:text-completion": dict(growth="mixed_long", composition=["prose"]),
    "hellobench:chat": dict(domains=["instruction", "writing"]),
    "hellobench:open-ended-qa": dict(domains=["instruction", "writing"]),
}

BABILONG = {  # qaN -> (archetype, dependency)
    1: (SR, ["point"]), 2: (MH, ["multi"]), 3: (MH, ["multi"]), 4: (SR, ["point"]), 5: (SR, ["point"]),
    6: (SR, ["point"]), 7: (MH, ["multi"]), 8: (MH, ["multi"]), 9: (SR, ["point"]), 10: (SR, ["point"]),
    11: (MH, ["multi"]), 12: (SR, ["point"]), 13: (MH, ["multi"]), 14: (MH, ["multi"]), 15: (MH, ["multi"]),
    16: (MH, ["multi"]), 17: (MH, ["multi"]), 18: (MH, ["multi"]), 19: (MH, ["multi"]), 20: (SR, ["point"]),
}

# ----------------------------------------------------------------------------------------------------
# Tasks outside the long-context table: short-context, reasoning, chat, agentic, streaming, KV-eval sets.
# Lengths marked "measured" come from results/length_profiles.csv / agent_summary.csv (Llama-3 tokens).
def extra(id, suite, task, s, input_len, output_len, metric, source, confidence, notes=""):
    return dict(id=id, suite=suite, task=task, input_len=input_len, output_len=output_len, metric=metric,
                sources=[source] if isinstance(source, str) else source, confidence=confidence, notes=notes, **s)


MC_SIG = sig(CMP, ["parametric"], "exact_low_entropy", ["instructions"], ["understanding"], G="compact",
             X=["parametric_leakage", "short_answer"])
LMEVAL = "https://github.com/EleutherAI/lm-evaluation-harness"
EXTRA_TASKS = [
    extra("mmlu", "MMLU", "MMLU (5-shot)", {**MC_SIG, "domains": ["knowledge"], "composition": ["demonstrations"]},
          "<1K tokens (5-shot)", "1 token (option)", "accuracy", LMEVAL, "[likely]"),
    extra("mmlu-pro", "MMLU-Pro", "MMLU-Pro (5-shot CoT)",
          sig(SG, ["parametric", "self"], "exact_low_entropy", ["demonstrations"], ["knowledge", "reasoning"], G="compact",
              Q="evolving"), "~1-2K tokens (5-shot CoT)", "CoT, hundreds of tokens", "accuracy",
          "https://github.com/TIGER-AI-Lab/MMLU-Pro", "[likely]"),
    *[extra(f"lm-eval:{n.lower()}", "lm-eval-harness (short MC)", n, {**MC_SIG, "domains": d}, "tens-hundreds of tokens",
            "log-likelihood over options", "accuracy", LMEVAL, "[likely]",
            "used by H2O / Scissorhands / InfiniGen-era eviction papers")
      for n, d in [("ARC-Challenge", ["knowledge", "understanding"]), ("HellaSwag", ["understanding"]),
                   ("Winogrande", ["understanding"]), ("PIQA", ["understanding"]), ("COPA", ["understanding"]),
                   ("RTE", ["understanding"]), ("OpenBookQA", ["knowledge"]), ("MathQA", ["math"]),
                   ("BoolQ", ["reading"]), ("CommonsenseQA", ["understanding"]), ("TruthfulQA-MC", ["knowledge", "safety"])]],
    extra("gsm8k", "GSM8K", "GSM8K (8-shot CoT)",
          sig(SG, ["self", "point"], "exact_high_entropy", ["demonstrations"], ["math"], G="compact", Q="evolving"),
          "measured: p50 712 tokens (lm-eval 8-shot CoT); 59 zero-shot", "measured: reference CoT p50 76 tokens",
          "exact match", "https://github.com/openai/grade-school-math", "[verified]",
          "KVFundaBench: arithmetic is the most compression-sensitive short task"),
    extra("math-500", "MATH-500", "MATH-500 (non-thinking CoT)",
          sig(SG, ["self", "point"], "exact_high_entropy", ["instructions"], ["math"], G="compact", Q="evolving"),
          "measured: p50 49 tokens", "measured: reference solution p50 158 tokens", "exact match",
          "https://github.com/openai/prm800k", "[verified]"),
    extra("math-500-thinking", "MATH-500", "MATH-500 (thinking model)",
          sig(LR, ["self", "point"], "exact_high_entropy", ["generated_trace"], ["math"], G="decode_dominated",
              Q="evolving", X=["output_length_unreported"]),
          "measured: p50 49 tokens", "thousands of tokens of reasoning (R-KV, kvpress decoding eval)", "exact match",
          "https://github.com/Zefan-Cai/R-KV", "[likely]"),
    extra("aime", "AIME", "AIME 2024 / 2025 (thinking model)",
          sig(LR, ["self", "point", "persistent"], "exact_high_entropy", ["generated_trace"], ["math"],
              G="decode_dominated", Q="evolving", X=["output_length_unreported"]),
          "measured: p50 86 tokens", "measured: human reference 1.3K tokens; thinking models emit ~10^4 tokens",
          "exact match (integer)", "https://github.com/QwenLM/Qwen2.5-Math", "[likely]",
          "30 problems per year: high variance, report several seeds"),
    extra("gpqa-diamond", "GPQA", "GPQA-Diamond (thinking model)",
          sig(LR, ["self", "parametric"], "exact_low_entropy", ["generated_trace"], ["reasoning", "knowledge"],
              G="decode_dominated", Q="evolving", X=["parametric_leakage"]),
          "short (question + 4 options)", "long reasoning trace", "accuracy", "https://github.com/idavidrein/gpqa",
          "[likely]"),
    extra("bbh", "BIG-Bench Hard", "BBH (27 tasks, 3-shot CoT)",
          sig(SG, ["self", "point"], "exact_low_entropy", ["demonstrations"], ["reasoning"], G="compact", Q="evolving"),
          "measured: p50 853 tokens", "measured: exemplar rationales p50 ~119 tokens", "exact match",
          "https://github.com/suzgunmirac/BIG-Bench-Hard", "[verified]"),
    extra("musr", "MuSR", "MuSR (multistep soft reasoning)",
          sig(SG, ["multi", "self"], "exact_low_entropy", ["prose"], ["reasoning"], G="compact", Q="evolving"),
          "~1K-token narratives", "CoT + option", "accuracy", "https://github.com/Zayne-sprague/MuSR", "[likely]"),
    extra("humaneval", "HumanEval", "HumanEval",
          sig(SG, ["self", "point", "persistent"], "exact_high_entropy", ["code"], ["code"], G="compact", Q="evolving"),
          "measured: p50 117 tokens", "measured: canonical solution p50 46 tokens", "pass@k",
          "https://github.com/openai/human-eval", "[verified]"),
    extra("mbpp", "MBPP", "MBPP",
          sig(SG, ["self", "point"], "exact_high_entropy", ["code"], ["code"], G="compact", Q="evolving"),
          "measured: p50 94 tokens", "measured: reference code p50 46 tokens", "pass@k",
          "https://github.com/google-research/google-research/tree/master/mbpp", "[verified]"),
    extra("livecodebench-thinking", "LiveCodeBench", "LiveCodeBench (thinking model)",
          sig(LR, ["self", "point", "persistent"], "exact_high_entropy", ["generated_trace"], ["code"],
              G="decode_dominated", Q="evolving"),
          "problem statement, hundreds of tokens", "long reasoning + program", "pass@1",
          "https://github.com/LiveCodeBench/LiveCodeBench", "[likely]"),
    extra("hle", "Humanity's Last Exam", "HLE (thinking model)",
          sig(LR, ["self", "parametric"], "exact_low_entropy", ["generated_trace"], ["knowledge", "reasoning"],
              G="decode_dominated", Q="evolving", X=["parametric_leakage"]),
          "short", "long reasoning trace", "accuracy", "https://github.com/centerforaisafety/hle", "[likely]"),
    extra("ifeval", "IFEval", "IFEval",
          sig(SG, ["persistent", "self"], "open", ["instructions"], ["instruction"], G="compact", Q="evolving"),
          "measured: p50 41 tokens; 1.5 constraints per prompt", "hundreds of tokens",
          "instruction-level accuracy", "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
          "[verified]", "Pitfalls (ACL'26): multi-instruction prompts lose individual constraints under compression"),
    extra("alpacaeval", "AlpacaEval 2", "AlpacaEval 2 (805 instructions)",
          sig(SG, ["self", "persistent"], "open", ["instructions"], ["instruction", "writing"], G="compact",
              Q="evolving", X=["fuzzy_metric"]),
          "measured: p50 21 tokens", "measured: p50 278-461 tokens across 6 models", "LC win rate (judge)",
          "https://github.com/tatsu-lab/alpaca_eval", "[verified]"),
    extra("arenahard-v0.1", "Arena-Hard", "Arena-Hard v0.1",
          sig(SG, ["self", "persistent"], "open", ["instructions"], ["instruction", "code", "reasoning"], G="compact",
              Q="evolving", X=["fuzzy_metric"]),
          "measured: p50 30 tokens", "measured: gpt-4-0314 p50 391 tokens", "win rate (judge)",
          "https://github.com/lm-sys/arena-hard-auto", "[verified]"),
    extra("arenahard-v2-hard-thinking", "Arena-Hard", "Arena-Hard v2 hard prompts (thinking model)",
          sig(LR, ["self", "point", "persistent"], "open", ["generated_trace"], ["instruction", "code", "reasoning"],
              G="decode_dominated", Q="evolving", X=["fuzzy_metric"]),
          "measured: p50 94 tokens", "measured: R1 p50 3.6K, QwQ-32B p50 5.1K tokens (thought + answer)",
          "win rate (judge)", "https://github.com/lm-sys/arena-hard-auto", "[verified]",
          "measured MR@512 = 0.42-0.46: a 512-token window misses almost half of the verbatim dependencies"),
    extra("arenahard-v2-creative-thinking", "Arena-Hard", "Arena-Hard v2 creative writing (thinking model)",
          sig(LG, ["self", "persistent"], "open", ["generated_trace"], ["writing"], G="decode_dominated", Q="evolving",
              X=["fuzzy_metric"]),
          "measured: p50 60 tokens", "measured: R1 p50 1.3K, QwQ-32B p50 1.7K tokens", "win rate (judge)",
          "https://github.com/lm-sys/arena-hard-auto", "[verified]"),
    extra("mtbench", "MT-Bench", "MT-Bench (80 questions x 2 turns)",
          sig(MT, ["persistent", "point", "self"], "open", ["dialogue"], ["instruction", "dialogue_memory"],
              G="accumulating", Q="deferred", R=["session"], X=["fuzzy_metric"]),
          "measured: turn-2 context p50 283 tokens", "measured: GPT-4 reference p50 224 tokens", "judge score (1-10)",
          "https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge", "[verified]",
          "turn 2 depends on turn 1, but the whole session fits any budget: weak KV stress"),
    extra("mtbench-101", "MT-Bench-101", "MT-Bench-101 (13 tasks, 1388 dialogues)",
          sig(MT, ["point", "persistent"], "open", ["dialogue"], ["dialogue_memory"], G="accumulating", Q="deferred",
              R=["session"], X=["fuzzy_metric"]),
          "measured: final-turn context p50 176 tokens (3.0 turns avg)", "measured: gold p50 62 tokens", "judge score",
          "https://github.com/mtbench101/mt-bench-101", "[verified]", "multi-turn structure, but short: weak KV stress"),
    extra("xsum", "HELM summarization", "XSum", sig(SG, ["global"], "semantic", ["prose"], ["summarization"],
                                                     G="compact", X=["fuzzy_metric"]),
          "~400-500 tokens", "1 sentence", "ROUGE", "https://github.com/stanford-crfm/helm", "[likely]", "used by H2O"),
    extra("cnn-dm", "HELM summarization", "CNN/DailyMail", sig(SG, ["global"], "semantic", ["prose"], ["summarization"],
                                                                G="compact", X=["fuzzy_metric"]),
          "~700-900 tokens", "3-4 sentences", "ROUGE", "https://github.com/stanford-crfm/helm", "[likely]", "used by H2O"),
    extra("crosscodeeval", "CrossCodeEval", "CrossCodeEval (python/java/ts/c#, BM25 cross-file context)",
          sig(CODE, ["point", "local"], "exact_high_entropy", ["code", "multidoc"], ["code"], P="uniform"),
          "measured: p50 606-988 tokens", "measured: one line, p50 9-14 tokens", "exact/edit similarity",
          "https://github.com/amazon-science/cceval", "[verified]",
          "measured: 97% of copied identifiers come from the prompt, median 341 tokens back"),
    extra("swe-bench-verified", "SWE-bench", "SWE-bench Verified (500 issues; mini-SWE-agent)",
          sig(AGENT, ["persistent", "point", "self"], "exact_high_entropy", ["tool_observations", "code", "instructions"],
              ["agentic", "code"], G="accumulating", Q="evolving", R=["agentic_loop", "shared_prefix"]),
          "measured: peak context p50 13-17K (p90 26-42K, max 157K) tokens; 12-47 steps",
          "measured: 16-35% of the peak context is model output", "resolved rate",
          "https://github.com/SWE-bench/experiments", "[verified]",
          "observations are 53-71% of the context and highly compressible (gzip 4.5)"),
    extra("swe-bench-lite", "SWE-bench", "SWE-bench Lite",
          sig(AGENT, ["persistent", "point", "self"], "exact_high_entropy", ["tool_observations", "code", "instructions"],
              ["agentic", "code"], G="accumulating", Q="evolving", R=["agentic_loop", "shared_prefix"]),
          "as SWE-bench Verified", "as SWE-bench Verified", "resolved rate", "https://github.com/SWE-bench/SWE-bench",
          "[likely]"),
    extra("tau-bench", "tau-bench", "tau-bench airline / retail",
          sig(AGENT, ["persistent", "point"], "exact_high_entropy", ["instructions", "tool_observations", "dialogue"],
              ["agentic", "dialogue_memory"], G="accumulating", Q="evolving", P="early",
              R=["agentic_loop", "shared_prefix", "session"]),
          "measured: peak p50 6.1-7.9K tokens; policy + 14-16 tool schemas = 3.7-4.1K persistent tokens",
          "measured: 13-18% of the final context is model output", "pass^k (task success)",
          "https://github.com/sierra-research/tau-bench", "[verified]",
          "persistent prefix is 52-61% of the median peak context: a direct probe of instruction retention"),
    extra("tau2-bench", "tau-bench", "tau2-bench (dual control)",
          sig(AGENT, ["persistent", "point"], "exact_high_entropy", ["instructions", "tool_observations", "dialogue"],
              ["agentic"], G="accumulating", Q="evolving", P="early", R=["agentic_loop", "shared_prefix", "session"]),
          "similar to tau-bench", "similar to tau-bench", "pass^k", "https://github.com/sierra-research/tau2-bench",
          "[likely]"),
    extra("bfcl-single", "BFCL", "BFCL (single-turn function calling)",
          sig(SG, ["persistent", "point"], "exact_high_entropy", ["instructions"], ["agentic"], G="compact",
              Q="known_last", P="early"),
          "function schemas + query, ~0.5-2K tokens", "one call (tens of tokens)", "AST / execution accuracy",
          "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", "[likely]"),
    extra("bfcl-multi", "BFCL", "BFCL v3 multi-turn",
          sig(AGENT, ["persistent", "point", "multi"], "exact_high_entropy", ["instructions", "tool_observations"],
              ["agentic"], G="accumulating", Q="evolving", R=["agentic_loop", "shared_prefix"]),
          "grows over turns", "tool calls", "state / response checks",
          "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", "[likely]"),
    *[extra(i, s, t, sig(AGENT, ["persistent", "point", "self"], "exact_high_entropy",
                         ["tool_observations", "instructions"], ["agentic"], G="accumulating", Q="evolving",
                         R=["agentic_loop", "shared_prefix"]),
            "grows with steps (observations dominate)", "actions / tool calls", "task success", u, "[likely]", n)
      for i, s, t, u, n in [
          ("terminal-bench", "Terminal-Bench", "Terminal-Bench", "https://github.com/laude-institute/terminal-bench", ""),
          ("webarena", "WebArena", "WebArena", "https://github.com/web-arena-x/webarena",
           "observations are accessibility trees / HTML (structured, large)"),
          ("osworld", "OSWorld", "OSWorld", "https://github.com/xlang-ai/OSWorld", "multimodal observations"),
          ("gaia", "GAIA", "GAIA", "https://huggingface.co/gaia-benchmark", "web search + file tools"),
          ("agentbench", "AgentBench", "AgentBench (8 environments)", "https://github.com/THUDM/AgentBench", "")]],
    extra("pg19-ppl", "Language modelling", "PG19 perplexity (streaming)",
          sig(STR, ["local"], "open", ["prose"], ["language_modeling"], G="decode_dominated", Q="evolving", P="recent"),
          "unbounded stream (books, up to millions of tokens)", "next-token prediction", "perplexity",
          "https://github.com/google-deepmind/pg19", "[verified]",
          "StreamingLLM: 4 sink tokens + recent window gives PPL 5.40 vs 5158 without sinks"),
    extra("wikitext-ppl", "Language modelling", "WikiText-2 / C4 / PTB perplexity",
          sig(STR, ["local"], "open", ["prose"], ["language_modeling"], G="decode_dominated", Q="evolving", P="recent"),
          "fixed windows (1-4K)", "next-token prediction", "perplexity", LMEVAL, "[likely]"),
    extra("streameval", "StreamingLLM", "StreamEval / streaming ARC QA",
          sig(STR, ["local", "point"], "exact_low_entropy", ["prose"], ["language_modeling", "reading"],
              G="accumulating", Q="evolving", P="recent", R=["session"]),
          "continuous stream; query every 10 lines", "short answers", "accuracy",
          "https://github.com/mit-han-lab/streaming-llm", "[verified]", "answers depend on recent context by design"),
    extra("sirllm-dialogue", "SirLLM", "DailyDialog / Grocery shopping / Rock-paper-scissors (streaming dialogue)",
          sig(MT, ["point", "local"], "semantic", ["dialogue"], ["dialogue_memory"], G="accumulating", Q="deferred",
              R=["session"]),
          "long streaming dialogues", "short turns", "memory accuracy", "https://github.com/Zoeyyao27/SirLLM", "[likely]",
          "StreamingLLM forgets early-turn facts"),
    extra("raccoonbench", "Pitfalls of KV Cache Compression", "System-prompt leakage (RaccoonBench + defenses)",
          sig(SG, ["persistent"], "open", ["instructions"], ["safety"], G="compact", Q="evolving", P="early"),
          "system prompt + defense + attack", "response", "leakage rate",
          "https://github.com/alexluchen/pitfalls-of-kv-cache-compression", "[verified]"),
    extra("jailbreakv", "KVFundaBench", "JailBreakV (safety)",
          sig(CMP, ["persistent", "parametric"], "open", ["instructions"], ["safety"], G="compact", Q="known_last"),
          "short", "refusal / compliance", "attack success rate", "https://arxiv.org/abs/2502.01941", "[likely]"),
    *[extra(f"reasoning:{n.lower()}", "Hold Onto That Thought", n,
            sig(SG, ["self", "point"], "exact_low_entropy", ["prose"], d, G="compact", Q="evolving"),
            "short", "CoT (budget-limited to 2K)", "accuracy", "https://github.com/minghui-liu/kvpress", "[verified]")
      for n, d in [("FOLIO", ["reasoning"]), ("DROP", ["reading", "math"]), ("StrategyQA", ["reasoning", "knowledge"]),
                   ("ReClor", ["reasoning", "reading"]), ("LogiQA", ["reasoning"])]],
]

# suites used to compute coverage (task id lists or id prefixes ending with ':')
SUITES = {
    "LongBench v1 (21)": ["longbench:"],
    "LongBench v1 EN-16 (SnapKV/PyramidKV protocol)": [
        "longbench:narrativeqa", "longbench:qasper", "longbench:multifieldqa-en", "longbench:hotpotqa",
        "longbench:2wikimultihopqa", "longbench:musique", "longbench:govreport", "longbench:qmsum",
        "longbench:multinews", "longbench:trec", "longbench:triviaqa", "longbench:samsum", "longbench:passagecount",
        "longbench:passageretrieval-en", "longbench:lcc", "longbench:repobench-p"],
    "LongBench v2": ["longbench-v2:"], "RULER": ["ruler:"], "InfiniteBench": ["infinitebench:"], "HELMET": ["helmet:"],
    "L-Eval": ["leval:"], "ZeroSCROLLS": ["zeroscrolls:"], "LooGLE": ["loogle:"], "BABILong": ["babilong:"],
    "LV-Eval": ["lv-eval:"], "Loong": ["loong:"], "NeedleBench": ["needlebench:"], "LongICLBench": ["longiclbench:"],
    "Michelangelo": ["michelangelo:"], "SCBench": ["scbench:"], "LongMemEval": ["longmemeval:"], "LoCoMo": ["locomo:"],
    "LongProc": ["longproc:"], "LongBench-Write": ["longbench-write:"], "LongGenBench (Wu)": ["longgenbench-wu:"],
    "HelloBench": ["hellobench:"],
    "NIAH + passkey": ["niah:", "passkey:"],
    "Open LLM Leaderboard v2": ["ifeval", "bbh", "math-500", "gpqa-diamond", "musr", "mmlu-pro"],
    "H2O (NeurIPS'23) evaluation": ["lm-eval:copa", "lm-eval:mathqa", "lm-eval:openbookqa", "lm-eval:piqa",
                                    "lm-eval:rte", "lm-eval:winogrande", "xsum", "cnn-dm"],
    "StreamingLLM (ICLR'24) evaluation": ["pg19-ppl", "streameval", "longbench:narrativeqa", "longbench:qasper",
                                          "longbench:hotpotqa", "longbench:2wikimultihopqa", "longbench:govreport",
                                          "longbench:multinews"],
    "SnapKV-era standard (LongBench EN-16 + NIAH)": ["@LongBench v1 EN-16 (SnapKV/PyramidKV protocol)",
                                                     "niah:original-single-needle-niah"],
    "KVFundaBench": ["mmlu", "lm-eval:commonsenseqa", "gsm8k", "humaneval", "jailbreakv", "longgenbench-liu:gsm8k"],
    "kvpress default evaluation": ["ruler:", "longbench:", "longbench-v2:", "loogle:", "zeroscrolls:", "infinitebench:",
                                   "niah:original-single-needle-niah", "aime", "math-500-thinking"],
    "Hold Onto That Thought": ["gsm8k", "math-500", "reasoning:", "lm-eval:commonsenseqa", "lm-eval:openbookqa", "aime"],
    "Agentic trio (SWE-bench, tau-bench, BFCL)": ["swe-bench-verified", "tau-bench", "bfcl-multi"],
    # --- recommended by this study (docs/recommendations.md) ---
    "KV-eviction core suite (this study)": [
        "mmlu", "gsm8k", "ifeval", "humaneval", "aime", "math-500-thinking", "longbench-write:longbench-write",
        "ruler:niah-single-3", "helmet:json-kv", "longbench:narrativeqa", "ruler:vt", "longbench:hotpotqa",
        "ruler:cwe", "longbench:govreport", "helmet:banking77", "longbench:repobench-p", "helmet:alce-asqa",
        "scbench:retr-kv", "scbench:en-qa", "scbench:math-find", "locomo:qa-multi-hop", "swe-bench-verified",
        "tau-bench", "pg19-ppl", "longproc:html-to-tsv"],
    "KV-eviction lite suite (this study)": [
        "mmlu", "gsm8k", "ifeval", "aime", "longbench-write:longbench-write", "ruler:niah-single-3", "ruler:vt", "ruler:cwe",
        "longbench:hotpotqa", "helmet:banking77", "longbench:repobench-p", "scbench:retr-kv", "scbench:en-qa",
        "tau-bench"],
}

# ----------------------------------------------------------------------------------------------------


def build():
    tasks = []
    for r in map(json.loads, open(HERE / "sources/longctx_tasks.jsonl")):
        s = SUITE_SHORT[r["suite"]]
        tid = f"{s}:{slug(r['task'])}"
        if s == "longbench-v2":
            tid = f"{s}:{slug(r['task'].replace('/', ' '))}"
        base = CATEGORY_RULES.get((s, r["suite_category"])) or CATEGORY_RULES.get((s, None))
        if base is None:
            raise KeyError(f"no rule for {s} / {r['suite_category']}")
        f = json.loads(json.dumps(base))
        if s == "babilong":
            n = int(re.match(r"qa(\d+)", r["task"]).group(1))
            f["archetype"], f["dependency"] = BABILONG[n]
        ov = TASK_OVERRIDES.get(tid, {})
        f.update({k: v for k, v in ov.items() if k != "notes_extra"})
        if f.get("alt") is None:
            f.pop("alt", None)
        qp = r["query_position"]
        if f["query"] in ("known_last", "known_first") and s not in ("scbench",):
            f["query"] = "known_first" if qp.startswith("before") else "known_last"
        notes = r.get("notes", "")
        if "notes_extra" in ov:
            notes = (ov["notes_extra"] + ". " + notes).strip()
        tasks.append(dict(id=tid, suite=r["suite"], task=r["task"], suite_category=r["suite_category"],
                          input_len=r["input_len"], output_len=r["output_len"], n_samples=r.get("n_samples"),
                          metric=r["metric"], query_position=qp, evidence_structure=r["evidence_structure"],
                          synthetic_or_natural=r["synthetic_or_natural"], language=r.get("language"),
                          sources=r["sources"], confidence=r["confidence"], notes=notes, **f))
    tasks += EXTRA_TASKS
    ids = [t["id"] for t in tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate ids: {dup}"
    return tasks


def validate(tasks):
    fac = {k: {v["id"] for v in f["values"]} for k, f in TAX["facets"].items()}
    arch = {a["id"] for a in TAX["archetypes"]}
    dom = {d["id"] for d in TAX["domains"]}
    flags = {f["id"] for f in TAX["validity_flags"]}
    for t in tasks:
        assert t["archetype"] in arch, (t["id"], t["archetype"])
        for s in t.get("secondary", []):
            assert s in arch, (t["id"], s)
        assert t["growth"] in fac["growth"], (t["id"], t["growth"])
        assert t["query"] in fac["query"], (t["id"], t["query"])
        assert t["fidelity"] in fac["fidelity"], (t["id"], t["fidelity"])
        assert t["position"] in fac["position"], (t["id"], t["position"])
        for k in ("dependency", "composition", "reuse"):
            for v in t[k]:
                assert v in fac[k], (t["id"], k, v)
        for d in t["domains"]:
            assert d in dom, (t["id"], d)
        for x in t["flags"]:
            assert x in flags, (t["id"], x)
        if "alt" in t:
            assert t["alt"]["query"] in fac["query"]


def resolve_suites(tasks):
    ids = [t["id"] for t in tasks]
    out = {}
    for name, items in SUITES.items():
        sel = []
        for it in items:
            if it.startswith("@"):
                sel += out[it[1:]]
            elif it.endswith(":"):
                sel += [i for i in ids if i.startswith(it)]
            else:
                assert it in ids, f"suite {name}: unknown task {it}"
                sel.append(it)
        out[name] = list(dict.fromkeys(sel))
    return out


if __name__ == "__main__":
    tasks = build()
    validate(tasks)
    suites = resolve_suites(tasks)
    doc = {"generated_by": "taxonomy/build_catalog.py", "taxonomy_version": TAX["version"],
           "n_tasks": len(tasks), "suites": suites, "tasks": tasks}
    with open(HERE / "benchmarks.yaml", "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=120)
    from collections import Counter
    print(len(tasks), "tasks;", len(suites), "suites")
    print(Counter(t["archetype"] for t in tasks))
