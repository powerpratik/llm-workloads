"""M1 - Prefill/decode length profiles of benchmark workloads (Llama-3 tokenizer).

For every workload we measure, per sample, the number of tokens that must be prefilled
(prompt, incl. few-shot template where that is the standard setting) and the number of
tokens that are decoded (reference output or real model output). Outputs:
  results/length_profiles.csv       one row per workload (percentiles, decode share, ...)
  results/length_samples.json       per-sample (in, out) pairs for plotting
Model outputs are used where available; otherwise the reference answer (flagged in
`out_source`), which UNDER-estimates what a chat/reasoning model actually decodes.
"""
import glob
import json
import re
import zipfile

import numpy as np
import yaml

from common import CHAT_MSG_OVERHEAD, RAW, RESULTS, ntok, read_jsonl, summarize, write_csv, write_json

DS = RAW / "datasets"
REPOS = RAW / "repos"
rows, samples = [], {}


def add(workload, dataset, archetype, ins, outs, out_source, notes="", extra=None, qwen_check=None):
    ins = [int(x) for x in ins]
    outs = [None if o is None else int(o) for o in outs]
    s_in, s_out = summarize(ins), summarize([o for o in outs if o is not None])
    shares = [o / (i + o) for i, o in zip(ins, outs) if o is not None and i + o > 0]
    tot = [i + o for i, o in zip(ins, outs) if o is not None]
    row = dict(workload=workload, dataset=dataset, archetype=archetype, n=len(ins),
               in_p10=s_in["p10"], in_p50=s_in["p50"], in_p90=s_in["p90"], in_mean=s_in["mean"], in_max=s_in["max"],
               out_p10=s_out["p10"], out_p50=s_out["p50"], out_p90=s_out["p90"], out_mean=s_out["mean"], out_max=s_out["max"],
               decode_share_p50=float(np.median(shares)) if shares else None,
               total_p50=float(np.median(tot)) if tot else None, total_p90=float(np.percentile(tot, 90)) if tot else None,
               out_source=out_source, notes=notes)
    if qwen_check is not None:
        row["qwen2_over_llama3_in"] = qwen_check
    if extra:
        row.update(extra)
    rows.append(row)
    samples[workload] = dict(archetype=archetype, pairs=list(zip(ins, outs)))
    print(f"{workload:38s} n={len(ins):5d} in_p50={s_in['p50']:9.0f} out_p50={s_out['p50'] if s_out['p50'] is not None else float('nan'):8.0f}")


def qwen_ratio(texts, k=200):
    texts = texts[:k]
    a, b = ntok(texts, "llama3"), ntok(texts, "qwen2")
    return float(np.sum(b) / max(1, np.sum(a)))


# ---------------------------------------------------------------- short-context reasoning / code
def gsm8k():
    data = read_jsonl(DS / "gsm8k/test.jsonl")
    cot = yaml.safe_load(open(DS / "gsm8k_cot/gsm8k-cot.yaml"))
    shots = "".join(f"Q: {s['question']}\nA: {s['target']}\n\n" for s in cot["fewshot_config"]["samples"])
    q0 = [f"Q: {d['question']}\nA:" for d in data]
    q8 = [shots + q for q in q0]
    out = [re.sub(r"<<.*?>>", "", d["answer"].split("####")[0]).strip() + f" The answer is {d['answer'].split('####')[1].strip()}." for d in data]
    add("gsm8k-0shot", "GSM8K test", "short_gen", ntok(q0), ntok(out), "reference CoT", qwen_check=qwen_ratio(q0))
    add("gsm8k-8shot-cot", "GSM8K test (lm-eval 8-shot CoT)", "short_gen", ntok(q8), ntok(out), "reference CoT")


def math500():
    data = read_jsonl(DS / "math500/test.jsonl")
    add("math500", "MATH-500", "short_gen", ntok([d["problem"] for d in data]), ntok([d["solution"] for d in data]),
        "reference solution", qwen_check=qwen_ratio([d["solution"] for d in data]))


def aime24():
    data = read_jsonl(REPOS / "qwenmath/evaluation/data/aime24/test.jsonl")
    add("aime24-reference", "AIME 2024", "short_gen", ntok([d["problem"] for d in data]), ntok([d["solution"] for d in data]),
        "reference solution", notes="human solutions; thinking models emit 10-30x more (see arenahard-v2 thinking rows)")


def bbh():
    ins, outs = [], []
    for f in sorted(glob.glob(str(REPOS / "bbh/bbh/*.json"))):
        task = f.split("/")[-1][:-5]
        cot = open(REPOS / f"bbh/cot-prompts/{task}.txt").read()
        cot = cot.split("-----", 1)[-1].strip()
        rationales = re.findall(r"A: Let's think step by step\.(.*?)(?=\n\nQ:|\Z)", cot, flags=re.S)
        est = int(np.mean(ntok(rationales))) if rationales else None
        for ex in json.load(open(f))["examples"]:
            ins.append(cot + "\n\nQ: " + ex["input"] + "\nA: Let's think step by step.")
            outs.append(est)
    add("bbh-3shot-cot", "BIG-Bench Hard (27 tasks)", "short_gen", ntok(ins), outs,
        "per-task mean length of exemplar rationales", notes="output estimated from the task's CoT exemplars")


def humaneval():
    data = read_jsonl(DS / "humaneval/HumanEval.jsonl.gz")
    add("humaneval", "HumanEval", "short_gen", ntok([d["prompt"] for d in data]),
        ntok([d["canonical_solution"] for d in data]), "canonical solution")


def mbpp():
    data = read_jsonl(DS / "mbpp/mbpp.jsonl")
    data = [d for d in data if 11 <= d["task_id"] <= 510]  # standard test split
    ins = [f"You are an expert Python programmer, and here is your task: {d['text']} Your code should pass these tests:\n\n"
           + "\n".join(d["test_list"]) for d in data]
    add("mbpp", "MBPP (test split)", "short_gen", ntok(ins), ntok([d["code"] for d in data]), "reference code")


def ifeval():
    data = read_jsonl(DS / "ifeval/input_data.jsonl")
    add("ifeval", "IFEval", "short_gen", ntok([d["prompt"] for d in data]), [None] * len(data), "n/a",
        notes="outputs not released; constraint-following over the whole response", extra=dict(
            constraints_per_prompt_mean=float(np.mean([len(d["instruction_id_list"]) for d in data]))))


# ---------------------------------------------------------------- chat / open-ended
def alpacaeval():
    instr = None
    for m in ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3.1-70B-Instruct-Turbo", "gpt-4o-2024-05-13",
              "claude-3-5-sonnet-20240620", "Qwen2-72B-Instruct", "gpt4_1106_preview"]:
        data = json.load(open(REPOS / f"alpaca_eval/results/{m}/model_outputs.json"))
        instr = instr or [d["instruction"] for d in data]
        add(f"alpacaeval:{m}", "AlpacaEval 2 (805 instr.)", "short_gen", ntok([d["instruction"] for d in data]),
            ntok([d["output"] for d in data]), f"model output ({m})")


def arenahard():
    q1 = {d["uid"]: d for d in read_jsonl(DS / "arenahard/v0.1_question.jsonl")}
    a = read_jsonl(REPOS / "arenahard/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl")
    ins = [q1[r["uid"]]["prompt"] for r in a if r["uid"] in q1]
    outs = [(r["messages"][-1]["content"]["answer"] if isinstance(r["messages"][-1]["content"], dict)
             else r["messages"][-1]["content"]) for r in a if r["uid"] in q1]
    add("arenahard-v0.1:gpt-4-0314", "Arena-Hard v0.1", "short_gen", ntok(ins), ntok(outs), "model output (gpt-4-0314)")
    q2 = {d["uid"]: d for d in read_jsonl(DS / "arenahard/v2.0_question.jsonl")}
    for m, arch in [("o3-mini-2025-01-31", "short_gen"), ("gemini-2.0-flash-001", "short_gen"),
                    ("deepseek-r1", "long_reason"), ("qwq-32b", "long_reason")]:
        a = [r for r in read_jsonl(REPOS / f"arenahard/data/arena-hard-v2.0/model_answer/{m}.jsonl") if r["uid"] in q2]
        for cat in ["hard_prompt", "creative_writing"]:
            sel = [r for r in a if q2[r["uid"]]["category"] == cat]
            ins = [q2[r["uid"]]["prompt"] for r in sel]
            c = [r["messages"][-1]["content"] for r in sel]
            outs = [(x.get("thought", "") + "\n" + x.get("answer", "")) if isinstance(x, dict) else x for x in c]
            src = "model output incl. visible thinking" if arch == "long_reason" else "model answer (hidden reasoning not counted)"
            wl_arch = arch if cat == "hard_prompt" else ("long_gen" if arch == "long_reason" else "short_gen")
            add(f"arenahard-v2:{cat}:{m}", f"Arena-Hard v2.0 {cat}", wl_arch,
                ntok(ins), ntok(outs), src)


def mtbench():
    qs = read_jsonl(DS / "mtbench/question.jsonl")
    ref = {d["question_id"]: d for d in read_jsonl(DS / "mtbench/reference_gpt4.jsonl")}
    ins1, ins2, out1, out2 = [], [], [], []
    for q in qs:
        if q["question_id"] not in ref:
            continue
        a1, a2 = ref[q["question_id"]]["choices"][0]["turns"]
        ins1.append(q["turns"][0]); out1.append(a1)
        ins2.append(q["turns"][0] + "\n" + a1 + "\n" + q["turns"][1]); out2.append(a2)
    add("mtbench-turn1(ref)", "MT-Bench (math/reasoning/coding w/ GPT-4 ref.)", "short_gen", ntok(ins1), ntok(out1), "GPT-4 reference")
    add("mtbench-turn2(ref)", "MT-Bench turn 2 (history+q2)", "shared_context_multiturn",
        [x + 3 * CHAT_MSG_OVERHEAD for x in ntok(ins2)], ntok(out2), "GPT-4 reference")


def mtbench101():
    data = read_jsonl(DS / "mtbench101/mtbench101.jsonl")
    ins, outs, turns = [], [], []
    for d in data:
        h = d["history"]
        ctx = "\n".join(f"{t['user']}\n{t['bot']}" for t in h[:-1]) + "\n" + h[-1]["user"]
        ins.append(ctx); outs.append(h[-1]["bot"]); turns.append(len(h))
    n_in = [x + (2 * len(h) - 1) * CHAT_MSG_OVERHEAD for x, h in zip(ntok(ins), [d["history"] for d in data])]
    add("mtbench101-final-turn", "MT-Bench-101 (13 tasks)", "shared_context_multiturn", n_in, ntok(outs), "gold response",
        extra=dict(turns_mean=float(np.mean(turns)), turns_max=int(np.max(turns))))


# ---------------------------------------------------------------- long output
def longbench_write():
    data = read_jsonl(DS / "longbench_write/longbench_write.jsonl")
    # required length is given in words; convert with the words->tokens ratio measured on long outputs
    ref = json.load(open(REPOS / "alpaca_eval/results/gpt-4o-2024-05-13/model_outputs.json"))
    txt = [d["output"] for d in ref]
    ratio = float(np.sum(ntok(txt)) / np.sum([len(t.split()) for t in txt]))
    add("longbench-write(required)", "LongBench-Write (120 prompts)", "long_gen", ntok([d["prompt"] for d in data]),
        [int(d["length"] * ratio) for d in data], f"required length x {ratio:.2f} tok/word",
        notes="English + Chinese prompts; requested lengths 100 to 20,000 words")


# ---------------------------------------------------------------- long-context
def leval():
    for f in sorted(glob.glob(str(REPOS / "leval/LEval-data/*/*.jsonl"))):
        task = f.split("/")[-1][:-6]
        group = f.split("/")[-2]
        docs = read_jsonl(f)
        doc_tok = ntok([d["input"] for d in docs])
        ins, outs, nq = [], [], []
        instr_tok = [ntok(d["instructions"]) for d in docs]
        out_tok = [ntok([str(o) for o in d["outputs"]]) for d in docs]
        for dt, it, ot in zip(doc_tok, instr_tok, out_tok):
            nq.append(len(it))
            for i, o in zip(it, ot):
                ins.append(dt + i); outs.append(o)
        arch = {"gov_report_summ": "aggregation", "meeting_summ": "aggregation", "news_summ": "aggregation",
                "patent_summ": "aggregation", "review_summ": "aggregation", "tv_show_summ": "aggregation",
                "codeU": "lc_code_struct", "gsm100": "many_shot_icl",
                "topic_retrieval_longchat": "sparse_retrieval", "paper_assistant": "aggregation"}.get(task, "sparse_retrieval")
        add(f"leval:{task}", f"L-Eval {group}", arch, ins, outs, "reference output",
            extra=dict(queries_per_context_mean=float(np.mean(nq)), contexts=len(docs)))


def quality():
    arts = read_jsonl(DS / "quality/QuALITY.v1.0.1.htmlstripped.dev")
    ins, nq = [], []
    art_tok = ntok([a["article"] for a in arts])
    for a, at in zip(arts, art_tok):
        qs = a["questions"]
        nq.append(len(qs))
        qt = ntok([q["question"] + "\n" + "\n".join(q["options"]) for q in qs])
        ins += [at + x for x in qt]
    add("quality-dev", "QuALITY dev (multiple choice)", "sparse_retrieval", ins, [1] * len(ins), "answer letter",
        notes="evidence often spread: questions written to require reading the whole article",
        extra=dict(queries_per_context_mean=float(np.mean(nq)), contexts=len(arts)))


def cceval():
    ins, outs = [], []
    for lang in ["python", "java", "typescript", "csharp"]:
        data = read_jsonl(DS / f"cceval/{lang}/line_completion_rg1_bm25.jsonl")
        ctx = ["\n".join(c["retrieved_chunk"] for c in d["crossfile_context"]["list"]) + "\n" + d["prompt"] for d in data]
        i, o = ntok(ctx), ntok([d["groundtruth"] for d in data])
        add(f"crosscodeeval:{lang}", f"CrossCodeEval {lang} (BM25 cross-file ctx)", "lc_code_struct", i, o, "ground-truth line")


def locomo():
    data = json.load(open(DS / "locomo/locomo10.json"))
    ins, outs, nq = [], [], []
    for d in data:
        c = d["conversation"]
        sess = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        text = "\n".join(f"[{c[s + '_date_time']}]\n" + "\n".join(f"{t['speaker']}: {t['text']}" for t in c[s]) for s in sess)
        ct = ntok(text)[0]
        qa = [q for q in d["qa"] if "answer" in q]
        nq.append(len(qa))
        for q, qt, at in zip(qa, ntok([q["question"] for q in qa]), ntok([str(q["answer"]) for q in qa])):
            ins.append(ct + qt); outs.append(at)
    add("locomo-qa", "LoCoMo (10 long conversations)", "shared_context_multiturn", ins, outs, "gold answer",
        extra=dict(queries_per_context_mean=float(np.mean(nq)), contexts=len(data)))


def niah_design():
    # Synthetic suites are parameterised: record the design envelope rather than measure.
    essays = sorted(glob.glob(str(REPOS / "niah/needlehaystack/PaulGrahamEssays/*.txt")))
    tot = sum(ntok([open(e).read() for e in essays]))
    add("niah(design)", "Needle-in-a-Haystack (PG essays)", "sparse_retrieval", [1000, 8000, 32000, 128000], [20] * 4,
        "design", notes=f"haystack corpus = {len(essays)} essays / {tot} tokens; depth swept 0-100%")


if __name__ == "__main__":
    for fn in [gsm8k, math500, aime24, bbh, humaneval, mbpp, ifeval, alpacaeval, arenahard, mtbench, mtbench101,
               longbench_write, leval, quality, cceval, locomo, niah_design]:
        fn()
    write_csv(rows, RESULTS / "length_profiles.csv")
    write_json(samples, RESULTS / "length_samples.json")
    print("wrote", RESULTS / "length_profiles.csv")
