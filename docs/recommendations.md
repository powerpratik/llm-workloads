# Recommendations: testing a KV-eviction policy for coverage

This chapter turns the taxonomy and measurements into a concrete test plan. It is written for evaluating *one
particular eviction policy*, which is the goal stated for this project. The suites below are defined in
`taxonomy/build_catalog.py` and checked with `tools/coverage.py`.

## 1. Quick start

1. **Place your policy in its family (§4).** This tells you which archetypes are its highest-risk workloads.
2. **Run the lite suite (§3) first**, with the protocol in §5.
3. **Report per archetype, not a single average.** Rankings flip across archetypes (KVDiagnosis; Yuan et al.).
4. **Before claiming generality, run the core suite (§2).** It reaches 100% weighted coverage; the best
   established suite (kvpress default) reaches 59%, and the LongBench + NIAH standard 41%.
5. **For a cross-request (prefix-cache) policy**, replay the stream set (§3.2).

## 2. Core suite (25 tasks, 100% coverage)

| archetype | task (catalog id) | what it stresses | protocol notes |
|---|---|---|---|
| compact | MMLU 5-shot (`mmlu`) | negative control; fundamental-ability regression | report under the same *ratio* budget as the other tasks |
| short_gen | GSM8K 8-shot CoT (`gsm8k`) | exact numbers copied within ~100 tokens | KVFundaBench found arithmetic the most sensitive short task |
| short_gen | IFEval (`ifeval`) | `persistent` constraints over the whole response | instruction-level accuracy; also run the system-prompt variant (Pitfalls) |
| short_gen | HumanEval (`humaneval`) | exact identifiers, code syntax | pass@1 |
| long_reason | AIME 2024/25 with a thinking model (`aime`) | decode-dominated, far-back self-reference (MR@512 ≈ 0.4–0.5) | ≥ 4 samples per problem; report accuracy, output length and cap-hit rate |
| long_reason | MATH-500 with a thinking model (`math-500-thinking`) | same, with lower variance | as above |
| long_gen | LongBench-Write (`longbench-write:longbench-write`) | 2K–20K-token outputs; coherence and length adherence | judge score + length-adherence score + repetition rate |
| sparse_retrieval | RULER NIAH-single-3, UUID in essays (`ruler:niah-single-3`) | high-entropy long span, natural haystack, swept depth | sweep depth × length; exact match |
| sparse_retrieval | HELMET JSON-KV (`helmet:json-kv`) | incompressible structured store with dense distractors | UUID exact match |
| sparse_retrieval | LongBench NarrativeQA (`longbench:narrativeqa`) | natural single-evidence QA over books | F1; add a closed-book baseline (leakage) |
| multi_hop | RULER variable tracking (`ruler:vt`) | chained bindings invisible to the question | |
| multi_hop | LongBench HotpotQA (`longbench:hotpotqa`) | natural 2-hop evidence among distractors | closed-book baseline |
| multi_hop / multi-turn | LoCoMo multi-hop (`locomo:qa-multi-hop`) | 3.1 evidence turns, 15K tokens back, in dialogue | run in **shared-context mode**: compress the conversation once, ask all questions |
| aggregation | RULER common-words extraction (`ruler:cwe`) | non-redundant counting: flat attention | |
| aggregation | LongBench GovReport (`longbench:govreport`) | redundant synthesis: expected to degrade gracefully | ROUGE + judge |
| many_shot_icl | HELMET BANKING77, random labels (`helmet:banking77`) | label mapping across many demonstrations | vary the shot count |
| lc_code_struct | LongBench RepoBench-P (`longbench:repobench-p`) | exact cross-file identifiers | edit similarity + exact match |
| lc_code_struct (mixed-long) | LongProc HTML→TSV (`longproc:html-to-tsv`) | long input *and* long exact output | row-level F1 |
| multi_hop (query first) | HELMET ALCE-ASQA (`helmet:alce-asqa`) | question *before* the context; citations | citation recall/precision |
| shared_context_multiturn | SCBench Retr.KV, En.QA, Math.Find (`scbench:*`) | `deferred` queries: follow-ups against a compressed cache | **both** multi-turn and multi-request modes; per-turn accuracy |
| agentic | SWE-bench Verified with mini-SWE-agent (`swe-bench-verified`) | accumulating context, observations, far-back identifiers | resolved rate + steps + tokens; compress between steps |
| agentic | τ-bench airline/retail (`tau-bench`) | persistent policy + tool schemas (52–61% of the context) | pass^k and **policy-violation rate** |
| streaming_lm | PG19 perplexity (`pg19-ppl`) | stability over unbounded streams | sanity check only |

Why these tasks and not others:

- **Availability.** Each task has public data and an official harness: RULER, LongBench and HELMET via their
  repositories or kvpress; SCBench via MInference; LoCoMo, τ-bench, SWE-bench and mini-SWE-agent via their
  repositories.
- **Established baselines.** Each has published eviction results to compare against.
- **Deliberate pairs.** Some archetypes get two tasks that differ on exactly one facet, so a failure can be
  attributed to that facet:

| pair | facet that differs |
|---|---|
| UUID-in-essays vs JSON-KV | redundancy / composition |
| CWE vs GovReport | redundant vs non-redundant aggregation |
| AIME vs LongBench-Write | exact vs open fidelity |
| HotpotQA vs ALCE-ASQA | query after vs before the context |

## 3. Lite suite and the stream set

### 3.1 Lite suite (14 tasks, 91% coverage)

`mmlu`, `gsm8k`, `ifeval`, `aime`, `longbench-write:longbench-write`, `ruler:niah-single-3`, `ruler:vt`,
`ruler:cwe`, `longbench:hotpotqa`, `helmet:banking77`, `longbench:repobench-p`, `scbench:retr-kv`, `scbench:en-qa`,
`tau-bench`.

It drops the streaming control, query-before-context and mixed-long growth. Use it for iteration; use the core
suite for claims.

### 3.2 Stream set (cross-request / prefix-cache eviction)

Replay each trace with prefix-contiguous hit semantics, using `analysis/serving_traces.py` as a reference
simulator. For each trace:

- report the hit ratio at 3–5 capacities around the knee (where LRU reaches 50% and 90% of the ideal);
- always include Belady's OPT as the upper bound;
- report LRU as the baseline.

| stream archetype | trace | what separates good from bad policies |
|---|---|---|
| shared_prefix_fanin | Qwen-Bailian to-B API | frequency / popularity (LFU 0.41 vs LRU 0.29 at 64K tokens) |
| session_continuation | Qwen-Bailian to-C chat; Mooncake conversation | minute-scale think-time gaps (LRU 0.30 vs OPT 0.52 at 1M tokens) |
| agentic_loop | Mooncake tool&agent; SWE-bench / τ-bench trajectories re-played as prefix streams | short-range recency; pinning of persistent prefixes |
| reasoning_heavy | Qwen-Bailian thinking | admission control for decode KV that the next turn never reads |
| coder | Qwen-Bailian coder | mixed same-session (45%) and cross-session reuse |
| fanout | synthetic (SGLang-style self-consistency / tree search) | branch-point retention |

Vary two regimes explicitly:

- **capacity-bound**: an engine-local cache;
- **TTL/cost-bound**: a provider cache with 5-minute TTLs.

The same policy ranks differently in the two.

## 4. Policy-family risk map: what to test first

| your policy resembles | what it assumes | highest-risk archetypes / facets | first tasks to run |
|---|---|---|---|
| sink + recent window (StreamingLLM, LM-Infinite) | dependencies are local | sparse retrieval in the middle; multi-hop; memory; persistent instructions beyond the sinks | RULER NIAH depth sweep; LoCoMo; τ-bench |
| accumulated attention / heavy hitters (H2O, Scissorhands, TOVA) | importance persists | `deferred` follow-ups; exact long spans; flat-attention aggregation | SCBench multi-turn; RULER UUID; CWE |
| end-of-prompt observation window (SnapKV, PyramidKV, Ada-KV, CAKE) | the query is the prompt's tail and fixed | `deferred` and `known_first` queries; decode-dominated growth (a no-op there); multi-instruction prompts | SCBench multi-request; ALCE; AIME; system-prompt IFEval |
| head-level full/streaming split (DuoAttention, RazorAttention, HeadKV) | the heads that matter are retrieval heads | reasoning heads ≠ retrieval heads (RLKV); GQA models need more heads | AIME + RULER on a GQA model |
| query-agnostic scoring / reconstruction (KVzip, Expected Attention, KeyDiff) | future queries are unknown | natural multi-hop QA vs synthetic (KeyDiff: 90.7 on RULER vs 51.5 on HotpotQA); model dependence | HotpotQA, LoCoMo, RULER on ≥ 2 model families |
| decode-time / reasoning-aware (R-KV, LazyEviction, RPC, ThinKV, SCOPE, MorphKV) | self-generated text is redundant; importance recurs | prompt-anchored exact retrieval during long output; persistent constraints; output-length inflation | MRCR / LongProc HTML→TSV; IFEval; SWE-bench |
| merging / quantization hybrids (CaM, MiniCache, KIVI) | small errors are tolerable | exact high-entropy spans; realized compression varies per workload | JSON-KV; CrossCodeEval; report the realized ratio |
| cross-request LRU / LFU / TTL / learned | reuse structure is stationary | the source of reuse differs per service (§3.2) | all stream archetypes, both regimes |

## 5. Evaluation protocol checklist

These are the conditions under which results are comparable and failures attributable. Each item is traced to the
evidence in [literature §3](literature.md#3-what-the-evaluation-studies-found).

1. **Budgets.** Report absolute token budgets *and* ratios, and whether they are per layer or per head. For
   decode-dominated tasks, set a *decode-time* budget: prefill-only compression is a no-op there. Report the
   realized compression for threshold- and merging-based methods.
2. **Query visibility.** Run every long-context task both *query-aware* (question inside the compressed context)
   and *query-agnostic* (compress the context, then append the question), as kvpress does. For multi-question
   datasets, add **shared-context mode**: L-Eval, QuALITY, LoCoMo, LongMemEval and NoCha carry this alternative
   protocol in the catalog (`--alt`).
3. **Paired per-sample reporting.** Compare against the full-cache run on the same samples and seeds. Report
   right→wrong and wrong→right flips and bootstrap confidence intervals. Averages hide failures that are specific
   to one method (KVDiagnosis: 12,520 vs 1,004 flips).
4. **Output length.** Report mean, p90 and p99 output length, the length-cap hit rate and the loop/repetition
   rate. Also report end-to-end decoded tokens: compression can lengthen outputs and raise total cost
   (Rethinking-KV; Hold Onto That Thought).
5. **Fidelity and answer span.** Include long exact spans (UUIDs, 32–64-digit keys, verbatim reproduction as in
   MRCR). Single-token answers are decoded from an uncompressed prefill and hide eviction.
6. **Parametric leakage.** Add closed-book baselines, or prefer counterfactual or synthetic content (L-Eval
   sci_fi, InfiniteBench fake-book, NoLiMa). Otherwise knowledge masks damage.
7. **Evidence position.** Sweep depth, and include early constraints (system prompt, tool schemas). Record how
   the benchmark truncates over-long inputs: LongBench cuts the middle.
8. **Persistent constraints.** Score instruction-level compliance (IFEval), τ-bench policy violations and
   system-prompt leakage under compression.
9. **Growth over time.** For decode-dominated and accumulating workloads, report cache size vs. step (peak and
   area under the curve), not just a single budget.
10. **Interaction with prefix caching.** State whether compressed KV stays shareable across requests and steps.
    Cross-step reuse saves 9–44× prefill on SWE-bench; a policy that rewrites earlier KV can forfeit it.
11. **Model diversity.** Use at least one MHA and one GQA model, and at least one thinking model. Head-level
    redundancy differs by architecture (DuoAttention), and reasoning models behave differently (Hold Onto That
    Thought; RLKV).
12. **Report per archetype.** Group results by the archetypes of the tasks (the coverage tool's grouping) and flag
    each task's validity flags (`parametric_leakage`, `short_answer`, `fuzzy_metric`, …).

## 6. Next step: model-based validation of the taxonomy

The taxonomy predicts *which* workloads a policy family will fail on, from model-free workload properties. The
natural next study tests this prediction directly and complements the characterization with attention-level
measurements. We did not run it here, because no model weights were reachable from the research environment.

1. **Grid.**
   - 2 tasks per archetype (the core suite).
   - Models: Llama-3.1-8B-Instruct (GQA), Qwen2.5/3-8B, and one R1-distilled 8B thinking model.
   - Policies: full cache, StreamingLLM, H2O, SnapKV, Ada-KV, KVzip, R-KV (the six families above).
   - Budgets: 4 per policy, from 128 tokens to 50%.
   - Both query-visibility protocols.
2. **Outcomes.** Per-sample flips vs. full cache, output-length change, and realized memory.
3. **Probes.**
   - evidence-token retention rate per policy (where gold evidence exists: RULER, LoCoMo, HotpotQA);
   - per-head attention entropy and top-k mass;
   - importance drift, measured as the Jaccard of the top-k sets across decoding steps.
4. **Test.** Regress degradation on the workload signature: MR(W), scope and spread, query timing, fidelity and
   redundancy. The taxonomy is validated if the signature explains most of the variance in degradation *within*
   capability domains, and capability domain adds little once the signature is known.
