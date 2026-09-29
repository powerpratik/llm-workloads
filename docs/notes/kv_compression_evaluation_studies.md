> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# How task types affect KV-cache compression: evaluation studies and mechanistic evidence

Purpose: evidence base for a workload taxonomy for evaluating KV-cache eviction policies.
Compiled 2026-09-29. Everything below comes from sources I could reach. Numbers I could not confirm are marked "unknown".

## Confidence legend
- **[verified]**: read directly in an author-controlled primary artifact (repo README, figure, released data or config, PR text, bibtex), or computed by me from the authors' released data (marked **[derived]**).
- **[likely]**: taken from search-engine snippets of the paper's abstract or HTML pages. Consistent with other snippets, but I did not read the full paper.
- **[unverified]**: a single secondary snippet, a search-tool summary of a table, or my own background knowledge.

## Access constraints (they limit what could be verified)
- Blocked: arxiv.org, openreview.net, aclanthology.org, huggingface.co, semanticscholar, researchgate, and also github.io, par.nsf.gov, medium.com, liner.com, alphaxiv, papernotes, awesomepapers, lacuna, lsh-ai.com and research.nvidia.com (the proxy denied CONNECT).
- WebSearch hit the session budget (200/200) partway through. Later items were verified only from GitHub, through raw.githubusercontent.com and github.com pages.
- As a result, most "full-paper" details such as per-task tables are **[likely]** or **unknown**. The strongest quantitative evidence comes from released artifacts: the SCBench task table, the KVzip benchmark figure, the KVDiagnosis result CSVs, the Rethinking-KV length logs and the HELMET correlation matrix.

---

# PART A: Evaluation and analysis studies

## A1. SCBench: A KV Cache-Centric Analysis of Long-Context Methods
- **Citation**: Yucheng Li, Huiqiang Jiang, Qianhui Wu, Xufang Luo, Surin Ahn, Chengruidong Zhang, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, Lili Qiu. ICLR 2025 (also ENLSP @ NeurIPS'24). arXiv 2412.10319. **[verified: MInference README]**
- **Repo**: https://github.com/microsoft/MInference/tree/main/scbench. Data: huggingface.co/datasets/microsoft/SCBench. **[verified]**
- **Task categories (12 tasks, 4 capabilities)** **[verified: repo Table 2 image `images/scbench/overview.png`]**

  | Task | Capability | Avg input | Avg output | Sessions/Turns |
  |---|---|---|---|---|
  | Retr.KV | String retrieval | 125K | 943 | 100/500 |
  | Retr.Prefix-Suffix | String retrieval | 112K | 914 | 100/500 |
  | Retr.MultiHop (variable tracking) | String retrieval | 124K | 410 | 90/450 |
  | Code.RepoQA | Semantic retrieval | 65K | 6,058 | 88/440 |
  | En.QA | Semantic retrieval | 198K | 272 | 69/351 |
  | Zh.QA | Semantic retrieval | 1.5M | 322 | 35/189 |
  | En.MultiChoice | Semantic retrieval | 188K | 215 | 58/299 |
  | Math.Find | Global information | 120K | 172 | 100/240 |
  | ICL.ManyShot | Global information | 22K | 975 | 54/270 |
  | En.Sum | Global information | 104K | 1,170 | 79/350 |
  | Mix.Sum+NIAH | Multi-tasking | 105K | 3,441 | 70/560 |
  | Mix.RepoQA+KV | Multi-tasking | 68K | 5,318 | 88/704 |
  | **Total** | | **227K** | **1,684** | **931/4,853** |

  The repo describes Retr.KV as "key-value lookup in large JSON objects with random, **incompressible** content". **[verified: README]**
- **Shared-context modes** **[verified: onepage figure and readme]**
  1. Multi-turn: long context, then Q1, A1, Q2, A2 and so on, with the KV cache reused across turns. This is the default mode.
  2. Multi-request: a "hinted" (pre-built or compressed) KV cache is shared by independent queries. Enabled with `--same_context_different_query`.
- **Methods**: 8 categories on 8 LLMs (Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, GLM-4-9B, Codestral Mamba, Jamba-1.5). The categories are gated linear RNNs, hybrids, sparse attention (MInference, A-shape, Tri-shape), KV dropping (StreamingLLM, SnapKV, PyramidKV), quantization (KIVI), retrieval (CacheBlend), loading (Quest, RetrievalAttention) and prompt compression (LLMLingua-2). **[likely: abstract; method list verified in README/onepage]**
- **Default budgets in the released code** **[verified: scbench/readme.md]**: SnapKV/PyramidKV `max_capacity_prompt=4096` (window 32); StreamingLLM and A-shape `n_init=128 + n_local=3968`; Tri-shape adds `n_last=100`; Quest `token_budget=1024`; KIVI 2-bit.
  - [derived] A 4K budget is about 1.8% of the 227K average input, or about 3.3% of Retr.KV's 125K.
- **Key findings**
  - **[verified: onepage figure, verbatim]**
    - "Sub-O(n) memory is almost infeasible in multi-turn decoding"
    - "Task performance shows varying decline trends"
    - "All long-context methods experience performance degradation as the compression rate decreases"
    - "Long-generation scenarios exhibit distribution shift issues"
  - **[likely: abstract]** O(n)-memory methods do well in multi-request scenarios. Sub-O(n) methods do well in single-turn settings but struggle with complex interactions. Sparse encoding with O(n) memory and sub-O(n²) prefill is robust. Dynamic sparsity gives more expressive KV caches than static patterns. Layer-level sparsity in hybrids reduces memory with strong performance.
  - **[likely: search snippets of the paper body]**
    - SnapKV and static Tri-shape struggle when the query is not visible. Dynamic MInference generalizes better.
    - StreamingLLM and SnapKV are consistently the weakest, especially in multi-turn mode.
    - Accuracy on follow-up turns degrades sharply, often near zero on retrieval tasks.
  - **[unverified: search-tool summary of a table]** Llama-3.1-8B on Math.Find: MInference 30.8, A-shape 18.5, Tri-shape 23.2, StreamingLLM 0.1, SnapKV 0.0. A claim that SnapKV reaches 0% on string retrieval in multi-request mode and on consecutive turns is also unverified.
- **Mechanism.** Query-aware prefill compression, such as SnapKV's observation window, keeps what the first query needs. Later queries in the session need different tokens, which were irreversibly dropped. String-retrieval payloads are random and incompressible, so nothing redundant is left to fall back on.
- **Methodological lessons**
  - Evaluate the full KV lifecycle (generation, compression, retrieval, loading).
  - Include follow-up turns and independent requests over a shared prefix.
  - Report per-turn accuracy, not just turn-1.
  - Long outputs (RepoQA and Mix tasks average 5–6K output tokens) expose a distribution shift that prefill-only benchmarks miss.
- **Sources**: https://github.com/microsoft/MInference (README, images/SCBench_onepage.png, images/scbench/overview.png, comparison.png); https://arxiv.org/abs/2412.10319

## A2. KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches
- **Citation**: Jiayi Yuan*, Hongyi Liu*, Shaochen (Henry) Zhong*, et al. (corresponding: Zhong, Zirui Liu). EMNLP 2024 Findings. arXiv 2407.01527. **[verified: repo README]**
- **Repo**: https://github.com/henryzhongsc/longctx_bench. Logs and raw results are on Google Drive, which I did not access.
- **Tasks** **[verified: README]**
  - 15 LongBench datasets: NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, Musique, GovReport, QMSum, MultiNews, TREC, TriviaQA, SAMSum, PassageRetrieval, LCC, RepoBench-P. These are grouped as single-doc QA, multi-doc QA, summarization, few-shot, synthetic and code.
  - A Paul-Graham passkey NIAH: 20,480 words, 7-digit passkey, 10×10×3 grid.
  - Models: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, LongChat-7B-32K, plus Mamba, Mamba-2, RWKV-5 and RecurrentGemma.
  - Methods: KIVI and FlexGen quantization; token dropping with StreamingLLM, H2O and InfLLM at 2×/4×/6×/8×; LLMLingua-2 prompt compression.
- **Findings**
  - **[likely: search snippets of the paper text]**
    - **Keeping prefill uncompressed is crucial.** Compressing during prefill makes later-layer representations of the prompt inaccurate.
    - Quantization gives the most reliable performance across tasks.
    - Token dropping (StreamingLLM, H2O) retains near-baseline **code completion** even at high ratios, so it "excels in specific domains".
    - Hard prompt compression (LLMLingua-2) is the worst on the needle test, because deleting the needle makes retrieval impossible.
    - Linear-time models (Mamba, RWKV) struggle with retrieval. Mixing in attention (RecurrentGemma) helps on every task.
  - **[likely: snippet quoting the paper] Answer-length artifact.** H2O is 100% on the 7-digit passkey at 4×, but drops to **35.0%** when the passkey is 64 digits.
    - KIVI-2bit goes from 100% to 91.0%. InfLLM at 4× goes from 20.7% to 19.0%.
    - The authors' explanation: H2O does not evict during prefill. The first generated token, which carries at least 3 digits because of tokenization, is decoded with a full cache. The first digits are therefore correct "for free", and short-answer needle tests overstate the robustness of decode-time eviction.
  - **[verified: figure `visualization/plotted/cover_vis.png`]** Radar profiles over Multi-Doc QA, Single-Doc QA, Code, Synthetic, Few-shot and Summarization. Quantization (KIVI-2/4bit) stays close to baseline. Token dropping at 4× and LLMLingua-2 at 2–8× shrink **unevenly** across categories. This is my qualitative reading; the plot is too small to extract numbers.
- **Methodological lessons**
  - Compressing prefill versus decode are different treatments.
  - Short single-token answers leak full-cache information.
  - Answer-span length is a hidden workload parameter.
  - Code completion is compatible with local-context retention and can inflate token-dropping averages.
- **Sources**: https://github.com/henryzhongsc/longctx_bench; https://arxiv.org/abs/2407.01527; https://aclanthology.org/2024.findings-emnlp.266/

## A3. Can LLMs Maintain Fundamental Abilities under KV Cache Compression? (KVFundaBench, ShotKV)
- **Citation**: Xiang Liu, Zhenheng Tang, et al. (first-author list per Semantic Scholar slug "Liu-Tang") **[likely]**. arXiv 2502.01941 **[likely]**. A later version appears retitled "Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression", listed under ICML 2026 by a paper-notes site **[likely]**. OpenReview id 4zwyuDgbWD. Repo: none found on GitHub (searched "KVFundaBench", "ShotKV").
- **Tasks** **[likely]**
  - World knowledge: MMLU.
  - Commonsense: CommonsenseQA.
  - Arithmetic: GSM8K.
  - Code: HumanEval.
  - Safety: JailBreakV.
  - Long-context understanding and generation: LongGenBench-GSM8K.
  - LongGenBench (Liu et al., EMNLP 2024 Findings, arXiv 2410.04199) asks for many questions answered in one long response. With no compression, it reports 1.2–47.1% degradation relative to answering the questions individually **[verified: LongGenBench README]**.
- **Findings** **[likely: abstract snippets]**
  - Arithmetic reasoning is the most compression-sensitive: **17.4%–43.3%** drops across methods under aggressive compression.
  - World knowledge and commonsense are resilient. Arithmetic, code and safety "collapse" when retained budget is below about 20%. This threshold was reported by one snippet only, so treat it as **[unverified]**.
  - Multi-step reasoning models (R1-distilled; o1 is also mentioned) are more robust than instruction-tuned models.
  - **Shorter prompts are more vulnerable** than long ones.
  - Chunk-level compression, which preserves semantic units, is better on complex long-context reasoning.
  - Long-context generation loses more than 20% (snippet phrasing unclear, **[unverified]**).
  - ShotKV keeps few-shot examples whole during prefill and applies separate token-level eviction during decoding. It gains **9–18%** on long-context generation under aggressive ratios. The later version reports an 11% latency reduction vs full cache **[likely]**.
- **Mechanism (authors' framing).** Reasoning chains and few-shot exemplars are "high-density" semantic units. Token-level eviction fragments them, breaking CoT links, while fact-lookup tasks draw on parametric knowledge that is not in the KV cache.
- **Methodological lessons**
  - Retrieval-centric benchmarks mask reasoning degradation.
  - Include short-prompt tasks, since relative budget matters.
  - Separate prefill and decode budgets.
  - Treat exemplar and shot boundaries as structure.
- **Sources**: https://arxiv.org/abs/2502.01941 (snippets); https://openreview.net/forum?id=4zwyuDgbWD; https://github.com/Dominic789654/LongGenBench

## A4. The Pitfalls of KV Cache Compression
- **Citation**: Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Mingyi Israel. ACL 2026 (long), pp. 41530–41553. arXiv 2510.00231. **[verified: repo bibtex]**
- **Repo**: https://github.com/alexluchen/pitfalls-of-kv-cache-compression. It contains a kvpress fork with "fair eviction" presses. **[verified]**
- **Setup**
  - StreamingLLM, SnapKV, TOVA, H2O (ObservedAttention in code) and K-Norm, on Llama-3.1-8B and Qwen2.5-14B **[likely: abstract]**.
  - Multi-instruction prompting uses IFEval, adapted from SystemCheck `sys_ifeval`, with system-prompt instructions **[verified: README]**.
  - Leakage uses RaccoonBench with defense templates **[verified: README]**.
  - Compression-ratio sweep defaults to 0.0 to 0.95 in 100 steps, with max_new_tokens 1280 **[verified: CLI defaults]**.
- **Findings** **[likely: abstract]**
  - Some instructions degrade much faster than others under compression and are effectively ignored.
  - System-prompt leakage rises under compression.
  - Leakage depends on the compression method, **instruction order** and **KV eviction bias**.
  - Two simple changes, **whitelist retention** and **fair eviction**, reduce leakage and stabilize instruction following.
  - Which IFEval instruction categories fail first is **unknown**, as are the leakage rates.
- **Mechanism** **[verified: code docstring; interpretation mine]**
  - `SnapKVFairEvictionPress` splits the observation window between the defense or earlier span and the later instruction span. Each window "votes" only within its own span, and budgets are proportional to span length.
  - This implies the failure mode: the scoring window at the end of the prompt, the latest instruction or user turn, preferentially keeps tokens relevant to itself and evicts earlier instructions such as the system prompt or defenses.
- **Methodological lessons**
  - Prompts with **multiple independent constraints spread across positions**, such as system prompt, tools and user turn, are a distinct workload class.
  - Instruction position relative to the scoring window matters.
  - Aggregate IFEval accuracy hides per-instruction collapse.
  - Security properties like leakage need to be measured explicitly.
- **Sources**: https://github.com/alexluchen/pitfalls-of-kv-cache-compression; https://arxiv.org/abs/2510.00231; https://aclanthology.org/2026.acl-long.1926/

## A5. Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving (MLSys 2025)
- **Citation**: MLSys 2025, arXiv 2503.24000 **[verified: repo README; venue per repo title]**. The first author is **unverified**; the PDF is hosted on Tianwei Zhang's page, and I believe the first author is Wei Gao.
- **Repo**: https://github.com/LLMkvsys/rethink-kv-compression. It includes a throughput predictor, a length predictor, a negative-sample evaluator, and cached length and negative-sample data. **[verified]**
- **Findings**
  - **[verified: README]** (1) Current compression implementations do not fit production serving stacks (FlashAttention, PagedAttention), so throughput gains are suboptimal. (2) "Compressing KV cache may lead to longer outputs, resulting in increased end-to-end latency."
  - **[likely: snippets]** LongBench averages:

    | Model | Baseline | KIVI | GEAR | H2O | StreamingLLM |
    |---|---|---|---|---|---|
    | Llama-3.1-8B-Instruct | 41.2 | 41.3 | 40.9 | 39.1 | 38.9 |
    | Mistral-7B | 33.3 | 33.4 | 33.4 | 31.8 | 30.4 |

  - On Mistral-7B, compression "considerably affects" **summarization and QA**. **[likely]**
  - Negative samples are samples that degrade by more than a 10% threshold (exact definition unverified). They are numerous even when the average loss is small, and quantization and sparsity methods show a similarly **unbalanced fragility across task types**. **[likely]**
  - "Over 20% of samples showed at least a 1.5× increase in response length", and higher compression makes the verbosity worse. **[likely]**
- **My re-analysis of the released length logs** **[derived]**
  - Setup: ShareGPT, Llama-3-8B-Instruct, 1,000 requests, max 1,024 new tokens. Prompts average 1,222 tokens (median 768, minimum 513).
  - Script: a working script (not retained); data: `benchmark_len/*/sharegpt.jsonl`.

    | Config (dir name) | % samples ≥1.5× longer | % ≤0.67× shorter | naturally-terminating samples that now hit the 1,024 cap | "degenerate/looping" outputs (heuristic) |
    |---|---|---|---|---|
    | baseline | — | — | — | 174/1000 |
    | H2O c256 | 21.4 | 14.6 | 15.2% | 204 |
    | H2O c512 | 12.1 | 9.9 | 8.8% | 166 |
    | StreamingLLM c256 | 26.4 | 23.4 | 17.0% | 189 |
    | StreamingLLM c1024 | 9.0 | 11.3 | 8.4% | 166 |
    | KIVI 2-bit | 24.6 | 17.2 | 20.8% | 233 |
    | KIVI 4-bit | 9.3 | 9.9 | 9.1% | 157 |
    | GEAR "4bits" dir | 27.1 | 10.8 | 27.5% | 310 |
    | GEAR "8bits" dir | 16.8 | 12.1 | 17.8% | 236 |

  - How to read the table:
    - The median length ratio is 1.00 everywhere. Compression mostly **widens the length distribution** and adds a heavy right tail of runaway generations that hit the cap and often loop.
    - Shorter outputs also occur (10–23%).
    - Noise floor: with StreamingLLM c1024, 274 of 300 samples that never needed eviction produced byte-identical output. Mild settings at about 8–9% cap-hitting approximate this noise floor.
    - Sampling mode is not documented in the artifact.
- **Methodological lessons**
  - Report **output length and time to completion** alongside accuracy. Compression can raise end-to-end cost despite lower memory.
  - Evaluate per sample, since averages hide negative samples.
  - Measure real throughput in a serving engine, not just the compression ratio.
- **Sources**: https://github.com/LLMkvsys/rethink-kv-compression; https://arxiv.org/abs/2503.24000; https://proceedings.mlsys.org/paper_files/paper/2025/hash/26289c647c6828e862e271ca3c490486-Abstract-Conference.html

## A6. NVIDIA kvpress: benchmark suite and leaderboard
- **Repo**: https://github.com/NVIDIA/kvpress. Paper to cite: Devoto, Jeblick, Jégou, "Expected Attention", arXiv 2510.00636. Leaderboard: huggingface.co/spaces/nvidia/kvpress-leaderboard, which was blocked. **[verified: README]**
- **Datasets** **[verified: `evaluation/evaluate_registry.py`]**
  - Prefill: `loogle`, `ruler` (4k default), `zero_scrolls`, `infinitebench`, `longbench`, `longbench-e`, `longbench-v2`, `needle_in_haystack` (Paul Graham essays; has a `needle_depth` knob).
  - Decoding compression: `aime25` and `math500`.
- **Protocol** **[verified]**
  - The default is `query_aware: false`: the question is excluded from the compressed context.
  - The README FAQ says including the question "would artificially favor methods such as SnapKV. Ideally, we want a compression method that works whatever comes after the context (e.g. chat or document QA)."
  - Setting `query_aware: true` puts the question in the compressed context.
- **Press families** **[verified]**
  - Attention-score (SnapKV, TOVA, ObservedAttention/H2O).
  - Statistical and query-agnostic (Knorm, KeyDiff, LagKV, ExpectedAttention, Leverage/CUR).
  - Reconstruction-based (KVzip, KVzap, FastKVzip).
  - Structural (StreamingLLM, DuoAttention, SimLayerKV, PyramidKV, ThinK).
  - Wrappers (AdaKV head-wise, CriticalKV, ChunkKV, DecodingPress, DMSPress threshold-based).
  - DecodingPress is "experimental", with default interval 512 and target 2048.
- **Leaderboard content**
  - The main plot is average RULER-4K accuracy vs compression ratio, for Qwen3-8B and Llama-3.1-8B-Instruct **[likely]**.
  - Data points quoted in GitHub PRs:
    - Qwen3-8B RULER-4K baseline is **95.29**. KVzip scores 95.24, 95.08 and 92.38 at ratios 0.5, 0.75 and 0.875 **[verified: PR #279 table]**.
    - At 87.5% compression, the query-aware presses on the leaderboard score ChunkKV **61.2**, Finch **61.7** and SnapKV **55.5**, vs KVzip **92.2** **[verified as quoted in PR #292; contributor-reported]**.
  - The PR #292 contributor's diagnosis is **[verified quote; claim itself likely]**: "the needle is usually found but miscopied" (a UUID corrupted), because "question queries often attend only to the start of the relevant span; in decode the entire span is often required."
  - Per the KVzap paper **[likely: snippet]**, Expected Attention's apparent gains over full cache on LongBench are "largely driven by outlier accuracies on the TREC subset". KeyDiff is strong on Llama-3.1-8B but weak on Gemma3-12B and Qwen3-8B (Expected Attention paper).
- **Methodological lessons**
  - The query-agnostic default is the right deployment proxy.
  - Averages over RULER hide per-subtask collapse, such as UUID copying.
  - Single-subset outliers like TREC can dominate LongBench averages.
  - Method rankings are model-dependent.
- **Sources**: https://github.com/NVIDIA/kvpress (README, evaluation/README.md, evaluate_registry.py, evaluate_config.yaml); https://github.com/NVIDIA/kvpress/pull/279; https://github.com/NVIDIA/kvpress/pull/292

## A7. KVzip (method paper, but its query-agnostic benchmark is key evidence)
- **Citation**: Jang-Hyun Kim et al. (first author **[likely]**). NeurIPS 2025 Oral. arXiv 2505.23416. **[verified: README]**
- **Repo**: https://github.com/snu-mllab/KVzip
- **Setup** **[verified: README and `images/benchmark.png`]**: query-agnostic multi-query evaluation. The context is compressed once and queried many times. Model: Qwen2.5-7B-Instruct-1M. Tasks: SQuAD, NIAH, 10 SCBench tasks, GSM8K. The figure groups tasks into **Retrieval / Contextual QA / Redundancy** rows.
- **Readings from the figure** (±3 points, by eye): baselines SnapKV / PyramidKV / H2O vs full cache, as a function of KV-cache ratio.
  - **Retrieval**
    - NIAH: full 100. SnapKV about 52 at 0.5 and about 79 at 0.8. PyramidKV about 9 at ≤0.5. H2O about 0 until 0.8 and about 49 at 0.9.
    - Retr.KV: full about 60. SnapKV/PyramidKV about 9 at 0.5 and about 51–58 at 0.9. H2O about 0 until 0.9.
    - Retr.Prefix-Suffix: full about 50. SnapKV about 4 at 0.5 and about 31 at 0.9.
    - Code.RepoQA: full about 62. SnapKV about 24 at 0.5.
  - **Contextual QA**
    - SQuAD: full about 93. SnapKV about 52 at 0.5.
    - GSM8K (problem as context): full about 69. SnapKV about 32 at 0.5.
    - En.QA and En.MultiChoice degrade much less: SnapKV about 39/41 and about 71/78 at 0.5.
  - **"Redundancy"**: En.Summary, Retr.MultiHop, Math.Find and ICL.ManyShot degrade only gradually.
    - SnapKV stays within about 1–4 points of full at 0.3–0.5. At 0.1 it loses about 15–30% relative, e.g. En.Summary about 29 vs 37 ROUGE.
    - PyramidKV is somewhat worse: En.Summary about 31 at 0.5, Math.Find about 20 at 0.1.
    - ICL.ManyShot is flat throughout.
    - H2O degrades strongly on this row too.
  - KVzip itself stays near full at 0.3 on most tasks.
- **Claim** **[likely: abstract]**: query-aware eviction methods "suffer from performance degradation even at a 90% cache budget ratio under multi-query scenarios".
- **Lessons**
  - The same budget is harmless for redundant, global tasks and catastrophic for exact retrieval once the query is hidden.
  - Arithmetic contexts (GSM8K) behave like retrieval, because every number matters.
- **Sources**: https://github.com/snu-mllab/KVzip; https://arxiv.org/abs/2505.23416

## A8. Hold Onto That Thought: Assessing KV Cache Compression on Reasoning
- **Citation**: Minghui Liu*, Aadi Palnitkar*, Tahseen Rabbani*, et al. 2025. arXiv 2512.12008. Listed at NeurIPS 2025, likely a workshop **[likely]**.
- **Repo**: https://github.com/minghui-liu/kvpress, a kvpress fork with decoding-time presses and a reasoning harness. **[verified]**
- **Setup** **[verified: README]**
  - Every press is made to act during decoding.
  - Datasets: GSM8K, MATH-500, FOLIO, DROP, StrategyQA, ReClor, CommonsenseQA, OpenBookQA, LogiQA, AIME24/25. They are grouped as reading comprehension (DROP, ReClor), logical (StrategyQA, FOLIO), commonsense (OpenBookQA, CSQA) and math (MATH-500, GSM8K).
  - Budgets {128, 256, 384, 512} tokens; 2k decoding limit.
  - Models: Llama-3.1-8B-Instruct, Llama-3.1-Nemotron-Nano-8B-v1, DeepSeek-R1-Distill Qwen/Llama.
- **Findings** **[verified: README "Findings"]**
  1. The non-reasoning Llama-3.1-8B has no single best press. H2O and StreamingLLM are good on reading comprehension. SnapKV-D is stronger on **short prompts with long CoT**.
  2. On reasoning models, heavy-hitter tracking dominates. H2O and SnapKV-D "frequently match or exceed the uncompressed baseline even at 256-token budgets".
  3. At low budgets, similarity-based pruning (R-KV, K-Norm) "can lengthen reasoning traces", which trades cache size against total decoding cost. The abstract, **[likely]**, states this more generally: eviction at low budgets can produce longer reasoning traces.
- **Lessons**
  - For short-prompt, long-output workloads, prefill-only compression is irrelevant, so decode-time eviction must be evaluated.
  - Output length is an outcome variable.
  - Rankings depend on the dataset family.
- **Sources**: https://github.com/minghui-liu/kvpress; https://arxiv.org/abs/2512.12008; https://openreview.net/forum?id=OtZtLYAdQY

## A9. Long-output and reasoning evidence from 2025–26 method papers (their motivation analyses double as workload evidence)
- **SCOPE** (ACL 2025 long, arXiv 2412.13649; repo Linking-ai/SCOPE) **[verified: README "Key Observations"]**
  - "Excessive compression during the prefill phase, which requires specific full context, impairs the comprehension of the reasoning task."
  - "Deviation of heavy hitters occurs in the reasoning tasks with long outputs."
  - **[likely]** It reaches near-full performance with 35% of KV memory on LongGenBench. Keeping the full prefill cache matters most for multi-step tasks (GSM8K+).
- **R-KV** (arXiv 2505.24133; repo Zefan-Cai/R-KV) **[verified: README]**
  - "Existing compression tools focus on long prompts and falter on long generations — often pruning the wrong tokens because redundant self-checks still attend heavily to themselves."
  - At 16% cache (R1-Distill-Llama-8B) or 33% (Qwen-14B), R-KV reaches 105% of full-KV accuracy on MATH-500 and AIME24. Snippet claim **[likely]**: about 100% at 10% budget vs about 60% for baselines.
  - The README also warns that a data bug froze GSM8K scores at about 40% before commit e9f54c45. This is an example of benchmark-harness fragility.
- **LazyEviction** (arXiv 2506.15969; ACL 2026 long per ACL Anthology listing **[likely]**; repo Halo-949/LazyEviction)
  - **[verified: README]** A "**Token Importance Recurrence**" phenomenon in reasoning: tokens show recurrent attention patterns. Both current-attention and cumulative-attention eviction "fail to preserve recurring tokens during their low-attention intervals".
  - **[unverified]** A figure of ">95% of tokens" was mentioned in one snippet.
  - **[likely]** 50–70% KV reduction at comparable accuracy.
- **ThinKV** (arXiv 2510.01290; ICLR 2026 Oral **[likely]**) **[likely]**
  - Attention sparsity separates CoT into thought types with different importance. The paper names reasoning, execution and transition thoughts **[unverified]**.
  - A hybrid quantization and eviction scheme is near-lossless at under 5% of the KV cache on R1-Distill, GPT-OSS and AceReason (math and code).
- **RLKV: Which Heads Matter for Reasoning?** (arXiv 2510.08525; LIT @ ICLR 2026 workshop) **[verified: README]**
  - Reasoning LLMs are "highly fragile to information loss during decoding".
  - "Token-dropping methods directly disrupt reasoning chains … head-reallocation methods, designed for retrieval tasks, fail to preserve the heads essential for generative reasoning." Heads also "control generation termination".
  - Full cache on a subset of heads gives 20–50% reduction at near-lossless quality.
  - Qwen-2.5-7B-R1 sparsifies less, attributed to a GQA group size of 7.
- **MorphKV: "Dialogue Without Limits"** (ICML 2025; arXiv 2503.00979) **[verified: README]**
  - Relevant context "dynamically shifts as token generation progresses".
  - At outputs of 12K tokens, MorphKV degrades about 10% vs 15–18% for SnapKV and H2O-style baselines. SnapKV prunes only once at the start of the response.
- **Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression** (arXiv 2608.01631, Aug 2026) **[likely: abstract]**
  - Uses fixed-trace replay: 10 token-eviction methods and 1 quantization method, on 3 models, across math, scientific QA, clinical calculation and long-context retrieval.
  - Final-answer accuracy and the validity of the supporting rationale are preserved **at different rates**, so accuracy-only evaluation is insufficient. Specific numbers are unknown.
- **KVzap** (arXiv 2601.07891; README in kvpress) **[verified README; results likely]**
  - Uses threshold-based (DMS) pruning, so the realized compression ratio adapts per input.
  - Reported 2–4× compression with negligible loss on Qwen3-8B, Llama-3.1-8B and Qwen3-32B for long-context and reasoning tasks (AIME25 evaluated with sampling).

## A10. KVDiagnosis: A Diagnostic Benchmark for KV-Cache Compression in Long-Context LMs (2026)
- **Citation**: arXiv 2608.09412 (Aug 2026), KAUST. The KDD 2027 listing comes from one snippet **[likely]**.
- **Repo**: https://github.com/ChosenQC/KVDiagnosis **[verified: README and released CSVs]**
- **Setup** **[verified]**
  - Model: Qwen3-8B (fixed revision), deterministic decoding, kvpress 0.5.3.
  - 8 methods: StreamingLLM, SnapKV, TOVA, KeyDiff, ThinK, ChunkKV(Knorm), AdaKV, QuantizedCache.
  - Retained ratios 75%, 50% and 25%.
  - Workloads: RULER-8K and RULER-16K (1,100 sources each), Qasper and HotpotQA (200 each).
  - It is **unknown** whether the protocol was query-aware.
  - Scale: 59,800 supported runs and 12,520 C→W rows (full cache correct, compressed wrong).
  - PyramidKV was **excluded after an audit**: its adapter silently used SnapKV's path, giving identical outputs across 7,800 pairs.
- **Per-workload compressed task score** **[verified: `paper_artifacts/generated/full_evaluation_results.csv`]**. Full cache: RULER-8K 97.1, RULER-16K 96.8, Qasper 29.5, HotpotQA 74.0.

  | Method @ retained 50% | RULER-8K | RULER-16K | Qasper | HotpotQA |
  |---|---|---|---|---|
  | StreamingLLM | 57.2 | 54.9 | 21.5 | 57.5 |
  | SnapKV | 90.8 | 95.3 | 27.5 | 69.0 |
  | TOVA | 95.2 | 95.5 | 29.0 | 70.0 |
  | AdaKV | 93.4 | 96.3 | 27.5 | 70.0 |
  | KeyDiff | 90.7 | 90.9 | **18.0** | **51.5** |
  | ChunkKV | 76.6 | 78.3 | 21.0 | 45.0 |
  | ThinK (channel) | **0.0** | **0.0** | 6.5 | 12.5 |
  | QuantizedCache | 97.1 | 96.0 | 29.5 | 70.0 |

  At 25% retained: StreamingLLM on RULER-8K falls to 32.9. KeyDiff falls to 25.5 on HotpotQA and 13.5 on Qasper. QuantizedCache collapses (20.0 / 9.2 / 8.5 / 21.0), which is a cliff rather than a slope.
- **Other findings** **[verified]**
  - Transitions: C→C 36,378; **C→W 12,520; W→C 1,004**. Compression rarely helps, but it does flip some wrong answers to right.
  - **Failure sets differ by method**: SnapKV–TOVA failure Jaccard is 0.14–0.38 in 10 of 12 dataset×ratio cells. The exceptions are the small Qasper cells, at 0.0 and 0.70.
  - "Low or partial mapped [evidence] coverage is common". Only 19 rows combine high evidence coverage with severe gold-answer likelihood drift. Most failures therefore coincide with losing the evidence tokens, not with representation drift.
- **Lessons**
  - Method rankings flip across workloads. KeyDiff is fine on synthetic RULER but poor on real multi-hop QA, while StreamingLLM is the reverse in relative terms.
  - Evaluate per sample and pair each run with a full-cache control.
  - Audit implementations.
  - Quantization and eviction budgets are not byte-equivalent.

## A11. How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit (arXiv 2607.11942, Jul 2026)
- **Findings** **[likely: abstract snippets]**
  - Six published methods were compared with three trivial baselines on three open 7–9B models: 144,300 paired evaluations on RULER-8192 and 40,800 on LongBench.
  - Under the **query-agnostic** protocol, among the five methods sharing an attention backend, **only KeyDiff beats the best-of-3 trivial baseline consistently (31/36 cells)**. **SnapKV loses to simple baselines.**
  - The paper's motivation is that the economic case for compression is reuse, so compression must happen before the question is seen.
  - The identity of the trivial baselines (presumably recency or random) is unknown; there is no repo.
- **Lesson**: published rankings obtained with the query appended are not deployment rankings.

## A12. Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving (arXiv 2607.05399, 2026)
- **Findings** **[likely: abstract snippet]**
  - KIVI, TurboQuant, SnapKV and CaM (merging) were run on LongBench-style multi-doc QA, single-doc QA, few-shot and summarization, with Llama-3.1-8B and Mistral-7B. Metrics were TTFT, throughput and prefill KV memory.
  - "Compression ratio alone is a poor predictor of end-to-end performance."
  - KIVI-4 is the most stable in quality. SnapKV has the best long-context throughput.
  - **CaM shows substantial workload sensitivity in both quality and realized compression ratio.**
  - The authors recommend workload-aware selection of mechanisms.

## A13. Agentic and multi-turn KV compression (2026, mostly method papers). Leads only, **[unverified]** beyond titles
- CommitKV, "Lifecycle-Aware KV Cache Compression via Commit Transitions for Multi-Turn Agents" (arXiv 2608.07855). Reported up to +22.24 pp over the strongest compressed baseline and 5.00× end-to-end speedup.
- AgentKV, "Phase-Aware KV Eviction for Agentic LLMs" (arXiv 2609.14872).
- IntentKV, "Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference" (arXiv 2606.09916).
- "Practical Online KV Cache Compaction for LLM Agents: An Empirical Study" (arXiv 2608.00902).
- Tool-use evaluation groups in this literature include BFCL, FRAMES, GAIA, ToolHop and xbench-DeepSearch. I could not tie each to a specific paper.
- EpiCache (arXiv 2509.17396; repo apple/ml-epicache) targets **long conversational QA** with episodic (clustered) KV management. Its baselines are KVzip, KeyDiff, SnapKV and InfiniPot **[verified: README]**. Results are unknown.
- "Agentic AI Workload Characterization" (arXiv 2605.26297) **[unverified snippet]**: with context caching, most input tokens are reused across turns, so execution is decode-dominated with long-lived KV state.

---

# PART B: Mechanistic evidence that attention importance differs by task

## B1. Retrieval Head Mechanistically Explains Long-Context Factuality
- **Citation**: Wenhao Wu et al. arXiv 2404.15574. ICLR 2025 **[likely]**. Repo: https://github.com/nightdessert/Retrieval_Head **[verified]**.
- **Findings** **[likely: abstract; masking tool verified]**
  - Retrieval heads are **universal** and **sparse**: fewer than 5% of heads.
  - They are intrinsic, emerging from pretraining, and are dynamically activated.
  - When retrieval heads fire, the needle is retrieved. When they are partly or not activated, the model hallucinates.
  - Masking them breaks NIAH and hurts **CoT reasoning**, which must refer back to the question and earlier context. Tasks answered from **intrinsic (parametric) knowledge are less affected**.
  - Example top scores for Llama-2-7B-80K: head [16,19] 0.94, [11,15] 0.92, [8,26] 0.80 **[verified: README]**.
- **Implication**
  - Eviction that starves the few retrieval heads is what kills copy-from-context workloads.
  - Workloads answerable from parametric knowledge mask cache damage.

## B2. DuoAttention: retrieval vs streaming heads
- **Citation**: Guangxuan Xiao et al. ICLR 2025, arXiv 2410.10819. Repo: https://github.com/mit-han-lab/duo-attention **[verified]**.
- **Findings** **[verified: README]**
  - Only a fraction of heads, the retrieval heads, need full attention over all tokens. The others "primarily focus on recent tokens and attention sinks".
  - Needle-in-a-haystack parity with full attention needs a **25% retrieval-head ratio for MHA (Llama-2-7B) and 50% for GQA (Llama-3-8B)**.
  - Memory reduction is up to 2.55× (MHA) and 1.67× (GQA).
  - **[unverified]** Performance may plateau at 16 sink and 64 recent tokens.
- **Implication**: GQA models have less head-level redundancy, so budgets that work for MHA do not transfer. The RLKV README makes a similar point about KV group size.

## B3. MagicPIG: attention is not always sparse
- **Citation**: Zhuoming Chen et al. ICLR 2025 **[likely]**, arXiv 2410.16179. Repo: https://github.com/Infini-AI-Lab/MagicPIG. Evaluated on RULER with Llama-3.1-8B/70B and MegaBeam-Mistral-7B-512K **[verified: README]**.
- **Findings** **[likely: abstract snippets]**
  - TopK attention "preserves accuracy for retrieval tasks that only require a minimal subset of the context, but severely degrades for **aggregation tasks that leverage the full context** (common word extraction, frequent word extraction)".
  - Attention in these tasks is flatter, with less peaked scores.
  - Sampling with estimation guarantees beats TopK. MagicPIG uses about 2–3% of tokens.
  - Exact CWE/FWE numbers are **unknown**.
- **Implication**: "scope" (how much of the context the answer depends on) sets the attention entropy. Aggregation and counting workloads are adversarial for any top-k or heavy-hitter policy, even with a full cache available.

## B4. DynamicKV: task-dependent layer patterns
- **Citation**: Xiabin Zhou et al. EMNLP 2025 (the ACL Anthology id suggests Findings), arXiv 2412.14838. Repo: https://github.com/DreamMr/DynamicKV **[verified]**.
- **Findings**
  - **[verified: README]** "Different tasks (QA, summarization, code completion) exhibit distinct token importance distributions across transformer layers; fixed-pattern compression (pyramid, sliding window) fails to capture this."
  - **[verified]** About 90% of full-KV performance at **1.7%** retention.
  - **[verified: README]** LongBench at KV=512 (6.9% of context), Llama-3-8B: Full 41.95, StreamingLLM 34.70, H2O 37.20, SnapKV 40.30, PyramidKV 40.18, DynamicKV 40.73.
  - **[verified: README]** NIAH (32K, cache 64): Full 92%, StreamingLLM 26%, PyramidKV 72%, DynamicKV 83%.
  - **[likely]** Code completion shows a "wave-like" layer pattern. Synthetic and summarization tasks are "pyramid-like".

## B5. Attention sinks: StreamingLLM
- **Citation**: Guangxuan Xiao et al. ICLR 2024, arXiv 2309.17453. Repo: https://github.com/mit-han-lab/streaming-llm **[verified]**.
- **Findings** **[verified: README]**
  - Window attention fails once text exceeds the cache.
  - Keeping the initial tokens, the "sink", recovers most of it. Sinks exist "even if they are not semantically important". Four sink tokens suffice **[likely]**.
  - FAQ: StreamingLLM does not extend context. For a book, it "might only summarize the concluding paragraphs". It is aimed at streaming dialogue that depends on recent context.
- **Implication**
  - Recency-dominated workloads, such as chat continuation and code completion (Yuan et al.), tolerate sink+window policies.
  - Workloads whose evidence sits in the middle do not.

## B6. Lost in the Middle
- **Citation**: Nelson F. Liu et al. TACL 2024, arXiv 2307.03172. Repo: https://github.com/nelson-liu/lost-in-the-middle **[verified]**.
- **Findings** **[likely: multiple consistent snippets]**
  - U-shaped accuracy vs position of the relevant information in multi-document QA and key-value retrieval.
  - GPT-3.5-Turbo with the answer document in the **middle of 20 documents scores below its closed-book accuracy (56.1%)**.
- **Implication**
  - Evidence position (depth) is a first-order workload variable.
  - Parametric knowledge solves many NQ-style questions without context, which dilutes measured compression damage.

## B7. HELMET: synthetic tasks vs downstream
- **Citation**: Howard Yen et al. ICLR 2025, arXiv 2410.02694. Repo: https://github.com/princeton-nlp/HELMET **[verified]**.
- **Findings** **[likely: abstract]**
  - (1) Synthetic tasks like NIAH are poor predictors of downstream performance.
  - (2) The categories show distinct trends and low correlation with each other.
  - (3) Open models lag on full-context reasoning and complex instructions.
  - RAG correlates best with downstream tasks, and harder recall (RULER MK, JSON KV) is more informative than vanilla NIAH.
- **Category correlation matrix** **[verified: repo figure `assets/task_correlation.png`]**. This is correlation across models, not across compression methods.
  - Recall vs RAG 0.87, Re-rank 0.86, LongQA 0.84, Summ 0.87, Cite 0.74, ICL 0.63.
  - **ICL is the least correlated with everything** (0.36–0.63).
  - Cite is moderately distinct (0.74–0.80).
- **Implication**: a taxonomy needs several task families, with ICL and citation separate from recall/QA. Whether these correlations hold for compression sensitivity is **unknown**.

## B8. Goldman et al., "Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP"
- **Citation**: Omer Goldman, Alon Jacovi, Aviv Slobodkin, Aviya Maimon, Ido Dagan, Reut Tsarfaty. EMNLP 2024 (main), arXiv 2407.00402 **[likely: ACL Anthology listing]**.
- **Findings** **[likely]**
  - Two orthogonal difficulty axes:
    - **Diffusion**, called "dispersion" in later versions: how hard it is to find the needed information, including obscurity, sparsity and redundancy of indicators.
    - **Scope**: how much information is needed.
  - The high-diffusion, long-scope quadrant is "severely under-explored".
- **Implication**: this is a natural two-axis frame for the workload taxonomy.
  - Low-scope, low-diffusion tasks (NIAH) are sensitive only to whether a few tokens survive.
  - High-scope tasks (aggregation, summarization) stress breadth. They are robust to eviction when the context is redundant (KVzip "Redundancy" row) and fragile when it is not (CWE/FWE, per MagicPIG).

## B9. RULER task families (reference, used by the kvpress leaderboard)
- Hsieh et al., COLM 2024 (venue from memory), arXiv 2404.06654. Repo NVIDIA/RULER **[verified: README]**.
- 13 tasks in 4 categories:
  - Retrieval: NIAH variants with words, 7-digit numbers or **32-digit UUIDs**, multi-key, multi-value and multi-query.
  - Multi-hop tracing: variable tracking.
  - Aggregation: common-words and frequent-words extraction.
  - QA.
- "Despite nearly perfect performance on vanilla NIAH, most models exhibit large degradation" on the other tasks as length grows.

## B10. Is token importance stable over decoding, or does it drift?
- **Stability or persistence camp**
  - H2O (Zhang et al.; NeurIPS 2023, venue from memory). **[verified: README]** A small set of heavy-hitter tokens "contributes most of the value" in attention. Their emergence "strongly correlates with the frequent co-occurrence of tokens". Removing them degrades performance.
  - Scissorhands' "persistence of importance" hypothesis. **[unverified: background knowledge]**
  - SnapKV (NeurIPS 2024). **[unverified: background knowledge of abstract]** Heads "consistently focus on specific prompt attention features during generation", detectable from an observation window at the end of the prompt. This mostly holds for long-prompt, short-answer tasks.
- **Drift or recurrence camp**
  - Quest (ICML 2024). **[verified: README]** "The criticality of a token highly depends on the query", so importance must be re-estimated for each decode step and query.
  - InfiniGen (OSDI 2024). **[verified: README abstract; drift claim likely]** Speculates the important tokens for the next layer each step. Attention patterns change across decoding iterations, and one static retained set loses accuracy.
  - SCOPE: **[verified]** "deviation of heavy hitters" in long-output reasoning.
  - LazyEviction: **[verified]** token importance recurrence.
  - MorphKV: **[verified]** relevant context shifts during generation.
  - R-KV: **[verified]** redundant self-checks attract attention to themselves, so attention-only scoring keeps repeated content.
  - Hold Onto That Thought: **[verified]** H2O's cumulative scores work well for reasoning models. This suggests cumulative heavy hitters approximate recurrence for CoT, but not for every dataset.
- **Synthesis**: stability is a property of the workload, not of the model.
  - It holds for single-question, short-answer tasks over a long context.
  - It breaks for long generation, multi-turn follow-ups and new queries.

---

# PART C: Evidence-backed claims for the taxonomy (with supporting studies)

1. **Exact retrieval of high-entropy, incompressible spans is the most eviction-fragile workload, especially query-agnostic.**
   - Retr.KV, prefix-suffix, UUID, passkey and needle tasks collapse at moderate budgets when the query is hidden.
   - KVzip figure: SnapKV about 9/60 on Retr.KV at 50%. SCBench string retrieval. KVDiagnosis: StreamingLLM 57.2 vs 97.1 on RULER-8K at 50%. DynamicKV NIAH at cache 64: StreamingLLM 26% vs full 92%.
2. **Answer-span length is a hidden sensitivity parameter.**
   - Single-token or short answers get a free full-cache first step, while long verbatim spans expose eviction.
   - Yuan et al.: H2O is 100% on 7-digit passkeys but 35% on 64-digit ones. kvpress PR #292: needles are found but miscopied, because the question attends to the span start only.
3. **Query visibility changes absolute scores and method rankings.**
   - Query-aware prefill compression breaks on follow-up turns and independent requests.
   - kvpress FAQ; KVzip (degradation even at a 90% budget in multi-query); SCBench (multi-turn infeasibility, follow-ups); Query-Visibility audit 2026 (SnapKV loses to trivial baselines; only KeyDiff wins in 31/36 cells).
4. **Redundant global tasks are robust to eviction; non-redundant aggregation is not.**
   - Summarization, many-shot ICL, Math.Find and variable tracking degrade only gradually (KVzip "Redundancy" row). SnapKV stays within a few points of full down to 30–50% retention even when query-agnostic.
   - Counting and frequency aggregation has dense, flat attention and defeats top-k (MagicPIG; RULER aggregation; Goldman's scope axis).
5. **Arithmetic and code contexts behave like retrieval: every token (number or identifier) matters.**
   - KVFundaBench: arithmetic drops 17.4–43.3%, while knowledge and commonsense are resilient. KVzip: GSM8K about 32/69 at 50%. The code-generation fragility found by KVFundaBench contrasts with code-completion robustness under local retention (Yuan et al.).
6. **Parametric-knowledge solvability masks compression damage.**
   - Retrieval-head masking barely hurts knowledge-driven answers (Wu et al.).
   - Closed-book GPT-3.5 at 56.1% beats a mid-context open-book placement (Lost in the Middle).
   - MMLU and CSQA are robust while context-dependent tasks are not (KVFundaBench).
7. **Evidence position and recency structure matter.**
   - Sinks and recent windows are nearly free to keep, while mid-context evidence is both least used and first evicted by sink+window policies (StreamingLLM, Lost in the Middle, DuoAttention streaming heads).
   - Code completion is local and tolerates dropping (Yuan et al.).
8. **Long-output and reasoning workloads need decode-time policies, because importance drifts or recurs.**
   - Prefill-only compression does not bound memory there.
   - SCOPE, LazyEviction, MorphKV, R-KV, RLKV, Hold Onto That Thought, SCBench's long-generation "distribution shift".
9. **Compression changes output length, typically adding a heavy tail of runaway or looping outputs.** Accuracy per token of memory is therefore misleading.
   - Rethinking-KV: over 20% of samples are at least 1.5× longer. In my re-analysis, aggressive settings push 15–28% of naturally-terminating responses to the length cap, and looping outputs rise from 174 to 204–310 per 1,000.
   - Hold Onto That Thought: longer traces at low budgets. RLKV: some heads control termination.
10. **Prompts with multiple instructions or constraints are a distinct fragile class.** Eviction bias toward the scoring window's own span drops earlier constraints such as system prompts and defenses, which causes ignored instructions and leakage (Pitfalls, ACL 2026). Short prompts are more vulnerable (KVFundaBench).
11. **Task sensitivity interacts with the method family, and rankings flip across workloads.**
    - KVDiagnosis: KeyDiff scores 90.7 on RULER-8K but 51.5 on HotpotQA and 18.0 on Qasper (vs full 74.0 and 29.5).
    - Yuan et al.: token dropping is strong on code. Hold Onto That Thought: no single best press for Llama. kvpress: KeyDiff is good on Llama but weak on Qwen3 and Gemma3.
    - Chunk or semantic-unit methods help reasoning and few-shot workloads (ShotKV, ChunkKV).
12. **Attention structure is heterogeneous across heads and layers, and task-dependent.**
    - Fewer than 5% of heads are retrieval heads (Wu). 25–50% of heads need full KV for NIAH (DuoAttention). Reasoning-critical heads differ from retrieval heads (RLKV). Layer patterns vary by task (DynamicKV).
    - Uniform budgets misallocate. In KVDiagnosis, AdaKV's head-wise allocation is at least as good as plain SnapKV in 7 of 8 dataset×ratio cells at 25–50%; the exception is Qasper at 25%, 27.0 vs 27.5.
13. **Aggregate scores hide per-sample failures and non-overlapping failure sets.**
    - Rethinking-KV negative samples; KVDiagnosis (12,520 C→W vs 1,004 W→C; SnapKV–TOVA Jaccard 0.14–0.38); Expected Attention's TREC-outlier-driven averages; R-KV at 105% of full.
    - Pair each run with a full-cache control and report per-sample transitions.
14. **Synthetic NIAH alone is not representative.**
    - HELMET: synthetic tasks predict poorly, and category correlations are as low as 0.36 for ICL. RULER: NIAH saturates. Goldman: benchmarks cluster in the easy quadrant.
    - Use harder recall (multi-key, UUID, JSON-KV) plus RAG, QA, ICL, summarization, citation and reasoning.
15. **Systems caveat: nominal compression ratio poorly predicts end-to-end benefit.**
    - Kernel incompatibility (Rethinking-KV). Realized ratio varies with the workload for merging and threshold methods (2607.05399 CaM; KVzap/DMS). Quantization vs eviction "75/50/25%" settings are not byte-equivalent (KVDiagnosis).

# PART D: Gaps and unknowns worth resolving if access allows
- Full SCBench per-task tables by mode (multi-turn vs multi-request), and per-turn curves.
- Per-instruction-type degradation curves and leakage rates in the Pitfalls paper.
- MagicPIG's exact TopK accuracies on CWE and FWE.
- KVFundaBench per-task numbers and the exact compression ratios.
- The kvpress leaderboard's per-subtask RULER breakdown (NIAH-UUID vs VT vs CWE/FWE vs QA).
- Whether the HELMET category correlation also holds for compression sensitivity. No study found measures cross-task correlation of compression-induced deltas.
- Agentic studies: CommitKV, AgentKV, IntentKV and the online-compaction empirical study are unread. Their workload features (tool outputs, turn lifecycle) are likely relevant.
