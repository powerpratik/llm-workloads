> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# Foundational token-level KV-cache eviction / compression methods (2023 – mid-2024): evaluation workloads and reported task sensitivity

Part of: "A workload taxonomy for evaluating KV-cache eviction policies in LLM inference".
Compiled 2026-09-29.

## 0. How this was sourced (read first)

- arXiv / OpenReview / ACL Anthology / proceedings.* / Semantic Scholar / alphaxiv / github.io were all egress-blocked (both WebFetch and curl). Only `github.com` (WebFetch), `raw.githubusercontent.com` (curl) and PyPI were reachable.
- Evidence came from three kinds of source:
  1. **WebSearch result summaries.** These paraphrase or quote arXiv HTML/PDF, ACL Anthology or proceedings pages. About 45 queries were issued for this sub-task before the session-wide 200-search cap, shared with other agents, was exhausted; after that no more searches were possible.
  2. **Official GitHub repos.** READMEs plus the actual eval scripts and configs, e.g. the task lists in `figure11.py` or `bash_experiments/*.sh`.
  3. **Paper PDFs or slides hosted on GitHub.** These were parsed with PyMuPDF: the ArkVale NeurIPS'24 paper (`pku-liang/ArkVale/media/arkvale-nips24-paper.pdf`), the StreamingLLM slides, and the Quest poster and slides.
- Confidence tags:
  - **[verified]**: an official repo/script/PDF states it, or a search summary directly quotes or paraphrases the paper text with specifics.
  - **[likely]**: indirect evidence, a secondary summary, or partial sources.
  - **[unverified]**: from memory only, not confirmed. Such facts are kept out of the JSONL benchmark lists or explicitly flagged there.
- Numbers are only reported where a source gave them. "unknown" means not found.
- A caveat for the whole field: many 2023 papers (H2O, VATP, parts of CaM/D2O/InfiniGen) evaluate lm-eval-harness tasks by **masking attention** ("simulation") rather than physically dropping KV. VATP's README says so explicitly. Budgets are also defined inconsistently: some as a % of prompt length, some as absolute tokens per layer or head, some as a fraction of all tokens.

---

## 1. StreamingLLM (attention sinks)

- **Canonical:** "Efficient Streaming Language Models with Attention Sinks". Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. 2023 (arXiv), **ICLR 2024**. arXiv **2309.17453**. Repo: https://github.com/mit-han-lab/streaming-llm [verified: README, slides, search]
- **Mechanism:** keeps the KV of the first few tokens ("attention sinks", 4 by default) plus a rolling window of recent tokens. Position-based FIFO eviction of everything in between; positions are re-assigned within the cache. [verified: slides p.15-16]
- **Phase / query-awareness:** continuous rolling eviction while streaming input and output (both). Query-agnostic. [verified]
- **Workloads:**
  - **PG19 language modelling:** perplexity on the concatenated PG19 test set (100 books), streaming up to 4M tokens. [verified]
    - Models: Llama-2-7/13/70B, MPT-7/30B, Falcon-7/40B, Pythia-2.9/6.9/12B.
    - Cache 2048 for Llama-2 and 1024 for Falcon/Pythia/MPT.
    - Ablations on the number of sink tokens: 4 is "generally enough". [verified: slides p.20]
  - **Streaming QA:** ARC-Challenge and ARC-Easy Q/A pairs concatenated into a continuous stream, fed to Llama-2-7/13/70B-Chat, scored by exact match at each answer position. [verified]
  - **StreamEval:** the authors' own benchmark, inspired by LongEval line-retrieval. The model is queried every 10 lines of new information. The repo TODO still lists "Release StreamEval dataset". [verified]
  - **LongBench (appendix):** Llama-2-7B-chat (4K) on NarrativeQA, Qasper, HotpotQA, 2WikiMQA, GovReport and MultiNews, against a truncation baseline (first 1750 + last 1750 tokens). [verified: search summary of arXiv v3/ICLR supplementary]
  - **Context-extension combination:** LongChat-7B-v1.5-32K and Llama-2-7B-32K-Instruct (paper Fig. 9). [verified: README FAQ]
  - **Demo only:** the chatbot demo streams MT-Bench questions (`run_streaming_llama.py` downloads FastChat `mt_bench/question.jsonl`). This is not a scored evaluation. [verified: repo code]
  - Efficiency: up to 22.2x speedup over sliding window with re-computation. [verified]
- **Task sensitivity:**
  - **Evicting the initial tokens is catastrophic for language modelling.** Llama-2-13B on the first PG19 book (65K tokens): window attention (0+1024) gives PPL 5158, while 4 sinks + 1020 recent gives PPL 5.40. Sliding window with re-computation gives 5.43. [verified: slides + search]
  - **StreamingLLM does not extend the context window or give long-term memory.** The FAQ says: "if a book is an input, StreamingLLM might only summarize the concluding paragraphs". It is "not suitable for tasks that demand long-term memory and extensive data dependency, such as long document QA and summarization". [verified: README FAQ + search]
  - **On LongBench the 4+3496 configuration underperforms truncation,** "likely due to the loss of crucial initial input prompt information". Raising sinks to 1750 (1750+1750) only restores truncation-level performance. Instruction and preamble retention matters. [verified]
  - It works well in streaming QA and dialog, where answers depend on recent context. [verified]
- **Sources:** https://github.com/mit-han-lab/streaming-llm ; https://raw.githubusercontent.com/mit-han-lab/streaming-llm/main/assets/StreamingLLM.pdf ; https://arxiv.org/abs/2309.17453 ; https://arxiv.org/html/2309.17453v3 ; https://proceedings.iclr.cc/paper_files/paper/2024/file/5e5fd18f863cbe6d8ae392a93fd271c9-Supplementary-Conference.pdf
- **Overall confidence:** verified.

## 2. H2O (Heavy-Hitter Oracle)

- **Canonical:** "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models". Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen. **NeurIPS 2023**. arXiv **2306.14048**. Repo: https://github.com/FMInference/H2O (`h2o_hf` for accuracy, `h2o_flexgen` for throughput). [verified]
- **Mechanism:** keeps "heavy hitters", i.e. tokens with the largest *accumulated* attention scores, plus a window of recent tokens. Greedy eviction, framed as a dynamic submodular problem. Default budget is 20% (10% heavy + 10% recent) of prompt length. [verified: README, scripts `--heavy_ratio 0.1 --recent_ratio 0.1`]
- **Phase / query-awareness:** both. The prompt KV is reduced to the budget using accumulated attention, then per-step eviction continues during decoding. Query-agnostic (accumulated history). [verified; phase characterization likely]
- **Workloads:**
  - Eight tasks from two frameworks. [verified]
    - **lm-eval-harness:** COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande.
    - **HELM:** XSUM and CNN/DailyMail, 1000 test samples.
  - Few-shot, e.g. the repo example is 5-shot OpenBookQA. [verified: repo]
  - Models: OPT-6.7B up to OPT-175B, LLaMA (7B in scripts, 13B in the text-generation example) and GPT-NeoX-20B. [verified: abstract + `scripts/lm_eval/experiments.sh`, `scripts/helm/experiments.sh`]
  - KV budgets: 4%, 10%, 20%, 60% of prompt length. [likely: search summary]
  - Throughput with FlexGen / DeepSpeed / Accelerate on OPT-6.7B/30B: up to 29x/29x/3x. [verified]
  - Appendix content beyond these 8 tasks is **unknown** (unverified).
- **Task sensitivity:**
  - Heavy hitters "strongly correlate with frequent co-occurrence", and "removing them results in significant performance degradation". [verified: abstract]
  - Recent-only ("Local") and static-sparsity baselines suffer large drops at low budgets (a search summary says up to ~37%). H2O holds near full-cache accuracy at 20%. [likely]
  - All tasks are short-context (<2K tokens) multiple-choice or short summarization. There are no long-context retrieval or multi-turn workloads in the original paper. [verified by task list]
- **Sources:** https://github.com/FMInference/H2O ; https://raw.githubusercontent.com/FMInference/H2O/main/h2o_hf/README.md ; https://raw.githubusercontent.com/FMInference/H2O/main/h2o_hf/scripts/lm_eval/experiments.sh ; https://arxiv.org/abs/2306.14048 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html
- **Overall confidence:** verified.

## 3. Scissorhands

- **Canonical:** "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time". Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava. **NeurIPS 2023**. arXiv **2305.17118**. Repo: https://github.com/lzcemma/Scissorhands (only the C4 perplexity code is released; "Fewshot evaluation code: coming soon"). [verified]
- **Mechanism:** the "persistence of importance" hypothesis: tokens that were pivotal (high attention) at one step stay pivotal later. The cache is held at a fixed budget. When it is full, tokens that were repeatedly low-attention within a history window are dropped, and recent tokens are protected. [verified: abstract; KVCache-Factory backlog describes "historical importance accumulation with fixed-budget pivotal-token selection"]
- **Phase / query-awareness:** decode-time. The budget is enforced as tokens are appended, including in perplexity streaming. Query-agnostic. [likely]
- **Workloads:** [verified: search summary quoting paper; repo]
  - C4 language-modelling perplexity. The repo has `run_infer_opt_66b_c4.sh` and `run_infer_opt_66b_sparse_c4.sh`, built on DS3Lab Decentralized_FM_alpha.
  - Few-shot downstream tasks: HellaSwag, MathQA, PIQA, Winogrande.
  - Models: OPT family, 6.7B/13B/30B/66B. The authors say compute limited them to OPT-66B. [likely]
  - Compression 2x to 5x. Combined with 4-bit weight quantization it reaches up to 20x. [verified]
- **Task sensitivity:**
  - "Up to 5x" KV reduction without quality loss. For OPT-66B, Winogrande and MathQA accuracy is maintained even at 5x. [verified]
  - Language-modelling perplexity tolerates less compression than few-shot accuracy. A search summary says perplexity is "maintained until 50% [OPT-13B] ... 75% [OPT-66B] of the original KV cache". The exact wording is ambiguous. [likely]
  - The accuracy-vs-compression curve is flatter for larger models, so the method "can scale with model size". [verified]
  - Per the cold-compress README, Scissorhands found that heavy hitters are more variable in higher layers, which calls for *larger* caches there. That is the opposite of PyramidKV's allocation. [likely: secondary]
- **Sources:** https://github.com/lzcemma/Scissorhands ; https://arxiv.org/abs/2305.17118 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html ; https://github.com/AnswerDotAI/cold-compress
- **Overall confidence:** verified for the benchmark list; model list and exact ratios are likely.

## 4. FastGen ("Model Tells You What to Discard")

- **Canonical:** "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs". Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, Jianfeng Gao. **ICLR 2024 (oral)**. arXiv **2310.01801**. Repo: https://github.com/machilusZ/FastGen. It contains **no code**: it points to the community reimplementation in AnswerDotAI/cold-compress and to Microsoft MInference as "a close implementation". [verified]
- **Mechanism:** lightweight per-head attention profiling on the prompt at the end of prefill. Each head is assigned a hybrid policy from a small portfolio:
  - special tokens only;
  - special tokens + punctuation;
  - local window;
  - frequency / heavy-hitter;
  - full cache.

  The cheapest policy that recovers a target fraction of attention mass is chosen. [verified: README "special tokens, local, topk...", search summaries; exact portfolio likely]
- **Phase / query-awareness:** both. Profiling happens at prefill, and the adaptive policy governs the cache during generation. Query-agnostic. [likely]
- **Workloads:** [verified: search summaries of arXiv HTML]
  - **AlpacaEval:** 805 prompts from diverse domains. Instruction-finetuned Llama 1 models; the metric is win rate against the full-cache model.
  - **"Standard generation tasks":** GSM8k (math), HumanEval (code), NQ (QA) and TQA ("reading comprehension"; probably TriviaQA; identity [likely]), on Llama 1 and fine-tuned Llama 1.
  - Models: Llama 1 7B/13B/30B/65B. [verified]
  - Budgets: KV cache budget swept from 30% to 100%. [verified]
- **Task sensitivity:**
  - **Model scale dominates compressibility.** At a 45% win rate, FastGen prunes 44.9% of the cache on Llama 1-65B but only 16.9% on 7B. With win rate above 45%, memory reduction is about 40% on 65B, about 30% on 30B and about 20% on 13B/7B. [verified]
  - On 30B, FastGen at 50% compression beats non-adaptive methods at 15% compression. [verified]
  - Heads that attend mostly to special tokens must never lose them ("anchors"). [likely: secondary summary]
  - The paper reports no long-context (>4K) workload. [likely]
- **Sources:** https://github.com/machilusZ/FastGen ; https://arxiv.org/abs/2310.01801 ; https://arxiv.org/html/2310.01801v4 ; https://proceedings.iclr.cc/paper_files/paper/2024/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html ; https://www.microsoft.com/en-us/research/blog/llm-profiling-guides-kv-cache-optimization/ ; https://github.com/AnswerDotAI/cold-compress
- **Overall confidence:** verified for benchmarks; policy details likely.

## 5. TOVA ("Transformers are Multi-State RNNs")

- **Canonical:** "Transformers are Multi-State RNNs". Matanel Oren, Michael Hassid, (Nir Yarden in the EMNLP version), Yossi Adi, Roy Schwartz. arXiv Jan 2024, **EMNLP 2024** (main; aclanthology 2024.emnlp-main.1043). arXiv **2401.06104**. Repo: https://github.com/schwartz-lab-NLP/TOVA (a `TOVACache` for LLaMA/Mistral, with no eval scripts). [verified]
- **Mechanism:** Token Omission Via Attention. When the fixed-size "multi-state" is full, drop the token with the lowest attention score from the *current* query, averaged over heads. It is training-free. [verified]
- **Phase / query-awareness:** per-step eviction as each token is processed, so it applies to both incremental input processing and generation. It uses only the current token's attention, with no knowledge of future queries, so it is query-agnostic. [verified/likely]
- **Workloads:** [verified: search summaries]
  - PG-19 perplexity.
  - Long-range understanding: **SQuALITY** (query-based summarization, ROUGE) and **QASPER** (QA over papers). These were likely taken from ZeroSCROLLS [likely].
  - Long **story generation**. The full model's average story length is 1566 tokens. Judging protocol (GPT-4?) is [unverified].
  - Models: LLaMA-2-7B, Mistral-7B, Yi-6B. The base vs chat variants are [unverified].
  - Multi-state sizes run from small (1/8 of context) to full. The exact grid 64 to 4096 is [unverified].
  - Baselines: window, window+sinks, and H2O [likely].
- **Task sensitivity:** [verified]
  - **Language modelling is robust:** within 0.4 PPL of topline at 1/8 of the context.
  - **Summarization (SQuALITY):** within 1 ROUGE at 1/4 (Mistral, Yi) or 1/8 (LLaMA-2) of context.
  - **QA (QASPER) is the most sensitive:** it "requires a half of the full multi-state size" to be comparable to topline.
  - Throughput reaches up to 4.8x.
  - Retained tokens are the first token, punctuation, possessive endings and proper nouns. The paper analyses tokens kept by part of speech.
- **Sources:** https://github.com/schwartz-lab-NLP/TOVA ; https://arxiv.org/abs/2401.06104 ; https://aclanthology.org/2024.emnlp-main.1043/
- **Overall confidence:** verified.

## 6. Keyformer

- **Canonical:** "Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference". Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J. Nair, Ilya Soloveychik, Purushotham Kamath (UBC and d-Matrix). **MLSys 2024**. arXiv **2403.09054**. Repo: https://github.com/d-matrix-ai/keyformer-llm [verified]
- **Mechanism:** keeps a recent window w plus k−w "key tokens". Key tokens are scored by a Gumbel-noise-regularized softmax of the unnormalized logits with a temperature schedule (tau_init 1 to tau_end 2), accumulated over steps, and biased toward initial tokens. The cache is held at a constant k. [verified: blog README]
- **Phase / query-awareness:** both. The prompt KV is reduced to k after prompt processing, and per-step discards keep it at k during generation. Query-agnostic. [verified: blog Fig. 2]
- **Workloads:** [verified: `blog/README.md`, `summarization/`, `conversation/` READMEs]
  - **Summarization:** CNN/DailyMail from HELM. XSUM and GovReport have download scripts.
  - **Long-context summarization:** GovReport with MPT-7B-storywriter at 8K sequence length.
  - **Conversation:** SODA with MPT-7B-chat.
  - **lm-eval-harness few-shot:** PIQA, Winogrande, OpenBookQA, COPA [likely: search summary]. The repo example is OpenBookQA 0-shot.
  - Models: GPT-J-6B (fine-tuned for summarization), Cerebras-GPT-6.7B, MPT-7B, plus the MPT-chat and storywriter variants. [verified]
  - Budgets: KV cache as a % of prompt length, swept. Example defaults are `--kv_cache 60 --recent 30` with 128 new tokens and beam 4. [verified]
- **Task sensitivity:** [verified: blog]
  - Keyformer "achieves the baseline accuracy with 70% prompt KV cache size for Summarization ... while 90% of prompt KV cache for Conversation task". Other baselines (window attention, H2O) did not reach baseline accuracy. **Dialogue is more eviction-sensitive than summarization.**
  - Long-document summarization (GovReport, 8K) reaches baseline accuracy at 50% of the prompt KV, with a 2.1x latency improvement and 2.4x throughput.
- **Sources:** https://github.com/d-matrix-ai/keyformer-llm ; https://raw.githubusercontent.com/d-matrix-ai/keyformer-llm/main/blog/README.md ; https://arxiv.org/abs/2403.09054 ; https://proceedings.mlsys.org/paper_files/paper/2024/hash/48fecef47b19fe501d27d338b6d52582-Abstract-Conference.html
- **Overall confidence:** verified (the lm-eval task list is likely).

## 7. SnapKV

- **Canonical:** "SnapKV: LLM Knows What You are Looking for Before Generation". Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen. arXiv Apr 2024, **NeurIPS 2024**. arXiv **2404.14469**. Repo: https://github.com/FasterDecoding/SnapKV [verified]
- **Mechanism:** an "observation window" of the last prompt tokens (window 32) votes, via attention, for important prefix positions per head. The votes are pooled (kernel 7, max-pool) to keep clusters, and the top positions plus the window are kept. The config name is `ablation_c4096_w32_k7_maxpool`. [verified: repo]
- **Phase / query-awareness:** **prefill-only**, a one-shot compression of the prompt KV; generated tokens are not compressed. It is **implicitly query-aware**, because the question or instruction usually sits in the observation window at the end of the prompt. [verified/likely]
- **Workloads:** [verified: repo `experiments/LongBench/pred_snap.py` + search summaries]
  - **LongBench, all 16 English datasets:** NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, GovReport, QMSum, MultiNews, TREC, TriviaQA, SAMSum, PassageCount, PassageRetrieval-en, LCC, RepoBench-P.
  - **Needle-in-a-Haystack** with LWM-Text-Chat-1M up to 380K tokens on one A100-80GB. The baseline hits OOM at about 33K.
  - **Command-R (35B, 128K) case study:** NIAH, RAG citation generation (keeps about 98.8% of performance, F1 −1.2%), and a lost-in-the-middle style position analysis. [verified]
  - Observation analyses used UltraChat (multi-turn; responses >512 tokens, prompts >3K) and QMSum. [verified]
  - Models: LWM-Text-Chat-1M, LongChat-7b-v1.5-32k, Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1, and Command-R. [verified]
  - Budgets: prompt KV compressed to 1024, 2048 or 4096 tokens. [verified]
- **Task sensitivity:**
  - At 1024 KV tokens (about 92% compression) performance is comparable to full KV across the 16 LongBench datasets. [verified]
  - Attention "features" chosen by the observation window stay consistent through generation. [verified]
  - The paper does not stress-test the case where **the question arrives after compression**, e.g. multi-turn follow-ups or a query placed before the context. Its design assumes the query is in the window. [inference; later work such as kvpress exposes `query_aware` as an explicit evaluation toggle — verified kvpress README]
- **Sources:** https://github.com/FasterDecoding/SnapKV ; https://raw.githubusercontent.com/FasterDecoding/SnapKV/main/experiments/LongBench/pred_snap.py ; https://arxiv.org/abs/2404.14469 ; https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html ; https://github.com/NVIDIA/kvpress
- **Overall confidence:** verified.

## 8. PyramidKV

- **Canonical:** "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling". Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, Wen Xiao. arXiv **2406.02069** (June 2024; later versions exist up to at least v4). An OpenReview forum exists (id jZVNmDiU86). The peer-reviewed venue is **unknown**. Repo: https://github.com/Zefan-Cai/PyramidKV, renamed **KVCache-Factory** on 2024-11-28. [verified]
- **Mechanism:** SnapKV-style observation-window selection, but the per-layer budget is **pyramidal**: larger in lower layers, where attention is scattered, and smaller in higher layers, where it is concentrated ("information funneling"). The total budget is fixed. [verified]
- **Phase / query-awareness:** prefill-only. Implicitly query-aware through the observation window. [verified/likely]
- **Workloads:**
  - LongBench. The search summary says "17 datasets"; the repo runner defaults to the 16-dataset English list. [verified]
  - Needle-in-a-Haystack. [verified: README fig and runner]
  - Models: LLaMA-3-8B-Instruct and Mistral-7B-Instruct(-v0.2). LLaMA-3-70B-Instruct is used for NIAH ("retaining just 128 KV cache entries enables LLaMA-3-70B to achieve 100% Acc"). [verified]
  - Budgets: "the PyramidKV paper reports results at budgets of 128 and 2048". The quickstart uses 128 and NIAH uses 96. [verified: README]
  - RULER support was added to the repo later. It is not part of the paper. [verified]
- **Task sensitivity:** [verified]
  - Gains over uniform-budget methods are largest at **small budgets** (KV=128).
  - At 0.7% retention PyramidKV is up to +20.5 absolute accuracy on **TREC** (few-shot classification over in-context examples). This implies TREC collapses under uniform tiny budgets.
  - It matches full KV "while retaining only 12% of the KV cache".
- **Sources:** https://github.com/Zefan-Cai/KVCache-Factory ; https://raw.githubusercontent.com/Zefan-Cai/PyramidKV/main/README.md ; https://arxiv.org/abs/2406.02069 ; https://openreview.net/forum?id=jZVNmDiU86
- **Overall confidence:** verified (venue unknown).

## 9. PyramidInfer

- **Canonical:** "PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference". Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, Hai Zhao. **Findings of ACL 2024** (2024.findings-acl.195). arXiv **2405.12532**. Repo: https://github.com/mutonix/pyramidinfer. Still marked "[WIP]"; the OpenCompass eval code is unreleased. [verified]
- **Mechanism:** layer-wise decaying retention of "pivotal contexts" (PvCs). Attention from the most recent tokens (a `recent_ratio` window, distance-weighted) selects the keys and values to keep, and fewer are kept in deeper layers (`prefill_decay_ratio`, linear or cosine). This reduces the KV that is *computed* in prefill. During generation it compresses the extra generated tokens once they exceed a threshold, but "we do not compress the prompt kv from the prefilling stage". [verified: README]
- **Phase / query-awareness:** both. The recent-window proxy means it is implicitly query-aware when the question is at the end. [verified/likely]
- **Workloads:** [verified: search summary of paper; configs]
  1. Language modelling on wikitext-2.
  2. LLM benchmarks: MMLU and BBH (understanding), GSM8K (math), HumanEval (code).
  3. Conversation: MT-Bench.
  4. Long context: LEval (long-text summarization).
  - Evaluated via OpenCompass.
  - Models: LLaMA 2, LLaMA 2-Chat, Vicuna 1.5-16k, CodeLLaMA, in 7B/13B/34B/70B sizes. The repo configs cover llama2_7b/13b/70b and llama3_8b/70b.
  - Budget: "over 54% GPU memory reduction in KV cache". Throughput is 2.2x vs Accelerate, 1.4x vs DeepSpeed and 2.4x vs H2O. [verified]
- **Task sensitivity:** no per-task statements recovered. **unknown**
- **Sources:** https://github.com/mutonix/pyramidinfer ; https://arxiv.org/abs/2405.12532 ; https://aclanthology.org/2024.findings-acl.195/
- **Overall confidence:** verified (benchmarks); findings unknown.

## 10. NACL

- **Canonical:** "NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time". Yilong Chen, Guoxia Wang, Junyuan Shang, Shiyao Cui, Zhenyu Zhang, Tingwen Liu, Shuohuan Wang, Yu Sun, Dianhai Yu, Hua Wu (Baidu / CAS). **ACL 2024** (long, pp. 7913–7926; 2024.acl-long.428). arXiv **2408.03675**. Repo: https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL [verified]
- **Mechanism:** a single "global optimal" eviction during the encoding phase, applied progressively layer by layer and head-wise. It combines:
  - **Proxy-Tokens Eviction:** attention statistics from a small set of task-specific proxy tokens, e.g. the question.
  - **Random Eviction:** head-wise sampling from the score distribution. This counters attention-score bias and improves robustness.

  [verified: README + abstract]
- **Phase / query-awareness:** prefill-only (encoding-time). Query-aware through the proxy tokens. [verified]
- **Workloads:**
  - Paper: "short- and long-text tasks". The paper criticises prior work for relying "on perplexity on inadequate short-text evaluation". The exact benchmark lists are **unknown**: LongBench [likely/unverified], lm-eval short tasks [unverified].
  - Repo (post-paper release): **InfiniteBench-128K**, all 12 tasks, with Llama-3.1-8B-Instruct at **80% eviction**. [verified]
- **Task sensitivity:**
  - Paper: improves short- and long-text tasks by 80% and 76% respectively. Up to 5x KV reduction with over 95% of performance maintained. [verified]
  - Repo InfiniteBench table (Llama-3.1-8B, full vs NACL with 80% of the KV evicted): [verified]

    | Task | Full | NACL (80% evicted) |
    |---|---|---|
    | Retrieve.PassKey | 1.000 | 1.000 |
    | Retrieve.Number | 0.9949 | 0.9661 |
    | **Retrieve.KV** | **0.592** | **0.036** |
    | En.Sum | 0.2761 | 0.2653 |
    | En.QA | 0.1303 | 0.1441 |
    | En.MC | 0.6637 | 0.6638 |
    | En.Dia | 0.170 | 0.155 |
    | Math.Find | 0.3285 | 0.3286 |

    Averages: 0.391 vs 0.338 overall, and 0.371 vs 0.368 without Retrieve.KV.
  - Quote: "Retrieve.KV is one of the most challenging tasks for KV cache eviction-based methods under an extremely low KV cache eviction budget, such as 20%." Exact-match retrieval of high-entropy key-value strings is the worst case for eviction; passkey is easy.
- **Sources:** https://raw.githubusercontent.com/PaddlePaddle/Research/master/NLP/ACL2024-NACL/README.md ; https://aclanthology.org/2024.acl-long.428/ ; https://arxiv.org/abs/2408.03675
- **Overall confidence:** likely. The paper's benchmark list is not confirmed; the repo results are verified.

## 11. L2-norm KV compression (Devoto et al.)

- **Canonical:** "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression". Alessio Devoto, Yu Zhao, Simone Scardapane, Pasquale Minervini. **EMNLP 2024** (main, pp. 18476–18499; 2024.emnlp-main.1027). arXiv **2406.11430**. Repo: https://github.com/alessiodevoto/l2compress. The method was later also added to NVIDIA kvpress. [verified]
- **Mechanism:** keeps tokens whose **key** embeddings have the lowest L2 norm, since low key norm correlates with high attention. It needs no attention scores, so it is FlashAttention-compatible. Parameters: `keep_ratio`, `prune_after` (a token threshold), and `skip_layers` (e.g. 0,1 or 0,1,12). [verified]
- **Phase / query-awareness:** it can be applied to the cache after any forward pass, so prefill and decode (both). Query-agnostic. [verified]
- **Workloads:** [verified: repo]
  - Language modelling on Wikipedia (20220301.en chunks) with Llama-3-8B (default), Llama-2-7B and Gemma.
  - Needle-in-a-Haystack and passkey retrieval with long-context Llama-2-7B variants (LongLoRA-32k-ft, Llama-2-7B-80k), up to 32K tokens.
  - Keep ratios swept 0.1 to 0.9.
- **Task sensitivity:** [verified]
  - It reduces the KV cache "by 50% on language modelling and needle-in-a-haystack tasks and 90% on passkey retrieval tasks without losing accuracy".
  - So NIAH (a natural-text needle) is more sensitive than synthetic passkey (a number in repetitive filler).
- **Sources:** https://github.com/alessiodevoto/l2compress ; https://raw.githubusercontent.com/alessiodevoto/l2compress/main/scripts/needle_loop.sh ; https://arxiv.org/abs/2406.11430 ; https://aclanthology.org/2024.emnlp-main.1027/
- **Overall confidence:** verified.

## 12. SirLLM

- **Canonical:** "SirLLM: Streaming Infinite Retentive LLM". Yao Yao, Zuchao Li, Hai Zhao. **ACL 2024** (long; 2024.acl-long.143). arXiv **2405.12528**. Repo: https://github.com/Zoeyyao27/SirLLM, built on the StreamingLLM code. [verified]
- **Mechanism:** a StreamingLLM cache (4 sink tokens) plus retention of high **token-entropy** tokens (key phrases). A **memory decay** ratio (`--decay_ratio` 0.7 to 1.0) progressively forgets older retained tokens. [verified]
- **Phase / query-awareness:** streaming, multi-turn dialogue: inputs and outputs are continuously trimmed. Query-agnostic (entropy-based). [verified]
- **Workloads:** three datasets constructed for the paper. [verified]
  1. **DailyDialog**, multi-turn chit-chat, with turns included.
  2. **Grocery Shopping:** remember items mentioned early in a long dialogue.
  3. **Rock-Paper-Scissors:** tracks the user's move preferences over many rounds.
  - Models: the repo examples use Yi-6B-Chat [verified]; other models are **unknown**.
  - Budget: `token_entropy_size` 508 or 1020 plus 4 sinks, i.e. about a 1K cache. [verified: repo commands]
- **Task sensitivity:** StreamingLLM-style streaming "significantly impair[s] the model's long-term memory capabilities by losing information from earlier parts of the conversation". SirLLM improves memory of early facts across tasks and models. These are the only workloads in this set that explicitly test **cross-turn recall of early facts**. [verified]
- **Sources:** https://github.com/Zoeyyao27/SirLLM ; https://arxiv.org/abs/2405.12528 ; https://aclanthology.org/2024.acl-long.143/
- **Overall confidence:** verified (the model list is partial).

## 13. D2O (Dynamic Discriminative Operations)

- **Canonical:** "D2O: Dynamic Discriminative Operations for Efficient (Generative →) Long-Context Inference of Large Language Models". The v1 title said "Generative Inference". Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong, (Longyue Wang,) Mi Zhang. arXiv **2406.13035** (June 2024). **ICLR 2025**. Repo: https://github.com/AIoT-MLSys-Lab/D2O [verified]
- **Mechanism:**
  - **Layer level:** a dynamic budget allocation driven by attention density, so shallow and deep layers get different eviction.
  - **Token level:** H2O-style eviction plus a compensation mechanism. An EMA similarity threshold re-discriminates evicted tokens and **merges** them back into similar retained tokens.

  [verified]
- **Phase / query-awareness:** both (prompt budget ratio + decoding). Query-agnostic. [likely]
- **Workloads:** [verified: search summary + repo `LLM_merge_new/bash_experiments/*`]
  - **LongBench.** Five models: Falcon-7B, Mistral-7B, Llama-2-7B, Llama-2-13B, Llama-3-8B. Default budget ratio ρ=0.2 of prompt length.
  - lm-eval **generation** tasks: GSM8K, CoQA (EM), TruthfulQA (gen, BLEU). Scripts sweep budget ratios 0.2/0.4/0.6/0.8, D2O ("merge") vs H2O.
  - Needle-in-a-Haystack.
  - The repo also contains PIQA and OpenBookQA-5shot classification runners and a HELM summarization runner. Their use in the paper is [likely].
- **Task sensitivity:** [verified: search summary]
  - Gains are largest on **reasoning / multi-step generation** (GSM8K, CoQA) for Llama-3-8B.
  - On TruthfulQA, D2O even *exceeds* full cache across Llama backbones and most budget ratios. The authors interpret this as pruning irrelevant context.
- **Sources:** https://github.com/AIoT-MLSys-Lab/D2O ; https://arxiv.org/abs/2406.13035 ; https://openreview.net/forum?id=HzBfoUdjHt
- **Overall confidence:** verified.

## 14. CaM (Cache Merging)

- **Canonical:** "CaM: Cache Merging for Memory-efficient LLMs Inference". Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, Rongrong Ji. **ICML 2024** (PMLR v235, zhang24n). arXiv: none found (**unknown**). Repo: https://github.com/zyxxmu/cam [verified]
- **Mechanism:** instead of discarding to-be-evicted KV, it adaptively **merges** them (values) into the retained cache, using a sampling strategy governed by the prominence of attention scores at the discarded positions. It is layered on H2O/StreamingLLM-style eviction. [verified: search summary + kvpress `CAMPress` "decoding press that merges the kv cache of evicted tokens into keep tokens"]
- **Phase / query-awareness:** decode-time. Query-agnostic. [verified via kvpress description]
- **Workloads:** [verified in repo; the paper-level exact set is likely]
  - lm-eval-harness QA/MC: OpenBookQA, MathQA, BoolQ, COPA, Winogrande "...".
  - HELM-style summarization: XSUM, CNN/DailyMail, MultiNews.
  - Long-generation perplexity: WikiText, PG-19.
  - Models: LLaMA-7B (huggyllama), OPT, GPT-NeoX.
  - Budgets: `start_ratio` and `recent_ratio` of 0.1 to 0.2; perplexity runs use 32 start + 32 recent.
- **Task sensitivity:** not recovered. **unknown**
- **Sources:** https://github.com/zyxxmu/cam ; https://proceedings.mlr.press/v235/zhang24n.html ; https://openreview.net/forum?id=LCTmppB165 ; https://github.com/NVIDIA/kvpress
- **Overall confidence:** likely.

## 15. InfiniPot

- **Canonical:** "InfiniPot: Infinite Context Processing on Memory-Constrained LLMs". Minsoo Kim, Kyuhong Shim, Jungwook Choi, Simyung Chang. **EMNLP 2024** (main; 2024.emnlp-main.897). arXiv **2410.01518** (Oct 2024). Repo: none found (**unknown**). [verified]
- **Mechanism:** Continual Context Distillation (CCD). A long input is streamed into a fixed-size KV "pot", and whenever the pot overflows it is compressed. Importance comes from two signals:
  - a **Catalyst Prompt (CaP):** a generic auxiliary prompt such as "Summarize the critical points", used only for scoring;
  - **Novelty under Compression (NuC).**

  It needs no training and works without the future query. [verified: search summary]
- **Phase / query-awareness:** prefill (chunked, iterative compression of the input before answering). Query-agnostic with respect to the actual user question; CaP is a generic proxy. [likely]
- **Workloads:** [likely: search summary with specific numbers]
  - **LongBench** under a 4K memory constraint. Example: "M3-InfiniPot-4K" scores 44.22 vs GPT-3.5-16K 43.74 and M3-PT-32K 47.84.
  - Baselines in the memory-constrained setting: StreamingLLM 29.87, H2O 31.01, SirLLM 35.43, TOVA 36.26, truncation 31.24.
  - **NIH / passkey** from 4K to 1M context at depths 0.1/0.5/0.9, with a 4K pot.
  - Models: LLaMA and Mistral families. Exact versions and the meaning of "M3" are **unknown**.
- **Task sensitivity:**
  - Streaming (StreamingLLM) and accumulated-attention (H2O) policies lose most in the memory-constrained long-input setting.
  - Methods with better importance estimates (SirLLM, TOVA, InfiniPot) retain more.
  - Passkey retrieval can be kept up to 1M tokens with a 4K pot.

  [likely]
- **Sources:** https://arxiv.org/abs/2410.01518 ; https://aclanthology.org/2024.emnlp-main.897/
- **Overall confidence:** likely.

## 16. Quest (query-aware sparsity; non-evicting contrast)

- **Canonical:** "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference". Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. **ICML 2024**. arXiv **2406.10774**. Repo: https://github.com/mit-han-lab/Quest [verified]
- **Mechanism:** keeps the **full** KV cache, organised in pages. Each page stores the element-wise min and max of its keys. At each decode step the current query scores each page with an upper bound on q·k, and attention is computed only over the top-K pages (the token budget). **Nothing is permanently evicted.** [verified: README, poster]
- **Phase / query-awareness:** decode-time. Query-aware (per step). Non-evicting; it saves memory bandwidth, not capacity. [verified]
- **Workloads:** [verified: repo scripts, slides, search]
  - **Passkey retrieval:** 10K context on LongChat-7B-v1.5-32K and 100K on Yarn-Llama-2-7B-128K. A later repo version adds Llama-3.1-8B-Instruct at 100K with budgets 512 to 4096.
  - **LongBench, 6 tasks:** Qasper, NarrativeQA, HotpotQA, MultiFieldQA-en, GovReport, TriviaQA. Budgets 512/1024/2048/4096.
  - **PG-19 perplexity:** budget 4096, 30K tokens.
  - Baselines: H2O, TOVA, StreamingLLM. Kernels are compared against FlashInfer.
- **Task sensitivity:** [verified]
  - "KV cache eviction algorithms such as H2O, TOVA, and StreamingLLM incorrectly discard the KV cache of the answer before receiving the question". In the poster's example, query-agnostic eviction scores **Acc: 2%** vs 100% for dense attention and Quest.
  - Quest is near-perfect on passkey with 64 tokens (10K) and 1024 tokens (100K).
  - On LongBench, "Baselines need nearly full cache to achieve lossless performance". Quest is lossless at about 2K budget.
  - From the poster: "critical tokens depend on the input query. For e.g., summary task will attend on different paragraphs, sequentially" — i.e. summarization needs *moving* attention, not a static retained set.
- **Sources:** https://github.com/mit-han-lab/Quest ; https://raw.githubusercontent.com/mit-han-lab/Quest/main/scripts/longbench.sh ; https://raw.githubusercontent.com/mit-han-lab/Quest/main/assets/quest_poster.pdf ; https://raw.githubusercontent.com/mit-han-lab/Quest/main/assets/quest_slides.pdf ; https://arxiv.org/abs/2406.10774
- **Overall confidence:** verified.

## 17. InfiniGen

- **Canonical:** "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management". Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim (SNU). **OSDI 2024**. arXiv **2406.19707**. Repo: https://github.com/snu-comparch/InfiniGen [verified]
- **Mechanism:** an offloading-based design (built on FlexGen). The KV lives in CPU memory. For the *next* layer, important tokens are speculated by a minimal "rehearsal": the current layer's input times partial query weights times a skewed partial key cache (offline SVD-based skewing). Only those KV entries are prefetched. A CPU-side KV pool with FIFO, LRU or counter-based eviction handles memory limits. [verified: README + `table2.sh`]
- **Phase / query-awareness:** decode-time. Query-aware (speculated per step). Mostly non-evicting (selective fetch). [verified]
- **Workloads:** [verified: `accuracy/lm_eval/figure11.py`, `figure13.sh`, `perplexity/table2.sh`, `figure12.sh`]
  - **lm-eval-harness 5-shot:** PIQA, OpenBookQA, WinoGrande, COPA, RTE.
  - **Perplexity:** WikiText-2 and PTB (seq 2048). Also block-wise perplexity per 256-token decoding chunk on WikiText-2, with OPT-13B at 2048 and Llama-2-13B at 4096.
  - Models: OPT-6.7B, 13B, 30B and Llama-2-7B, 13B.
  - Baselines: H2O (e.g. 1.875% heavy + 1.875% recent in the perplexity test), quantization, full cache.
  - Speedup experiments use FlexGen with PG-19 text as input.
- **Task sensitivity:** H2O's perplexity **diverges increasingly from the full-cache baseline as decoding proceeds**, i.e. with more decoding chunks, while InfiniGen stays close. Permanent eviction errors accumulate over long generation. [verified: search summary + `figure12.sh` design]
- **Sources:** https://github.com/snu-comparch/InfiniGen ; https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/accuracy/lm_eval/figure11.py ; https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/accuracy/perplexity/table2.sh ; https://arxiv.org/abs/2406.19707 ; https://www.usenix.org/system/files/osdi24-lee.pdf
- **Overall confidence:** verified.

## 18. ArkVale

- **Canonical:** "ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction". Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei, Shengen Yan, Meng Li, Yun Liang (PKU). **NeurIPS 2024**. arXiv: **unknown** (the PDF is hosted in the repo). Repo: https://github.com/pku-liang/ArkVale [verified: paper PDF]
- **Mechanism:** a page-based KV manager in the style of vLLM. Filled pages are asynchronously backed up to CPU memory and summarized as a **digest**, a bounding volume of their keys; "cuboid-mean" works best. Before attention, every page's importance is estimated from its digest and the current query. Important evicted pages are **recalled**, unimportant ones evicted, and only the top pages are attended. [verified]
- **Phase / query-awareness:** decode-time. Query-aware. Evictions are recallable. [verified]
- **Workloads:** [verified: paper §6]
  - **LongBench, 6 datasets:** HotpotQA, NarrativeQA, Qasper, GovReport, TriviaQA, PassageRetrieval.
  - **Passkey retrieval** at 10K, 20K and 30K; 20 cases per length at depths 0% to 95%.
  - Model: LongChat-7b-v1.5-32k. The first two layers are not compressed.
  - Budgets: 512, 1024, 2048, 4096; page sizes 16 and 32.
  - Baselines: StreamingLLM, H2O, TOVA, using a **two-phase prefill** where the context comes first and the question afterwards.
  - Latency on GovReport samples of about 10K, 20K and 30K.
- **Task sensitivity:** [verified]
  - **Token importance is dynamic.** In a GovReport sample, page 256 is unimportant at first but becomes crucial about 4,500 tokens later.
  - Passkey accuracy for baselines that "permanently evict": StreamingLLM 0–40%, H2O 0–40%, TOVA 5–40%. Accuracy falls further with longer context or smaller budget; e.g. at 30K with 512–4096 budget they reach only 0–15%. ArkVale stays at 95% or above.
  - On LongBench, baselines show "noticeable disparities when the budget dips below 2048 and even 4096". ArkVale matches Origin at: 1024 for HotpotQA and Qasper; 2048 for NarrativeQA and PassageRetrieval; 512 for TriviaQA. **TriviaQA (few-shot) is least sensitive; NarrativeQA and PassageRetrieval are most sensitive.**
- **Sources:** https://github.com/pku-liang/ArkVale ; https://raw.githubusercontent.com/pku-liang/ArkVale/main/media/arkvale-nips24-paper.pdf ; https://neurips.cc/virtual/2024/poster/96635
- **Overall confidence:** verified.

## 19. LESS (Low-rank Embedding Sidekick with Sparse policy)

- **Canonical:** "Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference". Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, Beidi Chen. **ICML 2024** (PMLR v235, pp. 11437–11452). arXiv **2402.09398**. Repo: https://github.com/hdong920/LESS [verified]
- **Mechanism:** an eviction policy (H2O, or Λ-shaped sinks + recent) combined with a small, constant-size **learned low-rank recurrent state**. The state accumulates the residual of the evicted tokens, so "all tokens can be queried at later decoding steps". It needs cheap per-layer training. [verified]
- **Phase / query-awareness:** both (per-step). Query-agnostic. Requires training. [verified/likely]
- **Workloads:** [verified: repo `src/eval_gen.py`, `example_scripts`]
  - Language modelling: WikiText via lm-eval-harness, 0-shot. Other lm-harness tasks are "similarly" supported; the exact classification task list is **unknown**.
  - Summarization, 5-shot on 1000 samples: CNN/DailyMail, XSum, MultiNews.
  - Models: Llama 2 7B and Falcon 7B. Other sizes are **unknown**.
  - Budgets: e.g. H2O at 5% (2.5% heavy + 2.5% recent), with a "Baseline+" control given matching extra memory; the README also mentions 10% H2O for Falcon.
- **Task sensitivity:** eviction methods "can have limited success in tasks that require recollecting a majority of previous tokens". LESS narrows the gap to full cache, sometimes matching it. [verified: README abstract]
- **Sources:** https://github.com/hdong920/LESS ; https://raw.githubusercontent.com/hdong920/LESS/main/src/eval_gen.py ; https://arxiv.org/abs/2402.09398 ; https://proceedings.mlr.press/v235/dong24f.html
- **Overall confidence:** likely (partial task list).

## 20. VATP ("Attention Score is not All You Need ... Value Also Matters")

- **Canonical:** "Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters". Zhiyu Guo, Hidetaka Kamigaito, Taro Watanabe (NAIST). **EMNLP 2024** (main, pp. 21158–21166; 2024.emnlp-main.1178). arXiv **2406.12335**. Repo: https://github.com/guozhiyu/vatp [verified]
- **Mechanism:** Value-Aware Token Pruning. Token importance is the accumulated attention score multiplied by the ℓ1 norm of the token's value vector. It plugs into H2O (`--h2o`) or Scissorhands, with attention-sink tokens protected (`--sink_len 20`). [verified]
- **Phase / query-awareness:** both, as a scoring rule for H2O/Scissorhands. Query-agnostic. The eval **masks** tokens rather than dropping them, per the README. [verified]
- **Workloads:** **LongBench, 16 English tasks.** Models: LLaMA2-7B-chat(-4k) and Vicuna-v1.5-7B. Budget: 50% of the KV cache (`--heavy_ratio 0.25` + recent). [verified]
- **Task sensitivity:** VATP beats attention-score-only baselines on more than 12 of 16 tasks. Which tasks it loses on is **unknown**. [verified; per-task losers unknown]
- **Sources:** https://github.com/guozhiyu/vatp ; https://arxiv.org/abs/2406.12335 ; https://aclanthology.org/2024.emnlp-main.1178/
- **Overall confidence:** verified.

---

## Additional methods found (2023 – mid-2024)

## 21. LM-Infinite

- **Canonical:** "LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models". Chi Han et al. (UIUC; first author per repo owner "Glaciohound" [likely]). **NAACL 2024** (Outstanding Paper; 2024.naacl-long.222). arXiv **2308.16137**. Repo: https://github.com/Glaciohound/LM-Infinite [verified]
- **Mechanism:** a Λ-shaped attention mask: the first n_starting tokens plus the most recent L_pretrain tokens, combined with a relative-distance ceiling. This is functionally the same retained set as StreamingLLM, published concurrently. [verified]
- **Phase / query-awareness:** both (streaming). Query-agnostic. [verified]
- **Workloads:** [verified: README/scripts]
  - Perplexity on ArXiv and OpenWebText2 (Pile subsets) up to **200M** tokens.
  - Generation quality (BLEU/ROUGE) on ArXiv at positions 4K to 16K.
  - Passkey retrieval, 6K to 16K.
  - **Qasper.**
  - Models: LLaMA, Llama-2, GPT-J, MPT-7B.
  - Efficiency: 2.7x decoding speedup and 7.5x memory saving.
- **Task sensitivity:** for passkey, the repo's eval script adds `--top_k_attention 5 --top_k_from_layer 4`, which lets higher layers attend to top-k middle tokens. **The pure Λ-mask cannot retrieve information outside its window.** It improves Qasper and passkey over vanilla models only in a zero-shot length-extrapolation sense. [likely: inferred from scripts + search summary]
- **Sources:** https://github.com/Glaciohound/LM-Infinite ; https://arxiv.org/abs/2308.16137 ; https://aclanthology.org/2024.naacl-long.222/
- **Overall confidence:** verified.

## 22. RoCo / EasyKV ("On the Efficacy of Eviction Policy ...")

- **Canonical:** "On the Efficacy of Eviction Policy for Key-Value Constrained Generative Language Model Inference". Siyu Ren, Kenny Q. Zhu. arXiv **2402.06262** (Feb 2024). The venue is **unknown**; an OpenReview entry exists. Repo: https://github.com/DRSY/EasyKV [verified]
- **Mechanism:** RoCo, a "robust cache omission" policy. It scores tokens by **temporal (averaged) attention** plus a robustness term (standard deviation of attention); the EasyKV policy names include `h2o_head_std_avg`. It is a study of importance-score and eviction-scope choices, i.e. a persistence-of-importance follow-up. [verified/likely]
- **Phase / query-awareness:** both. EasyKV has "encoding" (prefill with a stride), "decoding" and "auto" modes. Query-agnostic. [verified]
- **Workloads:**
  - Repo examples: passkey retrieval (Vicuna-7B-16K at 10K; DynamicNTK LLaMa2-7B-Chat at 5K), summarization (news article) and instruction following (LLaMa2-7B-Chat, budget 300/150 tokens), plus perplexity with LLaMa2-13B. Supports LLaMa, LLaMa2, Mistral and Zephyr. [verified]
  - Paper: language modelling, summarization, "context reconstruction", and AlpacaEval, where RoCo "matches full cache performance at 40% budget". [likely: secondary ArXivQA summary]
- **Task sensitivity:** gains are larger on generative tasks than on language modelling, because errors accumulate. [likely]
- **Sources:** https://github.com/DRSY/EasyKV ; https://arxiv.org/abs/2402.06262 ; https://raw.githubusercontent.com/taesiri/ArXivQA/main/papers/2402.06262.md
- **Overall confidence:** likely.

## 23. Ada-KV (adaptive head-wise budget)

- **Canonical:** "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference". Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, S. Kevin Zhou. arXiv **2407.11550** (July 2024). **NeurIPS 2025**. Repo: https://github.com/FFY0/AdaKV [verified]
- **Mechanism:** within each layer, the total budget is reallocated across heads according to their attention concentration. It is plugged into SnapKV/PyramidKV and uses a flattened variable-length cache with `flash_attn_varlen_func`. [verified]
- **Phase / query-awareness:** prefill-only (SnapKV-style). Implicitly query-aware. [verified/likely]
- **Workloads:** the README shows gains on **RULER** sub-tasks (added later) [verified]. The original evaluation on LongBench (16 datasets) and NIAH is [unverified: memory].
- **Task sensitivity:** unknown from verified sources.
- **Sources:** https://github.com/FFY0/AdaKV ; https://arxiv.org/abs/2407.11550
- **Overall confidence:** likely.

## 24. SqueezeAttention

- **Canonical:** "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget". Zihao Wang, Shaoduo Gan. arXiv **2404.04793**. The venue is **unknown**. Repo: https://github.com/hetailang/SqueezeAttention [verified]
- **Mechanism:** clusters layers by importance, measured by the cosine similarity of hidden states before and after attention. It reallocates the per-layer budget and then applies a token policy (sliding window, StreamingLLM or H2O) per layer. [verified]
- **Phase / query-awareness:** both. Query-agnostic. [verified]
- **Workloads:** repo examples [verified]:
  - LongBench **SAMSum** with Mistral-7B at a 21% budget;
  - HELM **XSUM** with LLaMA-2-7B-32K and StreamingLLM at 40% (300 samples).

  The paper claims "a wide range of LLMs and benchmarks", but the full list is **unknown**.
- **Task sensitivity:** 30–70% memory reduction and up to 2.2x throughput. Per-task sensitivity is unknown. [verified]
- **Sources:** https://github.com/hetailang/SqueezeAttention ; https://arxiv.org/abs/2404.04793
- **Overall confidence:** likely.

## 25. InfLLM (training-free block memory; non-evicting contrast)

- **Canonical:** "InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory". Chaojun Xiao et al. (THUNLP). arXiv **2402.04617**. Venue: **unverified** (possibly NeurIPS 2024). Repo: https://github.com/thunlp/InfLLM [verified title and repo]
- **Mechanism:** sinks + local window, with distant context stored in block-level memory units (optionally offloaded) and representative tokens used for lookup. Relevant blocks are retrieved per step via top-k (optionally with faiss). Nothing is lost; it trades memory for retrieval. [verified/likely]
- **Phase / query-awareness:** both (streaming chunks). Query-aware retrieval. [likely]
- **Workloads:** **InfiniteBench and LongBench.** Models: Mistral-7B-Instruct-v0.2, Vicuna, Llama-3-Instruct, Qwen, MiniCPM. [verified: README]
- **Task sensitivity:** not recovered.
- **Sources:** https://github.com/thunlp/InfLLM ; https://arxiv.org/abs/2402.04617
- **Overall confidence:** verified (benchmarks); findings unknown.

## 26. Q-Hitter (stub)

- "Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache". **MLSys 2024**; the proceedings PDF URL appeared in search results. It is an H2O-lineage method that combines heavy hitters with quantization-friendliness. Authors, benchmarks and repo are **unknown**. [title/venue likely; rest unverified]
- Source: https://proceedings.mlsys.org/paper_files/paper/2024/file/bbb7506579431a85861a05fff048d3e1-Paper-Conference.pdf

## Candidates identified but NOT characterized (search budget exhausted)

A2SF (arXiv 2407.20485; accumulative attention with forgetting factor), CORM (arXiv 2404.15949), KVMerger ("Model Tells You Where to Merge", 2024), MiniCache (NeurIPS 2024; cross-layer, depth dimension), RazorAttention (2024; retrieval heads), SubGen (2024), ALISA (ISCA 2024), SparQ Attention (non-evicting, query-aware), Dynamic Memory Compression and Dynamic Context Pruning (both need training), HeadKV (Oct 2024; outside the window). All are **unverified**.

## Related benchmark study (not a method; useful for the taxonomy)

- "KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches". Jiayi Yuan, Hongyi Liu, Shaochen Zhong et al. **EMNLP 2024 Findings**. arXiv 2407.01527. Repo: https://github.com/henryzhongsc/longctx_bench [verified]
- It evaluates token eviction (StreamingLLM, H2O, InfLLM), quantization (KIVI, FlexGen), prompt compression (LLMLingua2) and RNNs/hybrids (Mamba, Mamba-2, RecurrentGemma, RWKV-5).
- Workloads: 15 LongBench datasets (NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, MuSiQue, GovReport, QMSum, MultiNews, TREC, TriviaQA, SAMSum, PassageRetrieval, LCC, RepoBench-P) plus a Paul-Graham-essay passkey NIAH (20,480 words, 10x10x3, 7 digits).
- Models: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, LongChat-7B-v1.5-32K. Compression ratios 2x, 4x, 6x and 8x.
- Its published findings were not retrieved (search budget): **unverified**.

---

## Cross-cutting observations

### A. Benchmark-family coverage matrix (Y = evaluated; r = repo-only or later; ? = likely/unverified)

| Method | lm-eval MC (COPA/PIQA/…) | short summ. (XSUM/CNN/MultiNews) | LM perplexity | LongBench | NIAH/passkey | other long (InfiniteBench/LEval/SQuALITY/QASPER/GovReport-8K) | chat/instr. (AlpacaEval/MT-Bench/SODA) | math/code gen (GSM8K/HumanEval) | streaming / multi-turn memory |
|---|---|---|---|---|---|---|---|---|---|
| StreamingLLM | – | – | PG19 (4M tok) | Y (6, appx.) | – | – | (demo) | – | ARC-stream, StreamEval |
| H2O | Y (6) | Y | – | – | – | – | – | (MathQA MC) | – |
| Scissorhands | Y (4) | – | C4 | – | – | – | – | – | – |
| FastGen | – | – | – | – | – | – | AlpacaEval | GSM8k, HumanEval (+NQ, TQA) | – |
| TOVA | – | – | PG-19 | – | – | SQuALITY, QASPER, story gen | – | – | – |
| Keyformer | Y? | CNN/DM (XSUM r) | – | – | – | GovReport 8K | SODA | – | – |
| SnapKV | – | – | – | Y (16) | Y (380K) | Command-R RAG | – | (LCC/RepoBench in LB) | – |
| PyramidKV | – | – | – | Y (16/17) | Y | (RULER r) | – | (in LB) | – |
| PyramidInfer | MMLU/BBH | – | wikitext-2 | – | – | LEval | MT-Bench | GSM8K, HumanEval | – |
| NACL | short-text ? | – | – | long-text ? | (passkey r) | InfiniteBench r | – | – | – |
| L2-norm | – | – | Wikipedia | – | Y | – | – | – | – |
| SirLLM | – | – | – | – | – | – | DailyDialog | – | Grocery, RPS |
| D2O | (PIQA/OBQA r) | (HELM r) | – | Y | Y | – | – | GSM8K (+CoQA, TruthfulQA) | – |
| CaM | Y r | Y r | WikiText/PG19 r | – | – | – | – | – | – |
| InfiniPot | – | – | – | Y | Y (1M) | – | – | – | – |
| Quest | – | – | PG-19 | Y (6) | Y (10K/100K) | – | – | – | – |
| InfiniGen | Y (5) | – | WikiText-2/PTB | – | – | – | – | – | – |
| ArkVale | – | – | – | Y (6) | Y (10–30K) | – | – | – | – |
| LESS | Y (?) | Y | WikiText | – | – | – | – | – | – |
| VATP | – | – | – | Y (16) | – | – | – | – | – |
| LM-Infinite | – | – | ArXiv/OWT2 (200M) | – | Y | Qasper | – | – | – |
| RoCo/EasyKV | – | Y | Y | – | Y r | – | AlpacaEval ?, instr. r | – | – |
| Ada-KV | – | – | – | ? | ? | RULER r | – | – | – |
| SqueezeAttention | – | XSUM r | – | SAMSum r | – | – | – | – | – |
| InfLLM | – | – | – | Y | (in ∞Bench) | InfiniteBench | – | – | – |

### B. Dominant benchmark families, by era

1. **2023 decode-time "accumulated-attention" wave** (H2O, Scissorhands, Keyformer, CaM, LESS, InfiniGen, VATP-style simulation):
   - short-context lm-eval-harness multiple-choice: COPA, MathQA, OpenBookQA, PIQA, RTE, Winogrande, HellaSwag, BoolQ;
   - HELM short summarization: XSUM, CNN/DM, MultiNews;
   - perplexity: C4, WikiText-2, PTB, PG-19.

   Prompts are mostly under 2K tokens, and the MC tasks score log-likelihoods, often with **masking** rather than physical eviction. Outputs are short: summaries of about 128 tokens, or single-token MC answers.
2. **Streaming wave** (StreamingLLM, LM-Infinite, SirLLM, TOVA, InfiniPot): very-long-stream perplexity (PG-19, ArXiv, OWT2), concatenated-QA streams (ARC, StreamEval), passkey, and bespoke multi-turn memory tasks (SirLLM).
3. **Mid-2024 prefill-compression wave** (SnapKV, PyramidKV, NACL, VATP, D2O, Ada-KV, ArkVale, Quest): **LongBench** (16 English datasets or a 6-task subset) plus **Needle-in-a-Haystack / passkey**. A few add InfiniteBench (NACL repo, InfLLM), LEval (PyramidInfer), SQuALITY/QASPER (TOVA) or RULER (only in later repo updates).
4. **Chat / generation quality** is rare: AlpacaEval (FastGen, RoCo?), MT-Bench (PyramidInfer), SODA (Keyformer), GSM8K/HumanEval (FastGen, PyramidInfer, D2O).

### C. Reported sensitivity patterns (converging evidence)

- **Exact retrieval of arbitrary, high-entropy content is the most fragile workload.**
  - NACL: InfiniteBench Retrieve.KV falls from 0.592 to 0.036 at 80% eviction, while passkey, summarization, QA and MC are near baseline.
  - TOVA: QASPER needs 1/2 of the cache vs 1/8 for PG-19 and 1/4–1/8 for SQuALITY.
  - L2: NIAH tolerates 50% compression, passkey 90%.
  - ArkVale/Quest: query-agnostic permanent eviction gives passkey ≤40% (often 0–5%) when the question arrives after the context.
- **When the query is known matters as much as the policy.**
  - Prefill-time, observation-window methods (SnapKV, PyramidKV, NACL proxy tokens, Ada-KV) assume the question sits at the end of the prompt.
  - Decode-time query-agnostic methods (H2O, TOVA, StreamingLLM) fail when the question comes later (Quest: "discard the KV cache of the answer before receiving the question"; ArkVale two-phase prefill).
  - Non-evicting or recallable designs (Quest, ArkVale, InfiniGen, InfLLM) close this gap.
  - kvpress later made this an explicit `query_aware` evaluation switch.
- **Token importance drifts during long decoding.** ArkVale shows a page becoming critical about 4.5K tokens later. InfiniGen shows H2O's per-chunk perplexity diverging over decoding. Quest notes that summarization attends to different paragraphs sequentially. Permanent eviction hurts long generations and summarization over long inputs.
- **Preamble and instruction retention matters.**
  - StreamingLLM with 4+3496 underperforms simple truncation on LongBench because it loses the initial prompt.
  - Sink tokens are essential (window-only PPL 5158 vs 5.40).
  - FastGen's special-token heads and TOVA's retention of the first token and punctuation point the same way.
- **Few-shot / in-context-learning prompts collapse at tiny budgets** (PyramidKV's +20.5 on TREC at 0.7% retention), while TriviaQA few-shot is the least sensitive in ArkVale.
- **Dialogue is more sensitive than summarization** (Keyformer: 90% vs 70% of prompt KV). Multi-turn recall of early facts breaks StreamingLLM (SirLLM).
- **Short-context MC and perplexity look robust** (H2O at 20%, Scissorhands at 5x on MC). Perplexity is somewhat more sensitive than MC accuracy (Scissorhands).
- **Bigger models are more compressible** (FastGen: 44.9% pruned on 65B vs 16.9% on 7B; Scissorhands has flatter curves with size).
- **Reasoning and truthfulness can even improve** with pruning plus merging (D2O on GSM8K/CoQA; TruthfulQA above full cache). This is a single source and needs replication.

### D. Workload types systematically missing (in the methods' own evaluations)

1. **Multi-turn conversations over a shared, already-compressed context** where later turns ask about different parts. Only StreamEval/ARC streams and SirLLM's bespoke tasks come close, and none use real multi-session chat logs or shared-prefix serving.
2. **Long-output generation / long reasoning:** chain-of-thought or long-form answers with thousands of generated tokens. Decode-phase eviction is mostly scored on outputs under 256 tokens; the exceptions are TOVA story generation and InfiniGen's perplexity-by-chunk.
3. **Agentic / tool-use traces, structured data** (JSON, tables, logs) and **code repositories beyond LCC/RepoBench-P**. HumanEval appears only in FastGen and PyramidInfer.
4. **RAG with many retrieved passages and citation fidelity.** This appears only in SnapKV's Command-R case study.
5. **Aggregation / counting / multi-hop tracking** (e.g. RULER variable-tracking or common-word extraction). These appear only later, in repo updates (Ada-KV, KVCache-Factory).
6. **Non-English workloads:** LongBench Chinese tasks are almost always dropped; the NACL repo's Zh.QA is an exception.
7. **Serving-realistic mixes:** batched heterogeneous request-length distributions, prefix sharing, and interplay with paged or prefix caches. Throughput results use synthetic fixed lengths: FlexGen (H2O), OSDI (InfiniGen), kernels (Quest, ArkVale).
8. **Contexts above 128K** are rare: SnapKV NIAH to 380K, InfiniPot passkey to 1M, StreamingLLM/LM-Infinite perplexity streams.
9. **Comparability problems:** budgets are defined as a % of prompt, tokens per layer, per head or of all tokens; masking vs real eviction; different model generations (OPT and LLaMA-1 in 2023, Llama-3 and Mistral in 2024). Cross-paper numbers are rarely directly comparable.
