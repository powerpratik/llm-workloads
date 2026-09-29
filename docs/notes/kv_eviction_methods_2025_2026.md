> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# Recent token-level KV-cache eviction / compression methods (late 2024 - 2026): evaluated workloads and task-sensitivity findings

Scope: head/layer-adaptive budgets, retrieval-head methods, query-agnostic / reusable compression, decode-heavy
(reasoning traces, long-form generation) and multi-turn / agentic workloads, plus non-evicting sparse/retrieval
contrasts and the analysis papers that report task sensitivity. Compiled 2026-09-29.

## Provenance and confidence conventions

- **[verified]**: read directly this session in the official repo README / scripts / config files
  (raw.githubusercontent.com or github.com), or the venue is shown by an official proceedings / anthology /
  conference URL returned in search results.
- **[likely]**: taken from web-search snippets or summaries of the abstract or paper, not checked against the PDF.
  Re-check any [likely] number before citing it.
- **[unverified]**: recalled from memory or inferred, not confirmed this session. Treat it as a lead only.
- **unknown**: not found.
- Access limits: arxiv.org, openreview.net, aclanthology.org, huggingface.co, alphaxiv, semanticscholar,
  github.io pages and most paper-summary sites were blocked (curl and WebFetch). The shared web-search budget
  (200 queries) ran out before the 2026 agentic papers could be characterized.
- **Two different benchmarks are both called LongGenBench.** SCOPE uses Liu et al.'s LongGenBench. It
  concatenates many GSM8K/MMLU/CSQA questions so the answer becomes one long output (GSM8K+, MMLU+, CSQA+;
  github.com/Dominic789654/LongGenBench) [verified: SCOPE README]. MorphKV uses Wu et al.'s LongGenBench, which is
  open-ended long writing such as floor plans and diaries, scored by CR/Once/Range/Periodic
  (github.com/mozhu621/LongGenBench) [verified: MorphKV README].

---------------------------------------------------------------------------------------------------------------

## A. Head / layer-adaptive budgets and retrieval-head methods

### A1. Ada-KV
- **Paper**: "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference".
  Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, S. Kevin Zhou. arXiv 2407.11550 (2024). **NeurIPS 2025**.
  Repo https://github.com/FFY0/AdaKV [verified: README + bib].
- **Mechanism**: head-wise adaptive budget allocation. Budget moves from attention-concentrated heads to
  attention-dispersed heads, which minimizes an upper bound on eviction loss. It plugs into SnapKV/PyramidKV
  scorers and uses a flattened variable-length cache with `flash_attn_varlen` [verified README; loss-bound
  framing likely].
- **Phase / query**: prefill (SnapKV-style observation window). Query-aware by default, and also evaluated
  question-agnostic [likely].
- **Workloads**: 13 RULER sub-tasks and 16 LongBench datasets, each under question-aware and question-agnostic
  settings [likely: abstract snippet]. The README figure shows gains on RULER sub-tasks when Ada-KV is added to
  SnapKV/PyramidKV [verified].
  - Models: unknown this session. Early versions used Mistral-7B-Instruct-v0.2 and LWM-Text-Chat-1M
    [unverified recall].
  - Default config: base_capacity=512, window=32 [verified].
- **Findings**: heads differ strongly in attention concentration, so uniform per-head budgets waste capacity
  [verified README].
  - Adopted by NVIDIA kvpress (AdaKVPress), vLLM kvcompress and Sparse Frontier [verified].
  - Follow-ups: CriticalKV (arXiv 2502.03805) and DefensiveKV, "Taming the Fragility of KV Cache Eviction"
    (arXiv 2510.13334). Titles are verified; content is not.
- **Sources**: https://github.com/FFY0/AdaKV ; https://arxiv.org/abs/2407.11550 ;
  https://neurips.cc/virtual/2025/poster/115578 ; https://mlanthology.org/neurips/2025/feng2025neurips-adakv/

### A2. CAKE
- **Paper**: "CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences". Ziran Qin, Yuchen Cao,
  Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, Jianguo Li. **ICLR 2025**. arXiv 2503.12491.
  Repo https://github.com/antgroup/cakekv [verified: README/bib; arXiv ID from search listing].
- **Mechanism**: sets each layer's budget from its spatial attention dispersion and temporal attention shift,
  allocated in a cascade during prefill. The eviction indicator also tracks how token importance shifts over
  time [verified README].
- **Phase / query**: prefill; query-aware (observation window).
- **Workloads**: LongBench and NeedleBench [verified README].
  - Models (repo config): Llama-2-7B-chat, Llama-2-13B-chat, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
    Qwen2.5-7B-Instruct [verified config].
  - Example budget: cache_size=1024, window=32 [verified].
- **Findings**: layers differ in the spatial dispersion and temporal shift of their attention, so uniform or
  fixed-pyramid layer budgets are suboptimal. CAKE claims preserved performance with 3.2% of the KV cache, with the
  largest gains at low memory [verified README].
- **Sources**: https://github.com/antgroup/cakekv ;
  https://raw.githubusercontent.com/antgroup/cakekv/main/experiments/LongBench/config/model2path.json ;
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/dfae940651f3e690a12e19c874edad7c-Abstract-Conference.html

### A3. HeadKV / HeadKV-R2
- **Paper**: "Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and
  Reasoning". Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, Wen Xiao. **ICLR 2025**. arXiv 2410.19258.
  Repo https://github.com/FYYFU/HeadKV [verified].
- **Mechanism**: offline head-importance scores define a global head-level budget pool. The scores come either from
  retrieval heads or from "retrieval-reasoning" heads estimated on retrieval+reasoning examples. Tokens inside each
  head are chosen SnapKV-style; `beta` sets the size of the global pool [verified README].
- **Phase / query**: prefill. Token selection is query-aware; head importance is static.
- **Workloads**:
  - LongBench and LooGLE [likely].
  - Needle-in-a-Haystack at 1K-8K in scripts [verified].
  - "Reason-in-a-haystack" built on BABILong [verified].
  - Models: Llama-3-8B-Instruct [verified scripts]; Mistral-7B-Instruct [likely].
- **Budgets**: average per-head capacity 64/128/256/512/1024 [verified].
- **Findings**: keeping 1.5% of the KV cache retains about 97% of full performance on contextual QA [likely].
  Retrieval-reasoning head scores are denser and more discriminative than retrieval-only scores, and head-level
  allocation wins most at low budgets [likely].
- **Sources**: https://github.com/FYYFU/HeadKV ; https://arxiv.org/abs/2410.19258 ;
  https://www.microsoft.com/en-us/research/publication/not-all-heads-matter-a-head-level-kv-cache-compression-method-with-integrated-retrieval-and-reasoning/

### A4. DuoAttention
- **Paper**: "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads". Guangxuan Xiao,
  Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han. **ICLR 2025**. arXiv 2410.10819.
  Repo https://github.com/mit-han-lab/duo-attention [verified; ICLR via proceedings URL].
- **Mechanism**: an optimization on synthetic passkey data identifies retrieval heads, which keep the full KV. The
  other "streaming" heads keep only sink + recent tokens [verified].
- **Phase / query**: both (prefill and decode); query-agnostic (static head pattern).
- **Workloads**:
  - NIAH and LongBench [verified]; short-context MMLU, MBPP and MT-Bench [likely].
  - Models: Llama-2-7B-32K-Instruct, Llama-3-8B-Instruct-Gradient-1048k/4194k, Mistral-7B-Instruct-v0.2/v0.3, plus
    Llama-3.1-8B-Instruct patterns [verified]; Llama-3-70B on the short tasks [likely].
- **Budgets and efficiency** [verified]:
  - Retrieval-head ratio of 25% (MHA) or 50% (GQA) matches full attention on NIAH; the example uses sink 64 and
    recent 256.
  - Memory falls by 2.55x (MHA) and 1.67x (GQA). Decoding is 2.18x / 1.50x faster and pre-filling 1.73x / 1.63x
    faster.
  - With quantization it fits 3.3M tokens on one A100.
- **Findings**: only a fraction of heads needs the full context. On LongBench with Llama-3-8B-Instruct-1048K at a 50%
  budget, it scores 40.21 vs H2O 35.76, TOVA 35.55 and StreamingLLM 32.26 [likely]. RLKV (D11) later reports that
  retrieval-oriented head selection misses reasoning-critical heads [verified in RLKV README].
- **Sources**: https://github.com/mit-han-lab/duo-attention ; https://arxiv.org/abs/2410.10819 ;
  https://proceedings.iclr.cc/paper_files/paper/2025/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf

### A5. RazorAttention
- **Paper**: "RazorAttention: Efficient KV Cache Compression Through Retrieval Heads". **ICLR 2025** [verified:
  iclr.cc poster URL]. arXiv 2407.15891 [verified: search listing]. First author Hanlin Tang [unverified]. No
  official repo found on GitHub.
- **Mechanism**: a few retrieval heads keep the full KV. Non-retrieval heads keep only a local window, and their
  dropped tokens are summarized into a "compensation token". Training-free [likely]. Heads are identified via
  echo/induction-head scores [unverified recall].
- **Phase / query**: both; query-agnostic (static head classification).
- **Workloads**: LongBench with Qwen1.5-7B/72B-Chat, Llama3-8B-Instruct and Baichuan2-13B [likely]; NIAH
  [unverified].
- **Budgets**: about 70% KV compression without noticeable degradation [likely].
- **Findings**: most heads attend locally. On Qwen1.5-7B-Chat, the LongBench average is 35.87 vs full 36.03,
  StreamingLLM 17.00 and H2O 34.16 [likely].
- **Sources**: https://iclr.cc/virtual/2025/poster/28028 ; https://openreview.net/forum?id=tkiZQlL04w ;
  https://proceedings.iclr.cc/paper_files/paper/2025/file/2a98af4fea6a24b73af7b588ca95f755-Paper-Conference.pdf

### A6. SqueezeAttention
- **Paper**: "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget".
  Zihao Wang, Bin Cui, Shaoduo Gan. **ICLR 2025**. arXiv 2404.04793. Repo https://github.com/hetailang/SqueezeAttention
  [verified README; authors/venue via ICLR proceedings URL].
- **Mechanism**:
  - Layer importance is the cosine similarity of the prompt's hidden states before and after self-attention
    [likely].
  - Layers are clustered into groups and budgets reallocated between groups [verified].
  - It composes with sequence-wise policies: sliding window and StreamingLLM [verified], and H2O [likely].
- **Phase / query**: both. Query-awareness depends on the base policy (JSON null).
- **Workloads**: README examples use SAMSum (LongBench) and XSum (HELM) with Mistral-7B and LLaMA-2-7B-32K
  [verified]. The paper covers "a wide range of LLMs and benchmarks"; the full list is unknown.
- **Budgets**: e.g. 21% or 40% of prompt length per layer, with 8% or 25% for the reallocated layer class
  [verified].
- **Findings**: layers differ in sensitivity. It reports 30-70% memory reduction and up to 2.2x throughput
  [verified].
- **Sources**: https://github.com/hetailang/SqueezeAttention ; https://arxiv.org/abs/2404.04793 ;
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/3b0a8df568ec496a717566a7f8158aaa-Abstract-Conference.html

### A7. DynamicKV
- **Paper**: "DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs". Xiabin Zhou, Wenbin Wang,
  Minyan Zeng, Jiaxian Guo, Xuebo Liu, Li Shen, Min Zhang, Liang Ding. arXiv 2412.14838 (2024).
  **EMNLP 2025** (Findings per the ACL Anthology URL; the repo bib says the EMNLP 2025 conference) [verified].
  Repo https://github.com/DreamMr/DynamicKV (README copy fetched during the survey)
  [verified].
- **Mechanism**: at each layer, keeps the top-K tokens by attention from the recent window. Every m layers, the
  budgets of earlier layers are globally re-normalized to respect the total. Only the prefill phase is modified
  [verified README].
- **Phase / query**: prefill; query-aware.
- **Workloads**:
  - LongBench (single/multi-doc QA, summarization, synthetic tasks, code completion) and Needle-in-a-Haystack
    [verified].
  - Models: Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen2-7B-Instruct, InternLM-2.5-7B-Chat-1M [verified].
- **Budgets** [verified README]:
  - LongBench at KV=512 (a 6.9% context ratio): Llama-3-8B 40.73 vs FullKV 41.95, SnapKV 40.30, PyramidKV 40.18,
    H2O 37.20, StreamingLLM 34.70.
  - NIAH at 32K context with a 64-token cache: DynamicKV 83%, FullKV 92%, PyramidKV 72%, StreamingLLM 26%.
  - At 1.7% retention the README headline is about 90% of FullKV, while the abstract snippet says about 85%
    (90/87/78/83% for the four models) [likely].
- **Findings (task sensitivity)**: tasks such as QA, summarization and code show different token-importance
  distributions across layers, so fixed pyramid or sliding-window patterns fail [verified README]. Summarization
  needs small upper-layer caches, while code completion needs larger ones [likely].
- **Sources**: https://github.com/DreamMr/DynamicKV ; https://arxiv.org/abs/2412.14838 ;
  https://aclanthology.org/2025.findings-emnlp.426/

### A8. Locret
- **Paper**: "Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade
  Devices" (v1 title: "Locret: Accelerating Long-Context LLM Inference with Retaining Heads"). Yuxiang Huang,
  Binhang Yuan, Xu Han, Chaojun Xiao, Zhiyuan Liu. arXiv 2410.01805. **TMLR 2025** [likely: mlanthology].
  Repo https://github.com/huangyuxiang03/Locret [verified].
- **Mechanism**: small trained "retaining heads" predict a causal importance score (CIS) for each cache unit.
  Eviction is interleaved with chunked prefill, the backbone stays frozen, and training takes under 1 GPU-hour
  [verified/likely].
- **Phase / query**: prefill (chunked); query-agnostic (the score depends only on preceding units).
- **Workloads**: InfiniteBench and L-Eval (the repo's benchmark folders); the example task is R.PassKey [verified].
  Models: Phi-3-mini-128K and Llama-3.1-8B-Instruct [verified].
- **Budgets**: 20x (Phi-3-mini-128K) and 8x (Llama-3.1-8B) compression; 128K+ context on a single RTX 4090
  [verified]; under 10% average degradation [likely].
- **Findings**: no per-task breakdown found this session.
- **Sources**: https://github.com/huangyuxiang03/Locret ; https://arxiv.org/abs/2410.01805 ;
  https://mlanthology.org/tmlr/2025/huang2025tmlr-locret/

### A9. ThinK
- **Paper**: "ThinK: Thinner Key Cache by Query-Driven Pruning". Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu,
  Aojun Zhou, Amrita Saha, Caiming Xiong, Doyen Sahoo. **ICLR 2025** (Spotlight [likely]). arXiv 2407.21018.
  Repo https://github.com/SalesforceAIResearch/ThinK [verified].
- **Mechanism**: prunes the key-cache channels (head dimensions) with the lowest query-driven importance. It
  composes with token eviction (SnapKV/H2O) and with KIVI quantization [verified].
- **Phase / query**: prefill; query-aware.
- **Workloads**: LongBench [verified] and NIAH [likely]. Models: LLaMA-2-7B-chat, LLaMA-3-8B-Instruct,
  LLaMA-3-70B-Instruct, Mistral-7B-Instruct-v0.2 [likely].
- **Budgets**: about 40% key-cache reduction with comparable accuracy, at the cost of a slight TTFT increase
  [likely].
- **Findings**: the channel dimension is redundant (uneven magnitudes, low rank) [likely]. No task breakdown found.
- **Sources**: https://github.com/SalesforceAIResearch/ThinK ; https://arxiv.org/abs/2407.21018 ;
  https://proceedings.iclr.cc/paper_files/paper/2025/file/8edb116d5b288b6a9bba4c16ab647702-Paper-Conference.pdf

### A10. MiniCache (depth-dimension merging; not token eviction)
- **Paper**: "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models". Akide Liu et al.
  **NeurIPS 2024** [verified: proceedings URL]. arXiv 2405.14366 [unverified]. Repo https://github.com/AkideLiu/MiniCache
  (README is empty) [verified].
- **Mechanism**: merges KV states of adjacent middle-to-deep layers. It decomposes each state into magnitude and
  direction, interpolates the directions, and keeps highly distinct token pairs unmerged [likely].
- **Phase / query**: both; query-agnostic.
- **Workloads**: GSM8K, COQA and TruthfulQA (all-layer merging), plus LongBench [likely]. Models: LLaMA-2, LLaMA-3,
  Phi-3, Mistral, Mixtral [likely].
- **Budgets**: up to 5.02x compression when combined with 4-bit quantization [likely].
- **Sources**: https://proceedings.neurips.cc//paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html ;
  https://neurips.cc/virtual/2024/poster/93380

### A11. SepLLM
- **Paper**: "SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator". Guoxuan Chen
  et al. **ICML 2025** (PMLR 267). arXiv 2412.12094. Repo https://github.com/HKUDS/SepLLM [verified].
- **Mechanism**: keeps initial tokens, separator (punctuation) tokens and a neighboring window, on the premise that
  a segment's information condenses into its separator. It has training-free, from-scratch and post-training
  variants, and ships a portable SepCache [verified].
- **Phase / query**: both; query-agnostic (structural).
- **Workloads** [verified]:
  - Training-free: GSM8K-CoT and MMLU via lm-eval-harness.
  - Streaming: PG19 language modeling up to 4M tokens.
  - Training: Pythia / GPT-NeoX on the Pile.
  - Models: Llama-3-8B(-Instruct), Falcon, Pythia-160M.
- **Budgets**: over 50% KV reduction on GSM8K-CoT with Llama-3-8B at comparable accuracy [verified].
- **Findings**: separators receive disproportionate attention [verified]. It was not evaluated on reasoning-model
  traces.
- **Sources**: https://github.com/HKUDS/SepLLM ; https://arxiv.org/abs/2412.12094 ;
  https://proceedings.mlr.press/v267/chen25bf.html

### A12. WindowKV
- **Paper**: "WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference".
  arXiv 2503.17922 (2025). Venue: arXiv (none stated on the repo) [verified]. First author Youhui Zuo [unverified].
  Repo https://github.com/optim996/WindowKV [verified].
- **Mechanism** [verified README]:
  - Keeps contiguous semantic windows instead of discrete tokens.
  - A BERT-base task classifier picks the window-selection statistic (max vs average suffix scoring).
  - Layers in a group share KV indices to cut overhead.
- **Phase / query**: prefill; query-aware.
- **Workloads**: LongBench, and NIAH scored by Rouge-1 F1 as in PyramidKV [verified].
  - Model: Meta-Llama-3-8B-Instruct in scripts [verified]; other backbones are unnamed.
  - Budgets: 512/1024/2048 average per layer [verified].
- **Findings**: matches the full KV cache with 12% of it [verified]; 97.9% NIAH at 8K [likely]. The task type
  decides which selection statistic works [verified]. The classifier classes may correspond to localization vs
  aggregation tasks [unverified].
- **Sources**: https://github.com/optim996/WindowKV ; https://arxiv.org/abs/2503.17922

### A13. RLKV (reasoning heads): see D11.

---------------------------------------------------------------------------------------------------------------

## B. Retrieval / sparse-attention methods (non-evicting contrast)

### B1. ShadowKV
- **Paper**: "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference". Hanshi Sun, Li-Wen Chang,
  Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen. **ICML 2025 Spotlight**.
  arXiv 2410.21465. Repo https://github.com/bytedance/ShadowKV [verified].
- **Mechanism**: keeps low-rank pre-RoPE keys on GPU and offloads values to CPU. Chunk landmarks select a sparse KV
  set at each decode step, which is then reconstructed; nothing is permanently evicted [verified/likely].
- **Phase / query**: decode (prefill builds the low-rank keys and landmarks); query-aware per step.
- **Workloads**:
  - RULER at 128K: niah_single_1/2/3, niah_multikey_1/2, niah_multiquery, niah_multivalue, vt, fwe, qa_1, qa_2
    [verified command]. Note that CWE is not in the command list [verified].
  - LongBench and NIAH [likely].
  - Models [verified]: Llama-3-8B-1M, GLM-4-9B-1M, Llama-3.1-8B-Instruct, Yi-9B-200K, plus Phi-3-Mini-128K and
    Qwen2-7B-128K (NIAH only).
- **Budgets**: sparse_budget 2048, rank 160, chunk 8 [verified]. Over 6x less GPU memory, up to 6x larger batches
  and 3.04x throughput on A100 [likely].
- **Sources**: https://github.com/bytedance/ShadowKV ; https://arxiv.org/abs/2410.21465 ;
  https://icml.cc/virtual/2025/poster/44053

### B2. MagicPIG
- **Paper**: "MagicPIG: LSH Sampling for Efficient LLM Generation". Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou,
  Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, Beidi Chen. **ICLR 2025
  Spotlight**. arXiv 2410.16179. Repo https://github.com/Infini-AI-Lab/MagicPIG [verified].
- **Mechanism**: LSH-based sampling of keys/values on the CPU estimates the attention output instead of taking a
  TopK [verified/likely].
- **Phase / query**: decode; query-aware per step.
- **Workloads**:
  - 13 RULER tasks [likely].
  - LongBench: QASPER, LCC, RepoBench-P, TriviaQA, PRE, TREC [likely].
  - lm-eval tasks: GSM8K-CoT, MMLU-Flan-CoT-Fewshot, COQA [likely].
  - Models: Llama-3.1-8B/70B-Instruct and MegaBeam-Mistral-7B-512K [verified].
- **Budgets**: about 2% of the attention computation with under 2% degradation; up to about 3.9x throughput
  [likely].
- **Findings (key task-sensitivity result)**: attention is not always sparse. TopK is fine on retrieval (NIAH)
  but degrades severely on aggregation tasks (RULER CWE/FWE) because attention mass is long-tailed. The top 20% of
  tokens cover only about 70-80% of the attention mass, a 15-20% estimation error [likely; consistent across several
  snippets].
- **Sources**: https://github.com/Infini-AI-Lab/MagicPIG ; https://arxiv.org/abs/2410.16179 ;
  https://mlanthology.org/iclr/2025/chen2025iclr-magicpig/

### B3. RetrievalAttention
- **Paper**: "RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval". Di Liu, Meng Chen,
  Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, et al.
  **NeurIPS 2025** [verified: repo bib + proceedings URL]. arXiv 2409.10516 [unverified].
  Repo https://github.com/microsoft/RetrievalAttention (now hosts RetroInfer) [verified].
- **Mechanism**: an attention-aware ANNS index over keys, held in CPU memory, retrieves the relevant KV at each
  decode step. It is built to handle the query/key distribution mismatch that breaks off-the-shelf ANNS [likely].
- **Phase / query**: decode; query-aware per step.
- **Workloads**: InfiniteBench (∞-Bench) and RULER [likely]. It touches only 1-3% of the data, and an 8B model runs 128K context
  on an RTX 4090 at 0.107 s/token [likely].
- **Sources**: https://github.com/microsoft/RetrievalAttention ;
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e36d4049fb0fea195a8267c8dcd0824-Abstract-Conference.html

### B4. RetroInfer (RetrievalAttention follow-up)
- **Paper**: "RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference". Yaoqi Chen et al.
  **PVLDB 19(5), 2026**. arXiv 2505.02922 [verified: README bib].
- **Mechanism**: wave index (steady / retrieval / estimation zones with accuracy-bounded estimation) plus a
  GPU-CPU wave buffer [verified].
- **Phase / query**: decode; query-aware per step.
- **Workloads** [verified]:
  - RULER: niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2.
  - LongBench categories: SQA, MQA, SUM, FSL, ST, CC.
  - Long reasoning: AIME24 and GPQA with DeepSeek-R1-Distill-Llama-8B / Qwen-7B.
  - Models: Llama-3.1-8B, Llama-3-8B-1048K, Qwen2.5-7B/72B.
- **Budgets**: retrieval budget ratio (e.g. 0.018) and estimation ratio (e.g. 0.232) [verified].
- **Sources**: https://github.com/microsoft/RetrievalAttention

### B5. ClusterKV
- **Paper**: "ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression". Guangda Liu,
  Chengwei Li, Jieru Zhao, Chenqi Zhang, Minyi Guo. **DAC 2025**. arXiv 2412.03213.
  Repo https://github.com/sjtu-zhao-lab/ClusterKV [verified].
- **Mechanism**: clusters keys in semantic space and recalls clusters per decode step, so evicted tokens stay
  recallable [verified/likely].
- **Phase / query**: decode; query-aware.
- **Workloads**: LongBench (e.g. hotpotqa) and PG19 perplexity, with Quest as the baseline [verified].
  - Models: Llama-3-8B-Instruct, Llama-3.1-8B-Instruct, GLM-4-9B-chat at 4K/8K/32K [verified config].
- **Findings**: no task-sensitivity claims found this session.
- **Sources**: https://github.com/sjtu-zhao-lab/ClusterKV ; https://arxiv.org/abs/2412.03213

### B6. PQCache
- **Paper**: "PQCache: Product Quantization-based KVCache for Long Context LLM Inference". **SIGMOD 2025**.
  arXiv 2407.12820. First author Hailin Zhang [unverified]. Repo https://github.com/HugoZHL/PQCache [verified].
- **Mechanism**: product-quantizes keys at prefill. At decode, MIPS over the PQ codes picks the top tokens, which are
  fetched from CPU with overlap and caching to hide latency [verified].
- **Phase / query**: decode; query-aware.
- **Workloads**: LongBench v1 [verified]; other benchmarks [unverified]. Models: Llama-3.1-8B-Instruct and
  Mistral-7B-Instruct-v0.2 [verified].
- **Budgets**: quality holds with only 1/5 of tokens in attention [verified]; +4.60% over baselines [likely].
- **Sources**: https://github.com/HugoZHL/PQCache ; https://arxiv.org/abs/2407.12820 ; https://dl.acm.org/doi/abs/10.1145/3725338

### B7. SeerAttention-R
- **Paper**: SeerAttention-R, sparse attention for long reasoning. arXiv 2506.08889 [verified README]. Venue unknown;
  first author unknown this session. The original SeerAttention is Yizhao Gao et al., arXiv 2410.13276 [verified bib].
  Repo https://github.com/microsoft/SeerAttention [verified].
- **Mechanism**: self-distilled block-level AttnGates pick KV blocks at each decode step under a token budget; the
  backbone stays frozen. Nothing is evicted [verified].
- **Phase / query**: decode; query-aware (learned gate).
- **Workloads** [verified]:
  - AIME24 (64 samples), AIME25, MATH-500 (8 samples), GPQA-Diamond (16 samples); averaged pass@1.
  - Models: Qwen3-4B/8B/14B, DeepSeek-R1-Distill-Qwen-14B.
- **Budgets and results (1k-8k token budgets)** [verified tables]:

  | Benchmark | Model | Budget | Score | Full attention |
  |---|---|---|---|---|
  | AIME24 | Qwen3-4B | 4k | 69.32 | 71.25 |
  | AIME24 | Qwen3-4B | 2k | 55.83 | 71.25 |
  | AIME24 | Qwen3-8B | 4k | 71.35 | 74.48 |
  | AIME25 | Qwen3-4B | 4k | 58.59 | 66.41 |
  | MATH-500 | Qwen3-4B | 4k | 94.10 | 93.93 |

- **Findings**: harder, longer-trace tasks (AIME25) need larger budgets than MATH-500, and 1-2k budgets cause large
  drops [verified from tables].
- **Sources**: https://github.com/microsoft/SeerAttention ;
  https://raw.githubusercontent.com/microsoft/SeerAttention/main/eval/reasoning_tasks/README.md

### B8. RocketKV (hybrid: eviction + sparse attention, with a multi-turn variant): see E3.

---------------------------------------------------------------------------------------------------------------

## C. Query-agnostic / reusable compression

### C1. KVzip
- **Paper**: "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction". Jang-Hyun Kim, Jinuk Kim,
  Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song. **NeurIPS 2025 Oral**. arXiv 2505.23416.
  Repo https://github.com/snu-mllab/KVzip [verified].
- **Mechanism** [verified]:
  - Scores each KV pair by the attention it receives while the LLM reconstructs (repeats) the context from the cache.
  - Evicts once, then reuses the compressed cache for any future query.
  - A context-independent head-level variant produces DuoAttention-style head scores in under a minute.
- **Phase / query**: prefill; query-agnostic.
- **Workloads** [verified data loader]:
  - SQuAD with multiple questions per context.
  - NIAH at 500/2K/8K.
  - GSM8K, with the context being the problem statement minus its final question.
  - SCBench: kv, vt, many_shot, mf, repoqa, choice_eng, prefix_suffix, summary, qa_eng, summary_with_needles,
    repoqa_and_kv, with tiny/short/mid variants of about 8K/20K/60K tokens.
- **Models**: Qwen2.5-7B/14B-Instruct-1M, LLaMA3.1-8B, Gemma3-12B, LLaMA3-8B-W8A8KV4 (QServe), Qwen3 [verified].
  Contexts up to about 170K [likely].
- **Budgets**: KV ratio swept from 0.1 to 1.0. It gets 3-4x KV reduction and about 2x lower decode latency with
  minimal loss; head-level ratio 0.6 is recommended [verified].
- **Findings**:
  - Query-aware methods such as SnapKV and PyramidKV keep only the KV relevant to the first query. They degrade on
    multi-query context reuse even at a 90% cache ratio [likely].
  - For coding, head scores computed on repoqa improve robustness [verified tip].
  - update_cache=True supports multi-turn [verified].
- **Sources**: https://github.com/snu-mllab/KVzip ; https://raw.githubusercontent.com/snu-mllab/KVzip/main/data/load.py ;
  https://arxiv.org/abs/2505.23416 ; https://neurips.cc/virtual/2025/poster/118741

### C2. Fast KVzip (2026)
- **Paper**: "Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction". Jang-Hyun Kim, Dongyoon Han,
  Sangdoo Yun. arXiv 2601.17668 (2026) [verified README]. Repo https://github.com/Janghyun1230/FastKVzip.
- **Mechanism**: low-rank "sink attention" gates, distilled from KVzip scores in under one H100-hour, drive eviction
  in both prefill and decoding [verified].
- **Workloads**: "prefill-intensive" tasks (the KVzip suite) and "decoding-intensive" math reasoning (R-KV codebase)
  [verified].
  - Gates are released for Qwen2.5-7B/14B-Instruct-1M, Qwen3-8B/14B/8B-FP8, Qwen3-4B-Instruct-2507 and Gemma-3-12B-it.
- **Budgets**: near-lossless at up to 70% eviction [verified].

### C3. KVzap (2026)
- **Paper**: arXiv 2601.07891 [verified: kvpress kvzap README]. Authors unknown.
- **Mechanism**: a linear or MLP surrogate predicts KVzip+ scores from hidden states. KV pairs below a threshold are
  pruned in both prefill and decoding, following DMS (DMSPress) [verified].
- **Workloads**: trained on a Nemotron pretraining sample; evaluated with the kvpress CLI and on AIME25 with sampling,
  e.g. Qwen3-8B [verified].
- **Sources**: https://raw.githubusercontent.com/NVIDIA/kvpress/main/kvzap/README.md

### C4. Expected Attention (NVIDIA kvpress)
- **Paper**: "Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution".
  Alessio Devoto, Maximilian Jeblick, Simon Jégou. arXiv 2510.00636 (2025); venue: arXiv [verified: kvpress bib].
  Code https://github.com/NVIDIA/kvpress (ExpectedAttentionPress) [verified].
- **Mechanism**: models the distribution of future queries and computes each KV pair's expected attention in closed
  form, without materializing the attention matrix [likely].
- **Phase / query**: both (prefill, and decoding via DecodingPress); query-agnostic.
- **Workloads**:
  - Prefill: LongBench, RULER, NIAH with Llama-3.1-8B, Qwen3-8B, Gemma3-12B [likely].
  - Decode: AIME25 and MATH-500 with Qwen-R1 1.5B (the snippet says "Qwen-15B-R1"), Qwen-R1 7B and
    OpenMath-Nemotron-14B [likely].
  - The kvpress evaluation suite covers Loogle, RULER, Zero-Scrolls, InfiniteBench, LongBench, LongBench-v2 and NIAH,
    with a `query_aware` flag [verified].
- **Findings**: matches or beats baselines in both settings [likely].
- **Sources**: https://github.com/NVIDIA/kvpress ; https://raw.githubusercontent.com/NVIDIA/kvpress/main/evaluation/README.md ;
  https://arxiv.org/abs/2510.00636

### C5. Q-Filters
- **Paper**: "Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression". Nathan Godey, Alessio Devoto,
  Yu Zhao, Simone Scardapane, Pasquale Minervini, Éric de la Clergerie, Benoît Sagot. arXiv 2503.02812. ICLR 2025
  workshop per mlanthology [likely]; the repo lists only arXiv. Repo https://github.com/NathanGodey/qfilters [verified].
- **Mechanism**: projects keys onto the principal SVD direction of each head's queries, a context-agnostic
  projection that approximates attention scores. Filters are precomputed per model and work with FlashAttention
  [verified].
- **Phase / query**: both (streaming generation and prefill); query-agnostic.
- **Workloads**: streaming-generation perplexity and NIAH [verified]; RULER-type retrieval [likely]. The demo uses
  DeepSeek-R1-Distill-Llama-8B generation [verified]; the paper uses Llama-3.1 and Qwen-2.5 [likely].
- **Budgets**: 99% NIAH at x32 compression, and up to 65% smaller perplexity drop than StreamingLLM [verified].
- **Findings**: competitive with SnapKV on retrieval and better than StreamingLLM on generation [verified].
- **Sources**: https://github.com/NathanGodey/qfilters ; https://arxiv.org/abs/2503.02812 ;
  https://mlanthology.org/iclrw/2025/godey2025iclrw-qfilters/

### C6. KeyDiff
- **Paper**: "KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained
  Environments". Junyoung Park et al. (Qualcomm) [likely]. **NeurIPS 2025** [likely: poster URL]. arXiv 2504.15364.
  No official repo found; implemented in kvpress as KeyDiffPress and BlockPress [verified].
- **Mechanism**: keeps diverse keys and evicts keys most similar to the average key direction. It needs no
  attention scores and supports block-wise prefill under a hard memory cap [likely].
- **Phase / query**: prefill (block-wise); query-agnostic.
- **Workloads**: LongBench with Llama-3.1-8B and Llama-3.2-3B (plus Qwen), and Math500 with
  DeepSeek-R1-Distill-Llama-8B [likely].
- **Budgets**: an 8K budget (about 23% reduction) stays within 0.04% of the full cache on LongBench; up to 30% lower
  E2E latency than other eviction methods [likely].
- **Sources**: https://arxiv.org/abs/2504.15364 ; https://neurips.cc/virtual/2025/poster/115521

### C7. Compactor
- **Paper**: "Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores".
  Vivek Chari, Benjamin Van Durme. arXiv 2507.08143 (2025); venue unknown. Implemented in kvpress as CompactorPress,
  LeverageScorePress and NonCausalAttnPress [verified]; the compactor-vllm repo URL is unknown.
- **Mechanism**: blends approximate leverage scores (outliers in key space) with non-causal chunked attention. A
  context-calibrated procedure estimates the maximum compression each context supports [likely].
- **Phase / query**: prefill; query-agnostic.
- **Workloads**: 27 RULER and LongBench tasks, with the Qwen 2.5 and Llama 3.1 families [likely].
- **Budgets**: matches competitors while keeping 20% fewer tokens; calibrated mode reaches full-KV LongBench
  performance with 68% less KV on average [likely].
- **Findings**: more task-robust than other methods, and compressibility varies per context [likely].
- **Sources**: https://arxiv.org/abs/2507.08143 ; https://github.com/NVIDIA/kvpress

### C8. TRIM-KV (and DBTrimKV)
- **Paper**: "Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs". Ngoc Bui, Shubham Sharma,
  Simran Lamba, Saumitra Mishra, Rex Ying. arXiv 2512.03324. **ICLR 2026** [likely: mlanthology].
  Repo https://github.com/ngocbh/trimkv [verified].
- **Variant**: DBTrimKV ("Make Each Token Count", arXiv 2605.09649 [likely: README badge]) adds a dynamic global
  budget with PagedTrimKVCache.
- **Mechanism**: a learned retention gate gives each token a query-agnostic intrinsic importance at creation time,
  which decays exponentially; the lowest are evicted once the budget is exceeded. Trained by distillation plus a
  capacity loss with the backbone frozen [verified].
- **Phase / query**: both; query-agnostic.
- **Workloads** [verified]:
  - Long-horizon generation: GSM8K, MATH-500, AIME-24, LongProc.
  - Long-context understanding: SCBench, LongMemEval, LongBench, LongBench-v2.
  - Multimodal: lmms-eval and MMDU.
  - Baselines: R-KV, SeerAttention, SnapKV, StreamingLLM, H2O, KeyDiff, Locret.
  - Models: Qwen3-1.7B/4B/8B/14B (math), Qwen3-4B-Instruct-2507, Phi-3-mini-128k.
- **Budgets**: training M=512 for math (16K max) and 2048-4096 for long context; DBTrimKV M=128 [verified].
- **Findings**: up to 198% relative gain over heuristic baselines on math, often above the full cache; about 2x decode
  throughput at 32K; the biggest gains are at low memory [likely].
- **Sources**: https://github.com/ngocbh/trimkv ; https://arxiv.org/abs/2512.03324 ;
  https://mlanthology.org/iclr/2026/bui2026iclr-cache/

---------------------------------------------------------------------------------------------------------------

## D. Decode-heavy workloads: reasoning traces and long-form generation

### D1. SCOPE
- **Paper**: "SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation". Jialong Wu, Zhenglin Wang,
  Linhai Zhang, Yilong Lai, Yulan He, Deyu Zhou. **ACL 2025** (long; oral per repo) [verified: anthology URL].
  arXiv 2412.13649. Repo https://github.com/Linking-ai/SCOPE [verified].
- **Mechanism**: separates prefill and decode. The prefill KV is kept (or handled by any prefill compressor), and
  decode-phase heavy hitters are selected with slide, adaptive and discontinuous strategies [verified].
- **Phase / query**: both (decode-focused); query-aware.
- **Workloads**: LongGenBench (Liu et al.) [verified scripts].
  - 4K: GSM8K+ (K=30), MMLU+ (30), CSQA+ (40).
  - 8K: GSM8K++ (60), MMLU++ (60), CSQA++ (80).
  - Model: Llama-3.1-8B-Instruct; others unknown.
  - Baselines: StreamingLLM, H2O, PyramidInfer, SnapKV, PyramidKV.
- **Budgets**: prefill 2048 ("in paper"), decoding window 512/1024, recent window 256 [verified].
- **Output lengths**: about 4K / 8K tokens [likely].
- **Findings** [verified README]:
  - Excessive prefill compression impairs comprehension of reasoning tasks.
  - Heavy hitters drift during long outputs: tokens important at prefill or early decode stop being the ones
    needed later.
- **Sources**: https://github.com/Linking-ai/SCOPE ; https://aclanthology.org/2025.acl-long.529/ ; https://arxiv.org/abs/2412.13649

### D2. MorphKV
- **Paper**: "Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs". Ravi Ghadia et al.
  **ICML 2025** (PMLR 267). arXiv 2503.00979. Repo https://github.com/ghadiaravi13/MorphKV [verified].
- **Mechanism**: a constant-size cache holds a recent window plus older tokens chosen by their correlation with the
  recent tokens' attention. The cache is re-evaluated at every step, unlike SnapKV's one-shot pruning [verified].
- **Phase / query**: both; query-aware.
- **Workloads** [verified]:
  - LongGenBench (Wu et al.): CR/Once/Range/Periodic.
  - LongWriter: Mistral-Large-123B judge plus length.
  - LongBench: 15 tasks.
  - Models: Llama-3.1-8B, Mistral-7B-v0.2, Qwen2.5-7B, Phi-4-14B [likely; README tables show all four families].
- **Output lengths**: up to 12K-token responses [verified].
- **Findings**:
  - Up to 52.9% memory savings and 18.2% higher accuracy vs SnapKV/H2O [verified].
  - Accuracy drops about 10% as outputs grow to 12K, vs 15-18% for prior methods [verified].
  - SnapKV's cache keeps growing with response length, up to 13x more KV [likely].
  - On LongWriter, compressed variants can beat full attention: Llama with MorphKV scores 69.5 vs 66.5 [verified].
- **Sources**: https://github.com/ghadiaravi13/MorphKV ; https://arxiv.org/abs/2503.00979 ;
  https://proceedings.mlr.press/v267/ghadia25a.html

### D3. R-KV
- **Paper**: "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models". Zefan Cai et al. [likely].
  **NeurIPS 2025** [verified: repo title]. arXiv 2505.24133. Repo https://github.com/Zefan-Cai/R-KV [verified].
- **Mechanism** [verified]:
  - New tokens are staged in a 128-token buffer.
  - Score = lambda x attention importance (from the last 8 observation queries) - (1 - lambda) x redundancy (key
    cosine similarity, keeping the most recent near-duplicates), with lambda = 0.1.
  - The top-k tokens are kept within the budget.
- **Phase / query**: decode only (explicitly not a prefill compressor); query-aware plus redundancy.
- **Workloads** [verified]:
  - MATH-500 and AIME 2024; pass@1 over 64 samples at T=0.6, top-p 0.95.
  - Models: DeepSeek-R1-Distill-Llama-8B and DeepSeek-R1-Distill-Qwen-14B.
  - The serving ports (vLLM/SGLang) are validated on GSM8K with Qwen2.5-Math-7B.
  - SnapKV is adapted to decode by compressing every 128 tokens.
- **Budgets** [verified]:

  | Model | Dataset | Lossless at ratio | Lossless at fixed tokens |
  |---|---|---|---|
  | R1-Llama-8B | MATH-500 | 34% | 1024 |
  | R1-Llama-8B | AIME24 | 10% | 1536 |
  | R1-Qwen-14B | MATH-500 | 54% | 1536 |
  | R1-Qwen-14B | AIME24 | 25% | 3072 |

  - It reaches 105% of full accuracy at a 16% budget (Llama-8B) and 33% (Qwen-14B).
- **Output lengths** [verified]:
  - Average tokens per solution: MATH-500 2,979 (max 16,384) and AIME24 15,536 (max 32,768).
  - One R1-8B run can produce 32K tokens and 4.1 GB of KV.
- **Findings** [verified]:
  - Prompt-oriented compressors keep only about 60% of accuracy at a 10% cache on reasoning traces.
  - Redundant self-checks attend heavily to themselves, so attention-only scoring keeps repetitive tokens.
  - Removing redundancy can push accuracy above the full cache.
  - Throughput rises up to 6.6x (ratio budget) or 9x (fixed budget) through larger batches.
- **Sources**: https://github.com/Zefan-Cai/R-KV ; https://arxiv.org/abs/2505.24133

### D4. LazyEviction
- **Paper**: "LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning".
  Haoyue Zhang, Hualei Zhang, Xiaosong Ma, Jie Zhang, Song Guo. arXiv 2506.15969. **ACL 2026** (long)
  [likely: anthology URL 2026.acl-long.1683]. Repo https://github.com/Halo-949/LazyEviction [verified].
- **Mechanism**: tracks each token's maximum recurrence interval of high attention. It evicts only every W steps,
  within an observation window, and deprioritizes tokens whose recurrence pattern predicts they will be needed
  again; the recent W tokens are always kept [verified/likely].
- **Phase / query**: decode; query-aware.
- **Workloads**: GSM8K, MATH(-500), AIME, with DeepSeek-R1-Distill-Llama-8B and DeepSeek-R1-Distill-Qwen-7B
  [verified scripts].
- **Budgets**: 30-50% budgets come close to the full cache, i.e. 50-70% memory saved; scripts use
  max_kv_capacity 1334/1492 [verified].
- **Output limits**: max_new_tokens 4096 (GSM8K), 8192 (MATH), 16384 (AIME) [verified].
- **Findings** [verified README]:
  - "Token Importance Recurrence": many tokens regain high attention after many steps.
  - Step-wise greedy eviction (on current or cumulative attention) drops these periodically critical tokens.
  - It beats the full cache on MATH-500 and AIME at a 50% budget.
- **Sources**: https://github.com/Halo-949/LazyEviction ; https://arxiv.org/abs/2506.15969 ;
  https://aclanthology.org/2026.acl-long.1683/

### D5. Reasoning Path Compression (RPC)
- **Paper**: "Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning".
  Jiwon Song, Dongwon Jo, Yulhwa Kim, Jae-Joon Kim. **NeurIPS 2025**. arXiv 2505.13866.
  Repo https://github.com/jiwonsong-dev/ReasoningPathCompression [verified].
- **Mechanism**: every P generated tokens, it compresses the trajectory's KV by ratio c, using importance from the R
  most recent queries [verified].
- **Phase / query**: decode; query-aware.
- **Workloads**: AIME 2024 (n=8), LiveCodeBench v5 (n=4), IFEval (n=1), with DeepSeek-R1-Distill-Qwen-7B and QwQ-32B
  [verified].
- **Budgets**: P=1024 or 4096, R=32, c=4 (4x) [verified].
- **Findings**: reasoning paths are semantically sparse. Throughput rises up to 1.60x with a 1.2% pass@1 drop on
  AIME 2024 [verified]. IFEval probes instruction following; its outcome is not verified.
- **Sources**: https://github.com/jiwonsong-dev/ReasoningPathCompression ; https://arxiv.org/abs/2505.13866

### D6. ThinKV
- **Paper**: "ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models". arXiv 2510.01290.
  **ICLR 2026 Oral** [likely: iclr.cc oral URL]. First author unknown; no repo found.
- **Mechanism**: attention sparsity splits the chain of thought into reasoning, execution and transition thoughts.
  A hybrid quantization-eviction scheme sets precision by thought importance and progressively evicts the less
  critical thoughts. A PagedAttention extension reuses the freed slots without compaction [likely].
- **Phase / query**: decode; query-aware.
- **Workloads**: MATH-500, AIME, GSM8K, LiveCodeBench [likely].
  - Models: R1-Distill-Llama-8B/70B, R1-Distill-Qwen-14B, GPT-OSS-20B/120B, QwQ-32B, AceReason-Nemotron-14B,
    MobileLLM-R1-950M [likely].
- **Budgets**: near-lossless with under 5% of the KV; up to 5.8x throughput; a 1024-token budget is competitive
  [likely].
- **Output lengths**: average generation 9,020 tokens (AIME), 14,166 (LiveCodeBench), 2,468 (MATH-500) [likely].
- **Findings**: harder data (AIME) contains more transition thoughts than MATH-500, and thought types differ in
  importance [likely].
- **Sources**: https://arxiv.org/abs/2510.01290 ; https://iclr.cc/virtual/2026/oral/10009981 ; https://openreview.net/forum?id=M3CeHnZKNC

### D7. Think Clearly
- **Paper**: "Think Clearly: Improving Reasoning via Redundant Token Pruning". arXiv 2507.08806. **Findings of
  EMNLP 2025** [verified: anthology URL]; also listed at ICML 2025, probably a workshop [likely]. From Amazon; first
  author unknown this session.
- **Mechanism**: an instruction and an end-of-thinking token are inserted at the end of each reasoning step. Tokens
  are scored by their attention to that token, and structure-aware pruning removes low-contribution reasoning chunks
  first [likely].
- **Phase / query**: decode; query-aware.
- **Workloads**: MATH-500, Minerva, GaoKao, AIME2024, AIME2025, AMC2023 [likely]; model DeepSeek-R1-Distill-Qwen-7B
  (others unknown) [likely].
- **Findings**: reasoning paths are redundant. Attention is widely scattered, more so for incorrect answers. Pruning
  raises AMC2023 accuracy from 75.0% to 82.5% while cutting KV memory by 10.3% [likely].
- **Sources**: https://arxiv.org/abs/2507.08806 ; https://aclanthology.org/2025.findings-emnlp.1169/ ;
  https://www.amazon.science/publications/think-clearly-improving-reasoning-via-redundant-token-pruning

### D8. G-KV
- **Paper**: "G-KV: Decoding-Time KV Cache Eviction with Global Attention". Mengqi Liao, Lu Wang, Chaoyun Zhang,
  Zekai Shen, Xiaowei Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Huaiyu Wan. arXiv 2512.00504 (2025).
  Venue unknown (OpenReview submission). Repo https://github.com/microsoft/G-KV [verified].
- **Mechanism**: a global score mixes local attention with cached historical attention (decayed by alpha), plus
  R-KV-style redundancy suppression. Post-training (GRPO RL, SFT, distillation) adapts the model to decoding with a
  compressed KV [verified].
- **Phase / query**: decode; query-aware.
- **Workloads**: AMC 2023, AIME 2024 (pass@1) and LiveCodeBench [verified/likely].
  - Models: DeepSeek-R1-Distill-Qwen-7B plus an RL checkpoint [verified], and an R1-distilled Llama-3.1 [likely].
  - Baselines: FullKV, StreamingLLM, H2O, SnapKV, R-KV [verified].
- **Budgets**: e.g. budget 2048 with 4096 max new tokens [verified script].
- **Findings**: eviction on local attention overlooks tokens' long-term importance, and training the model under
  compression narrows the gap [likely].
- **Sources**: https://github.com/microsoft/G-KV ; https://arxiv.org/abs/2512.00504

### D9. LaCache
- **Paper**: "LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models".
  Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov,
  Yingyan (Celine) Lin. **ICML 2025**. arXiv 2507.14204. Repo https://github.com/GATECH-EIC/LaCache [verified].
- **Mechanism**: a training-free ladder pattern in which each layer keeps a different span of positions (span and
  overlap parameters), plus iterative compaction for continuous generation [verified/likely].
- **Phase / query**: both; query-agnostic.
- **Workloads** [verified]:
  - Wikitext-2 and PG19 perplexity, LongBench, NIAH up to 128K.
  - Models: Llama-3-8B, Llama-2-7B-chat, SmolLM2-1.7B-Instruct, Llama-3.2-3B-Instruct.
- **Budgets**: 256/512 tokens (perplexity); 25%/50% (LongBench/NIAH) [verified].
- **Sources**: https://github.com/GATECH-EIC/LaCache ; https://arxiv.org/abs/2507.14204

### D10. DMS: "Inference-Time Hyper-Scaling with KV Cache Compression"
- **Paper**: first author Adrian Łańcucki [likely: OpenReview PDF header]; co-authors unverified. **NeurIPS 2025**
  [likely: poster URL]. arXiv 2506.05345. No official repo found: kvpress has DMSPress without the trained evictors
  [verified]; a third-party port is shisa-ai/FastDMS.
- **Mechanism**: retrofits a learned per-token eviction decision with delayed eviction, trained by logit
  distillation in about 1K steps. The saved memory is spent on longer or more parallel reasoning ("hyper-scaling")
  [likely].
- **Phase / query**: both; query-agnostic (learned).
- **Workloads** [likely]:
  - Reasoning: AIME 24, MATH 500, GPQA Diamond, LiveCodeBench with Qwen-R1 1.5B/7B/32B.
  - Long context: NIAH and variable tracking.
  - Short context: MMLU, GSM8K, HellaSwag.
- **Budgets**: 4x-8x compression [likely].
- **Findings** [likely]:
  - Generation cost is bottlenecked by KV size rather than token count.
  - At equal memory and compute, it gains about +9.1 (AIME24), +7.6 (GPQA) and +9.6 (LiveCodeBench) points,
    reported for Qwen-R1 32B.
  - It beats training-free sparse attention at high compression, and exceeds vanilla on NIAH and variable tracking.
- **Sources**: https://arxiv.org/abs/2506.05345 ; https://neurips.cc/virtual/2025/poster/119605 ; https://openreview.net/forum?id=8ZiElzQxf1

### D11. RLKV: "Which Heads Matter for Reasoning? RL-Guided KV Cache Compression"
- **Paper**: Wenjie Du, Li Jiang, Keda Tao, Xue Liu, Huan Wang. arXiv 2510.08525. **ICML 2026** [likely: icml.cc
  poster URL]; also LIT@ICLR 2026 workshop [verified README]. Repo https://github.com/Kurt232/RLKV [verified].
- **Mechanism** [verified]:
  - A gate on each KV head mixes full attention with sink+local attention.
  - Gates are trained with GRPO on verifiable math rewards plus an L1 sparsity penalty, stabilized by
    self-distillation sampling and adaptive penalty weighting.
  - At inference the top-k heads keep the full KV; the rest keep 16 sink + 64 local tokens.
- **Phase / query**: both; query-agnostic (static head allocation).
- **Workloads** [verified]:
  - Reasoning: GSM8K, MATH-500, AIME24, MBPP.
  - Knowledge: MMLU-Pro Chemistry, CS, Law, Physics.
  - Long-context reasoning: LongReason-64K (400 samples, 70K context).
  - Models: DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, Qwen3-4B-Thinking-2507.
  - Baselines: H2O, R-KV, DuoAttention, KVzip.
- **Budgets**: sparsity 0.2-0.8 (fraction of heads compressed); near-lossless at 20-50% [verified].
- **Output lengths**: average sequence length about 3000 on MATH-500 (Llama-3.1-8B-R1) [verified].
- **Findings** [verified]:
  - Token dropping breaks reasoning chains. On LongReason-64K, R-KV scores 0.0 at every sparsity (outputs
    degenerate into repetitive loops) vs 49.25 for the full cache.
  - Retrieval-oriented head reallocation (DuoAttention, KVzip head scores) preserves some reasoning but degrades
    steeply. At sparsity 0.8, DuoAttention scores 1.5 and KVzip 4.75, vs RLKV 15.0.
  - A small fraction of heads carries reasoning, and those heads also control when generation terminates.
  - GQA group size limits sparsification (Qwen-7B-R1 has a group size of 7).
  - End-to-end speedup in SGLang is 1.19-2.06x at 20-60% sparsity.
- **Sources**: https://github.com/Kurt232/RLKV ; https://arxiv.org/abs/2510.08525 ; https://icml.cc/virtual/2026/poster/62599

### D12. Analysis: "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning"
- **Paper**: arXiv 2512.12008. NeurIPS 2025 listing, probably a workshop [likely]. First author likely Minghui Liu
  (repo owner) [likely]. Code https://github.com/minghui-liu/kvpress, a kvpress fork with decoding-aware presses
  [verified].
- **Setup** [verified]:
  - Presses: decode-time H2O, SnapKV-D, StreamingLLM, K-Norm, R-KV, ShadowKV, and 20+ others.
  - Data: GSM8K, MATH-500, FOLIO, DROP, StrategyQA, ReClor, CommonsenseQA, OpenBookQA, LogiQA, AIME24/25.
  - Budgets {128, 256, 384, 512}, with a 2K decoding limit.
  - Models: Llama-3.1-8B-Instruct, Llama-3.1-Nemotron-Nano-8B-v1, DeepSeek-R1-Distill Qwen/Llama.
- **Findings** [verified README]:
  1. For the non-reasoning Llama-3.1-8B-Instruct no single press is best. H2O and StreamingLLM do well on reading
     comprehension; SnapKV-D is stronger on short prompts with long chains of thought.
  2. For reasoning models, heavy-hitter tracking (H2O, SnapKV-D) dominates and often matches or beats the full cache
     at a 256-token budget.
  3. At low budgets, similarity pruning (R-KV, K-Norm) can lengthen reasoning traces, trading cache size against
     total decode cost.
- **Sources**: https://github.com/minghui-liu/kvpress ; https://raw.githubusercontent.com/minghui-liu/kvpress/main/reason/README.md ;
  https://arxiv.org/abs/2512.12008

### D13. Analysis + method: KVFundaBench / ShotKV
- **Paper**: "Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression",
  arXiv 2502.01941. ICML 2026 per a paper-notes index [likely]. Authors unknown this session.
- **Findings** [likely]:
  - Retrieval tasks stay robust, but reasoning shows task-dependent degradation because broken links in the chain of
    thought cannot be recovered.
  - Arithmetic reasoning (about 4K-token generations) loses more than 20% below a 30% compression ratio.
  - Few-shot examples must be kept as whole units. ShotKV, which retains whole shots and separates prefill from
    decode, keeps 80.37% on many-shot arithmetic at a 10% ratio vs about 51% for random shots.
  - It gains 9-18% on long-context generation.
- **Sources**: https://arxiv.org/html/2502.01941 ;
  https://en.papernotes.org/ICML2026/model_compression/semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in/

### D14. Analysis: "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving"
- **Paper**: **MLSys 2025** [verified: proceedings URL]. arXiv 2503.24000. Artifact
  https://github.com/LLMkvsys/rethink-kv-compression [verified: README read from the copy fetched during the survey].
- **Findings** [verified README]:
  1. Current implementations (FlashAttention, PagedAttention) are not optimized for production serving, so
     compressed KV gives suboptimal throughput.
  2. KV compression can lengthen outputs, which raises end-to-end latency.
  - The artifact includes length analysis, a negative-sample analysis, and throughput / length predictors.
  - A per-sample analysis showing intrinsic task limits is reported [likely].
- **Output lengths, re-computed this session from the artifact's `benchmark_len` outputs** (Llama-3-8B-Instruct,
  1000 ShareGPT requests, max 1024 new tokens) [verified computation]:

  | Config | Mean length | Ratio to full | Requests ≥1.5x longer | Hit 1024 cap | Degenerate outputs |
  |---|---|---|---|---|---|
  | Full | 412.9 | 1.00 | n/a | 253 | 174 |
  | H2O, 256-token cache | 444.2 | 1.08 | 21.4% | 285 | 204 |
  | H2O, 512-token cache | 414.5 | 1.00 | 12.1% | n/r | n/r |
  | StreamingLLM, 256 | 426.0 | 1.03 | 26.4% (23.4% ≤0.67x) | n/r | n/r |
  | KIVI 2-bit | n/r | 1.12 | n/r | n/r | n/r |
  | GEAR 4-bit | n/r | 1.24 | n/r | n/r | 310 |

  - The mean shift is modest at tight eviction budgets, but per-request length variance is large.
  - 15.2% of requests that terminated under the full cache hit the cap under H2O-256.
  - 107 outputs are empty in every config (a data artifact).
  - Degenerate outputs use the repeated-line heuristic from a working script (not retained).
  - n/r = not recorded in these notes.
- **Sources**: https://github.com/LLMkvsys/rethink-kv-compression ; https://arxiv.org/abs/2503.24000 ;
  https://proceedings.mlsys.org/paper_files/paper/2025/hash/26289c647c6828e862e271ca3c490486-Abstract-Conference.html

---------------------------------------------------------------------------------------------------------------

## E. Multi-turn / conversation / agentic workloads

### E1. EpiCache
- **Paper**: "EpiCache: Episodic KV Cache Management for Long Conversational Question Answering" (v4 title: "...for
  Long-Term Conversation on Resource-Constrained Environments"). Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit,
  Minsik Cho (Apple). arXiv 2509.17396; venue: arXiv [verified repo]. Repo https://github.com/apple/ml-epicache
  [verified].
- **Mechanism** [verified]:
  - Block-wise prefill eviction bounds cache growth.
  - History is clustered into topical episodes (4 clusters, 8 medoids, sentence or Qwen embeddings).
  - Each episode gets its own KV eviction, driven by medoid "patched prompts"; the incoming question is matched to
    an episode.
  - Layer budgets follow sensitivities profiled on BookSum, with an optional head-wise allocation.
- **Phase / query**: prefill; query-agnostic at compression time (episode-conditioned).
- **Workloads** [verified]:
  - LoCoMo; Realtalk (GPT-judge); LongMemEval as custom ~100K-token conversations of stacked sessions.
  - Models: Qwen2.5-3B/7B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct.
  - Baselines: KVzip, KeyDiff, SnapKV, InfiniPot.
- **Budgets**: near-full accuracy at 4-6x compression [likely].
- **Findings**: up to 40% accuracy gain over baselines, with up to 2.4x lower latency and 3.5x lower memory [likely].
  Topic-conditioned caches matter in long chats.
- **Sources**: https://github.com/apple/ml-epicache ; https://arxiv.org/abs/2509.17396

### E2. FlowKV
- **Paper**: "FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management".
  Xiang Liu, Hong Chen, Xuming Hu, Xiaowen Chu. arXiv 2505.15347. NeurIPS 2025 Workshop on Multi-Turn Interactions
  [likely]. No repo found.
- **Mechanism**: a training-free wrapper that leaves the already-compressed KV of earlier turns untouched and
  compresses only the latest completed turn, so old turns are never re-compressed [likely].
- **Phase / query**: applied at turn boundaries; query-awareness comes from the base method.
- **Workloads**: Multi-IF (multi-turn instruction following) and PrefEval (preference retention). Base methods:
  SnapKV, StreamingLLM, ExpectedAttention, ChunkKV [likely]. Models unknown.
- **Findings**: re-compressing early turns causes forgetting that grows in later turns. Isolation lifts later-turn
  instruction-following / preference retention from 10.90% to 75.40% [likely].
- **Sources**: https://arxiv.org/abs/2505.15347 ; https://neurips.cc/virtual/2025/loc/san-diego/127972

### E3. RocketKV (including its multi-turn variant)
- **Paper**: "RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression". Payman Behnam
  et al. [likely]. **ICML 2025**. arXiv 2502.14051. Repo https://github.com/NVlabs/RocketKV [verified].
- **Mechanism** [verified]:
  - Stage 1: SnapKV permanently evicts input tokens at a coarse grain.
  - Stage 2: hybrid sparse attention at decode reduces the sequence and head dimensions to approximate scores, then
    takes a top-k.
  - The target ratio is split adaptively between the stages, and there is a multi-turn variant.
- **Phase / query**: both; query-aware.
- **Workloads** [verified]:
  - LongBench, NIAH (Paul Graham passkey), RULER, SCBench (multi-turn).
  - Models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, LongChat-7B-v1.5-32k.
- **Budgets** [verified]: token budgets of 256-512; up to 400x compression; 3.7x speedup; 32.6% lower peak memory.
- **Findings**: permanent eviction is fragile in multi-turn use; the multi-turn variant approaches oracle top-k
  attention [verified].
- **Sources**: https://github.com/NVlabs/RocketKV ; https://arxiv.org/abs/2502.14051 ; https://proceedings.mlr.press/v267/behnam25a.html

### E4. LoopServe
- **Paper**: "LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues".
  arXiv 2507.13681. Venue, authors and repo unknown.
- **Mechanism**: online attention sparsification in prefill, plus progressive KV compression during decode driven by
  recent output tokens [likely].
- **Phase / query**: both; query-aware.
- **Workloads**: a new benchmark of 11 multi-turn datasets with realistic query positions and cross-turn
  dependencies [likely].
- **Findings**: static, position-based heuristics fail when the query position and the relevant context shift
  across turns [likely].
- **Sources**: https://arxiv.org/abs/2507.13681 ; https://openreview.net/forum?id=iyIzaoDVrT

### E5. Analysis: SCBench
- **Paper**: "SCBench: A KV Cache-Centric Analysis of Long-Context Methods". **ICLR 2025** [verified: poster URL].
  arXiv 2412.10319. Code https://github.com/microsoft/MInference/tree/main/scbench [verified].
- **Setup**:
  - 12 tasks [verified]:
    - String retrieval: kv, prefix_suffix, vt.
    - Semantic retrieval: repoqa, qa_eng, qa_chn, choice_eng.
    - Global information: many_shot, mf, summary.
    - Multi-task: summary_with_needles, repoqa_and_kv.
  - Two shared-context modes, multi-turn and multi-request [verified].
  - Models: Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, GLM-4-9B, plus Codestral-Mamba and Jamba-1.5-Mini
    [likely].
- **Findings** [likely]:
  - Methods with sub-O(n) memory suffer in multi-turn use, while O(n)-memory sparse encoding stays robust.
  - Dynamic sparsity beats static patterns.
  - Attention distributions shift during long generation.
- **Sources**: https://arxiv.org/abs/2412.10319 ; https://iclr.cc/virtual/2025/poster/28803

### E6. Multi-turn evidence in other entries
KVzip (C1: SCBench multi-query, SQuAD multi-question, update_cache), TRIM-KV (C8: LongMemEval, SCBench) and RocketKV (E3:
SCBench multi-turn) [verified].

### E6b. Analysis + fix: "The Pitfalls of KV Cache Compression" (multi-instruction / system-prompt workloads)
- **Paper**: **ACL 2026** (long; anthology 2026.acl-long.1926). arXiv 2510.00231. Repo
  https://github.com/Itisalex2/pitfalls-of-kv-cache-compression [verified: README copy fetched during the survey].
- **Setup** [verified README]:
  - Compression methods: StreamingLLM, SnapKV, ObservedAttention, TOVA, Knorm, plus "fair eviction" variants of each.
  - Benchmarks: a system-prompt IFEval variant (instruction following) and RaccoonBench (prompt leakage).
  - Compression-ratio sweeps from 0 to 0.95.
  - Models: Llama-3.2-1B-Instruct (default) and Qwen2.5-14B-Instruct (example).
- **Findings** [verified TL;DR]: with prompts that contain multiple instructions, KV compression can cause some
  instructions to be ignored and can affect leakage. Simple changes to the eviction policy fix this. This matters
  for agent-style prompts with long system prompts and many constraints.
- **Sources**: https://github.com/Itisalex2/pitfalls-of-kv-cache-compression ; https://arxiv.org/abs/2510.00231 ;
  https://aclanthology.org/2026.acl-long.1926/

### E7. Agentic KV eviction (2026). Titles and IDs come from search listings; content is [likely] only where a snippet described it.
- **AgentKV**: "Phase-Aware KV Eviction for Agentic LLMs", arXiv 2609.14872. Multi-turn agent trajectories mix query
  distributions from different phases (principal-angle analysis), which biases recency-based eviction; it keeps a
  persistent cache lifecycle across turns [likely]. Benchmarks unknown.
- **StepKV**: "Step-Aware KV Cache Compression for LLM Agents", arXiv 2609.22158. Evaluated on HotpotQA, 2WikiMultihopQA,
  MuSiQue and BrowseComp-Plus (trajectories over 100K tokens) [likely].
- **Nexus Sampling**: "Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction Under Fixed Budgets",
  arXiv 2606.23961. Runs a multi-turn coding agent over a growing trace, scored by Resolved/Pass@1 on 50 SWE-bench
  tasks [likely].
- **CommitKV**: "Lifecycle-Aware KV Cache Compression via Commit Transitions for Multi-Turn Agents", arXiv 2608.07855
  [title only].
- **IntentKV**: "Cross-Turn Intent-Aware KV Cache Pruning for Agent Inference", arXiv 2606.09916 [title only].
- "Practical Online KV Cache Compaction for LLM Agents: An Empirical Study", arXiv 2608.00902 [title only].
- "Efficient Long-Horizon GUI Agents via Training-Free KV Cache Compression", arXiv 2603.00188 [title only].
- "Learning Agent Execution for KV-Cache Management in Agentic Serving", arXiv 2608.14624 [title only].
- **Continuum** (earlier CacheTTL): "Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live",
  arXiv 2511.02230. A serving-level policy that pins whole requests' KV across tool calls with a TTL, not token
  eviction. Evaluated on SWE-Bench, BFCL and OpenHands with Llama-3.1-8B/70B, Gemma-3-12B and GLM-4.5-355B; over 8x
  better average job completion time [likely].
- "Agentic AI Workload Characterization", arXiv 2605.26297 [title only].

---------------------------------------------------------------------------------------------------------------

## F. Other leads, not characterized (titles and IDs seen in search listings or kvpress)

- **Reasoning-decode eviction (2025-26)**:
  - ForesightKV (2602.03203)
  - "Value-Aware Stochastic KV Cache Eviction for Reasoning Models" (2606.03928)
  - "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning" (2609.03430)
  - "Epiphany-Aware KV Cache Eviction Without the Attention Matrix" (2606.26472)
  - Zipage (2603.08743)
  - "Information-Aware KV Cache Compression for Long Reasoning" (2606.26875)
  - "Adaptive Mass-Segmented KV Compression for Long-Context Reasoning" (2605.23200)
  - "Learning to Evict from Key-Value Cache" (2602.10238)
  - DELTA (2510.09883)
  - "Beyond Speedup -- Utilizing KV Cache for Sampling and Reasoning" (2601.20326)
- **Head/layer budgets**:
  - DefensiveKV (2510.13334)
  - CriticalKV (2502.03805)
  - "Exploring a Layer-Wise Design Space for KV Cache Eviction" (2606.15157)
  - HeadWiseKV (2609.02029)
  - ZigzagAttention (2508.12407)
  - CateKV (2608.30295)
  - KV-Compress (2410.00161)
  - LUKV (2602.08585)
  - "Beyond Token Eviction: Mixed-Dimension Budget Allocation" (2603.20616)
- **Query-agnostic / reusable**:
  - ChunkKV (2502.00299)
  - LagKV (2504.04704)
  - KVCompose (2509.05165)
  - CURPress (2509.15038)
  - RestoreKV (2608.01247)
  - CapPress (2604.25975)
  - DropKV and KVgrad (OpenReview)
  - "Still: Amortized KV Cache Compaction in a Single Forward Pass" (2606.07878)
  - Draft-based approximate inference / SpecKV (2506.08373)
- **Benchmarks and analyses**:
  - The Sparse Frontier (2504.17768); its repo added AIME24/25, MATH500 and Qwen3 in Dec 2025 [verified].
  - "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving"
    (2607.05399)
  - KVDiagnosis (2608.09412)
  - "How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit" (2607.11942)
  - "A Probabilistic Interpretation of KV Cache Eviction" (2608.28293)
- **Recalled only, not found this session [unverified]**: Lethe, Multipole Attention, LessIsMore, Kinetics, LAVa.

---------------------------------------------------------------------------------------------------------------

## G. Cross-cutting observations for the workload taxonomy

### 1. Workload expansion
Newer head/layer-budget methods still test mostly on LongBench + NIAH, sometimes with RULER, NeedleBench, LooGLE or
BABILong. That is long-prompt, short-answer, single-query prefill. This covers Ada-KV, CAKE, HeadKV, DuoAttention,
RazorAttention, DynamicKV, ThinK and WindowKV. From 2025 on, papers add new workload types:

- **Decode-heavy reasoning**:
  - Benchmarks: AIME24/25, MATH-500, GSM8K, GPQA-D, LiveCodeBench, AMC23, Minerva, GaoKao, MBPP, IFEval.
  - Models: R1-distilled Qwen/Llama, QwQ-32B, Qwen3, GPT-OSS, AceReason, Nemotron.
- **Long-form generation**: both LongGenBench variants, LongWriter, LongProc.
- **Context reuse / multi-turn**: SCBench, SQuAD multi-question, LoCoMo, Realtalk, LongMemEval, Multi-IF, PrefEval.
- **Long-context reasoning**: LongReason-64K.
- **Agentic traces**: SWE-bench, BFCL, OpenHands, BrowseComp-Plus. These appear only in 2026 preprints.

### 2. Recurring task-sensitivity findings
- **Retrieval vs aggregation**: TopK/eviction works on NIAH but fails on CWE/FWE because attention mass is long-tailed
  (MagicPIG). Global-information and multi-turn tasks hurt sub-O(n) methods (SCBench).
- **Retrieval vs reasoning**: retrieval stays robust while reasoning degrades task-dependently (KVFundaBench). Retrieval-head
  criteria miss the heads reasoning needs (RLKV). Token dropping can collapse into repetition loops on long-context
  reasoning (RLKV on LongReason-64K).
- **Decode-time dynamics**: heavy hitters drift over long outputs (SCOPE). Important tokens recur after long gaps
  (LazyEviction). Tokens have long-term as well as local importance (G-KV). Thought types show phase structure
  (ThinKV). Agent phases have shifting query distributions (AgentKV).
- **Redundancy**: pruning redundant reasoning can beat the full cache (R-KV at 105%, Think Clearly, LazyEviction,
  RLKV on AIME24).
- **Output-length side effects**: compression can lengthen outputs and E2E latency (MLSys "Rethinking"; R-KV and
  K-Norm at low budgets in "Hold Onto That Thought"). Budgets defined as a percentage of output length are therefore
  endogenous. Re-computed from the MLSys artifact (Llama-3-8B-Instruct, ShareGPT): H2O with a 256-token cache raises
  mean output length only 8%, but 21% of requests become ≥1.5x longer and cap hits rise 253→285. The effect lies in
  the tail and per-request variance, not the mean.
- **Multi-instruction prompts**: compression can drop some instructions and change leakage behaviour ("Pitfalls",
  ACL 2026).
- **Budget vs difficulty**: AIME needs larger absolute budgets than MATH-500, yet can tolerate a smaller ratio (R-KV:
  10% vs 34%; SeerAttention-R tables).
- **Query dependence**: query-aware compression fails on follow-up queries (KVzip, SCBench). Re-compressing earlier
  turns causes forgetting (FlowKV). Topic shifts favour episode-specific caches (EpiCache).
- **Task-dependent layer/head profiles**: DynamicKV (summarization vs code), CAKE, WindowKV's classifier, and Ada-KV's
  question-agnostic setting.
- **Architecture**: the GQA group size caps head sparsification (RLKV).

### 3. Still uncovered
- Token-level eviction on real tool-use agent traces: tau-bench, BFCL, WebArena, terminal/SWE agents. Aside from
  2026 preprints, Continuum works at scheduling level only.
- Reasoning models in multi-turn chat, i.e. thinking tokens carried across turns.
- Long prompt + long output + multi-turn together.
- Cost-normalized reporting that counts extra tokens generated and E2E latency.
- Serving realism: continuous batching, paged memory, prefix-cache sharing.
- MLA, MoE and hybrid linear-attention models.
- Code and structured outputs beyond LiveCodeBench/MBPP, and multilingual workloads.
- Safety and instruction-following under compression are thinly covered: RPC uses IFEval, FlowKV Multi-IF, and
  "Pitfalls" system-prompt IFEval plus RaccoonBench leakage.
