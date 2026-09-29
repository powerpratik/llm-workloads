> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# General LLM task taxonomies and task-level facts for KV-cache workload design

Scope: (A) how academia and industry categorize LLM evaluation tasks; (B) KV-relevant facts for short-context, reasoning, code, instruction-following, multi-turn chat and agentic/tool-use benchmarks. The companion file `general_tasks.jsonl` has one record per benchmark (31 records).

## 0. Method and caveats

- **Token counts.** Unless stated otherwise, lengths were computed in this session from primary data with the **Llama-3 tokenizer** (tiktoken BPE, 128k vocab, loaded from `meta-llama/llama-models`). Other tokenizers differ by roughly ±10–25% (code and LaTeX vary the most).
- **Data sources actually used.** Most data came from GitHub mirrors, plus public S3/GCS buckets:
  - MMLU: `FranxYao/chain-of-thought-hub`
  - ARC: AI2 S3 zip, read with HTTP range requests
  - HellaSwag: OpenCompass core-data release zip
  - Winogrande: AI2 GCS
  - GSM8K: `openai/grade-school-math`
  - MATH-500: `openai/prm800k`
  - AIME24: `QwenLM/Qwen2.5-Math`
  - GPQA: `idavidrein/gpqa`
  - BBH CoT prompts: lm-eval-harness
  - HumanEval, MBPP, IFEval (+ GPT-4 responses), MT-Bench, MT-Bench-101, AlpacaEval (+ GPT-4-Turbo outputs and leaderboard CSV), Arena-Hard v0.1/v2.0 questions and v2.0 model answers, CrossCodeEval, BFCL v4, τ-bench (+ historical trajectories), τ²-bench, Terminal-Bench 2.0 task metadata
  - SWE-bench Verified trajectories: public bucket `s3://swe-bench-submissions` (100 SWE-agent+Claude-3.5-Sonnet, 40 SWE-agent-1.0+Claude-4-Sonnet, 100 OpenHands+Claude-4-Sonnet)
  - MMLU-Pro model outputs: `TIGER-AI-Lab/MMLU-Pro/eval_results`
  - Report PDFs: DeepSeek-R1 PDF (GitHub), Qwen3 Technical Report PDF (GitHub), Gemini 2.5 report (GCS), Microsoft Eureka inference-scaling report (microsoft.com)
- **Blocked sources.** arXiv, HF, LMArena, Artificial Analysis, OpenAI, Anthropic, Meta and Qwen blogs could not be fetched. Web search was exhausted early in the session (shared budget). Facts that come only from search snippets or from memory are tagged `[likely]`. Numbers read from a published figure are tagged `[likely, approx.]`.
- **Confidence tags.**
  - `[verified]`: computed from primary data, or read from a primary source in this session.
  - `[likely]`: a secondary source or confident recall.
  - `[unverified]`: an estimate.
  - `unknown`: no source was reached. No number was invented.

---

## PART A — Capability taxonomies

### A.1 Per-source summaries

**HELM (Liang et al., 2022; `stanford-crfm/helm` schemas)** `[verified from schema_classic.yaml / schema_lite.yaml / schema_capabilities.yaml]`
- **Scenario taxonomy.** Each scenario is tagged with *task* × *what* (domain/genre) × *who* (author/audience) × *when* (period) × *language*. The paper's core idea is a 2-D taxonomy of **scenarios × metrics**.
- **Core scenarios (Classic).** The task tags are:
  - question answering
  - information retrieval
  - summarization
  - sentiment analysis
  - toxicity detection
  - miscellaneous text classification
- **Targeted evaluations (Classic).** These cover:
  - Language: The Pile, TwitterAAE, ICE, BLiMP.
  - Knowledge: NaturalQuestions closed-book, HellaSwag, OpenBookQA, TruthfulQA, MMLU, WikiFact.
  - Reasoning, including code and math: synthetic reasoning, bAbI, Dyck, GSM8K, MATH, APPS, HumanEval, LSAT, LegalSupport, entity matching and imputation.
  - Harms: copyright, disinformation, BBQ, BOLD, RealToxicityPrompts.
  - Efficiency, calibration, robustness (contrast sets / individual perturbations), and in-context/MC/prompt ablations.
- **Metric taxonomy (7 metric groups).** Accuracy, calibration, robustness, fairness, bias, toxicity, efficiency.
- **HELM Lite.** NarrativeQA, NaturalQuestions (open and closed book), OpenbookQA, MMLU, MATH (CoT), GSM8K, LegalBench, MedQA, WMT14. Metrics are accuracy and efficiency only.
- **HELM Capabilities.** MMLU-Pro and GPQA (question answering), IFEval and WildBench (instruction following), Omni-MATH (mathematics). This is a reasoning/IF-era refresh.

**BIG-bench keyword taxonomy (`google/BIG-bench/keywords.md`)** `[verified]`
- There are 9 top-level keyword groups:
  1. traditional NLP tasks (QA, reading comprehension, summarization, translation, dialogue system, coreference, …)
  2. logic, math, code (algorithms, logical reasoning, arithmetic, mathematical proof, computer code, …)
  3. understanding the world (causal, physical, common sense, visual reasoning)
  4. understanding humans (theory of mind, emotional understanding, social reasoning, humor, figurative language)
  5. scientific and technical understanding (biology, chemistry, physics, medicine, domain specific)
  6. mechanics of interaction with the model (self-play, multiple choice vs free response, repeated interaction, zero-/one-/many-shot, show work)
  7. targeting common LM technical limitations (**context length**, **multi-step**, out-of-distribution, instructions, tokenization)
  8. pro-social behavior (alignment, social bias, toxicity, truthfulness, …)
  9. other (creativity, multilingual, low-resource language, riddles, …)
- BIG-bench has 204 tasks `[likely]`.

**BIG-Bench Hard (Suzgun et al., 2022)**
- 23 tasks (27 subtasks) on which earlier LMs underperformed the average human rater `[verified via lm-eval]`.
- The paper groups them into algorithmic/multi-step-arithmetic tasks and NLU/world-knowledge tasks `[likely]`.
- Standard protocol is 3-shot CoT.

**lm-evaluation-harness (EleutherAI)** `[verified]`
- Groups tasks by *benchmark family* (`mmlu`, `bbh`, `minerva_math`, `gsm8k`, …) rather than by capability.
- Uses tags such as `math_word_problems` and meta-groups such as `leaderboard` (Open LLM Leaderboard v2) and `benchmarks/openllm.yaml` (v1).
- The tasks README table lists each family with a description and its language(s).

**Open LLM Leaderboard (Hugging Face)**
- v1 `[verified: lm-eval openllm.yaml]`:
  - ARC-Challenge 25-shot
  - HellaSwag 10-shot
  - TruthfulQA 0-shot (MC2)
  - MMLU 5-shot
  - Winogrande 5-shot
  - GSM8K 5-shot
- v2 `[verified: lm-eval leaderboard README]`:
  - IFEval 0-shot generative
  - BBH 3-shot MC
  - MATH Level-5 4-shot generative (Minerva)
  - GPQA 0-shot MC
  - MuSR 0-shot MC
  - MMLU-Pro 5-shot MC
- Stated intent of v2 `[likely, search snippet]`: knowledge testing, reasoning, complex math, and human-preference-like instruction following, using uncontaminated, harder data.

**OpenCompass** `[verified: README v0.2.0]`
- Dataset categories are:
  - Language: word definition, idiom, semantic similarity, coreference, translation, multilingual QA/summary
  - Knowledge: knowledge QA
  - Reasoning: textual entailment, commonsense, math, theorem application, comprehensive reasoning
  - Examination: junior-high to professional exams, medical exams
  - Understanding: reading comprehension, content summary, content analysis
  - Long Context
  - Safety: safety, robustness
  - Code
- The README's "five capability dimensions" are Language, Knowledge, Reasoning, Examination, Understanding `[likely]`.
- The later CompassRank dimensions add Math, Code, Instruction-following and Agent `[likely]`.

**LiveBench** `[verified: README and changelog]`
- 6 categories: Math, Coding, Reasoning, Language, Data Analysis, Instruction Following.
- A 7th category, **Agentic Coding**, was added on 2025-05-30. It is multi-turn repository issue resolution run with SWE-agent, and since 2025-10-03 with mini-SWE-agent at a **250-step limit**.
- New questions are released monthly to limit contamination.

**Chatbot Arena / LMArena** `[verified: FastChat monitor code and category.py; blog source in lm-sys.github.io]`
- **Task categories:**
  - Coding: the conversation contains code snippets.
  - Math: LLM classifier.
  - Instruction Following: LLM classifier, 0–5 score.
  - Creative Writing: LLM classifier.
  - **Hard Prompts:** a prompt that satisfies 6 or more of 7 criteria (specificity, domain knowledge, complexity, problem-solving, creativity, technical accuracy, real-world application). The criteria were labelled with Llama-3-70B over 1M prompts, and about 20% of prompts score 6 or more.
- **Shape categories:**
  - **Longer Query:** ≥ 500 tokens, about 10% of prompts `[likely]`.
  - **Multi-Turn:** ≥ 2 turns.
  - Exclude Short Query: < 5 tokens.
  - Exclude Refusal.
  - Exclude Ties.
- **Language categories:** English, Chinese, French, German, Spanish, Russian, Japanese, Korean. Vision categories also exist.
- **Arena Expert and occupational categories** (2025) `[likely, search snippet]`:
  - About 5.5% of prompts are tagged "expert".
  - 23 occupational fields map to 8 leaderboards:
    - Software & IT Services, about 28% of prompts
    - Writing/Literature/Language, about 25%
    - Life/Physical/Social Science, about 17%
    - Entertainment/Sports/Media
    - Business/Management/Financial Ops
    - Mathematical
    - Legal & Government
    - Medicine & Healthcare
- Separate arenas cover WebDev, Search, Vision, Copilot and text-to-image.
- Reported vote shares `[likely, LMArena posts]`: Instruction Following about 35%, Math about 13%, Creative Writing about 15%.

**Artificial Analysis Intelligence Index** `[likely, AA pages seen via search snippets; site blocked]`
- **v3** (late 2025): MMLU-Pro, GPQA Diamond, HLE, LiveCodeBench, SciCode, AIME 2025, IFBench, AA-LCR, Terminal-Bench Hard, τ²-Bench Telecom.
- **v4.0** (Jan 2026):
  - Added AA-Omniscience, GDPval-AA and CritPt.
  - Retired MMLU-Pro, AIME25 and LiveCodeBench.
  - Four categories weighted 25% each: **Agents, Coding, Scientific Reasoning, General**.
- **v4.1** (Jun 2026):
  - Terminal-Bench Hard became Terminal-Bench 2.1.
  - τ²-Telecom became τ³-Banking.
  - GDPval-AA became v2, with the turn limit raised from **100 to 250**.
  - IFBench was removed as saturated.
  - Weights: GDPval-AA 20, TB 2.1 16, τ³-Banking 14, HLE 12, Omniscience-Acc 8, SciCode 8, GPQA 6, AA-LCR 6, CritPt 6, Omniscience-non-hallucination 4.
- **v4.2** (Sep 4 2026):
  - Added AA-Briefcase (15%): long-horizon knowledge work with thousands of source files.
  - Added GDP.pdf (10%): 4,592 PDF pages across 100 tasks.
  - Removed GPQA Diamond.
  - Category weights: Agents 30 / Coding 20 / Scientific Reasoning 20 / General 30.
- **v4.3 / v4.3.2** (Sep 7 2026, current):
  - Components: AA-Briefcase v1.1, GDPval-AA v2.1, **AutomationBench-AA** (with Zapier, 657 held-out tasks), Terminal-Bench 4.0, SciCode, HLE, GDP.pdf, CritPt, AA-Omniscience, AA-LCR v1.1.
  - Category weights stay 30/20/30/20.
  - One snippet claimed 25% each for v4.3; AA's own post states 30/20/30/20.

**Model reports**
- **Llama 3 / 3.1** `[verified: llama-models MODEL_CARD.md]`:
  - Pre-trained evaluation: General (MMLU, MMLU-Pro CoT, AGIEval, CommonSenseQA, Winogrande, BBH CoT, ARC-C), Knowledge reasoning (TriviaQA-Wiki), Reading comprehension (SQuAD, QuAC, BoolQ, DROP).
  - Instruct evaluation: General (MMLU, MMLU CoT, MMLU-Pro CoT, IFEval), Reasoning (ARC-C, GPQA), Code (HumanEval, MBPP++, MultiPL-E), Math (GSM8K CoT, MATH CoT), Tool Use (API-Bank, BFCL, Gorilla, Nexus), Multilingual (MGSM).
  - The "Herd" paper adds Long context (ZeroSCROLLS, InfiniteBench, NIH) `[likely]`.
- **Qwen3 report** `[verified: PDF]`:
  - Base models: General, Math & STEM, Coding, Multilingual.
  - Post-trained models:
    - **General** (MMLU-Redux, GPQA-D, C-Eval, LiveBench)
    - **Alignment** (IFEval, Arena-Hard, AlignBench, Creative Writing, WritingBench)
    - **Math & Text Reasoning** (MATH-500, AIME'24/'25, ZebraLogic, AutoLogi)
    - **Agent & Coding** (BFCL v3, LiveCodeBench, CodeForces)
    - **Multilingual** (Multi-IF, INCLUDE, MMMLU, MT-AIME, PolyMath, MLogiQA)
- **Qwen2.5 report:** the same base-model scheme (General / Math & Science / Coding / Multilingual) plus alignment benchmarks `[likely]`.
- **DeepSeek-V3 and R1** `[verified: V3 README, R1 PDF]`:
  - Groups are English (knowledge, IF, QA, long-context FRAMES, open-ended AlpacaEval/Arena-Hard), Code, Math, Chinese.
  - The base model adds Multilingual.
- **Gemini 2.5** `[verified: report Table 3]`:
  - Code (LiveCodeBench, Aider Polyglot, SWE-bench Verified)
  - Reasoning (GPQA-D, HLE)
  - Factuality (SimpleQA, FACTS Grounding)
  - Multilinguality (Global-MMLU-Lite, ECLeKTic)
  - Math (AIME 2025, HiddenMath-Hard)
  - Long-context (LOFT, MRCR-V2 at 128K and 1M)
  - Image, Video and Audio understanding
  - Agentic sections (Deep Research, "Gemini Plays Pokémon")
- **OpenAI (GPT-4o / o1 / o3)** `[verified: simple-evals README]`:
  - Text capability suite: MMLU, GPQA, MATH, HumanEval, MGSM, DROP, SimpleQA (plus BrowseComp and HealthBench).
  - Launch posts group results into competition math (AIME), competition code (Codeforces), PhD-level science (GPQA), software engineering (SWE-bench Verified, SWE-Lancer), multimodal (MMMU, MathVista), and instruction-following/agentic tool use (MultiChallenge, BrowseComp, τ-bench) `[likely]`.
  - System cards are safety-structured: disallowed content, jailbreaks, hallucination, bias, and Preparedness categories (bio/chem, cyber, persuasion, model autonomy / self-improvement) `[likely]`.
- **Anthropic Claude:**
  - Claude 3 model card: reasoning/coding/QA, multilingual, factuality, long context, vision, behavioral design `[likely]`.
  - Claude 4 / 4.x launch tables: agentic coding (SWE-bench Verified), agentic terminal coding (Terminal-Bench), agentic tool use (τ-/τ²-bench), graduate-level reasoning (GPQA-D), multilingual Q&A (MMMLU), visual reasoning (MMMU), high-school math competition (AIME 2025). Later additions: computer use (OSWorld), scaled tool use (MCP-Atlas), novel problem solving (ARC-AGI-2) `[likely]`.
  - System cards are safety/RSP-structured (CBRN, cyber, autonomy, alignment, agentic safety) `[likely]`.

**Surveys**
- **Chang et al. 2023, "A Survey on Evaluation of LLMs"** `[verified: MLGroupJLU/LLM-eval-survey README]`. Under *What to evaluate*:
  - Natural language processing: NLU (sentiment, text classification, NLI), reasoning, NLG (summarization, dialogue, translation, QA), multilingual, factuality
  - Robustness, ethics, bias, trustworthiness
  - Social science
  - Natural science and engineering: mathematics, general science, engineering
  - Medical applications
  - **Agent applications**
  - Other applications: education, search and recommendation, personality testing, specific tasks
  - The survey also covers *where* (benchmarks) and *how* (automatic vs human) to evaluate.
- **Guo et al. 2023, "Evaluating LLMs: A Comprehensive Survey"** `[verified: tjunlp-lab README]`:
  - **Knowledge & Capability:** QA, knowledge completion, reasoning (commonsense, logical, multi-hop, mathematical), **tool learning**
  - **Alignment:** ethics and morality, bias, toxicity, truthfulness
  - **Safety:** robustness, risk evaluation including evaluating LLMs as agents
  - **Specialized LLMs:** biology and medicine, education, legislation, computer science, finance
  - **Evaluation organization:** NLU/NLG benchmarks, knowledge and reasoning benchmarks, holistic benchmarks

### A.2 Which categories appear where (● = explicit category/group; ○ = covered by benchmarks but not a named group)

| Category | HELM | BIG-bench | OLL v1 | OLL v2 | OpenCompass | LiveBench | LMArena | AA Index v4.x | Llama 3 | Qwen3 | DeepSeek V3/R1 | Gemini 2.5 | OpenAI | Claude | Chang'23 | Guo'23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Knowledge / expert QA | ● | ● | ○ | ○ | ● | – | ○ (Expert) | ● General | ● | ● General | ● English | ● Factuality | ○ | ○ | ● | ● |
| Language understanding / commonsense / reading | ● | ● | ○ | – | ● | ● Language | – | – | ● | – | ○ | – | ○ DROP | ○ | ● NLU | ● |
| General / multi-step reasoning | ● | ● | ○ | ○ | ● | ● | ● Hard Prompts | ● Sci. reasoning | ● | ● | ○ | ● | ○ | ● | ● | ● |
| Math | ○ | ● | ○ | ○ | ○ | ● | ● | ○ (AIME until v3) | ● | ● | ● | ● | ○ | ○ | ● | ● |
| Coding / software eng. | ○ | ● | – | – | ● | ● + Agentic | ● | ● Coding | ● | ● | ● | ● | ○ | ● | ○ | ○ |
| Instruction following / alignment / chat preference | ● (Capabilities) | ● | – | ○ | ○ | ● | ● | ○ (IFBench until v4.1) | ○ | ● Alignment | ○ | – | ○ | – | ○ | ● |
| Creative writing | – | ● | – | – | – | – | ● | – | – | ○ | ○ | – | – | – | – | – |
| Multi-turn dialogue | ○ | ● | – | – | – | – | ● | – | – | ○ Multi-IF | – | – | ○ | – | ○ | – |
| Tool use / agents | – | – | – | – | ○ (later) | ● | ○ | ● Agents | ● Tool use | ● Agent | – | ● | ○ | ● | ● | ● |
| Long context | ○ | ● | – | – | ● | – | ● Longer Query | ○ AA-LCR, GDP.pdf | ● | ○ | ○ | ● | – | ○ | – | – |
| Multilingual | ● | ● | – | – | ● | – | ● | – | ● | ● | ● | ● | ○ | ○ | ● | – |
| Safety / truthfulness / bias / calibration | ● | ● | ○ | – | ● | – | – | ○ | ○ | – | – | ○ | ● (system cards) | ● (system cards) | ● | ● |
| Domain-specific / professional work | ○ | ● | – | – | ● Exam | – | ● Occupational | ● GDPval / Briefcase | – | – | ○ | – | ○ | – | ● | ● |
| Efficiency | ● | – | – | – | – | – | – | (separate) | – | – | – | – | – | – | – | – |
| Multimodal | – | ○ | – | – | – | – | ● | – | – | – | – | ● | ○ | ○ | – | – |

(– = not a category in that source; this table is a synthesis from the per-source summaries above.)

### A.3 Consolidated view: recurring top-level categories

1. **Knowledge & expert QA.** MMLU, MMLU-Pro, GPQA, HLE, SimpleQA. Present in essentially every taxonomy.
2. **Reasoning, with math as its own category.** BBH, MuSR, ZebraLogic; GSM8K, MATH, AIME. Every taxonomy has math. Reasoning appears as general, scientific, or "hard prompts".
3. **Coding.** Coding evaluation has moved from function-level (HumanEval, MBPP) to contamination-free (LiveCodeBench) to repo-level and agentic (SWE-bench, Terminal-Bench).
4. **Instruction following, alignment and open-ended chat preference.** IFEval, AlpacaEval, Arena-Hard, WildBench, MT-Bench, and LMArena itself. This category also absorbs **creative writing** and **multi-turn**, which only arena-style taxonomies and BIG-bench name separately.
5. **Tool use and agents.** BFCL, τ-bench, Terminal-Bench, GAIA, WebArena, OSWorld, AgentBench, GDPval. This category is new in 2024–26. It is now the largest weight in the AA Index (30% Agents) and dominates Claude and Gemini headline tables.
6. **Long context.** Present in model reports, OpenCompass, BIG-bench keywords and LMArena "Longer Query". Handled by other workstreams.
7. **Multilingual.**
8. **Safety, trustworthiness and factuality.** Usually handled in separate system cards or metric axes (HELM) rather than capability leaderboards.
9. **Domain or professional tasks.** Exams, medicine, law, finance, and "economically valuable work" (GDPval-AA, AA-Briefcase).
10. Legacy **language understanding and commonsense** (HellaSwag, Winogrande, ARC). Now mostly used for base-model or pretraining evaluation.

**Trends relevant to workload design:**
- Leaderboards moved from log-likelihood MC (Open LLM LB v1, HELM Classic) to generative CoT (OLL v2, HELM Capabilities), then to long reasoning (AIME, GPQA with 32k budgets), and now to agentic multi-step work (AA v4.x, LiveBench agentic coding, Claude and Gemini tables).
- Every step shifts the KV profile from prefill-only toward decode-dominated growth, and then toward append-only multi-call contexts.

---

## PART B — Task-level facts (KV-relevant)

### B.1 Summary table
Token counts use the Llama-3 tokenizer; "p90" is the 90th percentile. "Few-shot prefix" means the shared exemplar block.

| Benchmark | n | Prompt tokens (standard template) | Output tokens | Turns / steps | Context growth |
|---|---|---|---|---|---|
| MMLU (5-shot) | 14,042 | mean 683, median 502, p90 1,577, max 3,081 (history subjects about 2.2–2.8k); 0-shot 105 | 1 (log-likelihood) | 1 | none |
| MMLU-Pro | 12,032 | question + 10 options: mean 195, p90 349; 5-shot CoT ≈ 2.4k (estimate) | GPT-4o CoT mean 391; Gemini-3.1-Pro visible 246 | 1 | none |
| ARC-C (25-shot) | 1,172 | mean 931, max 1,272 | log-likelihood | 1 | none |
| HellaSwag (10-shot) | 10,042 | mean 789, max 1,071 | log-likelihood (4 endings of about 14 tokens) | 1 | none |
| Winogrande (5-shot) | 1,267 | about 140 | log-likelihood | 1 | none |
| TruthfulQA (MC) | 817 | 161 (143-token fixed primer + question) | log-likelihood | 1 | none |
| GSM8K (8-shot CoT) | 1,319 | 653-token prefix + 63 → mean 716, max 842 (5-shot LB format: 872) | reference 103; R1-Distill-1.5B 431–641 | 1 | decode |
| MATH-500 | 500 | problem mean 69 (Minerva 4-shot prefix 612) | reference 210; **R1-distill 8B/14B avg 2,979** | 1 | decode |
| AIME 2024/25 | 30 per year | problem mean 103; prompt about 160–200 | **R1-distill 8B/14B avg 15,536**; R1-Distill-1.5B 11,166 (16k cap, 13/30 truncated); AIME25 about 10–26k for frontier reasoners (figure) | 1 | extreme decode |
| GPQA-Diamond | 198 | question + options mean 199 | reasoners about 3.5k (o1) to 18k (Claude 3.7 thinking) (figure) | 1 | decode |
| BBH (3-shot CoT) | 6,511 | prefix 192–1,816 (median 686) | exemplar CoT mean 158 | 1 | none |
| HLE | 2,500 | unknown | harness recommends ≥ 8,192 max tokens for reasoners | 1 | decode |
| HumanEval | 164 | mean 131, max 391 | canonical 54 | 1 | none |
| MBPP (3-shot) | 500 | 505 prefix + 120 → mean 625 | reference 58 | 1 | none |
| LiveCodeBench | 1,055 (v6) | unknown | reasoners run with caps up to 32k | 1 (self-repair: 2) | decode |
| SWE-bench Verified | 500 | issue mean 313; static agent prefix 1.2–5.9k | about 5–18k generated per task | **34–69 LLM calls (max 206)** | **final context 34–48k mean (max about 100k); cumulative prefill 1.3–2.0M per task** |
| RepoBench | – | 2k to 16k buckets | 1 line (≤ 128) | 1 | none |
| CrossCodeEval | 9,928 | about 1.4k (left context about 800 + retrieved about 600) | 10–16 | 1 | none |
| IFEval | 541 | mean 46 | GPT-4 mean 275 (p90 555) | 1 | decode |
| MT-Bench | 80 × 2 | turn-1 66, turn-2 23 | GPT-4 references about 204 | 2 | per-turn append |
| MT-Bench-101 | 1,388 dialogues / 4,208 turns | whole dialogue mean 252 (max 967) | about 70 per turn (golden) | mean 3, max 7 | per-turn append |
| AlpacaEval 2 | 805 | mean 35 | GPT-4-Turbo reference 432; leaderboard median about 1.5k chars | 1 | decode |
| Arena-Hard v0.1 / v2 | 500 / 750 | mean 95 / 285 | **R1 thought 2,992 + answer 701; QwQ-32B 4,040 + 943** | 1 | decode |
| WildBench | 1,024 | query 978.5 chars; with history 3,402 chars (about 0.8k tokens) | unknown | ≤ 5 (static history) | static |
| BFCL multi-turn | 4 × 200 | tool docs about 5.0k + config 315 | calls | 4–5 user turns (max 8) | append; long-context variant injects big outputs |
| τ-bench | 115 + 50 | static policy + tools about 3.7–4.1k | about 1–1.8k over the trajectory | about 26–30 messages, 6–8 user turns, 6–8 tool calls | **final 6.2–8.2k mean (max 20.6k)** |
| Terminal-Bench 2.0 | 89 | instruction mean 231 | unknown | timeouts 10 min to 3.3 h (median 15 min) | append (unknown size) |
| GAIA | 466 | short question + files | short answer | L1 ≤ 5 steps, L2 5–10, L3 long | append (unknown) |
| WebArena | 812 | about 3k per step (965 prompt + ≤ 1,920 observation) | ≤ 384 per step | ≤ 30 steps | none in the reference agent (Markovian) |
| OSWorld | 369 | a11y tree ≤ 10k + screenshots, last 3 steps | ≤ 1,500 per step | 15 (also 50/100) | sliding window |
| AgentBench | 1,091 (test) | unknown | short actions | average turns OS 8, DB 5, KG 15, DCG 30, LTP 25, HH 35, WS 5, WB 10 | append |

### B.2 Published or measured reasoning-trace lengths
| Benchmark | Model | Avg output/thinking tokens | Setting | Source | Conf. |
|---|---|---|---|---|---|
| MATH-500 | DeepSeek-R1-Distill-Llama-8B / Qwen-14B | **2,979** | max gen 16,384 | R-KV README | [verified] |
| AIME 2024 | same | **15,536** | max gen 32,768 | R-KV README | [verified] |
| AIME 2024 | R1-Distill-Qwen-1.5B | 11,166 mean (median 14,168; 13/30 hit the 16k cap) | 16k cap, n=30 | R-KV validation README and outputs | [verified] |
| GSM8K | R1-Distill-Qwen-1.5B | 431 (chat template), 641 (plain; max 4,005) | n=50 | R-KV validation outputs | [verified] |
| AIME 2025 | Claude 3.7 Sonnet (thinking) / o3-mini-high / DeepSeek-R1 / o1 | about 26k / 11.5k / 10.7k / 9.7k | output + reasoning tokens | Eureka ITS report, Fig. 4 | [likely, approx.] |
| GPQA | Claude 3.7 (thinking) / o3-mini-high / Gemini-2-Flash-Thinking / R1 / o1 | about 18k / 7.3k / 5.8k / 5k / 3.5k | same | same | [likely, approx.] |
| Omni-MATH | Claude 3.7 / R1 / o3-mini / o1 | about 12.8k / 8.8k / 7.4k / 6.1k | same | same | [likely, approx.] |
| Arena-Hard v2 (all 750) | DeepSeek-R1 | thought 2,992 (median 1,686, p90 7,372, max 20,396); answer 701 | repo model answers | arena-hard-auto | [verified] |
| Arena-Hard v2 math / coding / creative | DeepSeek-R1 | 4,932 / 3,274 / 789 | same | same | [verified] |
| Arena-Hard v2 (all) | QwQ-32B | thought 4,040 (median 2,398, p90 9,470, max 28,569); answer 943 | same | same | [verified] |
| Arena-Hard v2 math / coding / creative | QwQ-32B | 6,910 / 4,198 / 1,044 | same | same | [verified] |
| Arena-Hard | DeepSeek-R1 (summary only) | 689 | – | R1 report | [verified] |
| AlpacaEval 2 | DeepSeek-R1 (summary only) | 2,218 chars | – | R1 report | [verified] |
| R1-Zero RL training | DeepSeek-R1-Zero | average response length grows steadily ("hundreds to thousands of reasoning tokens"), Fig. 3 | training set | R1 report | [verified text; figure values not extracted] |

**Generation caps and budgets used in official evaluations** `[verified]`:
- DeepSeek-R1: max 32,768 for all benchmarks; 64 samples per query for pass@1.
- Qwen3: 32,768 for all tasks, **38,912 for AIME'24/'25**; thinking budget 8,192 on RULER to avoid verbose thinking; BFCL multi-turn served at 64k via YaRN.
- Gemini 2.5: thinking budget swept 1,024 to 32,768 on AIME 2025, LiveCodeBench and GPQA.
- HLE harness: recommends max_completion_tokens ≥ 8,192 for reasoning models.
- s1 / LIMO / QwQ READMEs use max_tokens 32,768.

### B.3 Agent trajectory lengths (measured from public trajectories)
| Benchmark / scaffold + model | LLM calls per task | Final context (tokens) | Other |
|---|---|---|---|
| SWE-bench Verified, SWE-agent + Claude-3.5-Sonnet (2024-06), n=100 | mean 34, median 29, p90 68, max 78 | mean 33.7k, median 28.3k, p90 59.8k, max 98.6k | cumulative input 528k mean (417k median, p90 1.27M); avg 14.4k per call; output 5.3k per task; system+demo prefix 5,876; per-observation median 416, max 206k |
| SWE-bench Verified, SWE-agent 1.0 + Claude-4-Sonnet (2025-05), n=40 | mean 56, median 54, max 100 | mean 45.7k, median 40.1k, p90 67.7k, max 95.2k | sum of per-call contexts 1.33M mean (28× final); median per-call context about 20.6k; observation median 159 |
| SWE-bench Verified, OpenHands + Claude-4-Sonnet (2025-05), n=100 | mean 69, median 60, p90 110, max 206 | mean 48.1k, median 45.5k, p90 71.6k, max 104k (excluding tool schemas) | sum of per-call contexts 2.0M mean (37× final, max 11.4M); tool-output median 256, mean 464; assistant tokens about 18k |
| τ-bench retail, GPT-4o (4 trials × 115) | 7.1 tool calls, 8.2 user turns, 30.5 messages | mean 7.2k, max 15.3k (includes 4.1k static prefix) | tool outputs 2.0k, assistant 1.0k |
| τ-bench airline, GPT-4o (4 × 50) | 5.8 tool calls, 7.5 user turns | mean 6.2k, max 13.3k | – |
| τ-bench retail / airline, Claude-3.5-Sonnet (new) | 7.7 / 6.9 tool calls | 8.2k / 7.5k mean, max 19.0k / 20.6k | – |
| AgentBench | average turns 5–35 per environment | – | [verified from paper Table 2 image] |
| WebArena reference agent | ≤ 30 steps | about 3k per step, no growth | observation truncated to 1,920 |
| OSWorld reference agent | ≤ 15 steps (50/100 in Verified) | ≤ 3 recent observations, a11y ≤ 10k each | – |
| LiveBench agentic coding | step limit 250 (mini-SWE-agent) | – | changelog |
| AA GDPval-AA v2 | turn limit raised 100 → 250 | – | [likely] |
| Gemini Plays Pokémon (Gemini 2.5 report) | thousands of turns; summarization every 100 turns and re-compression every 1,000 | agent became repetitive once context grew **well beyond 100k tokens** | [verified] |

### B.4 KV footprint conversion (for sizing workloads)
- Llama-3.1-8B-class GQA model (32 layers, 8 KV heads, head dim 128, bf16): KV ≈ 2×32×8×128×2 B = **128 KiB per token**.
  - A 15.5k-token AIME trace needs about 1.9 GiB.
  - A 32k trace needs about 4 GiB. The R-KV README quotes 4.1 GB.
  - A 48k SWE-bench final context needs about 6 GiB per sequence.
- Llama-3.1-70B-class (80 layers, 8 KV heads) ≈ **320 KiB per token**, so a 48k agent context needs about 15 GiB.
- These two model configs are recalled, not verified in this session `[likely]`.

### B.5 Per-benchmark notes on retention (what eviction must not drop)
- **MC knowledge (MMLU, ARC, HellaSwag, Winogrande, TruthfulQA).** The answer depends on the final question and options. Few-shot exemplars only set the format. KV is essentially prefill-only.
- **Few-shot CoT (GSM8K, BBH, MATH 4-shot).**
  - The exemplar prefix is identical across items, so it is a prefix-cache target.
  - The CoT keeps running state (numbers, object positions, truth values), and evicting it breaks the chain.
  - The measured BBH prefix size varies 10× across subtasks (192–1,816).
- **Long reasoning (AIME, MATH-500, GPQA, HLE, LiveCodeBench).**
  - The KV is mostly self-generated.
  - The problem statement (100–300 tokens) must stay resident for the whole decode.
  - Traces are highly redundant: R-KV reports lossless accuracy at a 10–54% KV budget for R1-distill models (8B: 10% on AIME-24, 34% on MATH-500; 14B: 25% and 54%).
  - Too-aggressive budgets make models re-derive facts and lengthen outputs by about 25% (R-KV validation).
- **IF (IFEval).** Constraints stated once in a 40–70-token prompt must govern hundreds of output tokens. Early prompt tokens are critical during the whole decode.
- **Multi-turn chat (MT-Bench, MT-Bench-101, WildBench).**
  - Contexts are short (< 1k typical), but specific earlier-turn facts and the model's own prior answers are referenced later.
  - Examples are Context Memory and anaphora tasks, and "rewrite your previous response".
- **Agents (SWE-bench, τ-bench, BFCL, Terminal-Bench, GAIA).**
  - There is a large static prefix: system prompt, policy, tool schemas, demos (1–6k).
  - It is followed by append-only observations of very uneven size: median of a few hundred tokens, with tails of 10k–200k.
  - Each step re-reads everything, so cumulative prefill is 28–37× the final context without prefix caching.
  - Retention needs are sparse but unpredictable: IDs, file paths and line numbers, policy clauses, and earlier tool results.
  - Scaffolds already evict at the application level: WebArena keeps only the current observation, OSWorld keeps the last 3 steps, and SWE-agent collapses old observations.

---

## Sources (primary unless noted)

**Taxonomies**
- HELM: https://arxiv.org/abs/2211.09110 ; https://github.com/stanford-crfm/helm (schema_classic.yaml, schema_lite.yaml, schema_capabilities.yaml)
- BIG-bench: https://github.com/google/BIG-bench/blob/main/keywords.md ; https://arxiv.org/abs/2206.04615
- BBH: https://arxiv.org/abs/2210.09261 ; https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bbh
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
- Open LLM Leaderboard: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/benchmarks/openllm.yaml ; https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md
- OpenCompass: https://github.com/open-compass/opencompass (README at tag 0.2.0)
- LiveBench: https://github.com/LiveBench/LiveBench ; changelog.md
- LMArena / Chatbot Arena:
  - Categories and classifiers: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/monitor_md.py ; https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/classify/category.py
  - Hard Prompts blog source: https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2024-05-17-category-hard.md
  - https://arena.ai/blog/arena-category and https://news.lmarena.ai/arena-expert/ (search snippets only)
  - Chatbot Arena paper: https://arxiv.org/abs/2403.04132
- Artificial Analysis (search snippets only):
  - https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index
  - https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-1
  - https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-2
  - https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3
  - https://x.com/ArtificialAnlys/status/2097025650200924626
- Llama 3.1 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md ; Llama 3 paper: https://arxiv.org/abs/2407.21783
- Qwen3 report: https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf ; Qwen2.5 report: https://arxiv.org/abs/2412.15115
- DeepSeek:
  - V3: https://github.com/deepseek-ai/DeepSeek-V3
  - R1: https://github.com/deepseek-ai/DeepSeek-R1 (DeepSeek_R1.pdf) ; https://arxiv.org/abs/2501.12948
- Gemini 2.5: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
- OpenAI simple-evals: https://github.com/openai/simple-evals
- Surveys:
  - Chang et al.: https://github.com/MLGroupJLU/LLM-eval-survey ; https://arxiv.org/abs/2307.03109
  - Guo et al.: https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers ; https://arxiv.org/abs/2310.19736

**Benchmarks and measurements**
- MMLU: https://github.com/FranxYao/chain-of-thought-hub/tree/main/MMLU/data ; MMLU-Pro: https://github.com/TIGER-AI-Lab/MMLU-Pro
- ARC: https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip ; HellaSwag via https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1 ; Winogrande: https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip ; TruthfulQA: https://github.com/sylinrl/TruthfulQA
- GSM8K: https://github.com/openai/grade-school-math ; MATH-500: https://github.com/openai/prm800k ; AIME24: https://github.com/QwenLM/Qwen2.5-Math ; GPQA: https://github.com/idavidrein/gpqa ; HLE: https://github.com/centerforaisafety/hle
- Code:
  - HumanEval: https://github.com/openai/human-eval ; MBPP: https://github.com/google-research/google-research/tree/master/mbpp ; LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
  - SWE-bench: https://github.com/SWE-bench/SWE-bench ; https://github.com/SWE-bench/experiments ; trajectories in https://swe-bench-submissions.s3.amazonaws.com/
  - RepoBench: https://github.com/Leolty/repobench ; CrossCodeEval: https://github.com/amazon-science/cceval
- Instruction following and chat:
  - IFEval: https://github.com/google-research/google-research/tree/master/instruction_following_eval ; MT-Bench: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge ; MT-Bench-101: https://github.com/mtbench101/mt-bench-101
  - AlpacaEval: https://github.com/tatsu-lab/alpaca_eval ; Arena-Hard: https://github.com/lmarena/arena-hard-auto ; WildBench: https://github.com/allenai/WildBench (docs/wb_table.png, docs/wb_stat.png)
- Agents:
  - BFCL: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
  - τ-bench: https://github.com/sierra-research/tau-bench ; τ²/τ³: https://github.com/sierra-research/tau2-bench
  - Terminal-Bench: https://github.com/laude-institute/terminal-bench ; https://github.com/harbor-framework/terminal-bench-2
  - GAIA: https://arxiv.org/abs/2311.12983 ; WebArena: https://github.com/web-arena-x/webarena ; OSWorld: https://github.com/xlang-ai/OSWorld ; AgentBench: https://github.com/THUDM/AgentBench (assets/statistics.png)
- Reasoning lengths:
  - R-KV: https://github.com/Zefan-Cai/R-KV (README; results/validation-2026-07-02-a100)
  - Microsoft Eureka inference-time scaling report: https://www.microsoft.com/en-us/research/publication/inference-time-scaling-for-complex-tasks-where-we-stand-and-what-lies-ahead/
- Tokenizer used for all counts: https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.model
