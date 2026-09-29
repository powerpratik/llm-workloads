> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# Long-context and long-output benchmark suites: task-level notes

Companion to `longctx_tasks.jsonl`, which has 228 task rows from 29 suites. This file holds the
suite-level summaries, the prompt-template observations, the conceptual taxonomies and the gaps
that matter for a KV-cache-eviction workload taxonomy.

## 0. Method, access limits, field conventions

**How the data was gathered.** Most facts come from official GitHub repos, read through
`raw.githubusercontent.com` or WebFetch on github.com. I read READMEs, stats tables (including PNG
tables in the repos), prompt templates, generation-length configs and eval code. For Loong,
LoCoMo, LongProc, Counting-Stars, LongBench-Write, LongGenBench (Wu) and HelloBench I downloaded
the official data files and computed the statistics myself.

**What was blocked.** arXiv, ACL Anthology, OpenReview, Hugging Face, Semantic Scholar, project
`github.io` pages, fiction.live, openai.com and deepwiki were all blocked. The WebSearch budget
was used up partway through (200/200, shared across the session). As a result:
- Michelangelo, Fiction.LiveBench and Goldman et al. could not be checked against primary text.
  They are marked `[unverified]` or `[likely]`.
- OpenAI-MRCR details come from a third-party port (EvalScope) and are marked `[likely]`.
- The LCLM survey (Liu et al. 2025) PDF *is* hosted on GitHub, so its taxonomy text was read
  directly (`[verified]`).

**Field conventions in the JSONL**
- `query_position` records where the instance-specific information need sits: the question, key,
  claim, target function, and so on. Values:
  - `after context`: the need appears only after the long context. A generic task description
    may come before it.
  - `before context`: the need appears before the context.
  - `both`: the need appears before and after the context.
  - `n/a`: there is no instance-specific need (e.g. "summarize this", a fixed counting
    instruction), or there is no long input.
- `instruction_position` (extra field) records where the generic task instruction sits: before,
  after, both, or unknown. This matters because, for example, RULER announces "special magic
  numbers are hidden" before the context but gives the key only after it.
- Length units are kept as the source reports them: words, characters (for Chinese), or tokens
  (tokenizer noted where known). The units are **not comparable across suites**.
- Confidence levels:
  - `[verified]` = read in this session from an official repo file or data.
  - `[likely]` = secondary source (harness port, search snippet) or consistent recollection.
  - `[unverified]` = recollection only.

**Aggregate counts over the 228 rows**
- `query_position`: after context 189 (83%), n/a 34, both 3, before context 2.
- Explicit question-before-context appears only in HELMET's ALCE (ASQA, QAMPARI).
- NeedleBench can optionally put the question at the start.
- InfiniteBench's Kimi-specific template puts the question first.
- Rough synthetic/natural split: natural 131, synthetic 54, hybrid 43. Hybrid means a synthetic
  task on natural text, or natural QA inside a synthetic haystack.

---

## 1. Suite summaries

### LongBench (v1), THUDM, ACL 2024 `[verified]`
- **Tasks:** 21 tasks in 6 categories: single-doc QA, multi-doc QA, summarization, few-shot
  learning, synthetic, code. By language: 14 EN, 5 ZH, 2 code. 4,750 test items (200 per task;
  MultiFieldQA-en has 150; LCC and RepoBench-P have 500).
- **Lengths:** average per task 1,235–22,337. Units are words for EN/code and characters for ZH.
  Most tasks average 5–15k. LongBench-E resamples 13 tasks into uniform 0–4k, 4–8k and 8k+
  buckets.
- **Templates** (`config/dataset2prompt.json`):
  - Every QA task repeats the instruction before and after the context. The `{input}` question
    appears **only after** the context.
  - Summarization has a generic instruction on both sides. VCSUM has it before only.
  - Few-shot tasks put the demos in the context and the test input after them.
  - LCC has no explicit query (continue the code). RepoBench-P puts the in-file prefix after the
    cross-file snippets.
- **Generation caps:** 32 tokens (multi-doc QA, retrieval), 64–128 tokens (single-doc QA,
  classification), 512 tokens (summarization).
- **Truncation:** over-length inputs are **truncated from the middle**, keeping the head and
  tail. This interacts with evidence position.
- **KV notes:** almost entirely short-output, single-query-at-end. Few-shot demos are resampled
  per item, so there is no shared prefix across items.

### LongBench v2, THUDM 2024 `[verified]`
- **Items:** 503 four-option MCQs in 6 categories and 20 subdomains. The official table (#data and
  average length in words) is reproduced in the JSONL.
  - Single-Doc QA: 175, 51k average.
  - Multi-Doc QA: 125, 34k.
  - Long ICL: 81, 71k.
  - Long-dialogue history: 39, 25k.
  - Code repo: 50, 167k.
  - Long structured data: 33, 49k.
- **Lengths:** 8k–2M words overall; most items are under 128k.
- **Template:** `<text>$DOC$</text>` then the question and choices. The question comes after.
- **Output:** a letter (cap 128 tokens), or chain-of-thought up to 1,024 tokens.
- **Human baseline:** experts reach 53.7% under a 15-minute limit.
- **Length/difficulty splits** (180/215/108 and 192/311): recollection, `[unverified]`.

### RULER, NVIDIA 2024 `[verified]`
- **Tasks:** 13 synthetic tasks in 4 categories:
  - Retrieval: 8 NIAH variants.
  - Multi-hop tracing: VT.
  - Aggregation: CWE, FWE.
  - QA: SQuAD and HotpotQA inside distractor paragraphs.
- **Lengths:** configurable. Default 4K–128K; 500 samples per task per length.
- **NIAH configs** (`synthetic.yaml`):
  - Haystack types: noise sentences, Paul Graham essays, or "needle" (the haystack is all
    distractor key-value lines).
  - Key/value types: words, 7-digit numbers, UUIDs.
  - MK-1 has 4 keys in essays. MK-2 and MK-3 use all-needle haystacks.
  - MV has 4 values per key. MQ has 4 queried keys.
  - Needle depths are sampled from 40 evenly spaced points.
- **Templates:**
  - A needle-type instruction comes before the context; the key comes after.
  - VT and CWE prepend a one-shot example.
  - CWE and FWE have a fixed question.
- **Generation budgets:** NIAH 128, VT 30, CWE 120, FWE 50, QA 32.
- **Metrics:** string_match_all (NIAH/VT/CWE/FWE) and string_match_part (QA).
- **README caveat:** the 13 configs were chosen because models score well at 4K. RULER
  "cannot replace realistic tasks".

### InfiniteBench (∞Bench), OpenBMB, ACL 2024 `[verified]`
- **Tasks:** 12 tasks.
  - Realistic context: En.Sum, En.QA, En.MC, En.Dia, Zh.QA, Code.Debug. The "fake book" novels
    use core-entity substitution to limit memorization.
  - Synthetic context: Code.Run, Math.Calc, Math.Find, Retrieve.PassKey, Retrieve.Number,
    Retrieve.KV.
- **Lengths:** average input 43.9k (Math.Calc) to 2,068.6k tokens (Zh.QA); most are 75–192k.
- **Outputs:**
  - Average ≤ 23 tokens except En.Sum (1.1k) and **Math.Calc, whose output is as long as its
    input (43.9k)**. Math.Calc is the only comprehension task with a huge output.
  - Caps (`DATA_NAME_TO_MAX_NEW_TOKENS`): 3–50 tokens for most tasks; 1,200 for Sum; 30,000 for
    Calc.
- **Templates are model-specific** (`src/prompt.py`):
  - GPT-4/YaRN put the question after the context.
  - Kimi's template places the question before the attached file.
  - Math.Find has the target type both before and after. Code.Run names the function before and
    gives the call after.
  - En.Dia's target is a `$$MASK$$` inside the context.

### HELMET, Princeton, ICLR 2025 `[verified]`
- **Structure:** 7 application-centric categories, 21 datasets. Lengths 8K/16K/32K/64K/128K,
  controlled by the number of passages/demos or by truncation.
  - Synthetic recall: JSON KV, RULER MK-2, MK-3, MV.
  - RAG: KILT NQ, TriviaQA, HotpotQA, PopQA.
  - Re-rank: MS MARCO.
  - Cite: ALCE ASQA, QAMPARI.
  - LongQA: NarrativeQA, ∞Bench QA, ∞Bench MC.
  - Summ: ∞Bench Sum, Multi-LexSum.
  - ICL: TREC-coarse/fine, NLU, BANKING77, CLINC150.
- **Scaling knobs from the configs:**
  - JSON KV: 105/220/440/900/1800 pairs.
  - RAG: 50/105/220/440/1000 passages.
  - Re-rank: 50/130/285/600/1000 passages.
  - ALCE: 30/75/165/345/700 docs.
  - ICL: up to 6,600–8,296 shots.
- **Samples:** 100 per dataset per length (500 for ICL). Recall/RAG/re-rank/cite use 2-shot
  demos.
- **Template facts** (`data.py`):
  - **ALCE puts the question BEFORE the documents.** It is the only such case among the
    mainstream suites.
  - JSON KV puts the context first and then the instruction, demos and key.
  - ICL maps labels to random integers by default, so the model must actually learn the mapping
    from the demos.
  - LongQA and Summ use GPT-4o model-based evaluation. The summarization key points ship with the
    data.
- **Generation caps:** 20 (RAG, ICL), 50–100 (recall), 200 (re-rank), 300 (cite), 10–100
  (LongQA), 400/1,200 (summ).
- **LongProc add-on:** HELMET also hosts `longproc_addon` configs.

### L-Eval, OpenLMLab 2023 `[verified]`
- **Tasks:** 20 sub-tasks, 508 documents, more than 2,000 instructions. Input 3k–200k tokens.
  - 7 closed-ended (exact-match) tasks: TOEFL, GSM-16shot, QuALITY, Coursera, TopicRet,
    SFiction, CodeU.
  - 13 open-ended (generation) tasks: MultiDoc2Dial, Qasper, LongFQA, NQ, CUAD, NarrativeQA,
    Multi-News, GovReport, BigPatent, SummScreen, Openreview, QMSum, SPACE.
- **Per-task stats:** average/maximum token length and #instr/#doc come from `figs/data.png`.
- **Multiple instructions per document** (e.g., NarrativeQA has 182 instructions over 20 docs).
  The baselines issue one request per instruction: system prompt + document + instruction, with
  the instruction after.
- **Evaluation:** open-ended tasks use Length-Instruction-Enhanced evaluation (the reference
  length goes in the prompt) plus a GPT-4 judge that battles Turbo-16k.

### ZeroSCROLLS, TAU, Findings EMNLP 2023 (`[verified]` code; stats partly `[likely]`)
- **Tasks:** 10.
  - Summarization: GovReport, SummScreenFD.
  - Query-based summarization: QMSum, SQuALITY.
  - QA: Qasper, NarrativeQA, MuSiQue.
  - MC QA: QuALITY.
  - Aggregation: SpaceDigest (percentage of 50 hotel reviews that are positive; exponential
    similarity), BookSumSort (reorder chapter summaries; concordance index).
- **Generation budgets** (`api.py`): GovReport 1,024; SummScreenFD, QMSum and SQuALITY 512;
  Qasper 128; NarrativeQA 64; MuSiQue 32; QuALITY 10; SpaceDigest 36; BookSumSort 256.
- **Truncation:** the document is truncated from its end while the query suffix is kept, so the
  query is always at the end.
- **Average words** from search snippets of Table 1: GovReport 7,273; QMSum 10,839; Qasper 3,531;
  NarrativeQA 49,384; QuALITY 4,248; MuSiQue 1,749. The other four tasks are "unknown".
- **Samples:** at most 500 per task. Exact per-task counts are not verified.

### LooGLE, BIGAI 2023 `[verified]`
- **Data:** 776 post-2022 documents, 6,448 questions, 7 tasks. Sources:
  - arXiv papers (516): average 16,988 words / 20,887 tokens.
  - Wikipedia (105): 21,017 tokens.
  - Movie/TV scripts (155): 36,412 tokens.
- **Short-dependency tasks:** Wikipedia QA (1,951) and script cloze (2,880).
- **Long-dependency tasks:**
  - Summarization (516).
  - Comprehension & reasoning (152 + 254).
  - Multiple information retrieval (158 + 222).
  - Timeline reorder (83 + 132).
  - Computation (66 + 34).
- **Template:** instruction, then the text, then `Question: {Q}`. Question after.
- **Caps:** short-dep QA 300, long-dep QA/summ 500, cloze 50.
- **Multi-query:** each document carries many QA pairs.

### BABILong, AIRI/NeurIPS 2024 `[verified]` (qa11–20 details `[likely]`)
- **Design:** 20 bAbI tasks hidden in PG19 text. Lengths 0K to 10M tokens (13 levels). 100
  samples per task per length, or 1,000 in the 1k version (which goes to 128K).
- **README table:** facts and supporting facts per task for qa1–qa10, e.g. qa2 has 2–68 facts
  with 2 supporting; qa7 counting has 1–10 supporting facts.
- **Template:** instruction, few-shot examples, post-prompt, then `<context>...</context>` and
  `Question:`. Question after.
- **KV-relevant features:** the tasks need the most recent state (qa1–qa3), counting (qa7), sets
  (qa8), and chains (qa2/qa3/qa19).

### Michelangelo (Latent Structure Queries), Google DeepMind 2024 `[unverified]`
- **Tasks:** Latent List (Python list operations; query the final list state), MRCR (reproduce
  the i-th of several similar writing requests from a long synthetic chat; SequenceMatcher
  score), and IDK (answer "I don't know" when the context lacks the answer).
- **Lengths:** up to 128K, and 1M for Gemini.
- **Framing:** the context hides a latent structure. Irrelevant content must be discarded and
  queries probe that structure. The LCLM survey's Table 6 lists it as "Latent structure queries,
  128k, free-form, automatic metrics" (`[verified]` via the survey).

### LV-Eval, Infinigence 2024 `[verified]`
- **Data:** 11 bilingual QA datasets, 6 single-hop and 5 multi-hop. Each comes at 5 length levels
  (16k/32k/64k/128k/256k words); the same QA pairs are reused across levels.
- **Perturbations:**
  - Confusing-fact insertion (CFI).
  - Keyword/phrase replacement (KPR) against knowledge leakage.
  - Answer-keyword-gated F1.
- **Template:** instruction, article, instruction again, question. Question after.
- **Caps:** 64 tokens (16 for factrecall).

### NoCha, UMass/AI2, 2024 `[verified]`
- **Data:** 1,001 true/false minimal claim pairs over 67 recent novels (mean 127K tokens, range
  49K–336K, cl100k tokenizer), about 15 claims per book.
- **Metric:** pair accuracy.
- **Prompt:** instructions, `<context>` book, `<statement>` claim, question. Output is an
  explanation paragraph plus TRUE/FALSE.
- **Release:** only 4 classic books are released as samples.

### Loong, Alibaba, EMNLP 2024 `[verified]`
- **Items:** 1,600, computed from `data/loong.jsonl`. About 10 documents per item (mean 10.2).
  Tokens counted with tiktoken (gpt-3.5): mean 110K, range 11K–336K.
- **Four evidence-structure levels:**
  - Spotlight Locating: 250.
  - Comparison: 300.
  - Clustering: 641.
  - Chain of Reasoning: 409.
- **Domains:** financial 700, legal 500, paper 400. Languages: ZH 905, EN 695.
- **Length sets:** 10–50K (323), 50–100K (564), 100–200K (481), >200K (232).
- **Metric:** GPT-4 judge score plus Perfect Rate.
- **Templates:** 1,470 items use `{docs}{instruction}{question}`. 130 paper chain items put the
  instruction first.

### NeedleBench v2, OpenCompass `[verified]`/`[likely]`
- **Tasks:** Single-Needle Retrieval, Multi-Needle Retrieval, Multi-Needle Reasoning (v2 uses
  fictional needles), and Ancestral Trace Challenge (information-dense kinship chains with no
  filler; 2^k needles).
- **Lengths:** 4k–1000k configs, English and Chinese. For example, the 128k config sweeps 8
  lengths from 1K–128K × 11 depths.
- **Question position is a parameter** (`quesiton_position` End/Start; End is the default). This
  is one of the few suites that can test question-first prompts.

### Counting-Stars, Tencent, COLING 2025 `[verified]`
- **Tasks:** multi-evidence acquisition, and multi-evidence reasoning (each count is corrected in
  place).
- **Design:** 32 evidence sentences ("The little penguin counted N ★") spaced evenly. 32 context
  lengths up to about 128K. English uses a Paul Graham essay haystack; Chinese uses a novel
  haystack.
- **Question:** a fixed question comes **only after** the context. The output is a JSON list of 32
  numbers.
- **Samples:** 32 per file (4 files).

### LongICLBench, TIGER-Lab, TMLR 2025 `[verified]`
- **Data:** 6 extreme-label datasets (28–174 classes), 1K–50K tokens of demos. Demos come in
  rounds of one example per class.
- **Metric:** accuracy (F1 for Few-NERD and DialogRE).
- **Shared prefix:** the demo prompt is built once and reused for all test items. This is a clean
  shared-prefix workload.

### NoLiMa, Adobe/LMU, ICML 2025 `[likely]`
- **Design:** NIAH where the needle and question share minimal lexical overlap (latent
  association), with one-hop and two-hop variants.
- **Lengths:** 250 tokens to 32K, and 64K/128K for some models.
- **Grid:** 26 placements per length (11 at 64K/128K) and 5 shuffled-book haystacks. Metric is
  "contains"; the effective length is where the score stays ≥85% of the base score.
- **Size:** 58 question–needle pairs, per a secondary source. NoLiMa-Hard uses the 10 hardest
  pairs.

### LongProc, Princeton, COLM 2025 `[verified]`
- **Tasks:** 6 tasks, 16 task-levels, with target outputs of 0.5K/2K/8K tokens:
  - HTML→TSV: 0.5K/2K/8K.
  - Pseudocode→C++: 0.5K/2K.
  - Path traversal, ToM tracking and Countdown: 0.5K/2K/8K each.
  - Travel planning: 2K/8K.
- **Inputs** (computed here as characters/4):
  - HTML pages are about 1K–72K (mean about 20K).
  - Path-traversal graphs are about 1K/3.9K/11.7K.
  - The others are short.
- **Generation caps:** 1,024/3,072/10,240. HELMET's add-on uses 100 items per level.
- **Why it matters:** these are the only procedural long-output tasks. Output tokens must stay
  faithful to specific input tokens far back (HTML→TSV, path traversal) or to the model's own
  earlier output (Countdown, ToM).

### Needle-in-a-Haystack (gkamradt) and passkey retrieval
- **Original NIAH (2023)** `[likely]`:
  - The San Francisco sandwich needle in Paul Graham essays.
  - GPT-4 run: 15 lengths from 1K–128K × 15 depths (sigmoid-spaced). Claude 2.1 run: to 200K.
  - The repo was rewritten in 2026 as v2, with single/multi/uuid/uuid_chain tasks swept over a
    length × depth grid `[verified]`.
- **Passkey retrieval (Landmark Attention)** `[verified]`:
  - "The pass key is N. Remember it." inside repeated filler, at a random position.
  - 13 filler sizes of 0–38,000 characters, 50 tests each, max 10 new tokens.
  - Reused by InfiniteBench and SCBench.

### SCBench (SharedContextBench), Microsoft, ICLR 2025 `[verified]`
- **Data:** 12 tasks in 4 capability groups; 931 sessions / 4,853 turns; average input 227K,
  average output 1,684 tokens per session.
  - String retrieval: KV, Prefix-Suffix, MultiHop (VT).
  - Semantic retrieval: RepoQA, En.QA, Zh.QA, En.MultiChoice.
  - Global information: Math.Find, ICL.ManyShot, En.Sum.
  - Multi-tasking: Sum+NIAH, RepoQA+KV.
- **Two modes:**
  - **Multi-turn:** the context is sent once, and follow-up queries see earlier answers.
  - **Multi-request:** the same context is reused by independent queries.
- **Purpose:** explicitly built around the KV-cache lifecycle (generation, compression, retrieval,
  loading). Queries are *not* visible when the context KV is built. This is the benchmark
  closest to a KV-eviction workload.

### OpenAI MRCR `[likely]` (via EvalScope port)
- **Design:** multi-turn synthetic chat with 2, 4 or 8 identical asks hidden in it. The last turn
  asks for the i-th instance and requires a random-string prefix.
- **Lengths:** 8 bins from 4K–8K up to 512K–1M tokens (o200k tokenizer).
- **Score:** SequenceMatcher ratio, and 0 if the prefix is missing.
- **Output:** long, a verbatim copy of an earlier assistant turn. The number of rows (about
  2,400?) is `[unverified]`.

### Fiction.LiveBench `[unverified]`
- Questions about deep fiction comprehension over stories trimmed to lengths of 0–192K tokens.
- Site blocked, so all details are recollection.

### LongMemEval, UCLA/Tencent, ICLR 2025 `[verified]` (type counts `[likely]`)
- **Question types** (500 total): single-session-user 70, single-session-assistant 56,
  single-session-preference 30, multi-session 133, knowledge-update 78, temporal-reasoning 133.
  About 30 abstention variants.
- **Settings:** S (about 115K tokens, about 40 sessions), M (about 500 sessions, about 1.5M
  tokens per the survey table), and oracle.
- **Format:** timestamped sessions, then the dated question.
- **Metric:** GPT-4o judge.

### LoCoMo, Snap/UNC, ACL 2024 (`[likely]`; counts computed `[verified]`)
- **Data:** locomo10 has 10 conversations (mean 27 sessions, 588 turns, 13.4K words) and 1,986
  QA.
- **QA categories:** single-hop 841, multi-hop 282, temporal 321, open-domain 96, adversarial
  446. The category-number-to-name mapping is by convention.
- **Other tasks:** event summarization and multimodal dialogue generation.
- **Multi-query:** about 199 QAs share each conversation.

### Long-output suites
- **LongBench-Write** `[verified]`:
  - 120 prompts (a 60-prompt EN subset); required lengths 100–20,000 words (mean 2,772).
  - 7 writing types.
  - Scores: S_l, a piecewise length score, and S_q, a GPT-4o quality judge.
  - LongWrite-Ruler has 48 prompts at 1K–30K words.
- **LongGenBench (Liu et al., Findings EMNLP 2024; "HUST" in the survey):**
  - Packs K questions into one prompt (GSM8K K=35; MMLU K=20 per subtask) and requires all K
    answers in order in one response `[verified]`.
  - CSQA inclusion is `[unverified]`.
- **LongGenBench (Wu et al., ICLR 2025; "NUS" in the survey)** `[verified]`:
  - 800 prompts: 4 scenarios × (100 short with about 16K-token outputs + 100 long with about
    32K-token outputs).
  - Scenarios: diary, menu, skyscraper, urban planning.
  - Constraints are single-instance, range or periodic, and each must be honored at the right
    unit deep into the output.
- **HelloBench** `[verified]`:
  - 647 prompts in 5 tasks: open-ended QA 200, summarization 100, chat 147, text completion 77,
    heuristic generation 123.
  - Inputs: about 5.7K tokens for summarization, 2.9K for completion, otherwise short.
  - Evaluated with the checklist-based HelloEval.

---

## 2. Prompt-template observations (query position and related details)

| Pattern | Where observed |
|---|---|
| Generic instruction before, instance question **after** (dominant) | LongBench, LongBench v2, RULER, ∞Bench (GPT-4/YaRN templates), HELMET (except ALCE), L-Eval, ZeroSCROLLS, LooGLE, BABILong, LV-Eval, NoCha, NIAH, passkey, NoLiMa, SCBench, LongMemEval, LoCoMo |
| Instruction repeated before and after the context | LongBench QA/summarization, RULER QA, LV-Eval |
| Question **before** the context | HELMET ALCE (`{instruction} Question: {q} {docs}`); Loong paper chain items (instruction first); ∞Bench Kimi template; NeedleBench "Start" option |
| Target stated before and after | ∞Bench Math.Find (prefix plus question) and Code.Run (function name before, call after) |
| Instruction only after the context (none before) | Counting-Stars; Loong (docs, then instruction, then question); HELMET JSON KV (context first) |
| Target marked inside the context | ∞Bench En.Dia (`$$MASK$$`); LCC/RepoBench (continue from the end) |
| One-shot/few-shot demos before the context | RULER VT/CWE; BABILong; HELMET recall/RAG/re-rank/cite (2-shot); LongProc (worked examples) |
| Shared prefix across test items | LongICLBench (one demo prompt for all items); SCBench (by design); L-Eval, LooGLE, NoCha, LoCoMo (many questions per document, but issued as separate prompts) |
| Truncation policy | LongBench v1/v2: middle truncation (keep head and tail). ZeroSCROLLS: truncate the document end and keep the query. HELMET: truncate or scale the number of passages/demos to the target length |

Implication. With the question at the end, observation-window compressors such as SnapKV and
PyramidKV score the context with the query already in view. This is a best case for query-aware
eviction. Very few suites measure the query-agnostic case, where the context is compressed before
the query is known (multi-turn chat, prefix caching). SCBench does, and so does the NeedleBench
"Start" option, though only partially.

---

## 3. Conceptual taxonomies of long-context tasks

### 3.1 Goldman et al. 2024, "Is It Really Long Context if All You Need Is Retrieval?" (EMNLP 2024)

(Axis definitions `[likely]`; per-task placements `[unverified]`.)

Link: https://aclanthology.org/2024.emnlp-main.924.pdf. The paper could not be opened; the axes
are well remembered and are consistent with the LCLM survey's paraphrase "dispersion of key
information and difficulty of extraction [153]".

- **Claim:** grouping tasks by input length alone is unproductive. Long-context tasks should be
  described by what makes them harder as context grows.
- **Axis I, Diffusion:** how hard it is to *find and extract* the needed information. Diffusion
  is high when the information is dispersed, implicit or paraphrased (non-literal), obscured by
  similar distractors, or needs interpretation to recognize.
- **Axis II, Scope:** *how much* needed information there is. It ranges from a single fact to a
  large share of the input.
- **Placement (gist):**
  - NIAH, passkey and KV retrieval are low diffusion and low scope. They test retrieval only.
  - Single-doc QA over natural text raises diffusion while scope stays low.
  - Aggregation and counting tasks, and summarization, raise scope.
  - The **high-diffusion, high-scope** quadrant is the most difficult and interesting, and it is
    **severely under-explored**.
- **Our tentative mapping of the suites here** (derived by us, not the paper's figure):
  - Low/low: RULER S-NIAH, MK-NIAH; ∞Bench Retrieve.*; passkey; NIAH; HELMET JSON KV; LV-Eval
    factrecall.
  - High diffusion, low scope: NoLiMa; LongBench PassageRetrieval (paraphrased abstracts);
    LV-Eval CFI tasks; NoCha claims; IDK.
  - Low diffusion, high scope: CWE/FWE; PassageCount; ∞Bench Math.Find; Counting-Stars;
    SpaceDigest; HTML→TSV; LongICLBench.
  - High/high (rare): LooGLE long-dependency tasks; Loong Clustering and Chain of Reasoning;
    LongBench v2 Detective and Event-ordering; BookSumSort; book summarization (∞Bench Sum);
    Fiction.LiveBench.

### 3.2 LCLM survey (Liu et al. 2025, "A Comprehensive Survey on Long Context Language Modeling", arXiv 2503.17407) `[verified]`

(Read from the PDF on GitHub.)

**Long-context comprehension** is organized as a five-level capability hierarchy:
- **Language modeling** (perplexity versus position/window).
- **Retrieval:** explicit (string match) versus semantic.
- **Aggregation:** statistical (counts, max/median, frequent words, variable tracking) versus
  semantic (e.g. SummHay).
- **Reasoning:** parallel (gather, then reason) versus iterative (each step decides the next
  lookup). Multi-needle reasoning, BABILong and NeedleBench are the examples.
- **Real-world adaptation:** QA, summarization, document retrieval and re-ranking, RAG, many-shot
  ICL, and code. Each is tagged with the core capabilities it relies on.

Benchmarks are split into synthetic (Table 6: Michelangelo, RULER, NoLiMa, BABILong, ...) and
real-world (Table 7). The survey observes that:
- Synthetic suites are mostly NIAH variants.
- Real-world suites are dominated by QA.
- "Excellence in synthetic tasks alone does not guarantee downstream competence."

**Long-form generation** task types are QA, Summarization, Instruction-Following and Mixed. An
output counts as "long" at more than 1,000 words for writing or more than 500 words for QA. The
survey names the two LongGenBench papers "HUST" (Liu) and "NUS" (Wu). For the "Thus Spake
Long-Context LLM" survey (arXiv 2502.17129) only the README was read. It lists evaluation papers
under "Long-Context Evaluation" (including Goldman et al. and Hyper-multi-step) and ends with 10
"unanswered questions". Its evaluation taxonomy was not checked.

### 3.3 Suite-internal taxonomies
- **RULER** `[verified]`: retrieval, multi-hop tracing, aggregation, QA. It adds complexity knobs:
  distractor needles, number of keys/values/queries, chain hops, word frequencies.
- **HELMET** `[verified]` categories, with the rationale `[likely]` from the paper:
  - Seven *application-centric* categories (recall, RAG, re-rank, cite, LongQA, summ, ICL).
  - Reasons: synthetic NIAH does not predict downstream performance; categories behave
    differently (RAG and recall correlate; re-rank, cite and ICL are distinct); n-gram metrics
    are unreliable for long QA and summarization (hence GPT-4o judging); base models need
    few-shot demos.
- **LooGLE** `[verified]`:
  - Short dependency: the answer comes from local evidence.
  - Long dependency: needs inter-dependency across multiple evidence spans widely spread over the
    text. Its subtypes are comprehension & reasoning, computation, timeline reorder, multiple
    information retrieval, and summarization.
- **Michelangelo LSQ** `[unverified]`: the context is a latent structure plus irrelevant content;
  queries must reveal the structure (list state, turn ordering, absence of information).
  Difficulty knobs, such as the number of relevant updates, are independent of length.
- **SCBench** `[verified]`:
  - Capabilities: string retrieval, semantic retrieval, global information processing,
    multi-tasking.
  - Two shared-context modes (multi-turn, multi-request) framed around the KV-cache lifecycle.
- **Loong** `[verified]`: evidence-structure levels, in order: spotlight locating, comparison,
  clustering, chain of reasoning.
- **NeedleBench** `[verified]`: retrieval (single/multi) versus reasoning, plus information-dense
  contexts (ATC, where all text is relevant).
- **Counting-Stars** `[verified]`: multi-evidence acquisition versus reasoning with in-place
  corrections, with controlled evidence positions.
- **LV-Eval** `[verified]`: single-hop versus multi-hop, plus confusing-fact insertion and keyword
  replacement (distractor and leakage controls).
- **NoLiMa** `[verified]`: literal versus latent (associative) matching between question and
  needle.
- **Other conceptual work** (titles verified via survey READMEs; content from memory):
  - ETHIC: an "information coverage" requirement, close to *scope*.
  - "Retrieval or Global Context Understanding? On Many-Shot ICL" (ManyICLBench): many-shot ICL
    tasks are either retrieval-like or need global understanding.
  - "Hyper-multi-step: The Truth Behind Difficult Long-context Tasks" (arXiv 2410.04422):
    difficulty comes from multi-matching and logic-based retrieval.
  - Chroma "Context Rot" (July 2025; repo verified): degradation depends on needle–question
    similarity, distractors and haystack structure, not length alone.
- **Directly relevant to the KV-eviction study** `[verified]`: **KVDiagnosis** (arXiv 2608.09412;
  https://github.com/ChosenQC/KVDiagnosis).
  - A failure-focused benchmark for KV-cache compression.
  - Uses 8 methods at 75/50/25% KV budgets on RULER-8K, RULER-16K, Qasper and HotpotQA.
  - Isolates 12,520 rows that are correct with the full cache but wrong under compression.
  - Measures per-layer and per-head evidence retention (ERR_slot, ECov_slot) and gold-answer
    likelihood drift.
  - Finding: low or partial evidence coverage is common. SnapKV and TOVA fail on largely
    different items.

---

## 4. Systematic gaps relevant to a KV-cache eviction workload taxonomy

1. **The question almost always comes at the end.** 83% of rows put the instance query after the
   context; only 2 rows (HELMET ALCE) put it before. Most suites therefore evaluate eviction in
   the easiest, query-aware regime. Query-agnostic prefill compression is tested by:
   - SCBench (multi-turn and multi-request);
   - optionally NeedleBench "Start";
   - indirectly Counting-Stars, whose fixed instruction comes only after the context, so nothing
     tells the model what to keep during prefill.
2. **One query per compressed cache.** Datasets with many questions per document (L-Eval,
   LooGLE, NoCha about 15 per book, LoCoMo about 199 per conversation, LongICLBench's shared demo
   prefix) are scored as independent prompts. Only SCBench makes a second, different query hit
   the *same* compressed cache. RULER MQ (4 keys), Counting-Stars (32 items) and LongGenBench-Liu
   (K questions) pack several needs into one prompt.
3. **Outputs are short.** Most tasks answer in ≤128 tokens: spans, letters, labels, retrieved
   values. Long outputs of about 1K tokens or more come mainly from no-context generation suites:
   LongBench-Write, LongGenBench, HelloBench. Long input combined with long output is rare:
   - ∞Bench Math.Calc (44K in, 44K out) and En.Sum (171K in, 1.1K out);
   - LongProc HTML→TSV (up to about 72K in, 8K out) and path traversal;
   - SCBench RepoQA and the Mix tasks;
   - MRCR (long verbatim copy);
   - HELMET ∞Bench Sum.

   So decode-time KV growth and eviction during generation are under-tested. Copy-heavy outputs
   (MRCR, RepoQA, HTML→TSV) are especially sensitive to losing exact token KVs.
4. **Evidence is mostly one or a few spans with high lexical overlap.** Whole-context
   aggregation exists (CWE/FWE, PassageCount, Math.Find, Counting-Stars, Loong Clustering,
   SpaceDigest/BookSumSort, summarization), but the high-diffusion, high-scope quadrant is thin
   (Goldman et al.).
5. **Updates and overwrites are rare.** Only a few tasks require keeping the *latest* value and
   dropping stale ones:
   - BABILong qa1–qa3 and qa8;
   - RULER VT;
   - Counting-Stars reasoning;
   - LongMemEval knowledge-update;
   - LongProc ToM;
   - Latent List.

   Recency-biased eviction such as sliding windows or StreamingLLM behaves very differently on
   these tasks.
6. **Absence and abstention are rare:** Qasper "unanswerable", IDK, LongMemEval abstention,
   LoCoMo adversarial. Eviction can turn a correct "not present" into a hallucination, and this
   is almost never measured.
7. **Evidence position is controlled only in synthetic suites:** RULER's 40 depths, NIAH
   sweeps, NoLiMa's 26 placements, Counting-Stars' even spacing, NeedleBench depths. Natural
   suites leave position uncontrolled, and truncation policies differ: middle truncation in
   LongBench versus keep-suffix in ZeroSCROLLS.
8. **Conversation structure is present but flattened.** LongMemEval, LoCoMo, MRCR and LongBench v2
   dialogue history are all chats, yet they are fed as one prompt with the question at the end.
   Only SCBench and MRCR preserve turn structure at inference time.
9. **Units and tokenizers vary:** words, characters, cl100k, o200k, Llama-3 or the model's own
   tokenizer. Length bins must be renormalized before cross-suite comparison.
10. **Language and domain:** English dominates. Chinese appears in LongBench, LV-Eval, ∞Bench,
    Loong, NeedleBench and Counting-Stars. Code appears in LongBench LCC/RepoBench, ∞Bench
    Code.*, LongBench v2 code repo, SCBench RepoQA and L-Eval CodeU.
11. **Contamination controls vary:** entity substitution (∞Bench), keyword replacement (LV-Eval),
    fresh books (NoCha), fictional needles (NeedleBench v2) and non-literal needles (NoLiMa).
    Many "natural" suites (NarrativeQA, GovReport, HotpotQA) risk parametric answering, which
    makes an eviction policy look better than it is.

### Suggested workload axes (derived from the above)

Suggested for the study's taxonomy:
- Input length.
- Output length (short answer, paragraph, multi-K).
- Query timing relative to prefill (before, after, later turn, none).
- Queries per cached context (1, k packed, k sequential).
- Evidence scope (1 span, k spans, global).
- Evidence diffusion (literal, paraphrased, distractor-dense, latent).
- Evidence position (depth-controlled or not).
- Temporal semantics (static, overwrite/latest-value, ordered).
- Output-copy fidelity (exact copy versus abstractive).
- Abstention requirement.
- Synthetic versus natural, and contamination control.
