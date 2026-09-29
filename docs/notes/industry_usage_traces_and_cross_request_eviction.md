> **Provenance.** Working evidence notes compiled during this study's literature survey (September 2026). "This session" refers to the survey run. Tags: [verified] = read in an official repository, released data or a primary document; [likely] = search-engine abstract or secondary source; [unverified] = recollection only. arXiv, OpenReview, ACL Anthology and Hugging Face were not reachable during the survey, so paper-body details are mostly [likely]. Working scripts mentioned below were not retained unless they appear under `analysis/`. The curated synthesis is in [`../literature.md`](../literature.md).

# Real-world LLM workloads and cross-request KV-cache reuse/eviction: literature and trace notes

Compiled 2026-09-29 for the study "A workload taxonomy for evaluating KV-cache eviction policies in LLM inference".

## Evidence tags

- **[verified]**: I read the primary source in this session (anthropic.com page, GitHub README or source file, or a paper PDF hosted on GitHub or microsoft.com), or I computed the number myself from the released trace file. Working scripts were not retained; comparable scripts are in `analysis/`.
- **[likely]**: The number comes from search-engine summaries of the primary source, and the snippets agree with each other. I could not open the primary source because arXiv, NBER, openai.com, openrouter.ai and USENIX were blocked.
- **[unverified]**: The claim comes from my background knowledge or from a single secondary source. Check it before citing.

## Access limits

- **Blocked:** arxiv, openreview, usenix, acm, huggingface, nber, openai.com, openrouter.ai, lmarena, arena.ai, lmsys.org and most blogs.
- **Worked:** anthropic.com, platform.claude.com, github.com, raw.githubusercontent.com, media.githubusercontent.com (Git LFS) and microsoft.com (Splitwise PDF).
- **Search budget:** the WebSearch budget ran out partway through Part 3. Several Part 3 policy details therefore rest on GitHub READMEs and source code, or are tagged [likely] or [unverified].

## Reproducibility

- All trace statistics marked "computed" come from working scripts (not retained): `stream_stats.py`, `burst1_stats.py`, `burst_sessions.py`, `mooncake_stats.py`, `bailian_stats.py`, `bailian_bytype.py`, `tracelab_stats.py` and `sharegpt_stats.py`.
- Their outputs are the matching `*.txt` files in the same folder.
- The Bailian `qwen_*.jsonl` files and the Mooncake JSONL files are still in that folder. I deleted the large ShareGPT and TraceLab downloads after the analysis.

---

## Part 1: What people use LLMs for (usage taxonomies with shares)

### 1.1 Anthropic Clio (Dec 2024)

Source: https://www.anthropic.com/research/clio. The paper is arXiv 2412.13678.

- **Sample** [verified]: 1M Claude.ai conversations (Free and Pro).
- **Web and mobile application development** [verified]: "over 10% of all conversations". Search snippets give the exact figure as **10.4%** [likely].
- **Content creation and communication:** **9.2%** [likely, search snippet].
- **Education, teaching and learning** [verified]: "more than 7%".
- **Business strategy and operations** [verified]: "nearly 6%".
- **Long tail** [verified]: Clio found "thousands of smaller conversation clusters", for example dream interpretation and D&D. Usage also varies by language.

### 1.2 Anthropic Economic Index (AEI)

All AEI numbers below were checked against verbatim quotes from the pages.

**Sep 2025 report (data from Aug 2025, "V3")** [verified]. Source: https://www.anthropic.com/research/anthropic-economic-index-september-2025-report

- **Coding:** coding "continues to dominate our total sample at 36%" (Computer & Mathematical tasks).
- **Educational tasks:** rose from 9.3% to 12.4%. **Scientific tasks:** rose from 6.3% to 7.2%.
- **Coding mix:** creating new code rose from 4.1% to 8.6%. Debugging and error correction fell from 16.1% to 13.3%.
- **Directive conversations** (the user delegates the whole task): rose from 27% to 39%.
- **API vs Claude.ai:** "77% of business [1P API] uses involve automation usage patterns, compared to about 50% for Claude.ai users".
- **API coding share:** "44% of API traffic in our sample was matched to a task characteristic of a Computer and Mathematical occupation".
- **Categories under-represented in the API:** education and library tasks are 12.3% on Claude.ai but 3.6% in the API. Arts and entertainment are 8.2% on Claude.ai but 5.2% in the API.
- **Input vs output length:** "each 1% increase in input length is associated with a less-than-proportional 0.38% increase in output length".

**Jan 2026 report (data from Nov 2025)** [verified]. Source: https://www.anthropic.com/research/anthropic-economic-index-january-2026-report

- **Sample size:** about 2M conversations, split evenly between Claude.ai and the 1P API [likely, snippet only].
- **Claude.ai coding share:** fell "from a peak of 40% in March 2025 to 34% in November 2025".
- **API coding share:** Computer & Mathematical went from 44% in August to 46% in November 2025.
- **Task concentration:** the top 10 tasks are 24% of Claude.ai conversations and 32% of API traffic (up from 28%). The single largest task, "modifying software to correct errors", is 6% of usage.
- **Claude.ai interaction mode:** augmented 52% (+5pp), automated 45% (−4pp).
- **Work vs personal:** Claude.ai is 46% work, 19% coursework and 35% personal. The API is 74% work-related.

**Mar 2026 report (data from Feb 2026)** [verified]. Source: https://www.anthropic.com/research/economic-index-march-2026-report

- **Coding:** Computer & Mathematical tasks are 35% of Claude.ai conversations. This category grew 14% in the API and shrank 18% on Claude.ai since Aug 2025.
- **Task concentration:** Claude.ai top-10 tasks fell from 24% to 19%. API top-10 rose from 28% to 33%.
- **Work vs personal:** personal use rose from 35% to 42%. Coursework fell from 19% to 12%.
- **Automation:** "automation decreased sharply in the 1P API data". Augmentation on Claude.ai increased slightly.

**Jun 2026 "Cadences" report (chat and Cowork conversations, Apr 10 to Jun 10 2026)** [verified]. Source: https://www.anthropic.com/research/economic-index-june-2026-report

- **Weekly cycle:** the personal-use share rises "from around 35% on weekdays to just under 50% on weekends".
- **Output types:** 93% of conversations produce an identifiable artifact. Explanations are 17%, documents and reports 15%, and guidance 11%. Code and technical work is "about a sixth".
- **Token use varies widely:** a typical explanation uses about a fifth of the median conversation's tokens.
- **Chat vs Claude Code:** the median blog-post conversation on chat or Cowork has 13 rounds of back-and-forth, versus a single human prompt in Claude Code. Claude Code runs with 0.37 points more autonomy on a 1–5 scale.
- **Occupations:** top-tercile occupations use 2.07× the tokens and 1.53× the turns of bottom-tercile ones.

**Apr 2025 software-development report (500K coding interactions, Apr 6–13 2025)** [verified]. Source: https://www.anthropic.com/research/impact-software-development

- **Automation:** 79% of Claude Code conversations are "automation", versus 49% on Claude.ai.
- **Interaction subtypes:** feedback loop is 35.8% on Claude Code vs 21.3% on Claude.ai. Directive is 43.8% vs 27.5%.
- **Languages:** JavaScript and TypeScript 31%, HTML and CSS 28%, Python 14%.
- **Adoption:** 33% of Claude Code conversations serve startup work.

**Affective-use study (Jun 2025)** [verified]. Source: https://www.anthropic.com/news/how-people-use-claude-for-support-advice-and-companionship

- Only 2.9% of Claude.ai conversations are affective.
- Companionship plus roleplay is under 0.5%.
- The starting sample was about 4.5M conversations.

### 1.3 OpenAI/NBER "How People Use ChatGPT" (NBER w34255, Sep 2025)

These numbers are [likely]: several search snippets agree, but the NBER and OpenAI PDFs were blocked. Sources: https://www.nber.org/papers/w34255 and https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf

- **Scale:** about 700M weekly active users and about 18B messages per week by July 2025. The analysis sample is about 1.1–1.5M conversations; snippets differ on the exact figure.
- **Topic shares, around June 2025:**
  - Practical Guidance: about 29%, stable.
  - Seeking Information: 24%, up from 14%.
  - Writing: 24%, down from 36%. Writing is 40% of work-related messages.
  - Practical Guidance, Seeking Information and Writing together: about 78–80%.
  - Technical Help: about 5%, down from 12%.
  - Multimedia: just over 7%, up from 2%.
  - Computer programming: 4.2% of messages, compared with 33% of work-related Claude conversations.
  - Relationships and personal reflection: 1.9%.
  - Games and roleplay (companionship): 0.4%.
- **Non-work share:** rose from 53% (Jun 2024) to 73% (Jun 2025).
- **Intent:** Asking 49%, Doing 40%, Expressing 11%.

### 1.4 OpenRouter "State of AI: an empirical 100-trillion-token study" (a16z/OpenRouter, Dec 2025; arXiv 2601.10088)

These numbers are [likely], from search snippets only. Sources: https://openrouter.ai/state-of-ai and https://a16z.com/state-of-ai/

- **Scope:** more than 100T tokens. The main analysis window is the 13 months ending Nov 2025.
- **Programming share of tokens:** grew from about 11% to over 50% by late 2025.
- **Roleplay and creative:** the second-largest category, and about 52% of open-source-model tokens.
- **Prompt length:** average prompt tokens per request grew about 4× (roughly 1.5K to over 6K). Programming requests "routinely exceed 20K input tokens".
- **Completion length:** average completion tokens grew about 3× (roughly 150 to 400). Average sequence length grew from under 2K to over 5.4K.
- **Reasoning:** reasoning-model share rose from negligible to over 50% of tokens by late 2025.
- **Agentic inference:** described as the fastest-growing behaviour. I have no numeric tool-call share.

### 1.5 WildChat (ICLR 2024; arXiv 2405.01470)

These numbers are [likely], from search snippets of the paper's Table 1.

- **Scale:** about 1M ChatGPT conversations (Table 1 lists 1,009,245; the full HF release lists 1,039,785) from 196,927 users.
- **Turns:** mean 2.52–2.54 per conversation; paper versions differ.
- **Lengths** (Llama-2 tokenizer): user tokens 295.58 ± 1609.18, assistant tokens 441.34 ± 410.91.
- **Languages:** 68.
- **Categories**, from 1,000 sampled first turns labelled by GPT-4:
  - Assisting or creative writing: 61.9%.
  - Analysis or decision explanation: 13.6%.
  - Coding: 6.7%.
  - Factual information: 6.3%.
- **Timestamps:** I believe WildChat has per-turn timestamps and hashed IPs, so session gaps could be derived from it [unverified; check the dataset card].

### 1.6 LMSYS-Chat-1M (arXiv 2309.11998) and ShareGPT, as reported in WildChat's Table 1

These numbers are [likely].

- **LMSYS-Chat-1M:** 1M conversations from 210,479 users. Mean 2.02 turns. User tokens 69.83 ± 143.49; assistant tokens 215.71 ± 1858.09. 65 languages.
- **LMSYS-Chat-1M topics:** coding and software-error clusters are the largest [likely, qualitative]. **I could not retrieve the exact cluster shares.**
- **ShareGPT:** 94,145 conversations, mean 3.51 turns, user tokens 94.46, assistant tokens 348.45, 41 languages.

### 1.7 ShareGPT as used in vLLM benchmarks

**Computed** [verified] from `ShareGPT_V3_unfiltered_cleaned_split.json`, a 673 MB Git LFS copy at github.com/cyril-k/load-generator. The canonical file is on HF: anon8231489123/ShareGPT_Vicuna_unfiltered.

- **Entries and splits:** 94,145 entries come from only 50,142 base conversation ids, because long conversations are split into segments (`_0`, `_1`, …).
- **Turn counts:** mean 7.46 messages per entry and 3.55 human turns (median 3, p90 7, p99 13). 73.6% of entries have at least 2 human turns.
- **Split artifact:** 34,538 entries (36.7%) begin with a `gpt` message.
- **Lengths in characters:**
  - First human message: median 604, mean 1,356.
  - First reply: median 555.
  - Full history before the last message: median 5,485, p90 8,942. The segment splitting bounds this.
- **Repeated opening prompts:** "hi" (240), "Explain quantum computing in simple terms" (190), TypingMind/ChatHub share headers, and similar.

**vLLM sampler** [verified]. Source: `vllm/benchmarks/datasets/datasets.py`, https://github.com/vllm-project/vllm

- It keeps entries with at least 2 messages and uses `conversations[0]` as the prompt and the length of `conversations[1]` as the output length.
- It drops sequences with a prompt or output under 4 tokens, a prompt over 1024 tokens, or prompt plus output over 2048 tokens.
- **So the standard ShareGPT benchmark has no multi-turn structure and essentially no cross-request prefix reuse.** For the 36.7% of split entries that start with `gpt`, the "prompt" is actually an assistant message.

### 1.8 Chatbot Arena / LMArena

- **Hard prompts** [verified]: "approximately 20% of prompts have a [hardness] score of 6 or higher", which defines the Hard Prompts category. Source: https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2024-05-17-category-hard.md
- **De-duplication** [verified, same source]: the top 0.1% most common prompts (about 1,000, mostly greetings) were downsampled, which removed about 8.6% of votes.
- **Llama-3 analysis** [verified]: about 27% of a 3.5K-battle sample was hard, and about 9% of prompts in Llama-3 battles were duplicates. Source: blog/2024-05-08-llama3.md
- **Category trends** [likely, snippet]: hard prompts and coding increased over time, while instruction-following, math and creative writing decreased. **I could not retrieve per-category shares.**
- **Open data release** [likely]: `lmarena-ai/arena-human-preference-140k` has about 140K conversations from Apr 17 to Jul 25, 2025.

---

## Part 2: Production traces and workload characterizations

### 2.1 Azure LLM inference trace 2023 (Splitwise, ISCA'24)

- **Repo and files:** https://github.com/Azure/AzurePublicDataset, files `data/AzureLLMInferenceTrace_code.csv` and `data/AzureLLMInferenceTrace_conv.csv`. Description: `AzureLLMInferenceDataset2023.md`.
- **Schema:** `TIMESTAMP, ContextTokens, GeneratedTokens`. There are no session ids and no content hashes, so reuse cannot be measured.
- **Paper** [verified, Splitwise PDF from microsoft.com]:
  - Traces come from two Azure services, coding and conversation. The characterization traces are 20 minutes long.
  - Coding: median prompt 1,500 tokens, median output 13 tokens ("generates the next few words").
  - Conversation: median prompt 1,020 tokens, median output 129 tokens. The output distribution is almost bimodal.
- **Computed from the released CSVs** [verified]:
  - The released timestamps are dated 2023-11-16 and cover about 57 minutes.
  - **Code:** n = 8,819 (2.57 req/s). Context median 1,469, mean 2,048, p99 7,436. Generated tokens median 13, mean 28, p99 249.
  - **Conversation:** n = 19,366 (5.53 req/s). Context median 1,020, mean 1,155, p99 4,142, max 14,050. Generated tokens median 129, mean 211, p99 601.

### 2.2 Azure LLM inference trace 2024 (DynamoLLM, HPCA'25), one week

- **Files:** `https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_{code,conv}_1week.csv` (692 MB and 1.14 GB). The schema is the same as 2023.
- **Computed** [verified, streamed]:
  - **Code** (May 10–16, 2024): n = 16,803,695. Context median 1,930, mean 2,511, p99 7,685, max 7,743. Generated median 8, mean 23, p99 271. Hourly request counts range from 9,086 to 301,105, a **peak/trough ratio of 33×**.
  - **Conversation** (May 12–18, 2024): n = 27,303,999. Context median 928, mean 1,632, p99 6,683, max 7,999. Generated median 41, mean 106, p99 694. Hourly peak/trough ratio **3.6×**.
  - Context appears capped at about 8K, which is probably the model's context limit.
- **Multimodal trace (ModServe, SoCC'25)** [verified]: `data/AzureLMMInferenceTrace_multimodal.csv.gz` covers Oct 15–22, 2024. Schema: `TIMESTAMP, NumImages, ContextTokens, GeneratedTokens`. I did not analyze it.

### 2.3 BurstGPT (Azure OpenAI GPT-3.5/GPT-4 at a university service; KDD'25; arXiv 2401.17644)

**Repo and files** [verified]:

- https://github.com/HPMLL/BurstGPT. `data/BurstGPT_1.csv` is in the repo.
- Release v2.0 has `BurstGPT_{1,2,3}.csv` and `BurstGPT_without_fails_{1,2,3}.csv` at `https://github.com/HPMLL/BurstGPT/releases/download/v2.0/<file>`.
- Coverage: parts 1+2 span 121 days (about 5.29M lines). Part 3 spans 110 days (5.34M lines).

**Schema** [verified]:

- Base columns: `Timestamp` (seconds since the first day), `Model` (ChatGPT, GPT-4), `Request tokens`, `Response tokens`, `Total tokens`, `Log Type` (Conversation log or API log).
- `BurstGPT_3` adds `Session ID` (conversation mode only) and `Elapsed time`.
- A response-token value of 0 means a failure.

**Computed from BurstGPT_1** (61 days, 1,429,737 rows) [verified]:

| Model and log type | Share of rows | Request tokens (median / p90) | Response tokens, successful only (median / p90) | Failure rate |
|---|---|---|---|---|
| ChatGPT API | 77.4% | 218 / 2,161 | 26 / 187 | 1.4% |
| GPT-4 API | 11.8% | 460 / 1,105 | 40 / 1,038 | 1.0% |
| ChatGPT conversation | 7.2% | 481 / 1,664 | 229 / 543 | 5.5% |
| GPT-4 conversation | 3.6% | 520 / 1,843 | 240 / 549 | 5.5% |

**Computed from BurstGPT_3** (5,344,021 rows) [verified]:

- **Sessions:** conversation logs are only 233,617 rows (4.4%), and all of them carry a session id. They form 55,920 sessions.
- **Turns per session:** mean 4.18, median 2, p90 9, p99 31, max 436. 35.5% of sessions have a single turn, and 91.5% of conversation requests belong to multi-turn sessions.
- **Think time** (gap minus previous elapsed time): median 124 s, p10 17 s, p90 2,331 s, p99 78,782 s.
- **History re-sending:** 60% of follow-up turns have request tokens ≥ the previous request plus 0.9 × the previous response, which is consistent with the full history being re-sent. The median ratio of next to previous request tokens is 1.27.
- **No content hashes**, so prefix reuse can only be inferred from session structure.

### 2.4 Mooncake (Moonshot AI / Kimi; FAST'25 Best Paper)

**Repo and files** [verified]:

- `https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release`.
- `traces/conversation_trace.jsonl`, `traces/toolagent_trace.jsonl`, `traces/synthetic_trace.jsonl`, plus the older `arxiv-trace/mooncake_trace.jsonl`. The paper with its trace appendix is `FAST25-release/Mooncake-FAST25.pdf`.
- Raw URL pattern: `https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/<file>`.

**Schema** [verified]:

- One JSON object per line: `{"timestamp": ms, "input_length", "output_length", "hash_ids": [...]}`.
- `hash_ids` are remapped **prefix** block hashes, with a **block size of 512 tokens**. Identical id means the prefix KV is reusable.
- Conversation and tool&agent are 1 hour of online traffic each. Synthetic is built from ShareGPT, L-Eval and LooGLE mixed 1:1:1, with Poisson arrivals.

**Paper, Table 2** [verified, PDF]:

| Trace | Requests | Avg input tokens | Avg output tokens | Cache ratio |
|---|---|---|---|---|
| Conversation | 12,031 | 12,035 | 343 | 40% |
| Tool&agent | 23,608 | 8,596 | 182 | 59% |
| Synthetic | 3,993 | 15,325 | 149 | 66% |

**Paper, other statements** [verified]:

- Conversation requests reach up to 128K tokens. Tool&agent prompts carry "pre-designed, often lengthy, system prompts that are fully repetitive".
- The synthetic trace's cache hits "are quite dispersed, thus requiring a substantial cache capacity".
- "The capacity of local DRAM supports only up to 50% of the theoretical cache hit rate."
- Mooncake Store uses LRU. The global cache gives up to 2.36× the hit rate of a local cache and saves up to 48% of prefill compute.
- "System prompts are accessed by almost every request, whereas caches storing content from a local long document may be used by only one user."
- Mooncake handles hotspots with heuristic replication of hot blocks.

**Earlier arXiv version** [likely; search snippet plus a third-party reproduction table]:

- LRU performed best among LRU, LFU and LengthAwareCache.
- Hit ratio was 30% at 1K blocks and 50% at 50K blocks, then flat. The published curve is 0.30 / 0.40 / 0.48 / 0.50 / 0.51 / 0.51 at 1k / 10k / 30k / 50k / 100k / unlimited blocks.

**Computed** [verified]. Ideal hit ratio means an unlimited cache with prefix matching. The last column is plain block-level LRU at 1% / 5% / 10% / 25% / 50% of the unique blocks.

| Trace | Duration, rate | Input median / p99 | Ideal hit ratio | LRU hit ratio at 1% / 5% / 10% / 25% / 50% of unique blocks |
|---|---|---|---|---|
| Conversation | 3,537 s, 3.40 req/s | 6,909 / 85,399 | 0.366 | .052 / .195 / .278 / .353 / .363 |
| Tool&agent | 3,537 s, 6.67 req/s | 6,346 / 61,525 (output median 30) | 0.553 | .346 / .449 / .508 / .551 / .552 |
| Synthetic | 1,022 s | 11,587 / 66,456 | 0.640 | .034 / .155 / .260 / .443 / .589 |

**Structure and caveats** (computed):

- **Shared first block:** every conversation request shares first block 0, a common system prompt. The tool&agent trace has only 4 distinct first blocks; 3 of them cover 10,938, 9,203 and 3,449 requests.
- **New-block reuse:** the fraction of new blocks that are ever reused is 24.2% (conversation), 21.5% (tool&agent) and 41.6% (synthetic). Median time to first reuse is 147 s, 123 s and 153 s.
- **Tool&agent and arXiv traces overlap:** `toolagent_trace.jsonl` is essentially the arXiv trace: same n = 23,608 and hash structure, with lengths a few tokens apart.
- **Coarse timestamps:** there are only 1,180 distinct timestamps per hour, so many requests tie and "recency" is ambiguous in replay.
- **Offset from published numbers:** my ideal ratios are 3–4 pp below the paper's Table 2. The independent `agentic-kv-cache` repo also reports a 4–6 pp offset against the arXiv curve (measured 0.553 vs 0.51 published). Treat absolute values as definition-dependent.

### 2.5 Alibaba Bailian / Qwen traces ("KVCache Cache in the Wild", USENIX ATC'25, arXiv 2506.02634)

**Repo and files** [verified]:

- https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon. The files are Git LFS; download them from `https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/<file>`.
- `qwen_traceA_blksz_16.jsonl`: to-C chat, 56 MB.
- `qwen_traceB_blksz_16.jsonl`: to-B API "task automation", collected Dec 2024, 96 MB.
- `qwen_thinking_blksz_16.jsonl`: reasoning, 28 MB.
- `qwen_coder_blksz_16.jsonl`: coder, 132 MB.
- Each file is a two-hour sample from one Qwen serving cluster.
- The official replayer is https://github.com/blitz-serving/trace-replayer. NVIDIA AIPerf supports the format as `--custom-dataset-type bailian_trace`.

**Schema** [verified]:

- Fields: `chat_id`, `parent_chat_id` (−1 for a root request), `timestamp` (seconds), `input_length`, `output_length`, `type` (text/search/image/file; the newer files also use api/thinking/coder), `turn`, `hash_ids`.
- `hash_ids` are salted SipHash-2-4 **16-token** blocks remapped to integers. They are computed after the chat template is applied.
- The FAQ explains two quirks: `<think>` tokens are stripped before the next turn, and the last block's hash can change because of padding.

**Paper** [likely, search snippets]:

- Ideal hit rates are **62% (Trace A) and 54% (Trace B)**.
- "KV$ reuses are skewed across requests, where reuses between single-turn requests are equally important as multi-turn requests."
- Reuse time and probability are diverse overall but predictable within a request category, where they fit an exponential distribution.
- The required cache capacity is moderate. For Trace A, Llama3-70B needs about 4× the available HBM. For to-B, about 2× the per-GPU HBM is enough.
- The proposed policy is workload-aware eviction by the reuse-probability distribution of each category. It gives +3.9% hits and up to 41.4% lower mean response time than LRU.

**Computed** [verified]. Ideal hit ratio uses prefix-contiguous matching with 16-token blocks and an unlimited cache. LRU columns are at 1% / 5% / 10% / 25% / 50% of the unique blocks.

| Trace | Requests (rate) | Input mean / p50 / p99 | Output mean / p50 / p99 | Single-request sessions | Follow-up turns (share of requests) | Parent→child gap p50 / p90 | Ideal hit ratio | Share of hits same-session / cross-session | LRU hit ratio at 1% / 5% / 10% / 25% / 50% |
|---|---|---|---|---|---|---|---|---|---|
| A (to-C) | 43,058 (5.98/s) | 2,331 / 1,046 / 14,359 | 430 / 375 / 1,641 | 61.0% | 46.3% | 110.6 s / 718.5 s | 0.579 | 0.711 / 0.289 | .176 / .410 / .500 / .562 / .578 |
| B (to-B) | 172,800 (24.0/s) | 915 / 574 / 6,294 | 92 / 39 / 1,006 | 100% (all turn 1) | 0 | – | 0.542 | 0 / 1.000 | .462 / .516 / .528 / .537 / .541 |
| Coder | 43,011 (5.97/s) | 5,748 / 4,540 / 14,406 | 808 / 469 / 5,835 | 74.7% | 38.6% | 117.7 s / 941.8 s | 0.664 | 0.398 / 0.602 | .404 / .553 / .604 / .645 / .660 |
| Thinking | 10,812 (1.50/s) | 4,763 / 3,680 / 24,962 | **3,651 / 1,665 / 34,529** | 94.7% | 11.1% | 28.6 s / 254.9 s | 0.462 | 0.282 / 0.718 | .367 / .430 / .445 / .457 / .460 |

- **Trace A composition:** 31,744 text, 8,187 search, 1,617 image and 1,510 file requests. Ideal hit ratio by type is text 0.692, image 0.781, file 0.551 and search 0.488.
- **Trace A reuse:** median reuse interval is 47.7 s for text, 4.8 s for image, 111.8 s for search and 152.4 s for file. About 42–52% of newly created blocks are ever reused, with median time to first reuse 123–153 s.
- **Trace B composition:** 150,936 api and 21,864 text requests. Median reuse interval is **2.3 s** for api and 39.2 s for text.
- **Trace B concentration:** only **2.3%** of new blocks created by api requests are ever reused. The hits come from a small set of hot shared prefixes, which is why LRU with 1% of unique blocks already reaches 0.46 of the 0.54 ideal.
- **Session split in Trace A:** requests in single-request sessions have an ideal hit ratio of 0.247 (24% of blocks). Requests in multi-request sessions have 0.684.

### 2.6 ServeGen (Alibaba Model Studio / Bailian; NSDI'26; arXiv 2505.09999)

**Repo and data** [verified]:

- https://github.com/alibaba/ServeGen.
- `data/{language/m-large|m-mid|m-small, reason/deepseek-r1, multimodal/mm-image}/chunk-*-dataset.json` and `chunk-*-trace.csv` hold per-client length distributions and rate/CV time series, not per-request traces.
- `data/conversations/conversations_hashed.json` holds multi-turn conversations. Turn fields are `turn, timestamp, input_token_count, output_token_count, input_tokens, output_tokens` (hashed).
- `data/offline/trace_{a..f}.jsonl` are batch jobs from ACDC (SOSP'26). Files a–c have `hash_ids` with 16-token blocks in a global namespace, plus `img_ids`. Files d–f have `input_length, output_length, duration`. Each has about 50K requests, and there are no arrival timestamps.

**README findings** [verified]:

- Arrivals are bursty beyond Poisson.
- Input and output length distributions shift over days and weeks.
- Multimodal (Qwen-VL) data composition is heterogeneous.
- Reasoning (DeepSeek-R1) output lengths are bimodal.
- Coverage is 12 models and billions of requests over 4 months. Naive workload generation under-provisions by 50%.

**Computed from `conversations_hashed.json`** [verified]:

- 1,616 conversations, all multi-turn (a selective release). Mean 3.54 turns (p50 2, p90 7, max 45) over a 23.9-hour span.
- Input tokens per turn: mean 20,220, p50 7,740, p99 161,963. Output: mean 1,066, p50 891.
- **Inter-turn gap:** median 308 s, p90 3,222 s, p99 22,354 s.
- **Growth per turn** (input minus previous input minus previous output): median +2,106 tokens.
- Caution: the hashed `input_tokens` list holds only about 0.2% as many entries as `input_token_count`. Check its semantics before using it for reuse analysis.

### 2.7 Agentic-coding traces (2025–2026)

**TraceLab** (UW SyFI; arXiv 2606.30560)

- **Repo and file** [verified]: https://github.com/uw-syfi/TraceLab. Latest release asset: `https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.jsonl.gz` (100 MB gz), plus a `.duckdb` file. License CC BY 4.0.
- **Content** [verified]: real Claude Code and Codex sessions from 52 developers, with no content hashes.
- **Fields** [verified]:
  - `session_id`, `round_index`, `model`.
  - Token accounting: `input_tokens_total = prefix_tokens + newly_append_tokens`. Claude rounds also have `claude_cache_{read,creation}_input_tokens`.
  - `output_tokens`, `reasoning_output_tokens`.
  - `timing_events[]`: ISO timestamps for `user_message`, `tool_result`, `text`, `reasoning`, `tool_call`.
  - `tools[]`: `tool_name`, `emitted_at`, `result_at`, `tool_wall_latency_ms`, `result_chars`.
  - `first_input_event_type`.
- **Computed from the latest release** [verified]:
  - **Size:** 665,453 rounds, 8,058 sessions, 52 users. 305,445 rounds are Claude and 360,008 are Codex. Top models are gpt-5.5, claude-opus-4-8, claude-opus-4-7 and gpt-5.6-sol.
  - **Input per round:** median **132,092 tokens**, p90 338,653, p99 856,444, max 999,944.
  - **Cached prefix:** 95.6% of all input tokens are cached prefix; per-round prefix/input median 0.992. On Claude rounds, cache read is 95.2%, cache write 4.7% and uncached 0.1% of input tokens.
  - **Appended per round:** median 1,045 tokens, p90 6,963. Consecutive-round growth median +743 tokens; 1.9% of transitions shrink (compaction).
  - **Output per round:** median 249 tokens.
  - **Tool calls:** 86.3% of rounds emit at least one tool call. Tool wall latency median **0.22 s**, p90 10.05 s, p99 151.6 s.
  - **Rounds per session:** median 16, mean 82.6, p99 1,124, max 21,351.
  - **Idle gap before the next round:** after a tool result, median **1.0 s** (p90 23.5 s; 0.7% over 5 min). Before a human message, median **121.5 s** (p90 1,058 s; **25.7% over 5 min**).
  - Caution: `prefix_tokens` reflects the provider's cache state (TTL and eviction), not an ideal cache.

**WEKA kv-cache-tester traces** (Claude Code captured through a proxy)

- **Repo** [verified]: https://github.com/callanjfox/kv-cache-tester, directory `traces/`: 739 traces with 59,204 requests; 19 traces contain nested sub-agents (70 total).
- **Schema** [verified]:
  - Top level: `id, models, block_size (64), tool_tokens, system_tokens, requests[]`.
  - Each request: `t` (seconds from conversation start), `type`, `model`, `in`, `out`, `hash_ids` (64-token blocks, **local, per-conversation scope**), `input_types`, `output_types`, `stop` (tool_use or end_turn), plus `api_time` and `think_time`.
- **README statistics** [verified]:
  - Requests per trace: median 48, mean 80.
  - Input per request: p50 109,903 tokens (p10 43,475; p90 300,118). Output per request: p50 218.
  - Think time: median 10 s. API time: median 6.5 s. Conversation duration: median 62 min.
  - Cache hit between consecutive requests: **median 96%** (mean 93%).
  - Tools plus system prefix: median about 15K tokens (about 10.5K tools and 2.6K system), roughly 60% of the median first request.

**SemiAnalysis "AgentX" corpus** (InferenceX)

- **Location** [verified via aiperf docs]: HF `semianalysisai/cc-traces-weka-062126` (393 traces with sub-agent SPAWN/JOIN fan-out; the canonical AgentX MVP corpus), `…-062126-256k` (capped at 256K), and `…-no-subagents-051826` (98 traces). Format is WEKA with 64-token blocks.
- **Models:** usually Claude Opus for the agent and Haiku for sub-agents.
- **AIPerf replay docs:** https://github.com/ai-dynamo/aiperf/tree/main/docs/tutorials (weka-trace.md, agentx-mvp.md, tracelab-trace.md, bailian-trace.md, baseten-trace.md, burst-gpt-trace.md).
- **Characterization by the independent repo `gauravapiscean/agentic-kv-cache`** [verified README; not peer-reviewed]:
  - **Size:** 393 sessions and 68,266 requests. Session span median 1.84 h.
  - **Gaps:** inter-request gap median **2.1 s**, p90 51.1 s, p99 3,426 s. 9.5% of gaps exceed 60 s, 3.3% exceed 300 s and 1.0% exceed 3,600 s.
  - **Lengths and turns:** input median 88,768 tokens. Requests per session median 70.
  - **Duty cycle:** median 13.9%, and 85.5% of sessions are active less than 50% of the time. The README contrasts this with a "most-cited" characterization that reports a 20% median and 70% of sessions under 50%; it does not name that source [unverified].

**AgentSysBench** ("From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems", arXiv 2608.15127)

These numbers are [likely]: a search snippet, also quoted in the agentic-kv-cache README.

- 35,037 sessions; 59.4% have at least one eviction event.
- "Cache evictions contribute 55.9% of the total cache-create tokens and account for 31.5% of aggregate monetary cost", driven by a 5-minute provider TTL colliding with 1–10 minute idle gaps.

**SGLang HiCache production anecdote** [verified; LMSYS blog 2025-09-10-sglang-hicache.md]

- A coding agent on Qwen3-Coder-480B had dialogues "past 25K tokens around 8 turns per session".
- With HiCache and 3FS, the hit rate went from 40% to 80% and average TTFT fell 56%.
- In a DeepSeek-R1 general-QA deployment, a cache hit cut TTFT by 84% versus recomputation.

### 2.8 Production prompt-cache TTL policy (provider side) [verified]

Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

- **Anthropic prompt caching:**
  - The default TTL is 5 minutes and is refreshed free on each hit. A 1-hour TTL is available.
  - Writes cost 1.25× base price (5 min) or 2× (1 h). Reads cost 0.1× base for standard models.
  - Lookup is by prefix hash at breakpoints, with at most 4 breakpoints and a 20-block lookback.
  - The minimum cacheable prompt is 512–4,096 tokens depending on the model.
  - The invalidation hierarchy is tools → system → messages.
- This is the "TTL-bound" regime, as opposed to the capacity-bound regime of an engine-local cache.

### 2.9 Other systems papers' workloads

These are [likely] or [unverified], from memory; verify before citing.

- DistServe (OSDI'24) evaluated on ShareGPT (chat), HumanEval (code completion) and LongBench (summarization).
- Llumnix (OSDI'24) used ShareGPT, BurstGPT (GPT-4 conversation) and synthetic length distributions.
- Marconi (MLSys'25) used LMSys, ShareGPT and SWE-bench traces. This one is [verified] from `artifact_evaluation.md`.
- The Bailian traces are also used by LMetric ("Simple yet effective LLM scheduling", OSDI'26) [verified, Bailian README].

---

## Part 3: Cross-request KV-cache management and eviction policies

**Terminology.** "KV eviction" in the compression literature (H2O, SnapKV, StreamingLLM, and most ICML/ACL 2025–26 "eviction" papers) drops tokens within a single request. That is **not** cross-request prefix-cache eviction. See the "Eviction" section of https://github.com/jjiantong/Awesome-KV-Cache-Optimization, where nearly all entries are intra-request. The taxonomy should keep the two apart.

### 3.1 Engine defaults

**vLLM automatic prefix caching** [verified]. Source: `docs/design/prefix_caching.md`

- The cache is hash-based and holds full blocks only.
- Blocks with ref_cnt = 0 sit in a doubly-linked free queue, and eviction pops the head (LRU).
- When a request frees its blocks, they are appended in reverse order, so tail blocks are evicted first: the last block "hash[es] more tokens and is less likely to be reused".
- `cache_salt` gives per-tenant isolation.
- The documented target workloads are long-document QA and multi-round conversation.

**SGLang RadixAttention** [verified]. Sources: LMSYS blog 2024-01-17-sglang.md; source `python/sglang/srt/mem_cache/evict_policy.py`, `cache_init_params.py` (default `"lru"`) and `arg_groups/choices.py`

- The cache is a radix tree with LRU over leaves, evicted recursively, plus cache-aware (longest-prefix) scheduling.
- The `--radix-eviction-policy` choices are `lru, lfu, slru, priority, tlru`. The code also has FIFO, MRU and FILO strategies.
- The paper's KV-sharing patterns are few-shot examples, self-consistency questions, multi-turn chat history and tree-of-thought search history.
- Benchmarks: MMLU 5-shot, HellaSwag 20-shot, ReAct agent, ToT on GSM-8K, JSON decode, synthetic 4-turn chat (short and long outputs), DSPy RAG and LLaVA-bench.

**SGLang T-LRU** ("Tail-Optimized LRU", Zhang et al., arXiv 2510.15152) [verified, docstring]

- For a conversation with history L and expected next prompt Q̂, only L + Q̂ − threshold tokens need to stay cached to keep the next prefill within the TTFT budget.
- Tokens beyond that are "TEL-safe" and are evicted first. After that, eviction is plain LRU.

**SGLang HiCache** [verified, blog]

- Tiers run from GPU to host memory to storage (3FS, Mooncake, NIXL, file).
- Write policies are write-through, write-through-selective (hit-count based) and write-back.
- Storage prefetch can be best-effort, timeout or wait_complete.

**LMCache** [verified]. Source: `lmcache/v1/config.py`, `cache_policy/`

- `cache_policy` defaults to "LRU". LFU, FIFO and MRU are also available.
- It supports CPU, disk and remote tiers, and is the home of CacheGen and CacheBlend.

**Mooncake Store** [verified]. Source: `docs/source/design/store/mooncake-store.md`

- Eviction is approximate LRU. It triggers at a 90% high watermark and frees 5% per pass.
- Leases (default 10 s TTL) protect objects in use.
- "Soft pin" for hot objects such as system prompts has a default 30-minute TTL and is evicted last. "Hard pin" objects are never evicted.
- Object groups follow an all-or-none rule with a shared TTL.
- In the FAST'25 paper, Mooncake uses LRU plus hotspot replication.

### 3.2 Research systems

The table below lists each system's policy, the workload property it exploits, and what it was evaluated on.

| System | Venue | Policy mechanism | Workload property exploited | Evaluation workloads | Tag |
|---|---|---|---|---|---|
| CachedAttention / AttentionStore | ATC'24 | Per-session KV kept in HBM→DRAM→SSD. Prefetch and eviction are scheduler-aware: the job queue's look-ahead tells it which sessions arrive next. Layer-wise preloading, async saving, and truncation decoupled from positional encoding. | Multi-turn session structure; the next-turn order is known from the queue | ShareGPT multi-turn | [likely; paper blocked, listing verified] |
| Pensieve | EuroSys'25 | GPU plus CPU two-tier stateful cache for conversations. Eviction weighs recompute cost against recency, and drops a conversation's leading tokens first because they are cheaper to recompute. | Per-session history reuse; position-dependent recompute cost | Multi-turn chat datasets (e.g., ShareGPT) | [unverified details; listing verified] |
| IMPRESS | FAST'25 | Multi-tier (GPU/CPU/SSD) prefix KV store. It loads only the KV of important tokens from SSD, and its cache management is importance-aware. | Prefix reuse at scale with I/O cost; token-importance sparsity | Long-context prefix-sharing QA | [unverified details; listing verified] |
| Marconi | MLSys'25 | Prefix caching for hybrid (Mamba plus attention) models. An admission filter caches SSM states only at likely-reuse points. Eviction is FLOP-aware, combining recency with FLOPs saved per byte (weight α); there is also an offline static-α oracle. | Reuse likelihood of branch points; unequal recompute value per byte | LMSys, ShareGPT and SWE-bench traces | [verified AE doc; mechanism likely] |
| RAGCache | TOCS'25 | Knowledge tree over retrieved documents with PGDSF (prefix-aware Greedy-Dual-Size-Frequency) replacement. | Document popularity, size and position (RAG) | RAG QA benchmarks | [likely] |
| CacheBlend | EuroSys'25 | Reuses KV of **non-prefix** chunks, recomputing only a small subset of tokens to repair cross-attention. | RAG chunks recur in different positions or orders | Musique, SAMSum, 2WikiMQA | [README verified; details likely] |
| CacheGen | SIGCOMM'24 | Encodes KV into compact bitstreams for storage and network streaming, with adaptive quality. | Cost of loading and storing reused context across the network | Long-context QA and summarization | [README verified] |
| InferCept | ICML'24 | During an augmentation or tool pause, it chooses per request between discard (recompute), preserve (keep in GPU) and swap (to CPU), picking the option with the least wasted GPU memory-time. | Tool-call pauses; interception duration | Arithmetic, QA/search, virtual environment, chatbot, image and TTS augmentations | [README verified; details likely] |
| KVFlow (+ ScaleSim) | NeurIPS'25 | Workflow-aware eviction by priority, evicting agents whose next activation is furthest away ("steps-to-execution" in an agent step graph), with overlapped CPU→GPU prefetch of KV and LoRA. ScaleSim adds invocation-distance-based memory management. Built on SGLang. | Known multi-agent workflow order; future invocation distance | Multi-agent workflows and simulations | [README verified; step-graph detail likely] |
| Continuum | 2025 preprint | On a tool call, pins the agent's KV in GPU memory with an **adaptive TTL** based on predicted tool duration and reuse benefit; program-level scheduling. | Tool-call pause length; multi-turn agent continuation | Agentic benchmarks (e.g., SWE-bench, BFCL) | [adaptive TTL confirmed by the agentic-kv-cache README; the rest is unverified] |
| TokenCake | EuroSys'27 | Spatial: agent-aware admission and priority (agent importance, DAG structure, progress, prefix affinity). Temporal: tool-lifecycle events offload idle KV to CPU during tool stalls and restore it afterwards. | Tool stalls; multi-agent DAG criticality | 24 static DAGs (648 calls). Input-cache reuse rose from 44.1% to 83.8% and end-to-end time fell 34% at 1 QPS. On 20 SWE-bench-Verified tasks with mini-swe-agent, throughput rose 27.6%. | [verified README] |
| KVCache Cache in the Wild policy | ATC'25 | Eviction priority is the predicted reuse probability from exponential reuse-time distributions profiled per request category (text, file, image, search, API). | Category-specific reuse time; single- vs multi-turn mix | Bailian traces A and B. +3.9% hits and up to 41.4% lower mean response time vs LRU. | [likely] |
| Mooncake (Conductor + Store) | FAST'25 | Global disaggregated cache with LRU, cache-aware prefill scheduling (prefix-match vs load) and hot-block replication. | System-prompt popularity; cross-node locality | Kimi conversation, tool&agent and synthetic traces | [verified] |
| Preble | ICLR'25 (my recollection; the awesome list's badge says ICLR 2024) | Distributed prompt scheduling that balances prefix-cache locality against load. | Shared long prefixes (tools, agents, few-shot, video QA) | Agent, tool, programming and QA workloads | [README verified; details likely] |
| Learning Agent Execution for KV-Cache Management | arXiv 2608.14624 | Learned model of agent execution drives agent-aware eviction and proactive prefix prefetching. | Agent execution pattern, predicted next call | Not retrieved | [verified title and one-line summary only] |
| HotPrefix / PRISM / Strata / DualPath | 2026 (SIGMOD, preprint, OSDI, SIGCOMM) | HotPrefix: hotness-aware admission and reuse. PRISM: hot-segment admission and retention. Strata: GPU/CPU/SSD tiers with cache-loading-aware scheduling. DualPath: storage bandwidth for agentic KV loading. | Prefix hotness; long-context tiering | Not retrieved | [verified titles and summaries only] |

### 3.3 Evidence on whether workload-aware eviction beats LRU

**Independent replay study `gauravapiscean/agentic-kv-cache`** [verified README; not peer-reviewed]

- **Setup:** a block-granular simulator with prefix-contiguous hits, LRU over radix leaves and pinning of the in-flight chain, replaying AgentX and Mooncake.
- **Hit rate** on 40 AgentX sessions (4,751 requests):

| Cache size (blocks) | LRU-leaf | TTL-300s | LFU-leaf | + hazard-based liveness | + recompute cost | + coherent session eviction |
|---|---|---|---|---|---|---|
| 8K | 83.48% | 83.48% | 63.61% | 82.89% | 71.77% | 68.63% |
| 20K | 93.92% | 93.92% | 69.96% | 93.61% | 84.86% | 78.89% |
| 50K | 95.76% | 95.76% | 79.58% | 95.68% | 94.45% | 91.40% |

- Every addition made hit rate worse than LRU-leaf.
- **TTL-300s never fired** under capacity pressure; its results were byte-identical to LRU.
- **Where recompute comes from** (40K-block cache): gaps under 10 s cause 33.1% of recompute tokens; gaps over 5 min cause 17.5%.
- **Harness warning:** Belady lost to LRU until in-flight chains were pinned.
- Flat block LRU and radix-leaf LRU differed by only 0.02 pp on Mooncake.

**Provider TTL regime** (AgentSysBench, [likely])

- Evictions account for 31.5% of cost when a 5-minute TTL collides with 1–10 minute idle gaps.
- Together with the study above, this implies policy value depends on the regime: capacity-bound versus TTL/cost-bound.

**KVCache-in-the-wild** [likely]

- On Bailian traces, a workload-aware, category-based policy improves hits modestly (+3.9%) but latency substantially (up to 41.4%) at limited capacity.

---

## Part 4: Implications for the taxonomy and test-workload design

These are my synthesis, drawn from the data above.

1. **Reuse source is the primary axis.**
   - (a) A small hot set of shared prefixes, such as system prompts, tool definitions and templates. Examples: Bailian to-B is 100% single-turn and all its hits are cross-request, only 2.3% of new blocks are ever reused, and LRU at 1% of capacity reaches 85% of ideal. Mooncake tool&agent has 3 prefixes covering nearly all requests.
   - (b) Per-session history. Bailian to-C has 71% same-session hits and a median of about 110 s to the next turn.
   - (c) Agentic self-reuse. TraceLab has 95.6% cached-prefix input and a 1 s median gap.
   - (d) Non-prefix or chunk reuse (RAG and CacheBlend-style), which no public trace hashes capture.
2. **Inter-request gaps are bimodal.** Tool loops take about 1–2 s (TraceLab median 1.0 s; AgentX 2.1 s). Human think time is about 2 minutes (TraceLab 121.5 s, BurstGPT 124 s, Bailian 111–118 s, ServeGen 308 s), with a heavy tail: 25.7% of human gaps exceed 5 min. The ratio between the two modes and the tail should be generator parameters. The tail sets TTL behaviour; tool loops set capacity pressure.
3. **Per-session working-set size spans about 3 orders of magnitude.** It runs from about 1K tokens (Azure 2023/2024 conversation median 928–1,020; ShareGPT first turns) through 7–12K (Mooncake, ServeGen) to 90–130K per request (AgentX, WEKA, TraceLab; p99 above 850K). Multi-session working sets exceed GPU KV capacity, so partial-prefix retention (T-LRU, Pensieve) and tiering (HiCache, Mooncake Store) become first-order.
4. **Test two cache regimes explicitly.**
   - Capacity-bound, engine-local caches, where LRU-leaf is strong and TTL is irrelevant.
   - TTL- or cost-bound provider caches: Anthropic's 5-minute default, where AgentSysBench attributes 31.5% of cost to evictions.
   - Liveness or idle-aware policies can only win when gaps are long relative to cache churn.
5. **Popularity skew and pinning.** One system prompt can be block 0 of every request (Mooncake conversation). Claude Code's tools-plus-system prefix is about 15K tokens, about 60% of the first request. Protecting hot shared prefixes (Mooncake soft pin, SLRU, `priority`) is a policy dimension separate from recency. Test workloads should vary the number, size and Zipf skew of shared prefixes.
6. **Reasoning outputs are large and often not reused.** Bailian thinking has mean output 3,651 and p99 34.5K, and `<think>` content is stripped from the next turn. Generated reasoning KV therefore has low reuse value, which calls for admission control (Marconi-style) and "don't cache decode KV" variants.
7. **Mix and temporal non-stationarity matter.**
   - Coding and agentic work dominates token volume: OpenRouter programming is over 50%, Anthropic API is 44–46% Computer & Math, and Claude Code is 79% automation.
   - Consumer chat is mostly guidance, information and writing: ChatGPT coding is 4.2%.
   - Load is diurnal: Azure 2024 code has a 33× peak/trough in hourly requests, while conversation is 3.6×.
   - The mix shifts weekly: Anthropic personal share goes from 35% on weekdays to about 50% on weekends.
   - Weight categories by tokens, not requests, and drive the mix over time.
8. **Standard benchmarks and many traces cannot evaluate eviction.**
   - vLLM's ShareGPT sampler keeps only the first turn (no reuse). Azure 2023/2024 and Splitwise have no sessions or hashes. BurstGPT has sessions but no hashes.
   - Traces that carry hashes use block sizes of 16 (Bailian, ServeGen offline), 64 (WEKA, AgentX) or 512 (Mooncake). Re-blocking, which is only possible to coarser sizes, is needed to compare them.
   - AgentX and WEKA hashes are session-local, so cross-session sharing is invisible. TraceLab has no hashes; its prefix tokens reflect provider cache state.
9. **Simulator fidelity checklist** (from the agentic-kv-cache study and the trace quirks above):
   - Use prefix-contiguous hits, radix-leaf eviction and in-flight pinning.
   - Check that Belady beats LRU as a sanity test.
   - Handle coarse and tied timestamps: Mooncake has only 1,180 distinct timestamps per hour.
   - Validate against a published curve; Mooncake shows a 3–6 pp offset from its published figures.

---

## Items I could not retrieve (do not cite without checking)

- Exact LMSYS-Chat-1M topic-cluster percentages.
- Exact Chatbot Arena category shares.
- OpenRouter tool-call share.
- ServeGen and KVCache-in-the-wild paper figures beyond the snippets above.
- Evaluation details for Continuum, Pensieve and IMPRESS, and Continuum's arXiv id.
- Content of the 2026 preprints listed by title only: "A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing" (arXiv 2608.13573) and "The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems" (arXiv 2609.27746).
