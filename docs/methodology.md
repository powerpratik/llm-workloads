# Methodology, reproduction and threats to validity

## 1. Reproducing every number

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash analysis/fetch_data.sh   # 1.8 GB of public data (~2 min) into ./data (git-ignored; LLMW_DATA overrides)
bash analysis/run_all.sh      # every step below, under 10 minutes on 4 cores; per-step logs in data/logs/
git status                    # no changes = every committed output was reproduced
```

`run_all.sh` runs these steps; each can also be run on its own from `analysis/` (or the repository root for the
last three):

| step | script | outputs |
|---|---|---|
| tokenizers | `build_tokenizers.py` | Llama-3 and Qwen2 tokenizers rebuilt offline; asserts an exact match with llama.cpp's test vectors |
| M1 | `length_profiles.py` | `results/length_profiles.csv` |
| M2 | `reuse_distance.py` | `results/reuse_distance.csv` |
| M3, M4 | `context_scope.py` | `results/context_scope.csv`, `locomo_*.csv`, `context_redundancy.csv` |
| M5 | `agent_trajectories.py` | `results/agent_*.csv`, `agent_growth.json` |
| M6 | `serving_traces.py "<trace>"` as five parallel replays, then `serving_traces.py --merge` | `results/serving_*_all.csv` |
| literature | `literature_coverage.py` | `results/literature_coverage*.csv`, Fig. 7 |
| catalog | `taxonomy/build_catalog.py`, `tools/coverage.py --matrix`, `tools/render_catalog.py` | `taxonomy/benchmarks.yaml`, `results/suite_coverage.*`, `docs/benchmark_mapping.md` |
| figures, tables | `make_figures.py`, `render_tables.py` | `figures/`, `docs/tables.md` |

All scripts are deterministic; the reference run used Python 3.11 on 4 CPU cores. CSV, JSON and markdown
outputs reproduce exactly; figure bytes can vary with the matplotlib version and installed fonts.

## 2. Data provenance

| data | location | licence / notes |
|---|---|---|
| Qwen-Bailian traces (to-C, to-B, thinking, coder) | github.com/alibaba-edu/qwen-bailian-usagetraces-anon (Git LFS) | Apache-2.0; 2-hour samples; salted block hashes (16 tokens) |
| Mooncake traces | github.com/kvcache-ai/Mooncake `FAST25-release/traces` | 1-hour samples; 512-token block hashes; millisecond timestamps with many ties |
| Azure LLM inference 2023 | github.com/Azure/AzurePublicDataset | CC-BY; lengths only |
| BurstGPT (first two months) | github.com/HPMLL/BurstGPT `data/BurstGPT_1.csv` | lengths, model, conversation vs API |
| SWE-bench Verified trajectories | s3://swe-bench-submissions (public), mini-SWE-agent submissions for Claude Sonnet 4, Qwen3-Coder-480B, GPT-5 | full message histories |
| τ-bench trajectories + tool schemas | github.com/sierra-research/tau-bench | historical trajectories (GPT-4o, Claude 3.5 Sonnet) |
| Benchmarks | official GitHub repositories (see `analysis/fetch_data.sh`) | used for measurement only; not redistributed |

Nothing was fetched from Hugging Face: every source above is reachable through GitHub or the public S3 bucket.

## 3. Measurement definitions

- **Tokens.** Llama-3 BPE (128,256 vocabulary), rebuilt from llama.cpp's `ggml-vocab-llama-bpe.gguf`. It reproduces
  all 46 llama.cpp reference tokenizations exactly. Chat templates add ~5 tokens per message; this is added
  explicitly where messages are concatenated.
- **Length profiles.** Per sample, the prompt is what is prefilled in the standard setting (for example, lm-eval's
  8-shot GSM8K CoT exemplars). The output is the real model output where available (AlpacaEval, Arena-Hard,
  agent trajectories) and the reference answer otherwise; the table says which.
- **Lexical dependency distance.** Only informative output 4-grams count: those containing a digit or a
  non-stopword of three or more characters. A *copy event* is such a 4-gram that occurred earlier in prompt or
  output. Its distance is the gap to the most recent earlier occurrence, a lower bound on the reach of the
  dependency. MR(W) is the share of copy events with distance > W: the miss ratio of a pure W-token sliding window.
- **Evidence scope.** Contexts are split into 128-token chunks. The target is the content words of the reference
  answer that occur in the context (for QuALITY, the text of the gold option). Greedy set cover picks the fewest
  chunks covering 80% of them (k80). Union scope and pairwise Jaccard are computed over the questions attached to
  the same context.
- **Redundancy.** zlib level-9 compression ratio of the context text, plus the share of repeated token 4-grams.
- **Agent growth.** The prompt of LLM call *k* is all messages before assistant message *k*, plus ~5 template
  tokens per message. Peak context is prompt plus output of the last call. Messages logged after the last call
  never reach the model and are excluded; this matters for trajectories that log a huge final diff.
- **Serving replay.**
  - Prefix-contiguous hits: the leading run of resident block hashes.
  - Within a request, blocks are touched in reverse order, so ancestors are always more recent (as in vLLM /
    SGLang).
  - LRU miss-ratio curves come from Mattson stack distances. The distance of each lookup is measured against the
    pre-request cache state, which is exact for this touch order.
  - FIFO, LFU (LRU tie-break) and Belady's OPT are simulated explicitly.
  - Only input blocks are inserted; generated blocks are not cached. This is a conservative choice for multi-turn
    traces.

## 4. Threats to validity

**Internal**

- **Lexical proxies are lower bounds.** Paraphrased, recomputed or behaviorally obeyed dependencies (instructions,
  policies) are invisible to copy detection. For example, agents copy their policy text in only 1–5% of copy
  events, yet must obey it throughout.
- **Nearest-occurrence distances under-state aggregation**, which needs every occurrence.
- **Oracle scope depends on its parameters.** It depends on the chunk size (128) and the stop-list. Summaries
  paraphrase, so their lexical scope is a lower bound. Multiple-choice QuALITY uses the gold option text as a
  proxy answer.
- **Reference answers under-state decode length.** We show real outputs next to them where possible (AIME
  references are 1.3K tokens; R1-distilled models average 15.5K per R-KV).
- **Trace replay simplifications.** No pinning of in-flight requests. Generated blocks are not cached. Mooncake's
  coarse timestamps compress reuse-time distributions. Two independent re-implementations agree within ±2 points
  on the ideal hit ratios.

**External**

- **Model and harness choices.** Agent trajectories come from one scaffold (mini-SWE-agent) and three models.
  SWE-agent and OpenHands keep larger contexts: means of 34–48K were measured on public trajectories by the
  companion survey. Production agentic coding is larger still (median 110–132K input tokens).
- **Short samples.** Production traces are 1–2-hour samples from two providers, and diurnal and weekly patterns
  are not represented.
- **Tokenizer differences.** Llama-3 token counts differ from other tokenizers: Qwen2 gives +4–5% on the English
  samples checked, and code and Chinese differ more.

**Construct**

- **Judgment in the catalog.** The facet assignment of catalogued tasks follows documented rules and overrides
  (`taxonomy/build_catalog.py`). Most long-context facts are verified against official repositories (197 of 228
  rows); the rest are tagged.
- **Heuristic literature mapping.** The literature-coverage analysis maps free-text benchmark lists with keyword
  rules (`analysis/literature_coverage.py`). We audited the per-method rows in `results/literature_coverage.csv`
  and corrected false positives (demo-only uses, observation analyses, short summarization). Residual
  misclassification is possible, but it would not change the qualitative gap: in 2023–24, 0% of methods covered
  long reasoning or agents.
- **The taxonomy predicts sensitivity; it does not measure it.** It is designed to predict *where* eviction
  policies differ. The model-based validation in [recommendations §6](recommendations.md) is the test of that
  claim.
