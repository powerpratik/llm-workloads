# References

Keys are used in `taxonomy/taxonomy.yaml` (`evidence:` fields) and throughout the docs.

**How identifiers were checked.** arXiv IDs are given only where the survey confirmed them from an official
repository (README badge or BibTeX), a proceedings or anthology URL, or the released data. Entries marked † carry
venue or details confirmed only at abstract level. Entries marked ‡ are cited from general knowledge and could not
be re-checked in this environment. Per-fact confidence tags are in [`notes/`](notes) and
[`taxonomy/sources/`](../taxonomy/sources).

## Workload taxonomies, benchmarks and evaluation methodology

| key | reference |
|---|---|
| goldman2024 | O. Goldman, A. Jacovi, A. Slobodkin, A. Maimon, I. Dagan, R. Tsarfaty. *Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP.* EMNLP 2024. arXiv 2407.00402 |
| helm2022 | P. Liang et al. *Holistic Evaluation of Language Models.* TMLR 2023 ‡ (schemas verified in `stanford-crfm/helm`) |
| bigbench2022 | A. Srivastava et al. *Beyond the Imitation Game (BIG-bench).* TMLR 2023 ‡ (keyword taxonomy verified in `google/BIG-bench`) |
| bbh2022 | M. Suzgun et al. *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* Findings of ACL 2023 ‡ |
| chang2023survey | Y. Chang et al. *A Survey on Evaluation of Large Language Models.* 2023 ‡ (structure verified in `MLGroupJLU/LLM-eval-survey`) |
| guo2023survey | Z. Guo et al. *Evaluating Large Language Models: A Comprehensive Survey.* 2023 ‡ (structure verified in `tjunlp-lab/Awesome-LLMs-Evaluation-Papers`) |
| longbench2024 | Y. Bai et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL 2024. arXiv 2308.14508 |
| longbenchv2 | Y. Bai et al. *LongBench v2.* 2024/2025 ‡ (data and tables verified in `THUDM/LongBench`) |
| ruler2024 | C.-P. Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024 †. arXiv 2404.06654 |
| infinitebench2024 | X. Zhang et al. *∞Bench: Extending Long Context Evaluation Beyond 100K Tokens.* ACL 2024 ‡ |
| helmet2025 | H. Yen et al. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR 2025. arXiv 2410.02694 |
| leval2024 | C. An et al. *L-Eval: Instituting Standardized Evaluation for Long Context Language Models.* ACL 2024. arXiv 2307.11088 |
| zeroscrolls2023 | U. Shaham et al. *ZeroSCROLLS.* Findings of EMNLP 2023 ‡ |
| loogle2024 | J. Li et al. *LooGLE: Can Long-Context Language Models Understand Long Contexts?* ACL 2024 ‡ |
| babilong2024 | Y. Kuratov et al. *BABILong.* NeurIPS 2024 Datasets & Benchmarks (proceedings URL in repo) |
| michelangelo2024 | K. Vodrahalli et al. *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries.* 2024 ‡ |
| scbench2025 | Y. Li et al. *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR 2025. arXiv 2412.10319 |
| liu2024lostmiddle | N. F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv 2307.03172 |
| lclmsurvey2025 | J. Liu et al. *A Comprehensive Survey on Long Context Language Modeling.* 2025 (taxonomy read from the GitHub-hosted PDF) |
| locomo2024 | A. Maharana et al. *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo). ACL 2024 ‡ (category ids verified in `snap-research/locomo` evaluation code) |
| longmemeval2025 | D. Wu et al. *LongMemEval.* ICLR 2025 ‡ |
| quality2022 | R. Y. Pang et al. *QuALITY: Question Answering with Long Input Texts, Yes!* NAACL 2022 (abstract in the repo README) |
| crosscodeeval2023 | Y. Ding et al. *CrossCodeEval.* NeurIPS 2023 Datasets & Benchmarks (repo README) |
| longwriter2025 | Y. Bai et al. *LongWriter* (LongBench-Write). ‡ |
| longgenbench_liu | X. Liu et al. *LongGenBench: Long-context Generation Benchmark.* Findings of EMNLP 2024. arXiv 2410.04199 |
| longgenbench_wu | Y. Wu et al. *LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs.* ‡ |
| hellobench2024 | H. Que et al. *HelloBench.* 2024 ‡ |
| longproc2025 | X. Ye et al. *LongProc.* 2025 ‡ |
| nolima2025 | A. Modarressi et al. *NoLiMa.* 2025 ‡ |
| mtbench2023 | L. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 ‡ |
| mtbench101 | G. Bai et al. *MT-Bench-101.* ACL 2024 ‡ |
| alpacaeval | Y. Dubois et al. *Length-Controlled AlpacaEval.* 2024 ‡ (outputs from `tatsu-lab/alpaca_eval`) |
| arenahard | T. Li et al. *From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder.* 2024 ‡ (data from `lm-sys/arena-hard-auto`) |
| ifeval2023 | J. Zhou et al. *Instruction-Following Evaluation for Large Language Models.* 2023 ‡ |
| gsm8k2021 | K. Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021 ‡ |
| math2021 / prm800k | D. Hendrycks et al. *MATH.* NeurIPS 2021 D&B; H. Lightman et al. *Let's Verify Step by Step* (MATH-500 split). ICLR 2024 ‡ |
| humaneval2021 / mbpp2021 | M. Chen et al. *Evaluating LLMs Trained on Code* (HumanEval); J. Austin et al. *Program Synthesis with LLMs* (MBPP). 2021 ‡ |
| swebench2024 | C. E. Jimenez et al. *SWE-bench.* ICLR 2024 ‡; trajectories: `SWE-bench/experiments` (mini-SWE-agent submissions) |
| taubench2024 | S. Yao et al. *τ-bench.* 2024 ‡ (trajectories and tool schemas from `sierra-research/tau-bench`) |
| pg19 | J. W. Rae et al. *Compressive Transformers for Long-Range Sequence Modelling* (PG19). ICLR 2020 ‡ |

## KV-cache eviction / compression methods

| key | reference |
|---|---|
| streamingllm2024 | G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv 2309.17453 |
| h2o2023 | Z. Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* NeurIPS 2023. arXiv 2306.14048 |
| scissorhands2023 | Z. Liu et al. *Scissorhands.* NeurIPS 2023. arXiv 2305.17118 |
| fastgen2024 | S. Ge et al. *Model Tells You What to Discard.* ICLR 2024. arXiv 2310.01801 |
| tova2024 | M. Oren et al. *Transformers are Multi-State RNNs.* EMNLP 2024. arXiv 2401.06104 |
| keyformer2024 | M. Adnan et al. *Keyformer.* MLSys 2024. arXiv 2403.09054 |
| snapkv2024 | Y. Li et al. *SnapKV.* NeurIPS 2024. arXiv 2404.14469 |
| pyramidkv2024 | Z. Cai et al. *PyramidKV.* arXiv 2406.02069 |
| pyramidinfer2024 | D. Yang et al. *PyramidInfer.* Findings of ACL 2024. arXiv 2405.12532 |
| nacl2024 | Y. Chen et al. *NACL.* ACL 2024 †. arXiv 2408.03675 |
| devoto2024l2 | A. Devoto et al. *A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression.* EMNLP 2024. arXiv 2406.11430 |
| sirllm2024 | Y. Yao et al. *SirLLM.* ACL 2024. arXiv 2405.12528 |
| d2o2025 | Z. Wan et al. *D2O.* ICLR 2025. arXiv 2406.13035 |
| quest2024 | J. Tang et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. arXiv 2406.10774 |
| infinigen2024 | W. Lee et al. *InfiniGen.* OSDI 2024. arXiv 2406.19707 |
| arkvale2024 | R. Chen et al. *ArkVale.* NeurIPS 2024 (paper PDF in repo) |
| vatp2024 | Z. Guo et al. *Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters.* EMNLP 2024. arXiv 2406.12335 |
| lminfinite2024 | C. Han et al. *LM-Infinite.* NAACL 2024. arXiv 2308.16137 |
| adakv2025 | Y. Feng et al. *Ada-KV.* NeurIPS 2025. arXiv 2407.11550 |
| cake2025 | Z. Qin et al. *CAKE.* ICLR 2025. arXiv 2503.12491 |
| headkv2025 | Y. Fu et al. *Not All Heads Matter* (HeadKV). ICLR 2025. arXiv 2410.19258 |
| duoattention2025 | G. Xiao et al. *DuoAttention.* ICLR 2025. arXiv 2410.10819 |
| razorattention2025 | *RazorAttention.* ICLR 2025. arXiv 2407.15891 |
| dynamickv2025 | X. Zhou et al. *DynamicKV.* EMNLP 2025 †. arXiv 2412.14838 |
| magicpig2025 | Z. Chen et al. *MagicPIG.* ICLR 2025. arXiv 2410.16179 |
| shadowkv2025 | H. Sun et al. *ShadowKV.* ICML 2025. arXiv 2410.21465 |
| kvzip2025 | J.-H. Kim et al. *KVzip.* NeurIPS 2025. arXiv 2505.23416 |
| kvpress / expectedattention | NVIDIA kvpress (github.com/NVIDIA/kvpress); A. Devoto et al. *Expected Attention.* arXiv 2510.00636 |
| keydiff2025 | *KeyDiff.* 2025 †. arXiv 2504.15364 |
| trimkv2025 | N. Bui et al. *TRIM-KV.* 2025 †. arXiv 2512.03324 |
| scope2025 | J. Wu et al. *SCOPE.* ACL 2025. arXiv 2412.13649 |
| morphkv2025 | R. Ghadia et al. *Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs* (MorphKV). ICML 2025. arXiv 2503.00979 |
| rkv2025 | *R-KV: Redundancy-aware KV Cache Compression for Reasoning Models.* NeurIPS 2025 (repo). arXiv 2505.24133 |
| lazyeviction2025 | H. Zhang et al. *LazyEviction.* 2025 †. arXiv 2506.15969 |
| rpc2025 | J. Song et al. *Reasoning Path Compression.* NeurIPS 2025. arXiv 2505.13866 |
| thinkv2025 | *ThinKV.* 2025 †. arXiv 2510.01290 |
| rlkv2025 | W. Du et al. *Which Heads Matter for Reasoning?* (RLKV). 2025. arXiv 2510.08525 |
| epicache2025 | M. Kim et al. *EpiCache.* 2025. arXiv 2509.17396 |
| rocketkv2025 | *RocketKV.* ICML 2025. arXiv 2502.14051 |
| stepkv2026 / nexus2026 | *StepKV* arXiv 2609.22158; *Nexus Sampling* arXiv 2606.23961 (title-level) |

## Evaluation studies and mechanistic analyses

| key | reference |
|---|---|
| yuan2024whatmust | J. Yuan, H. Liu, S. Zhong et al. *KV Cache Compression, But What Must We Give in Return?* Findings of EMNLP 2024. arXiv 2407.01527 |
| kvfundabench2025 | X. Liu et al. *Can LLMs Maintain Fundamental Abilities under KV Cache Compression?* †. arXiv 2502.01941 |
| pitfalls2026 | A. Chen, R. Geh, A. Grover, G. Van den Broeck, D. M. Israel. *The Pitfalls of KV Cache Compression.* ACL 2026. arXiv 2510.00231 |
| rethinkkv2025 | *Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving.* MLSys 2025. arXiv 2503.24000 |
| holdonto2025 | M. Liu et al. *Hold Onto That Thought: Assessing KV Cache Compression on Reasoning.* †. arXiv 2512.12008 |
| kvdiagnosis2026 | *KVDiagnosis: A Diagnostic Benchmark for KV-Cache Compression in Long-Context LMs.* 2026. arXiv 2608.09412 (released CSVs) |
| queryvis2026 | *How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit.* 2026 †. arXiv 2607.11942 |
| wu2024retrievalhead | W. Wu et al. *Retrieval Head Mechanistically Explains Long-Context Factuality.* arXiv 2404.15574 |

## Serving systems, traces and usage studies

| key | reference |
|---|---|
| vllm2023 | W. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM). SOSP 2023 ‡; prefix-caching design doc in `vllm-project/vllm` |
| sglang2024 | L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention). NeurIPS 2024 ‡ |
| mooncake2025 | R. Qin et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* FAST 2025 ‡ (traces from `kvcache-ai/Mooncake`) |
| kvcachewild2025 | J. Wang et al. *KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider.* USENIX ATC 2025. arXiv 2506.02634 (traces from `alibaba-edu/qwen-bailian-usagetraces-anon`) |
| splitwise2024 | P. Patel et al. *Splitwise.* ISCA 2024 ‡ (Azure LLM inference trace 2023) |
| dynamollm2025 | J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, E. Choukse. *DynamoLLM.* HPCA 2025. arXiv 2408.00741 (Azure trace 2024) |
| burstgpt2025 | Y. Wang et al. *BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems.* KDD 2025. arXiv 2401.17644 |
| servegen2025 | *ServeGen: Workload Characterization of LLM Serving in Production* (Alibaba). arXiv 2505.09999 † |
| tracelab2026 | *TraceLab* (Claude Code / Codex traces, UW SyFI). arXiv 2606.30560 † |
| cachedattention2024 / pensieve2025 / marconi2025 / cacheblend2025 / infercept2024 / kvflow2025 / continuum2025 | cross-request KV systems as summarized in [literature §4.3](literature.md#43-cross-request-kv-management-and-eviction) ‡/† (details in `notes/industry_usage_traces_and_cross_request_eviction.md`) |
| clio2024 | A. Tamkin et al. *Clio: Privacy-Preserving Insights into Real-World AI Use.* arXiv 2412.13678 |
| aei2025 | Anthropic Economic Index reports (Sep 2025, Jan/Mar/Jun 2026), anthropic.com/research |
| chatgptusage2025 | A. Chatterji et al. *How People Use ChatGPT.* NBER Working Paper w34255, 2025 † |
| openrouter2025 | OpenRouter & a16z. *State of AI: An Empirical 100 Trillion Token Study.* 2025 † |
| wildchat2024 | W. Zhao et al. *WildChat.* ICLR 2024. arXiv 2405.01470 |
| lmsys2024 | L. Zheng et al. *LMSYS-Chat-1M.* ICLR 2024. arXiv 2309.11998 |

## Classical caching

| key | reference |
|---|---|
| mattson1970 | R. L. Mattson, J. Gecsei, D. R. Slutz, I. L. Traiger. *Evaluation Techniques for Storage Hierarchies.* IBM Systems Journal 9(2), 1970 |
| belady1966 | L. A. Belady. *A Study of Replacement Algorithms for a Virtual-Storage Computer.* IBM Systems Journal 5(2), 1966 |
| denning1968 | P. J. Denning. *The Working Set Model for Program Behavior.* Communications of the ACM 11(5), 1968 |
