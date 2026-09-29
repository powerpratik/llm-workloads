#!/usr/bin/env bash
# Reproducibly fetch every public dataset / trace used by the analysis scripts.
# Everything comes from GitHub (raw / LFS media / shallow sparse clones) or the public
# SWE-bench S3 bucket, so no Hugging Face access is required.
#
#   LLMW_DATA=/path/to/data bash analysis/fetch_data.sh      (default: ./data)
set -euo pipefail
DATA=${LLMW_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
mkdir -p "$DATA"; DATA=$(cd "$DATA" && pwd)   # absolute, so a relative LLMW_DATA works too
RAW=$DATA/raw; TOK=$DATA/tok
R=https://raw.githubusercontent.com; M=https://media.githubusercontent.com/media
mkdir -p "$RAW" "$TOK"; cd "$RAW"

get() { mkdir -p "$(dirname "$1")"; [ -s "$1" ] || curl -sSL --retry 3 --max-time 900 -o "$1" "$2"; echo "$(stat -c %s "$1") $1"; }
sparse() { # name url path...
  local name=$1 url=$2; shift 2
  [ -d "repos/$name" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --sparse "$url" "repos/$name"
  (cd "repos/$name" && git sparse-checkout set --no-cone "$@")
}

# --- tokenizers: llama.cpp vocab-only GGUF + reference test vectors -------------------
for f in ggml-vocab-llama-bpe.gguf ggml-vocab-qwen2.gguf; do
  for ext in "" .inp .out; do get "$TOK/$f$ext" "$R/ggml-org/llama.cpp/master/models/$f$ext"; done
done

# --- production serving traces ----------------------------------------------------------
for f in qwen_traceA_blksz_16 qwen_traceB_blksz_16 qwen_thinking_blksz_16 qwen_coder_blksz_16; do
  get traces/qwen_bailian/$f.jsonl $M/alibaba-edu/qwen-bailian-usagetraces-anon/main/$f.jsonl   # Git LFS objects
done
get traces/qwen_bailian/qa-context-growth-pattern.md $R/alibaba-edu/qwen-bailian-usagetraces-anon/main/docs/qa-context-growth-pattern.md
for f in conversation_trace toolagent_trace synthetic_trace; do
  get traces/mooncake/$f.jsonl $R/kvcache-ai/Mooncake/main/FAST25-release/traces/$f.jsonl
done
get traces/azure2023/AzureLLMInferenceTrace_code.csv $R/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv
get traces/azure2023/AzureLLMInferenceTrace_conv.csv $R/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
get traces/burstgpt/BurstGPT_1.csv $R/HPMLL/BurstGPT/main/data/BurstGPT_1.csv

# --- benchmark datasets -------------------------------------------------------------------
get datasets/humaneval/HumanEval.jsonl.gz $R/openai/human-eval/master/data/HumanEval.jsonl.gz
get datasets/mbpp/mbpp.jsonl $R/google-research/google-research/master/mbpp/mbpp.jsonl
get datasets/ifeval/input_data.jsonl $R/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl
get datasets/mtbench/question.jsonl $R/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
get datasets/mtbench/reference_gpt4.jsonl $R/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl
get datasets/arenahard/v0.1_question.jsonl $R/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl
get datasets/arenahard/v2.0_question.jsonl $R/lm-sys/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl
get datasets/locomo/locomo10.json $R/snap-research/locomo/main/data/locomo10.json
get datasets/longbench_write/longbench_write.jsonl $R/THUDM/LongWriter/main/evaluation/longbench_write.jsonl
get datasets/math500/test.jsonl $M/openai/prm800k/main/prm800k/math_splits/test.jsonl
get datasets/mtbench101/mtbench101.jsonl $R/mtbench101/mt-bench-101/main/data/subjective/mtbench101.jsonl
get datasets/gsm8k/test.jsonl $R/openai/grade-school-math/master/grade_school_math/data/test.jsonl
get datasets/gsm8k_cot/gsm8k-cot.yaml $R/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
get datasets/quality/QuALITY.v1.0.1.zip $R/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.zip
(cd datasets/quality && unzip -q -o QuALITY.v1.0.1.zip)
get datasets/cceval/crosscodeeval_data.tar.xz $R/amazon-science/cceval/main/data/crosscodeeval_data.tar.xz
(cd datasets/cceval && tar -xJf crosscodeeval_data.tar.xz)

sparse bbh https://github.com/suzgunmirac/BIG-Bench-Hard.git '/bbh/*' '/cot-prompts/*'
sparse niah https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git '/needlehaystack/PaulGrahamEssays/*'
sparse leval https://github.com/OpenLMLab/LEval.git '/LEval-data/*'
sparse taubench https://github.com/sierra-research/tau-bench.git '/historical_trajectories/*' '/tau_bench/envs/*'
sparse qwenmath https://github.com/QwenLM/Qwen2.5-Math.git '/evaluation/data/*'
sparse arenahard https://github.com/lm-sys/arena-hard-auto.git '/data/arena-hard-v2.0/model_answer/*' '/data/arena-hard-v0.1/model_answer/*'
sparse alpaca_eval https://github.com/tatsu-lab/alpaca_eval.git \
  '/results/Meta-Llama-3-8B-Instruct/model_outputs.json' '/results/Meta-Llama-3.1-8B-Instruct-Turbo/model_outputs.json' \
  '/results/Meta-Llama-3.1-70B-Instruct-Turbo/model_outputs.json' '/results/gpt-4o-2024-05-13/model_outputs.json' \
  '/results/claude-3-5-sonnet-20240620/model_outputs.json' '/results/Qwen2-72B-Instruct/model_outputs.json' \
  '/results/gpt4_1106_preview/model_outputs.json'

# --- SWE-bench Verified agent trajectories (mini-SWE-agent, public S3 bucket) ---------------
S3=https://swe-bench-submissions.s3.amazonaws.com
for sub in 20250726_mini-v1.0.0_claude-sonnet-4-20250514 20250802_mini-v1.0.0_qwen3-coder-480b-a35b-instruct 20250807_mini-v1.7.0_gpt-5; do
  mkdir -p "swebench_trajs/$sub"
  python3 - "$sub" > "swebench_trajs/keys_$sub.txt" <<'EOF'
import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
B = "https://swe-bench-submissions.s3.amazonaws.com/"; ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
prefix, token = f"bash-only/{sys.argv[1]}/trajs/", None
while True:
    q = {"list-type": "2", "prefix": prefix, "max-keys": "1000", **({"continuation-token": token} if token else {})}
    root = ET.fromstring(urllib.request.urlopen(B + "?" + urllib.parse.urlencode(q), timeout=60).read())
    for c in root.findall("s:Contents", ns):
        k = c.find("s:Key", ns).text
        if k.endswith(".traj.json"): print(k)
    t = root.find("s:NextContinuationToken", ns)
    if t is None: break
    token = t.text
EOF
  xargs -P 12 -I{} sh -c 'f=$(basename "{}"); [ -s "swebench_trajs/'"$sub"'/$f" ] || curl -s --retry 3 -o "swebench_trajs/'"$sub"'/$f" "'"$S3"'/{}"' < "swebench_trajs/keys_$sub.txt"
  echo "$sub: $(ls "swebench_trajs/$sub" | wc -l) trajectories"
done
echo "done -> $DATA ; next: python analysis/build_tokenizers.py"
