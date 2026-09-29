#!/usr/bin/env bash
# Re-run every measurement (M1-M6) from the raw data fetched by analysis/fetch_data.sh, then regenerate
# all tables, figures and generated docs. Every step is deterministic, so on a clean checkout
# `git status` should afterwards report no changed results or docs.
#
#   LLMW_DATA=/path/to/data bash analysis/run_all.sh      (default: ./data; ~10 min on 4 cores)
#
# Per-step logs go to $LLMW_DATA/logs/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${LLMW_DATA:-$REPO/data}
[ -d "$DATA/raw" ] || { echo "no raw data in $DATA; run analysis/fetch_data.sh first" >&2; exit 1; }
export LLMW_DATA=$(cd "$DATA" && pwd)
PY=${PYTHON:-python3}
LOG=$LLMW_DATA/logs
mkdir -p "$LOG"
t0=$SECONDS
trap 'kill $(jobs -p) 2> /dev/null || true' EXIT  # stop background replays if a step fails

run() {  # run <log name> <script> [args...]: one foreground step, stops on failure
  local name=$1 t=$SECONDS; shift
  if "$PY" "$@" > "$LOG/$name.log" 2>&1; then
    printf '  ok  %-22s %4ds\n' "$name" $((SECONDS - t))
  else
    echo "  FAILED $name; last lines of $LOG/$name.log:" >&2; tail -n 20 "$LOG/$name.log" >&2; exit 1
  fi
}

cd "$REPO/analysis"
echo "data: $LLMW_DATA"
run tokenizers build_tokenizers.py

# M6 trace replays are the slowest step: start them now and run M1-M5 meanwhile.
# The split into five processes matches the committed results (see serving_traces.py --merge).
pids=() names=()
for t in "qwen:to-C chat" "qwen:to-B API" "qwen:thinking" "qwen:coder" \
         "mooncake:conversation|mooncake:tool&agent|mooncake:synthetic"; do
  name="M6_$(echo "${t%%|*}" | tr -c 'a-zA-Z0-9\n' '_')"
  IFS='|' read -r -a args <<< "$t"
  "$PY" serving_traces.py "${args[@]}" > "$LOG/$name.log" 2>&1 &
  pids+=($!) names+=("$name")
done

run M1_length_profiles length_profiles.py
run M2_reuse_distance reuse_distance.py
run M3_M4_context_scope context_scope.py
run M5_agents agent_trajectories.py
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "  FAILED ${names[$i]}; see $LOG/${names[$i]}.log" >&2; exit 1; }
done
echo "  ok  M6 trace replays      $((SECONDS - t0))s since start"
run M6_merge serving_traces.py --merge
run literature literature_coverage.py

cd "$REPO"
run catalog taxonomy/build_catalog.py
run coverage_matrix tools/coverage.py --matrix
run benchmark_mapping tools/render_catalog.py
cd analysis
run figures make_figures.py
run tables render_tables.py

echo "done in $((SECONDS - t0))s"
if git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1; then
  changed=$(git -C "$REPO" status --porcelain -- results docs taxonomy)
  if [ -z "$changed" ]; then
    echo "all results, docs and catalog files reproduced exactly"
  else
    echo "changed files (compare with git diff):"; echo "$changed"
  fi
  # PNG bytes also depend on the matplotlib version and installed fonts, so figures are reported separately.
  figs=$(git -C "$REPO" status --porcelain -- figures)
  [ -z "$figs" ] && echo "figures identical" || echo "figures re-rendered with different bytes (expected across matplotlib/font versions)"
fi
