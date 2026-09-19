#!/usr/bin/env bash
# EXP-JEV-3 driver (run on gpu02 inside ~/AGM-jev). Writes only under evaluation/results/jev_lme2.
# Stage 1: `base` for all abilities in parallel (also builds the shared org-chart/vector caches).
# Stage 2: every other config, abilities in parallel. Stage 3: LLM-judge scoring.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
M=${LME_MODEL:-accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b}
JUDGE=${LME_JUDGE:-accounts/fireworks/models/gpt-oss-120b}
PY=${PY:-$HOME/miniconda3/envs/agm/bin/python}
OUT=evaluation/results/jev_lme2; mkdir -p "$OUT"
# gpu01's Ollama embedding server was unresponsive (2026-09-19): embed through Fireworks instead.
export EMBEDDING_BASE_URL=https://api.fireworks.ai/inference/v1 EMBEDDING_API_KEY=$FIREWORKS_API_KEY \
       EMBEDDING_MODEL=accounts/fireworks/models/qwen3-embedding-8b
# Server is compute-only: never touch its vLLM/Ollama. Default LLM route -> Fireworks as well.
export LLM_BACKEND=vllm VLLM_BASE_URL=https://api.fireworks.ai/inference/v1 VLLM_API_KEY=$FIREWORKS_API_KEY
export PYTHONPATH=. LLM_REASONING_EFFORT=none LLM_DEFAULT_MODEL=$M JEV_CACHE_DIR=$OUT/jev_cache
ABILITIES="knowledge-update multi-session temporal-reasoning single-session-user single-session-assistant single-session-preference"
CONFIGS=${LME_CONFIGS:-"cascade cascade_router cascade_full cascade_gate"}
run() { # config
  for A in $ABILITIES; do
    JEV_USAGE_LOG=$OUT/jev_usage_$1.jsonl LLM_USAGE_LOG=$OUT/llm_usage_$1.jsonl \
      $PY -u scripts/jev/lme_requery.py --config "$1" --ability "$A" --model "$M" --resume > "$OUT/run_${A}__$1.log" 2>&1 &
  done; wait
}
echo "[ALL] base $(date)"; run base
for C in $CONFIGS; do echo "[ALL] $C $(date)"; run "$C"; done
echo "[ALL] scoring $(date)"
for C in base $CONFIGS; do for A in $ABILITIES; do
  LLM_REASONING_EFFORT= $PY evaluation/LongMemEval/score_longmemeval.py --cache-dir "$OUT/${A}__$C" \
    --output "$OUT/scores_${A}__$C.json" --judge --judge-model "$JUDGE" > "$OUT/score_${A}__$C.log" 2>&1 &
done; wait; done
echo "LME_ALL_DONE $(date)"
