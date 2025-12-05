#!/bin/bash

set -euo pipefail

# 请根据实际路径导出以下环境变量，也可在命令行附加覆盖
: "${BASE_MODEL_PATH:?请先导出 BASE_MODEL_PATH，例如 export BASE_MODEL_PATH=/path/to/base}" 
: "${EAGLE_MODEL_PATH:?请先导出 EAGLE_MODEL_PATH，例如 export EAGLE_MODEL_PATH=/path/to/eagle}" 

MODEL_ID=${MODEL_ID:-specexit-demo}
BENCH_NAME=${BENCH_NAME:-gsm8k}
OUTPUT_DIR=${OUTPUT_DIR:-specexit_outputs}
TEMPERATURE=${TEMPERATURE:-1.0}
TOTAL_TOKEN=${TOTAL_TOKEN:-60}
DEPTH=${DEPTH:-5}
TOP_K=${TOP_K:-10}
TOP_P=${TOP_P:-1.0}
MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-1024}
NUM_GPUS_PER_MODEL=${NUM_GPUS_PER_MODEL:-1}
NUM_GPUS_TOTAL=${NUM_GPUS_TOTAL:-1}
EARLY_STOP_METHOD=${EARLY_STOP_METHOD:-confidence_progress_remain}
USE_CONFUSION_PRUNE=${USE_CONFUSION_PRUNE:-0}
CONFUSION_WINDOW=${CONFUSION_WINDOW:-16}
CONFUSION_THRESHOLD=${CONFUSION_THRESHOLD:-25.0}

CONFUSION_FLAG=()
if [[ "$USE_CONFUSION_PRUNE" == "1" ]]; then
	CONFUSION_FLAG+=(--use-confusion-prune)
fi

python3 tools/run_specexit.py \
	--base-model-path "$BASE_MODEL_PATH" \
	--eagle-model-path "$EAGLE_MODEL_PATH" \
	--model-id "$MODEL_ID" \
	--bench-name "$BENCH_NAME" \
	--output-dir "$OUTPUT_DIR" \
	--temperature "$TEMPERATURE" \
	--total-token "$TOTAL_TOKEN" \
	--depth "$DEPTH" \
	--top-k "$TOP_K" \
	--top-p "$TOP_P" \
	--max-new-token "$MAX_NEW_TOKEN" \
	--num-gpus-per-model "$NUM_GPUS_PER_MODEL" \
	--num-gpus-total "$NUM_GPUS_TOTAL" \
	--experiments baseline,speculative,specexit \
	--specexit-stop-method "$EARLY_STOP_METHOD" \
	--confusion-window "$CONFUSION_WINDOW" \
	--confusion-threshold "$CONFUSION_THRESHOLD" \
	"${CONFUSION_FLAG[@]}" \
	"$@"
