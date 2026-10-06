#!/usr/bin/env bash
# GSM8k — grade-school math word problems (1,319 test questions), few-shot CoT.
# Single-turn, no tools, no sandbox. Capability sanity axis: is multi-step
# reasoning preserved after baking the refusal ablation into the weights?
#
# Reads the served model from LOCAL_BASE_URL / LOCAL_MODEL_NAME (exported by
# serve_and_eval.sh) and scores with inspect_ai's inspect_evals/gsm8k.
#
# Examples:
#   ./run_gsm8k.sh --limit 200             # smoke test
#   ./run_gsm8k.sh --max-connections 100   # full 1,319, 100-way parallel
#   GSM8K_FEWSHOT=5 ./run_gsm8k.sh         # override few-shot count (default 10)

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f "$HERE/.env" ]; then
  set -a; source "$HERE/.env"; set +a
fi

# Prefer a project-local venv's inspect, else fall back to inspect on PATH.
INSPECT_BIN="${INSPECT_BIN:-}"
if [ -z "$INSPECT_BIN" ]; then
  if [ -x "$HERE/venv/bin/inspect" ]; then INSPECT_BIN="$HERE/venv/bin/inspect"; else INSPECT_BIN="inspect"; fi
fi

export LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8000/v1}"
export LOCAL_API_KEY="${LOCAL_API_KEY:-EMPTY}"

LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-local-model}"
MODEL="openai-api/local/${LOCAL_MODEL_NAME}"
LOG_DIR=logs/gsm8k_${LOCAL_MODEL_NAME}

mkdir -p "$LOG_DIR"

# Optional chat-template kwargs (JSON object) forwarded to vLLM via the OpenAI
# extra_body. Used to switch off Qwen3 thinking (which otherwise ~5x's runtime):
#   CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' ./run_gsm8k.sh ...
# inspect's --generate-config only accepts a *file*, so we materialise one in the
# log dir. config.extra_body is merged straight into the vLLM chat request, where
# the Qwen3 template reads chat_template_kwargs.enable_thinking.
GC_FLAGS=()
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
  GC_FILE="$LOG_DIR/_generate_config.json"
  printf '{"extra_body": {"chat_template_kwargs": %s}}\n' "$CHAT_TEMPLATE_KWARGS" > "$GC_FILE"
  GC_FLAGS=(--generate-config "$GC_FILE")
fi

# --max-tokens caps each generation so a non-terminating prompt can't run to the
# model's context ceiling (128k for Mistral with MAX_MODEL_LEN=auto → ~75 min on
# one request, stalling the run). GSM8k CoT answers finish in a few hundred
# tokens; 2048 only truncates runaways. Override via MAX_TOKENS.
#
# fewshot defaults to inspect_evals' 10-shot; override via GSM8K_FEWSHOT.
exec "$INSPECT_BIN" eval inspect_evals/gsm8k \
    --model "$MODEL" \
    --log-dir "$LOG_DIR" \
    --max-connections "${GSM8K_MAX_CONN:-20}" \
    --max-tokens "${MAX_TOKENS:-2048}" \
    -T fewshot="${GSM8K_FEWSHOT:-10}" \
    "${GC_FLAGS[@]}" \
    "$@"
