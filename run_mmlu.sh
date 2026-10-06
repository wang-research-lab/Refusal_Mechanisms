#!/usr/bin/env bash
# MMLU 0-shot — 57-subject multiple-choice knowledge benchmark (~14k questions).
# Single-turn, no tools, no sandbox. Capability sanity axis: is base-model
# knowledge preserved after baking the refusal ablation into the weights?
#
# Reads the served model from LOCAL_BASE_URL / LOCAL_MODEL_NAME (exported by
# serve_and_eval.sh) and scores with inspect_ai's inspect_evals/mmlu_0_shot.
#
# Examples:
#   ./run_mmlu.sh --limit 500              # smoke test
#   ./run_mmlu.sh --max-connections 100    # full ~14k, 100-way parallel

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
LOG_DIR=logs/mmlu_${LOCAL_MODEL_NAME}

mkdir -p "$LOG_DIR"

# Optional chat-template kwargs (JSON object) forwarded to vLLM via the OpenAI
# extra_body. Used to switch off Qwen3 thinking (which otherwise ~5x's runtime:
# ~2h vs ~25m for one MMLU):
#   CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' ./run_mmlu.sh ...
# inspect's --generate-config only accepts a *file*, so we materialise one in the
# log dir. config.extra_body is merged straight into the vLLM chat request, where
# the Qwen3 template reads chat_template_kwargs.enable_thinking.
GC_FLAGS=()
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
  GC_FILE="$LOG_DIR/_generate_config.json"
  printf '{"extra_body": {"chat_template_kwargs": %s}}\n' "$CHAT_TEMPLATE_KWARGS" > "$GC_FILE"
  GC_FLAGS=(--generate-config "$GC_FILE")
fi

# cot=true so reasoning-format models get an uncapped output budget. With the
# default cot=false (max 16 tokens), thinking models burn the budget on internal
# reasoning and emit zero answer letters → all samples score Invalid → 0%.
# cot=true keeps the comparison apples-to-apples across model families.
#
# --max-tokens caps each generation. cot=true otherwise leaves the budget
# uncapped, so a prompt the model never stops on runs to the *context* ceiling
# — with MAX_MODEL_LEN=auto that's 128k for Mistral (~75 min/request at ~28
# tok/s), which stalls the run on a handful of stragglers. A real MMLU CoT
# answer is a few hundred tokens, so 2048 only ever truncates runaways (already
# Invalid for lacking an ANSWER line). Override via MAX_TOKENS.
exec "$INSPECT_BIN" eval inspect_evals/mmlu_0_shot \
    --model "$MODEL" \
    --log-dir "$LOG_DIR" \
    --max-connections "${MMLU_MAX_CONN:-20}" \
    --max-tokens "${MAX_TOKENS:-2048}" \
    -T cot=true \
    "${GC_FLAGS[@]}" \
    "$@"
