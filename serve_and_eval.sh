#!/usr/bin/env bash
# Serve ONE vLLM variant in a backgrounded podman container, run a configured
# set of benches against it, then tear the container down. Single-container
# at-a-time (no concurrent serves within this invocation).
#
# Usage:
#   ./serve_and_eval.sh <model_dir> <served_name> <parser> <port> <gpus>
#
# Args:
#   model_dir   — either host path under $MODEL_HUB_ROOT/... (auto-
#                 translated to /models/...) or a container path /models/...
#                 OR a HF hub id (e.g. ibm-granite/granite-4.1-30b) — those
#                 get mounted via HF_HOME read-only like the base evals.
#   served_name — vLLM --served-model-name (also the local model alias).
#   parser      — vLLM --tool-call-parser (granite4 / openai / glm47).
#   port        — host port (container always serves on 8000).
#   gpus        — comma-separated GPU indices. Length>1 → tensor-parallel.
#
# Env knobs:
#   BENCHES         — space-separated list to run (default "agentharm asb fortress").
#                     Supported: agentharm asb fortress agentdojo codeipi bfcl mmlu gsm8k labbench
#                     Capability sanity for the orthogonalized weights: BENCHES="mmlu gsm8k".
#   READY_TIMEOUT_S — seconds to wait for vLLM /v1/models (default 900).
#   MAX_MODEL_LEN   — vLLM --max-model-len (default 32768). Set to "auto" (or
#                     empty) to let vLLM derive it from the model config — needed
#                     for short-context models (e.g. gemma-2's 8192) where 32768
#                     is a fatal error.
#   REASONING_PARSER — if set, passed as vLLM --reasoning-parser. Required for
#                      thinking models so <think>...</think> blocks are stripped
#                      from the assistant content (e.g. qwen3, deepseek_r1).
#   MAX_NUM_SEQS     — if set, passed as vLLM --max-num-seqs. Lower for hybrid
#                      attention models (Qwen3.5/3.6 mixed attention) where the
#                      Mamba cache pool is tight; typical safe value: 512.
#   GPU_MEM_UTIL     — if set, passed as vLLM --gpu-memory-utilization. Default
#                      vLLM behaviour (0.9) is fine for most models.
#   MODEL_IMPL       — if set, passed as vLLM --model-impl (e.g. "transformers"
#                      to bypass vLLM's native arch registry when it lags
#                      transformers config-class renames).
#   CHAT_TEMPLATE    — if set, host path to a .jinja chat template, mounted into
#                      the container and passed as vLLM --chat-template. Use for
#                      gemma-2 (stock template raises on the system role that
#                      GSM8k's fewshot solver emits): CHAT_TEMPLATE=chat_templates/gemma_system.jinja
#   VLLM_IMAGE       — container image tag (default vllm/vllm-openai:latest).
#                      For Blackwell GPUs you may need
#                      docker.io/vllm/vllm-openai:cu130-nightly which ships
#                      newer arch support (Qwen3_5ForConditionalGeneration etc).
#
# Examples:
#   ./serve_and_eval.sh $MODEL_HUB_ROOT/granite-4.1-30b-ortho-wrap-layer31 \
#                       granite-wrap granite4 8006 2
#   BENCHES="asb fortress" ./serve_and_eval.sh \
#       $MODEL_HUB_ROOT/GLM-4.7-Flash-ortho-agent-layer22 \
#       glm-agent glm47 8008 5,6

set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <model_dir> <served_name> <parser> <port> <gpus>" >&2
    exit 1
fi

MODEL_DIR="$1"
SERVED="$2"
PARSER="$3"
PORT="$4"
GPUS="$5"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; source "$HERE/.env"; set +a; }

BENCHES="${BENCHES:-agentharm asb fortress}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:latest}"
# Root of the local HF hub cache holding the baked -ortho-* checkpoints.
# Override with MODEL_HUB_ROOT=... to run outside our machine.
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${HF_HUB_CACHE:-/data/vince/.hf_home/hub}}"
HF_HOME_ROOT="${HF_HOME_ROOT:-${HF_HOME:-/data/vince/.hf_home}}"
CTR_NAME="${SERVED}-vllm-$$"

# Translate $MODEL_HUB_ROOT/<X> → /models/<X> (matches our mount).
# Anything else (HF hub id like ibm-granite/granite-4.1-30b, or already-/models/
# path) passes through.
case "$MODEL_DIR" in
    "$MODEL_HUB_ROOT"/*) IN_CTR_MODEL="/models/${MODEL_DIR#$MODEL_HUB_ROOT/}" ;;
    *)                          IN_CTR_MODEL="$MODEL_DIR" ;;
esac

# Decide which volume to mount: -ortho-* models live in hub; baselines come
# from HF cache and need HF_HOME.
# HF cache mount cannot be :ro — huggingface_hub takes a filelock on every
# load (.locks/<repo>/<sha>.lock) even for fully-cached models, so a read-only
# mount blows up with `OSError: [Errno 30] Read-only file system` on first
# run after a remote revision change.
if [[ "$IN_CTR_MODEL" == /models/* ]]; then
    MOUNT_FLAGS=(-v "$MODEL_HUB_ROOT":/models:ro)
else
    MOUNT_FLAGS=(-v "$HF_HOME_ROOT":/hf_home -e HF_HOME=/hf_home)
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
DEV_FLAGS=()
for g in "${GPU_ARR[@]}"; do
    DEV_FLAGS+=(--device "nvidia.com/gpu=$g")
done
TP_SIZE="${#GPU_ARR[@]}"
TP_FLAGS=()
[ "$TP_SIZE" -gt 1 ] && TP_FLAGS=(--tensor-parallel-size "$TP_SIZE")

REASONING_FLAGS=()
[ -n "${REASONING_PARSER:-}" ] && REASONING_FLAGS=(--reasoning-parser "$REASONING_PARSER")

# --max-model-len: pass through unless set to "auto"/empty, in which case we let
# vLLM derive it from the model config. Forcing a value larger than the model's
# max_position_embeddings (e.g. 32768 vs gemma-2's 8192) is a fatal vLLM error.
MML_FLAGS=()
case "${MAX_MODEL_LEN,,}" in
    ""|auto) ;;  # omit → vLLM derives from config
    *)       MML_FLAGS=(--max-model-len "$MAX_MODEL_LEN") ;;
esac

EXTRA_FLAGS=()
[ -n "${MAX_NUM_SEQS:-}" ]  && EXTRA_FLAGS+=(--max-num-seqs "$MAX_NUM_SEQS")
[ -n "${GPU_MEM_UTIL:-}" ]  && EXTRA_FLAGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
[ -n "${MODEL_IMPL:-}" ]    && EXTRA_FLAGS+=(--model-impl "$MODEL_IMPL")

# --chat-template override (host .jinja path). Mounted read-only into the
# container and passed to vLLM. Needed for gemma-2, whose stock template raises
# on the system role that inspect's GSM8k fewshot solver emits; the override
# folds system into the first user turn (identical output when no system msg).
CT_MOUNT=()
CT_FLAGS=()
if [ -n "${CHAT_TEMPLATE:-}" ]; then
    if [ ! -f "$CHAT_TEMPLATE" ]; then
        echo "[serve_and_eval] CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2
        exit 1
    fi
    CT_HOST="$(cd "$(dirname "$CHAT_TEMPLATE")" && pwd)/$(basename "$CHAT_TEMPLATE")"
    CT_MOUNT=(-v "$CT_HOST:/chat_template.jinja:ro")
    CT_FLAGS=(--chat-template /chat_template.jinja)
fi

cleanup() {
    local rc=$?
    echo "[serve_and_eval] cleanup — stopping $CTR_NAME (exit rc=$rc)"
    podman stop -t 30 "$CTR_NAME" >/dev/null 2>&1 || true
    podman rm   -f    "$CTR_NAME" >/dev/null 2>&1 || true
    exit $rc
}
trap cleanup EXIT INT TERM

VLLM_LOG="/tmp/${CTR_NAME}.log"
echo "[serve_and_eval] launch  name=$CTR_NAME  gpus=$GPUS  port=$PORT  parser=$PARSER  model=$IN_CTR_MODEL"
echo "[serve_and_eval] vLLM stdout/err being mirrored to $VLLM_LOG"
# No --rm: keep the container around if it dies during startup so the cleanup
# trap can still pull its logs. The trap removes it explicitly afterwards.
podman run -d --name "$CTR_NAME" \
    "${DEV_FLAGS[@]}" --security-opt=label=disable --ipc=host \
    "${MOUNT_FLAGS[@]}" "${CT_MOUNT[@]}" -p "$PORT:8000" \
    "$VLLM_IMAGE" \
    --model "$IN_CTR_MODEL" \
    --served-model-name "$SERVED" \
    "${MML_FLAGS[@]}" \
    --enable-auto-tool-choice --tool-call-parser "$PARSER" \
    "${CT_FLAGS[@]}" \
    "${REASONING_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}" \
    "${TP_FLAGS[@]}" \
    >/dev/null
# Stream container logs to a persistent file so they survive cleanup.
podman logs -f "$CTR_NAME" >"$VLLM_LOG" 2>&1 &

echo "[serve_and_eval] waiting up to ${READY_TIMEOUT_S}s for /v1/models on :$PORT …"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[serve_and_eval] vLLM ready"
        break
    fi
    if ! podman inspect "$CTR_NAME" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
        echo "[serve_and_eval] container died before /v1/models came up — logs (tail of $VLLM_LOG):"
        tail -120 "$VLLM_LOG" 2>/dev/null || podman logs "$CTR_NAME" 2>&1 | tail -120 || true
        exit 1
    fi
    sleep 5
done

if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[serve_and_eval] timeout — vLLM did not come up. last container logs (tail of $VLLM_LOG):"
    tail -120 "$VLLM_LOG" 2>/dev/null || podman logs "$CTR_NAME" 2>&1 | tail -120 || true
    exit 1
fi

export LOCAL_BASE_URL="http://localhost:$PORT/v1"
export LOCAL_API_KEY="EMPTY"
export LOCAL_MODEL_NAME="$SERVED"

for bench in $BENCHES; do
    echo "[serve_and_eval] === $bench  →  $SERVED ==="
    # Each bench script reads LOCAL_BASE_URL/LOCAL_MODEL_NAME and writes to
    # logs/<bench>_<served>/. We don't want one bench failing to kill the
    # whole run — record the rc and continue.
    set +e
    case "$bench" in
        agentharm) "$HERE/run_agentharm.sh" --max-connections 100 ;;
        asb)       ASB_WORKERS="${ASB_WORKERS:-512}" "$HERE/run_asb.sh" ;;
        fortress)  "$HERE/run_fortress.sh" --max-connections 100 ;;
        agentdojo) "$HERE/run_agentdojo.sh" -T with_sandbox_tasks=no --max-connections 100 ;;
        codeipi)   "$HERE/run_codeipi.sh" --max-connections 100 ;;
        bfcl)      "$HERE/run_bfcl.sh" --max-connections 100 ;;
        mmlu)      "$HERE/run_mmlu.sh" --max-connections 100 ;;
        gsm8k)     "$HERE/run_gsm8k.sh" --max-connections 100 ;;
        labbench)  "$HERE/run_labbench.sh" ;;
        wmdp)      "$HERE/run_wmdp.sh" ;;
        redcode_gen) REDCODE_MAX_CONN="${REDCODE_MAX_CONN:-32}" "$HERE/run_redcode_gen.sh" ;;
        *)         echo "[serve_and_eval] unknown bench: $bench" ;;
    esac
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[serve_and_eval] !! $bench exited non-zero ($rc), continuing"
    fi
done

echo "[serve_and_eval] done — trap will tear down $CTR_NAME"
