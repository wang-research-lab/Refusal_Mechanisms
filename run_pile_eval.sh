#!/usr/bin/env bash
# Pile CE-loss / perplexity for all paper models, three conditions each
# (base / refusal-steered / mechanism-steered) via pipeline/eval_pile.py.
# Runtime hooks == the baked weights from save_ortho_weights.py, so no
# orthogonalized checkpoints need to exist on disk.
#
# Each model writes runs/<alias>/pile_loss.json.
#
# Env:
#   GPUS       — single GPU index for the small models (default 4).
#   N_BATCHES  — Pile batches per variant (default 64).
#   BATCH_SIZE — default 8.
#   MAX_SEQ    — max sequence length (default 512).

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

GPUS="${GPUS:-4}"
N_BATCHES="${N_BATCHES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SEQ="${MAX_SEQ:-512}"

MODELS=(
    google/gemma-2-2b-it
    Qwen/Qwen3-4B
    microsoft/Phi-4-mini-instruct
    mistralai/Mistral-Small-3.2-24B-Instruct-2506
)

for mp in "${MODELS[@]}"; do
    echo "############################################################"
    echo "# Pile eval: $mp"
    echo "############################################################"
    CUDA_VISIBLE_DEVICES="$GPUS" python -m pipeline.eval_pile \
        --model_path "$mp" \
        --n_batches "$N_BATCHES" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_length "$MAX_SEQ" || echo "[pile] !! failed for $mp, continuing"
done
echo "[pile] done — results in runs/<alias>/pile_loss.json"
