#!/bin/bash

# Usage:
# ./run_masks.sh "model_path" pool [mode]
#
# Modes:
# pool      → original behavior (default)
# all_pairs → use GPU pairs across all 0–7

MODEL_PATH=${1:-"microsoft/Phi-4-mini-instruct"}
POOL=${2:-1}
MODE=${3:-pool}

# Budget quantiles (top magnitude runs)
BUDGETS=(0.25 0.5 0.75 1)

########################################
# Define GPU assignments
########################################

if [[ $MODE == "pool" ]]; then
    
    if [[ $POOL == 1 ]]; then
        GPU_LIST=(0 1 2 3)
    elif [[ $POOL == 2 ]]; then
        GPU_LIST=(4 5 6 7)
    else
        echo "Invalid pool. Use 1 or 2."
        exit 1
    fi

    USE_ROUND_ROBIN=true

elif [[ $MODE == "all_pairs" ]]; then
    
    GPU_ASSIGNMENTS=("0,1" "2,3" "4,5" "6,7")
    USE_ROUND_ROBIN=false

else
    echo "Invalid mode. Use 'pool' or 'all_pairs'."
    exit 1
fi

########################################
# Phase 1: Original budget_quantile runs
########################################

for i in "${!BUDGETS[@]}"; do
    BUDGET=${BUDGETS[$i]}

    if [[ $USE_ROUND_ROBIN == true ]]; then
        GPU=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
        GPU_SET=$GPU
    else
        GPU_SET=${GPU_ASSIGNMENTS[$i]}
    fi

    echo "Running TOP magnitude on GPUs $GPU_SET with budget $BUDGET..."

    CMD="CUDA_VISIBLE_DEVICES=$GPU_SET python3 -m pipeline.run_random_masks \
        --model_path \"$MODEL_PATH\" \
        --use_existing \
        --budget_quantile $BUDGET"

    if [[ $BUDGET == 1 ]]; then
        CMD="$CMD --num_random_trials 1"
    fi

    eval "$CMD &"
done

wait
echo "Top magnitude runs completed."

########################################
# Phase 2: Bottom magnitude quadrant runs
# Split [0.25, 0.75] into 4 equal parts
########################################

LOW_START=0
HIGH_END=1
NUM_SPLITS=4
INTERVAL=$(echo "($HIGH_END - $LOW_START) / $NUM_SPLITS" | bc -l)

for i in $(seq 0 3); do

    LOW=$(echo "$LOW_START + $i * $INTERVAL" | bc -l)
    HIGH=$(echo "$LOW_START + ($i + 1) * $INTERVAL" | bc -l)

    if [[ $USE_ROUND_ROBIN == true ]]; then
        GPU=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
        GPU_SET=$GPU
    else
        GPU_SET=${GPU_ASSIGNMENTS[$i]}
    fi

    echo "Running BOTTOM magnitude on GPUs $GPU_SET with quantile range [$LOW, $HIGH]..."

    CMD="CUDA_VISIBLE_DEVICES=$GPU_SET python3 -m pipeline.run_random_masks \
        --model_path \"$MODEL_PATH\" \
        --use_existing \
        --run_bottom_mag \
        --low_quantile $LOW \
        --high_quantile $HIGH"

    eval "$CMD &"
done

wait
echo "Bottom magnitude quadrant runs completed."
echo "All runs completed."