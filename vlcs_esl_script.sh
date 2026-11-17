#!/bin/bash

# Experiment configuration
BASE_COMMAND="python trainer.py -c 10 -bs 64 -tbs 256 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset VLCS_ESL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb"
OUTPUT_DIR="./VLCS_New/ESL"
SEEDS=(43)
OOD_MODES=(0 1 2 3 4)
# OOD_MODES=(2 3 4)

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through OOD modes and seeds
for OOD in "${OOD_MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_FILE="${OUTPUT_DIR}/vlcs_esl_dirch0.3_${OOD}_${SEED}.txt"
        START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

        echo "Starting experiment for mode $OOD with seed $SEED at $START_TIME"
        EXECUTED_COMMAND="$BASE_COMMAND --ood $OOD --exp_name vlcs_esl_dirch0.3_${OOD}_${SEED} --seed $SEED"

        # Log the executed command and run the experiment
        echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
        nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
        wait

        END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "Finished experiment for mode $OOD with seed $SEED at $END_TIME"
    done
    echo "Mode $OOD experiments completed!"
done

echo "All ESL experiments completed!"