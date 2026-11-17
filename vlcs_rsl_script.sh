#!/bin/bash

# Base configuration for the experiments
BASE_COMMAND="python trainer.py -c 10 -bs 64 -tbs 256 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset VLCS_RSL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb"
SEEDS=(42 43 44 45 46)
OUTPUT_DIR="./VLCS_NEW_TRANSFORMS/RSL_05_seeds"
mkdir -p "$OUTPUT_DIR"

# Function to run experiments for a specific mode
run_experiment() {
    local mode=$1
    local command="${BASE_COMMAND} --ood ${mode}"
    
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_FILE="${OUTPUT_DIR}/vlcs_rsl_dirch0.3_${mode}_${SEED}.txt"
        START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "Starting experiment of mode ${mode} with seed ${SEED} at ${START_TIME}"
        
        EXECUTED_COMMAND="${command} --exp_name vlcs_rsl_dirch0.3_${mode}_${SEED} --seed ${SEED}"
        echo "Executing: ${EXECUTED_COMMAND}" > "$OUTPUT_FILE"  # Overwrite the file at the start of the experiment
        
        nohup $EXECUTED_COMMAND > "$OUTPUT_FILE" 2>&1 &  # Run in background and redirect output to the file
        
        # Wait for background processes to complete for this seed before moving on
        wait
        END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "Finished experiment of mode ${mode} with seed ${SEED} at ${END_TIME}"
    done
    
    echo "RSL mode ${mode} experiments completed!"
}

# Run experiments for modes 1 through 4
for MODE in {0..4}; do
    run_experiment "$MODE"
done