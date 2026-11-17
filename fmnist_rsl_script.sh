#!/bin/bash

COMMAND0="python trainer.py -c 10 -bs 64 -tbs 64 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset FMNIST_CL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb --ood 0"
SEEDS=(42)
OUTPUT_DIR="./FMNIST/RSL"
mkdir -p "$OUTPUT_DIR"
for SEED in "${SEEDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/fmnist_rsl_dirch0.3_0_${SEED}.txt"
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Starting experiment of mode 0 with seed $SEED at $START_TIME"
    EXECUTED_COMMAND="$COMMAND0 --exp_name fmnist_rsl_dirch0.3_0_${SEED} --seed $SEED"
    echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
    nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
    wait
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Finished experiment of mode 0 with seed $SEED at $END_TIME"
done
echo "RSL mode 0 experiments completed!"


SEEDS=(42)
COMMAND1="python trainer.py -c 10 -bs 64 -tbs 64 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset FMNIST_CL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb --ood 1"
for SEED in "${SEEDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/fmnist_rsl_dirch0.3_1_${SEED}.txt"
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Starting experiment of mode 1 with seed $SEED at $START_TIME"
    EXECUTED_COMMAND="$COMMAND1 --exp_name fmnist_rsl_dirch0.3_1_${SEED} --seed $SEED"
    echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
    nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
    wait
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Finished experiment of mode 1 with seed $SEED at $END_TIME"
done
echo "RSL mode 1 experiments completed!"


SEEDS=(42)
COMMAND2="python trainer.py -c 10 -bs 64 -tbs 64 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset FMNIST_CL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb --ood 2"
for SEED in "${SEEDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/fmnist_rsl_dirch0.3_2_${SEED}.txt"
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Starting experiment of mode 2 with seed $SEED at $START_TIME"
    EXECUTED_COMMAND="$COMMAND2 --exp_name fmnist_rsl_dirch0.3_2_${SEED} --seed $SEED"
    echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
    nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
    wait
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Finished experiment of mode 2 with seed $SEED at $END_TIME"
done
echo "RSL mode 2 experiments completed!"


SEEDS=(42)
COMMAND3="python trainer.py -c 10 -bs 64 -tbs 64 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset FMNIST_CL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb --ood 3"
for SEED in "${SEEDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/fmnist_rsl_dirch0.3_3_${SEED}.txt"
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Starting experiment of mode 3 with seed $SEED at $START_TIME"
    EXECUTED_COMMAND="$COMMAND3 --exp_name fmnist_rsl_dirch0.3_3_${SEED} --seed $SEED"
    echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
    nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
    wait
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Finished experiment of mode 3 with seed $SEED at $END_TIME"
done
echo "RSL mode 3 experiments completed!"


SEEDS=(42)
COMMAND4="python trainer.py -c 10 -bs 64 -tbs 64 -n 60 --client_lr 1e-3 --server_lr 1e-3 --dataset FMNIST_CL_IP --model resnet18 --split 1 -kv --kv_refresh_rate 0 --kv_factor 1 --wandb --ood 4"
for SEED in "${SEEDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/fmnist_rsl_dirch0.3_4_${SEED}.txt"
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Starting experiment of mode 4 with seed $SEED at $START_TIME"
    EXECUTED_COMMAND="$COMMAND4 --exp_name fmnist_rsl_dirch0.3_4_${SEED} --seed $SEED"
    echo "Executing: $EXECUTED_COMMAND" > "$OUTPUT_FILE"
    nohup $EXECUTED_COMMAND >> "$OUTPUT_FILE" 2>&1 &
    wait
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Finished experiment of mode 4 with seed $SEED at $END_TIME"
done
echo "RSL mode 4 experiments completed!"