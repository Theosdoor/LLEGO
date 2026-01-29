#!/bin/bash

# Ensure environment is set up once (uv run will reuse this)
echo "Setting up environment..."
uv sync
echo ""

# DATASET_LIST=(credit-g diabetes compas heart-statlog liver breast vehicle)
DATASET_LIST=(credit-g heart-statlog liver breast vehicle)
MAX_DEPTH_LIST=("3")
SEED_LIST=("0" "1" "2" "3" "4")
exp_name='classification'

for max_depth in "${MAX_DEPTH_LIST[@]}"
do
    for dataset in "${DATASET_LIST[@]}"
    do
        for seed in "${SEED_LIST[@]}"
        do
            echo "Running experiment for dataset: $dataset, max_depth: $max_depth, seed: $seed; exp_name: $exp_name"
            uv run python exp_llego.py dataset=$dataset max_depth=$max_depth exp_name=$exp_name seed=$seed
        done
    done
done


# DATASET_LIST=(cholesterol wine wage abalone cars)
# exp_name='regression'

# for max_depth in "${MAX_DEPTH_LIST[@]}"
# do
#     for dataset in "${DATASET_LIST[@]}"
#     do
#         for seed in "${SEED_LIST[@]}"
#         do
#             echo "Running experiment for dataset: $dataset, max_depth: $max_depth, seed: $seed; exp_name: $exp_name"
#             uv run python exp_llego.py dataset=$dataset max_depth=$max_depth exp_name=$exp_name seed=$seed
#         done
#     done
# done

