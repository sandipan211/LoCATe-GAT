#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf5/' --split_index=5

python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0/' --split_index=0

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf1_retry1/' --split_index=1

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf2_retry1/' --split_index=2

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf3/' --split_index=3

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf4_retry1/' --split_index=4

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf6/' --split_index=6

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf7/' --split_index=7

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf8/' --split_index=8

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf9/' --split_index=9


# For testing GAT visualizations
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf3_copy/' --split_index=3