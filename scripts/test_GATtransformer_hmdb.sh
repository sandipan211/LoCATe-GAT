#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb8_retry1/' --split_index=8

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb4/' --split_index=4

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb9_retry1/' --split_index=9

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb0/' --split_index=0

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb1/' --split_index=1

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb2/' --split_index=2

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb3/' --split_index=3

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb5/' --split_index=5

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb7/' --split_index=7

# python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb6/' --split_index=6