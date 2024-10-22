#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat


# python3 main_kg.py --mode=gat_lca_gzsl_test --action=gzsl_test  --dataset=olympics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics9/' --split_index=9

python3 main_kg.py --mode=gat_lca_gzsl_test --action=gzsl_test  --dataset=olympics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics2_retry1/' --split_index=2

python3 main_kg.py --mode=gat_lca_gzsl_test --action=gzsl_test  --dataset=olympics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics3_retry1/' --split_index=3