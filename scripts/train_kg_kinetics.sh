#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=kinetics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics0/' --split_index=0 --batch_size=32

# python3 main_kg.py --mode=kg --dataset=kinetics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics1/' --split_index=1 --batch_size=32

python3 main_kg.py --mode=kg --dataset=kinetics --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics2/' --split_index=2 --batch_size=32
