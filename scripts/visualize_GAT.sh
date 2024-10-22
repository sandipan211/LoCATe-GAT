#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# UCF splits
python3 visualize.py --dataset=ucf --semantic=clip --network=clip_transformer --model_path='../../log/log_transformer_ucf0/checkpoint_kg.pth.tar' --type=attention --split_index=0 --save_path='../../class_analysis_gray/'

# HMDB splits
python3 visualize.py --dataset=hmdb --semantic=clip --network=clip_transformer --model_path='../../log/log_transformer_hmdb2/checkpoint_kg.pth.tar' --type=attention --split_index=2 --save_path='../../class_analysis_gray/'