#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb0/' --split_index=0

# python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb1/' --split_index=1

# python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb2/' --split_index=2

# python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb3/' --split_index=3

python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb5/' --split_index=5

python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb7/' --split_index=7
