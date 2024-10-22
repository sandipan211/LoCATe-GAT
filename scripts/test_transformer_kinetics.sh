#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=test --dataset=kinetics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics0/' --split_index=0 --batch_size=32

# python3 main.py --action=test --dataset=kinetics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics1/' --split_index=1 --batch_size=32

python3 main.py --action=test --dataset=kinetics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_kinetics2/' --split_index=2 --batch_size=32