#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0/' --split_index=0 --n_epochs=16 --lr=2e-7 --early_stop_thresh=4