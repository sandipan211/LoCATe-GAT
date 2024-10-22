#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

python3 main.py --action=train --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb2/' --split_index=2 --n_epochs=28 --lr=2e-7 --early_stop_thresh=4