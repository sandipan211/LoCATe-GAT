#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline


python3 main.py --action=test --dataset=ucf --network=clip --semantic=clip --image_size=224 --save_path='../../log/log_clip_clip_ucf0/' --split_index=0