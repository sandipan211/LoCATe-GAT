#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

python3 viz.py --dataset=ucf --ckp_path='../../log/log_transformer_ucf0/checkpoint.pth.tar' --layer=lca --save_path='/workspace/udit/arijit/sandipan/zsar_divyam/viz'