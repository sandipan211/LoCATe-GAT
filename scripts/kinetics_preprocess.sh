#!/bin/bash

cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

python3 kinetics_utils.py --action=find_corrupt --dataset=k600 --data=val --split_index=2

# python3 kinetics_utils.py --action=find_corrupt --dataset=k400 --data=train
