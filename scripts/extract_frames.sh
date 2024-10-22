#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 gradcam_frames.py --dataset=ucf --class_name='all' --save_path='../../class_analysis/'  --n_videos=10

# python3 gradcam_frames.py --dataset=olympics --class_name='all' --save_path='../../misc/'  --n_videos=10

python3 gradcam_frames.py --dataset=ucf --class_name='all' --save_path='../../viz/'  --n_videos=10
