#!/bin/bash

cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 tsne_visual_copy.py --dataset=ucf --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=0 --save_path='../../log/log_transformer_ucf0_tsne_again/' --viz=clip
# python3 tsne_visual_copy.py --dataset=ucf --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=0 --save_path='../../log/log_transformer_ucf0_tsne_again/' --viz=transformer
# python3 tsne_visual_copy.py --dataset=hmdb --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=2 --save_path='../../log/log_transformer_hmdb2/' --viz=clip
# python3 tsne_visual_copy.py --dataset=hmdb --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=2 --save_path='../../log/log_transformer_hmdb2/' --viz=transformer

# python3 tsne_visual_copy.py --dataset=ucf --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=0 --save_path='../../log/log_transformer_ucf0_tsne_again/' --viz=clip
python3 tsne_visual_copy.py --dataset=ucf --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --network=clip --semantic=clip --image_size=224 --split_index=0 --save_path='../../log/log_transformer_ucf0_tsne_again/' --viz=transformer


# python3 tsne_visual.py --dataset=ucf --network=clip --semantic=clip --image_size=224 --split_index=0 --save_path='../../log/log_transformer_ucf0_tsne_again/' --viz=transformer

# python3 tsne_embed_copy.py --dataset=ucf --dataset_path="/workspace/udit/arijit/sandipan/zsar_divyam/datasets" --semantic=clip --split_index=0 --save_name='../../log/log_transformer_ucf0_tsne_again/sem_embed_clip.pdf'
