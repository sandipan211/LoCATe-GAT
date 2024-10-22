#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf_truze/' --split_index=777 --n_epochs=16 --lr=2e-7

python3 main.py --action=train --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb_truze/' --split_index=777 --n_epochs=28 --lr=2e-7


######## step 2: KG training #############
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf_truze/' --split_index=777

python3 main_kg.py --mode=kg --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb_truze/' --split_index=777


######## step 3: GAT+Transformer CZSL testing ###########
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf_truze/' --split_index=777

python3 main_kg.py --mode=gat_lca_test --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb_truze/' --split_index=777

######## step 4: Only Transformer CZSL testing ###########
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=test --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf_truze/' --split_index=777

python3 main.py --action=test --dataset=hmdb --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb_truze/' --split_index=777


######## step 5: GAT+Transformer GZSL testing ###########
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=gat_lca_gzsl_test --action=gzsl_test  --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf_truze/' --split_index=777

python3 main_kg.py --mode=gat_lca_gzsl_test --action=gzsl_test  --dataset=hmdb --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_hmdb_truze/' --split_index=777

