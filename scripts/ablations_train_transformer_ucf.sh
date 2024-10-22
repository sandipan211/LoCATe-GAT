#!/bin/bash



# ======================trying different dilation branches=================================

# with 2 branches

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca2_retry1/' --split_index=0 --n_epochs=6 --lr=2e-7 --early_stop_thresh=4 --lca_branch=2

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca2_retry1/' --split_index=0 --lca_branch=2

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca2_retry1/' --split_index=0 --lca_branch=2

# # with 1 branch
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca1_retry1/' --split_index=0 --n_epochs=6 --lr=2e-7 --early_stop_thresh=4 --lca_branch=1

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca1_retry1/' --split_index=0 --lca_branch=1

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_lca1_retry1/' --split_index=0 --lca_branch=1

###################################################################################################################


# =================trying different semantics===============================
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg_temp.py --mode=cls --dataset=ucf --network=clip_transformer_classifier --semantic=word2vec --image_size=224 --save_path='../../log/log_transformer_ucf0_word/' --split_index=0 --n_epochs=5
# python3 main_kg_temp.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=word2vec --image_size=224 --save_path='../../log/log_transformer_ucf0_word/' --split_index=0 --n_epochs=100000 --lr=1e-5
# python3 main_kg_temp.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=word2vec --image_size=224 --save_path='../../log/log_transformer_ucf0_word/' --split_index=0

# python3 main_kg_temp.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=sent2vec --image_size=224 --save_path='../../log/log_transformer_ucf0_sent/' --split_index=0 --n_epochs=100000 --lr=1e-5
# python3 main_kg_temp.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=sent2vec --image_size=224 --save_path='../../log/log_transformer_ucf0_sent/' --split_index=0


# ================trying different backbones for CLIP - ViT-B/16 (default), ViT-B/32, ViT-L/14===========
# ## cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# ## python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitB32/' --split_index=0 --n_epochs=16 --lr=2e-7 --early_stop_thresh=4 --vit_backbone='ViT-B/32'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitB32/' --split_index=0 --vit_backbone='ViT-B/32'

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitB32/' --split_index=0 --vit_backbone='ViT-B/32'


# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitL14/' --split_index=0 --n_epochs=6 --lr=2e-7 --early_stop_thresh=4 --vit_backbone='ViT-L/14' --batch_size=16

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitL14/' --split_index=0 --vit_backbone='ViT-L/14' --batch_size=16

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_vitL14/' --split_index=0 --vit_backbone='ViT-L/14' --batch_size=16


# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN50/' --split_index=0 --n_epochs=16 --lr=2e-7 --early_stop_thresh=4 --vit_backbone='RN50'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN50/' --split_index=0 --vit_backbone='RN50'

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN50/' --split_index=0 --vit_backbone='RN50'



# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN101/' --split_index=0 --n_epochs=16 --lr=2e-7 --early_stop_thresh=4 --vit_backbone='RN101'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN50/' --split_index=0 --vit_backbone='RN50'

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_RN50/' --split_index=0 --vit_backbone='RN50'

############################################################################################################


# ===============================Visual models - r2plus1d========================================
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=r2plus1d --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_r2plus1d/' --split_index=0 --n_epochs=150 --lr=2e-7 --early_stop_thresh=4 --val_freq=15

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=r2plus1d --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_r2plus1d/' --split_index=0

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=r2plus1d --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_r2plus1d/' --split_index=0


# ======================counting model parameters - change it accordingly=======================
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_copy/' --split_index=0 --count_params > ../../log/log_transformer_ucf0_copy/params.txt
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_copy/' --split_index=0 --count_params > ../../log/log_transformer_ucf0_copy/params_kg.txt

###############################################################################################


# =========================Without CBAM=========================================================
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_cbam/' --split_index=0 --n_epochs=16 --lr=2e-7 --early_stop_thresh=4

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat

# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_cbam/' --split_index=0

# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_cbam/' --split_index=0


# =========================Linear probe=====================================================
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main_baseline.py --dataset=ucf --image_size=224 --save_path='../../log/log_baseline_ucf0/' --split_index=0 --num_workers=4

# ======================trying MLP instead of LCA=================================


# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA_retry1/' --split_index=0 --n_epochs=4 --lr=2e-7 --early_stop_thresh=4 --novelty='noLCA'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 


# test after 3 epochs (epoch 2) 
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA_retry1/' --split_index=0 --ckpt_epoch=2 

# test after 12 epochs (epoch 11) 
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --ckpt_epoch=11


# running transformer training from epoch number 12 to 15
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --n_epochs=16 --lr=2e-7 --novelty='noLCA'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --ckpt_epoch=15

# test after 16 epochs (epoch 15) 
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=gat_lca_test --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --ckpt_epoch=15

# count the params
# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/kg_gat
# python3 main_kg.py --mode=kg --dataset=ucf --network=clip_transformer_classifier --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --count_params > ../../log/log_transformer_ucf0_noLCA/params_kg.txt --ckpt_epoch=15


# only LoCATe
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=test --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0/' --split_index=0

# GZSL for TETCI - to indicate polysemy mitigation
# python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0/' --split_index=0
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf1_retry1/' --split_index=1
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf2_retry1/' --split_index=2
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf3/' --split_index=3
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf4_retry1/' --split_index=4
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf5/' --split_index=5
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf6/' --split_index=6
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf7/' --split_index=7
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf8/' --split_index=8
python3 main.py --action=gzsl_test  --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf9/' --split_index=9

# test no-LCA after 16 epochs but without GAT

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main.py --action=test --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --ckpt_epoch=15 --novelty='noLCA'

# cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline
# python3 main.py --action=train --dataset=ucf --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_ucf0_noLCA/' --split_index=0 --count_params > ../../log/log_transformer_ucf0_noLCA/params.txt --ckpt_epoch=15 --novelty='noLCA'