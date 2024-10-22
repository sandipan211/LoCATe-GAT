#!/bin/bash
cd /workspace/udit/arijit/sandipan/zsar_divyam/own_zsar/baseline

# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics9/' --split_index=9 --n_epochs=80 --lr=2e-7


# for splits 0 and 1 may need to redo from beginning
# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics3_retry1/' --split_index=3 --n_epochs=80 --lr=2e-7

# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics2_retry1/' --split_index=2 --n_epochs=80 --lr=2e-7

python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics4_retry1/' --split_index=4 --n_epochs=80 --lr=2e-7

python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics5_retry1/' --split_index=5 --n_epochs=80 --lr=2e-7

python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics6_retry1/' --split_index=6 --n_epochs=80 --lr=2e-7

# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics7/' --split_index=7 --n_epochs=80 --lr=2e-7

# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics8/' --split_index=8 --n_epochs=80 --lr=2e-7





# trying training for 80 epochs on oly0
# python3 main.py --action=train --dataset=olympics --network=clip_transformer --semantic=clip --image_size=224 --save_path='../../log/log_transformer_olympics0_80eps/' --split_index=0 --n_epochs=80 --lr=2e-7




