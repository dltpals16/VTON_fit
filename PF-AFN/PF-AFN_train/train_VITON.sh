# #!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 train_PBAFN_viton.py --name=train_viton \
--resize_or_crop=none --verbose --tf_log --batchSize=32 --num_gpus=1 --gpu_ids=0 --label_nc=13 --dataroot=/mnt/aix22401/아르포아/Fit_data &
# nohup torchrun --standalone --nnodes=1 --nproc_per_node=2 \
# train_PBAFN_viton.py --name=train_viton_adaLN \
# --resize_or_crop=none --verbose --tf_log --batchSize=32 \
# --num_gpus=2 --label_nc=13  \
# --dataroot=/mnt/aix22401/아르포아/Fit_data &
# nohup torchrun --standalone --nnodes=1 --nproc_per_node=4 \
#   train_PBAFN_viton.py --name=train_viton_multi \
#   --resize_or_crop=none --verbose --tf_log --batchSize=32 \
#   --num_gpus=4 --label_nc=13 \
#   --dataroot=/mnt/aix23904/아르포아/virtual_tryon/Fit_data \
#   > train_viton_multi.out 2>&1 &