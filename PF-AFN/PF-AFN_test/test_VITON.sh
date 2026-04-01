#!/bin/bash
python -u eval_PBAFN_viton.py --name=cloth-warp --resize_or_crop=none --batchSize=32 --gpu_ids=3 \
  --warp_checkpoint=/mnt/aix23904/아르포아/virtual_tryon/DCI-VTON-Virtual-Try-On_fit/PF-AFN/PF-AFN_train/checkpoints/train_viton_2gpu/PBAFN_warp_epoch_089.pth --label_nc=13 --dataroot=/mnt/aix23904/아르포아/virtual_tryon/Fit_data \
  --fineSize=512 --unpaired
# python -u eval_PBAFN_viton.py --name=cloth-warp --resize_or_crop=none --batchSize=32 --gpu_ids=3 \
#   --warp_checkpoint=/mnt/aix23904/아르포아/virtual_tryon/DCI-VTON-Virtual-Try-On/PF-AFN/PF-AFN_train/checkpoints/train_viton_2gpu/PBAFN_warp_epoch_100.pth --label_nc=13 --dataroot=/mnt/aix23904/아르포아/virtual_tryon/Fit_data \
#   --fineSize=512 --unpaired