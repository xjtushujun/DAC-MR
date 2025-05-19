#!/usr/bin/env bash

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
# CUDA_VISIBLE_DEVICES=0 python co_tuning_kl40.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
#   --log logs/moco_pretrain_co_tuning_kl40/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
# CUDA_VISIBLE_DEVICES=0 python co_tuning_kl40.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
#   --log logs/moco_pretrain_co_tuning_kl40/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune


#!/usr/bin/env bash

# MoCo (Unsupervised Pretraining)
#CUB-200-2011
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/cub200 -d CUB200 -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cub200_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/cub200 -d CUB200 -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cub200_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/cub200 -d CUB200 -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cub200_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/cub200 -d CUB200 -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cub200_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune

# Standford Cars
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cars_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cars_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cars_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/cars_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune

# Aircrafts
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/aircraft -d Aircraft -sr 100 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/aircraft_100 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/aircraft -d Aircraft -sr 50 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/aircraft_50 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/aircraft -d Aircraft -sr 30 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/aircraft_30 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
CUDA_VISIBLE_DEVICES=5 python co_tuning_kl42.py data/aircraft -d Aircraft -sr 15 --seed 0 --lr 0.1 -i 2000 --lr-decay-epochs 3 6 9 --epochs 12 \
  --log logs/moco_pretrain_co_tuning_kl42/aircraft_15 --pretrained checkpoints/moco_v1_200ep_backbone.pth --finetune
