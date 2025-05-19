#!/usr/bin/env bash
# CUB-200-2011
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/l2_sp_kl42/cub200_100 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/l2_sp_kl42/cub200_50 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/l2_sp_kl42/cub200_30 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/l2_sp_kl42/cub200_15 --regularization-type l2_sp

# Stanford Cars
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/l2_sp_kl42/car_100 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/l2_sp_kl42/car_50 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/l2_sp_kl42/car_30 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/l2_sp_kl42/car_15 --regularization-type l2_sp

# Aircrafts
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/l2_sp_kl42/aircraft_100 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/l2_sp_kl42/aircraft_50 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/l2_sp_kl42/aircraft_30 --regularization-type l2_sp
CUDA_VISIBLE_DEVICES=3 python delta_kl42.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/l2_sp_kl42/aircraft_15 --regularization-type l2_sp



