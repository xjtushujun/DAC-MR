#!/usr/bin/env bash
# Supervised Pretraining
# CUB-200-2011
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log logs/erm_kl42/cub200_100
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log logs/erm_kl42/cub200_50
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log logs/erm_kl42/cub200_30
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log logs/erm_kl42/cub200_15

# Standford Cars
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/stanford_cars -d StanfordCars -sr 100 --seed 0 --finetune --log logs/erm_kl42/car_100
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/stanford_cars -d StanfordCars -sr 50 --seed 0 --finetune --log logs/erm_kl42/car_50
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/stanford_cars -d StanfordCars -sr 30 --seed 0 --finetune --log logs/erm_kl42/car_30
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/stanford_cars -d StanfordCars -sr 15 --seed 0 --finetune --log logs/erm_kl42/car_15

# Aircrafts
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/aircraft -d Aircraft -sr 100 --seed 0 --finetune --log logs/erm_kl42/aircraft_100
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/aircraft -d Aircraft -sr 50 --seed 0 --finetune --log logs/erm_kl42/aircraft_50
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/aircraft -d Aircraft -sr 30 --seed 0 --finetune --log logs/erm_kl42/aircraft_30
 CUDA_VISIBLE_DEVICES=6 python erm_kl42.py data/aircraft -d Aircraft -sr 15 --seed 0 --finetune --log logs/erm_kl42/aircraft_15


