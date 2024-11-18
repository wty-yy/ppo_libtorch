#!/bin/bash

./build/train_ppo --total-steps 1e9 --init-lr 1e-4 --final-lr 2e-5 --init-ent-coef 1e-3 --final-ent-coef 5e-5 --anneal-steps 9e8 --path-load-model /home/yy/Coding/GitHub/ppo_libtorch/best_ckpt/v3_seed1_size8_5e8/0500006912.pt
