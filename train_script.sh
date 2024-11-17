#!/bin/bash

./build/train_ppo --total-steps 1e8 --game-size 8 --path-load-model /home/yy/Coding/GitHub/ppo_libtorch/ckpt/seed1_hidden256_size8_20241117_195721/0100007936.pt --learning-rate 1e-5 --ent-coef 1e-4
