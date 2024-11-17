#!/bin/bash

./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 2.5e-4 --ent-coef 1e-3
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 1e-4 --ent-coef 1e-3
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 1e-4 --ent-coef 1e-4
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 5e-5 --ent-coef 1e-4

./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 2.5e-4 --ent-coef 1e-3 --reward-done -5.0
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 1e-4 --ent-coef 1e-3 --reward-done -5.0
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 1e-4 --ent-coef 1e-4 --reward-done -5.0
./build/train_ppo --total-steps 1e8 --game-size 8 --learning-rate 5e-5 --ent-coef 1e-4 --reward-done -5.0
