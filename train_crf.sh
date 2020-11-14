#!/bin/bash

python3 -B segment.py train_crf -d datasets/cityscapes/ -c 19 -s 896 \
    --arch drn_d_22 --batch-size 8 --epochs 250 --lr 0.01 --momentum 0.9 \
    --step 100 --save_path models/drn_d_22_crf --base_model models/drn_d_22/checkpoint_latest.pth.tar \
    --resume models/drn_d_22_crf/checkpoint_latest.pth.tar