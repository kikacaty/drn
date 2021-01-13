#!/bin/bash

python3 segment.py train -d datasets/cityscapes/ -c 19 -s 896 \
    --arch drn_d_22 --batch-size 8 \
    --epochs 500 --lr 0.01 --momentum 0.9 \
    --save_path models/drn_d_22 \
    --log_dir tmp_log $@

