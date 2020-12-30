#!/bin/bash

python3 segment.py train -d datasets/cityscapes/ -c 19 -s 840 \
    --arch drn_d_22 --random-scale 2 --random-rotate 10 --batch-size 8 \
    --epochs 500 --lr 0.01 --momentum 0.9 -j 16 --lr-mode poly \
    --save_path models/drn_d_22_ms \
    --log_dir tmp_log $@