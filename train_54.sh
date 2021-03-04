#!/bin/bash

python3 segment.py train -d datasets/cityscapes/ -c 19 -s 896 \
    --arch drn_d_54 --batch-size 8 \
    --epochs 500 --lr 0.01 --momentum 0.9 \
    --resume models/drn_d_54/checkpoint_latest.pth.tar
    --save_path models/drn_d_54 \
    --log_dir tmp_log $@

