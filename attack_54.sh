#!/bin/bash

python3 -B segment.py attack -d datasets/cityscapes -c 19 --arch drn_d_54 \
    --resume models/drn_d_54_ms/checkpoint_latest.pth.tar --phase val \
    --batch-size 1 --with-gt --pgd-steps 10 \
    --log_dir test --output_path test $@