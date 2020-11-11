#!/bin/bash

echo python3 -B segment.py attack -d /tmp/cityspcae/ -c 19 --arch duc_hdc_no \
    --resume models/duc_hdc_no/checkpoint_019.pth.tar --phase val \
    --batch-size 1 --with-gt \
    --pgd-steps $1 --eval-num $2 \
    --output_path output \
    --log_dir log &&

echo python3 -B segment.py attack -d /tmp/cityscapes -c 19 --arch duc_hdc_rf \
    --resume models/duc_hdc_rf/checkpoint_019.pth.tar --phase val \
    --batch-size 1 --with-gt \
    --pgd-steps $1 --eval-num $2 \
    --output_path output \
    --log_dir log &&

echo python3 -B segment.py attack -d /tmp/cityscapes -c 19 --arch duc_hdc_bigger \
    --resume models/duc_hdc_bigger/checkpoint_019.pth.tar --phase val \
    --batch-size 1 --with-gt \
    --pgd-steps $1 --eval-num $2 \
    --output_path output \
    --log_dir log
