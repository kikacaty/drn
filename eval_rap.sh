#!\bin\bash
for dist in 0 50 100 200
do
    for step in 20 50 
    do
        for arch in duc_hdc_bigger duc_hdc_rf duc_hdc_no
        do
            python3 -B segment.py attack -d datasets/cityscapes -c 19 --arch $arch  \
            --resume "models/$arch/checkpoint_019.pth.tar" --phase val --batch-size 1 \
            --with-gt --log_dir log_rap --output_path output_rap \
            --rap --patch-dist $dist --eval-num 100 --pgd-steps $step &&
        done
    done
done