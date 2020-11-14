#!\bin\bash
for dist in 0 50 100 200
do
    for step in 0 10 
    do
        for arch in drn_d_22 resnet_d_22
        do
            python3 -B segment.py attack -d datasets/cityscapes -c 19 --arch $arch  \
            --resume "models/$arch/checkpoint_latest.pth.tar" --phase val --batch-size 1 \
            --with-gt --log_dir log_rap --output_path output_rap \
            --rap --patch-dist $dist --eval-num 100 --pgd-steps $step
        done
    done
done