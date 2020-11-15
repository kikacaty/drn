ngc batch run \
  --instance dgx1v.32g.1.norm  \
  --name "ml-model.$3-$2-$4" \
  --image "nvidia/pytorch:20.02-py3" \
  --datasetid 68858:/cityscape \
  --result /result \
  --workspace ws-chaowei1:/workspace \
  --team onboarding\
  --commandline " cd /workspace/drn/; ls; pip install py3nvml; python3 -B segment.py attack -d /cityscape -c 19 --arch $1 --resume "models/$1/checkpoint_019.pth.tar" --phase val --batch-size 1 --with-gt --pgd-steps $2 --eval-num 200 --output_path output_restart_5 --log_dir log_restart_5 --patch_dist $4"
