#!/bin/bash
echo "Evaluating $2 images on hdc with pgd step: $1"
bash eval_hdc.sh 1 200
bash eval_hdc.sh 20 200
bash eval_hdc.sh 50 200
bash eval_hdc.sh 100 200
