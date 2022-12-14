#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N lm_train
#$ -j y -o ./rl4lm_exps/rl4lm_experiment/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=32G,mem_free=32G,gpu=1,hostname="b19|c0[2356789]|c1[0123456789]"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
for _ in $(seq 1); do source /home/gqin2/scripts/acquire-gpu; done
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

# Activate any environments, call your script, etc
source /home/cxiao7/miniconda3/etc/profile.d/conda.sh
conda activate rl4lm
python ./scripts/training/train_text_generation.py --config_path scripts/training/task_configs/common_gen/t5_nlpo_bleubert.yml --experiment_name t5_nlpo_on_supervised_bertscore