#!/bin/bash

#SBATCH --mail-type=none
#SBATCH --time=12:0:0
#SBATCH --mem=56g
#SBATCH --output=scs_all_%j.out
#SBATCH --error=scs_all_%j.err
#SBATCH --job-name=scs_all
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslaP100:1

# gpus teslaP100 gtx1080ti

./scs_all.sh
./scs_all_result.sh
