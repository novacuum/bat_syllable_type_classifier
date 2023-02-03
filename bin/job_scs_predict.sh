#!/bin/bash

#SBATCH --mail-type=none
#SBATCH --time=12:0:0
#SBATCH --mem=56g
#SBATCH --output=scs_predict_%j.log
#SBATCH --error=scs_predict_%j.log
#SBATCH --job-name=scs_predict
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslaP100:1

# gpus teslaP100 gtx1080ti


BASEDIR=$(dirname "$PWD")
cd $BASEDIR/src/

export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"
export BSC_DATASET_NAME='simple_call_seq'

python scripts/scs_predict.py
