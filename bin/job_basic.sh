#!/bin/bash

#SBATCH --mail-type=none
#SBATCH --time=12:00:0
#SBATCH --mem=32g
#SBATCH --output=job_basic_%j.log
#SBATCH --error=job_basic_%j.log
#SBATCH --job-name=job_basic_%j
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslaP100:2
#SBATCH --requeue

# gpus teslaP100 gtx1080ti
# gpu vs gpu-debug

BASEDIR=$(dirname "$PWD")
cd $BASEDIR/src/

export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"
export BSC_DATASET_NAME=$datasetName

python scripts/$pythonFile -v=$variant -i=$index
