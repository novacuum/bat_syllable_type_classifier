#!/bin/bash

#SBATCH --mail-type=none
#SBATCH --time=4:0:0
#SBATCH --mem=32g
#SBATCH --output=notebookrunner.log
#SBATCH --error=notebookrunner.log
#SBATCH --job-name=notebookrunner_%j
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslaP100:1

# gpus teslaP100 gtx1080ti

BASEDIR=$(dirname "$PWD")
cd $BASEDIR/src/

export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"

python scripts/notebookrunner.py
