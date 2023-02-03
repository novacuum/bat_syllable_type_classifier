#!/usr/bin/env bash

BASEDIR=$(dirname "$PWD")
cd $BASEDIR/src/

export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"
export BSC_DATASET_NAME='simple_call_seq'

python scripts/scs_all.py
