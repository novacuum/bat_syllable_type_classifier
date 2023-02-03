#!/usr/bin/env bash

BASEDIR=$(dirname "$PWD")
cd $BASEDIR/src/

export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"

python test/lrp.py

