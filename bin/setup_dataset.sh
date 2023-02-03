#!/usr/bin/env bash

BASEDIR=$(dirname "$PWD")

cd $1
export PYTHONPATH="$BASEDIR/src/:$PYTHONPATH"
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"

python setup.py
