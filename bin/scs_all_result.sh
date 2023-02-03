#!/usr/bin/env bash

cd ../experiments/
export PATH="$HOME/app/sox-14.4.2/bin:$PATH"
export BSC_DATASET_NAME='simple_call_seq'

# jupyter nbconvert --inplace --execute results_simple_call_seq.ipynb --allow-errors
jupyter nbconvert --to=html --execute results_simple_call_seq.ipynb --allow-errors
