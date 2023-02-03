#!/usr/bin/env bash

pythonFile="sct.py"
datasetName="simple_call_test"

for v in compressed padded variable_length; do
  for i in {0..3}; do
    log="sct_${v}_${i}.log"
    sbatch --job-name=sct_${v}_${i} --export=ALL,variant=$v,index=$i,pythonFile=$pythonFile,datasetName=$datasetName --error=$log --output=$log  job_basic.sh

#   used for initialization/generating files
#    if [ $i -eq 0 ]; then
#      sleep 5m
#    fi
  done
done
echo 0
