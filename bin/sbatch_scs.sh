#!/usr/bin/env bash

pythonFile="scs.py"
datasetName="simple_call_seq"

for v in r3; do
  for i in {0..3}; do
    log="scs_${v}_${i}.log"
    sbatch --job-name=scs_${v}_${i} --export=ALL,variant=$v,index=$i,pythonFile=$pythonFile,datasetName=$datasetName --error=$log --output=$log  job_basic.sh
  done
done
echo 0
