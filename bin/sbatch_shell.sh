#!/usr/bin/env bash
# gpus teslaP100 gtx1080ti
srun --time=2:00:0 --mem=24g --cpus-per-task=2 -p gpu --gres=gpu:teslaP100:1 --pty /bin/bash
