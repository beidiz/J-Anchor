#!/usr/bin/env bash
# Calibrate every supported model, one after another (needs a GPU), e.g.
#   bash scripts/calibrate_all.sh
#   MODELS="llama qwen3" bash scripts/calibrate_all.sh
#   JOBID=<slurm job id> bash scripts/gpu.sh bash scripts/calibrate_all.sh     # inside a Slurm allocation
cd "$(dirname "$0")/.."; mkdir -p logs
for m in ${MODELS:-llama gemma4 qwen3.5 qwen3}; do
  python -m janchor.calibrate --model $m > logs/calibrate_$m.log 2>&1 && echo "$m ok" || echo "$m FAILED (logs/calibrate_$m.log)"
done
