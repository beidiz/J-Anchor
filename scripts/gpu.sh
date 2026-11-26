#!/usr/bin/env bash
# Run a command inside an existing Slurm GPU allocation:
#   JOBID=<slurm job id> bash scripts/gpu.sh python -m janchor.calibrate --model llama
# (CUDA_VISIBLE_DEVICES=<i> in front of the command picks one GPU of a multi-GPU allocation.)
# Optional: JANCHOR_ENV=<shell file> is sourced first (e.g. one that activates the Python environment and exports
# JANCHOR_RES / HF_HOME / JANCHOR_WILDGUARD_PYTHON, see config.py and the README).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${JOBID:?set JOBID to a running Slurm job (squeue -u \$USER)}"
PRE=""
if [ -n "${JANCHOR_ENV:-}" ]; then PRE="source '$JANCHOR_ENV' >/dev/null 2>&1; "; fi
exec srun --overlap --jobid="$JOBID" --ntasks=1 bash -lc "${PRE}cd '$HERE'; $*"
