#!/usr/bin/env bash
# One short generation per model (harmful rows with the fixed prefill, benign rows without), to check the pipeline.
cd "$(dirname "$0")/.."; mkdir -p runs/smoke logs
for m in ${MODELS:-llama gemma4 qwen3.5 qwen3}; do
  python -m janchor.generate --model $m --input data/smoke.jsonl --prefill fixed --max_new 48 --batch 8 --out runs/smoke/${m}_fixed.jsonl > logs/smoke_$m.log 2>&1 \
    && python -m janchor.generate --model $m --input data/smoke.jsonl --prefill none --max_new 48 --batch 8 --out runs/smoke/${m}_none.jsonl >> logs/smoke_$m.log 2>&1 \
    && echo "$m ok" || echo "$m FAILED (logs/smoke_$m.log)"
done
