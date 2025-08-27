#!/bin/bash

# Run benchmark.py using uv run with specified arguments
uv run cs336_systems/benchmark.py \
  --include-backward \
  --batch-size 4 \
  --warmup-steps 5 \
  --execution-steps 10 \
  --vocab-size 10000 \
  --context-length 1024 \
  --d-model 768 \
  --num-layers 12 \
  --num-heads 12 \
  --d-ff 3072

# Run benchmark.py using uv run with specified arguments
uv run cs336_systems/benchmark.py \
  --include-backward \
  --apply-torch-compile \
  --batch-size 4 \
  --warmup-steps 5 \
  --execution-steps 10 \
  --vocab-size 10000 \
  --context-length 1024 \
  --d-model 768 \
  --num-layers 12 \
  --num-heads 12 \
  --d-ff 3072