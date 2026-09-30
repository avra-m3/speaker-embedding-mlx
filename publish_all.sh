#!/usr/bin/env bash
# Convert every released ReDimNet2 checkpoint from b0 to b4 and upload each one to
# Hugging Face as <namespace>/redimnet2-<name>-mlx. Needs HF_TOKEN with write access.
#
#   ./publish_all.sh [namespace]   # default namespace: causal
set -euo pipefail
NS="${1:-causal}"
OUT=build
uv sync --group convert
for size in b0 b1 b2 b3 b4; do
  for train in lm ptn; do
    uv run python convert.py --model "$size" --train-type "$train" --dataset vox2 \
      --out "$OUT/$size-vox2-$train"
  done
done
uv run python convert.py --model b3 --train-type lm --dataset "vb2+vox2+cnc2_v0" \
  --out "$OUT/b3-vb2+vox2+cnc2_v0-lm"
uv run python publish_hf.py --namespace "$NS" "$OUT"/*
