#!/usr/bin/env bash
# Single-node torchrun launcher for one arm. Run from the patched cosmos-framework checkout (or with it first on
# PYTHONPATH). Required env: ARM (c0|c1|c2|c5), DATASET_PATH (dir holding train/ and val/), BASE_CHECKPOINT_PATH
# (Cosmos3-Nano-dcp), WAN_VAE_PATH (Wan2.2_VAE.pth). Optional: OUTPUT_ROOT, RUN_NAME, MAX_ITER, SAVE_ITER,
# SAMPLES_PER_BATCH (packed samples per GPU per step), NPROC (GPUs; the TOML pins FSDP shard degree 8).
set -euo pipefail
: "${ARM:?}" "${DATASET_PATH:?}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}"
case "$ARM" in
  c0) export VIEW_DROPOUT_P=0.0 REROPE=0;;   # baseline
  c1) export VIEW_DROPOUT_P=0.0 REROPE=1;;   # + ReRoPE
  c2) export VIEW_DROPOUT_P=0.5 REROPE=1;;   # + ReRoPE + view dropout
  c5) export VIEW_DROPOUT_P=0.5 REROPE=0;;   # + view dropout only
  *) echo "ARM must be c0|c1|c2|c5" >&2; exit 2;;
esac
export MAX_ITER=${MAX_ITER:-10} SAVE_ITER=${SAVE_ITER:-10} SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-8}
export RUN_NAME=${RUN_NAME:-blocksv12_fd_$ARM} IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs} DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH
export PYTHONUNBUFFERED=1
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test -f "$DATASET_PATH/train/meta/info.json" && test -d "$BASE_CHECKPOINT_PATH/model" && test -f "$WAN_VAE_PATH"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC:-8}" -m cosmos_framework.scripts.train \
  --sft-toml "$BUNDLE/toml/action_fd_blocksv12_posttrain.toml"
# Checkpoints: $IMAGINAIRE_OUTPUT_ROOT/cosmos3_v12/action_fd/$RUN_NAME/checkpoints/iter_<N>/
