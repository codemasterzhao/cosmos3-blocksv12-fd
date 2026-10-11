#!/usr/bin/env bash
# Train one arm on one 8-GPU node, inside this repo's venv.
# Required: ARM (c0|c1|c2|c5), DATASET_PATH (the directory that contains train/), BASE_CHECKPOINT_PATH
# (Cosmos3-Nano-dcp from prep_base.sh), WAN_VAE_PATH (Wan2.2_VAE.pth). Optional: MAX_ITER (10), SAVE_ITER (10),
# SAMPLES_PER_BATCH (8 per GPU), RUN_NAME, OUTPUT_ROOT (<repo>/outputs), NPROC (8; the TOML pins FSDP shard 8).
# Rerunning with the same RUN_NAME and OUTPUT_ROOT resumes from the latest checkpoint.
set -euo pipefail
: "${ARM:?}" "${DATASET_PATH:?}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}"
case "$ARM" in
  c0) export VIEW_DROPOUT_P=0.0 REROPE=0;;   # baseline
  c1) export VIEW_DROPOUT_P=0.0 REROPE=1;;   # + ReRoPE
  c2) export VIEW_DROPOUT_P=0.5 REROPE=1;;   # + ReRoPE + view dropout
  c5) export VIEW_DROPOUT_P=0.5 REROPE=0;;   # + view dropout
  *) echo "ARM must be c0|c1|c2|c5" >&2; exit 2;;
esac
for p in "$DATASET_PATH/train/meta/info.json" "$BASE_CHECKPOINT_PATH/model" "$WAN_VAE_PATH"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export MAX_ITER=${MAX_ITER:-10} SAVE_ITER=${SAVE_ITER:-10} SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-8}
export RUN_NAME=${RUN_NAME:-blocksv12_fd_$ARM} IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC:-8}" -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_blocksv12_posttrain.toml
# checkpoints: $IMAGINAIRE_OUTPUT_ROOT/cosmos3_v12/action_fd/$RUN_NAME/checkpoints/iter_<N>/ (never deleted automatically)
