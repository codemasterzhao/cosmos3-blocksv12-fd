# Cosmos3-Nano forward dynamics on Blocks V12 (four tasks)

Fine-tune Cosmos3-Nano as an action-conditioned forward-dynamics world model on our simulated Blocks V12
four-task data, to benchmark Cosmos against our in-house world models. Four arms:

| arm | what | set by `train.sh` |
|---|---|---|
| **c0** | baseline forward dynamics | `REROPE=0 VIEW_DROPOUT_P=0.0` |
| **c1** | + ReRoPE (relative camera geometry in attention) | `REROPE=1 VIEW_DROPOUT_P=0.0` |
| **c2** | + ReRoPE + view dropout | `REROPE=1 VIEW_DROPOUT_P=0.5` |
| **c5** | + view dropout | `REROPE=0 VIEW_DROPOUT_P=0.5` |

Pinned versions: this repo = cosmos-framework `5e67049` plus one commit; base weights `nvidia/Cosmos3-Nano`
revision `411f42a8`; data `haibozzz/blocks-v12-4task-lerobot-v30-camgeom` revision `338c80ca`.

## Train

```bash
uv sync --all-extras --group=cu128-train && source .venv/bin/activate   # cu130-train on CUDA 13 (docs/setup.md)
hf auth login                                                             # anonymous downloads get rate-limited
DATA_ROOT=/local/blocksv12 bash blocksv12/download_data.sh                # 32 GB (train + val)
BASE_ROOT=/local/cosmos_base bash blocksv12/prep_base.sh                  # ~70 GB, Nano + Wan2.2 VAE -> DCP
ARM=c0 RUN_NAME=c0_e3 MAX_ITER=52950 SAVE_ITER=5000 \
DATASET_PATH=/local/blocksv12 BASE_CHECKPOINT_PATH=/local/cosmos_base/Cosmos3-Nano-dcp \
WAN_VAE_PATH=/local/cosmos_base/wan22_vae/Wan2.2_VAE.pth bash blocksv12/train.sh
```

- One node, 8×H100, FSDP, 8 packed samples per GPU (64 per step): ≈2.45 s/step. One epoch is 17,650 steps
  ≈ 11.9 h, so `MAX_ITER=52950` is 3 epochs ≈ 36 h per arm. Our reference model trained 10 epochs.
- `MAX_ITER=10` is a 10-step smoke (≈7 min, most of it compiling).
- Running the same command again resumes from the latest checkpoint. Checkpoints are DCP directories under
  `$OUTPUT_ROOT/cosmos3_v12/action_fd/$RUN_NAME/checkpoints/iter_<N>/` and are never deleted automatically.
- Optional CPU checks: `python blocksv12/test_rerope.py --root $DATASET_PATH/train --out /tmp/rerope.json` and
  `python blocksv12/test_blocksv12_dataset.py --root $DATASET_PATH/train --out /tmp/data.json`.

## Data

- Isaac sim, Franka FR3 + Robotiq 2F-85, cameras `front`, `top` (static left) and `wrist`, 25 fps, 640×360;
  tasks pick-and-place, flip, rotate, mixed.
- `train/`: 3,600 episodes, per task 450 successes + 450 failure twins from the same reset. `val/`: 400
  episodes (50 + 50 per task), the evaluation set: train on `train/` only, and do not re-split (twins share a reset).
- `action` = 8-D `[joint_target(7) rad, gripper_raw]`; row *i* is the command issued at frame *i*.
- Camera poses for ReRoPE are the `observation.camera_T_cw*` columns; intrinsics in `meta/camera_intrinsics.json`.
- The task strings in `meta/tasks.parquet` are provisional.

## What the commit on top of 5e67049 changes (`git diff 5e67049 HEAD`)

- `cosmos_framework/data/generator/action/datasets/blocksv12_fd_dataset.py`: a sample is 21 frames and the
  20 commands between them. Each view is resized to 192×320 and the three are stacked vertically (576×320,
  18×10 patches per latent frame). Actions are raw. View dropout blacks out one view of the conditioning frame.
  With `REROPE=1` the sample also carries per-latent-frame camera transforms.
- `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_blocksv12_posttrain.py`: the
  public DROID FD recipe pointed at this dataset; knobs come from the environment that `train.sh` sets.
- `cosmos_framework/model/generator/mot/rerope.py`: on the 8 lowest-frequency RoPE bins of every vision token,
  the rotation is replaced by the camera's projective transform (`Q ← PᵀQ`, `K ← P⁻¹K`, `P = K · T_cw`). Text
  and action tokens are unchanged.
- Small hooks in seven framework files (16 lines added, 3 replaced): register embodiment `blocksv12` (domain 22, 8-D actions),
  import the experiment, and carry the camera transforms from the batch into attention. They do nothing
  when `REROPE=0`.
- `examples/toml/sft_config/action_fd_blocksv12_posttrain.toml` and this `blocksv12/` directory.
- `.github/workflows/` removed: NVIDIA's CI does not run outside their organisation.

## Notes

- Domain id 22 is unregistered at 5e67049 and untrained in the Nano weights (its bias rows are exactly zero),
  so it starts from a clean row. Newer cosmos-framework assigns 22 to `behavior1k_lerobot`; there, 10, 11, 14,
  18 and 19 are free.
- ReRoPE does not support context parallel (the per-token camera matrices are not sharded). It is not needed
  at this sequence length; FSDP alone fits.
- A resumed run restarts the episode shuffle (the dataloader keeps no cursor), so epoch boundaries are approximate.
- Only Nano was tested. Edge or Super need their own experiment config and base weights.
- All four arms ran a 10-step smoke on Nano, 8×H100, on 2026-10-09. On 2026-10-11 this repo was checked from a fresh
  GitHub download: `download_data.sh` fetched the 10,812 train files (28.8 GB) in ~20 min on a free account, `prep_base.sh`
  reproduced our converted weights byte for byte, and `train.sh` ran a 10-step c1 run.

Contact: Haibo Zhao (haiboz@allenai.org).
