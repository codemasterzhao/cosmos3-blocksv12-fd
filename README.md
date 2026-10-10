# Cosmos3-Nano forward dynamics on Blocks V12 (four tasks)

Fine-tune Cosmos3-Nano as an action-conditioned forward-dynamics world model on our simulated Blocks V12
four-task data, in four arms:

| arm | what | env |
|---|---|---|
| **C0** | baseline forward dynamics | `REROPE=0 VIEW_DROPOUT_P=0.0` |
| **C1** | + ReRoPE (relative camera geometry in the attention) | `REROPE=1 VIEW_DROPOUT_P=0.0` |
| **C2** | + ReRoPE + view dropout | `REROPE=1 VIEW_DROPOUT_P=0.5` |
| **C5** | + view dropout only | `REROPE=0 VIEW_DROPOUT_P=0.5` |

Built against **cosmos-framework commit `5e67049`** and **`nvidia/Cosmos3-Nano` revision `411f42a8`**.
All four arms ran a 10-step smoke on 8×H100 (logs in the tarball's `smoke_logs/`).
Code: https://github.com/codemasterzhao/cosmos3-blocksv12-fd

## Dataset

**https://huggingface.co/datasets/haibozzz/blocks-v12-4task-lerobot-v30-camgeom** (public, 29 GB, one LeRobot
v3.0 root `train/`; `DATASET_PATH` is the directory that contains it).

- Isaac sim, Franka FR3 + Robotiq 2F-85, three cameras (`front`, `top` = static left, `wrist`), 25 fps, 640×360;
  tasks pick-and-place, flip, rotate, mixed.
- `train/`: 3,600 episodes (per task 450 successes + 450 failure twins from the same reset). Train on it as
  given; do not carve a validation split out of it at random (twins share a reset). Our 400-episode validation
  set is held out on our side.
- `action` = 8-D `[joint_target(7) rad, gripper_raw]`; row *i* is the command issued at frame *i*.
  Camera poses for ReRoPE are the `observation.camera_T_cw*` columns; intrinsics in `meta/camera_intrinsics.json`.
- The task strings in `meta/tasks.parquet` are provisional.

## Code

- `cosmos_framework/data/generator/action/datasets/blocksv12_fd_dataset.py` — dataset + factory. A sample is
  21 frames and the 20 commands between them; each view is resized to 192×320 and the three views are stacked
  vertically (576×320 → 18×10 patches per latent frame). Actions are raw. View dropout blacks out one view of
  the conditioning frame. With `REROPE=1` the sample also carries per-latent-frame camera transforms.
- `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_blocksv12_posttrain.py` — the
  public DROID FD recipe pointed at this dataset. Knobs come from the environment: `VIEW_DROPOUT_P`, `REROPE`,
  `MAX_ITER`, `SAVE_ITER`, `SAMPLES_PER_BATCH`, `RUN_NAME`, `DATASET_PATH`, `BASE_CHECKPOINT_PATH`.
- `cosmos_framework/model/generator/mot/rerope.py` — ReRoPE: on the 8 lowest-frequency RoPE bins of every
  vision token the rotation is replaced by the camera's projective transform (`Q ← PᵀQ`, `K ← P⁻¹K`,
  `P = K · T_cw`); text and action tokens are unchanged.
- `patches/apply_overlay.py` — copies the three files above into a checkout and applies nine small hooks
  (domain registry, experiment import, carrying the camera transforms into attention).

## Install

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git && cd cosmos-framework && git checkout 5e67049
# training venv as in the repo's docs/setup.md:  uv sync --all-extras --group=cu128-train   (or cu130-train)
python /path/to/bundle/patches/apply_overlay.py . --check     # every anchor must match exactly once
python /path/to/bundle/patches/apply_overlay.py .             # run once, on a clean checkout
BASE_ROOT=/local/cosmos_base bash /path/to/bundle/scripts/prep_base.sh   # Nano + Wan2.2 VAE -> DCP, ~70 GB local disk
```

Optional CPU checks: `tests/test_rerope.py --root $DATASET_PATH/train` and
`tests/test_blocksv12_dataset.py --root $DATASET_PATH/train`.

## Train

```bash
cd cosmos-framework
ARM=c1 DATASET_PATH=/data/blocks-v12-4task-lerobot-v30-camgeom \
BASE_CHECKPOINT_PATH=/local/cosmos_base/Cosmos3-Nano-dcp WAN_VAE_PATH=/local/cosmos_base/wan22_vae/Wan2.2_VAE.pth \
MAX_ITER=10 SAVE_ITER=10 RUN_NAME=smoke_c1 bash /path/to/bundle/scripts/train.sh
```

Single node, 8×H100, FSDP, 8 packed samples per GPU (64 per step): ≈2.45 s/step, 17,650 steps ≈ 11.9 h per
epoch per arm. For a real run set `MAX_ITER` to the budget: our in-house reference model trained 10 epochs on
this data; 3 epochs (≈36 h per arm) is the smallest budget we would compare against. Checkpoints land in
`$OUTPUT_ROOT/cosmos3_v12/action_fd/$RUN_NAME/checkpoints/iter_<N>/` (DCP).

## Notes

- Domain id **22** is free at commit 5e67049 but taken on newer `main`; on another commit change `DOMAIN_ID` in
  `patches/apply_overlay.py` to a free id. The patcher refuses to run if its anchors do not match.
- Context parallel is not supported with ReRoPE (the per-token camera matrices are not sharded). It is not
  needed at this sequence length; the smokes and the public recipe use FSDP only.
- A resumed run restarts the episode shuffle (the dataloader keeps no cursor), so epochs are approximate.
- Only Nano was smoked. Edge/Super need their own experiment config and base weights; the hooks and the data
  side are tier-independent.

## Files

```
README.md
patches/apply_overlay.py        copies the three new files and applies the hooks
cosmos_framework/...            dataset, experiment config, rerope
toml/action_fd_blocksv12_posttrain.toml
scripts/prep_base.sh            Cosmos3-Nano@411f42a8 + Wan2.2 VAE -> DCP
scripts/train.sh                torchrun launcher (ARM=c0|c1|c2|c5)
tests/                          ReRoPE unit tests, dataset smoke
LICENSE, NOTICE                 Apache-2.0; the experiment config is derived from NVIDIA's cosmos-framework (OpenMDW-1.1)
smoke_logs/, MANIFEST.sha256    tarball only: the four 10-step smokes on Nano 8×H100, and per-file sha256
```

Contact: Haibo Zhao (haiboz@allenai.org).
