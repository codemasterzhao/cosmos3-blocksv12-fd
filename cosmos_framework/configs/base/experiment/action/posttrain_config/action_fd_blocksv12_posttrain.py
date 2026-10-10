# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Derived from cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_droid_posttrain.py
# (cosmos-framework commit 5e67049), modified for the Blocks V12 four-task dataset.

"""``action_fd_blocksv12_posttrain`` — Cosmos3-Nano forward dynamics on Blocks V12 four-task (C0 / C5).

A copy of ``action_fd_droid_posttrain`` with the Blocks V12 dataset, the 480 tier, 21-frame windows and
local DCP checkpoints. The arm is selected by ``VIEW_DROPOUT_P`` (0.0 = C0, 0.5 = C5) and ``REROPE`` (1 = C1/C2); the budget by
``MAX_ITER`` / ``SAVE_ITER`` / ``SAMPLES_PER_BATCH``. ``DATASET_PATH`` is the packaged dataset root that
holds ``train/`` and ``val/``.
"""
from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_droid_posttrain import _make_model_config
from cosmos_framework.data.generator.action.datasets.blocksv12_fd_dataset import get_blocksv12_fd_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

_cfg = _make_model_config()
_cfg["resolution"] = "480"
_cfg["tokenizer"]["encode_exact_durations"] = [21]

_VIEW_DROPOUT_P = float(os.environ.get("VIEW_DROPOUT_P", "0.0"))
_REROPE = os.environ.get("REROPE", "0") == "1"  # C1/C2: dataset emits camera poses, attention applies ReRoPE
_MAX_ITER = int(os.environ.get("MAX_ITER", "10"))
_SAVE_ITER = int(os.environ.get("SAVE_ITER", "10"))
_SAMPLES_PER_BATCH = int(os.environ.get("SAMPLES_PER_BATCH", "8"))

action_fd_blocksv12_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "local"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_v12",
            group="action_fd",
            name=os.environ.get("RUN_NAME", "blocksv12_fd"),
            wandb_mode="disabled",
        ),
        model=dict(config=_cfg),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[20000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=_MAX_ITER,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            straggler_detection=dict(enabled=False, report_freq=50),  # the image lacks the `straggler` package
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=50, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=100),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                compile_tokenizer=dict(enabled=True, warmup_resolutions=["480"]),
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            keys_to_skip_loading=["net_ema."],  # warm start: EMA is seeded from net
            load_path="${oc.env:BASE_CHECKPOINT_PATH}",
            load_training_state=False,
            save_iter=_SAVE_ITER,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="blocksv12",
            max_samples_per_batch=_SAMPLES_PER_BATCH,
            max_sequence_length=None,
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=3,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    blocksv12=dict(
                        ratio=1,
                        dataset=L(get_blocksv12_fd_sft_dataset)(
                            root="${oc.env:DATASET_PATH}/train",
                            chunk_length=20,
                            view_dropout_p=_VIEW_DROPOUT_P,
                            return_camera_poses=_REROPE,
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            cfg_dropout_rate=0.1,
                            episode_shuffle_seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(group="experiment", package="_global_", name="action_fd_blocksv12_posttrain", node=action_fd_blocksv12_posttrain)
