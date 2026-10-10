"""Blocks V12 four-task LeRobot v3.0 (ArmNet-shaped, one AV1 file per episode and view) -> Cosmos3 FD samples.

Matches the FlexPiFwd four-task 25 Hz profile: a window is 21 frames (6 Wan latent frames) plus the 20
departure-indexed commands between them; every view is resized 360x640 -> 192x320 with antialiased
bilinear (FlexPiFwd's ``transforms_F.resize``) and the three views are stacked front / top (= the static
left camera) / wrist into a 576x320 composite (``vstack_3cam_576x320_uniform``). Actions stay raw 8-D
[joint_target(7) rad, gripper_raw]. The dataset rows are already departure-indexed (row i = frame i and
the command issued at frame i), so frames t..t+20 pair with actions t..t+19.

View dropout (arm C5): with probability ``view_dropout_p`` one view of the conditioning frame 0 is set
to black before the VAE; the 20 target frames are never touched (FlexPiFwd F5 semantics).
"""
from __future__ import annotations

import json, random
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import functional as transforms_F

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
    Gripper,
    Joint,
    build_action_spec,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.model.generator.mot import rerope

EMBODIMENT = "blocksv12"  # registered by overlay.sh: domain id 22 (free at 5e67049), raw action dim 8
VIEWS = ("front", "top", "wrist")
VIEW_HW = (192, 320)
NATIVE_HW = (360, 640)
POSE_KEYS = {v: f"observation.camera_T_cw.{v}" for v in VIEWS}
CMD_KEY = "observation.camera_T_cw_commanded_next.wrist"
TRANSLATION_SCALE = 2.0  # FlexFwd's pinned camera_pe_translation_scale on the same cameras / world frame


class BlocksV12FDDataset(BaseActionLeRobotDataset):
    def __init__(
        self,
        *,
        root: str,
        fps: float = 25.0,
        chunk_length: int = 20,
        view_dropout_p: float = 0.0,
        return_camera_poses: bool = False,
        tolerance_s: float = 1e-4,
    ) -> None:
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=0,
            split_val_ratio=0.0,
            split="full",  # train/ and val/ are separate roots; never re-split them
            mode="forward_dynamics",
            embodiment_type=EMBODIMENT,
            viewpoint="concat_view",
            pose_convention="backward_framewise",  # only gates idle-frame detection for a joint-space spec
            tolerance_s=tolerance_s,
        )
        assert 0.0 <= view_dropout_p <= 1.0, view_dropout_p
        self._view_dropout_p = float(view_dropout_p)
        obs_ts = [i * self._dt for i in range(self._chunk_length + 1)]
        act_ts = [i * self._dt for i in range(self._chunk_length)]
        self._delta_timestamps = {f"observation.images.{v}": obs_ts for v in VIEWS}
        self._delta_timestamps["action"] = act_ts
        self._return_camera_poses = bool(return_camera_poses)
        if self._return_camera_poses:
            # FlexFwd rule: latent frame 0 = achieved pose at the window start; latent frame j >= 1 = the
            # commanded pose of its arrival frame 4j, i.e. commanded_next at row 4j - 1 (Wan: 1 + 4k frames).
            for key in POSE_KEYS.values():
                self._delta_timestamps[key] = [0.0]
            self._delta_timestamps[CMD_KEY] = [(4 * j - 1) * self._dt for j in range(1, self._chunk_length // 4 + 1)]
            k = json.loads((Path(root) / "meta" / "camera_intrinsics.json").read_text())["K_row_major_3x3"]
            k_native = torch.tensor([k[f"observation.images.{v}"] for v in VIEWS], dtype=torch.float32)  # [V,3,3]
            self._k_norm = rerope.normalize_intrinsics(rerope.slot_K_from_native(k_native, NATIVE_HW, VIEW_HW), VIEW_HW)
        self._all_shard_roots = [root]
        self._register_sources()

    def _build_action_spec(self):
        return build_action_spec(Joint(7), Gripper())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)
        tiles = [
            transforms_F.resize(
                sample[f"observation.images.{v}"],  # [T,3,360,640] float in [0,1]
                list(VIEW_HW),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for v in VIEWS
        ]
        if self._view_dropout_p > 0.0 and random.random() < self._view_dropout_p:
            tiles[random.randrange(len(VIEWS))][0].zero_()  # conditioning frame only
        video = torch.cat(tiles, dim=-2)  # [T,3,576,320]
        action = sample["action"].float()  # [chunk_length,8]
        extras: dict[str, Any] = {}
        if self._return_camera_poses:
            n_lat = self._chunk_length // 4 + 1
            t_cw = torch.empty(n_lat, len(VIEWS), 4, 4)
            for v, view in enumerate(VIEWS):
                t_cw[:, v] = sample[POSE_KEYS[view]].reshape(-1, 4, 4)[0]  # static views; wrist row 0 = achieved at start
            t_cw[1:, 2] = sample[CMD_KEY].reshape(-1, 4, 4)
            extras["rerope_P_T"], extras["rerope_P_inv"] = rerope.projective_transforms(t_cw, self._k_norm, TRANSLATION_SCALE)
        assert tuple(video.shape[:2]) == (self._chunk_length + 1, 3) and tuple(action.shape) == (self._chunk_length, 8), (
            tuple(video.shape),
            tuple(action.shape),
        )
        return self._build_result(mode=mode, video=video, action=action, ai_caption=sample["task"], **extras)


def get_blocksv12_fd_sft_dataset(
    *,
    root: str,
    chunk_length: int = 20,
    view_dropout_p: float = 0.0,
    return_camera_poses: bool = False,
    resolution: str = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    episode_shuffle_seed: int = 42,
):
    """Same wrapper stack as the public DROID FD recipe (JSON prompt, idle frames, episode-shuffled stream)."""
    dataset = BlocksV12FDDataset(root=root, chunk_length=chunk_length, view_dropout_p=view_dropout_p, return_camera_poses=return_camera_poses)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_idle_frames=True,
        idle_frames_dropout=0.05,
        format_prompt_as_json=True,
    )
    return ActionIterableShuffleDataset(ActionSFTDataset(dataset, transform, resolution), seed=episode_shuffle_seed)
