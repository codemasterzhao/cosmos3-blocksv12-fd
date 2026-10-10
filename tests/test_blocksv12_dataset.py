"""CPU smoke of BlocksV12FDDataset against the packaged v3.0 root (run inside the Cosmos3 image after overlay.sh).

Checks: sample shapes/dtypes, caption, domain id, idle frames; the view-dropout contract (frame 0 of exactly one
view black, frames 1..20 untouched); frame/action alignment against a direct parquet read and a torchcodec
decode of the source video; and the ActionTransformPipeline output (480 tier 9:16 bucket, image_size, padding,
sequence plan).
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import functional as transforms_F

from cosmos_framework.data.generator.action.datasets.blocksv12_fd_dataset import VIEWS, VIEW_HW, BlocksV12FDDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="/results/data_smoke.json")
    a = ap.parse_args()
    root = Path(a.root)
    ds = BlocksV12FDDataset(root=str(root))
    info = json.loads((root / "meta/info.json").read_text())
    rep = {"root": str(root), "windows": len(ds), "episodes": info["total_episodes"],
           "expected_windows": info["total_frames"] - 20 * info["total_episodes"]}
    assert rep["windows"] == rep["expected_windows"], rep

    rng = random.Random(0)
    picked = [0, len(ds) // 2, rng.randrange(len(ds))]
    for idx in picked:
        s = ds[idx]
        assert tuple(s["video"].shape) == (3, 21, 576, 320) and s["video"].dtype == torch.uint8, s["video"].shape
        assert tuple(s["action"].shape) == (20, 8) and s["action"].dtype == torch.float32
        assert isinstance(s["ai_caption"], str) and s["ai_caption"], s["ai_caption"]
        assert int(s["domain_id"]) == 22 and s["mode"] == "forward_dynamics" and int(s["conditioning_fps"]) == 25
        assert "idle_frames" in s and 0 <= int(s["idle_frames"]) <= 20, s.get("idle_frames")
    rep["sample0"] = {"caption": ds[0]["ai_caption"], "idle_frames": int(ds[0]["idle_frames"]),
                      "action_row0": ds[0]["action"][0].tolist()}

    # view dropout: p=1 blacks frame 0 of exactly one view; all other frames identical to the p=0 sample
    dd = BlocksV12FDDataset(root=str(root), view_dropout_p=1.0)
    hits = []
    for idx in picked:
        clean, drop = ds[idx]["video"], dd[idx]["video"]
        bands = [bool((drop[:, 0, 192 * v:192 * (v + 1)] == 0).all()) for v in range(3)]
        assert sum(bands) == 1, bands
        assert torch.equal(clean[:, 1:], drop[:, 1:]), "target frames changed by view dropout"
        v = bands.index(True)
        assert torch.equal(clean[:, 0][:, [i for i in range(576) if not 192 * v <= i < 192 * (v + 1)]],
                           drop[:, 0][:, [i for i in range(576) if not 192 * v <= i < 192 * (v + 1)]])
        hits.append(v)
    rep["view_dropout_views_hit"] = hits

    # alignment: resolve window -> (episode, start row) through the base class, compare with the raw files
    ds_idx, row_idx, episode_id, frame_offset = ds._resolve_index(picked[-1])
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    start = eps["dataset_from_index"][episode_id] + frame_offset
    assert row_idx == start, (row_idx, start)
    data = root / info["data_path"].format(chunk_index=eps["data/chunk_index"][episode_id], file_index=eps["data/file_index"][episode_id])
    rows = pq.read_table(data, filters=[("episode_index", "=", episode_id)]).to_pydict()
    k = rows["index"].index(start)
    s = ds[picked[-1]]
    assert np.array_equal(s["action"].numpy(), np.asarray(rows["action"][k:k + 20], dtype=np.float32)), "action rows misaligned"
    for v in VIEWS:
        key = f"observation.images.{v}"
        path = root / info["video_path"].format(video_key=key, chunk_index=eps[f"videos/{key}/chunk_index"][episode_id],
                                               file_index=eps[f"videos/{key}/file_index"][episode_id])
        for t in (0, 20):
            ref = VideoDecoder(str(path))[frame_offset + t].float() / 255  # [3,360,640]
            ref = transforms_F.resize(ref, list(VIEW_HW), interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            got = s["video"][:, t, 192 * VIEWS.index(v):192 * (VIEWS.index(v) + 1)].float()
            err = (got - (ref * 255).round()).abs().max().item()
            assert err <= 1.0, (v, t, err)
    rep["alignment"] = {"episode": episode_id, "start_row": start, "max_pixel_err": "<=1"}

    # transform pipeline without tokenizer: bucket, padding, plan, action padding
    tf = ActionTransformPipeline(tokenizer_config=None, max_action_dim=64, append_idle_frames=True, format_prompt_as_json=True)
    out = tf(ds[picked[0]], "480")
    assert out["image_size"].tolist() == [832, 480, 576, 320], out["image_size"]
    assert tuple(out["video"].shape) == (3, 21, 832, 480), out["video"].shape
    assert tuple(out["action"].shape) == (20, 64) and int(out["raw_action_dim"]) == 8, (out["action"].shape, out.get("raw_action_dim"))
    plan = out["sequence_plan"]
    rep["transform"] = {"image_size": out["image_size"].tolist(), "caption": out["ai_caption"],
                        "plan": {k: str(getattr(plan, k)) for k in ("has_text", "has_vision", "has_action", "condition_frame_indexes_vision") if hasattr(plan, k)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    print("DATA_SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
