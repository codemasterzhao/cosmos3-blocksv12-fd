"""CPU tests for rerope.py (run from the repository root): channel plan, P=I no-op, world-frame
invariance of the transformed logits, per-token assignment on a synthetic pack, and (with --root) the dataset's
camera-pose output."""
from __future__ import annotations

import argparse, json, sys

import torch

from cosmos_framework.model.generator.mot import rerope as R


def rand_se3(g):
    a = torch.randn(3, generator=g)
    a = a / a.norm()
    th = torch.rand((), generator=g) * 3
    k = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rot = torch.eye(3) + torch.sin(th) * k + (1 - torch.cos(th)) * (k @ k)
    t = torch.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = torch.randn(3, generator=g)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="/results/rerope_tests.json")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    rep = {}

    groups = R.channel_groups()
    idx = groups.reshape(-1)
    assert idx.numel() == 16 and idx.unique().numel() == 16
    inv_freq = 5e6 ** (-torch.arange(64).double() / 64)
    assert sorted(R.BINS) == inv_freq.argsort()[:8].sort().values.tolist(), "not the 8 lowest-frequency bins"
    for row in groups.tolist():
        assert row[1] == row[0] + 64 and row[2] == row[0] + 1 and row[3] == row[2] + 64, row
    rep["groups"] = groups.tolist()

    # P = I reproduces the RoPE-masked baseline exactly
    n, h, hk = 7, 4, 2
    q_pre, k_pre = torch.randn(n, h, 128, generator=g), torch.randn(n, hk, 128, generator=g)
    q_rope, k_rope = torch.randn(n, h, 128, generator=g), torch.randn(n, hk, 128, generator=g)
    eye = torch.eye(4).repeat(n, 1, 1)
    q1, k1 = R.apply_rerope(q_pre, k_pre, q_rope, k_rope, eye, eye)
    q_ref, k_ref = q_rope.clone(), k_rope.clone()
    q_ref[..., idx], k_ref[..., idx] = q_pre[..., idx], k_pre[..., idx]
    assert torch.equal(q1, q_ref) and torch.equal(k1, k_ref)
    mask = torch.ones(128, dtype=torch.bool)
    mask[idx] = False
    assert torch.equal(q1[..., mask], q_rope[..., mask]), "untouched channels changed"

    # world-frame invariance: left-multiplying every T_cw by one rigid W leaves q^T P_i P_j^{-1} k unchanged
    k_norm = torch.tensor([[[0.8, 0, 0.0], [0, 1.2, -0.1], [0, 0, 1.0]]] * 2)
    t_cw = torch.stack([torch.stack([rand_se3(g) for _ in range(2)]) for _ in range(3)])  # [F=3, V=2, 4, 4]
    w = rand_se3(g)
    pt_a, pinv_a = R.projective_transforms(t_cw, k_norm, 2.0)
    pt_b, pinv_b = R.projective_transforms(t_cw @ w, k_norm, 2.0)
    assert torch.allclose(pt_a.transpose(-1, -2) @ pinv_a, torch.eye(4).expand(3, 2, 4, 4), atol=1e-5), "P P^-1 != I"
    tok_pt_a, tok_pinv_a = pt_a.reshape(6, 4, 4), pinv_a.reshape(6, 4, 4)
    tok_pt_b, tok_pinv_b = pt_b.reshape(6, 4, 4), pinv_b.reshape(6, 4, 4)
    q6, k6 = torch.randn(6, 1, 128, generator=g), torch.randn(6, 1, 128, generator=g)
    qa, ka = R.apply_rerope(q6, k6, q6, k6, tok_pt_a, tok_pinv_a)
    qb, kb = R.apply_rerope(q6, k6, q6, k6, tok_pt_b, tok_pinv_b)
    logits_a = torch.einsum("ihc,jhc->ij", qa[..., idx], ka[..., idx])
    logits_b = torch.einsum("ihc,jhc->ij", qb[..., idx], kb[..., idx])
    assert torch.allclose(logits_a, logits_b, atol=1e-3, rtol=1e-4), (logits_a - logits_b).abs().max()
    assert not torch.allclose(qa, qb, atol=1e-3), "transformed q should depend on the absolute pose"
    rep["world_frame_invariance_max_abs"] = float((logits_a - logits_b).abs().max())

    # per-token assignment on a synthetic pack: text(5) | vision(6x18x10) | action(20) | text(3) | vision | action
    t_lat, h_p, w_p = 6, 18, 10
    n_vis = t_lat * h_p * w_p
    sample = [5, n_vis, 20]
    full_idx, vis_idx, cur = [], [], 0
    for _ in range(2):
        cur += sample[0]
        vis_idx.append(torch.arange(cur, cur + n_vis)); full_idx.append(torch.arange(cur, cur + n_vis + 20)); cur += n_vis + 20
    n_total = cur
    p_t = [torch.randn(t_lat, 3, 4, 4, generator=g) for _ in range(2)]
    p_inv = [torch.randn(t_lat, 3, 4, 4, generator=g) for _ in range(2)]
    pt_gen, pinv_gen = R.gen_token_transforms(n_total, torch.cat(full_idx), torch.cat(vis_idx), [(t_lat, h_p, w_p)] * 2, 6, p_t, p_inv, 2 * (n_vis + 20) + 3)
    assert pt_gen.shape == (2 * (n_vis + 20) + 3, 4, 4)
    for s in range(2):
        base = s * (n_vis + 20)
        for (t, hh, ww) in [(0, 0, 0), (2, 5, 9), (3, 6, 0), (5, 17, 9), (4, 12, 3)]:
            pos = base + t * h_p * w_p + hh * w_p + ww
            assert torch.equal(pt_gen[pos], p_t[s][t, hh // 6]) and torch.equal(pinv_gen[pos], p_inv[s][t, hh // 6]), (s, t, hh, ww)
        assert torch.equal(pt_gen[base + n_vis: base + n_vis + 20], torch.eye(4).repeat(20, 1, 1)), "action tokens must be identity"
    assert torch.equal(pt_gen[-3:], torch.eye(4).repeat(3, 1, 1)), "padding must be identity"
    rep["gather"] = "ok"

    if a.root:
        from cosmos_framework.data.generator.action.datasets.blocksv12_fd_dataset import BlocksV12FDDataset
        ds = BlocksV12FDDataset(root=a.root, return_camera_poses=True)
        for i in (0, len(ds) // 2):
            s = ds[i]
            pt, pinv = s["rerope_P_T"], s["rerope_P_inv"]
            assert pt.shape == (6, 3, 4, 4) and pinv.shape == (6, 3, 4, 4) and torch.isfinite(pt).all() and torch.isfinite(pinv).all()
            assert torch.allclose(pt.transpose(-1, -2) @ pinv, torch.eye(4).expand(6, 3, 4, 4), atol=1e-4)
            for v in (0, 1):
                assert torch.allclose(pt[:, v], pt[:1, v].expand(6, 4, 4), atol=1e-6), f"static view {v} varies over latent frames"
            rep.setdefault("dataset", []).append({"idx": i, "wrist_frame_to_frame_max_change": float((pt[1:, 2] - pt[:-1, 2]).abs().max()),
                                                  "P_T_00": pt[0, 0].tolist()})
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("REROPE_TESTS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
