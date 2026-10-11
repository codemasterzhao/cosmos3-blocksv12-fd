"""ReRoPE for the Cosmos3 MoT: relative camera projective transforms on the lowest-frequency RoPE bins.

Graft of "Repurposing RoPE for Relative Camera Control" (arXiv 2602.08068, PRoPE operator arXiv 2507.10496)
as ported for FlexFwd (flex-pi ``camera_pe.py``), adapted to Qwen3-VL's interleaved 3-D MRoPE:

* head_dim 128 = 64 complex bins, bin ``j`` is channel pair ``(j, j+64)`` (rotate_half layout) with
  frequency ``theta^(-j/64)``; axes interleave T,H,W over bins 0..59 and T on 60..63.  We repurpose the
  8 globally lowest-frequency bins ``56..63`` (16 real channels, 4 projective groups).  Within a sample
  they rotate by < 1e-4 rad (T <= 6 latent frames, H <= 18, W <= 10 patches) and the 15000 text/video
  temporal margin moves them by <= 0.016 rad, so masking them is a no-op for the pretrained attention.
* For generation-side vision tokens the RoPE rotation on those channels is undone (pre-RoPE values are
  restored) and ``Q <- P^T Q``, ``K <- P^{-1} K`` is applied per 4-channel group, ``P = lift(K_norm) @ T_cw``.
  Action tokens and all understanding (text) tokens get the identity.  Logits between two vision tokens
  then see only ``P_i P_j^{-1}``: relative camera geometry, world-frame invariant.
"""
from __future__ import annotations

import torch

HEAD_DIM = 128
N_BINS = 8
BINS = list(range(HEAD_DIM // 2 - N_BINS, HEAD_DIM // 2))  # [56, ..., 63]


def channel_groups() -> torch.Tensor:
    """[4, 4] channel indices: bins (56,57),(58,59),(60,61),(62,63) -> [j, j+64, j+1, j+65]."""
    half = HEAD_DIM // 2
    return torch.tensor([[j, j + half, j + 1, j + 1 + half] for j in BINS[::2]], dtype=torch.long)


# ---- projective transforms (torch ports of the pinned FlexFwd / PRoPE math) ---------------------------

def invert_SE3(t: torch.Tensor) -> torch.Tensor:
    rot_inv = t[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(t)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_inv, t[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def lift_K(k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(k.shape[:-2] + (4, 4), device=k.device, dtype=k.dtype)
    out[..., :3, :3] = k
    out[..., 3, 3] = 1.0
    return out


def invert_K(k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(k)
    out[..., 0, 0] = 1.0 / k[..., 0, 0]
    out[..., 1, 1] = 1.0 / k[..., 1, 1]
    out[..., 0, 2] = -k[..., 0, 2] / k[..., 0, 0]
    out[..., 1, 2] = -k[..., 1, 2] / k[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def slot_K_from_native(k_native: torch.Tensor, native_hw, slot_hw) -> torch.Tensor:
    """Antialiased-bilinear resize maps pixel centres as x' = s(x+0.5)-0.5 (align_corners=False)."""
    sx, sy = slot_hw[1] / native_hw[1], slot_hw[0] / native_hw[0]
    out = k_native.to(torch.float64).clone()
    out[..., 0, 0] *= sx
    out[..., 0, 2] = sx * (out[..., 0, 2] + 0.5) - 0.5
    out[..., 1, 1] *= sy
    out[..., 1, 2] = sy * (out[..., 1, 2] + 0.5) - 0.5
    return out.to(k_native.dtype)


def normalize_intrinsics(k: torch.Tensor, slot_hw) -> torch.Tensor:
    """PRoPE normalization: fx/W, fy/H, cx/W - 0.5, cy/H - 0.5."""
    h, w = float(slot_hw[0]), float(slot_hw[1])
    out = torch.zeros_like(k)
    out[..., 0, 0] = k[..., 0, 0] / w
    out[..., 1, 1] = k[..., 1, 1] / h
    out[..., 0, 2] = k[..., 0, 2] / w - 0.5
    out[..., 1, 2] = k[..., 1, 2] / h - 0.5
    out[..., 2, 2] = 1.0
    return out


def projective_transforms(t_cw: torch.Tensor, k_norm: torch.Tensor, translation_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(P^T, P^{-1}) for T_cw [F, V, 4, 4] and K_norm [V, 3, 3]; translation divided by ``translation_scale``."""
    t = t_cw.to(torch.float32).clone()
    t[..., :3, 3] = t[..., :3, 3] / float(translation_scale)
    kl = lift_K(k_norm.to(torch.float32)).unsqueeze(0).expand(t.shape[0], -1, -1, -1)
    p = kl @ t
    p_inv = invert_SE3(t) @ lift_K(invert_K(k_norm.to(torch.float32))).unsqueeze(0).expand_as(kl)
    return p.transpose(-1, -2).contiguous(), p_inv.contiguous()


# ---- per-token assignment ------------------------------------------------------------------------------

def gen_token_transforms(n_total: int, full_indices: torch.Tensor, vision_indexes: torch.Tensor,
                         token_shapes: list[tuple[int, int, int]], rows_per_view: int,
                         p_t: list[torch.Tensor], p_inv: list[torch.Tensor], n_gen: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-item (frame, view) matrices to the pack's generation tokens.

    ``vision_indexes`` are the packed positions of all vision tokens, item by item in (t, h, w) order;
    ``token_shapes[i] = (T, H, W)`` for item i; a token at row h belongs to view ``h // rows_per_view``.
    ``full_indices`` are the packed positions of the generation sequence (vision + action tokens, in
    pack order); positions without a vision transform (action tokens, padding) get the identity.
    """
    assert len(token_shapes) == len(p_t) == len(p_inv), (len(token_shapes), len(p_t), len(p_inv))
    device = full_indices.device
    eye = torch.eye(4, device=device, dtype=torch.float32)
    pt_full = eye.repeat(n_total, 1, 1)
    pinv_full = eye.repeat(n_total, 1, 1)
    offset = 0
    for (t_lat, h_p, w_p), pt_i, pinv_i in zip(token_shapes, p_t, p_inv):
        n = t_lat * h_p * w_p
        idx = vision_indexes[offset:offset + n]
        offset += n
        pt_i = pt_i.to(device=device, dtype=torch.float32)
        pinv_i = pinv_i.to(device=device, dtype=torch.float32)
        assert pt_i.shape[0] == t_lat and h_p % rows_per_view == 0 and pt_i.shape[1] == h_p // rows_per_view, (pt_i.shape, t_lat, h_p)
        local = torch.arange(n, device=device)
        t = local // (h_p * w_p)
        v = (local // w_p) % h_p // rows_per_view
        pt_full[idx] = pt_i[t, v]
        pinv_full[idx] = pinv_i[t, v]
    assert offset == vision_indexes.numel(), (offset, vision_indexes.numel())
    pt_gen, pinv_gen = pt_full[full_indices], pinv_full[full_indices]
    pad = n_gen - pt_gen.shape[0]
    assert pad >= 0, (n_gen, pt_gen.shape[0])
    if pad:
        pt_gen = torch.cat([pt_gen, eye.repeat(pad, 1, 1)])
        pinv_gen = torch.cat([pinv_gen, eye.repeat(pad, 1, 1)])
    return pt_gen, pinv_gen


def attach_rerope(input_pack: dict, packed_seq, cp_enabled: bool, rows_per_view: int = 6) -> None:
    """Store per-generation-token (P^T, P^{-1}) on the pack; ``from_mode_splits`` copies them to every layer."""
    p_t = getattr(packed_seq, "rerope_P_T", None)
    if p_t is None:
        return
    assert not cp_enabled, "ReRoPE pack tensors are not sharded for context parallel"
    vision = packed_seq.vision
    n_gen = input_pack["full_only_seq"].shape[0]
    # packed batches carry per-sample tensors as [1, F, V, 4, 4]; drop the batch dim
    p_t = [p.reshape(-1, p.shape[-3], 4, 4) for p in p_t]
    p_inv = [p.reshape(-1, p.shape[-3], 4, 4) for p in packed_seq.rerope_P_inv]
    pt, pinv = gen_token_transforms(packed_seq.sequence_length, input_pack["_full_indices"], vision.sequence_indexes,
                                    list(vision.token_shapes), rows_per_view, p_t, p_inv, n_gen)
    input_pack["rerope_P_T"] = pt
    input_pack["rerope_P_inv"] = pinv


# ---- attention-side application --------------------------------------------------------------------

def apply_rerope(q_pre: torch.Tensor, k_pre: torch.Tensor, q_rope: torch.Tensor, k_rope: torch.Tensor,
                 p_t: torch.Tensor, p_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q_*: [N, H, 128], k_*: [N, Hk, 128], p_*: [N, 4, 4]. Returns RoPE'd q/k with the 16 repurposed channels
    replaced by P^T q_pre / P^{-1} k_pre (per 4-channel group), computed in float32."""
    idx = channel_groups().reshape(-1).to(q_pre.device)
    n, h, _ = q_pre.shape
    q_span = q_pre[..., idx].to(torch.float32).reshape(n, h, 4, 4)
    k_span = k_pre[..., idx].to(torch.float32).reshape(n, k_pre.shape[1], 4, 4)
    q_span = torch.einsum("nij,nhgj->nhgi", p_t.to(torch.float32), q_span)
    k_span = torch.einsum("nij,nhgj->nhgi", p_inv.to(torch.float32), k_span)
    q_out, k_out = q_rope.clone(), k_rope.clone()
    q_out[..., idx] = q_span.reshape(n, h, 16).to(q_rope.dtype)
    k_out[..., idx] = k_span.reshape(n, k_pre.shape[1], 16).to(k_rope.dtype)
    return q_out, k_out
