#!/usr/bin/env python3
"""Apply the Blocks V12 overlay to a cosmos-framework checkout (written against commit 5e67049).

    python patches/apply_overlay.py /path/to/cosmos-framework [--check]

Copies the three new files from ../cosmos_framework/ into the checkout and applies nine exact-match edits in
seven files (each anchor must occur exactly once; nothing is written if any anchor is missing):

  1-2. domain registry: embodiment "blocksv12" -> domain id 22, raw action dim 8
       (22 is a free row at 5e67049; on newer main pick a free id and change DOMAIN_ID below)
  3.   configs/base/config.py: import the new experiment module so Hydra registers it
  4.   GenerationDataClean: two optional fields rerope_P_T / rerope_P_inv (ReRoPE camera transforms)
  5.   OmniMoTModel.get_data_and_condition: pass data_batch["rerope_P_T"/"rerope_P_inv"] into GenerationDataClean
  6.   pack_input_sequence: carry the two lists on the finalized PackedSequence
  7.   Cosmos3VFMNetwork.forward: attach per-generation-token matrices to the pack (before any CP sharding)
  8.   PackedAttentionMoT.forward: apply ReRoPE to the generation-side Q/K after their RoPE

Patches 4-8 are inert unless the dataset is built with return_camera_poses=True (env REROPE=1).
Run it once, on a clean checkout: it is not idempotent (a second run stops at the first consumed anchor).
"""
from __future__ import annotations

import argparse, shutil, sys
from pathlib import Path

DOMAIN_ID = 22
HERE = Path(__file__).resolve().parent.parent

PATCHES = [
    ("data/generator/action/domain_utils.py", '    "drawanything": 21,\n', f'    "blocksv12": {DOMAIN_ID},\n', "after"),
    ("data/generator/action/domain_utils.py", '    "drawanything": 3,\n', '    "blocksv12": 8,\n', "after"),
    ("configs/base/config.py",
     "    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_droid_posttrain  # noqa: F401\n",
     "    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_blocksv12_posttrain  # noqa: F401\n", "after"),
    ("model/generator/utils/data_and_condition.py",
     "    control_weights: list[list[float]] | None = None\n",
     "    rerope_P_T: list[torch.Tensor] | None = None  # ReRoPE: per-sample [F_lat, V, 4, 4]\n"
     "    rerope_P_inv: list[torch.Tensor] | None = None\n", "after"),
    ("model/generator/omni_mot_model.py",
     "            control_weights=control_weights,\n        )\n",
     "            rerope_P_T=data_batch.get(\"rerope_P_T\"),\n            rerope_P_inv=data_batch.get(\"rerope_P_inv\"),\n", "before"),
    ("data/generator/sequence_packing/packers.py",
     "    return seq_builder.finalize(\n        gen_data_clean=gen_data_clean,\n    )\n",
     "    packed_seq_out = seq_builder.finalize(gen_data_clean=gen_data_clean)\n"
     "    packed_seq_out.rerope_P_T = getattr(gen_data_clean, \"rerope_P_T\", None)  # ReRoPE\n"
     "    packed_seq_out.rerope_P_inv = getattr(gen_data_clean, \"rerope_P_inv\", None)\n"
     "    return packed_seq_out\n", "replace"),
    ("model/generator/mot/cosmos3_vfm_network.py",
     "        input_pack, packed_position_ids = get_context_parallel_sharded_sequence(\n",
     "        from cosmos_framework.model.generator.mot.rerope import attach_rerope  # ReRoPE\n"
     "        attach_rerope(input_pack, packed_seq, bool(sequence_shard_parallel_dims is not None and sequence_shard_parallel_dims.cp_enabled))\n",
     "before"),
    ("model/generator/mot/unified_mot.py",
     "from cosmos_framework.utils import log\n",
     "from cosmos_framework.model.generator.mot.rerope import apply_rerope  # ReRoPE\n", "after"),
    ("model/generator/mot/unified_mot.py",
     "            unsqueeze_dim=1,\n        )  # q_gen_: [N_gen,num_heads,head_dim], k_gen_: [N_gen,num_kv_heads,head_dim]\n",
     "        if pack.get(\"rerope_P_T\") is not None:  # ReRoPE on the generation-side Q/K only\n"
     "            q_gen_, k_gen_ = apply_rerope(q_gen, k_gen, q_gen_, k_gen_, pack[\"rerope_P_T\"], pack[\"rerope_P_inv\"])\n",
     "after"),
]
NEW_FILES = [
    "data/generator/action/datasets/blocksv12_fd_dataset.py",
    "configs/base/experiment/action/posttrain_config/action_fd_blocksv12_posttrain.py",
    "model/generator/mot/rerope.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("framework_root", type=Path, help="cosmos-framework checkout (contains cosmos_framework/)")
    ap.add_argument("--check", action="store_true", help="verify every anchor occurs exactly once; change nothing")
    a = ap.parse_args()
    cf = a.framework_root / "cosmos_framework"
    assert (cf / "__init__.py").is_file(), f"not a cosmos-framework checkout: {a.framework_root}"
    problems = []
    for path, old, _, _ in PATCHES:
        n = (cf / path).read_text().count(old) if (cf / path).is_file() else -1
        if n != 1:
            problems.append(f"{path}: anchor count {n} (expected 1) for {old.strip()[:70]!r}")
    if problems:
        print("\n".join(problems))
        print("refusing to patch: the checkout differs from 5e67049 at the lines above; port the hooks by hand")
        return 1
    if a.check:
        print("all anchors found exactly once; nothing changed")
        return 0
    for rel in NEW_FILES:
        dst = cf / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "cosmos_framework" / rel, dst)
        print("copied", rel)
    for path, old, new, mode in PATCHES:
        p = cf / path
        s = p.read_text()
        p.write_text(s.replace(old, {"after": old + new, "before": new + old, "replace": new}[mode]))
        print("patched", path)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
