#!/usr/bin/env bash
# Fetch the base checkpoint and the Wan2.2 VAE and convert the checkpoint to the DCP layout the trainer loads.
#   BASE_ROOT=/local/disk/cosmos_base bash scripts/prep_base.sh
# Needs: a cosmos-framework venv on PATH (huggingface_hub, cosmos_framework), ~70 GB of local disk, HF access.
# Produces: $BASE_ROOT/Cosmos3-Nano-dcp/ and $BASE_ROOT/wan22_vae/Wan2.2_VAE.pth
set -euo pipefail
: "${BASE_ROOT:?set BASE_ROOT to a local directory}"
NANO_REV=${NANO_REV:-411f42a8fdfb8c5b2583cb8786e0938f49796eaa}   # nvidia/Cosmos3-Nano revision used for the smokes
WAN_REV=${WAN_REV:-921dbaf3f1674a56f47e83fb80a34bac8a8f203e}     # Wan-AI/Wan2.2-TI2V-5B revision (Wan2.2_VAE.pth)
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
mkdir -p "$BASE_ROOT"
python - "$BASE_ROOT" "$NANO_REV" "$WAN_REV" <<'EOF'
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
root, nano_rev, wan_rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
info = HfApi().model_info("nvidia/Cosmos3-Nano", revision=nano_rev, files_metadata=True)
snapshot_download("nvidia/Cosmos3-Nano", revision=nano_rev, local_dir=str(root / "Cosmos3-Nano"), max_workers=4)
bad = [s.rfilename for s in info.siblings if (root / "Cosmos3-Nano" / s.rfilename).stat().st_size != s.size]
assert not bad, bad
vae = Path(hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth", revision=wan_rev, local_dir=str(root / "wan22_vae")))
h = hashlib.sha256()
with open(vae, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
print("Wan2.2_VAE.pth", vae.stat().st_size, "bytes sha256", h.hexdigest())  # expected 2818839170 bytes, 20eb7896...92b36
EOF
python -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path "$BASE_ROOT/Cosmos3-Nano" -o "$BASE_ROOT/Cosmos3-Nano-dcp"
test -d "$BASE_ROOT/Cosmos3-Nano-dcp/model"
echo "BASE_READY $BASE_ROOT"
