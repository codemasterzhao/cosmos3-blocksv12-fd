#!/usr/bin/env bash
# Download Cosmos3-Nano and the Wan2.2 VAE and convert Nano to the DCP layout the trainer loads (~70 GB local disk).
#   BASE_ROOT=/local/cosmos_base bash blocksv12/prep_base.sh
# Produces $BASE_ROOT/Cosmos3-Nano-dcp/ (BASE_CHECKPOINT_PATH) and $BASE_ROOT/wan22_vae/Wan2.2_VAE.pth (WAN_VAE_PATH).
set -euo pipefail
: "${BASE_ROOT:?set BASE_ROOT to a local directory}"
NANO_REV=411f42a8fdfb8c5b2583cb8786e0938f49796eaa   # nvidia/Cosmos3-Nano revision used for our smokes
WAN_REV=921dbaf3f1674a56f47e83fb80a34bac8a8f203e    # Wan-AI/Wan2.2-TI2V-5B revision holding Wan2.2_VAE.pth
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
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
assert h.hexdigest() == "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36", h.hexdigest()
print("downloads verified", flush=True)
EOF
python -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path "$BASE_ROOT/Cosmos3-Nano" -o "$BASE_ROOT/Cosmos3-Nano-dcp"
test -d "$BASE_ROOT/Cosmos3-Nano-dcp/model"
echo "BASE_READY BASE_CHECKPOINT_PATH=$BASE_ROOT/Cosmos3-Nano-dcp WAN_VAE_PATH=$BASE_ROOT/wan22_vae/Wan2.2_VAE.pth"
