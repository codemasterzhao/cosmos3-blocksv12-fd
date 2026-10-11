#!/usr/bin/env bash
# Download the training data (public Hugging Face dataset, 29 GB) and check every file's size against the Hub.
#   DATA_ROOT=/local/blocksv12 bash blocksv12/download_data.sh      -> use DATASET_PATH=$DATA_ROOT
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to a local directory}"
python - "$DATA_ROOT" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, rev, root = "haibozzz/blocks-v12-4task-lerobot-v30-camgeom", "6fdef09b9b4654d125838641fe8746d350c08bb6", Path(sys.argv[1])
snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=str(root), max_workers=8)
files = [s for s in HfApi().dataset_info(repo, revision=rev, files_metadata=True).siblings if s.rfilename.startswith("train/")]
bad = [s.rfilename for s in files if (root / s.rfilename).stat().st_size != s.size]
assert not bad, bad[:20]
print(f"DATA_READY {len(files)} train files, {sum(s.size for s in files):,} bytes in {root}")
EOF
