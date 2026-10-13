#!/usr/bin/env bash
# Download the data (public Hugging Face dataset: train/ 29 GB + val/ 3.3 GB, ~12,000 files) and check every file's size.
#   hf auth login        # or export HF_TOKEN=...; required, anonymous downloads of this many files are blocked
#   DATA_ROOT=/local/blocksv12 bash blocksv12/download_data.sh      -> use DATASET_PATH=$DATA_ROOT
# Hugging Face rate-limits requests per 5 minutes (free account: 5,000 file requests). The script disables Xet
# so each file costs only file requests, and on HTTP 429 it waits for the window to reset and resumes; finished
# files are not requested again. Measured 2026-10-11: ~20 min for the 10,812 train files on a free account.
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to a local directory}"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
python - "$DATA_ROOT" <<'PY'
import re, sys, time
from pathlib import Path
from huggingface_hub import HfApi, get_token, snapshot_download
if get_token() is None:
    sys.exit("Log in to Hugging Face first (hf auth login, or export HF_TOKEN).")
repo, rev, root = "haibozzz/blocks-v12-4task-lerobot-v30-camgeom", "338c80cada56693078b05c69111b153ae641c13d", Path(sys.argv[1])
def rate_limit_wait(e):  # seconds until the window resets if a 429 is anywhere in the exception chain, else None
    while e is not None:
        r = getattr(e, "response", None)
        if getattr(r, "status_code", None) == 429:
            m = re.search(r"t=(\d+)", r.headers.get("RateLimit", ""))
            return int(m.group(1)) + 5 if m else 300
        e = e.__cause__ or e.__context__
    return None
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=str(root), max_workers=8)
        break
    except Exception as e:
        wait = rate_limit_wait(e)
        if wait is None:
            raise
        print(f"Hugging Face rate limit (HTTP 429); resuming in {wait} s", flush=True)
        time.sleep(wait)
else:
    sys.exit("still rate-limited after 60 waits")
files = [s for s in HfApi().dataset_info(repo, revision=rev, files_metadata=True).siblings if s.rfilename.startswith(("train/", "val/"))]
bad = [s.rfilename for s in files if (root / s.rfilename).stat().st_size != s.size]
assert not bad, bad[:20]
print(f"DATA_READY {len(files)} files (train/ + val/), {sum(s.size for s in files):,} bytes in {root}")
PY
