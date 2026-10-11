#!/usr/bin/env bash
# Download the training data (public Hugging Face dataset, 29 GB, ~10,800 files) and check every file's size.
#   hf auth login        # or export HF_TOKEN=...; required, anonymous downloads of this many files are blocked
#   DATA_ROOT=/local/blocksv12 bash blocksv12/download_data.sh      -> use DATASET_PATH=$DATA_ROOT
# Hugging Face rate-limits requests per 5 minutes (free account: 5,000 file requests). The script disables Xet
# so each file costs only file requests, and on HTTP 429 it waits for the window to reset and resumes; finished
# files are not requested again. Expect roughly half an hour on a free account.
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to a local directory}"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
python - "$DATA_ROOT" <<'PY'
import re, sys, time
from pathlib import Path
from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
if get_token() is None:
    sys.exit("Log in to Hugging Face first (hf auth login, or export HF_TOKEN).")
repo, rev, root = "haibozzz/blocks-v12-4task-lerobot-v30-camgeom", "6fdef09b9b4654d125838641fe8746d350c08bb6", Path(sys.argv[1])
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=str(root), max_workers=8)
        break
    except HfHubHTTPError as e:
        if e.response is None or e.response.status_code != 429:
            raise
        m = re.search(r"t=(\d+)", e.response.headers.get("RateLimit", ""))
        wait = int(m.group(1)) + 5 if m else 300
        print(f"Hugging Face rate limit (HTTP 429); resuming in {wait} s", flush=True)
        time.sleep(wait)
else:
    sys.exit("still rate-limited after 60 waits")
files = [s for s in HfApi().dataset_info(repo, revision=rev, files_metadata=True).siblings if s.rfilename.startswith("train/")]
bad = [s.rfilename for s in files if (root / s.rfilename).stat().st_size != s.size]
assert not bad, bad[:20]
print(f"DATA_READY {len(files)} train files, {sum(s.size for s in files):,} bytes in {root}")
PY
