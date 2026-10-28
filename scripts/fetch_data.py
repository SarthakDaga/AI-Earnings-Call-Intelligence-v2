"""
scripts/fetch_data.py

Deployment helper. Transcripts are third-party content and are NOT in the repo or
the Docker image (see README, Data & licensing). To deploy, put your prebuilt
`data/` folder (earnings.db + chroma_db/) in a PRIVATE Hugging Face dataset repo
(free) and set HF_DATASET_REPO and HF_TOKEN as secrets. No-op if unset.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main() -> int:
    repo, token = os.getenv("HF_DATASET_REPO"), os.getenv("HF_TOKEN")
    if not repo:
        print("HF_DATASET_REPO not set; using whatever is already in data/")
        return 0
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(config.DATA_DIR))
    print(f"fetched {repo} -> {config.DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
