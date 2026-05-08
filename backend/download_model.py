"""Pre-pull model weights at Docker build time so cold-starts are fast.

RingLeader gotcha #6: don't download weights inside a request handler.
First-PDF stalls for 60s while HuggingFace pulls 700MB - users assume broken.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# (1) RxnIM Pix2Seq detector - the 432MB checkpoint hosted on the HF Space repo.
SPACE_REPO = "CYF200127/RxnIM"
RXN_CKPT_FILE = "rxn/model/model.ckpt"
RXN_CKPT_DEST = Path("/app/rxn/model/model.ckpt")

# (2) MolScribe atom-level OCSR ckpt
MOLSCRIBE_REPO = "yujieq/MolScribe"
MOLSCRIBE_FILE = "swin_base_char_aux_1m680k.pth"


def main():
    print(f"[download_model] pulling {RXN_CKPT_FILE} from {SPACE_REPO} ...", flush=True)
    src = hf_hub_download(repo_id=SPACE_REPO, filename=RXN_CKPT_FILE, repo_type="space")
    RXN_CKPT_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, RXN_CKPT_DEST)
    sz = RXN_CKPT_DEST.stat().st_size
    print(f"[download_model] -> {RXN_CKPT_DEST} ({sz / 1_000_000:.1f} MB)", flush=True)
    if sz < 400_000_000 or sz > 500_000_000:
        print(f"[download_model] WARN: size {sz} outside expected 432MB +/- range", flush=True)

    print(f"[download_model] pulling {MOLSCRIBE_FILE} from {MOLSCRIBE_REPO} ...", flush=True)
    src = hf_hub_download(repo_id=MOLSCRIBE_REPO, filename=MOLSCRIBE_FILE)
    sz = Path(src).stat().st_size
    print(f"[download_model] MolScribe at {src} ({sz / 1_000_000:.1f} MB)", flush=True)

    # Pre-warm easyocr detection model (~110MB) so first request doesn't pay the
    # download cost.  easyocr stores in ~/.EasyOCR/ which we mount as a volume.
    print("[download_model] pre-warming easyocr ...", flush=True)
    import easyocr  # noqa: F401
    reader = easyocr.Reader(['en'], gpu=False)
    del reader
    print("[download_model] done.", flush=True)


if __name__ == "__main__":
    main()
