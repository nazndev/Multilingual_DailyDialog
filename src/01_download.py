"""
Download DailyDialog from roskoN/dailydialog.
Uses zip files directly. Writes to DATA_DIR/raw (from env).
"""
import io
import os
import sys
import traceback
import zipfile
from pathlib import Path
from itertools import zip_longest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs
from src.utils.logging_utils import setup_logger, banner, log_env_safely, timer

HF_ID = "roskoN/dailydialog"
SPLITS = {"train": "train.zip", "validation": "validation.zip", "test": "test.zip"}


def _get_hf_token():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _download_zips(token=None):
    from huggingface_hub import hf_hub_download
    out = {}
    for split, filename in SPLITS.items():
        path = hf_hub_download(repo_id=HF_ID, filename=filename, repo_type="dataset", token=token)
        out[split] = path
    return out


def _parse_zip(data_path: str):
    rows = []
    with zipfile.ZipFile(data_path) as zf:
        names = [str(n) for n in zf.namelist()]
        acts_file = next((f for f in names if "act" in f.lower()), None)
        emotions_file = next((f for f in names if "emotion" in f.lower()), None)
        utterances_file = next(
            (f for f in names if "act" not in f.lower() and "emotion" not in f.lower() and "dialogues" in f.lower()),
            None,
        )
        if not all([acts_file, emotions_file, utterances_file]):
            raise ValueError(f"Missing expected files in zip. Found: {names}")
        sentinel = object()
        with io.TextIOWrapper(zf.open(acts_file), encoding="utf-8") as af, \
             io.TextIOWrapper(zf.open(emotions_file), encoding="utf-8") as ef, \
             io.TextIOWrapper(zf.open(utterances_file), encoding="utf-8") as uf:
            for idx, (al, el, ul) in enumerate(zip_longest(af, ef, uf, fillvalue=sentinel)):
                if sentinel in (al, el, ul):
                    raise ValueError("Mismatched line counts in zip files")
                utts = [s.strip() for s in ul.strip().rstrip("__eou__").split("__eou__") if s.strip()]
                if not utts:
                    continue
                acts = (al or "").strip().split()
                emotions = (el or "").strip().split()
                n = len(utts)
                act_list = [
                    int(acts[i]) if i < len(acts) and str(acts[i]).strip().isdigit() else -1
                    for i in range(n)
                ]
                emotion_list = [
                    int(emotions[i]) if i < len(emotions) and str(emotions[i]).strip().isdigit() else -1
                    for i in range(n)
                ]
                rows.append({"dialogue": utts, "dialog_acts": act_list, "emotions": emotion_list})
    return rows


def main():
    logger = setup_logger("01_download")
    banner(logger, "Step 01: Download DailyDialog")
    log_env_safely(logger, ["DATA_DIR", "TARGET_LANGS", "SOURCE_LANG"])
    try:
        token = _get_hf_token()
        dirs = get_dirs()
        raw_dir = dirs["data"] / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info("output_dir=%s", raw_dir)
        with timer(logger, "download_zips"):
            paths = _download_zips(token=token)
        logger.info("input_sources=HuggingFace repo %s", HF_ID)
        for split, data_path in paths.items():
            with timer(logger, f"parse_and_save_{split}"):
                rows = _parse_zip(data_path)
                df = pd.DataFrame(rows)
                out = raw_dir / f"{split}.parquet"
                df.to_parquet(out, index=False)
            logger.info("saved split=%s records=%s path=%s", split, len(df), out)
        banner(logger, "Step 01: Done", char="-")
    except Exception:
        logger.exception("Step 01 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
