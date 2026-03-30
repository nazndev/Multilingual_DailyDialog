import hashlib
import json
import sys
import traceback
from pathlib import Path

import argparse
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_env_safely, timer, summarize_jsonl


def load_cfg(path: Optional[str]) -> dict:
    if not path:
        return {}
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm(t: str) -> str:
    return " ".join((t or "").strip().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Preprocess config (raw_dir, processed_dir, max_dialogues_per_split)")
    args = ap.parse_args()

    logger = setup_logger("02_preprocess")
    banner(logger, "Step 02: Preprocess")
    log_env_safely(logger, ["DATA_DIR", "TARGET_LANGS"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        raw_dir = resolve_path(cfg.get("raw_dir", "raw"), dirs["data"])
        processed_dir = resolve_path(cfg.get("processed_dir", "processed"), dirs["data"])
        max_per_split = cfg.get("max_dialogues_per_split")
        processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "input_dir=%s output_dir=%s max_dialogues_per_split=%s",
            raw_dir,
            processed_dir,
            max_per_split,
        )
        def _cap_for(split_name: str):
            if max_per_split is None:
                return None
            # Back-compat: integer applies to all splits.
            if isinstance(max_per_split, int):
                return int(max_per_split)
            # YAML may load numbers as int/float; accept numeric.
            if isinstance(max_per_split, (float,)):
                return int(max_per_split)
            # New: mapping per split, e.g. {train: 800, validation: 0, test: 200}
            if isinstance(max_per_split, dict):
                v = max_per_split.get(split_name)
                if v is None:
                    return None
                return int(v)
            raise ValueError(
                "max_dialogues_per_split must be int or mapping like {train: 800, test: 200}"
            )

        for split in ["train", "validation", "test"]:
            p = raw_dir / f"{split}.parquet"
            if not p.exists():
                logger.warning("missing input path=%s; run 01_download.py", p)
                continue
            with timer(logger, f"preprocess_{split}"):
                df = pd.read_parquet(p)
                if "dialogue" not in df.columns:
                    raise ValueError(f"No 'dialogue' column. Columns: {list(df.columns)}")
                logger.info("input path=%s records=%s", p, len(df))
                out = processed_dir / f"{split}_en.jsonl"
                written = 0
                cap = _cap_for(split)
                with open(out, "w", encoding="utf-8") as w:
                    for i, row in df.iterrows():
                        if cap is not None and written >= int(cap):
                            break
                        turns = [norm(x) for x in row["dialogue"]]
                        turns = [t for t in turns if t]
                        if not turns:
                            continue
                        n = len(turns)
                        _acts = row.get("dialog_acts")
                        acts = [int(x) for x in list(_acts)[:n]] if "dialog_acts" in df.columns and _acts is not None and hasattr(_acts, "__iter__") and not isinstance(_acts, str) else [-1] * n
                        _emos = row.get("emotions")
                        emotions = [int(x) for x in list(_emos)[:n]] if "emotions" in df.columns and _emos is not None and hasattr(_emos, "__iter__") and not isinstance(_emos, str) else [-1] * n
                        if len(acts) < n:
                            acts = acts + [-1] * (n - len(acts))
                        if len(emotions) < n:
                            emotions = emotions + [-1] * (n - len(emotions))
                        rec = {
                            "dialogue_id": sha1(f"{split}:{i}"),
                            "turns_en": turns,
                            "num_turns": n,
                            "dialog_acts": acts[:n],
                            "emotions": emotions[:n],
                        }
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                logger.info("output path=%s records=%s", out, written)
                summarize_jsonl(logger, out)
        banner(logger, "Step 02: Done", char="-")
    except Exception:
        logger.exception("Step 02 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
