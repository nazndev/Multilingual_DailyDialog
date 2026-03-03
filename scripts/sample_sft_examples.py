#!/usr/bin/env python3
"""Sample a fixed-size subset of SFT JSONL examples (e.g., 1000 Bangla rows).

This is used to create the professor-requested 1000-example dataset:
- 800 train examples
- 200 test examples

Input records are JSON objects with at least:
  - lang
  - messages (list)

Extra fields (dialogue_id, turn_index, emotion_at_turn, act_at_turn, etc.) are preserved.

Usage:
  /path/to/python scripts/sample_sft_examples.py \
    --inputs data/sft/multilingual_1000/train.jsonl data/sft/multilingual_1000/validation.jsonl data/sft/multilingual_1000/test.jsonl \
    --lang bn --total 1000 --train 800 --seed 42 \
    --out-dir data/sft/sampled_1000_bn
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def iter_jsonl(paths: list[Path]) -> Iterable[dict]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input SFT JSONL files")
    ap.add_argument("--lang", default="bn")
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.train >= args.total:
        raise SystemExit("--train must be < --total")

    in_paths = [Path(x) for x in args.inputs]
    missing = [str(p) for p in in_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all candidate records for the language.
    candidates: list[dict] = []
    for rec in iter_jsonl(in_paths):
        if rec.get("lang") != args.lang:
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        candidates.append(rec)

    if len(candidates) < args.total:
        raise SystemExit(
            f"Not enough examples for lang={args.lang}. Have {len(candidates)}, need {args.total}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked = candidates[: args.total]

    train_recs = picked[: args.train]
    test_recs = picked[args.train :]

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    with open(train_path, "w", encoding="utf-8") as w:
        for rec in train_recs:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as w:
        for rec in test_recs:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote train={len(train_recs)} to {train_path}")
    print(f"Wrote test={len(test_recs)} to {test_path}")


if __name__ == "__main__":
    main()
