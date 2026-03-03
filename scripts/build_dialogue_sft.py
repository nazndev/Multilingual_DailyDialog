#!/usr/bin/env python3
"""Build dialogue-level SFT JSONL (one row per dialogue).

Input: existing turn-level SFT JSONL (one row per assistant turn), e.g.
  data/sft/multilingual_1000/train.jsonl

Output: one row per dialogue_id by selecting a single representative turn.
Default selection: last assistant turn (max turn_index).

This is useful when you want an *exact* 800/200 example split by dialogue,
not multiple turn-rows per dialogue.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_last_turn(rows: list[dict]) -> dict:
    # Prefer max numeric turn_index; fall back to last row order.
    def key(r: dict):
        ti = r.get("turn_index")
        try:
            return int(ti)
        except Exception:
            return -1

    return max(rows, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--strategy",
        choices=["last"],
        default="last",
        help="How to choose one SFT row per dialogue_id",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    by_dialogue: dict[str, list[dict]] = defaultdict(list)
    for rec in iter_jsonl(in_path):
        did = rec.get("dialogue_id")
        if not did:
            continue
        by_dialogue[str(did)].append(rec)

    if args.strategy == "last":
        out_rows = [select_last_turn(rows) for _, rows in sorted(by_dialogue.items(), key=lambda x: x[0])]
    else:
        raise SystemExit(f"Unknown strategy: {args.strategy}")

    # Sanity: ensure 1 row per dialogue
    if len(out_rows) != len(by_dialogue):
        raise SystemExit("Internal error: output rows != unique dialogue count")

    write_jsonl(out_path, out_rows)
    print(f"input_rows={sum(len(v) for v in by_dialogue.values())} unique_dialogues={len(by_dialogue)}")
    print(f"wrote_rows={len(out_rows)} to {out_path}")


if __name__ == "__main__":
    main()
