#!/usr/bin/env python3
"""Validate a manually-translated JSONL against the original processed English JSONL.

Checks:
- Same number of records (optional if --allow-mismatch)
- Same dialogue_id sequence
- Required keys exist (turns_en, turns_bn, num_turns, translation_meta)
- turns_bn length matches turns_en length and num_turns

Example:
  python scripts/validate_manual_translation.py \
    --source data/processed_1000/train_en.jsonl \
    --translated data/translated_manual_1000/train.jsonl \
    --lang bn
"""

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield line_no, json.loads(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Processed EN JSONL (e.g., train_en.jsonl)")
    ap.add_argument("--translated", required=True, help="Manually translated JSONL (e.g., train.jsonl)")
    ap.add_argument("--lang", default="bn", help="Target lang key suffix (default: bn)")
    ap.add_argument("--allow-mismatch", action="store_true", help="Allow different record counts")
    args = ap.parse_args()

    src_path = Path(args.source)
    tr_path = Path(args.translated)
    lang_key = f"turns_{args.lang}"

    src_iter = list(iter_jsonl(src_path))
    tr_iter = list(iter_jsonl(tr_path))

    if not args.allow_mismatch and len(src_iter) != len(tr_iter):
        raise SystemExit(f"Count mismatch: source={len(src_iter)} translated={len(tr_iter)}")

    n = min(len(src_iter), len(tr_iter))
    bad = 0

    for i in range(n):
        src_ln, src = src_iter[i]
        tr_ln, tr = tr_iter[i]

        src_id = src.get("dialogue_id")
        tr_id = tr.get("dialogue_id")
        if src_id != tr_id:
            bad += 1
            print(f"ID mismatch at index={i} source_line={src_ln} translated_line={tr_ln}: {src_id} != {tr_id}")
            if bad >= 10:
                break
            continue

        for k in ("turns_en", "num_turns", "translation_meta"):
            if k not in tr:
                bad += 1
                print(f"Missing key '{k}' at translated_line={tr_ln} dialogue_id={tr_id}")
                break
        if lang_key not in tr:
            bad += 1
            print(f"Missing key '{lang_key}' at translated_line={tr_ln} dialogue_id={tr_id}")
            continue

        turns_en = tr.get("turns_en")
        turns_tgt = tr.get(lang_key)
        if not isinstance(turns_en, list) or not isinstance(turns_tgt, list):
            bad += 1
            print(f"Bad turns types at translated_line={tr_ln} dialogue_id={tr_id}")
            continue

        n_turns = int(tr.get("num_turns") or len(turns_en))
        if len(turns_en) != n_turns:
            bad += 1
            print(f"num_turns mismatch at translated_line={tr_ln} dialogue_id={tr_id}: num_turns={n_turns} len(turns_en)={len(turns_en)}")
        if len(turns_tgt) != len(turns_en):
            bad += 1
            print(f"Turn count mismatch at translated_line={tr_ln} dialogue_id={tr_id}: len(turns_en)={len(turns_en)} len({lang_key})={len(turns_tgt)}")
        if any((t is None) or (str(t).strip() == "") for t in turns_tgt):
            bad += 1
            print(f"Empty translation found at translated_line={tr_ln} dialogue_id={tr_id}")

    if bad:
        raise SystemExit(f"Validation failed: {bad} issue(s) found")

    print(f"OK: validated {n} records ({lang_key})")


if __name__ == "__main__":
    main()
