#!/usr/bin/env python3
"""Merge a manually translated JSONL chunk with the original source chunk.

Goal: preserve all original fields exactly (dialogue_id, turns_en, num_turns,
      dialog_acts, emotions, etc.) and inject ONLY:
        - turns_bn
        - translation_meta

Any extra fields present in the translated file are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on {path} line {line_no}: {e}")
    return rows


def _validate_source_row(row: dict[str, Any], *, path: Path) -> None:
    required = ["dialogue_id", "turns_en", "num_turns", "dialog_acts", "emotions"]
    missing = [k for k in required if k not in row]
    if missing:
        raise SystemExit(f"Source row missing keys {missing} in {path}")


def _extract_translated_fields(row: dict[str, Any], *, path: Path) -> tuple[list[str], dict[str, Any]]:
    if "dialogue_id" not in row:
        raise SystemExit(f"Translated row missing dialogue_id in {path}")
    if "turns_bn" not in row:
        raise SystemExit(f"Translated row missing turns_bn for dialogue_id={row.get('dialogue_id')} in {path}")

    turns_bn = row["turns_bn"]
    if not isinstance(turns_bn, list) or not all(isinstance(t, str) for t in turns_bn):
        raise SystemExit(
            f"turns_bn must be list[str] for dialogue_id={row.get('dialogue_id')} in {path}"
        )

    translation_meta = row.get("translation_meta")
    if translation_meta is None:
        translation_meta = {}
    if not isinstance(translation_meta, dict):
        raise SystemExit(
            f"translation_meta must be an object for dialogue_id={row.get('dialogue_id')} in {path}"
        )

    # Ensure some minimal metadata exists
    translation_meta.setdefault("backend", "manual")
    translation_meta.setdefault("model", "chatgpt")
    translation_meta.setdefault("quality_flags", [])

    return turns_bn, translation_meta


def merge_chunk(*, source_path: Path, translated_path: Path) -> list[dict[str, Any]]:
    source_rows = _read_jsonl(source_path)
    translated_rows = _read_jsonl(translated_path)

    translated_by_id: dict[str, dict[str, Any]] = {}
    for row in translated_rows:
        did = row.get("dialogue_id")
        if not isinstance(did, str) or not did:
            raise SystemExit(f"Invalid dialogue_id in translated file {translated_path}")
        if did in translated_by_id:
            raise SystemExit(f"Duplicate dialogue_id={did} in translated file {translated_path}")
        translated_by_id[did] = row

    out_rows: list[dict[str, Any]] = []
    for src in source_rows:
        _validate_source_row(src, path=source_path)
        did = src["dialogue_id"]
        if did not in translated_by_id:
            raise SystemExit(f"Missing translation for dialogue_id={did} (source {source_path})")

        turns_bn, translation_meta = _extract_translated_fields(translated_by_id[did], path=translated_path)

        turns_en = src["turns_en"]
        if not isinstance(turns_en, list) or not all(isinstance(t, str) for t in turns_en):
            raise SystemExit(f"turns_en must be list[str] for dialogue_id={did} in {source_path}")

        if len(turns_bn) != len(turns_en):
            raise SystemExit(
                f"Turn count mismatch for dialogue_id={did}: "
                f"turns_en={len(turns_en)} turns_bn={len(turns_bn)}"
            )

        merged = dict(src)
        merged["turns_bn"] = turns_bn
        merged["translation_meta"] = translation_meta
        out_rows.append(merged)

    extra_ids = set(translated_by_id.keys()) - {r["dialogue_id"] for r in source_rows}
    if extra_ids:
        # Not fatal, but likely a copy/paste mixup.
        raise SystemExit(
            f"Translated file has {len(extra_ids)} dialogue_id(s) not present in source chunk. "
            f"Example: {sorted(list(extra_ids))[:3]}"
        )

    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="Source chunk JSONL (English) as produced by chunking")
    ap.add_argument("--translated", required=True, type=Path, help="ChatGPT/manual translated JSONL for the same chunk")
    ap.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting (creates file if needed)",
    )
    args = ap.parse_args()

    out_rows = merge_chunk(source_path=args.source, translated_path=args.translated)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.out.open(mode, encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {args.out} ({'append' if args.append else 'overwrite'})")


if __name__ == "__main__":
    main()
