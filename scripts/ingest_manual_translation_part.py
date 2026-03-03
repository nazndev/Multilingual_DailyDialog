#!/usr/bin/env python3
"""Ingest a manually translated chunk/part into the cumulative translated split.

This is a convenience wrapper around the existing merge/validate logic, designed
so we can run everything with a single short command (stable in VS Code
integrated terminals).

It will:
- canonical-merge: preserve all source fields; inject only turns_bn + translation_meta
- validate basic structural invariants against the source
- check for duplicate dialogue_id against the cumulative output
- append (or overwrite) the cumulative output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	if not path.exists():
		raise SystemExit(f"File not found: {path}")
	with path.open("r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			s = line.strip()
			if not s:
				continue
			try:
				rows.append(json.loads(s))
			except json.JSONDecodeError as e:
				raise SystemExit(f"Invalid JSON on {path} line {line_no}: {e}")
	return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	mode = "a" if append else "w"
	with path.open(mode, encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_translation_meta(meta: Any) -> dict[str, Any]:
	if meta is None:
		meta = {}
	if not isinstance(meta, dict):
		raise SystemExit("translation_meta must be an object")
	meta.setdefault("backend", "manual")
	meta.setdefault("model", "chatgpt")
	meta.setdefault("quality_flags", [])
	return meta


def merge_source_with_translated(*, source_path: Path, translated_path: Path) -> list[dict[str, Any]]:
	source_rows = _read_jsonl(source_path)
	translated_rows = _read_jsonl(translated_path)

	translated_by_id: dict[str, dict[str, Any]] = {}
	for row in translated_rows:
		did = row.get("dialogue_id")
		if not isinstance(did, str) or not did:
			raise SystemExit(f"Invalid dialogue_id in translated file: {translated_path}")
		if did in translated_by_id:
			raise SystemExit(f"Duplicate dialogue_id={did} in translated file: {translated_path}")
		translated_by_id[did] = row

	out_rows: list[dict[str, Any]] = []
	source_ids: list[str] = []

	for src in source_rows:
		did = src.get("dialogue_id")
		if not isinstance(did, str) or not did:
			raise SystemExit(f"Invalid dialogue_id in source file: {source_path}")
		source_ids.append(did)

		if did not in translated_by_id:
			raise SystemExit(f"Missing translation for dialogue_id={did} (source {source_path})")

		tr = translated_by_id[did]
		turns_bn = tr.get("turns_bn")
		if not isinstance(turns_bn, list) or not all(isinstance(t, str) for t in turns_bn):
			raise SystemExit(f"turns_bn must be list[str] for dialogue_id={did} in {translated_path}")

		turns_en = src.get("turns_en")
		if not isinstance(turns_en, list) or not all(isinstance(t, str) for t in turns_en):
			raise SystemExit(f"turns_en must be list[str] for dialogue_id={did} in {source_path}")

		if len(turns_bn) != len(turns_en):
			raise SystemExit(
				f"Turn count mismatch for dialogue_id={did}: turns_en={len(turns_en)} turns_bn={len(turns_bn)}"
			)

		merged = dict(src)
		merged["turns_bn"] = turns_bn
		merged["translation_meta"] = _ensure_translation_meta(tr.get("translation_meta"))
		out_rows.append(merged)

	extra_ids = set(translated_by_id.keys()) - set(source_ids)
	if extra_ids:
		raise SystemExit(
			f"Translated file has {len(extra_ids)} extra dialogue_id(s) not in source. Example: {sorted(extra_ids)[:3]}"
		)

	return out_rows


def validate_canonical_against_source(*, source_path: Path, canonical_rows: list[dict[str, Any]]) -> None:
	source_rows = _read_jsonl(source_path)
	if len(source_rows) != len(canonical_rows):
		raise SystemExit(f"Count mismatch after merge: source={len(source_rows)} canonical={len(canonical_rows)}")

	for idx, (src, tr) in enumerate(zip(source_rows, canonical_rows, strict=True)):
		src_id = src.get("dialogue_id")
		tr_id = tr.get("dialogue_id")
		if src_id != tr_id:
			raise SystemExit(f"ID mismatch at index={idx}: source={src_id} canonical={tr_id}")

		for k in ("turns_en", "num_turns", "translation_meta", "turns_bn"):
			if k not in tr:
				raise SystemExit(f"Missing key '{k}' after merge for dialogue_id={tr_id}")

		turns_en = tr["turns_en"]
		turns_bn = tr["turns_bn"]
		if not isinstance(turns_en, list) or not isinstance(turns_bn, list):
			raise SystemExit(f"Bad turns types after merge for dialogue_id={tr_id}")

		n_turns = int(tr.get("num_turns") or len(turns_en))
		if len(turns_en) != n_turns:
			raise SystemExit(
				f"num_turns mismatch after merge for dialogue_id={tr_id}: num_turns={n_turns} len(turns_en)={len(turns_en)}"
			)
		if len(turns_bn) != len(turns_en):
			raise SystemExit(
				f"Turn count mismatch after merge for dialogue_id={tr_id}: len(turns_en)={len(turns_en)} len(turns_bn)={len(turns_bn)}"
			)
		if any((t is None) or (str(t).strip() == "") for t in turns_bn):
			raise SystemExit(f"Empty Bangla turn found for dialogue_id={tr_id}")


def _iter_ids_in_file(path: Path) -> set[str]:
	if not path.exists():
		return set()
	ids: set[str] = set()
	for row in _read_jsonl(path):
		did = row.get("dialogue_id")
		if isinstance(did, str) and did:
			ids.add(did)
	return ids


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--source", required=True, type=Path)
	ap.add_argument("--translated", required=True, type=Path)
	ap.add_argument("--canonical-out", required=True, type=Path)
	ap.add_argument("--final-out", required=True, type=Path)
	ap.add_argument("--append", action="store_true", help="Append to --final-out instead of overwriting")
	args = ap.parse_args()

	canonical_rows = merge_source_with_translated(source_path=args.source, translated_path=args.translated)
	validate_canonical_against_source(source_path=args.source, canonical_rows=canonical_rows)
	_write_jsonl(args.canonical_out, canonical_rows, append=False)

	if args.append:
		final_ids = _iter_ids_in_file(args.final_out)
		chunk_ids = {r["dialogue_id"] for r in canonical_rows}
		dups = sorted(final_ids.intersection(chunk_ids))
		if dups:
			raise SystemExit(f"Duplicate dialogue_id(s) already in final output: {dups[:5]}")

	before = 0
	if args.final_out.exists() and args.append:
		before = len(_read_jsonl(args.final_out))

	_write_jsonl(args.final_out, canonical_rows, append=args.append)
	after = len(_read_jsonl(args.final_out))

	print(f"OK: merged={len(canonical_rows)} wrote_canonical={args.canonical_out}")
	if args.append:
		print(f"OK: appended_to={args.final_out} rows_before={before} rows_after={after}")
	else:
		print(f"OK: wrote_final={args.final_out} rows={after}")


if __name__ == "__main__":
	main()
