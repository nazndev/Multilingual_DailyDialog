#!/usr/bin/env python3
"""Chunk a JSONL file into smaller JSONL parts for manual ChatGPT translation.

Example:
  python scripts/chunk_jsonl_for_chatgpt.py \
    --input data/processed_1000/train_en.jsonl \
    --out-dir data/manual_chunks_1000/train \
    --chunk-lines 20

This does NOT translate anything. It only helps you paste small chunks into ChatGPT.
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--out-dir", required=True, help="Directory for chunk JSONL files")
    ap.add_argument("--chunk-lines", type=int, default=20, help="Number of JSONL lines per chunk")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_lines <= 0:
        raise SystemExit("--chunk-lines must be > 0")

    chunk_idx = 0
    line_idx_in_chunk = 0
    out_f = None

    def _open_new_chunk(i: int):
        return (out_dir / f"chunk_{i:04d}.jsonl").open("w", encoding="utf-8")

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.rstrip("\n")
            if not s.strip():
                continue
            if out_f is None or line_idx_in_chunk >= args.chunk_lines:
                if out_f is not None:
                    out_f.close()
                chunk_idx += 1
                line_idx_in_chunk = 0
                out_f = _open_new_chunk(chunk_idx)

            out_f.write(s + "\n")
            line_idx_in_chunk += 1

    if out_f is not None:
        out_f.close()

    print(f"Wrote {chunk_idx} chunk files to: {out_dir}")


if __name__ == "__main__":
    main()
