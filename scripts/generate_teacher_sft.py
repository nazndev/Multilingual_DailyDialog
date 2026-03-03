#!/usr/bin/env python3
"""Generate teacher-labeled SFT JSONL using a Qwen 7B model.

Reads sampled SFT examples (JSONL). For each record:
- Uses the messages WITHOUT the final assistant message as the prompt.
- Generates a new assistant reply using the teacher model.
- Writes a new JSONL record where the final assistant message content is the teacher output.

This implements the professor requirement:
  Qwen 7B outputs are treated as "ground truth" targets for training Qwen 0.5B.

Usage:
  /path/to/python scripts/generate_teacher_sft.py \
    --input data/sft/sampled_1000_bn/train.jsonl \
    --output data/sft/teacher_1000_bn/train.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-new-tokens 96 --batch-size 1

Notes:
- This is compute-heavy; recommended to run on CUDA (Colab).
- Works on CPU/MPS too but will be slow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_load_kwargs(device: torch.device) -> dict:
    """Pick safe model loading kwargs for the detected device.

    Qwen 7B will typically OOM in fp32 on Colab GPUs.
    """

    kw: dict = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        # T4/L4/A100: fp16 is the safest default.
        kw.update({"torch_dtype": torch.float16, "device_map": "auto"})
    elif device.type == "mps":
        # MPS benefits from fp16, but does not support device_map.
        kw.update({"torch_dtype": torch.float16})
    return kw


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for rec in recs:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for quick smoke runs")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    device = _pick_device()

    load_kwargs = _pick_load_kwargs(device)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    # If accelerate device_map placed the model, don't move it again.
    if "device_map" not in load_kwargs:
        model.to(device)
    model.eval()

    out_recs = []
    n = 0
    for rec in _iter_jsonl(in_path):
        n += 1
        if args.limit and n > args.limit:
            break

        messages = rec.get("messages") or []
        if not messages:
            continue

        # Prompt = everything except the last assistant label.
        prompt_messages = messages[:-1]
        if not prompt_messages:
            continue

        enc = tok.apply_chat_template(prompt_messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
        attention_mask = enc.get("attention_mask") if isinstance(enc, dict) else None

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_len = input_ids.shape[1]
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(args.max_new_tokens),
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if args.temperature and args.temperature > 0.0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                }
            )
        gen_ids = model.generate(**gen_kwargs)
        new_ids = gen_ids[0][prompt_len:]
        teacher_text = tok.decode(new_ids, skip_special_tokens=True).strip()

        new_rec = dict(rec)
        # Keep original as optional reference.
        try:
            last = (messages[-1] or {})
            if isinstance(last, dict) and last.get("role") == "assistant":
                new_rec["reference_original"] = (last.get("content") or "")
        except Exception:
            pass

        new_rec["messages"] = list(prompt_messages) + [{"role": "assistant", "content": teacher_text}]
        new_rec["teacher_meta"] = {
            "model": args.model,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
        }
        out_recs.append(new_rec)

        if n % 50 == 0:
            print(f"Generated {n} teacher examples...")

    _write_jsonl(out_path, out_recs)
    print(f"Wrote {len(out_recs)} teacher examples to {out_path}")


if __name__ == "__main__":
    main()
