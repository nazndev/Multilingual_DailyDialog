"""
Step 08: Pairwise LLM Judge

Use a judge model (e.g., Qwen2.5-7B-Instruct) to compare two model outputs
for the same prompt/reference pair from existing generations JSONL artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, resolve_path
from src.utils.logging_utils import banner, log_config_safely, log_env_safely, setup_logger, timer


def load_cfg(path: str) -> dict:
    import yaml

    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _parse_torch_dtype(name: str | None):
    if not name or str(name).lower() in ("auto", "none", "null"):
        return None
    n = str(name).lower()
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16"):
        return torch.float16
    if n in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported judge.model.torch_dtype={name!r}")


def _load_model(model_cfg: dict, logger):
    model_name = model_cfg.get("base_model")
    if not model_name:
        raise ValueError("Missing judge.model.base_model")
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("judge.model.load_in_4bit and load_in_8bit cannot both be true.")
    device_map = model_cfg.get("device_map", "auto")
    dtype = _parse_torch_dtype(model_cfg.get("torch_dtype"))

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = device_map
    else:
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    logger.info("judge_model=%s", model_name)
    logger.info("judge_device_map=%s", kwargs.get("device_map"))
    return tok, model


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(row: dict) -> tuple[Any, Any]:
    return (row.get("dialogue_id"), row.get("turn_index"))


def _build_index(rows: list[dict]) -> dict[tuple[Any, Any], dict]:
    out: dict[tuple[Any, Any], dict] = {}
    for r in rows:
        out[_key(r)] = r
    return out


def _safe_text(v: Any) -> str:
    return (v or "").strip() if isinstance(v, str) else ""


def _judge_prompt(system_prompt: str, prompt: str, reference: str, cand_a: str, cand_b: str) -> list[dict]:
    user_content = (
        "Task: judge which candidate response is better for Bengali dialogue next-utterance generation.\n\n"
        "Scoring criteria (priority order):\n"
        "1) Relevance to the immediate prompt/context\n"
        "2) Faithfulness to reference intent (not exact wording)\n"
        "3) Fluency and natural Bengali\n"
        "4) Safety and non-hallucination\n"
        "5) Conciseness when appropriate\n\n"
        "Return STRICT JSON only with keys:\n"
        '{"winner":"A|B|TIE","reason":"short reason","score_a":0-10,"score_b":0-10}\n\n'
        f"Prompt:\n{prompt}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Candidate A:\n{cand_a}\n\n"
        f"Candidate B:\n{cand_b}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_winner(v: Any) -> str:
    s = str(v or "").strip().upper()
    if s in ("A", "B", "TIE"):
        return s
    return "TIE"


def _score_to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _run_judgement(tok, model, messages: list[dict], max_new_tokens: int) -> tuple[dict, str]:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
    attention_mask = enc.get("attention_mask") if hasattr(enc, "get") else None
    dev = getattr(model, "device", None) or next(model.parameters()).device
    input_ids = input_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    prompt_len = input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        num_beams=1,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    raw = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    parsed = _extract_json(raw) or {}
    return parsed, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    logger = setup_logger("08_judge")
    banner(logger, "Step 08: LLM Judge (Pairwise)")
    log_env_safely(logger, ["REPORTS_DIR", "CACHE_DIR", "OUTPUTS_DIR"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")

        judge_cfg = cfg.get("judge", {}) if isinstance(cfg.get("judge"), dict) else {}
        inputs_cfg = cfg.get("inputs", {}) if isinstance(cfg.get("inputs"), dict) else {}
        outputs_cfg = cfg.get("outputs", {}) if isinstance(cfg.get("outputs"), dict) else {}
        comp_cfg = cfg.get("comparison", {}) if isinstance(cfg.get("comparison"), dict) else {}

        a_name = str(comp_cfg.get("candidate_a_name", "model_a"))
        b_name = str(comp_cfg.get("candidate_b_name", "model_b"))
        max_cases = int(comp_cfg.get("max_cases", 200))
        max_new_tokens = int(judge_cfg.get("max_new_tokens", 120))
        system_prompt = str(
            judge_cfg.get(
                "system_prompt",
                "You are a strict and fair evaluator for Bengali dialogue responses.",
            )
        )

        path_a = resolve_path(inputs_cfg["candidate_a_predictions_path"], dirs["reports"])
        path_b = resolve_path(inputs_cfg["candidate_b_predictions_path"], dirs["reports"])
        out_json = resolve_path(outputs_cfg.get("judge_metrics_path", "judge_metrics.json"), dirs["reports"])
        out_jsonl = resolve_path(outputs_cfg.get("judge_decisions_path", "judge_decisions.jsonl"), dirs["reports"])
        out_md = resolve_path(outputs_cfg.get("judge_report_path", "judge_report.md"), dirs["reports"])

        logger.info("candidate_a=%s (%s)", a_name, path_a)
        logger.info("candidate_b=%s (%s)", b_name, path_b)

        rows_a = _load_jsonl(path_a)
        rows_b = _load_jsonl(path_b)
        idx_a = _build_index(rows_a)
        idx_b = _build_index(rows_b)
        shared_keys = [k for k in idx_a.keys() if k in idx_b]
        shared_keys = shared_keys[:max_cases]
        if not shared_keys:
            raise ValueError("No overlapping (dialogue_id, turn_index) rows between candidate A and B predictions.")
        logger.info("judge_cases=%s", len(shared_keys))

        with timer(logger, "load_judge_model"):
            tok, model = _load_model(judge_cfg.get("model", {}), logger)

        decisions: list[dict] = []
        wins = {"A": 0, "B": 0, "TIE": 0}
        score_a_sum = 0.0
        score_b_sum = 0.0
        score_a_n = 0
        score_b_n = 0

        with timer(logger, "run_judge"):
            for i, k in enumerate(shared_keys, start=1):
                ra = idx_a[k]
                rb = idx_b[k]
                prompt = _safe_text(ra.get("prompt")) or _safe_text(rb.get("prompt"))
                reference = _safe_text(ra.get("reference")) or _safe_text(rb.get("reference"))
                cand_a = _safe_text(ra.get("fine_tuned"))
                cand_b = _safe_text(rb.get("fine_tuned"))
                if not prompt or not cand_a or not cand_b:
                    continue

                messages = _judge_prompt(system_prompt, prompt, reference, cand_a, cand_b)
                parsed, raw = _run_judgement(tok, model, messages, max_new_tokens=max_new_tokens)
                winner = _normalize_winner(parsed.get("winner"))
                reason = _safe_text(parsed.get("reason"))
                score_a = _score_to_float(parsed.get("score_a"))
                score_b = _score_to_float(parsed.get("score_b"))

                wins[winner] += 1
                if score_a is not None:
                    score_a_sum += score_a
                    score_a_n += 1
                if score_b is not None:
                    score_b_sum += score_b
                    score_b_n += 1

                decisions.append(
                    {
                        "index": i,
                        "dialogue_id": k[0],
                        "turn_index": k[1],
                        "lang": ra.get("lang") or rb.get("lang"),
                        "reference": reference,
                        "candidate_a_name": a_name,
                        "candidate_b_name": b_name,
                        "candidate_a_text": cand_a,
                        "candidate_b_text": cand_b,
                        "winner": winner,
                        "reason": reason,
                        "score_a": score_a,
                        "score_b": score_b,
                        "raw_judge_output": raw,
                    }
                )

        total = len(decisions)
        if total == 0:
            raise ValueError("No valid judge decisions were produced.")

        avg_a = round(score_a_sum / score_a_n, 4) if score_a_n else None
        avg_b = round(score_b_sum / score_b_n, 4) if score_b_n else None
        win_rate_a = round(wins["A"] / total, 4)
        win_rate_b = round(wins["B"] / total, 4)
        tie_rate = round(wins["TIE"] / total, 4)
        preferred = a_name if wins["A"] > wins["B"] else b_name if wins["B"] > wins["A"] else "TIE"

        payload = {
            "script": "08_judge.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_cfg.get("model", {}).get("base_model"),
            "candidate_a_name": a_name,
            "candidate_b_name": b_name,
            "candidate_a_predictions_path": str(path_a),
            "candidate_b_predictions_path": str(path_b),
            "cases_requested": max_cases,
            "cases_judged": total,
            "wins": wins,
            "win_rates": {"candidate_a": win_rate_a, "candidate_b": win_rate_b, "tie": tie_rate},
            "average_scores": {"candidate_a": avg_a, "candidate_b": avg_b},
            "preferred_candidate": preferred,
        }

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with out_jsonl.open("w", encoding="utf-8") as w:
            for row in decisions:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")

        md_lines = [
            "# LLM Judge Report (Qwen 7B)\n\n",
            f"- Judge model: `{payload['judge_model']}`\n",
            f"- Candidate A: `{a_name}`\n",
            f"- Candidate B: `{b_name}`\n",
            f"- Cases judged: `{total}`\n",
            f"- Wins: A=`{wins['A']}`, B=`{wins['B']}`, Tie=`{wins['TIE']}`\n",
            f"- Win rates: A=`{win_rate_a}`, B=`{win_rate_b}`, Tie=`{tie_rate}`\n",
            f"- Avg score: A=`{avg_a}`, B=`{avg_b}`\n",
            f"- Preferred candidate: `{preferred}`\n\n",
            "## Sample decisions\n\n",
            "| # | dialogue_id | turn_index | winner | reason |\n",
            "|---:|---|---:|---|---|\n",
        ]
        for row in decisions[:10]:
            reason = (row.get("reason") or "").replace("|", "\\|").replace("\n", " ")
            md_lines.append(
                f"| {row['index']} | {row.get('dialogue_id')} | {row.get('turn_index')} | {row.get('winner')} | {reason} |\n"
            )
        out_md.write_text("".join(md_lines), encoding="utf-8")

        logger.info("judge_metrics_path=%s", out_json)
        logger.info("judge_decisions_path=%s", out_jsonl)
        logger.info("judge_report_path=%s", out_md)
        banner(logger, "Step 08: Done", char="-")
    except Exception:
        logger.exception("Step 08 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
