"""
Evaluation: language-ID consistency, BLEU (optional), baseline comparison (optional).
Report written to REPORTS_DIR (from env) or path in config.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from datasets import load_dataset
from langdetect import detect
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, get_langs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def get_reference(rec):
    messages = rec.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _format_prompt(rec, max_chars: int = 800) -> str:
    messages = rec.get("messages") or []
    msgs = messages[:-1] if messages else []
    parts = []
    for m in msgs:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"[system] {content}")
        elif role == "user":
            parts.append(f"[user] {content}")
        elif role == "assistant":
            parts.append(f"[assistant] {content}")
        else:
            parts.append(content)
    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


def run_model_on_buckets(model, tok, buckets, langs, max_new_tokens=128):
    results = {l: {"refs": [], "hyps": [], "prompts": [], "meta": [], "langid_ok": 0} for l in langs}
    for l in langs:
        for rec in buckets[l]:
            ref = get_reference(rec)
            prompt_str = _format_prompt(rec)
            meta = {
                "dialogue_id": rec.get("dialogue_id"),
                "turn_index": rec.get("turn_index"),
            }
            messages = rec["messages"][:-1] if rec.get("messages") else []
            enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            # Transformers 5.x returns a BatchEncoding; generate() expects tensors.
            input_ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
            attention_mask = None
            try:
                attention_mask = enc.get("attention_mask") if hasattr(enc, "get") else None
            except Exception:
                attention_mask = None

            input_ids = input_ids.to(model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            prompt_len = input_ids.shape[1]
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            new_ids = out[0][prompt_len:]
            gen = tok.decode(new_ids, skip_special_tokens=True).strip()
            results[l]["refs"].append(ref)
            results[l]["hyps"].append(gen)
            results[l]["prompts"].append(prompt_str)
            results[l]["meta"].append(meta)
            try:
                dl = detect(gen[:200]) if gen else "unknown"
            except Exception:
                dl = "unknown"
            if dl == l:
                results[l]["langid_ok"] += 1
    return results


def _append_examples(lines, title: str, langs: list, base_results: dict, lora_results: dict, max_examples: int = 3):
    lines.append(f"## {title}\n\n")
    for l in langs:
        lines.append(f"### {l}\n\n")
        n = min(
            max_examples,
            len(base_results.get(l, {}).get("hyps", [])) if base_results else 0,
            len(lora_results.get(l, {}).get("hyps", [])) if lora_results else 0,
        )
        if n <= 0:
            lines.append("*(No samples available.)*\n\n")
            continue
        for i in range(n):
            prompt = (base_results[l].get("prompts") or [""])[i]
            meta = (base_results[l].get("meta") or [{}])[i] or {}
            ref = (base_results[l].get("refs") or [""])[i]
            zh = (base_results[l].get("hyps") or [""])[i]
            fh = (lora_results[l].get("hyps") or [""])[i]
            did = meta.get("dialogue_id") or "unknown"
            tidx = meta.get("turn_index")
            lines.append(f"**Sample {i+1}**\n\n")
            lines.append(f"- dialogue_id: `{did}`\n")
            lines.append(f"- turn_index: `{tidx}`\n\n")
            if prompt:
                lines.append("Prompt:\n\n")
                lines.append("```\n" + prompt + "\n```\n\n")
            if ref:
                lines.append("Reference (gold):\n\n")
                lines.append("```\n" + ref + "\n```\n\n")
            lines.append("Zero-shot output:\n\n")
            lines.append("```\n" + (zh or "") + "\n```\n\n")
            lines.append("Fine-tuned (LoRA) output:\n\n")
            lines.append("```\n" + (fh or "") + "\n```\n\n")
        lines.append("\n")


def _md_escape_cell(s: str, max_chars: int = 220) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Keep tables readable.
    if max_chars and len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    # Basic markdown table escaping.
    s = s.replace("|", "\\|")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "<br>")
    return s


def _append_samples_table(
    lines,
    title: str,
    langs: list,
    base_results: dict,
    lora_results: dict,
    max_rows: int = 3,
):
    lines.append(f"## {title}\n\n")
    for l in langs:
        lines.append(f"### {l}\n\n")
        base = base_results.get(l) if base_results else None
        lora = lora_results.get(l) if lora_results else None
        if not base or not lora:
            lines.append("*(No samples available.)*\n\n")
            continue

        n = min(
            max_rows,
            len(base.get("hyps", [])),
            len(lora.get("hyps", [])),
        )
        if n <= 0:
            lines.append("*(No samples available.)*\n\n")
            continue

        lines.append("| # | dialogue_id | turn_index | Prompt | Reference (gold) | Zero-shot output | Fine-tuned (LoRA) output |\n")
        lines.append("|---:|---|---:|---|---|---|---|\n")
        for i in range(n):
            meta = (base.get("meta") or [{}])[i] or {}
            did = str(meta.get("dialogue_id") or "unknown")
            tidx = meta.get("turn_index")
            prompt = _md_escape_cell((base.get("prompts") or [""])[i], max_chars=240)
            ref = _md_escape_cell((base.get("refs") or [""])[i], max_chars=220)
            zh = _md_escape_cell((base.get("hyps") or [""])[i], max_chars=220)
            fh = _md_escape_cell((lora.get("hyps") or [""])[i], max_chars=220)
            lines.append(
                f"| {i+1} | {did} | {tidx} | {prompt} | {ref} | {zh} | {fh} |\n"
            )
        lines.append("\n")


def compute_bleu(refs, hyps):
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return round(bleu.score, 2)
    except Exception:
        return None


def _fmt_bleu(x):
    return "-" if x is None else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logger = setup_logger("07_eval")
    banner(logger, "Step 07: Evaluate")
    log_env_safely(logger, ["DATA_DIR", "OUTPUTS_DIR", "REPORTS_DIR", "TARGET_LANGS", "BASE_MODEL"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        # Prefer the config's model for reproducibility; fall back to env only if config omits it.
        base = cfg.get("model", {}).get("base_model") or get_env("BASE_MODEL")
        adapter_rel = cfg["model"].get("lora_adapter_dir")
        adapter = resolve_path(adapter_rel, dirs["outputs"]) if adapter_rel else None
        test_path = resolve_path(cfg["data"]["test_path"], dirs["data"])
        logger.info("test_path=%s base_model=%s adapter=%s", test_path, base, adapter)

        with timer(logger, "load_dataset"):
            ds = load_dataset("json", data_files=str(test_path), split="train")
        logger.info("test_dataset_size=%s", len(ds))

        run_baseline = cfg.get("evaluation", {}).get("run_baseline", False)
        compute_bleu_flag = cfg.get("evaluation", {}).get("compute_bleu", True)
        max_new_tokens = int(cfg.get("evaluation", {}).get("max_new_tokens", 128))
        out_cfg = cfg.get("outputs", {}) or {}
        include_examples = bool(out_cfg.get("include_samples", out_cfg.get("include_examples", False)))
        num_examples = int(out_cfg.get("num_samples_per_lang", out_cfg.get("num_examples_per_lang", 3)))
        samples_format = (out_cfg.get("samples_format") or "blocks").strip().lower()
        env_targets = get_langs()["targets"]
        cfg_langs = cfg.get("evaluation", {}).get("langs")
        # Prefer config so eval is deterministic even if TARGET_LANGS is set.
        langs = cfg_langs if cfg_langs else env_targets
        n = int(cfg["evaluation"]["num_samples_per_lang"])
        logger.info("evaluation_langs=%s num_samples_per_lang=%s run_baseline=%s compute_bleu=%s", langs, n, run_baseline, compute_bleu_flag)

        buckets = {l: [] for l in langs}
        for rec in ds:
            l = rec.get("lang")
            if l in buckets and len(buckets[l]) < n:
                buckets[l].append(rec)
            if all(len(buckets[l]) >= n for l in langs):
                break
        for l in langs:
            logger.info("bucket lang=%s count=%s", l, len(buckets[l]))

        with timer(logger, "load_model"):
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            # Keep the model on a single device for reliable generation on macOS.
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            model = AutoModelForCausalLM.from_pretrained(base)
            model.to(device)
            logger.info("model_device=%s", device)

        lines = ["# Evaluation Report: Zero-shot vs Fine-tuned\n\n"]
        lines.append("## Run settings\n\n")
        lines.append(f"- Base model: {base}\n")
        lines.append(f"- Adapter: {adapter if adapter else 'none'}\n")
        lines.append(f"- Languages: {', '.join(langs)}\n")
        lines.append(f"- Samples per language: {n}\n\n")

        base_results = None

        # Zero-shot: base model only (no LoRA)
        if run_baseline:
            lines.append("## Zero-shot (base model, no fine-tuning)\n\n")
            lines.append("Generated outputs using the base model only (no LoRA). Details are summarized below.\n\n")
            base_results = run_model_on_buckets(model, tok, buckets, langs, max_new_tokens)
            # Header later; we present a combined comparison table.

        if adapter and adapter.exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()

        if adapter and adapter.exists():
            lines.append("## Fine-tuned (LoRA)\n\n")
            lines.append("Generated outputs using the same base model with the trained LoRA adapter applied. Details are summarized below.\n\n")
            lora_results = run_model_on_buckets(model, tok, buckets, langs, max_new_tokens)
        else:
            lines.append("## Fine-tuned (LoRA)\n\n")
            lines.append("*Adapter not found; run step 6 first.*\n\n")
            lora_results = {l: {"refs": [], "hyps": [], "prompts": [], "meta": [], "langid_ok": 0} for l in langs}

        for l in langs:
            count = len(buckets[l])
            pass

        # Summary comparison table
        lines.append("## Summary (Zero-shot vs Fine-tuned)\n\n")
        if compute_bleu_flag:
            lines.append("| Lang | Samples | Zero-shot LangID | Zero-shot BLEU | Fine-tuned LangID | Fine-tuned BLEU |\n")
            lines.append("|---|---:|---:|---:|---:|---:|\n")
        else:
            lines.append("| Lang | Samples | Zero-shot LangID | Fine-tuned LangID |\n")
            lines.append("|---|---:|---:|---:|\n")

        overall_base_refs, overall_base_hyps = [], []
        overall_lora_refs, overall_lora_hyps = [], []

        for l in langs:
            count = len(buckets[l])
            z_ok = base_results[l]["langid_ok"] if base_results and run_baseline else 0
            f_ok = lora_results[l]["langid_ok"] if lora_results else 0

            if compute_bleu_flag:
                z_bleu = None
                f_bleu = None
                if base_results and run_baseline and base_results[l].get("refs"):
                    z_bleu = compute_bleu(base_results[l]["refs"], base_results[l]["hyps"])
                    overall_base_refs.extend(base_results[l]["refs"])
                    overall_base_hyps.extend(base_results[l]["hyps"])
                if lora_results and lora_results[l].get("refs"):
                    f_bleu = compute_bleu(lora_results[l]["refs"], lora_results[l]["hyps"])
                    overall_lora_refs.extend(lora_results[l]["refs"])
                    overall_lora_hyps.extend(lora_results[l]["hyps"])

                lines.append(
                    f"| {l} | {count} | {z_ok}/{count} | {_fmt_bleu(z_bleu)} | {f_ok}/{count} | {_fmt_bleu(f_bleu)} |\n"
                )
            else:
                lines.append(f"| {l} | {count} | {z_ok}/{count} | {f_ok}/{count} |\n")
        lines.append("\n")

        if compute_bleu_flag:
            lines.append("## Overall BLEU\n\n")
            if run_baseline and overall_base_refs:
                overall_z = compute_bleu(overall_base_refs, overall_base_hyps)
                lines.append(f"- Zero-shot BLEU (all languages): {_fmt_bleu(overall_z)}\n")
            if overall_lora_refs:
                overall_f = compute_bleu(overall_lora_refs, overall_lora_hyps)
                lines.append(f"- Fine-tuned BLEU (all languages): {_fmt_bleu(overall_f)}\n")
            lines.append("\n")

        if include_examples and run_baseline and base_results is not None and adapter and adapter.exists():
            if samples_format == "table":
                _append_samples_table(
                    lines,
                    title="Samples Table (from test set): Zero-shot vs Fine-tuned",
                    langs=langs,
                    base_results=base_results,
                    lora_results=lora_results,
                    max_rows=num_examples,
                )
            else:
                _append_examples(
                    lines,
                    title="Samples (from test set): Zero-shot vs Fine-tuned",
                    langs=langs,
                    base_results=base_results,
                    lora_results=lora_results,
                    max_examples=num_examples,
                )

        report_path = cfg["outputs"].get("report_path", "eval_report.md")
        out_path = resolve_path(report_path, dirs["reports"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")
        logger.info("eval_report_path=%s eval_sample_count_total=%s", out_path, sum(len(buckets[l]) for l in langs))
        banner(logger, "Step 07: Done", char="-")
    except Exception:
        logger.exception("Step 07 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
