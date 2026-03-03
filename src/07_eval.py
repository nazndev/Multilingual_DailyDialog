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
            prompt_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            out = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            new_ids = out[0][prompt_ids.shape[1] :]
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


def compute_bleu(refs, hyps):
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return round(bleu.score, 2)
    except Exception:
        return None


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
        base = get_env("BASE_MODEL") or cfg["model"]["base_model"]
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
        env_targets = get_langs()["targets"]
        langs = env_targets if env_targets else cfg["evaluation"]["langs"]
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
            model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")

        lines = ["# Evaluation Report: Zero-shot vs Fine-tuned\n\n"]

        base_results = None

        # Zero-shot: base model only (no LoRA)
        if run_baseline:
            lines.append("## Zero-shot (base model, no fine-tuning)\n\n")
            base_results = run_model_on_buckets(model, tok, buckets, langs, max_new_tokens)
            for l in langs:
                count = len(buckets[l])
                lines.append(f"### {l}\n")
                lines.append(f"- Samples: {count}\n")
                lines.append(f"- LangID match: {base_results[l]['langid_ok']}/{count}\n")
                if compute_bleu_flag and base_results[l]["refs"]:
                    bleu = compute_bleu(base_results[l]["refs"], base_results[l]["hyps"])
                    if bleu is not None:
                        lines.append(f"- BLEU: {bleu}\n")
                lines.append("\n")
            lines.append("\n")

        if adapter and adapter.exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()

        if adapter and adapter.exists():
            lines.append("## Fine-tuned (LoRA)\n\n")
            lora_results = run_model_on_buckets(model, tok, buckets, langs, max_new_tokens)
        else:
            lines.append("## Fine-tuned (LoRA)\n\n*Adapter not found; run step 6 first.*\n\n")
            lora_results = {l: {"refs": [], "hyps": [], "langid_ok": 0} for l in langs}

        for l in langs:
            count = len(buckets[l])
            lines.append(f"### {l}\n")
            lines.append(f"- Samples: {count}\n")
            lines.append(f"- LangID match: {lora_results[l]['langid_ok']}/{count}\n")
            if compute_bleu_flag and lora_results[l]["refs"]:
                bleu = compute_bleu(lora_results[l]["refs"], lora_results[l]["hyps"])
                if bleu is not None:
                    lines.append(f"- BLEU: {bleu}\n")
            lines.append("\n")

        if compute_bleu_flag:
            all_refs, all_hyps = [], []
            for l in langs:
                all_refs.extend(lora_results[l]["refs"])
                all_hyps.extend(lora_results[l]["hyps"])
            if all_refs:
                overall_bleu = compute_bleu(all_refs, all_hyps)
                if overall_bleu is not None:
                    lines.append("### Overall\n")
                    lines.append(f"- BLEU (all languages): {overall_bleu}\n\n")

        if include_examples and run_baseline and base_results is not None and adapter and adapter.exists():
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
