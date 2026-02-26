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


def run_model_on_buckets(model, tok, buckets, langs, max_new_tokens=128):
    results = {l: {"refs": [], "hyps": [], "langid_ok": 0} for l in langs}
    for l in langs:
        for rec in buckets[l]:
            ref = get_reference(rec)
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
            try:
                dl = detect(gen[:200]) if gen else "unknown"
            except Exception:
                dl = "unknown"
            if dl == l:
                results[l]["langid_ok"] += 1
    return results


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
