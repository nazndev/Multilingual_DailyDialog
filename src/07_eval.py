"""
Evaluation for multilingual next-utterance generation:
- language-ID consistency
- zero-shot (base) vs fine-tuned (LoRA) comparison
- BLEU / chrF / optional BERTScore automatic metrics
Artifacts are written under REPORTS_DIR using config paths.
"""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from langdetect import detect
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, get_langs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer
from src.utils.prompting import (
    MODEL_FAMILY_DEFAULT,
    MODEL_FAMILY_GEMMA,
    messages_for_generation_from_record,
)


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def get_reference(rec):
    messages = rec.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _format_messages_for_display(messages: list, max_chars: int = 800) -> str:
    parts = []
    for m in messages or []:
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
        elif role == "model":
            parts.append(f"[model] {content}")
        else:
            parts.append(content)
    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


def _resolve_generation_kwargs(cfg_eval: dict, tok) -> tuple[dict, dict]:
    """
    Return (generation_kwargs_for_model, metadata_dict).
    Defaults match deterministic benchmark-style decoding.
    """
    g = cfg_eval.get("generation") if isinstance(cfg_eval.get("generation"), dict) else {}
    max_new_tokens = int(g.get("max_new_tokens", cfg_eval.get("max_new_tokens", 128)))
    do_sample = bool(g.get("do_sample", False))
    temperature = float(g.get("temperature", 1.0))
    top_p = float(g.get("top_p", 1.0))
    num_beams = int(g.get("num_beams", 1))
    repetition_penalty = float(g.get("repetition_penalty", 1.0))
    eos_mode = str(g.get("eos_token_id_mode", "auto")).strip().lower()

    gen = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
    }
    if eos_mode == "eos":
        gen["eos_token_id"] = tok.eos_token_id
        gen["pad_token_id"] = tok.pad_token_id or tok.eos_token_id
    else:
        gen["pad_token_id"] = tok.pad_token_id or tok.eos_token_id

    meta = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "eos_token_id_mode": eos_mode,
    }
    return gen, meta


def run_model_on_buckets(
    model,
    tok,
    buckets,
    langs,
    gen_kwargs: dict,
    prompt_kwargs: dict,
):
    results = {l: {"refs": [], "hyps": [], "prompts": [], "meta": [], "langid_ok": 0} for l in langs}
    for l in langs:
        for rec in buckets[l]:
            ref = get_reference(rec)
            messages = messages_for_generation_from_record(rec, **prompt_kwargs)
            if not messages:
                continue
            prompt_str = _format_messages_for_display(messages)
            meta = {
                "dialogue_id": rec.get("dialogue_id"),
                "turn_index": rec.get("turn_index"),
                "emotion_at_turn": rec.get("emotion_at_turn"),
                "act_at_turn": rec.get("act_at_turn"),
                "lang": l,
            }
            enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            input_ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
            attention_mask = None
            try:
                attention_mask = enc.get("attention_mask") if hasattr(enc, "get") else None
            except Exception:
                attention_mask = None

            dev = getattr(model, "device", None) or next(model.parameters()).device
            input_ids = input_ids.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)

            prompt_len = input_ids.shape[1]
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
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
    if max_chars and len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
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


def _resolve_bleu_tokenizer(eval_cfg: dict) -> str:
    tok = str(eval_cfg.get("bleu_tokenizer", "flores200")).strip()
    return tok or "flores200"


def compute_bleu(refs, hyps, bleu_tokenizer: str = "flores200"):
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tokenizer)
        return round(bleu.score, 2)
    except Exception:
        return None


def compute_chrf(refs, hyps):
    try:
        import sacrebleu
        score = sacrebleu.corpus_chrf(hyps, [refs])
        return round(score.score, 2)
    except Exception:
        return None


_BERTSCORE_LANG_MAP = {"bn": "bn", "en": "en", "de": "de", "fr": "fr"}


def resolve_bertscore_lang(lang: str) -> str:
    """Map eval language codes to bert-score ``lang``; unknown codes pass through for a best-effort try."""
    key = (lang or "en").strip().lower()
    return _BERTSCORE_LANG_MAP.get(key, key)


def compute_bertscore_optional(hyps, refs, lang: str):
    """Return mean F1 or None if bert-score is unavailable or fails."""
    try:
        from bert_score import score as bert_score_fn
    except Exception:
        return None
    primary = resolve_bertscore_lang(lang)
    candidates: list[str] = []
    for cand in (primary, "en"):
        if cand and cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        try:
            _, _, f1 = bert_score_fn(hyps, refs, lang=cand)
            return round(f1.mean().item(), 4)
        except Exception:
            continue
    return None


def _is_gemma_model(model_id: str) -> bool:
    return "gemma" in (model_id or "").lower()


def _fmt_bleu(x):
    return "-" if x is None else str(x)


def _safe_ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


def _group_metrics_by_label(
    result_pack: dict,
    label_key: str,
    *,
    compute_bleu_flag: bool,
    compute_chrf_flag: bool,
    compute_bert_flag: bool,
    bert_lang: str,
    bleu_tokenizer: str,
):
    """Aggregate BLEU/chrF/BERTScore per label when labels are present on metadata rows."""
    groups: dict[str, dict[str, Any]] = {}
    refs = result_pack.get("refs", [])
    hyps = result_pack.get("hyps", [])
    metas = result_pack.get("meta", [])
    for idx in range(min(len(refs), len(hyps), len(metas))):
        label = (metas[idx] or {}).get(label_key)
        if label is None or label == "":
            continue
        key = str(label)
        if key not in groups:
            groups[key] = {"refs": [], "hyps": []}
        groups[key]["refs"].append(refs[idx])
        groups[key]["hyps"].append(hyps[idx])
    out: dict[str, Any] = {}
    for label, val in groups.items():
        r, h = val["refs"], val["hyps"]
        out[label] = {
            "count": len(r),
            "bleu": compute_bleu(r, h, bleu_tokenizer) if compute_bleu_flag and r else None,
            "chrf": compute_chrf(r, h) if compute_chrf_flag and r else None,
            "bertscore_f1": compute_bertscore_optional(h, r, bert_lang) if compute_bert_flag and r else None,
        }
    return out


def _load_eval_model(base: str, model_cfg: dict, logger):
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("model.load_in_4bit and model.load_in_8bit cannot both be true.")
    device_map = model_cfg.get("device_map")
    dtype_str = model_cfg.get("torch_dtype")
    torch_dtype = None
    if dtype_str and str(dtype_str).lower() not in ("auto", "none", "null"):
        n = str(dtype_str).lower()
        torch_dtype = torch.bfloat16 if n in ("bfloat16", "bf16") else torch.float16 if n in ("float16", "fp16") else None

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = device_map or "auto"
    else:
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if quantization_config is None and device_map is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
        logger.info("model_device=%s", device)
    else:
        logger.info("model_device_map=%s", kwargs.get("device_map"))
    return model


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
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        base = model_cfg.get("base_model") or get_env("BASE_MODEL")
        if not base:
            raise ValueError("Missing base model: set `model.base_model` in the eval config or BASE_MODEL.")
        adapter_rel = cfg["model"].get("lora_adapter_dir")
        adapter = resolve_path(adapter_rel, dirs["outputs"]) if adapter_rel else None
        test_path = resolve_path(cfg["data"]["test_path"], dirs["data"])
        logger.info("test_path=%s base_model=%s adapter=%s", test_path, base, adapter)
        base_l = (base or "").lower()
        test_path_l = str(test_path).lower()
        if "qwen" in base_l:
            assert "qwen_bn" in test_path_l, (
                f"Evaluation dataset mismatch: model '{base}' requires a qwen_bn test set, got '{test_path}'."
            )
        if "gemma" in base_l:
            assert "gemma_bn" in test_path_l, (
                f"Evaluation dataset mismatch: model '{base}' requires a gemma_bn test set, got '{test_path}'."
            )

        eval_cfg = cfg.get("evaluation", {}) or {}
        bleu_tokenizer = _resolve_bleu_tokenizer(eval_cfg)
        logger.info("bleu_tokenizer=%s", bleu_tokenizer)
        model_family = MODEL_FAMILY_GEMMA if _is_gemma_model(base) else MODEL_FAMILY_DEFAULT
        logger.info("model_family=%s", model_family)
        prompt_cfg = eval_cfg.get("prompt") if isinstance(eval_cfg.get("prompt"), dict) else {}
        prompt_kwargs = {
            "use_emotion_tag": bool(prompt_cfg.get("use_emotion_tag", eval_cfg.get("use_emotion_tag", False))),
            "use_dialog_act_tag": bool(prompt_cfg.get("use_dialog_act_tag", eval_cfg.get("use_dialog_act_tag", False))),
            "short_reply_hint": bool(prompt_cfg.get("short_reply_hint", False)),
            "style": str(prompt_cfg.get("style", "default")).strip(),
            "system_template": prompt_cfg.get("system_template"),
            "model_family": model_family,
        }

        with timer(logger, "load_dataset"):
            ds = load_dataset("json", data_files=str(test_path), split="train")
        logger.info(f"Using eval dataset: {test_path}")
        logger.info("test_dataset_size=%s", len(ds))

        run_baseline = bool(eval_cfg.get("run_baseline", False))
        compute_bleu_flag = bool(eval_cfg.get("compute_bleu", True))
        compute_chrf_flag = bool(eval_cfg.get("compute_chrf", True))
        compute_bert_flag = bool(eval_cfg.get("compute_bertscore", False))

        out_cfg = cfg.get("outputs", {}) or {}
        include_examples = bool(out_cfg.get("include_samples", out_cfg.get("include_examples", False)))
        num_examples = int(out_cfg.get("num_samples_per_lang", out_cfg.get("num_examples_per_lang", 3)))
        samples_format = (out_cfg.get("samples_format") or "blocks").strip().lower()
        env_targets = get_langs()["targets"]
        cfg_langs = eval_cfg.get("langs")
        langs = cfg_langs if cfg_langs else env_targets
        n = int(eval_cfg["num_samples_per_lang"])
        logger.info(
            "evaluation_langs=%s num_samples_per_lang=%s run_baseline=%s compute_bleu=%s compute_chrf=%s bertscore=%s",
            langs,
            n,
            run_baseline,
            compute_bleu_flag,
            compute_chrf_flag,
            compute_bert_flag,
        )

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
            gen_kwargs, gen_meta = _resolve_generation_kwargs(eval_cfg, tok)

        with timer(logger, "load_model_weights"):
            model = _load_eval_model(base, model_cfg, logger)

        lines = [
            "# Evaluation Report: Zero-shot vs Fine-tuned\n\n",
            "## Run settings\n\n",
            f"- Base model: {base}\n",
            f"- Adapter: {adapter if adapter else 'none'}\n",
            f"- Languages: {', '.join(langs)}\n",
            f"- Eval dataset: {test_path}\n",
            f"- Eval dataset rows (file): {len(ds)}\n",
            f"- Requested samples per language (cap): {n}\n",
            f"- Actually evaluated per language: {', '.join(f'{l}={len(buckets[l])}' for l in langs)}\n",
            f"- Decoding: `do_sample={gen_meta['do_sample']}`, `temperature={gen_meta['temperature']}`, "
            f"`top_p={gen_meta['top_p']}`, `num_beams={gen_meta['num_beams']}`, "
            f"`max_new_tokens={gen_meta['max_new_tokens']}`, `repetition_penalty={gen_meta['repetition_penalty']}`, "
            f"`eos_token_id_mode={gen_meta['eos_token_id_mode']}`\n",
            f"- Metrics: BLEU={compute_bleu_flag}, chrF={compute_chrf_flag}, BERTScore={compute_bert_flag} (optional)\n",
            f"- BLEU tokenizer (sacrebleu): `{bleu_tokenizer}`\n",
            f"- Chat formatting: `{model_family}` (Gemma uses `user`/`model` roles with system folded into the first user turn)\n\n",
        ]

        base_results = None

        if run_baseline:
            lines.append("## Zero-shot (base model, no fine-tuning)\n\n")
            lines.append("Generated outputs using the base model only (no LoRA).\n\n")
            base_results = run_model_on_buckets(model, tok, buckets, langs, gen_kwargs, prompt_kwargs)

        if adapter and adapter.exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()

        if adapter and adapter.exists():
            lines.append("## Fine-tuned (LoRA)\n\n")
            lines.append("Generated outputs using the base model with the trained LoRA adapter applied.\n\n")
            lora_results = run_model_on_buckets(model, tok, buckets, langs, gen_kwargs, prompt_kwargs)
        else:
            lines.append("## Fine-tuned (LoRA)\n\n")
            lines.append("*Adapter not found; run step 6 first.*\n\n")
            lora_results = {l: {"refs": [], "hyps": [], "prompts": [], "meta": [], "langid_ok": 0} for l in langs}

        lines.append("## Summary (metrics table)\n\n")
        if compute_bleu_flag or compute_chrf_flag:
            extra_headers = ""
            extra_separator = ""
            if compute_bleu_flag:
                extra_headers += " | Zero-shot BLEU | Fine-tuned BLEU"
                extra_separator += "|---:|---:"
            if compute_chrf_flag:
                extra_headers += " | Zero-shot chrF | Fine-tuned chrF"
                extra_separator += "|---:|---:"
            if compute_bert_flag:
                extra_headers += " | Zero-shot BERT F1 | Fine-tuned BERT F1"
                extra_separator += "|---:|---:"
            lines.append(f"| Lang | Samples | Zero-shot LangID | Fine-tuned LangID{extra_headers} |\n")
            lines.append(f"|---|---:|---:|---:{extra_separator}|\n")
        else:
            lines.append("| Lang | Samples | Zero-shot LangID | Fine-tuned LangID |\n")
            lines.append("|---|---:|---:|---:|\n")

        overall_base_refs, overall_base_hyps = [], []
        overall_lora_refs, overall_lora_hyps = [], []
        per_lang_metrics = {}

        for l in langs:
            count = len(buckets[l])
            z_ok = base_results[l]["langid_ok"] if base_results and run_baseline else 0
            f_ok = lora_results[l]["langid_ok"] if lora_results else 0
            z_bleu = f_bleu = z_chrf = f_chrf = z_bert = f_bert = None

            if base_results and run_baseline and base_results[l].get("refs"):
                if compute_bleu_flag:
                    z_bleu = compute_bleu(base_results[l]["refs"], base_results[l]["hyps"], bleu_tokenizer)
                if compute_chrf_flag:
                    z_chrf = compute_chrf(base_results[l]["refs"], base_results[l]["hyps"])
                if compute_bert_flag:
                    z_bert = compute_bertscore_optional(base_results[l]["hyps"], base_results[l]["refs"], l)
                overall_base_refs.extend(base_results[l]["refs"])
                overall_base_hyps.extend(base_results[l]["hyps"])
            if lora_results and lora_results[l].get("refs"):
                if compute_bleu_flag:
                    f_bleu = compute_bleu(lora_results[l]["refs"], lora_results[l]["hyps"], bleu_tokenizer)
                if compute_chrf_flag:
                    f_chrf = compute_chrf(lora_results[l]["refs"], lora_results[l]["hyps"])
                if compute_bert_flag:
                    f_bert = compute_bertscore_optional(lora_results[l]["hyps"], lora_results[l]["refs"], l)
                overall_lora_refs.extend(lora_results[l]["refs"])
                overall_lora_hyps.extend(lora_results[l]["hyps"])

            if compute_bleu_flag or compute_chrf_flag or compute_bert_flag:
                suffix = ""
                if compute_bleu_flag:
                    suffix += f" | {_fmt_bleu(z_bleu)} | {_fmt_bleu(f_bleu)}"
                if compute_chrf_flag:
                    suffix += f" | {_fmt_bleu(z_chrf)} | {_fmt_bleu(f_chrf)}"
                if compute_bert_flag:
                    suffix += f" | {_fmt_bleu(z_bert)} | {_fmt_bleu(f_bert)}"
                lines.append(f"| {l} | {count} | {z_ok}/{count} | {f_ok}/{count}{suffix} |\n")
            else:
                lines.append(f"| {l} | {count} | {z_ok}/{count} | {f_ok}/{count} |\n")
            per_lang_metrics[l] = {
                "samples": count,
                "zero_shot": {
                    "langid_ok": z_ok,
                    "langid_ratio": _safe_ratio(z_ok, count),
                    "bleu": z_bleu,
                    "chrf": z_chrf,
                    "bertscore_f1": z_bert,
                },
                "fine_tuned": {
                    "langid_ok": f_ok,
                    "langid_ratio": _safe_ratio(f_ok, count),
                    "bleu": f_bleu,
                    "chrf": f_chrf,
                    "bertscore_f1": f_bert,
                },
            }
        lines.append("\n")

        overall_metrics = {
            "zero_shot": {"bleu": None, "chrf": None, "bertscore_f1": None},
            "fine_tuned": {"bleu": None, "chrf": None, "bertscore_f1": None},
        }
        if compute_bleu_flag or compute_chrf_flag or compute_bert_flag:
            lines.append("## Overall metrics\n\n")
            if run_baseline and overall_base_refs:
                if compute_bleu_flag:
                    overall_metrics["zero_shot"]["bleu"] = compute_bleu(overall_base_refs, overall_base_hyps, bleu_tokenizer)
                    lines.append(f"- Zero-shot BLEU (pooled): {_fmt_bleu(overall_metrics['zero_shot']['bleu'])}\n")
                if compute_chrf_flag:
                    overall_metrics["zero_shot"]["chrf"] = compute_chrf(overall_base_refs, overall_base_hyps)
                    lines.append(f"- Zero-shot chrF (pooled): {_fmt_bleu(overall_metrics['zero_shot']['chrf'])}\n")
                if compute_bert_flag:
                    overall_metrics["zero_shot"]["bertscore_f1"] = compute_bertscore_optional(
                        overall_base_hyps, overall_base_refs, langs[0] if langs else "en"
                    )
                    lines.append(f"- Zero-shot BERTScore F1 (pooled): {_fmt_bleu(overall_metrics['zero_shot']['bertscore_f1'])}\n")
            if overall_lora_refs:
                if compute_bleu_flag:
                    overall_metrics["fine_tuned"]["bleu"] = compute_bleu(overall_lora_refs, overall_lora_hyps, bleu_tokenizer)
                    lines.append(f"- Fine-tuned BLEU (pooled): {_fmt_bleu(overall_metrics['fine_tuned']['bleu'])}\n")
                if compute_chrf_flag:
                    overall_metrics["fine_tuned"]["chrf"] = compute_chrf(overall_lora_refs, overall_lora_hyps)
                    lines.append(f"- Fine-tuned chrF (pooled): {_fmt_bleu(overall_metrics['fine_tuned']['chrf'])}\n")
                if compute_bert_flag:
                    overall_metrics["fine_tuned"]["bertscore_f1"] = compute_bertscore_optional(
                        overall_lora_hyps, overall_lora_refs, langs[0] if langs else "en"
                    )
                    lines.append(f"- Fine-tuned BERTScore F1 (pooled): {_fmt_bleu(overall_metrics['fine_tuned']['bertscore_f1'])}\n")
            lines.append("\n")

        lines.append("## Language consistency (langdetect)\n\n")
        for l in langs:
            count = len(buckets[l])
            z_ok = per_lang_metrics[l]["zero_shot"]["langid_ok"]
            f_ok = per_lang_metrics[l]["fine_tuned"]["langid_ok"]
            lines.append(f"- `{l}`: zero-shot `{z_ok}/{count}`, fine-tuned `{f_ok}/{count}`\n")
        lines.append("\n")

        grouped: dict[str, Any] = {"emotion": {}, "dialog_act": {}}
        has_emotion = any(
            (m or {}).get("emotion_at_turn") not in (None, "")
            for la in langs
            for m in lora_results[la].get("meta", [])
        )
        has_act = any(
            (m or {}).get("act_at_turn") not in (None, "")
            for la in langs
            for m in lora_results[la].get("meta", [])
        )
        if has_emotion or has_act:
            lines.append("## Grouped metrics (when labels exist)\n\n")
            for l in langs:
                grouped[l] = {}
                bl = base_results[l] if base_results and run_baseline else {}
                if has_emotion:
                    z_grp = _group_metrics_by_label(
                        bl,
                        "emotion_at_turn",
                        compute_bleu_flag=compute_bleu_flag,
                        compute_chrf_flag=compute_chrf_flag,
                        compute_bert_flag=compute_bert_flag,
                        bert_lang=l,
                        bleu_tokenizer=bleu_tokenizer,
                    )
                    f_grp = _group_metrics_by_label(
                        lora_results[l],
                        "emotion_at_turn",
                        compute_bleu_flag=compute_bleu_flag,
                        compute_chrf_flag=compute_chrf_flag,
                        compute_bert_flag=compute_bert_flag,
                        bert_lang=l,
                        bleu_tokenizer=bleu_tokenizer,
                    )
                    grouped[l]["by_emotion"] = {"zero_shot": z_grp, "fine_tuned": f_grp}
                    lines.append(
                        f"- `{l}` emotion groups: zero-shot `{len(z_grp)}` labels, fine-tuned `{len(f_grp)}` labels\n"
                    )
                if has_act:
                    z_grp = _group_metrics_by_label(
                        bl,
                        "act_at_turn",
                        compute_bleu_flag=compute_bleu_flag,
                        compute_chrf_flag=compute_chrf_flag,
                        compute_bert_flag=compute_bert_flag,
                        bert_lang=l,
                        bleu_tokenizer=bleu_tokenizer,
                    )
                    f_grp = _group_metrics_by_label(
                        lora_results[l],
                        "act_at_turn",
                        compute_bleu_flag=compute_bleu_flag,
                        compute_chrf_flag=compute_chrf_flag,
                        compute_bert_flag=compute_bert_flag,
                        bert_lang=l,
                        bleu_tokenizer=bleu_tokenizer,
                    )
                    grouped[l]["by_dialog_act"] = {"zero_shot": z_grp, "fine_tuned": f_grp}
                    lines.append(
                        f"- `{l}` dialog-act groups: zero-shot `{len(z_grp)}` labels, fine-tuned `{len(f_grp)}` labels\n"
                    )
            lines.append("\n")

        if include_examples and run_baseline and base_results is not None and adapter and adapter.exists():
            if samples_format == "table":
                _append_samples_table(
                    lines,
                    title="Sample outputs (table)",
                    langs=langs,
                    base_results=base_results,
                    lora_results=lora_results,
                    max_rows=num_examples,
                )
            else:
                _append_examples(
                    lines,
                    title="Sample outputs (blocks)",
                    langs=langs,
                    base_results=base_results,
                    lora_results=lora_results,
                    max_examples=num_examples,
                )

        lines.append("## Note on sample cap\n\n")
        lines.append(
            f"The evaluator stops after `num_samples_per_lang={n}` per language (see config). "
            f"Metrics and generations reflect up to that many examples per language, not necessarily the full test file.\n\n"
        )

        report_path = cfg["outputs"].get("report_path", "eval_report.md")
        out_path = resolve_path(report_path, dirs["reports"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")
        predictions_path = resolve_path(cfg["outputs"].get("predictions_path", "generations.jsonl"), dirs["reports"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_rows = []
        for l in langs:
            totals = [len(lora_results[l]["hyps"]), len(lora_results[l]["refs"])]
            if base_results and run_baseline:
                totals.append(len(base_results[l]["hyps"]))
            total = min(totals) if totals else 0
            for i in range(total):
                meta = (lora_results[l].get("meta") or [{}])[i] or {}
                prediction_rows.append(
                    {
                        "lang": l,
                        "dialogue_id": meta.get("dialogue_id"),
                        "turn_index": meta.get("turn_index"),
                        "emotion_at_turn": meta.get("emotion_at_turn"),
                        "act_at_turn": meta.get("act_at_turn"),
                        "prompt": (lora_results[l].get("prompts") or [""])[i],
                        "reference": (lora_results[l].get("refs") or [""])[i],
                        "zero_shot": (base_results[l].get("hyps") or [""])[i] if base_results else "",
                        "fine_tuned": (lora_results[l].get("hyps") or [""])[i],
                    }
                )
        with predictions_path.open("w", encoding="utf-8") as w:
            for row in prediction_rows:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")

        metrics_path = resolve_path(cfg["outputs"].get("metrics_path", "eval_metrics.json"), dirs["reports"])
        metrics_payload = {
            "script": "07_eval.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": base,
            "adapter_dir": str(adapter) if adapter else None,
            "eval_dataset_path": str(test_path),
            "requested_samples_per_lang": n,
            "evaluated_samples_total": int(sum(len(buckets[l]) for l in langs)),
            "bleu_tokenizer": bleu_tokenizer,
            "generation": gen_meta,
            "prompt_options": prompt_kwargs,
            "langs": langs,
            "metrics": {
                "per_lang": per_lang_metrics,
                "overall": overall_metrics,
                "grouped": grouped,
            },
            "language_consistency": {
                l: {
                    "zero_shot_ok": per_lang_metrics[l]["zero_shot"]["langid_ok"],
                    "fine_tuned_ok": per_lang_metrics[l]["fine_tuned"]["langid_ok"],
                    "samples": per_lang_metrics[l]["samples"],
                }
                for l in langs
            },
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("eval_report_path=%s eval_sample_count_total=%s", out_path, sum(len(buckets[l]) for l in langs))
        logger.info("predictions_path=%s metrics_path=%s", predictions_path, metrics_path)
        banner(logger, "Step 07: Done", char="-")
    except Exception:
        logger.exception("Step 07 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
