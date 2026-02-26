import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, get_langs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer, summarize_jsonl

LANG_MAP = {"bn": "ben_Beng", "ar": "arb_Arab", "es": "spa_Latn", "en": "eng_Latn"}
# Human-readable names for LLM translation prompts
LANG_NAMES = {"bn": "Bengali", "ar": "Arabic", "es": "Spanish", "en": "English"}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def translate_one_local(model, tok, text: str, src: str = "en", tgt: str = "bn", device: str = "cpu", max_new_tokens: int = 256) -> str:
    """Translate one sentence using local NLLB model."""
    tok.src_lang = LANG_MAP[src]
    enc = tok([text], return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(LANG_MAP[tgt]), max_new_tokens=max_new_tokens)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]


def translate_one_api(cfg: dict, text: str, src: str = "en", tgt: str = "bn") -> str:
    """
    Translate one sentence using an LLM API (e.g. OpenAI GPT) for higher quality.
    Requires OPENAI_API_KEY in env when provider is openai.
    """
    api_cfg = cfg.get("api", {})
    provider = (api_cfg.get("provider") or "openai").lower()
    model_name = api_cfg.get("model") or "gpt-4o-mini"
    max_retries = int(api_cfg.get("max_retries", 3))
    sleep_sec = float(api_cfg.get("sleep_between_retries_sec", 1))
    target_lang = LANG_NAMES.get(tgt, tgt)

    prompt = (
        f"Translate the following English text to {target_lang}. "
        "Preserve meaning, tone, and naturalness. Output only the translation, no explanation or quotes.\n\n"
        f"English: {text}"
    )

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("OpenAI backend requires: pip install openai")
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set in environment for API translation")
        client = OpenAI(api_key=api_key)
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(api_cfg.get("max_tokens", 256)),
                    temperature=0.0,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(sleep_sec)
        raise last_err or RuntimeError("No response from API")

    raise ValueError(f"Unsupported translation API provider: {provider}. Use backend: local for NLLB or provider: openai with OPENAI_API_KEY.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logger = setup_logger("03_translate")
    banner(logger, "Step 03: Translate")
    log_env_safely(logger, ["DATA_DIR", "CACHE_DIR", "TARGET_LANGS", "TRANSLATION_MODEL"])
    try:
        dirs = get_dirs()
        langs = get_langs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        targets = langs["targets"] if langs["targets"] else cfg.get("targets", ["bn"])
        processed_dir = resolve_path(cfg.get("processed_dir", "processed"), dirs["data"])
        out_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        cache_dir = resolve_path(cfg["cache"]["dir"], dirs["cache"])
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("targets=%s data_dir=%s out_dir=%s cache_dir=%s", targets, dirs["data"], out_dir, cache_dir)

        backend = get_env("TRANSLATION_BACKEND") or cfg.get("backend") or "local"
        use_api = backend.lower() == "api"
        model_name = None
        tok, model, device = None, None, None

        if use_api:
            model_name = cfg.get("api", {}).get("model") or "gpt-4o-mini"
            logger.info("backend=api model=%s (set OPENAI_API_KEY for OpenAI)", model_name)
        else:
            model_name = get_env("TRANSLATION_MODEL") or cfg["local"]["model_name"]
            device = "cuda" if (cfg["local"]["device"] == "auto" and torch.cuda.is_available()) else "cpu"
            logger.info("backend=local model=%s device=%s", model_name, device)
            with timer(logger, "load_model"):
                tok = NllbTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

        max_per_split = cfg.get("max_dialogues_per_split")
        for split in ["train", "validation", "test"]:
            in_path = processed_dir / f"{split}_en.jsonl"
            if not in_path.exists():
                logger.warning("missing input path=%s", in_path)
                continue
            logger.info("input path=%s", in_path)
            summarize_jsonl(logger, in_path)
            out_path = out_dir / f"{split}.jsonl"
            cache_hits = 0
            cache_misses = 0
            written = 0
            records_with_flags = 0
            with open(in_path, "r", encoding="utf-8") as r, open(out_path, "w", encoding="utf-8") as w:
                for count, line in enumerate(tqdm(r, desc=f"Translate {split}")):
                    if max_per_split is not None and count >= max_per_split:
                        break
                    rec = json.loads(line)
                    turns_en = rec["turns_en"]
                    out = {
                        "dialogue_id": rec["dialogue_id"],
                        "turns_en": turns_en,
                        "num_turns": rec["num_turns"],
                        "translation_meta": {"backend": "api" if use_api else "local", "model": model_name, "quality_flags": []},
                    }
                    if "dialog_acts" in rec:
                        out["dialog_acts"] = rec["dialog_acts"]
                    if "emotions" in rec:
                        out["emotions"] = rec["emotions"]
                    for lang in targets:
                        t_out = []
                        for t in turns_en:
                            key = sha256(lang + "\n" + t)
                            c = cache_dir / f"{key}.txt"
                            if c.exists():
                                t_out.append(c.read_text(encoding="utf-8"))
                                cache_hits += 1
                            else:
                                cache_misses += 1
                                try:
                                    if use_api:
                                        tr = translate_one_api(cfg, t, "en", lang)
                                    else:
                                        tr = translate_one_local(model, tok, t, "en", lang, device, cfg["local"]["max_new_tokens"])
                                    c.write_text(tr, encoding="utf-8")
                                    t_out.append(tr)
                                except Exception:
                                    out["translation_meta"]["quality_flags"].append(f"translate_fail_{lang}")
                                    t_out.append("")
                        out[f"turns_{lang}"] = t_out
                        if len(t_out) != len(turns_en) or any(x == "" for x in t_out):
                            out["translation_meta"]["quality_flags"].append(f"integrity_{lang}")
                    if out["translation_meta"]["quality_flags"]:
                        records_with_flags += 1
                    w.write(json.dumps(out, ensure_ascii=False) + "\n")
                    written += 1
            total_lookups = cache_hits + cache_misses
            rate = (cache_hits / total_lookups * 100) if total_lookups else 0
            logger.info("output path=%s records=%s cache_hits=%s cache_misses=%s cache_hit_rate_pct=%.1f", out_path, written, cache_hits, cache_misses, rate)
            if records_with_flags:
                logger.warning("records_with_quality_flags=%s (of %s)", records_with_flags, written)
            summarize_jsonl(logger, out_path)
        banner(logger, "Step 03: Done", char="-")
    except Exception:
        logger.exception("Step 03 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
