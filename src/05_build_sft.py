"""Build chat-format SFT data for multilingual next-utterance generation."""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_langs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer, summarize_jsonl


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _infer_langs(in_dir: Path) -> list:
    for split in ["train", "validation", "test"]:
        p = in_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            line = f.readline()
        if not line.strip():
            continue
        rec = json.loads(line)
        langs = [k.replace("turns_", "") for k in rec if k.startswith("turns_") and isinstance(rec.get(k), list)]
        if langs:
            return sorted(langs)
    return ["bn"]


def _text_or_empty(value):
    return value.strip() if isinstance(value, str) else ""


def _load_sft_cfg(cfg: dict) -> dict:
    """Read SFT builder options with backward-compatible defaults."""
    sft = cfg.get("sft", {}) if isinstance(cfg.get("sft", {}), dict) else {}
    return {
        "source_lang_field": sft.get("source_lang_field", "turns_en"),
        "target_lang_field": sft.get("target_lang_field", "turns_{lang}"),
        "context_window": int(sft.get("context_window", -1)),
        "use_context_window": int(sft.get("use_context_window", -1)),
        "use_emotion_tag": bool(sft.get("use_emotion_tag", False)),
        "use_dialog_act_tag": bool(sft.get("use_dialog_act_tag", False)),
        "min_turns": int(sft.get("min_turns", 2)),
        "max_samples": int(sft.get("max_samples", 0)),
        "output_format": str(sft.get("output_format", "jsonl")).strip().lower(),
    }


def _system_prompt(lang: str, use_emotion_tag: bool, use_dialog_act_tag: bool, emotion_tag: str, act_tag: str) -> str:
    """Task-aligned system prompt for multilingual next-utterance generation."""
    constraints = [
        f"You are an assistant for multilingual next-utterance generation in language '{lang}'.",
        "Generate only the next assistant utterance based on the dialogue context.",
        "Do not repeat the full history and do not add explanations or labels in the answer.",
        f"The output must be in '{lang}'.",
    ]
    if use_emotion_tag and emotion_tag:
        constraints.append(f"Target turn emotion label: {emotion_tag}.")
    if use_dialog_act_tag and act_tag:
        constraints.append(f"Target turn dialog act label: {act_tag}.")
    return " ".join(constraints)


def _build_examples_for_dialogue(
    turns,
    lang,
    did,
    emotions,
    dialog_acts,
    context_window,
    use_emotion_tag,
    use_dialog_act_tag,
):
    n = len(turns)
    for i in range(1, n, 2):
        target = _text_or_empty(turns[i]) if i < n else ""
        if not target:
            continue
        start = 0 if context_window <= 0 else max(0, i - context_window)
        hist = []
        for j in range(start, i):
            content = _text_or_empty(turns[j]) if j < n else ""
            if not content:
                continue
            hist.append({"role": "user" if j % 2 == 0 else "assistant", "content": content})
        if not hist:
            continue

        emotion_tag = ""
        act_tag = ""
        if isinstance(emotions, list) and i < len(emotions):
            emotion_tag = _text_or_empty(emotions[i])
        if isinstance(dialog_acts, list) and i < len(dialog_acts):
            act_tag = _text_or_empty(dialog_acts[i])

        sys_prompt = _system_prompt(
            lang=lang,
            use_emotion_tag=use_emotion_tag,
            use_dialog_act_tag=use_dialog_act_tag,
            emotion_tag=emotion_tag,
            act_tag=act_tag,
        )
        ex = {
            "dialogue_id": did,
            "turn_index": i,
            "lang": lang,
            "messages": [{"role": "system", "content": sys_prompt}] + hist + [{"role": "assistant", "content": target}],
        }
        if emotion_tag:
            ex["emotion_at_turn"] = emotion_tag
        if act_tag:
            ex["act_at_turn"] = act_tag
        ex["context_turns"] = len(hist)
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/translation.yaml", help="Translation config (for out_dir)")
    args = ap.parse_args()
    logger = setup_logger("05_build_sft")
    banner(logger, "Step 05: Build SFT")
    log_env_safely(logger, ["DATA_DIR", "TARGET_LANGS"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        sft_cfg = _load_sft_cfg(cfg)
        if sft_cfg["output_format"] not in {"jsonl", "json"}:
            raise ValueError(f"Unsupported sft.output_format={sft_cfg['output_format']}. Use 'jsonl' or 'json'.")
        context_window = sft_cfg["context_window"] if sft_cfg["context_window"] >= 0 else sft_cfg["use_context_window"]
        in_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        sft_dir = resolve_path(cfg.get("sft_dir", "sft/multilingual"), dirs["data"])
        sft_dir.mkdir(parents=True, exist_ok=True)
        logger.info("input_dir=%s output_dir=%s context_window=%s", in_dir, sft_dir, context_window)
        env_targets = get_langs()["targets"]
        inferred = _infer_langs(in_dir)
        langs = env_targets if env_targets else inferred
        logger.info("sft_languages=%s (env_targets=%s inferred=%s)", langs, env_targets, inferred)
        run_summary = {
            "script": "05_build_sft.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(in_dir),
            "output_dir": str(sft_dir),
            "langs": langs,
            "config": sft_cfg,
            "splits": {},
        }
        for split in ["train", "validation", "test"]:
            inp = in_dir / f"{split}.jsonl"
            if not inp.exists():
                logger.warning("missing input path=%s", inp)
                continue
            logger.info("input path=%s", inp)
            summarize_jsonl(logger, inp)
            outp = sft_dir / f"{split}.{'jsonl' if sft_cfg['output_format'] == 'jsonl' else 'json'}"
            written = 0
            processed_dialogues = 0
            skipped_dialogues = 0
            skipped_empty_turns = 0
            skipped_malformed = 0
            context_turn_sum = 0
            examples_buffer = []
            stop_early = False
            with timer(logger, f"build_sft_{split}"):
                with open(inp, "r", encoding="utf-8") as r, open(outp, "w", encoding="utf-8") as w:
                    for line in r:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            skipped_malformed += 1
                            continue
                        did = rec.get("dialogue_id")
                        if did is None:
                            skipped_malformed += 1
                            continue
                        processed_dialogues += 1
                        emotions = rec.get("emotions")
                        dialog_acts = rec.get("dialog_acts")
                        has_valid_lang = False
                        for lang in langs:
                            target_key = str(sft_cfg["target_lang_field"]).replace("{lang}", lang)
                            turns = rec.get(target_key, [])
                            if not isinstance(turns, list):
                                continue
                            if len(turns) < sft_cfg["min_turns"]:
                                continue
                            if any(not _text_or_empty(t) for t in turns):
                                skipped_empty_turns += 1
                                continue
                            has_valid_lang = True
                            for ex in _build_examples_for_dialogue(
                                turns=turns,
                                lang=lang,
                                did=did,
                                emotions=emotions,
                                dialog_acts=dialog_acts,
                                context_window=context_window,
                                use_emotion_tag=sft_cfg["use_emotion_tag"],
                                use_dialog_act_tag=sft_cfg["use_dialog_act_tag"],
                            ):
                                if sft_cfg["output_format"] == "jsonl":
                                    w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                                else:
                                    examples_buffer.append(ex)
                                written += 1
                                context_turn_sum += int(ex.get("context_turns", 0))
                                if sft_cfg["max_samples"] > 0 and written >= sft_cfg["max_samples"]:
                                    stop_early = True
                                    break
                            if stop_early:
                                break
                        if not has_valid_lang:
                            skipped_dialogues += 1
                        if stop_early:
                            break
                    if sft_cfg["output_format"] == "json":
                        w.write(json.dumps(examples_buffer, ensure_ascii=False, indent=2))
            avg_context_turns = round(context_turn_sum / written, 2) if written > 0 else 0.0
            logger.info(
                "output path=%s sft_examples=%s processed_dialogues=%s skipped_dialogues=%s skipped_malformed=%s skipped_empty=%s avg_context_turns=%s",
                outp,
                written,
                processed_dialogues,
                skipped_dialogues,
                skipped_malformed,
                skipped_empty_turns,
                avg_context_turns,
            )
            run_summary["splits"][split] = {
                "input_path": str(inp),
                "output_path": str(outp),
                "processed_dialogues": processed_dialogues,
                "generated_samples": written,
                "skipped_dialogues": skipped_dialogues,
                "skipped_malformed": skipped_malformed,
                "skipped_empty_or_invalid_turns": skipped_empty_turns,
                "avg_context_turns": avg_context_turns,
            }
            if sft_cfg["max_samples"] > 0 and written >= sft_cfg["max_samples"]:
                logger.warning("Reached sft.max_samples=%s for split=%s", sft_cfg["max_samples"], split)
            logger.info("output path=%s sft_examples=%s", outp, written)
            if sft_cfg["output_format"] == "jsonl":
                summarize_jsonl(logger, outp)
        summary_path = sft_dir / "build_sft_summary.json"
        summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("sft_summary_path=%s", summary_path)
        banner(logger, "Step 05: Done", char="-")
    except Exception:
        logger.exception("Step 05 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
