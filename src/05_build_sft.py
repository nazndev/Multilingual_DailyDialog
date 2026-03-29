"""Build chat-format SFT data for multilingual next-utterance generation."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_langs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer, summarize_jsonl
from src.utils.prompting import build_system_prompt

# -----------------------------------------------------------------------------
# DailyDialog label conventions (utterance-level emotion / act IDs).
# Integer IDs match common DailyDialog preprocessing; names are stable lowercase.
# -----------------------------------------------------------------------------
EMOTION_ID_TO_NAME: dict[int, str] = {
    0: "no_emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}

DIALOG_ACT_ID_TO_NAME: dict[int, str] = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}

_KNOWN_EMOTION_NAMES = frozenset(EMOTION_ID_TO_NAME.values())
_KNOWN_ACT_NAMES = frozenset(DIALOG_ACT_ID_TO_NAME.values())

ALLOWED_PROMPT_KEYS = frozenset({"style", "short_reply_hint", "system_template"})


def map_emotion_label(value: int | str | None) -> str | None:
    """
    Map a DailyDialog emotion field to a canonical symbolic name, or None if unknown / empty.

    Accepts integer IDs, numeric strings, or already-symbolic strings (case-insensitive).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return EMOTION_ID_TO_NAME.get(value)
    if isinstance(value, float):
        if value != value or int(value) != value:  # NaN or non-integral
            return None
        return EMOTION_ID_TO_NAME.get(int(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return EMOTION_ID_TO_NAME.get(int(s))
            except ValueError:
                return None
        low = s.lower()
        if low in _KNOWN_EMOTION_NAMES:
            return low
        return None
    return None


def map_dialog_act_label(value: int | str | None) -> str | None:
    """Map a DailyDialog dialog-act field to a canonical symbolic name, or None if unknown / empty."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return DIALOG_ACT_ID_TO_NAME.get(value)
    if isinstance(value, float):
        if value != value or int(value) != value:
            return None
        return DIALOG_ACT_ID_TO_NAME.get(int(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return DIALOG_ACT_ID_TO_NAME.get(int(s))
            except ValueError:
                return None
        low = s.lower()
        if low in _KNOWN_ACT_NAMES:
            return low
        return None
    return None


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


def _text_or_empty(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _valid_turns_list(turns) -> tuple[bool, str]:
    """Return (ok, reason) for a raw turns list."""
    if not isinstance(turns, list):
        return False, "not_a_list"
    for t in turns:
        if not isinstance(t, str):
            return False, "non_string_turn"
    return True, ""


def _turns_meet_min(turns: list, min_turns: int) -> bool:
    n = len([t for t in turns if isinstance(t, str) and t.strip()])
    return n >= min_turns


def _load_sft_cfg(cfg: dict) -> dict:
    """Read SFT builder options with backward-compatible defaults."""
    sft = cfg.get("sft", {}) if isinstance(cfg.get("sft", {}), dict) else {}
    prompt = sft.get("prompt") if isinstance(sft.get("prompt"), dict) else {}
    unknown = set(prompt.keys()) - ALLOWED_PROMPT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown sft.prompt key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(ALLOWED_PROMPT_KEYS)}."
        )
    return {
        "source_lang_field": str(sft.get("source_lang_field", "turns_en")),
        "target_lang_field": str(sft.get("target_lang_field", "turns_{lang}")),
        "context_window": int(sft.get("context_window", -1)),
        "use_context_window": int(sft.get("use_context_window", -1)),
        "use_emotion_tag": bool(sft.get("use_emotion_tag", False)),
        "use_dialog_act_tag": bool(sft.get("use_dialog_act_tag", False)),
        "min_turns": int(sft.get("min_turns", 2)),
        "max_samples": int(sft.get("max_samples", 0)),
        "output_format": str(sft.get("output_format", "jsonl")).strip().lower(),
        "prompt_style": str(prompt.get("style", "default")).strip(),
        "short_reply_hint": bool(prompt.get("short_reply_hint", False)),
        "system_template": prompt.get("system_template"),
        "max_history_chars": int(sft.get("max_history_chars", 0)),
    }


def _trim_history_messages(
    history_messages: list[dict],
    max_history_chars: int,
) -> list[dict]:
    """Drop oldest turns until serialized history fits ``max_history_chars`` (0 = no limit)."""
    if max_history_chars <= 0 or not history_messages:
        return history_messages

    def _join(h: list[dict]) -> str:
        return "\n".join((m.get("content") or "") for m in h)

    h = list(history_messages)
    while len(h) > 1 and len(_join(h)) > max_history_chars:
        h = h[1:]
    return h


def _build_history_before_target(
    target_turns: list[str],
    target_index: int,
    context_window: int,
) -> tuple[list[dict[str, str]], str | None]:
    """
    Build chronological chat history immediately before the assistant target turn.

    DailyDialog-style layout: even indices = user, odd = assistant. The **target**
    is always an assistant turn at odd ``target_index``. Only prior indices
    ``0 .. target_index - 1`` are included.

    If ``context_window > 0``, only the **last** ``context_window`` utterances
    before the target are kept (same chronological order).

    Returns (history_messages, skip_reason): skip_reason is set if this example
    must be dropped (empty slot inside the selected window, or empty history).
    """
    n = len(target_turns)
    if context_window <= 0:
        start = 0
    else:
        start = max(0, target_index - context_window)

    history_messages: list[dict[str, str]] = []
    for j in range(start, target_index):
        if j >= n:
            return [], "context_hole"
        target_text = _text_or_empty(target_turns[j])
        if not target_text:
            return [], "context_hole"
        role = "user" if j % 2 == 0 else "assistant"
        history_messages.append({"role": role, "content": target_text})

    if not history_messages:
        return [], "empty_history"
    return history_messages, None


def _label_at_turn(seq, index: int):
    """Safe index into emotions / dialog_acts list."""
    if not isinstance(seq, list) or index >= len(seq):
        return None
    return seq[index]


def _collect_examples_for_dialogue(
    *,
    target_turns: list[str],
    lang: str,
    dialogue_id,
    emotions,
    dialog_acts,
    context_window: int,
    use_emotion_tag: bool,
    use_dialog_act_tag: bool,
    prompt_style: str,
    short_reply_hint: bool,
    system_template,
    max_history_chars: int,
):
    """
    One SFT row per **assistant** target turn (odd indices 1, 3, 5, ...).

    Returns (examples, skip_stats).
    """
    out: list[dict] = []
    skip_stats = {
        "skipped_target_empty": 0,
        "skipped_empty_history": 0,
        "skipped_context_hole": 0,
    }
    n = len(target_turns)
    # Odd indices are assistant replies; we predict each assistant utterance from prior context.
    for target_index in range(1, n, 2):
        target_text = _text_or_empty(target_turns[target_index]) if target_index < n else ""
        if not target_text:
            skip_stats["skipped_target_empty"] += 1
            continue

        history_messages, skip_reason = _build_history_before_target(
            target_turns, target_index, context_window
        )
        if skip_reason == "empty_history":
            skip_stats["skipped_empty_history"] += 1
            continue
        if skip_reason == "context_hole":
            skip_stats["skipped_context_hole"] += 1
            continue

        history_messages = _trim_history_messages(history_messages, max_history_chars)
        if not history_messages:
            skip_stats["skipped_empty_history"] += 1
            continue

        raw_emotion = _label_at_turn(emotions, target_index)
        raw_act = _label_at_turn(dialog_acts, target_index)
        emotion_mapped = map_emotion_label(raw_emotion)
        dialog_act_mapped = map_dialog_act_label(raw_act)

        sys_prompt = build_system_prompt(
            lang,
            style=prompt_style,
            use_emotion_tag=use_emotion_tag,
            use_dialog_act_tag=use_dialog_act_tag,
            emotion=emotion_mapped if use_emotion_tag else None,
            dialog_act=dialog_act_mapped if use_dialog_act_tag else None,
            short_reply_hint=short_reply_hint,
            system_template=system_template,
        )
        ex = {
            "dialogue_id": dialogue_id,
            "turn_index": target_index,
            "lang": lang,
            "messages": [{"role": "system", "content": sys_prompt}]
            + history_messages
            + [{"role": "assistant", "content": target_text}],
        }
        if emotion_mapped is not None:
            ex["emotion_at_turn"] = emotion_mapped
        if dialog_act_mapped is not None:
            ex["act_at_turn"] = dialog_act_mapped
        ex["context_turns"] = len(history_messages)
        out.append(ex)
    return out, skip_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/translation_1000_api_bn.yaml", help="Translation config (for out_dir)")
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
            raise ValueError(
                f"Unsupported sft.output_format={sft_cfg['output_format']}. "
                "Use 'jsonl' or 'json'."
            )
        context_window = sft_cfg["context_window"] if sft_cfg["context_window"] >= 0 else sft_cfg["use_context_window"]
        in_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        sft_dir = resolve_path(cfg.get("sft_dir", "sft/multilingual"), dirs["data"])
        sft_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "input_dir=%s output_dir=%s context_window=%s prompt_style=%s short_reply_hint=%s "
            "use_emotion_tag=%s use_dialog_act_tag=%s",
            in_dir,
            sft_dir,
            context_window,
            sft_cfg["prompt_style"],
            sft_cfg["short_reply_hint"],
            sft_cfg["use_emotion_tag"],
            sft_cfg["use_dialog_act_tag"],
        )
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
            "context_window_effective": context_window,
            "prompt_style": sft_cfg["prompt_style"],
            "short_reply_hint": sft_cfg["short_reply_hint"],
            "use_emotion_tag": sft_cfg["use_emotion_tag"],
            "use_dialog_act_tag": sft_cfg["use_dialog_act_tag"],
            "output_examples_per_split": {},
            "skipped_dialogues": 0,
            "skipped_turns": 0,
            "skipped_target_empty": 0,
            "skipped_empty_history": 0,
            "skipped_context_hole": 0,
            "average_turns_per_dialogue": 0.0,
            "average_examples_per_dialogue": 0.0,
            "splits": {},
        }
        total_input_dialogues = 0
        total_skipped_dialogues = 0
        sum_turns_for_avg = 0
        dialogues_with_examples = 0
        total_sk_target = total_sk_hist = total_sk_hole = 0

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
            skipped_malformed = 0
            skipped_empty_turns = 0
            skipped_source_mismatch = 0
            skipped_target_empty = 0
            skipped_empty_history = 0
            skipped_context_hole = 0
            context_turn_sum = 0
            examples_buffer = []
            stop_early = False
            examples_per_dialogue: list[int] = []
            turns_per_dialogue: list[int] = []

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
                            target_turns = rec.get(target_key, [])
                            ok, reason = _valid_turns_list(target_turns)
                            if not ok:
                                if reason == "non_string_turn":
                                    skipped_empty_turns += 1
                                else:
                                    skipped_malformed += 1
                                continue
                            if not _turns_meet_min(target_turns, sft_cfg["min_turns"]):
                                continue
                            if any(not _text_or_empty(t) for t in target_turns):
                                continue

                            src_key = sft_cfg["source_lang_field"]
                            if src_key and src_key in rec:
                                source_turns = rec.get(src_key)
                                if not isinstance(source_turns, list):
                                    skipped_malformed += 1
                                    continue
                                if len(source_turns) != len(target_turns):
                                    skipped_source_mismatch += 1
                                    continue

                            ex_list, sk = _collect_examples_for_dialogue(
                                target_turns=target_turns,
                                lang=lang,
                                dialogue_id=did,
                                emotions=emotions,
                                dialog_acts=dialog_acts,
                                context_window=context_window,
                                use_emotion_tag=sft_cfg["use_emotion_tag"],
                                use_dialog_act_tag=sft_cfg["use_dialog_act_tag"],
                                prompt_style=sft_cfg["prompt_style"],
                                short_reply_hint=sft_cfg["short_reply_hint"],
                                system_template=sft_cfg["system_template"],
                                max_history_chars=sft_cfg["max_history_chars"],
                            )
                            skipped_target_empty += sk["skipped_target_empty"]
                            skipped_empty_history += sk["skipped_empty_history"]
                            skipped_context_hole += sk["skipped_context_hole"]
                            ex_count = 0
                            for ex in ex_list:
                                if sft_cfg["output_format"] == "jsonl":
                                    w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                                else:
                                    examples_buffer.append(ex)
                                written += 1
                                ex_count += 1
                                context_turn_sum += int(ex.get("context_turns", 0))
                                if sft_cfg["max_samples"] > 0 and written >= sft_cfg["max_samples"]:
                                    stop_early = True
                                    break
                            if ex_count > 0:
                                has_valid_lang = True
                                examples_per_dialogue.append(ex_count)
                                turns_per_dialogue.append(len(target_turns))
                            if stop_early:
                                break
                        if not has_valid_lang:
                            skipped_dialogues += 1
                        if stop_early:
                            break
                    if sft_cfg["output_format"] == "json":
                        w.write(json.dumps(examples_buffer, ensure_ascii=False, indent=2))
            avg_context_turns = round(context_turn_sum / written, 2) if written > 0 else 0.0
            avg_turns_per_d = round(sum(turns_per_dialogue) / len(turns_per_dialogue), 2) if turns_per_dialogue else 0.0
            avg_ex_per_d = round(sum(examples_per_dialogue) / len(examples_per_dialogue), 2) if examples_per_dialogue else 0.0
            skipped_turns_split = skipped_target_empty + skipped_empty_history + skipped_context_hole

            logger.info(
                "split=%s output=%s output_examples=%s input_dialogues=%s skipped_dialogues=%s "
                "skipped_malformed=%s skipped_empty=%s skipped_source_mismatch=%s "
                "skipped_turns=%s (target_empty=%s empty_history=%s context_hole=%s) avg_context_turns=%s",
                split,
                outp,
                written,
                processed_dialogues,
                skipped_dialogues,
                skipped_malformed,
                skipped_empty_turns,
                skipped_source_mismatch,
                skipped_turns_split,
                skipped_target_empty,
                skipped_empty_history,
                skipped_context_hole,
                avg_context_turns,
            )
            run_summary["splits"][split] = {
                "input_path": str(inp),
                "output_path": str(outp),
                "input_dialogues": processed_dialogues,
                "output_examples": written,
                "skipped_dialogues": skipped_dialogues,
                "skipped_malformed": skipped_malformed,
                "skipped_empty": skipped_empty_turns,
                "skipped_source_mismatch": skipped_source_mismatch,
                "skipped_turns": skipped_turns_split,
                "skipped_target_empty": skipped_target_empty,
                "skipped_empty_history": skipped_empty_history,
                "skipped_context_hole": skipped_context_hole,
                "avg_context_turns": avg_context_turns,
                "avg_turns_per_dialogue": avg_turns_per_d,
                "context_window": context_window,
                "prompt_style": sft_cfg["prompt_style"],
                "short_reply_hint": sft_cfg["short_reply_hint"],
                "use_emotion_tag": sft_cfg["use_emotion_tag"],
                "use_dialog_act_tag": sft_cfg["use_dialog_act_tag"],
                "generated_samples": written,
                "skipped_empty_or_invalid_turns": skipped_empty_turns,
            }
            run_summary["output_examples_per_split"][split] = written
            if sft_cfg["max_samples"] > 0 and written >= sft_cfg["max_samples"]:
                logger.warning("Reached sft.max_samples=%s for split=%s", sft_cfg["max_samples"], split)
            if sft_cfg["output_format"] == "jsonl":
                summarize_jsonl(logger, outp)

            total_input_dialogues += processed_dialogues
            total_skipped_dialogues += skipped_dialogues
            sum_turns_for_avg += sum(turns_per_dialogue)
            dialogues_with_examples += len(turns_per_dialogue)
            total_sk_target += skipped_target_empty
            total_sk_hist += skipped_empty_history
            total_sk_hole += skipped_context_hole

        run_summary["input_dialogues"] = total_input_dialogues
        run_summary["skipped_dialogues"] = total_skipped_dialogues
        run_summary["skipped_turns"] = total_sk_target + total_sk_hist + total_sk_hole
        run_summary["skipped_target_empty"] = total_sk_target
        run_summary["skipped_empty_history"] = total_sk_hist
        run_summary["skipped_context_hole"] = total_sk_hole
        if dialogues_with_examples > 0:
            run_summary["average_turns_per_dialogue"] = round(sum_turns_for_avg / dialogues_with_examples, 4)
        tw = sum(run_summary["output_examples_per_split"].values())
        if dialogues_with_examples > 0:
            run_summary["average_examples_per_dialogue"] = round(tw / dialogues_with_examples, 4)

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
