"""Build last-turn-only chat-format SFT data for multilingual next-utterance generation."""

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
from src.utils.prompting import build_system_prompt, to_gemma_messages

# -----------------------------------------------------------------------------
# DailyDialog label conventions (utterance-level emotion / act IDs).
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
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return EMOTION_ID_TO_NAME.get(value)
    if isinstance(value, float):
        if value != value or int(value) != value:
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


def _safe_turn(turns: list, index: int) -> str:
    if not isinstance(turns, list) or index < 0 or index >= len(turns):
        return ""
    return _text_or_empty(turns[index])


def _valid_turns_list(turns) -> tuple[bool, str]:
    if not isinstance(turns, list):
        return False, "not_a_list"
    for t in turns:
        if not isinstance(t, str):
            return False, "non_string_turn"
    return True, ""


def _turns_meet_min(turns: list, min_turns: int) -> bool:
    n = len([t for t in turns if isinstance(t, str) and t.strip()])
    return n >= min_turns


def _load_sft_lastturn_cfg(cfg: dict) -> dict:
    """SFT options for last-turn builder; ignores context window / history settings."""
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
        "use_emotion_tag": bool(sft.get("use_emotion_tag", False)),
        "use_dialog_act_tag": bool(sft.get("use_dialog_act_tag", False)),
        "min_turns": int(sft.get("min_turns", 2)),
        "max_samples": int(sft.get("max_samples", 0)),
        "output_format": str(sft.get("output_format", "jsonl")).strip().lower(),
        "prompt_style": str(prompt.get("style", "default")).strip(),
        "short_reply_hint": bool(prompt.get("short_reply_hint", False)),
        "system_template": prompt.get("system_template"),
    }


def _label_at_turn(seq, index: int):
    if not isinstance(seq, list) or index >= len(seq):
        return None
    return seq[index]


def _build_qwen_lastturn_messages(
    *,
    lang: str,
    user_text: str,
    assistant_text: str,
    use_emotion_tag: bool,
    use_dialog_act_tag: bool,
    prompt_style: str,
    short_reply_hint: bool,
    system_template,
    emotion_mapped: str | None,
    dialog_act_mapped: str | None,
) -> list[dict[str, str]]:
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
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _build_gemma_lastturn_messages(qwen_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return to_gemma_messages(qwen_messages)


def _collect_lastturn_examples_for_dialogue(
    *,
    target_turns: list[str],
    lang: str,
    dialogue_id,
    emotions,
    dialog_acts,
    use_emotion_tag: bool,
    use_dialog_act_tag: bool,
    prompt_style: str,
    short_reply_hint: bool,
    system_template,
) -> tuple[list[dict], dict[str, int]]:
    out: list[dict] = []
    skip_stats = {
        "skipped_target_empty": 0,
        "skipped_user_empty": 0,
    }
    n = len(target_turns)
    for target_index in range(1, n, 2):
        user_text = _safe_turn(target_turns, target_index - 1)
        assistant_text = _safe_turn(target_turns, target_index)
        if not user_text:
            skip_stats["skipped_user_empty"] += 1
            continue
        if not assistant_text:
            skip_stats["skipped_target_empty"] += 1
            continue

        raw_emotion = _label_at_turn(emotions, target_index)
        raw_act = _label_at_turn(dialog_acts, target_index)
        emotion_mapped = map_emotion_label(raw_emotion)
        dialog_act_mapped = map_dialog_act_label(raw_act)

        qwen_messages = _build_qwen_lastturn_messages(
            lang=lang,
            user_text=user_text,
            assistant_text=assistant_text,
            use_emotion_tag=use_emotion_tag,
            use_dialog_act_tag=use_dialog_act_tag,
            prompt_style=prompt_style,
            short_reply_hint=short_reply_hint,
            system_template=system_template,
            emotion_mapped=emotion_mapped,
            dialog_act_mapped=dialog_act_mapped,
        )
        ex: dict = {
            "dialogue_id": dialogue_id,
            "turn_index": target_index,
            "lang": lang,
            "messages": qwen_messages,
        }
        if emotion_mapped is not None:
            ex["emotion_at_turn"] = emotion_mapped
        if dialog_act_mapped is not None:
            ex["act_at_turn"] = dialog_act_mapped
        out.append(ex)
    return out, skip_stats


def _messages_have_only_roles(messages: list[dict], allowed: set[str]) -> bool:
    return all(isinstance(m, dict) and (m.get("role") in allowed) for m in messages)


def _build_export_variants(example: dict) -> tuple[dict, dict]:
    qwen_messages = [dict(m) for m in (example.get("messages") or []) if isinstance(m, dict)]
    qwen = {
        "dialogue_id": example.get("dialogue_id"),
        "turn_index": example.get("turn_index"),
        "lang": example.get("lang"),
        "messages": qwen_messages,
        "target_role": "assistant",
        "target_text": (qwen_messages[-1].get("content") or "").strip() if qwen_messages else "",
    }
    if "emotion_at_turn" in example:
        qwen["emotion_at_turn"] = example["emotion_at_turn"]
    if "act_at_turn" in example:
        qwen["act_at_turn"] = example["act_at_turn"]

    gemma_messages = _build_gemma_lastturn_messages(qwen_messages)
    gemma = {
        "dialogue_id": example.get("dialogue_id"),
        "turn_index": example.get("turn_index"),
        "lang": example.get("lang"),
        "messages": gemma_messages,
        "target_role": "model",
        "target_text": (gemma_messages[-1].get("content") or "").strip() if gemma_messages else "",
    }
    if "emotion_at_turn" in example:
        gemma["emotion_at_turn"] = example["emotion_at_turn"]
    if "act_at_turn" in example:
        gemma["act_at_turn"] = example["act_at_turn"]
    return qwen, gemma


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_split(
    *,
    inp: Path,
    qwen_outp: Path,
    gemma_outp: Path,
    processed_dialogues: int,
    written: int,
    written_qwen: int,
    written_gemma: int,
    skipped_dialogues: int,
    skipped_malformed: int,
    skipped_empty_turns: int,
    skipped_source_mismatch: int,
    skipped_target_empty: int,
    skipped_user_empty: int,
    examples_per_dialogue: list[int],
    turns_per_dialogue: list[int],
    bad_qwen_roles: int,
    bad_gemma_roles: int,
    gemma_system_found: int,
    missing_target: int,
    sft_cfg: dict,
) -> dict:
    avg_turns_per_d = round(sum(turns_per_dialogue) / len(turns_per_dialogue), 2) if turns_per_dialogue else 0.0
    avg_ex_per_d = round(sum(examples_per_dialogue) / len(examples_per_dialogue), 2) if examples_per_dialogue else 0.0
    skipped_turns_split = skipped_target_empty + skipped_user_empty
    return {
        "input_path": str(inp),
        "output_paths": {
            "qwen": str(qwen_outp),
            "gemma": str(gemma_outp),
        },
        "input_dialogues": processed_dialogues,
        "output_examples": written,
        "output_examples_qwen": written_qwen,
        "output_examples_gemma": written_gemma,
        "skipped_dialogues": skipped_dialogues,
        "skipped_malformed": skipped_malformed,
        "skipped_empty": skipped_empty_turns,
        "skipped_source_mismatch": skipped_source_mismatch,
        "skipped_turns": skipped_turns_split,
        "skipped_target_empty": skipped_target_empty,
        "skipped_user_empty": skipped_user_empty,
        "avg_examples_per_dialogue": avg_ex_per_d,
        "avg_turns_per_dialogue": avg_turns_per_d,
        "prompt_style": sft_cfg["prompt_style"],
        "short_reply_hint": sft_cfg["short_reply_hint"],
        "use_emotion_tag": sft_cfg["use_emotion_tag"],
        "use_dialog_act_tag": sft_cfg["use_dialog_act_tag"],
        "generated_samples": written,
        "skipped_empty_or_invalid_turns": skipped_empty_turns,
        "validation": {
            "bad_qwen_roles": bad_qwen_roles,
            "bad_gemma_roles": bad_gemma_roles,
            "gemma_system_found": gemma_system_found,
            "missing_target": missing_target,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/translation_1000_api_bn.yaml", help="Translation config (for out_dir)")
    args = ap.parse_args()
    logger = setup_logger("05_build_sft_lastturn")
    banner(logger, "Step 05: Build SFT (Last Turn Only)")
    log_env_safely(logger, ["DATA_DIR", "TARGET_LANGS"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        sft_cfg = _load_sft_lastturn_cfg(cfg)
        if sft_cfg["output_format"] not in {"jsonl", "json"}:
            raise ValueError(
                f"Unsupported sft.output_format={sft_cfg['output_format']}. "
                "Use 'jsonl' or 'json'."
            )
        in_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        sft_dir = resolve_path(cfg.get("sft_dir", "sft/multilingual"), dirs["data"])
        qwen_dir = sft_dir.parent / "qwen_bn_lastturn"
        gemma_dir = sft_dir.parent / "gemma_bn_lastturn"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        gemma_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "input_dir=%s qwen_output_dir=%s gemma_output_dir=%s (last-turn-only; context/history settings ignored) "
            "prompt_style=%s short_reply_hint=%s use_emotion_tag=%s use_dialog_act_tag=%s",
            in_dir,
            qwen_dir,
            gemma_dir,
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
            "script": "05_build_sft_lastturn.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(in_dir),
            "output_dirs": {
                "qwen": str(qwen_dir),
                "gemma": str(gemma_dir),
            },
            "langs": langs,
            "config": sft_cfg,
            "builder_mode": "last_turn_only",
            "output_examples_per_split": {},
            "output_examples_per_split_by_model": {"qwen": {}, "gemma": {}},
            "skipped_dialogues": 0,
            "skipped_turns": 0,
            "skipped_target_empty": 0,
            "skipped_user_empty": 0,
            "average_turns_per_dialogue": 0.0,
            "average_examples_per_dialogue": 0.0,
            "splits": {},
            "validation": {
                "qwen_roles_only": True,
                "gemma_roles_only": True,
                "gemma_has_no_system": True,
                "target_span_present_non_empty": True,
            },
        }
        total_input_dialogues = 0
        total_skipped_dialogues = 0
        sum_turns_for_avg = 0
        dialogues_with_examples = 0
        total_sk_target = total_sk_user = 0

        for split in ["train", "validation", "test"]:
            inp = in_dir / f"{split}.jsonl"
            if not inp.exists():
                logger.warning("missing input path=%s", inp)
                continue
            logger.info("input path=%s", inp)
            summarize_jsonl(logger, inp)
            qwen_outp = qwen_dir / f"{split}.jsonl"
            gemma_outp = gemma_dir / f"{split}.jsonl"
            written = 0
            written_qwen = 0
            written_gemma = 0
            processed_dialogues = 0
            skipped_dialogues = 0
            skipped_malformed = 0
            skipped_empty_turns = 0
            skipped_source_mismatch = 0
            skipped_target_empty = 0
            skipped_user_empty = 0
            bad_qwen_roles = 0
            bad_gemma_roles = 0
            gemma_system_found = 0
            missing_target = 0
            stop_early = False
            examples_per_dialogue: list[int] = []
            turns_per_dialogue: list[int] = []

            qwen_rows: list[dict] = []
            gemma_rows: list[dict] = []
            with timer(logger, f"build_sft_lastturn_{split}"):
                with open(inp, "r", encoding="utf-8") as r:
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

                            src_key = sft_cfg["source_lang_field"]
                            if src_key and src_key in rec:
                                source_turns = rec.get(src_key)
                                if not isinstance(source_turns, list):
                                    skipped_malformed += 1
                                    continue
                                if len(source_turns) != len(target_turns):
                                    skipped_source_mismatch += 1
                                    continue

                            ex_list, sk = _collect_lastturn_examples_for_dialogue(
                                target_turns=target_turns,
                                lang=lang,
                                dialogue_id=did,
                                emotions=emotions,
                                dialog_acts=dialog_acts,
                                use_emotion_tag=sft_cfg["use_emotion_tag"],
                                use_dialog_act_tag=sft_cfg["use_dialog_act_tag"],
                                prompt_style=sft_cfg["prompt_style"],
                                short_reply_hint=sft_cfg["short_reply_hint"],
                                system_template=sft_cfg["system_template"],
                            )
                            skipped_target_empty += sk["skipped_target_empty"]
                            skipped_user_empty += sk["skipped_user_empty"]
                            ex_count = 0
                            for ex in ex_list:
                                qwen_ex, gemma_ex = _build_export_variants(ex)
                                qwen_roles_ok = _messages_have_only_roles(
                                    qwen_ex["messages"], {"system", "user", "assistant"}
                                )
                                gemma_roles_ok = _messages_have_only_roles(
                                    gemma_ex["messages"], {"user", "model"}
                                )
                                if not qwen_roles_ok:
                                    bad_qwen_roles += 1
                                if not gemma_roles_ok:
                                    bad_gemma_roles += 1
                                if any((m.get("role") == "system") for m in gemma_ex["messages"]):
                                    gemma_system_found += 1
                                if not qwen_ex.get("target_text") or not gemma_ex.get("target_text"):
                                    missing_target += 1

                                qwen_rows.append(qwen_ex)
                                gemma_rows.append(gemma_ex)
                                written += 1
                                written_qwen += 1
                                written_gemma += 1
                                ex_count += 1
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
            _write_jsonl(qwen_outp, qwen_rows)
            _write_jsonl(gemma_outp, gemma_rows)

            avg_turns_per_d = round(sum(turns_per_dialogue) / len(turns_per_dialogue), 2) if turns_per_dialogue else 0.0
            avg_ex_per_d = round(sum(examples_per_dialogue) / len(examples_per_dialogue), 2) if examples_per_dialogue else 0.0
            skipped_turns_split = skipped_target_empty + skipped_user_empty

            logger.info(
                "split=%s input_dialogues=%s output_examples=%s avg_examples_per_dialogue=%s "
                "skipped_dialogues=%s skipped_malformed=%s skipped_empty_turn_struct=%s "
                "skipped_source_mismatch=%s skipped_turns=%s (target_empty=%s user_empty=%s)",
                split,
                processed_dialogues,
                written,
                avg_ex_per_d,
                skipped_dialogues,
                skipped_malformed,
                skipped_empty_turns,
                skipped_source_mismatch,
                skipped_turns_split,
                skipped_target_empty,
                skipped_user_empty,
            )
            logger.info("split=%s qwen_out=%s gemma_out=%s avg_turns_per_dialogue=%s", split, qwen_outp, gemma_outp, avg_turns_per_d)

            run_summary["splits"][split] = _summarize_split(
                inp=inp,
                qwen_outp=qwen_outp,
                gemma_outp=gemma_outp,
                processed_dialogues=processed_dialogues,
                written=written,
                written_qwen=written_qwen,
                written_gemma=written_gemma,
                skipped_dialogues=skipped_dialogues,
                skipped_malformed=skipped_malformed,
                skipped_empty_turns=skipped_empty_turns,
                skipped_source_mismatch=skipped_source_mismatch,
                skipped_target_empty=skipped_target_empty,
                skipped_user_empty=skipped_user_empty,
                examples_per_dialogue=examples_per_dialogue,
                turns_per_dialogue=turns_per_dialogue,
                bad_qwen_roles=bad_qwen_roles,
                bad_gemma_roles=bad_gemma_roles,
                gemma_system_found=gemma_system_found,
                missing_target=missing_target,
                sft_cfg=sft_cfg,
            )
            run_summary["output_examples_per_split"][split] = written
            run_summary["output_examples_per_split_by_model"]["qwen"][split] = written_qwen
            run_summary["output_examples_per_split_by_model"]["gemma"][split] = written_gemma
            run_summary["validation"]["qwen_roles_only"] = run_summary["validation"]["qwen_roles_only"] and (
                bad_qwen_roles == 0
            )
            run_summary["validation"]["gemma_roles_only"] = run_summary["validation"]["gemma_roles_only"] and (
                bad_gemma_roles == 0
            )
            run_summary["validation"]["gemma_has_no_system"] = run_summary["validation"]["gemma_has_no_system"] and (
                gemma_system_found == 0
            )
            run_summary["validation"]["target_span_present_non_empty"] = run_summary["validation"][
                "target_span_present_non_empty"
            ] and (missing_target == 0)
            if sft_cfg["max_samples"] > 0 and written >= sft_cfg["max_samples"]:
                logger.warning("Reached sft.max_samples=%s for split=%s", sft_cfg["max_samples"], split)
            summarize_jsonl(logger, qwen_outp)
            summarize_jsonl(logger, gemma_outp)

            total_input_dialogues += processed_dialogues
            total_skipped_dialogues += skipped_dialogues
            sum_turns_for_avg += sum(turns_per_dialogue)
            dialogues_with_examples += len(turns_per_dialogue)
            total_sk_target += skipped_target_empty
            total_sk_user += skipped_user_empty

        run_summary["input_dialogues"] = total_input_dialogues
        run_summary["skipped_dialogues"] = total_skipped_dialogues
        run_summary["skipped_turns"] = total_sk_target + total_sk_user
        run_summary["skipped_target_empty"] = total_sk_target
        run_summary["skipped_user_empty"] = total_sk_user
        if dialogues_with_examples > 0:
            run_summary["average_turns_per_dialogue"] = round(sum_turns_for_avg / dialogues_with_examples, 4)
        tw = sum(run_summary["output_examples_per_split"].values())
        if dialogues_with_examples > 0:
            run_summary["average_examples_per_dialogue"] = round(tw / dialogues_with_examples, 4)

        summary_path = sft_dir.parent / "build_sft_lastturn_summary.json"
        summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("sft_lastturn_summary_path=%s", summary_path)
        banner(logger, "Step 05: Done (Last Turn Only)", char="-")
    except Exception:
        logger.exception("Step 05 (last turn) failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
