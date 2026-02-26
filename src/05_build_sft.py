import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, resolve_path
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


def build(turns, lang, did, emotions=None, dialog_acts=None):
    sys = f"You are a helpful assistant. Reply in {lang}."
    n = len(turns)
    for i in range(1, n, 2):
        hist = []
        for j in range(0, i):
            hist.append({"role": "user" if j % 2 == 0 else "assistant", "content": turns[j]})
        ex = {
            "dialogue_id": did,
            "turn_index": i,
            "lang": lang,
            "messages": [{"role": "system", "content": sys}] + hist + [{"role": "assistant", "content": turns[i]}],
        }
        if emotions is not None and i < len(emotions):
            ex["emotion_at_turn"] = emotions[i]
        if dialog_acts is not None and i < len(dialog_acts):
            ex["act_at_turn"] = dialog_acts[i]
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
        in_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        sft_dir = resolve_path(cfg.get("sft_dir", "sft/multilingual"), dirs["data"])
        sft_dir.mkdir(parents=True, exist_ok=True)
        logger.info("input_dir=%s output_dir=%s", in_dir, sft_dir)
        langs = _infer_langs(in_dir)
        logger.info("sft_languages=%s (from translated data)", langs)
        for split in ["train", "validation", "test"]:
            inp = in_dir / f"{split}.jsonl"
            if not inp.exists():
                logger.warning("missing input path=%s", inp)
                continue
            logger.info("input path=%s", inp)
            summarize_jsonl(logger, inp)
            outp = sft_dir / f"{split}.jsonl"
            written = 0
            with timer(logger, f"build_sft_{split}"):
                with open(inp, "r", encoding="utf-8") as r, open(outp, "w", encoding="utf-8") as w:
                    for line in r:
                        rec = json.loads(line)
                        did = rec["dialogue_id"]
                        emotions = rec.get("emotions")
                        dialog_acts = rec.get("dialog_acts")
                        for lang in langs:
                            turns = rec.get(f"turns_{lang}", [])
                            if not turns or any(t == "" for t in turns):
                                continue
                            for ex in build(turns, lang, did, emotions=emotions, dialog_acts=dialog_acts):
                                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                                written += 1
            logger.info("output path=%s sft_examples=%s", outp, written)
            summarize_jsonl(logger, outp)
        banner(logger, "Step 05: Done", char="-")
    except Exception:
        logger.exception("Step 05 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
