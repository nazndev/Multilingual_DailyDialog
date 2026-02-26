import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/translation.yaml", help="Translation config (for out_dir)")
    args = ap.parse_args()
    logger = setup_logger("04_quality_checks")
    banner(logger, "Step 04: Quality Checks")
    log_env_safely(logger, ["DATA_DIR", "REPORTS_DIR"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        out_dir = resolve_path(cfg.get("out_dir", "translated_dailydialog_en_bn_ar_es"), dirs["data"])
        reports_dir = dirs["reports"]
        reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("input_dir=%s reports_dir=%s", out_dir, reports_dir)

        flags = Counter()
        total = 0
        severe = 0
        for split in ["train", "validation", "test"]:
            p = out_dir / f"{split}.jsonl"
            if not p.exists():
                logger.warning("missing input path=%s", p)
                continue
            logger.info("input path=%s", p)
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    rec = json.loads(line)
                    qf = rec.get("translation_meta", {}).get("quality_flags", [])
                    for fl in qf:
                        flags[fl] += 1
                    if any(fl.startswith("integrity_") for fl in qf):
                        severe += 1
        logger.info("total_records=%s severe_integrity=%s (%.2f%%)", total, severe, (severe / max(1, total)) * 100)
        for k, v in flags.most_common():
            logger.info("flag %s: %s", k, v)

        with timer(logger, "write_reports"):
            report = [
                "# Translation Quality Report\n",
                f"- Total: {total}\n",
                f"- Severe(integrity): {severe} ({(severe / max(1, total)) * 100:.2f}%)\n\n",
                "## Flags\n",
            ]
            for k, v in flags.most_common():
                report.append(f"- {k}: {v}\n")
            report_path = reports_dir / "translation_quality_report.md"
            report_path.write_text("".join(report), encoding="utf-8")
            (out_dir / "quality_summary.json").write_text(
                json.dumps({"total": total, "severe": severe, "flags": dict(flags)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info("output report_path=%s quality_summary=%s", report_path, out_dir / "quality_summary.json")
        banner(logger, "Step 04: Done", char="-")
    except Exception:
        logger.exception("Step 04 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
