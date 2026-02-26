from __future__ import annotations

import os
import sys
from pathlib import Path

# Run from repo root so ui is importable
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports")).resolve()


def main() -> int:
    ok = True
    translated = None
    for cand in [DATA_DIR / "translated_dailydialog_en_bn_ar_es", DATA_DIR / "translated_dailydialog"]:
        if (cand / "test.jsonl").exists():
            translated = cand
            break
    if translated is None:
        print("[WARN] No translated dataset found under DATA_DIR.")
        ok = False
    else:
        print("[OK] translated_dir:", translated)

    report = None
    for cand in [REPORTS_DIR / "eval_report_full.md", REPORTS_DIR / "eval_report.md"]:
        if cand.exists():
            report = cand
            break
    if report is None:
        print("[WARN] No eval report found under REPORTS_DIR.")
    else:
        print("[OK] report:", report)

    app = Path("ui/app.py")
    if not app.exists():
        print("[ERR] ui/app.py missing")
        ok = False
    else:
        print("[OK] ui/app.py present")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
