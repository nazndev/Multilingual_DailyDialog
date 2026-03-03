#!/usr/bin/env bash
# Quick mode end-to-end; exit non-zero if outputs missing, JSONL empty, or sample missing expected keys.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

export DATA_DIR="${DATA_DIR:-./data}"
export OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs}"
export REPORTS_DIR="${REPORTS_DIR:-./reports}"

echo "[smoke] 01 Download"
"$PYTHON" src/01_download.py

echo "[smoke] 02 Preprocess"
"$PYTHON" src/02_preprocess.py --config configs/preprocess_1000.yaml

echo "[smoke] 03 Translate"
TARGET_LANGS=bn TRANSLATION_BACKEND=api "$PYTHON" src/03_translate.py --config configs/translation_1000_api_bn.yaml

echo "[smoke] 04 Build SFT"
TARGET_LANGS=bn "$PYTHON" src/05_build_sft.py --config configs/translation_1000_api_bn.yaml

echo "[smoke] 05 Train SFT"
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct "$PYTHON" src/06_train_sft.py --config configs/training_1000.yaml

echo "[smoke] 06 Eval"
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct "$PYTHON" src/07_eval.py --config configs/eval_1000.yaml

echo "[smoke] Verifying outputs..."
"$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from src.utils.env import get_dirs
dirs = get_dirs()
data_dir = dirs["data"]
out_dir = dirs["outputs"]
reports_dir = dirs["reports"]

translated_dir = data_dir / "translated_dailydialog_en_bn_ar_es"
translated_dir = data_dir / "translated_api_1000_bn"
sft_dir = data_dir / "sft" / "multilingual_1000"
adapter_dir = out_dir / "model_1000" / "lora_adapter"

checks = []
# Required files
for p in [translated_dir / "test.jsonl", sft_dir / "test.jsonl", reports_dir / "eval_report_1000.md"]:
    if not p.exists():
        checks.append(f"Missing: {p}")
if not adapter_dir.exists():
    checks.append(f"Missing: {adapter_dir}")

# JSONL non-empty and expected keys
for path, keys in [
    (translated_dir / "test.jsonl", ["dialogue_id", "turns_en", "num_turns", "translation_meta"]),
    (sft_dir / "test.jsonl", ["dialogue_id", "lang", "messages"]),
]:
    if not path.exists():
        checks.append(f"Missing: {path}")
    else:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            checks.append(f"Empty JSONL: {path}")
        else:
            rec = json.loads(lines[0])
            for k in keys:
                if k not in rec:
                    checks.append(f"Missing key '{k}' in first record of {path}")

if checks:
    for c in checks:
        print(c, file=sys.stderr)
    sys.exit(1)
print("[smoke] OK — outputs present and valid.")
PY
