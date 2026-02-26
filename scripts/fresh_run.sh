#!/usr/bin/env bash
# Remove existing pipeline data (optional), then download from Hugging Face and run pipeline step by step.
# Usage:
#   ./scripts/fresh_run.sh              # run pipeline only (no clean)
#   ./scripts/fresh_run.sh clean        # clean then run quick pipeline
#   ./scripts/fresh_run.sh clean quick  # same
#   ./scripts/fresh_run.sh clean full   # clean then run full pipeline (bn+ar+es, more steps)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

DATA_DIR="${DATA_DIR:-./data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs}"
REPORTS_DIR="${REPORTS_DIR:-./reports}"
CACHE_DIR="${CACHE_DIR:-./cache}"

DO_CLEAN=false
MODE=quick
for arg in "$@"; do
  case "$arg" in
    clean)   DO_CLEAN=true ;;
    quick)   MODE=quick ;;
    full)    MODE=full ;;
    *)       echo "Unknown: $arg. Use: clean, quick, full"; exit 1 ;;
  esac
done

if [[ "$DO_CLEAN" == true ]]; then
  echo "== Cleaning existing pipeline data =="
  [[ -d "$DATA_DIR"    ]] && rm -rf "$DATA_DIR"/*
  [[ -d "$OUTPUTS_DIR" ]] && rm -rf "$OUTPUTS_DIR"/*
  [[ -d "$REPORTS_DIR" ]] && rm -rf "$REPORTS_DIR"/*
  [[ -d "$CACHE_DIR"   ]] && rm -rf "$CACHE_DIR"/*
  echo "Cleaned DATA_DIR, OUTPUTS_DIR, REPORTS_DIR, CACHE_DIR."
  # If only "clean" was requested, stop here.
  if [[ $# -eq 1 ]]; then
    echo "Done (clean only). Run ./scripts/fresh_run.sh clean quick or make fresh to re-download and run pipeline."
    exit 0
  fi
fi

echo "== [1/6] Download from Hugging Face =="
"$PYTHON" src/01_download.py

echo "== [2/6] Preprocess =="
"$PYTHON" src/02_preprocess.py

if [[ "$MODE" == full ]]; then
  echo "== [3/6] Translate =="
  "$PYTHON" src/03_translate.py --config configs/translation.yaml
  echo "== [4/6] Build SFT =="
  "$PYTHON" src/05_build_sft.py --config configs/translation.yaml
  echo "== [5/6] Train SFT (LoRA) =="
  "$PYTHON" src/06_train_sft.py --config configs/training_full.yaml
  echo "== [6/6] Evaluate (Zero-shot + Fine-tuned) =="
  "$PYTHON" src/07_eval.py --config configs/eval_full.yaml
else
  echo "== [3/6] Translate =="
  "$PYTHON" src/03_translate.py --config configs/translation.yaml
  echo "== [4/6] Build SFT =="
  "$PYTHON" src/05_build_sft.py --config configs/translation.yaml
  echo "== [5/6] Train SFT (LoRA) =="
  "$PYTHON" src/06_train_sft.py --config configs/training.yaml
  echo "== [6/6] Evaluate (Zero-shot + Fine-tuned) =="
  "$PYTHON" src/07_eval.py --config configs/eval.yaml
fi

echo "DONE. Data in DATA_DIR, reports in REPORTS_DIR, adapter in OUTPUTS_DIR."
