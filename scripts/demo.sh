#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# REPORTS_DIR from env or default ./reports
REPORTS_DIR="${REPORTS_DIR:-./reports}"
LOG_DIR="$REPORTS_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/${STAMP}_pipeline.log"
echo "Master pipeline log: $MASTER_LOG" | tee -a "$MASTER_LOG"

run_step() {
  local n="$1" name="$2" rest=("${@:3}")
  echo "" | tee -a "$MASTER_LOG"
  echo "== [$n/6] $name ==" | tee -a "$MASTER_LOG"
  if "${rest[@]}" 2>&1 | tee -a "$MASTER_LOG"; then
    echo "[$n/6] $name OK" | tee -a "$MASTER_LOG"
  else
    echo "[$n/6] $name FAILED" | tee -a "$MASTER_LOG"
    exit 1
  fi
}

run_step "1" "Download" python src/01_download.py
run_step "2" "Preprocess" python src/02_preprocess.py
run_step "3" "Translate" python src/03_translate.py --config configs/translation.yaml
run_step "4" "Build SFT" python src/05_build_sft.py --config configs/translation.yaml
run_step "5" "Train SFT (LoRA)" python src/06_train_sft.py --config configs/training_full.yaml
run_step "6" "Evaluate (Zero-shot + Fine-tuned)" python src/07_eval.py --config configs/eval_full.yaml

echo "" | tee -a "$MASTER_LOG"
echo "DONE. Reports in REPORTS_DIR, adapters in OUTPUTS_DIR. Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
