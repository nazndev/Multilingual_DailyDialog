# Multilingual DailyDialog — Full Command Runbook (Colab / Local)

## 1. Mount Google Drive (Colab only)
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2. Setup Persistent Workspace
```bash
export WORKDIR="/content/drive/MyDrive/Multilingual_DailyDialog"
mkdir -p $WORKDIR
cd $WORKDIR
```

---

## 3. Clone Repo (first time only)
```bash
if [ ! -d "Multilingual_DailyDialog" ]; then
  git clone https://github.com/nazndev/Multilingual_DailyDialog.git
fi

cd Multilingual_DailyDialog
```

---

## 4. Always Pull Latest Code
```bash
git pull origin main
```

---

## 5. Setup Environment Variables
```bash
export DATA_DIR="$WORKDIR/data"
export OUTPUTS_DIR="$WORKDIR/outputs"
export REPORTS_DIR="$WORKDIR/reports"

mkdir -p $DATA_DIR $OUTPUTS_DIR $REPORTS_DIR
```

---

## 6. Install Dependencies
```bash
pip install -r requirements.txt
pip install bitsandbytes accelerate peft trl sacrebleu langdetect
```

---

## 7. Build SFT (MANDATORY after prompt change)
```bash
TARGET_LANGS=bn python src/05_build_sft.py \
  --config configs/translation_1000_api_bn.yaml
```

---

## 8. Train 0.5B Baseline
```bash
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

python src/06_train_sft.py \
  --config configs/training_final.yaml
```

---

## 9. Evaluate 0.5B
```bash
python src/07_eval.py \
  --config configs/eval_final.yaml
```

---

## 10. Train 7B QLoRA (Optional)
```bash
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

python src/06_train_sft.py \
  --config configs/training_7b_qlora_bn.yaml
```

---

## 11. Evaluate 7B
```bash
python src/07_eval.py \
  --config configs/eval_7b_qlora_bn.yaml
```

---

## 12. Check Outputs
```bash
ls -R $OUTPUTS_DIR
ls -R $REPORTS_DIR
```

---

## 13. Check Metrics
```bash
cat $REPORTS_DIR/eval_metrics_final.json
cat $REPORTS_DIR/eval_metrics_7b_qlora_bn.json
```

---

## 14. GPU Memory Fix (if needed)
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 15. Always Start Session With
```bash
cd /content/drive/MyDrive/Multilingual_DailyDialog/Multilingual_DailyDialog
git pull
```

---

# Final Flow Summary
1. Build SFT  
2. Train 0.5B  
3. Evaluate 0.5B  
4. Train 7B  
5. Evaluate 7B  
6. Compare results  

