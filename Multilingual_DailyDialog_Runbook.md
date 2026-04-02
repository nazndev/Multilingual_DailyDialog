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

## 11. Train 3B QLoRA (Optional fair smaller-model comparison)
```bash
python src/06_train_sft.py \
  --config configs/training_3b_qlora_bn.yaml
```

Expected training artifacts:
```bash
ls -R $OUTPUTS_DIR/model_3b_qlora_bn
```

Exact expected paths:
- `$OUTPUTS_DIR/model_3b_qlora_bn`
- `$OUTPUTS_DIR/model_3b_qlora_bn/lora_adapter`

Optional pre-flight checks before training:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
print(bool(getattr(tok, "chat_template", None)))
print((tok.chat_template or "")[:400])
```

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
found = {name.split(".")[-1] for name, _ in model.named_modules()}
print("missing:", sorted(wanted - found))
```

---

## 12. Evaluate 7B
```bash
python src/07_eval.py \
  --config configs/eval_7b_qlora_bn.yaml
```

---

## 13. Evaluate 3B
```bash
python src/07_eval.py \
  --config configs/eval_3b_qlora_bn.yaml
```

Expected evaluation artifacts:
```bash
ls $REPORTS_DIR/eval_report_3b_qlora_bn.md
ls $REPORTS_DIR/eval_metrics_3b_qlora_bn.json
ls $REPORTS_DIR/generations_3b_qlora_bn.jsonl
```

Exact expected report paths:
- `$REPORTS_DIR/eval_report_3b_qlora_bn.md`
- `$REPORTS_DIR/eval_metrics_3b_qlora_bn.json`
- `$REPORTS_DIR/generations_3b_qlora_bn.jsonl`

Metrics note: run locally to populate metrics for the new 3B path.

---

## 14. Check Outputs
```bash
ls -R $OUTPUTS_DIR
ls -R $REPORTS_DIR
```

---

## 15. Check Metrics
```bash
cat $REPORTS_DIR/eval_metrics_final.json
cat $REPORTS_DIR/eval_metrics_3b_qlora_bn.json
cat $REPORTS_DIR/eval_metrics_7b_qlora_bn.json
```

---

## 16. GPU Memory Fix (if needed)
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 17. Always Start Session With
```bash
cd /content/drive/MyDrive/Multilingual_DailyDialog/Multilingual_DailyDialog
git pull
```

---

# Final Flow Summary
1. Build SFT  
2. Train 0.5B  
3. Evaluate 0.5B  
4. Train 3B  
5. Evaluate 3B  
6. Train 7B  
7. Evaluate 7B  
8. Compare results locally after metrics are generated  

