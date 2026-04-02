# Multilingual DailyDialog — Colab Run Commands (Final)

## 1) Mount Drive & Setup Paths
from google.colab import drive
drive.mount("/content/drive")

import os

COLAB_ROOT = "/content/drive/MyDrive/Multilingual_DailyDialog"
os.makedirs(COLAB_ROOT, exist_ok=True)

os.environ["HF_HOME"] = f"{COLAB_ROOT}/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

os.environ["DATA_DIR"] = f"{COLAB_ROOT}/data"
os.environ["CACHE_DIR"] = f"{COLAB_ROOT}/cache"
os.environ["OUTPUTS_DIR"] = f"{COLAB_ROOT}/outputs"
os.environ["REPORTS_DIR"] = f"{COLAB_ROOT}/reports"

REPO_DIR = f"{COLAB_ROOT}/repo"

for p in [
    COLAB_ROOT,
    os.environ["HF_HOME"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_HUB_CACHE"],
    os.environ["DATA_DIR"],
    os.environ["CACHE_DIR"],
    os.environ["OUTPUTS_DIR"],
    os.environ["REPORTS_DIR"],
]:
    os.makedirs(p, exist_ok=True)

---

## 2) Sync Latest Repo

import os

if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    !git clone https://github.com/nazndev/Multilingual_DailyDialog.git "$REPO_DIR"

%cd $REPO_DIR
!git reset --hard
!git clean -fd
!git fetch origin
!git checkout main
!git reset --hard origin/main
!git status -sb
!git log --oneline -n 3


---

## 2.A) Pull 
%cd /content/drive/MyDrive/Multilingual_DailyDialog/repo
!git pull


## 3) Install Dependencies

%cd $REPO_DIR
!grep -vE '^\s*torch\b' requirements.txt > /tmp/requirements_no_torch.txt
!pip -q install -r /tmp/requirements_no_torch.txt
!pip -q install sentencepiece protobuf accelerate bitsandbytes


---

## 4) Check GPU
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))


---

## Hugging face Connet
from huggingface_hub import login
login()


## missing 
%cd $REPO_DIR
!pip install -q langdetect sacrebleu


## 5) Rebuild SFT (MANDATORY)

%cd $REPO_DIR
!TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml


Verify:
!find "$DATA_DIR/sft/multilingual_1000" -maxdepth 2 -type f | sort


---

## 6) Train (0.5B FINAL)
import torch
torch.cuda.empty_cache()

%cd $REPO_DIR
!BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/06_train_sft.py --config configs/training_final.yaml

Verify:
!find "$OUTPUTS_DIR/model_final" -maxdepth 4 -type f | sort


---

## 7) Evaluate (0.5B)

%cd $REPO_DIR
!BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/07_eval.py --config configs/eval_final.yaml


Verify:
```python
!find "$REPORTS_DIR" -maxdepth 4 -type f | sort
!cat "$REPORTS_DIR/eval_metrics_final.json"
```

---

## 8) Train (7B QLoRA — Optional)
```python
import torch, os
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

%cd $REPO_DIR
!BASE_MODEL=Qwen/Qwen2.5-7B-Instruct python src/06_train_sft.py --config configs/training_7b_qlora_bn.yaml
```

---

## 9) Train (3B QLoRA — Optional fair smaller-model comparison)
```python
import torch
torch.cuda.empty_cache()

%cd $REPO_DIR
!python src/06_train_sft.py --config configs/training_3b_qlora_bn.yaml
```

Optional pre-flight checks:
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

Expected artifacts after training:
```python
!find "$OUTPUTS_DIR/model_3b_qlora_bn" -maxdepth 4 -type f | sort
```

Exact expected training paths:
- `outputs/model_3b_qlora_bn`
- `outputs/model_3b_qlora_bn/lora_adapter`

---

## 10) Evaluate (7B QLoRA — Optional)
```python
%cd $REPO_DIR
!BASE_MODEL=Qwen/Qwen2.5-7B-Instruct python src/07_eval.py --config configs/eval_7b_qlora_bn.yaml
```

Verify:
```python
!cat "$REPORTS_DIR/eval_metrics_7b_qlora_bn.json"
```

---

## 11) Evaluate (3B QLoRA — Optional)
```python
%cd $REPO_DIR
!python src/07_eval.py --config configs/eval_3b_qlora_bn.yaml
```

Expected artifacts after evaluation:
```python
!find "$REPORTS_DIR" -maxdepth 4 -type f | sort
```

Metric placeholders:
```python
print("Run locally to populate eval_report_3b_qlora_bn.md, eval_metrics_3b_qlora_bn.json, and generations_3b_qlora_bn.jsonl")
```

Exact expected report paths:
- `reports/eval_report_3b_qlora_bn.md`
- `reports/eval_metrics_3b_qlora_bn.json`
- `reports/generations_3b_qlora_bn.jsonl`

---

## FINAL ORDER
1. Setup Drive
2. Sync Repo
3. Install Dependencies
4. Rebuild SFT
5. Train (0.5B)
6. Evaluate (0.5B)
7. Train (3B)
8. Evaluate (3B)
9. Train (7B)
10. Evaluate (7B)
