# Run on Google Colab (with persistence)

Use Google Drive so **data**, **trained adapter**, and **reports** survive after the runtime disconnects.

---

## 1. New notebook, enable GPU

- [colab.new](https://colab.new) → **Runtime** → **Change runtime type** → **T4 GPU** (or A100 if available) → Save.

---

## 2. Mount Drive and set persistent paths

```python
from google.colab import drive
drive.mount('/content/drive')

# Persistent folder for this project (create once)
import os
COLAB_ROOT = "/content/drive/MyDrive/Multilingual_DailyDialog"
os.makedirs(COLAB_ROOT, exist_ok=True)

# Point pipeline to Drive so everything is saved there
os.environ["DATA_DIR"]   = f"{COLAB_ROOT}/data"
os.environ["CACHE_DIR"]  = f"{COLAB_ROOT}/cache"
os.environ["OUTPUTS_DIR"] = f"{COLAB_ROOT}/outputs"
os.environ["REPORTS_DIR"] = f"{COLAB_ROOT}/reports"
```

---

## 3. Clone repo and install

```bash
%cd /content
!git clone https://github.com/nazndev/Multilingual_DailyDialog.git
%cd Multilingual_DailyDialog

!pip install -r requirements.txt
```

---

## 4. Optional: GPT translation and Hugging Face

If you use **GPT** for translation or need **HF token** for dataset download, set secrets in the notebook (don’t commit them):

```python
import os
# For GPT translation (step 3)
os.environ["OPENAI_API_KEY"] = "sk-..."      # your key
os.environ["TRANSLATION_BACKEND"] = "api"

# For dataset download (step 1) if required
# os.environ["HF_TOKEN"] = "hf_..."
```

Or use **Colab Secrets**: left sidebar → 🔑 **Secrets** → add `OPENAI_API_KEY` → in code: `from google.colab import userdata; os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")`.

---

## 5. Run pipeline (persistent because paths are on Drive)

**Quick run (0.5B model, good for free Colab):**

```bash
!python src/01_download.py
!python src/02_preprocess.py
!TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation.yaml
!python src/05_build_sft.py --config configs/translation.yaml
!BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/06_train_sft.py --config configs/training_quick.yaml
!BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/07_eval.py --config configs/eval_quick.yaml
```

**Or with Make (one shell cell; env vars apply to make):**

```bash
export DATA_DIR="/content/drive/MyDrive/Multilingual_DailyDialog/data"
export OUTPUTS_DIR="/content/drive/MyDrive/Multilingual_DailyDialog/outputs"
export REPORTS_DIR="/content/drive/MyDrive/Multilingual_DailyDialog/reports"
export CACHE_DIR="/content/drive/MyDrive/Multilingual_DailyDialog/cache"
make quick-small
```
(In Colab, run as `!bash -c 'export DATA_DIR=...; ...; make quick-small'` or run each `export` and `make` in one cell.)

---

## 6. Where everything is saved (on Drive)

After the run, in **Drive** → `My Drive/Multilingual_DailyDialog/`:

- `data/` — raw, processed, translated, SFT data  
- `outputs/model/lora_adapter/` — trained LoRA weights  
- `reports/` — e.g. `eval_report.md`  
- `cache/` — translation cache  

Re-run the notebook later: mount Drive, set the same `DATA_DIR`/`OUTPUTS_DIR`/`REPORTS_DIR`/`CACHE_DIR`, clone (or pull) the repo, and run again. No need to re-download or re-translate if you keep `data/` and `cache/` on Drive.

---

## 7. If you already ran and didn’t use Drive

Copy from Colab disk to Drive before runtime dies:

```python
import os
root = "/content/drive/MyDrive/Multilingual_DailyDialog"
os.makedirs(root, exist_ok=True)
for folder in ["data", "outputs", "reports", "cache"]:
    os.system(f"cp -r /content/Multilingual_DailyDialog/{folder} {root}/")
```

Then next time set env to `COLAB_ROOT/...` and run from step 5 (or skip 1–3 if data already there).
