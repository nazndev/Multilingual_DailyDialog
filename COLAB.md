# Run on Google Colab (with persistence)

Use Google Drive so **data**, **trained adapter**, and **reports** survive after the runtime disconnects.

---

## 1. New notebook, set runtime type

- [colab.new](https://colab.new) (or open your `Multilingual_DailyDialog.ipynb`).
- **Runtime** → **Change runtime type**.
- In the dialog:
  - **Runtime type:** `Python 3`
  - **Hardware accelerator:** `T4 GPU` (free; or `A100 GPU` / `L4 GPU` if you have compute units)
  - **Runtime version:** `Latest (recommended)`
- Click **Save**.

---

## 2. Mount Drive and set persistent paths

```python
from google.colab import drive
drive.mount('/content/drive')

# Persistent folder for this project (create once)
import os
COLAB_ROOT = "/content/drive/MyDrive/Multilingual_DailyDialog"
os.makedirs(COLAB_ROOT, exist_ok=True)

# Optional but recommended: persist pip's download cache on Drive to speed up re-installs
os.environ["PIP_CACHE_DIR"] = f"{COLAB_ROOT}/pip-cache"

# Point pipeline to Drive so everything is saved there
os.environ["DATA_DIR"]   = f"{COLAB_ROOT}/data"
os.environ["CACHE_DIR"]  = f"{COLAB_ROOT}/cache"
os.environ["OUTPUTS_DIR"] = f"{COLAB_ROOT}/outputs"
os.environ["REPORTS_DIR"] = f"{COLAB_ROOT}/reports"
```

---

## 3. Get the code + install deps

You have two good options:

### Option A (fast filesystem): clone into `/content` each session

This is usually the fastest to *run* (Colab local disk), but you’ll re-clone every runtime.

```bash
%cd /content
!rm -rf Multilingual_DailyDialog
!git clone https://github.com/nazndev/Multilingual_DailyDialog.git
%cd Multilingual_DailyDialog
!pip install -q -r requirements.txt
```

### Option B (persistent): keep the git clone on Drive and `git pull`

This avoids re-cloning when you come back later. It can be slightly slower at runtime because Drive I/O is slower than `/content`, but for this repo it’s usually fine.

```python
import os
REPO_DIR = f"{COLAB_ROOT}/repo"

if not os.path.exists(REPO_DIR):
  !git clone https://github.com/nazndev/Multilingual_DailyDialog.git "$REPO_DIR"
else:
  !git -C "$REPO_DIR" pull

%cd "$REPO_DIR"
!pip install -q -r requirements.txt
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

### A) End-to-end (1000-sample, Bangla-only)

This matches the **800/200/200 dialogue** subset and produces Bangla-only SFT.

```bash
!python src/01_download.py
!python src/02_preprocess.py --config configs/preprocess_1000.yaml
!TARGET_LANGS=bn TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation_1000_api_bn.yaml
!TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml
```

### B) Teacher generation (Qwen 7B, GPU)

For **dialogue-based** distillation (recommended): use the existing split that already corresponds to
**800 train dialogues** and **200 test dialogues**, while keeping **all assistant turns** from each dialogue.

Upload these two files to Drive (or generate them via Step A above):

- `${DATA_DIR}/sft/multilingual_1000/train.jsonl`  (all SFT rows from 800 dialogues)
- `${DATA_DIR}/sft/multilingual_1000/test.jsonl`   (all SFT rows from 200 dialogues)

Then run:

```bash
!python scripts/generate_teacher_sft.py \
  --input "$DATA_DIR/sft/multilingual_1000/train.jsonl" \
  --output "$DATA_DIR/sft/teacher_dialogue_1000_bn/train.jsonl" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 96 --temperature 0.0

!python scripts/generate_teacher_sft.py \
  --input "$DATA_DIR/sft/multilingual_1000/test.jsonl" \
  --output "$DATA_DIR/sft/teacher_dialogue_1000_bn/test.jsonl" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 96 --temperature 0.0
```

Outputs:

- `${DATA_DIR}/sft/teacher_dialogue_1000_bn/train.jsonl` (rows correspond to all turns from 800 dialogues)
- `${DATA_DIR}/sft/teacher_dialogue_1000_bn/test.jsonl` (rows correspond to all turns from 200 dialogues)

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
