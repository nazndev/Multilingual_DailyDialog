# Run on Google Colab (with persistence)

Use Google Drive so **data**, **trained adapter**, and **reports** survive after the runtime disconnects.

**Checklist:** 
(1) Translate dialogue EN→Bangla, keep act/emotion ✓ 
(2) Qwen 7B generates ground truth for translation ✓ 
(3) Fine-tune Qwen 0.5B ✓ 
(4) Zero-shot vs fine-tune ✓ 
  — Train 800, test 200, zero-shot vs fine-tune report on 1000 samples ✓ (section 5).

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

# Colab already includes a CUDA-enabled PyTorch build.
# Installing `torch` via pip can silently replace it with a CPU-only wheel.
!grep -vE '^\s*torch\b' requirements.txt > /tmp/requirements_no_torch.txt
!pip install -q -r /tmp/requirements_no_torch.txt
!pip install -q sentencepiece protobuf
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

# Colab already includes a CUDA-enabled PyTorch build.
# Installing `torch` via pip can silently replace it with a CPU-only wheel.
!grep -vE '^\s*torch\b' requirements.txt > /tmp/requirements_no_torch.txt
!pip install -q -r /tmp/requirements_no_torch.txt
!pip install -q sentencepiece protobuf
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

## 5. Pipeline (100% — Colab-ready)

**Requirements:** Translate EN→Bangla (keep act/emotion) → Qwen 7B ground truth → Fine-tune Qwen 0.5B → Zero-shot vs fine-tune. **Train: 800, Test: 200, Zero-shot vs fine-tune report: 1000 samples.**

Ensure **Runtime → Change runtime type → Hardware accelerator: T4 GPU** (or A100/L4). Set `DATA_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR`, `CACHE_DIR` in section 2, then `%cd` into the repo (section 3). Run each block in order.

### 5.1 Download + preprocess (800 train, 200 test)

```bash
!python src/01_download.py
!python src/02_preprocess.py --config configs/preprocess_1000.yaml
```

### 5.2 Translate to Bangla (keep dialog_acts + emotions)

Set `OPENAI_API_KEY` and `TRANSLATION_BACKEND=api` in section 4 if using API.

```bash
!TARGET_LANGS=bn TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation_1000_api_bn.yaml
```

### 5.3 Build turn-level SFT, then dialogue-level (800 + 200)

```bash
!TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml
!python scripts/build_dialogue_sft.py --input "$DATA_DIR/sft/multilingual_1000/train.jsonl" --output "$DATA_DIR/sft/dialogue_1000_bn/train.jsonl"
!python scripts/build_dialogue_sft.py --input "$DATA_DIR/sft/multilingual_1000/test.jsonl" --output "$DATA_DIR/sft/dialogue_1000_bn/test.jsonl"
```

### 5.3b (optional) Sanity-check subset alignment (800/200/1000)

```python
import os, json
from pathlib import Path

DATA_DIR = Path(os.environ["DATA_DIR"])

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

en = load_jsonl(DATA_DIR / "processed_1000" / "train_en.jsonl")
bn = load_jsonl(DATA_DIR / "translated_api_1000_bn" / "train.jsonl")
sft_train = load_jsonl(DATA_DIR / "sft" / "dialogue_1000_bn" / "train.jsonl")
sft_test = load_jsonl(DATA_DIR / "sft" / "dialogue_1000_bn" / "test.jsonl")

print("lens (en, bn, sft_train, sft_test):", len(en), len(bn), len(sft_train), len(sft_test))

ids_en = {r["dialogue_id"] for r in en}
ids_bn = {r["dialogue_id"] for r in bn}
ids_sft_train = {r["dialogue_id"] for r in sft_train}
ids_sft_test = {r["dialogue_id"] for r in sft_test}

print("unique ids (en, bn, sft_train, sft_test):", len(ids_en), len(ids_bn), len(ids_sft_train), len(ids_sft_test))
print("sft_train subset of en?", ids_sft_train <= ids_en)
print("sft_train subset of bn?", ids_sft_train <= ids_bn)

for r in sft_train[:3]:
    did = r.get("dialogue_id", "")[:8]
    msgs = r.get("messages") or []
    lang = r.get("lang")
    print(f"  did={did} lang={lang} msg_len={len(msgs)}")
    if msgs:
        print("    system:", msgs[0])
        print("    first user/assistant:", msgs[1:3])
```

### 5.4 Qwen 7B teacher ground truth (GPU)

```python
import os

OUT_DIR = os.path.join(os.environ["DATA_DIR"], "sft", "teacher_dialogue_1000_bn")
os.makedirs(OUT_DIR, exist_ok=True)

!python3 scripts/generate_teacher_sft.py --input "$DATA_DIR/sft/dialogue_1000_bn/train.jsonl" --output "{OUT_DIR}/train.jsonl" --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 96 --temperature 0.0
!python3 scripts/generate_teacher_sft.py --input "$DATA_DIR/sft/dialogue_1000_bn/test.jsonl" --output "{OUT_DIR}/test.jsonl" --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 96 --temperature 0.0

!wc -l "{OUT_DIR}/train.jsonl" "{OUT_DIR}/test.jsonl"
```

### 5.5 Fine-tune Qwen 0.5B on 800 teacher-labeled examples

```bash
!python src/06_train_sft.py --config configs/training_teacher_dialogue_1000.yaml
```

### 5.6 Zero-shot vs fine-tuned (200 test)

```bash
!python src/07_eval.py --config configs/eval_teacher_dialogue_1000.yaml
```

### 5.7 Zero-shot vs fine-tuned on 1000 samples (professor report)

```bash
!cat "$DATA_DIR/sft/teacher_dialogue_1000_bn/train.jsonl" "$DATA_DIR/sft/teacher_dialogue_1000_bn/test.jsonl" > "$DATA_DIR/sft/teacher_dialogue_1000_bn/all_1000.jsonl"
!python src/07_eval.py --config configs/eval_teacher_dialogue_1000_all.yaml
```

Reports: `$REPORTS_DIR/eval_report_teacher_dialogue_1000.md` (200 test), `$REPORTS_DIR/eval_report_teacher_dialogue_1000_all.md` (1000 samples).

---

### A) Quick data-only (no teacher/train): Bangla SFT from translation

```bash
!python src/01_download.py
!python src/02_preprocess.py --config configs/preprocess_1000.yaml
!TARGET_LANGS=bn TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation_1000_api_bn.yaml
!TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml
```

### B) If you already have `dialogue_1000_bn` (e.g. from another run)

Skip 5.1–5.3 and run 5.4–5.7 only. Inputs: `$DATA_DIR/sft/dialogue_1000_bn/train.jsonl` (800), `$DATA_DIR/sft/dialogue_1000_bn/test.jsonl` (200).

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
