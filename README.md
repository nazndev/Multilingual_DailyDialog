# Multilingual DailyDialog Translation + SFT

Dataset: https://huggingface.co/datasets/roskoN/dailydialog

Targets: Bangla (bn), Arabic (ar), Spanish (es). Pipeline preserves per-turn **dialog acts** and **emotion** labels; evaluation includes **BLEU** and optional **baseline** (base vs LoRA).

**Models:** Translation = NLLB (default) or **LLM API (e.g. GPT)** for higher accuracy; Chat = Qwen2.5-7B-Instruct + LoRA. Evaluation compares **Zero-shot** (base) vs **Fine-tuned** (LoRA).

**Run on Google Colab (persistent):** [COLAB.md](COLAB.md).

---

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Or: `make venv` then `make install`.

---

## BN-only mode via .env
For a **quick seminar demo** (Bangla only), copy `.env.example` to `.env` and set:
```bash
TARGET_LANGS=bn
```
Leave `DATA_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR` at defaults (`./data`, `./outputs`, `./reports`) or override them. **Never commit `.env`**—it may contain tokens; only `.env.example` is safe to commit.

---

## Switch to full (bn/ar/es)
In `.env` set:
```bash
TARGET_LANGS=bn,ar,es
```
Then run the full pipeline (`make full` or `make demo`). Quick config (`make quick`) still uses a small sample; full configs translate all dialogues and train longer.

---

## LLM-based translation (GPT for higher accuracy)
For more accurate translation, use an **LLM (e.g. OpenAI GPT)** instead of local NLLB. In `.env` set `OPENAI_API_KEY` and `TRANSLATION_BACKEND=api`, then run step 3 with the same configs (quick or full):

```bash
make translate-llm
# or: TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation.yaml
```

Then run steps 4–6 (Build SFT → Train → Eval) as usual. Translations are cached; same cache key for API and local so re-runs are cheap.

---

## Zero-shot vs Fine-tuned evaluation
Step 6 (Eval) compares **Zero-shot** (base Qwen, no LoRA) and **Fine-tuned** (base + LoRA) on the test set. In `configs/eval.yaml`, `run_baseline: true` runs the base model first and labels the report as **Zero-shot (base model)** and **Fine-tuned (LoRA)**. The eval report is written to `REPORTS_DIR/eval_report.md`.

**Build SFT is required before both.** Train uses SFT train/validation data; Eval uses SFT test data (each example has `messages`: system + user/assistant utterances). Build SFT converts translated **utterances** (per-turn dialogue text in bn/ar/es) into that chat format, so the pipeline order is: **Translate → Build SFT → Train → Eval (zero-shot + fine-tuned)**.

---

## Base directory overrides
In `.env` you can set:
- `DATA_DIR` — raw, processed, translated, and SFT data (default `./data`)
- `CACHE_DIR` — translation cache (default `./cache`)
- `OUTPUTS_DIR` — LoRA adapters and training outputs (default `./outputs`)
- `REPORTS_DIR` — evaluation reports and logs (default `./reports`)

All paths in configs are **relative** to these bases. No hardcoded `data/` in code.

---

## Fresh run (remove data, re-download, then run pipeline)
To wipe existing pipeline data and start from a clean Hugging Face download, then run the pipeline step by step:
```bash
make fresh          # clean + download + quick pipeline (bn-only by default)
make fresh-full     # clean + download + full pipeline (bn/ar/es)
```
Clean only (no download or run):
```bash
make clean          # empties DATA_DIR, OUTPUTS_DIR, REPORTS_DIR, CACHE_DIR
```
Or run the script directly: `./scripts/fresh_run.sh clean quick` or `./scripts/fresh_run.sh clean full`.

---

## Quick Demo
```bash
make quick
```
Uses `configs/translation.yaml` (all dialogues, bn/ar/es). Targets can be overridden via `TARGET_LANGS` in `.env`. Outputs go under `DATA_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR`.

---

## Full Demo
```bash
make full
```
or
```bash
make demo
```
Runs `scripts/demo.sh` with `configs/translation.yaml`, `training_full.yaml`, `eval_full.yaml` (bn/ar/es, all dialogues, baseline vs LoRA).

---

## Expected output paths
All relative to `DATA_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR` (defaults: `./data`, `./outputs`, `./reports`).

- **Translated:** `DATA_DIR/translated_dailydialog_en_bn_ar_es/{train,validation,test}.jsonl`
- **SFT:** `DATA_DIR/sft/multilingual/{train,validation,test}.jsonl`
- **Quick run adapter:** `OUTPUTS_DIR/model/lora_adapter/`
- **Full run adapter:** `OUTPUTS_DIR/model_full/lora_adapter/`
- **Reports:** `REPORTS_DIR/eval_report.md` (quick) or `REPORTS_DIR/eval_report_full.md` (full)

---

## Logging (show logs to your professor)
Every pipeline step logs to **console** and to a **per-run file** under `REPORTS_DIR/logs/` (default `reports/logs/`). A full run via `make full` or `scripts/demo.sh` also writes a **master pipeline log** so one file contains all steps.

- **Log file names:** `YYYYMMDD_HHMMSS_01_download.log`, … `07_eval.log`, and `YYYYMMDD_HHMMSS_pipeline.log` for the master log.
- **List latest logs:**  
  `make logs`  
  (or `ls -lt reports/logs | head -20` if you use the default `REPORTS_DIR`.)
- **Watch the current run live:**  
  `tail -f reports/logs/20250217_143022_pipeline.log`  
  (replace with the latest `*_pipeline.log` from `make logs`.)
- **Show a summary of dataset counts:**  
  - Per-step logs already print record counts, turns, and paths.  
  - For a quick count of translated dialogues per split:  
    `wc -l data/translated_dailydialog_en_bn_ar_es/*.jsonl`  
  (use `$DATA_DIR` if you override it in `.env`.)

Logs are sanitized: **tokens, keys, and secrets are never printed** (env and config values with names like `token`, `key`, `secret`, `password` are masked).

---

## 3-minute live demo checklist
1. **Show .env (no secrets):** `cat .env.example` — explain `TARGET_LANGS=bn` for BN-only.
2. **Run quick:** `make quick` (or run up to Build SFT and show SFT sample).
3. **Show dataset sample:**  
   `python -c "from src.utils.env import get_dirs; d=get_dirs()['data']; import json; print(list(json.loads((d/'translated_dailydialog_en_bn_ar_es/test.jsonl').read_text().splitlines()[0]).keys()))"`
4. **Show SFT sample:**  
   `python -c "from src.utils.env import get_dirs; d=get_dirs()['data']; import json; r=json.loads((d/'sft/multilingual/test.jsonl').read_text().splitlines()[0]); print(r.get('lang'), r['messages'][0]['content'][:80])"`
5. **Show eval report:**  
   `cat reports/eval_report.md` (or `REPORTS_DIR/eval_report.md`).

---

## Smoke test
Runs the full pipeline (download → preprocess → translate → build SFT → train → eval) then checks: required files exist, JSONL non-empty, sample record has expected keys. Exits non-zero on failure.

**Why it can take a while:** The run includes LoRA training (30 steps on the base model). On GPU this may take ~10–30 minutes; on CPU much longer.
```bash
bash scripts/smoke_test.sh
```

---

## Config and options
- **Translation (step 3):** Single config **`configs/translation.yaml`** — all dialogues, bn/ar/es. Backend: `backend: local` = NLLB; for GPT set `TRANSLATION_BACKEND=api` and `OPENAI_API_KEY` in `.env`.
- **Languages:** `TARGET_LANGS` in `.env` overrides config `targets` (03) and eval `langs` (07). Use `bn` for BN-only, `bn,ar,es` for all three.
- **Evaluation:** In `configs/eval.yaml`: `compute_bleu: true`, `run_baseline: true` for Zero-shot vs Fine-tuned.

## UI Demo (Streamlit)
A seminar-friendly dashboard to browse aligned dialogues, show base vs LoRA comparisons from saved generations, and view evaluation reports.

```bash
make ui
```

- UI reads from `DATA_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR` (set in `.env`).
- For reliability, use **saved generations** in `reports/generations.jsonl` or `reports/generations_full.jsonl` (keys e.g. `lang`, `baseline`, `lora`) instead of running heavy inference live.

Check that UI prerequisites are met:
```bash
make ui-check
```

## Notes
- Translation uses NLLB locally by default.
- Do not publish translated dataset unless derivative redistribution is allowed by the dataset license.

---

## CHANGELOG

**Repo fixes (cursor_repo_master.yml)**  
- `configs/translation.yaml`: targets `[bn, ar, es]`, no dialogue limit; relative `processed_dir`, `out_dir`, `cache.dir`.  
- `configs/eval.yaml`: `langs: [bn, ar, es]`; relative paths.  
- `src/03_translate.py`: `out_dir` from config; no hardcoded data paths.  
- `src/06_train_sft.py`: bf16/fp16 `auto` only when CUDA available.

**Env and paths (cursor_env_and_paths.yml)**  
- `.env.example`: `TARGET_LANGS`, `DATA_DIR`, `CACHE_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR`, `BASE_MODEL`, `TRANSLATION_MODEL`; no tokens.  
- `src/utils/env.py`: `get_dirs()`, `get_langs()`, `resolve_path()`, `get_env()`.  
- `.gitignore`: added `cache/`, `reports/`.  
- Configs: all data/cache/output/report paths relative; resolved in code via env bases.  
- `src/01_download.py`–`07_eval.py`: use env and config; no hardcoded `data/` paths.  
- `Makefile`: include and export `.env` when present.

**Full configs and demo (cursor_full_run_and_demo.yml)**  
- `configs/translation.yaml`, `training_full.yaml`, `eval_full.yaml`: full run, relative paths; base model `Qwen/Qwen2.5-7B-Instruct`.  
- `scripts/demo.sh`: executable; sources `.env`; runs full pipeline with _full configs.  
- `Makefile`: `quick`, `full`, `demo`, `venv`, `install`.

**README and smoke test**  
- README: BN-only via `TARGET_LANGS=bn`; full via `TARGET_LANGS=bn,ar,es`; base dir overrides; quick/full demo; expected paths; 3-minute demo checklist; never commit `.env`; note on smoke test duration (training step).  
- `scripts/smoke_test.sh`: quick run; verifies output files, JSONL non-empty, sample keys; exits non-zero on failure.

**UI (cursor_streamlit_ui.yml)**  
- `ui/app.py`: Streamlit dashboard (Dataset Viewer, Chat Compare from saved generations, Evaluation); uses `DATA_DIR`/`OUTPUTS_DIR`/`REPORTS_DIR` from env.  
- `ui/utils_io.py`: path helpers, `read_jsonl`/`read_text`, `guess_translated_dir`/`guess_sft_dir`, `extract_turns` for our schema.  
- `ui/README_UI.md`: run instructions and env.  
- `scripts/ui_smoke_check.py`: checks translated dir, eval report, ui/app.py.  
- `requirements.txt`: added `streamlit>=1.31.0`.  
- Makefile: `ui`, `ui-check`.  
- README: UI Demo section.
