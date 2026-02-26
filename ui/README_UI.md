# UI Demo (Streamlit)

This UI is a dashboard for:
- Viewing aligned multilingual DailyDialog turns with emotion/dialog-act labels (readable act/emotion names)
- Comparing Base vs LoRA generations (from saved reports/generations*.jsonl)
- Viewing evaluation reports
- **Try it**: type a message and get a response in Bangla, Arabic, or Spanish (Base or LoRA model; first run loads the model)

## Run
```bash
# single script (from anywhere): activates venv, loads .env, installs deps, runs UI
./scripts/run_ui.sh
```
Or from repo root:
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```
Or: `make ui`

## Environment variables
The UI uses the same base dirs as the pipeline:
- DATA_DIR (default: ./data)
- OUTPUTS_DIR (default: ./outputs)
- REPORTS_DIR (default: ./reports)

Set them via .env (e.g. copy .env.example to .env). Never commit .env.

## Seminar best practice
Precompute generations and show them from:
- reports/generations.jsonl (quick) or
- reports/generations_full.jsonl (full)

Each line can be JSON with keys: lang, baseline (or base/base_response), lora (or lora_response/adapted).
