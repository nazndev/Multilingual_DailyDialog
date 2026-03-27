# Multilingual DailyDialog

Academic submission version of a minimal, reproducible pipeline for multilingual next-utterance generation.

## Final Pipeline

The supported workflow is:

1. `src/03_translate.py`
2. `src/05_build_sft.py`
3. `src/06_train_sft.py`
4. `src/07_eval.py`

This repository keeps one canonical script per stage and only active configs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Required Data Prerequisite

Step 1 (translation) expects preprocessed English DailyDialog JSONL files under:

- `data/processed_1000/train_en.jsonl`
- `data/processed_1000/validation_en.jsonl`
- `data/processed_1000/test_en.jsonl`

If these files are missing, prepare them before running the final 4-step pipeline.

## Configs Kept

- Translation: `configs/translation_1000_api_bn.yaml`
- Training (demo): `configs/training_demo.yaml`
- Training (final): `configs/training_final.yaml`
- Eval: `configs/eval_demo.yaml` and `configs/eval_final.yaml`

## Run (Canonical Commands)

```bash
# 1) Translate
TARGET_LANGS=bn TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation_1000_api_bn.yaml

# 2) Build SFT
TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml

# 3a) Train demo
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/06_train_sft.py --config configs/training_demo.yaml

# 3b) Train final
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/06_train_sft.py --config configs/training_final.yaml

# 4) Evaluate
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/07_eval.py --config configs/eval_demo.yaml

# 4b) Evaluate final
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python src/07_eval.py --config configs/eval_final.yaml
```

## Make Targets

```bash
make translate
make build-sft
make train-demo
make train-final
make eval-demo
make eval-final
make pipeline-demo
make pipeline-final
```

## Outputs

Default output roots (overridable with env variables):

- `DATA_DIR` (default `./data`)
- `OUTPUTS_DIR` (default `./outputs`)
- `REPORTS_DIR` (default `./reports`)
- `CACHE_DIR` (default `./cache`)

Typical artifacts:

- Translated data: `data/translated_api_1000_bn/*.jsonl`
- SFT data: `data/sft/multilingual_1000/*.jsonl`
- LoRA adapter: `outputs/model_demo/lora_adapter/` or `outputs/model_final/lora_adapter/`
- Eval report and metrics: `reports/`

## Notes

- This cleaned submission removes Colab-only documentation and obsolete experiment paths.
- For reproducibility, keep config files and commands exactly as shown above.
