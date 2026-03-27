# Multilingual DailyDialog

Academic submission version of a minimal, reproducible pipeline for multilingual next-utterance generation.

## Final pipeline (four stages)

The supported workflow is:

1. `src/03_translate.py` — translate
2. `src/05_build_sft.py` — build SFT JSONL
3. `src/06_train_sft.py` — train (LoRA or QLoRA)
4. `src/07_eval.py` — evaluate

### Bengali-only path (canonical)

Use the API translation config and the same SFT output directory end-to-end:

```text
translate  →  build_sft  →  train (demo or final)  →  eval (demo or final)
configs/translation_1000_api_bn.yaml
  → data/sft/multilingual_1000/*.jsonl
  → configs/training_demo.yaml or configs/training_final.yaml
  → configs/eval_demo.yaml or configs/eval_final.yaml
```

Optional larger-model path (separate configs; does not replace the 0.5B baseline):

```text
configs/training_7b_qlora_bn.yaml  →  configs/eval_7b_qlora_bn.yaml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`bitsandbytes` is required only for 4-bit / 8-bit (QLoRA) training or 4-bit eval. `bert-score` is optional; evaluation uses it only when `evaluation.compute_bertscore: true` and the package imports successfully.

## Required data prerequisite

Step 1 (translation) expects preprocessed English DailyDialog JSONL files under:

- `data/processed_1000/train_en.jsonl`
- `data/processed_1000/validation_en.jsonl`
- `data/processed_1000/test_en.jsonl`

If these files are missing, prepare them before running the pipeline.

## Current validated setup

- **Base model (baseline):** `Qwen/Qwen2.5-0.5B-Instruct` as set in `configs/training_demo.yaml` and `configs/training_final.yaml`.
- **SFT data:** `data/sft/multilingual_1000/{train,validation,test}.jsonl` produced by `src/05_build_sft.py` with `configs/translation_1000_api_bn.yaml` (`sft_dir: sft/multilingual_1000`).
- **Training:** standard LoRA on full-precision (or bf16) weights; `model.load_in_4bit` and `model.load_in_8bit` are `false` in the 0.5B configs.
- **Evaluation:** decoding defaults are deterministic (`do_sample: false`, `temperature` / `top_p` at 1.0, `num_beams: 1`); see `evaluation.generation` in the eval YAML files.

## Optional larger-model setup (7B QLoRA)

- **Config:** `configs/training_7b_qlora_bn.yaml` uses `Qwen/Qwen2.5-7B-Instruct` with **4-bit loading** (`model.load_in_4bit: true`), **gradient checkpointing**, and a conservative batch size for T4-class GPUs.
- **Eval:** `configs/eval_7b_qlora_bn.yaml` loads the same base model in 4-bit and applies the adapter under `outputs/model_7b_qlora_bn/lora_adapter/`.
- **7B on Colab T4:** quantized training is **expected** for this path; the 0.5B baseline does not require it.

This repository does **not** claim numerical results for 7B until you run those configs locally.

## Configs

| Purpose | File |
|--------|------|
| Translation + SFT builder settings | `configs/translation_1000_api_bn.yaml` |
| Training (smoke) | `configs/training_demo.yaml` |
| Training (heavier) | `configs/training_final.yaml` |
| Training (7B QLoRA, optional) | `configs/training_7b_qlora_bn.yaml` |
| Eval (demo / final / 7B) | `configs/eval_demo.yaml`, `configs/eval_final.yaml`, `configs/eval_7b_qlora_bn.yaml` |

SFT prompts and builder options (`sft.*`, including `sft.prompt`) are read from the translation config. Training and evaluation share the same default system prompt text via `src/utils/prompting.py`.

## Run (canonical commands)

```bash
# 1) Translate
TARGET_LANGS=bn TRANSLATION_BACKEND=api python src/03_translate.py --config configs/translation_1000_api_bn.yaml

# 2) Build SFT
TARGET_LANGS=bn python src/05_build_sft.py --config configs/translation_1000_api_bn.yaml

# 3a) Train demo (0.5B)
python src/06_train_sft.py --config configs/training_demo.yaml

# 3b) Train final (0.5B)
python src/06_train_sft.py --config configs/training_final.yaml

# 3c) Optional 7B QLoRA
python src/06_train_sft.py --config configs/training_7b_qlora_bn.yaml

# 4) Evaluate
python src/07_eval.py --config configs/eval_demo.yaml
python src/07_eval.py --config configs/eval_final.yaml
python src/07_eval.py --config configs/eval_7b_qlora_bn.yaml
```

`BASE_MODEL` can still override the config when set in the environment; configs are preferred for reproducibility.

## Make targets

```bash
make translate
make build-sft
make train-demo
make train-final
make eval-demo
make eval-final
make train-7b
make eval-7b
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
- SFT build summary: `data/sft/multilingual_1000/build_sft_summary.json`
- LoRA adapter: `outputs/model_demo/lora_adapter/`, `outputs/model_final/lora_adapter/`, or `outputs/model_7b_qlora_bn/lora_adapter/`
- Training metadata: `outputs/<run>/train_run_metadata.json`
- Eval report, metrics, and JSONL generations: `reports/`

## Notes

- This cleaned submission keeps one canonical script per stage and only active configs under `configs/`.
- For reproducibility, keep config files and commands aligned with the tables above.
- Prompts are **not** duplicated in `05_build_sft.py` — they live in `src/utils/prompting.py` and are reapplied at eval time from `src/07_eval.py` so decoding uses the same template as training.
