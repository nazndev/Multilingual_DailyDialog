# Multilingual DailyDialog

Academic submission version of a minimal, reproducible pipeline for multilingual next-utterance generation.

**Repository:** [github.com/nazndev/Multilingual_DailyDialog](https://github.com/nazndev/Multilingual_DailyDialog)

## Final pipeline (four stages)

The supported workflow is:

1. `src/03_translate.py` — translate
2. `src/05_build_sft.py` — build SFT JSONL
3. `src/06_train_sft.py` — train (LoRA or QLoRA)
4. `src/07_eval.py` — evaluate
5. `src/08_judge.py` — optional pairwise LLM judge (Qwen 7B)

### Bengali-only path (canonical)

```text
translate  →  build_sft  →  train (demo or final)  →  eval (demo or final)
configs/translation_1000_api_bn.yaml
  → data/sft/multilingual_1000/*.jsonl
  → configs/training_demo.yaml or configs/training_final.yaml
  → configs/eval_demo.yaml or configs/eval_final.yaml
```

Optional larger-model path (separate configs; does not replace the 0.5B baseline):

```text
configs/training_3b_qlora_bn.yaml  →  configs/eval_3b_qlora_bn.yaml
configs/training_7b_qlora_bn.yaml  →  configs/eval_7b_qlora_bn.yaml
configs/training_gemma2_2b_it_qlora_bn.yaml  →  configs/eval_gemma2_2b_it_qlora_bn.yaml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`bitsandbytes` is required only for 4-bit / 8-bit (QLoRA) training or 4-bit eval. **`bert-score`** is listed in `requirements.txt`; evaluation calls it only when `evaluation.compute_bertscore: true` and the import succeeds. For Bangla, BERTScore uses language code `bn` when supported, with a fallback to `en` if the requested language fails inside bert-score. **BLEU with `tokenize=flores200`** needs the sacrebleu SPM backend: install **`sentencepiece`** (`pip install sentencepiece`) or BLEU may be omitted (`None`) if sacrebleu cannot load that tokenizer.

**Metrics (why both BLEU and BERTScore):** BLEU stays for strict n-gram overlap and continuity with past runs; **chrF** remains useful for Bangla morphology; **BERTScore** (when enabled) complements them because BLEU is very strict on open-ended dialogue where paraphrases are valid.

## Required data prerequisite

Step 1 (translation) expects preprocessed English DailyDialog JSONL files under:

- `data/processed_1000/train_en.jsonl`
- `data/processed_1000/validation_en.jsonl`
- `data/processed_1000/test_en.jsonl`

If these files are missing, prepare them before running the pipeline.

## How dialogue rows become SFT examples

Step `src/05_build_sft.py` reads translated JSONL (e.g. `turns_bn`, `turns_en`, `dialogue_id`, optional `emotions` / `dialog_acts`) and writes one JSONL training row per **assistant** reply you want the model to predict.

- **One dialogue → many examples:** For each dialogue, every **odd** turn index (`1, 3, 5, …`) is treated as an assistant utterance. The model is trained to produce that utterance given prior context. Even indices are user turns (DailyDialog-style alternation).
- **Targets only:** Only those assistant turns become supervised targets; user turns appear in the context, not as the prediction target.
- **`context_window`:** If `sft.context_window` is positive (e.g. `4`), each example includes only the **last N utterances** before the target turn, in order. If `context_window` is `-1`, the full prefix of the dialogue up to the target is used.
- **Prompts:** The system string is built only through `src/utils/prompting.py` (`build_system_prompt`), using `sft.prompt` from `configs/translation_1000_api_bn.yaml` (style, short-reply hint, optional `system_template`). Evaluation rebuilds the same logic when decoding so training and eval stay aligned.
- **Labels:** Integer emotion and dialog-act IDs are mapped to stable lowercase names (DailyDialog conventions) for optional prompt tags and for `emotion_at_turn` / `act_at_turn` fields in each row when the value maps successfully.

See `data/sft/.../build_sft_summary.json` after a run for per-split counts and effective settings.

## Current validated baseline (0.5B Bengali)

This is the **lightweight, reproducible baseline** you should report when you need a small, fast, cheap run.

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (`configs/training_demo.yaml`, `configs/training_final.yaml`).
- **SFT data:** `data/sft/multilingual_1000/{train,validation,test}.jsonl` from `src/05_build_sft.py` with `configs/translation_1000_api_bn.yaml` (`sft_dir: sft/multilingual_1000`). With `sft.prompt.short_reply_hint: true`, the builder and eval share the same short-reply hint as in `src/utils/prompting.py`.
- **Training:** standard LoRA on full-precision (or bf16) weights; `model.load_in_4bit` and `model.load_in_8bit` are `false` in the 0.5B configs.
- **Evaluation:** decoding is deterministic (`evaluation.generation`: `do_sample: false`, `temperature` / `top_p` at 1.0, `num_beams: 1`, `max_new_tokens: 96`); see `configs/eval_final.yaml` and `configs/eval_demo.yaml`. **BLEU** uses [sacrebleu](https://github.com/mjpost/sacrebleu) with `tokenize=flores200` by default (`evaluation.bleu_tokenizer`). **7B/3B QLoRA eval YAMLs** set `compute_bertscore: true` so you get BERTScore F1 alongside BLEU/chrF when `bert-score` is installed.

**Rerun 7B / 3B evaluation only** (after training or adapter changes), from the repo root with `DATA_DIR`, `OUTPUTS_DIR`, and `REPORTS_DIR` set as in the runbook:

```bash
python src/07_eval.py --config configs/eval_7b_qlora_bn.yaml
python src/07_eval.py --config configs/eval_3b_qlora_bn.yaml
```

The **final** 0.5B training config (`configs/training_final.yaml`) uses a longer training budget than the demo config for a fairer baseline; it does **not** switch to 7B.

## Optional larger-model experiment (7B QLoRA Bengali)

This path is **optional** and **stronger but heavier** than the 0.5B baseline. It is **not** required for a valid submission.

- **Config:** `configs/training_7b_qlora_bn.yaml` — `Qwen/Qwen2.5-7B-Instruct` with **4-bit loading** (`model.load_in_4bit: true`), **gradient checkpointing**, and T4-friendly batch settings.
- **Eval:** `configs/eval_7b_qlora_bn.yaml` — same Bengali test path and the same deterministic decoding block as `eval_final.yaml` for direct comparison.
- **7B on Colab T4:** quantized training is **expected** for this path; the 0.5B baseline does not require it.

This repository does **not** claim numerical results for 7B until you run those configs locally.

## Optional Qwen2.5-3B-Instruct experiment

This path is the **fair smaller instruct-model comparison** against `Qwen/Qwen2.5-7B-Instruct` because it stays within the same Qwen2.5 Instruct family while using a smaller parameter count.

- **Train:** `configs/training_3b_qlora_bn.yaml`
- **Eval:** `configs/eval_3b_qlora_bn.yaml`
- **Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Why the SFT format stays unchanged:** the repo already stores SFT rows as chat-style `messages`, and both `src/06_train_sft.py` and `src/07_eval.py` use `tokenizer.apply_chat_template(...)`, so adding the 3B path only requires separate configs and outputs.
- **Outputs:** `outputs/model_3b_qlora_bn/lora_adapter/`, `reports/eval_report_3b_qlora_bn.md`, `reports/eval_metrics_3b_qlora_bn.json`, `reports/generations_3b_qlora_bn.jsonl`

Optional pre-flight checks before launching the 3B run:

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

## Optional Gemma 2 2B IT (`google/gemma-2-2b-it`)

Separate **4-bit QLoRA** path using the same stored SFT files as Qwen (`data/sft/multilingual_1000/*.jsonl`). The dataset is **not** rebuilt: Gemma instruction-tuned chat templates expect **`user` / `model`** turns and **no `system` role**, so `src/06_train_sft.py` and `src/07_eval.py` normalize messages at formatting time (system text is prepended to the first user turn; `assistant` becomes `model`). Qwen configs are unchanged and still use Qwen-style roles end-to-end.

- **Train:** `configs/training_gemma2_2b_it_qlora_bn.yaml` → `outputs/model_gemma2_2b_it_qlora_bn/lora_adapter/`
- **Eval:** `configs/eval_gemma2_2b_it_qlora_bn.yaml` → `reports/eval_report_gemma2_2b_it_qlora_bn.md`, `eval_metrics_gemma2_2b_it_qlora_bn.json`, `generations_gemma2_2b_it_qlora_bn.jsonl`

**Preflight (chat template + LoRA target modules):**

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_fast=True)
print(bool(getattr(tok, "chat_template", None)))
print((tok.chat_template or "")[:400])
```

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
found = {name.split(".")[-1] for name, _ in model.named_modules()}
print("missing:", sorted(wanted - found))
```

**Commands:**

```bash
python src/06_train_sft.py --config configs/training_gemma2_2b_it_qlora_bn.yaml
python src/07_eval.py --config configs/eval_gemma2_2b_it_qlora_bn.yaml
```

This repository does **not** claim numerical results for Gemma until you run those configs locally. **Gemma 3 4B IT** is intentionally out of scope for this patch.

## Checkpoint selection

Training saves periodic checkpoints under the run output directory (e.g. `outputs/model_final/checkpoint-50`, `checkpoint-100`, …), with `save_steps` controlling spacing. The script also writes a final adapter to `outputs/<run>/lora_adapter/` after training completes.

**Do not assume the last checkpoint or the final `lora_adapter` folder is best on validation or test.** Early stopping can overfit; later checkpoints may underperform. For a paper or report, compare **several** checkpoints on the same evaluation setup—e.g. `checkpoint-50`, `checkpoint-100`, `checkpoint-150`, `checkpoint-200`, or whatever exists nearest to those steps given your `save_steps` and `max_steps`.

**Fair comparison:** use the **same** eval YAML (`evaluation`, `outputs`, `generation`, `num_samples_per_lang`) and only change which adapter you load.

**How to evaluate a specific checkpoint in this repo:** `src/07_eval.py` only accepts `--config` (there is no CLI flag to override the adapter path). The supported approach is:

1. Copy `configs/eval_final.yaml` (or `eval_7b_qlora_bn.yaml`) to a new file, e.g. `configs/eval_final_ckpt100.yaml`.
2. Set `model.lora_adapter_dir` to the checkpoint subdirectory **relative to `OUTPUTS_DIR`** (default `./outputs`), e.g. `model_final/checkpoint-100` if the checkpoint lives at `outputs/model_final/checkpoint-100/`.
3. Run: `python src/07_eval.py --config configs/eval_final_ckpt100.yaml`

Repeat for other checkpoints and compare metrics under `reports/` using the same sample cap.

## Configs

| Purpose | File |
|--------|------|
| Translation + SFT builder settings | `configs/translation_1000_api_bn.yaml` |
| Training (smoke) | `configs/training_demo.yaml` |
| Training (heavier) | `configs/training_final.yaml` |
| Training (3B QLoRA, optional) | `configs/training_3b_qlora_bn.yaml` |
| Training (7B QLoRA, optional) | `configs/training_7b_qlora_bn.yaml` |
| Training (Gemma 2 2B IT QLoRA, optional) | `configs/training_gemma2_2b_it_qlora_bn.yaml` |
| Eval (demo / final / 3B / 7B / Gemma) | `configs/eval_demo.yaml`, `configs/eval_final.yaml`, `configs/eval_3b_qlora_bn.yaml`, `configs/eval_7b_qlora_bn.yaml`, `configs/eval_gemma2_2b_it_qlora_bn.yaml` |
| Judge (Qwen 7B, 3B vs Gemma) | `configs/judge_7b_3b_vs_gemma.yaml` |

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

# 3d) Optional 3B QLoRA
python src/06_train_sft.py --config configs/training_3b_qlora_bn.yaml

# 3e) Optional Gemma 2 2B IT QLoRA
python src/06_train_sft.py --config configs/training_gemma2_2b_it_qlora_bn.yaml

# 4) Evaluate
python src/07_eval.py --config configs/eval_demo.yaml
python src/07_eval.py --config configs/eval_final.yaml
python src/07_eval.py --config configs/eval_3b_qlora_bn.yaml
python src/07_eval.py --config configs/eval_7b_qlora_bn.yaml
python src/07_eval.py --config configs/eval_gemma2_2b_it_qlora_bn.yaml

# 5) Optional judge (Qwen 7B compares 3B vs Gemma outputs)
python src/08_judge.py --config configs/judge_7b_3b_vs_gemma.yaml
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
make train-3b
make eval-3b
make train-7b
make eval-7b
make judge-7b-3b-vs-gemma
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
- LoRA adapter: `outputs/model_demo/lora_adapter/`, `outputs/model_final/lora_adapter/`, `outputs/model_3b_qlora_bn/lora_adapter/`, `outputs/model_7b_qlora_bn/lora_adapter/`, or `outputs/model_gemma2_2b_it_qlora_bn/lora_adapter/`
- Training checkpoints: `outputs/<run>/checkpoint-*`
- Training metadata: `outputs/<run>/train_run_metadata.json`
- Eval report, metrics, and JSONL generations: `reports/`
- Judge artifacts: `reports/judge_report_*.md`, `reports/judge_metrics_*.json`, `reports/judge_decisions_*.jsonl`

## Notes

- This cleaned submission keeps one canonical script per stage and only active configs under `configs/`.
- For reproducibility, keep config files and commands aligned with the tables above.
- Prompts are **not** duplicated in `05_build_sft.py` — they live in `src/utils/prompting.py` and are reapplied at eval time from `src/07_eval.py` so decoding uses the same template as training.
