# SFT Migration Note: Trainer Path -> Unsloth Path

This repository now has two Step 06 training paths:

- Legacy path: `src/06_train_sft.py` (HF `Trainer`-based, response-only masking)
- New path: `src/06_train_sft_unsloth.py` (Unsloth-native model/tokenizer + chat-template text SFT)

The migration is additive. The old script and old configs are kept unchanged.

## New Unsloth training flow

`src/06_train_sft_unsloth.py` uses:

- `FastLanguageModel.from_pretrained(...)` for model/tokenizer setup
- `get_chat_template(...)` to attach the right template
- `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)` to render each row into a `text` column
- `FastLanguageModel.get_peft_model(...)` for LoRA
- `SFTTrainer` after Unsloth model/tokenizer preparation

No custom response-only `labels`/`-100` masking is used in the Unsloth script.

## Dataset family alignment

Model and dataset family must match:

- Qwen models -> `sft/qwen_bn/{train,validation,test}.jsonl`
- Gemma model -> `sft/gemma_bn/{train,validation,test}.jsonl`

New Unsloth configs:

- `configs/training_3b_unsloth_bn.yaml`
- `configs/training_7b_unsloth_bn.yaml`
- `configs/training_gemma2_2b_it_unsloth_bn.yaml`

Evaluation configs must remain aligned:

- `configs/eval_3b_qlora_bn.yaml` -> `sft/qwen_bn/test.jsonl`
- `configs/eval_7b_qlora_bn.yaml` -> `sft/qwen_bn/test.jsonl`
- `configs/eval_gemma2_2b_it_qlora_bn.yaml` -> `sft/gemma_bn/test.jsonl`

## Chat template mapping

The Unsloth template is selected by model id:

- Qwen -> `qwen-2.5`
- Gemma -> `gemma2`

Helper for this mapping:

- `select_unsloth_chat_template(model_id)` in `src/utils/prompting.py`
