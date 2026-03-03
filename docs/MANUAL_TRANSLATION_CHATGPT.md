# Manual translation via ChatGPT (web)

Use this if you want to translate with ChatGPT in the browser instead of the API (e.g. no API key or quota).

## 1. Which file to give

- **Input:** The **processed** English dialogues (one JSON object per line = JSONL).
- **Paths (under `data/`, relative to repo root):**
  - `data/processed/train_en.jsonl`
  - `data/processed/validation_en.jsonl`
  - `data/processed/test_en.jsonl`

You must run **Step 01 (Download)** and **Step 02 (Preprocess)** first so these files exist.

Because of ChatGPT’s context limit, **don’t paste the whole file**. Use a **small chunk** (e.g. first 5–20 lines) per conversation, then combine the outputs (see below).

## 2. Format of each line (input)

Each line is a single JSON object, for example:

```json
{"dialogue_id": "abc123...", "turns_en": ["Hello!", "How are you?", "I'm fine."], "num_turns": 3, "dialog_acts": [1, 2, 2], "emotions": [0, 0, 0]}
```

- `turns_en`: list of English utterances to translate.
- Keep `dialogue_id`, `num_turns`, `dialog_acts`, `emotions` unchanged.

## 3. Prompt to use in ChatGPT

Copy-paste this (optionally adjust language list if you only want some of bn/ar/es):

```
I will give you JSONL: one JSON object per line. Each object has "turns_en" (list of English dialogue utterances).

For each line:
1. Translate every string in "turns_en" into:
   - Bengali (add key "turns_bn")
   - Arabic (add key "turns_ar")
   - Spanish (add key "turns_es")
2. Keep all original keys and values (dialogue_id, turns_en, num_turns, dialog_acts, emotions).
3. Add: "translation_meta": {"backend": "manual", "model": "chatgpt", "quality_flags": []}
4. Output must be valid JSONL: exactly one JSON object per line, no extra text or markdown.

Translate naturally and keep the same number of turns in each language. Output only the JSONL.
```

Then paste a few lines from one of the files (e.g. 5–20 lines from `train_en.jsonl`).

## 4. What you get back

ChatGPT should return lines like:

```json
{"dialogue_id": "abc123...", "turns_en": ["Hello!", ...], "num_turns": 3, "dialog_acts": [...], "emotions": [...], "translation_meta": {"backend": "manual", "model": "chatgpt", "quality_flags": []}, "turns_bn": ["...", ...], "turns_ar": ["...", ...], "turns_es": ["...", ...]}
```

Each line must have: `dialogue_id`, `turns_en`, `num_turns`, `turns_bn`, `turns_ar`, `turns_es`, `translation_meta`, and optionally `dialog_acts`, `emotions`.

## 5. Where to save

- **Directory:** `data/translated_dailydialog_en_bn_ar_es/`
- **Files:** Save as:
  - `data/translated_dailydialog_en_bn_ar_es/train.jsonl`
  - `data/translated_dailydialog_en_bn_ar_es/validation.jsonl`
  - `data/translated_dailydialog_en_bn_ar_es/test.jsonl`

Create the directory if needed. For each split, you can do multiple ChatGPT runs (e.g. lines 1–20, then 21–40, …) and **append** all output lines into the same file (e.g. `train.jsonl`). Order of lines should match the original split.

## 6. After that

Run the rest of the pipeline from **Step 05 (Build SFT)** (Step 03 and 04 are skipped when you translate manually):

```bash
make build_sft    # or: python src/05_build_sft.py --config configs/translation.yaml
make train_sft    # then train
make eval         # then evaluate
```

Step 05 reads from `data/translated_dailydialog_en_bn_ar_es/*.jsonl` and builds SFT data; it doesn’t care whether translations came from the API or from manual ChatGPT.
