from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path so "ui" package is found (e.g. when run via streamlit run ui/app.py)
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from ui.utils_io import (
    Paths,
    extract_turns,
    format_act_emotion,
    guess_sft_dir,
    guess_translated_dir,
    read_jsonl,
    read_text,
)

# Load .env so DATA_DIR / OUTPUTS_DIR / REPORTS_DIR are set
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "./outputs")).resolve()
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports")).resolve()
for p in (DATA_DIR, OUTPUTS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

paths = Paths(DATA_DIR, OUTPUTS_DIR, REPORTS_DIR)

st.set_page_config(page_title="Multilingual DailyDialog Demo", layout="wide")
st.title("Multilingual DailyDialog Translation + LoRA SFT Demo")

st.caption(
    "This dashboard visualizes (1) aligned multilingual dialogues with emotion/dialog acts, "
    "(2) base vs LoRA chat comparison, (3) evaluation outputs, and (4) try-it chat (type a message → response in Bangla/ar/es)."
)

translated_dir = guess_translated_dir(paths.data_dir)
sft_dir = guess_sft_dir(paths.data_dir)

colA, colB, colC = st.columns(3)
with colA:
    st.caption("**DATA_DIR**")
    st.text(str(paths.data_dir))
with colB:
    st.caption("**Translated dir**")
    st.text(str(translated_dir) if translated_dir else "Not found")
with colC:
    st.caption("**SFT dir**")
    st.text(str(sft_dir) if sft_dir else "Not found")

# LoRA adapter path (prefer final, then demo); resolved under OUTPUTS_DIR
_candidate_adapters = [paths.outputs_dir / "model_final/lora_adapter", paths.outputs_dir / "model_demo/lora_adapter"]
_lora_adapter_path = next((p for p in _candidate_adapters if p.exists()), _candidate_adapters[0])
st.caption("**LoRA adapter** (for Try it / evaluation)")
st.text(str(_lora_adapter_path))
if not _lora_adapter_path.exists():
    st.caption("Not found — run `make pipeline-demo` or `make pipeline-final` to train and save the adapter.")


@st.cache_resource
def _load_model_and_tokenizer_cached(_base: str, _adapter_path: str | None):
    """Load base model (and optional LoRA); cached so we only load once per (base, adapter) pair."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(_base, device_map="auto")
    if _adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, _adapter_path)
    model.eval()
    return model, tok


tab1, tab2, tab3, tab4 = st.tabs(["Dataset Viewer", "Chat Compare", "Evaluation", "Try it"])

# ---------------- Tab 1: Dataset Viewer ----------------
with tab1:
    st.subheader("Dataset Viewer (Aligned Turns + Emotion + Dialog Acts)")

    if not translated_dir:
        st.warning("No translated dataset found. Run: make translate first.")
    else:
        split = st.selectbox("Split", ["test", "validation", "train"], index=0)
        split_path = translated_dir / f"{split}.jsonl"
        rows = read_jsonl(split_path, limit=400)

        if not rows:
            st.error(f"No rows loaded from {split_path}.")
        else:
            idx = st.slider("Conversation index", 0, len(rows) - 1, 0)
            rec = rows[idx]

            available = [k.replace("turns_", "") for k in rec if k.startswith("turns_") and isinstance(rec.get(k), list)]
            lang = st.selectbox("Target language", available or ["bn"], index=0)

            turns = extract_turns(rec, lang)
            if not turns:
                st.json(rec)
            else:
                st.write(f"Showing conversation **#{idx}** from **{split}**")
                st.markdown("---")
                for i, t in enumerate(turns):
                    left, right = st.columns(2)
                    with left:
                        st.markdown(f"**Turn {i+1} (EN)**")
                        st.write(t.get("en") or "")
                    with right:
                        st.markdown(f"**Turn {i+1} ({lang})**")
                        st.write(t.get("target") or "")
                    meta = format_act_emotion(t.get("act"), t.get("emotion"))
                    if meta:
                        st.caption(meta)
                    st.markdown("---")

# ---------------- Tab 2: Chat Compare ----------------
with tab2:
    st.subheader("Chat Compare (Base vs LoRA)")

    st.info(
        "This tab shows saved generations from reports. "
        "For a smooth demo, precompute generations into reports/generations_full.jsonl or reports/generations.jsonl."
    )

    lang = st.selectbox("Language", ["bn", "ar", "es"], index=0, key="chat_lang")

    gen_path = paths.reports_dir / "generations_full.jsonl"
    if not gen_path.exists():
        gen_path = paths.reports_dir / "generations.jsonl"
    gens = read_jsonl(gen_path, limit=200)

    if not gens:
        st.warning("No saved generations found. Add reports/generations.jsonl or reports/generations_full.jsonl with keys such as lang, baseline, lora (or base_response, lora_response).")
    else:
        filtered = [g for g in gens if g.get("lang") == lang] or gens
        sample = filtered[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Base model")
            st.write(sample.get("baseline") or sample.get("base") or sample.get("base_response") or "(no baseline key)")
        with c2:
            st.markdown("### LoRA model")
            st.write(sample.get("lora") or sample.get("lora_response") or sample.get("adapted") or "(no lora key)")
        st.caption("Tip: Put hand-picked examples into reports/generations_full.jsonl for the live demo.")

# ---------------- Tab 3: Evaluation ----------------
with tab3:
    st.subheader("Evaluation Viewer")

    report_final = paths.reports_dir / "eval_report_final.md"
    report_demo = paths.reports_dir / "eval_report_demo.md"
    report_path = report_final if report_final.exists() else report_demo

    if not report_path.exists():
        st.warning("No eval report found yet. Run `make eval-demo` or `make eval-final`.")
    else:
        st.markdown(f"Using report: `{report_path.name}`")
        st.markdown(read_text(report_path))

    st.markdown("---")
    st.markdown("### Key files")
    st.write("Reports directory:", str(paths.reports_dir))
    for p in sorted(paths.reports_dir.glob("*.md")):
        st.write("-", p.name)

# ---------------- Tab 4: Try it (live chat in target language) ----------------
with tab4:
    st.subheader("Try it — Get a response in your chosen language")

    st.caption(
        "Type a message (e.g. in English) and the model will reply in Bangla, Arabic, or Spanish. "
        "First run may load the model (slow); then generation is fast."
    )

    lang_names = {"bn": "Bangla", "ar": "Arabic", "es": "Spanish"}
    target_lang = st.selectbox(
        "Reply in",
        options=list(lang_names.keys()),
        format_func=lambda k: lang_names[k],
        index=0,
        key="tryit_lang",
    )
    use_lora = st.checkbox("Use LoRA-adapted model (if available)", value=True, key="tryit_lora")
    user_input = st.text_area("Your message", placeholder="e.g. Hello, how are you?", height=100, key="tryit_input")
    gen_clicked = st.button("Generate response", type="primary", key="tryit_btn")

    if gen_clicked and user_input.strip():
        with st.spinner("Loading model and generating…"):
            try:
                from src.utils.env import get_dirs, resolve_path

                dirs = get_dirs()
                eval_cfg_path = _REPO_ROOT / "configs" / "eval.yaml"
                if eval_cfg_path.exists():
                    import yaml
                    cfg = yaml.safe_load(eval_cfg_path.read_text(encoding="utf-8"))
                else:
                    cfg = {"model": {"base_model": "Qwen/Qwen2.5-7B-Instruct", "lora_adapter_dir": "model/lora_adapter"}}
                base_model = os.getenv("BASE_MODEL") or cfg["model"].get("base_model", "Qwen/Qwen2.5-7B-Instruct")
                adapter_rel = cfg["model"].get("lora_adapter_dir")
                adapter = resolve_path(adapter_rel, dirs["outputs"]) if adapter_rel else None
                use_adapter = use_lora and adapter and adapter.exists()
                adapter_str = str(adapter) if use_adapter else None

                model, tokenizer = _load_model_and_tokenizer_cached(base_model, adapter_str)

                system = f"You are a helpful assistant. Reply only in {lang_names.get(target_lang, target_lang)}."
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_input.strip()},
                ]
                prompt_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                new_ids = out[0][prompt_ids.shape[1]:]
                response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                st.success("Response")
                st.write(response)
                st.caption(f"Model: {'LoRA-adapted' if use_adapter else 'Base'} | Language: {lang_names.get(target_lang, target_lang)}")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
    elif gen_clicked and not user_input.strip():
        st.warning("Please enter a message.")
