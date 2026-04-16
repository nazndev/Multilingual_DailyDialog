"""Shared prompt construction for SFT building and evaluation (next-utterance generation)."""

from __future__ import annotations

from typing import Any, Optional

MODEL_FAMILY_DEFAULT = "default"
MODEL_FAMILY_GEMMA = "gemma"

# Default template: concise next-utterance task; {lang} is substituted per record.
DEFAULT_SYSTEM_TEMPLATE = (
    "Generate only the next assistant utterance in '{lang}'. "
    "Reply naturally and conversationally to the last user message. "
    "Keep it brief and contextually appropriate. "
    "Do not repeat the history or add explanations, labels, or extra text. "
    "Output only the reply in '{lang}'."
)

SHORT_REPLY_HINT = (
    "Prefer a short human-like reply. "
    "Use one sentence when appropriate."
)


def _text_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_system_prompt(
    lang: str,
    style: str = "default",
    *,
    use_emotion_tag: bool = False,
    use_dialog_act_tag: bool = False,
    emotion: Optional[str] = None,
    dialog_act: Optional[str] = None,
    short_reply_hint: bool = False,
    system_template: Optional[str] = None,
) -> str:
    """
    Build the system prompt for multilingual next-utterance generation.

    ``system_template`` may contain ``{lang}`` and optionally ``{emotion}``, ``{dialog_act}``
    placeholders when tags are enabled.
    """
    lang = (lang or "bn").strip() or "bn"
    template = (system_template or DEFAULT_SYSTEM_TEMPLATE)
    template = template.strip() if isinstance(template, str) else ""
    if not template:
        template = DEFAULT_SYSTEM_TEMPLATE.strip()
    emo = _text_or_empty(emotion) if emotion is not None else ""
    act = _text_or_empty(dialog_act) if dialog_act is not None else ""

    try:
        base = template.format(lang=lang, emotion=emo, dialog_act=act)
    except KeyError:
        # Backward compatibility: only {lang} required.
        base = template.format(lang=lang)

    parts: list[str] = [base.rstrip()]

    if style and style != "default":
        parts.append(f"(Style hint: {style}.)")

    if use_emotion_tag and emo:
        parts.append(f"Target turn emotion label: {emo}.")
    if use_dialog_act_tag and act:
        parts.append(f"Target turn dialog act label: {act}.")

    if short_reply_hint:
        parts.append(SHORT_REPLY_HINT)

    return " ".join(p for p in parts if p).strip()


def normalize_messages_for_model(
    messages: list[dict[str, Any]],
    model_family: str,
) -> list[dict[str, str]]:
    """
    Adapt chat messages for the tokenizer's expected roles.

    Qwen-style instruct models use ``system`` / ``user`` / ``assistant``. Gemma IT chat
    templates use ``user`` / ``model`` only; system instructions must be folded into the
    first user turn. Stored SFT rows stay Qwen-shaped; call this before
    ``apply_chat_template`` when ``model_family`` is ``\"gemma\"``.
    """
    fam = (model_family or MODEL_FAMILY_DEFAULT).strip().lower()
    if fam != MODEL_FAMILY_GEMMA:
        return [dict(m) for m in messages if isinstance(m, dict)]

    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = _text_or_empty(m.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "user":
            turns.append(("user", content))
        elif role in ("assistant", "model"):
            turns.append(("assistant", content))

    system_text = "\n\n".join(system_parts).strip()

    if not turns:
        return []

    i_first_user = next((i for i, (r, _) in enumerate(turns) if r == "user"), None)
    if i_first_user is None:
        if system_text:
            turns.insert(0, ("user", system_text))
    elif system_text:
        _r, c = turns[i_first_user]
        turns[i_first_user] = (
            "user",
            f"{system_text}\n\n{c}".strip() if c else system_text,
        )

    out: list[dict[str, str]] = []
    for r, c in turns:
        gemma_role = "model" if r == "assistant" else "user"
        out.append({"role": gemma_role, "content": c})
    return out


def to_gemma_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Qwen-style messages into Gemma chat roles (`user`/`model`)."""
    return normalize_messages_for_model(messages, MODEL_FAMILY_GEMMA)


def select_unsloth_chat_template(model_id: str) -> str:
    """
    Select the Unsloth chat template name from the model id.

    - Qwen 2.5 models -> ``qwen-2.5``
    - Gemma 2 IT models -> ``gemma2``
    """
    lowered = (model_id or "").lower()
    if "qwen" in lowered:
        return "qwen-2.5"
    if "gemma" in lowered:
        return "gemma2"
    raise ValueError(f"Unsupported model for Unsloth chat template selection: {model_id!r}")


def build_generation_messages(
    history_messages: list[dict[str, Any]],
    lang: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """
    Build chat messages for ``tokenizer.apply_chat_template`` (generation prompt, no assistant target).

    ``history_messages`` should be user/assistant turns only (no system, no final target).
    """
    out: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in history_messages:
        role = (m.get("role") or "").strip()
        content = _text_or_empty(m.get("content"))
        if not content or role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": content})
    return out


def messages_for_generation_from_record(
    rec: dict[str, Any],
    *,
    use_emotion_tag: bool = False,
    use_dialog_act_tag: bool = False,
    short_reply_hint: bool = False,
    style: str = "default",
    system_template: Optional[str] = None,
    model_family: str = MODEL_FAMILY_DEFAULT,
) -> list[dict[str, str]]:
    """
    Rebuild generation-time messages from a JSONL record using the same system prompt logic as SFT.

    Strips the dataset's stored system message and replaces it with ``build_system_prompt`` output
    so eval matches the current template and YAML options.
    """
    msgs = [dict(m) for m in list(rec.get("messages") or []) if isinstance(m, dict)]
    if len(msgs) < 2:
        return []
    lang = (rec.get("lang") or "bn").strip() or "bn"
    emotion = rec.get("emotion_at_turn")
    act = rec.get("act_at_turn")
    system = build_system_prompt(
        lang,
        style=style,
        use_emotion_tag=use_emotion_tag,
        use_dialog_act_tag=use_dialog_act_tag,
        emotion=str(emotion) if emotion is not None else None,
        dialog_act=str(act) if act is not None else None,
        short_reply_hint=short_reply_hint,
        system_template=system_template,
    )
    normalized = normalize_messages_for_model(msgs, MODEL_FAMILY_GEMMA if any((m.get("role") or "").strip() == "model" for m in msgs) else MODEL_FAMILY_DEFAULT)
    if not normalized:
        return []
    body: list[dict[str, str]] = []
    for m in normalized[:-1]:
        role = (m.get("role") or "").strip()
        content = _text_or_empty(m.get("content"))
        if not content:
            continue
        if role == "model":
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        body.append({"role": role, "content": content})
    qwen_style = build_generation_messages(body, lang, system)
    return normalize_messages_for_model(qwen_style, model_family)
