"""Shared prompt construction for SFT building and evaluation (next-utterance generation)."""

from __future__ import annotations

from typing import Any, Optional

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
) -> list[dict[str, str]]:
    """
    Rebuild generation-time messages from a JSONL record using the same system prompt logic as SFT.

    Strips the dataset's stored system message and replaces it with ``build_system_prompt`` output
    so eval matches the current template and YAML options.
    """
    msgs = list(rec.get("messages") or [])
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
    body: list[dict[str, str]] = []
    for m in msgs[:-1]:
        role = (m.get("role") or "").strip()
        if role == "system":
            continue
        content = _text_or_empty(m.get("content"))
        if not content or role not in ("user", "assistant"):
            continue
        body.append({"role": role, "content": content})
    return build_generation_messages(body, lang, system)
