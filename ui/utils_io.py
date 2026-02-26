from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Paths:
    data_dir: Path
    outputs_dir: Path
    reports_dir: Path


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(items) >= limit:
                break
    return items


def read_text(path: Path, max_chars: int = 80_000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="replace")
    return txt[:max_chars]


def list_jsonl_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.jsonl") if p.is_file()])


def guess_translated_dir(data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "translated_dailydialog_en_bn_ar_es",
        data_dir / "translated_dailydialog",
    ]
    for c in candidates:
        if (c / "train.jsonl").exists() and (c / "test.jsonl").exists():
            return c
    best: Optional[Path] = None
    best_mtime = 0.0
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "train.jsonl").exists() and (d / "test.jsonl").exists():
            m = d.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = d, m
    return best


def guess_sft_dir(data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "sft" / "multilingual",
        data_dir / "sft",
    ]
    for c in candidates:
        if (c / "train.jsonl").exists() and (c / "test.jsonl").exists():
            return c
    best: Optional[Path] = None
    best_mtime = 0.0
    for d in data_dir.rglob("sft"):
        if d.is_dir() and (d / "train.jsonl").exists():
            m = d.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = d, m
    return best


# Code → label mapping: single source of truth in src/utils/dailydialog_labels.py
try:
    from src.utils.dailydialog_labels import ACT_LABELS, EMOTION_LABELS
except ImportError:
    ACT_LABELS = ["inform", "question", "directive", "commissive"]
    EMOTION_LABELS = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def format_act_emotion(act: Any = None, emotion: Any = None) -> str:
    """Return human-readable 'Act: X | Emotion: Y' for display; -1 or invalid → '—'."""
    a = "—"
    if act is not None and isinstance(act, int) and 0 <= act < len(ACT_LABELS):
        a = ACT_LABELS[act]
    elif act is not None and act != -1:
        a = str(act)
    e = "—"
    if emotion is not None and isinstance(emotion, int) and 0 <= emotion < len(EMOTION_LABELS):
        e = EMOTION_LABELS[emotion]
    elif emotion is not None and emotion != -1:
        e = str(emotion)
    return f"Act: {a} | Emotion: {e}"


def extract_turns(record: Dict[str, Any], lang: str = "bn") -> List[Dict[str, Any]]:
    """Build turn list from our schema: turns_en, turns_<lang>, dialog_acts, emotions."""
    turns_en = record.get("turns_en") or []
    turns_tgt = record.get(f"turns_{lang}") or []
    acts = record.get("dialog_acts") or []
    emotions = record.get("emotions") or []
    n = min(len(turns_en), len(turns_tgt))
    out = []
    for i in range(n):
        out.append({
            "en": turns_en[i] if i < len(turns_en) else "",
            "target": turns_tgt[i] if i < len(turns_tgt) else "",
            "act": acts[i] if i < len(acts) else None,
            "emotion": emotions[i] if i < len(emotions) else None,
        })
    if out:
        return out
    for key in ("turns", "dialogue", "messages"):
        if key in record and isinstance(record[key], list):
            return record[key]
    return []
