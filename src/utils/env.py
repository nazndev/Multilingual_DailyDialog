from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


_load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else (default if default is not None else "")


def get_langs() -> Dict[str, List[str]]:
    source = get_env("SOURCE_LANG", "en")
    targets_raw = get_env("TARGET_LANGS", "bn")
    targets = [x.strip() for x in targets_raw.split(",") if x.strip()]
    return {"source": source, "targets": targets}


def get_dirs() -> Dict[str, Path]:
    data_dir = Path(get_env("DATA_DIR", "./data")).resolve()
    cache_dir = Path(get_env("CACHE_DIR", "./cache")).resolve()
    outputs_dir = Path(get_env("OUTPUTS_DIR", "./outputs")).resolve()
    reports_dir = Path(get_env("REPORTS_DIR", "./reports")).resolve()
    for p in (data_dir, cache_dir, outputs_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "data": data_dir,
        "cache": cache_dir,
        "outputs": outputs_dir,
        "reports": reports_dir,
    }


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()
