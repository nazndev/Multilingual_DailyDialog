"""
Shared logging for pipeline scripts: console + file, sanitized config, banners, timers.
Log files: REPORTS_DIR/logs/YYYYMMDD_HHMMSS_<script>.log
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator

# Repo root for path insertion (caller sets up get_dirs)
def _reports_dir() -> Path:
    from src.utils.env import get_dirs
    return get_dirs()["reports"]


def _mask_secret(key: str) -> bool:
    k = key.lower()
    return any(x in k for x in ("token", "key", "secret", "password", "auth"))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: "***" if _mask_secret(str(k)) else _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def setup_logger(script_name: str) -> logging.Logger:
    """Return a logger that writes to console and to REPORTS_DIR/logs/YYYYMMDD_HHMMSS_<script>.log."""
    reports = _reports_dir()
    log_dir = reports / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stamp}_{script_name}.log"

    logger = logging.getLogger(f"pipeline.{script_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    date_fmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger


def banner(logger: logging.Logger, title: str, char: str = "=") -> None:
    """Log a start/end banner."""
    line = char * 60
    logger.info(line)
    logger.info("  %s", title)
    logger.info(line)


def log_config_safely(logger: logging.Logger, cfg: Dict[str, Any], prefix: str = "config") -> None:
    """Log config with secret-like keys masked."""
    safe = _sanitize(cfg)
    logger.info("%s: %s", prefix, json.dumps(safe, indent=2, default=str))


def log_env_safely(logger: logging.Logger, keys: list[str]) -> None:
    """Log selected env vars (DATA_DIR, TARGET_LANGS, etc.); never log token/key/secret."""
    import os
    out = {}
    for k in keys:
        if _mask_secret(k):
            continue
        v = os.environ.get(k)
        if v is not None:
            out[k] = v
    if out:
        logger.info("env: %s", json.dumps(out, sort_keys=True))


@contextmanager
def timer(logger: logging.Logger, step_name: str) -> Iterator[None]:
    """Context manager to log duration of a step."""
    start = datetime.now()
    logger.info("START %s", step_name)
    try:
        yield
    finally:
        elapsed = (datetime.now() - start).total_seconds()
        logger.info("END %s | elapsed %.2f s", step_name, elapsed)


def summarize_jsonl(logger: logging.Logger, path: Path, parse_limit: int = 50000) -> None:
    """Count lines, show first record keys, and average turns if available."""
    if not path.exists():
        logger.warning("summarize_jsonl: path does not exist: %s", path)
        return
    count = 0
    total_turns = 0
    first_keys = None
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            count += 1
            if count <= parse_limit:
                try:
                    rec = json.loads(line)
                    if first_keys is None:
                        first_keys = list(rec.keys())
                    n = rec.get("num_turns")
                    if n is not None:
                        total_turns += n
                    elif "turns_en" in rec:
                        total_turns += len(rec["turns_en"])
                    elif "messages" in rec:
                        total_turns += len(rec["messages"])
                except json.JSONDecodeError:
                    pass
    logger.info("path=%s | records=%s | first_record_keys=%s", path, count, first_keys)
    if count > 0 and total_turns > 0:
        avg = total_turns / min(count, parse_limit)
        logger.info("avg_turns_per_record=%.2f", avg)
