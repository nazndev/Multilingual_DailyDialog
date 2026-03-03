#!/usr/bin/env python3
"""
Local test: verify GPT translation works. Requires .env with OPENAI_API_KEY.
Run from repo root: python scripts/test_gpt_translate.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env from repo root before any other imports that read env
_env_file = REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=True)
except ImportError:
    pass

# Fallback: if OPENAI_API_KEY still missing (e.g. dotenv didn't load it), parse .env for that key
try:
    if not os.environ.get("OPENAI_API_KEY") and _env_file.exists():
        for line in _env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and value:
                value = value.strip().strip("'\"").strip()
                if value:
                    os.environ["OPENAI_API_KEY"] = value
                    break
except Exception:
    pass

from src.utils.env import get_env


def load_cfg(path):
    try:
        import yaml
    except ModuleNotFoundError:
        print("ERROR: Missing dependency. Install with: pip install pyyaml")
        print("  Or from repo root: pip install -r requirements.txt")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def translate_one_api(cfg, text: str, src: str = "en", tgt: str = "bn") -> str:
    """Same logic as 03_translate.translate_one_api."""
    LANG_NAMES = {"bn": "Bengali", "ar": "Arabic", "es": "Spanish", "en": "English"}
    api_cfg = cfg.get("api", {})
    provider = (api_cfg.get("provider") or "openai").lower()
    model_name = api_cfg.get("model") or "gpt-4o-mini"
    target_lang = LANG_NAMES.get(tgt, tgt)
    prompt = (
        f"Translate the following English text to {target_lang}. "
        "Preserve meaning, tone, and naturalness. Output only the translation, no explanation or quotes.\n\n"
        f"English: {text}"
    )
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}")
    from openai import OpenAI
    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env or environment")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(api_cfg.get("max_tokens", 256)),
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def _env_keys(path: Path) -> list[str]:
    """Return names of variables defined in .env (non-empty, uncommented). No values."""
    names = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                names.append(line.split("=", 1)[0].strip())
    except Exception:
        pass
    return names


def main():
    key = get_env("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        env_path = REPO_ROOT / ".env"
        print(f"  .env path: {env_path}")
        print(f"  .env exists: {env_path.exists()}")
        if env_path.exists():
            keys = _env_keys(env_path)
            print(f"  Variables in .env: {keys}")
            if "OPENAI_API_KEY" not in keys:
                print("  Tip: Add uncommented line: OPENAI_API_KEY=sk-... (no spaces around =)")
        sys.exit(1)

    cfg_path = REPO_ROOT / "configs" / "translation.yaml"
    cfg = load_cfg(cfg_path)
    model = cfg.get("api", {}).get("model", "gpt-4o-mini")
    print(f"Testing GPT translation (model: {model})")
    print()

    tests = [("Hello, how are you?", "bn"), ("The weather is nice today.", "es")]
    for text, lang in tests:
        try:
            out = translate_one_api(cfg, text, "en", lang)
            print(f"  EN: {text}")
            print(f"  {lang.upper()}: {out}")
            print()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "insufficient_quota" in err_str:
                print("  API returned 429 (quota exceeded). Key and config are OK; add billing/credits at https://platform.openai.com/account/billing")
            else:
                print(f"  ERROR: {e}")
            sys.exit(1)

    print("GPT translation test OK.")


if __name__ == "__main__":
    main()
