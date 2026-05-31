"""Small config helper for QUBO GUI and Copilot.

Stores/loads a JSON config file (defaults to the current working directory
`.qubo_config.json`). This keeps Gemini API key and other small settings.
"""
from pathlib import Path
import json
from typing import Dict, Any


def _default_path() -> Path:
    return Path.cwd() / '.qubo_config.json'


def load_config(path: Path | None = None) -> Dict[str, Any]:
    p = path or _default_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def save_config(data: Dict[str, Any], path: Path | None = None) -> bool:
    p = path or _default_path()
    try:
        p.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return True
    except Exception:
        return False


def get_gemini_key(path: Path | None = None) -> str | None:
    """Return Gemini API key if present in env or config file.

    Checks the `GEMINI_API_KEY` environment variable first, then looks in the
    config file under the `gemini_key` key.
    """
    import os
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ.get('GEMINI_API_KEY')
    cfg = load_config(path)
    return cfg.get('gemini_key')
