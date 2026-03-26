"""Load editable persona text from disk each turn."""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG.parent

def persona_path() -> Path:
    raw = os.environ.get("PERSONA_FILE", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        return p
    return _PROJECT_ROOT / "persona.txt"


def _strip_hash_comments(text: str) -> str:
    """Drop full lines that start with # so you can annotate the file."""
    out: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def load_persona() -> str:
    path = persona_path()
    if not path.is_file():
        return ""
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    return _strip_hash_comments(raw)


def clear_persona_cache() -> None:
    # Backward-compatible no-op: persona is read fresh every turn.
    return None
