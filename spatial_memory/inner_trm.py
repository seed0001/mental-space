"""
Tiny recursive emotional model (inner TRM): idle-time inner monologue over memory + personas.

Streams NDJSON lines: step, token, step_end, done, error.
Configure with INNER_TRM_MODEL, INNER_TRM_TEMPERATURE, INNER_TRM_NUM_PREDICT, etc.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from spatial_memory.config import LLAMA_MODEL
from spatial_memory.models import MemoryNode
from spatial_memory.ollama_client import chat_stream
from spatial_memory.persona import load_inner_persona, load_persona
from spatial_memory import store

INNER_TRM_MODEL = os.environ.get("INNER_TRM_MODEL", "").strip() or None
INNER_TRM_TEMPERATURE = float(os.environ.get("INNER_TRM_TEMPERATURE", "0.65"))
INNER_TRM_NUM_PREDICT = int(os.environ.get("INNER_TRM_NUM_PREDICT", "340"))
OUTER_PERSONA_MAX = int(os.environ.get("INNER_TRM_OUTER_PERSONA_CHARS", "2800"))
DIGEST_MAX_NODES = int(os.environ.get("INNER_TRM_DIGEST_NODES", "11"))
DIGEST_SNIPPET_CHARS = int(os.environ.get("INNER_TRM_DIGEST_CHARS", "240"))
CARRY_MAX_CHARS = int(os.environ.get("INNER_TRM_CARRY_CHARS", "800"))


def _inner_persona_block() -> str:
    t = load_inner_persona().strip()
    if t:
        return t
    return (
        "(No inner_persona.txt yet — improvise a restrained, embodied inner voice: first person, "
        "emotional honesty, no performance for an audience.)"
    )


def _format_digest(*, db_path: str | None) -> str:
    nodes = store.all_nodes(db_path=db_path)
    if not nodes:
        return ""

    def rank_key(n: MemoryNode) -> float:
        return float(n.current_relevance) * max(0.05, n.confidence)

    ranked = sorted(nodes, key=rank_key, reverse=True)
    lines: list[str] = []
    for n in ranked[:DIGEST_MAX_NODES]:
        u = (n.understanding or "").strip().replace("\r\n", "\n")
        if len(u) > DIGEST_SNIPPET_CHARS:
            u = u[: DIGEST_SNIPPET_CHARS - 1] + "…"
        lines.append(u)
    return "\n---\n".join(lines)


def _build_system(inner_voice: str) -> str:
    return (
        "You are one character's private inner voice — not their public chat persona.\n\n"
        "INNER LAYER DIRECTIVE (who this underside is):\n"
        f"{inner_voice}\n\n"
        "Rules:\n"
        "- First person only. Prose inner monologue; no bullet lists unless they feel like natural thought.\n"
        "- Embodied emotion: dread, tenderness, anger, relief, loneliness — whatever fits. No sadism, no bigotry.\n"
        "- The MEMORY NOTES are things this person carries; reflect on them, don't obediently summarize.\n"
        "- The OUTER PERSONA excerpt is how they present socially; you may feel gaps between inner truth and that mask.\n"
        "- Never address a reader, operator, or therapist. No 'As an AI'. No offers to help.\n"
        "- Do not output JSON or stage directions.\n"
    )


def _build_user(step: int, total: int, digest: str, outer: str, carry: str) -> str:
    parts = [
        "MEMORY NOTES (fragments they remember):",
        digest or "(Empty — almost no stored notes yet.)",
        "",
        "OUTER PERSONA — social face (excerpt):",
        outer if outer else "(No outer persona file — stay with the inner voice only.)",
        "",
    ]
    if step == 1:
        parts.append(
            f"Inner pass {step} of {total}: What rises first when you sit with these memories? "
            "Let associations move; stay raw and interior."
        )
    else:
        parts.append("Where your mind just went (carry forward; do not repeat verbatim):\n")
        parts.append(carry or "…")
        parts.append("")
        parts.append(
            f"Inner pass {step} of {total}: Go one layer deeper, or connect two threads emotionally — "
            "what sits underneath, or what you're afraid to say out loud?"
        )
    return "\n".join(parts)


def iter_inner_trm_ndjson(*, db_path: str | None, steps: int) -> Iterator[str]:
    store.init_db(db_path)
    outer = load_persona().strip()
    if len(outer) > OUTER_PERSONA_MAX:
        outer = outer[: OUTER_PERSONA_MAX - 1] + "…"
    digest = _format_digest(db_path=db_path)
    inner_voice = _inner_persona_block()
    model = INNER_TRM_MODEL or LLAMA_MODEL
    options: dict = {"num_predict": INNER_TRM_NUM_PREDICT}
    carry = ""
    total = max(1, min(int(steps), 10))

    for step in range(1, total + 1):
        yield json.dumps({"event": "step", "step": step, "total": total}, ensure_ascii=False) + "\n"
        system = _build_system(inner_voice)
        user = _build_user(step, total, digest, outer, carry)
        buf: list[str] = []
        try:
            for token in chat_stream(
                system,
                user,
                model=model,
                temperature=INNER_TRM_TEMPERATURE,
                options=options,
            ):
                buf.append(token)
                yield json.dumps({"event": "token", "text": token}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
            return
        text = "".join(buf).strip()
        carry = text[-CARRY_MAX_CHARS:] if text else ""
        yield json.dumps({"event": "step_end", "step": step}, ensure_ascii=False) + "\n"

    yield json.dumps({"event": "done"}, ensure_ascii=False) + "\n"


__all__ = ["iter_inner_trm_ndjson"]
