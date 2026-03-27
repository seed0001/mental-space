from __future__ import annotations

import json
from typing import Any

import httpx

from spatial_memory.config import EMBED_MODEL, LLAMA_MODEL, OLLAMA_BASE_URL


def _client() -> httpx.Client:
    return httpx.Client(base_url=OLLAMA_BASE_URL, timeout=120.0)


def _merge_ollama_options(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if not extra:
        return out
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out


def chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    options: dict[str, Any] | None = None,
) -> str:
    m = model or LLAMA_MODEL
    payload: dict[str, Any] = {
        "model": m,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": _merge_ollama_options({"temperature": temperature}, options),
    }
    if json_mode:
        payload["format"] = "json"
    with _client() as c:
        r = c.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return data["message"]["content"]


def embed(text: str, *, model: str | None = None) -> list[float]:
    m = model or EMBED_MODEL
    with _client() as c:
        r = c.post("/api/embeddings", json={"model": m, "prompt": text})
        r.raise_for_status()
        data = r.json()
    return data["embedding"]


def chat_stream(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    options: dict[str, Any] | None = None,
):
    m = model or LLAMA_MODEL
    payload: dict[str, Any] = {
        "model": m,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": _merge_ollama_options({"temperature": temperature}, options),
    }
    with _client() as c:
        with c.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("done"):
                    break
                msg = chunk.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield piece


def parse_json_loose(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]!r}")
