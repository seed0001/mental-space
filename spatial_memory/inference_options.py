from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResponseInferenceOptions:
    """Generation settings for the main spatial-memory chat reply (Ollama /api/chat)."""

    model: str | None = None
    temperature: float = 0.4
    top_p: float | None = None
    top_k: int | None = None
    num_predict: int | None = None
    repeat_penalty: float | None = None

    def to_ollama_options(self) -> dict[str, Any]:
        o: dict[str, Any] = {"temperature": float(self.temperature)}
        if self.top_p is not None:
            o["top_p"] = float(self.top_p)
        if self.top_k is not None:
            o["top_k"] = int(self.top_k)
        if self.num_predict is not None:
            o["num_predict"] = int(self.num_predict)
        if self.repeat_penalty is not None:
            o["repeat_penalty"] = float(self.repeat_penalty)
        return o
