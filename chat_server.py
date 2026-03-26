"""
Spatial memory chat UI. Run:

  uvicorn chat_server:app --reload --host 127.0.0.1 --port 8765

Or: python launch.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from spatial_memory.pipeline import process_message, process_message_stream
from spatial_memory import store

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="Spatial Memory Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)


class ChatOut(BaseModel):
    reply: str
    x: float
    y: float
    z: float
    commitment: str
    confidence: float
    caution: bool
    rule_id: str = ""
    rationale: str = ""


class TtsIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=32000)


@app.get("/")
async def index():
    path = STATIC / "index.html"
    if not path.is_file():
        raise HTTPException(500, "static/index.html missing")
    return FileResponse(path)


@app.get("/api/space")
def space_snapshot():
    db = os.environ.get("SPATIAL_MEMORY_DB")
    try:
        return store.graph_snapshot(db_path=db)
    except Exception as e:
        raise HTTPException(502, f"space snapshot failed: {e!s}") from e


@app.post("/api/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    db = os.environ.get("SPATIAL_MEMORY_DB")
    try:
        out = process_message(text, db_path=db)
    except Exception as e:
        raise HTTPException(502, f"pipeline failed: {e!s}") from e
    d = out.decision
    return ChatOut(
        reply=out.response,
        x=out.coordinate[0],
        y=out.coordinate[1],
        z=out.coordinate[2],
        commitment=out.commitment_type.value,
        confidence=d.confidence_level,
        caution=d.caution_internal_conflict,
        rule_id=d.rule_id,
        rationale=d.rationale,
    )


@app.post("/api/chat/stream")
def chat_stream_ndjson(body: ChatIn):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    db = os.environ.get("SPATIAL_MEMORY_DB")

    def gen():
        try:
            for ev in process_message_stream(text, db_path=db):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"event": "done", "data": {"reply": "", "commitment": "error", "confidence": 0.0, "caution": True}},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/tts")
async def synthesize_tts(body: TtsIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    try:
        communicate = edge_tts.Communicate(
            text,
            voice="en-GB-RyanNeural",
            rate="+0%",
            pitch="+0Hz",
        )
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data", b""))
        if not audio:
            raise RuntimeError("no audio returned")
        return Response(content=bytes(audio), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(502, f"tts failed: {e!s}") from e
