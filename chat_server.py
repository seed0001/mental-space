"""
Spatial memory chat UI. Run:

  uvicorn chat_server:app --reload --host 127.0.0.1 --port 8765

Or: python launch.py
"""

from __future__ import annotations

import json
import os
import platform
import socket
from pathlib import Path

# Resolve persona / DB paths against this app directory (must run before spatial_memory imports).
ROOT = Path(__file__).resolve().parent
if not os.environ.get("SPATIAL_MEMORY_PROJECT_ROOT", "").strip():
    os.environ["SPATIAL_MEMORY_PROJECT_ROOT"] = str(ROOT)

import edge_tts
import httpx
import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from spatial_memory.config import LLAMA_MODEL, OLLAMA_BASE_URL
from spatial_memory.inference_options import ResponseInferenceOptions
from spatial_memory.ollama_client import chat as ollama_chat
from spatial_memory.inner_trm import iter_inner_trm_ndjson
from spatial_memory.persona import inner_persona_path, persona_path
from spatial_memory.pipeline import process_message, process_message_stream
from spatial_memory import store

STATIC = ROOT / "static"


def _response_inference(body: ChatIn) -> ResponseInferenceOptions | None:
    if body.inference is None:
        return None
    i = body.inference
    return ResponseInferenceOptions(
        model=i.model,
        temperature=i.temperature,
        top_p=i.top_p,
        top_k=i.top_k,
        num_predict=i.num_predict,
        repeat_penalty=i.repeat_penalty,
    )


app = FastAPI(title="Spatial Memory Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatInferenceBody(BaseModel):
    """Ollama /api/chat options for the assistant reply (classifier / memory pipeline unchanged)."""

    model: str | None = Field(default=None, max_length=256)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=1_000_000)
    num_predict: int | None = Field(default=None, ge=1, le=256_000)
    repeat_penalty: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("model", mode="before")
    @classmethod
    def _empty_model_to_none(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    inference: ChatInferenceBody | None = None


class ChatOut(BaseModel):
    reply: str
    x: float
    y: float
    z: float
    w: float = 0.0
    v: float = 0.0
    commitment: str
    confidence: float
    caution: bool
    rule_id: str = ""
    rationale: str = ""
    deep_remember: bool = False
    consolidation: dict | None = None


class TtsIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=32000)


class PersonaPut(BaseModel):
    content: str = Field(default="", max_length=120_000)


class InnerPersonaPut(BaseModel):
    content: str = Field(default="", max_length=120_000)


class PersonaEnhanceIn(BaseModel):
    hint: str = Field(..., min_length=1, max_length=8000)
    include_current: bool = Field(
        default=True,
        description="Pass current persona file text as context for the model.",
    )


class PersonaEnhanceOut(BaseModel):
    enhanced: str


_ENHANCE_SYSTEM = """You expand short directions into rich persona text for someone who will speak in first person in chat—as a single believable human, not as software.
Output ONLY the enhanced persona (sections, bullets, or prose). No preamble, no "Here is", no markdown code fences.
Keep directions concrete and behavioral. If the user asks for traits, make them actionable on the page."""


@app.get("/")
async def index():
    path = STATIC / "index.html"
    if not path.is_file():
        raise HTTPException(500, "static/index.html missing")
    return FileResponse(path)


def _primary_disk_path() -> str:
    if os.name == "nt":
        drive = os.environ.get("SystemDrive", "C:")
        path = drive + "\\"
        return path if os.path.isdir(path) else "C:\\"
    return "/"


@app.get("/api/ollama/models")
def ollama_models():
    """List tags from the local Ollama daemon (for UI model picker)."""
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=10.0) as c:
            r = c.get("/api/tags")
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise HTTPException(
            502,
            f"Could not reach Ollama at {OLLAMA_BASE_URL}: {e!s}",
        ) from e
    names = [m.get("name", "").strip() for m in data.get("models", []) if m.get("name")]
    return {"models": sorted(set(names)), "default_model": LLAMA_MODEL, "base_url": OLLAMA_BASE_URL}


@app.get("/api/system")
def system_metrics():
    """Live host metrics for HUD (local use only)."""
    disk_path = _primary_disk_path()
    try:
        du = psutil.disk_usage(disk_path)
    except OSError:
        du = psutil.disk_usage("/")
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    mount_label = disk_path.rstrip("\\/") or disk_path
    return {
        "hostname": socket.gethostname(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "status": "NOMINAL",
        "cpu_percent": round(cpu, 1),
        "cpu_cores_logical": psutil.cpu_count(logical=True) or 0,
        "ram_percent": round(vm.percent, 1),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "disk_mount": mount_label,
        "disk_percent": round(du.percent, 1),
        "disk_used_gb": round(du.used / (1024**3), 2),
        "disk_total_gb": round(du.total / (1024**3), 2),
    }


@app.get("/api/persona")
def get_persona():
    path = persona_path().resolve()
    exists = path.is_file()
    raw = ""
    if exists:
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError as e:
            raise HTTPException(502, f"read persona failed: {e!s}") from e
    return {
        "content": raw,
        "path": str(path),
        "exists": exists,
    }


@app.put("/api/persona")
def put_persona(body: PersonaPut):
    path = persona_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.content, encoding="utf-8", newline="\n")
    except OSError as e:
        raise HTTPException(502, f"write persona failed: {e!s}") from e
    return {"ok": True, "path": str(path)}


@app.get("/api/inner-persona")
def get_inner_persona():
    path = inner_persona_path().resolve()
    exists = path.is_file()
    raw = ""
    if exists:
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError as e:
            raise HTTPException(502, f"read inner persona failed: {e!s}") from e
    return {
        "content": raw,
        "path": str(path),
        "exists": exists,
    }


@app.put("/api/inner-persona")
def put_inner_persona(body: InnerPersonaPut):
    path = inner_persona_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.content, encoding="utf-8", newline="\n")
    except OSError as e:
        raise HTTPException(502, f"write inner persona failed: {e!s}") from e
    return {"ok": True, "path": str(path)}


@app.post("/api/inner/trm/stream")
def inner_trm_stream(steps: int = Query(3, ge=1, le=10, description="Recursive inner passes")):
    """NDJSON stream: step, token, step_end, done, error. For idle emotional background thinking."""
    db = os.environ.get("SPATIAL_MEMORY_DB")

    def gen():
        try:
            for line in iter_inner_trm_ndjson(db_path=db, steps=steps):
                yield line
        except Exception as e:
            yield json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/persona/enhance", response_model=PersonaEnhanceOut)
def enhance_persona(body: PersonaEnhanceIn):
    current = ""
    if body.include_current:
        path = persona_path()
        if path.is_file():
            try:
                current = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                current = ""
    user_msg = (
        "Current persona file (verbatim; may be empty):\n---\n"
        f"{current}\n---\n\n"
        "Enhancement direction from the operator:\n"
        f"{body.hint.strip()}\n\n"
        "Produce the full revised or expanded persona text only."
    )
    try:
        enhanced = ollama_chat(_ENHANCE_SYSTEM, user_msg, temperature=0.35).strip()
    except Exception as e:
        raise HTTPException(502, f"persona enhance failed (is Ollama running?): {e!s}") from e
    if enhanced.startswith("```"):
        lines = enhanced.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        enhanced = "\n".join(lines).strip()
    if not enhanced:
        raise HTTPException(502, "model returned empty enhancement")
    return PersonaEnhanceOut(enhanced=enhanced)


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
        out = process_message(text, db_path=db, inference=_response_inference(body))
    except Exception as e:
        raise HTTPException(502, f"pipeline failed: {e!s}") from e
    d = out.decision
    return ChatOut(
        reply=out.response,
        x=out.coordinate[0],
        y=out.coordinate[1],
        z=out.coordinate[2],
        w=out.coordinate[3],
        v=out.coordinate[4],
        commitment=out.commitment_type.value,
        confidence=d.confidence_level,
        caution=d.caution_internal_conflict,
        rule_id=d.rule_id,
        rationale=d.rationale,
        deep_remember=out.deep_remember,
        consolidation=out.consolidation,
    )


@app.post("/api/chat/stream")
def chat_stream_ndjson(body: ChatIn):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    db = os.environ.get("SPATIAL_MEMORY_DB")

    def gen():
        try:
            for ev in process_message_stream(text, db_path=db, inference=_response_inference(body)):
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
