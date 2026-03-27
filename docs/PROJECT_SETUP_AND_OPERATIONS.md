# Project setup and operations

This document describes **repository layout**, **how to run** the spatial-memory application, **environment variables**, **HTTP APIs**, and **how the pieces fit together** at deployment level.

---

## 1. What this project is

- **Backend:** Python 3.12+ with **FastAPI** (`chat_server.py`), **SQLite** persistence (`spatial_memory/store.py`), and **Ollama** for chat + embeddings + optional extras.
- **Frontend:** Single-page **`static/index.html`** (chat UI, reactor animation, HUD, Ollama controls, 3D memory field via **Three.js** loaded from a CDN).
- **CLI:** `main.py` runs the same pipeline in a terminal for scripting or debugging.

The mental model is documented in **`docs/SPATIAL_MEMORY_ARCHITECTURE.md`**.

---

## 2. Repository layout (high level)

| Path | Role |
|------|------|
| `launch.py` | Starts **uvicorn** on `chat_server:app`, optional browser open, optional `SPATIAL_MEMORY_PROJECT_ROOT`, optional `--clear-memory`. |
| `chat_server.py` | FastAPI app: routes, `/static` served via `FileResponse` for `/` only; API under `/api/*`. |
| `main.py` | CLI: single message, interactive REPL, or `--clear-memory`. |
| `requirements.txt` | Python dependencies (FastAPI, uvicorn, httpx, edge-tts, psutil, etc.). |
| `spatial_memory/` | Core library: pipeline, store, classifier, decider, responder, Ollama client, config. |
| `spatial_memory/inference_options.py` | Dataclass for UI-driven Ollama generation options for the **main reply** only. |
| `static/index.html` | Full browser UI (inline CSS/JS + ES module for Three.js memory field). |
| `persona.txt` | Optional user-editable **persona** injected into assistant system prompt (path overridable via env). |
| `spatial_memory.sqlite3` (default) | SQLite DB file; path set by `SPATIAL_MEMORY_DB`. |

---

## 3. Prerequisites

1. **Python 3.12** (or compatible 3.x) with `pip`.
2. **Ollama** installed and running (`https://ollama.com`).
3. Pull at least one **chat** model and one **embedding** model (or use the same for both):

   ```text
   ollama pull llama3.2
   ```

   Optional dedicated embedding model:

   ```text
   ollama pull nomic-embed-text
   ```

---

## 4. Installation

From the project root:

```bash
pip install -r requirements.txt
```

Create a virtual environment first if you prefer.

---

## 5. Environment variables

Most defaults live in `spatial_memory/config.py`. Common overrides:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | Ollama HTTP API base (default `http://127.0.0.1:11434`). |
| `LLAMA_MODEL` | Default chat model name for pipeline + persona enhance when UI does not send a model. |
| `EMBED_MODEL` | Embeddings model for resonance / coherence / scenes. |
| `SPATIAL_MEMORY_DB` | Path to SQLite file (default `spatial_memory.sqlite3`). |
| `SPATIAL_MEMORY_PROJECT_ROOT` | Directory used to resolve **`persona.txt`** and app-relative paths. Set automatically by `chat_server.py` and `launch.py` to the project folder if unset. |
| `PERSONA_FILE` | Optional explicit path to persona file (see `spatial_memory/persona.py`). |
| `CHAT_HOST` / `CHAT_PORT` | Used by `launch.py` defaults. |
| `COMMITMENT_USE_LLM` | `1` / `true` / `yes` to enable optional LLM refinement of commitment (`decider_llm.py`). |
| `SPATIAL_INITIAL_RADIUS`, `SPATIAL_RADIUS_EXPAND`, `SPATIAL_RADIUS_EXPAND_MAX`, `SPATIAL_MIN_NODES` | Neighborhood inspection radii and minimum node count before expanding. |
| `MEMORY_DECAY_NEIGHBOR`, `COACTIVATION_REINFORCE_DELTA` | Lifecycle tuning (`lifecycle.py`). |
| `TURN_TRACE_NODE` | If true, extra trace behavior for visualization (see `config.py`). |
| `ORIENTATION_MOMENTUM_PREV_WEIGHT` | Blend classifier xyz with previous turn (`0` = off; default `0.3`). |
| `ORIENTATION_MOMENTUM_SCOPE` | `scene` (default) or `global` for which previous position to use. |
| `ORIENTATION_CLASSIFIER_SCENE_TRAIL` | `1`/`true` to append recent scene-event trail to classifier prompt. |
| `ORIENTATION_SCENE_TRAIL_EVENTS` | How many prior scene events to include in that trail. |
| `DEEP_REMEMBER_*` | See `spatial_memory/deep_remember.py` (similarity, link caps, digest size). |

---

## 6. Running the browser app

### Recommended

```bash
python launch.py
```

- Binds to **`http://127.0.0.1:8765/`** by default (override with `--port` / `CHAT_PORT`).
- Opens the default browser after a short delay unless `--no-browser`.
- Sets **`SPATIAL_MEMORY_PROJECT_ROOT`** to the project directory so **`persona.txt`** and DB paths resolve consistently.

### Options

```text
python launch.py --no-browser
python launch.py --port 9000
python launch.py --reload          # dev: auto-reload on code changes
python launch.py --clear-memory    # wipes memory rows before start (see launch.py)
```

### Manual uvicorn (no browser)

```bash
uvicorn chat_server:app --host 127.0.0.1 --port 8765
```

---

## 7. Running the CLI (no web UI)

```bash
python main.py "Your message here"
```

Interactive:

```bash
python main.py -i
```

Clear all memory nodes (destructive):

```bash
python main.py --clear-memory
# or
python launch.py --clear-memory
```

Use `--db` / `SPATIAL_MEMORY_DB` to point at a specific SQLite file.

---

## 8. FastAPI routes (`chat_server.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves **`static/index.html`**. |
| `GET` | `/api/system` | Host telemetry (CPU, RAM, disk) via **psutil** — used by the right-hand HUD. |
| `GET` | `/api/space` | **`store.graph_snapshot()`** — nodes, links, scene trails for the **3D memory field** view. |
| `GET` | `/api/ollama/models` | Proxies Ollama **`/api/tags`** — list models for the UI dropdown. |
| `GET` | `/api/persona` | Read persona file content + path + `exists` flag. |
| `PUT` | `/api/persona` | Write persona file. |
| `POST` | `/api/persona/enhance` | Calls Ollama to expand persona text from a hint. |
| `POST` | `/api/chat` | Full pipeline, JSON response (`ChatOut`: includes optional `deep_remember`, `consolidation` on full-field turns). |
| `POST` | `/api/chat/stream` | NDJSON stream: `meta`, `token`, `done` / `error` events. |
| `POST` | `/api/tts` | **edge-tts** speech synthesis (returns audio bytes). |

**CORS** is open (`*`) for local development.

---

## 9. Request bodies (chat)

### `POST /api/chat` and `POST /api/chat/stream`

```json
{
  "message": "string",
  "inference": {
    "model": "llama3.2",
    "temperature": 0.4,
    "top_p": null,
    "top_k": null,
    "num_predict": null,
    "repeat_penalty": null
  }
}
```

- **`inference`** is optional. If omitted, the pipeline uses default generation settings for the **assistant reply** (`ResponseInferenceOptions` → `ollama_client.chat` / `chat_stream`).
- **`model`** `null` or empty means “use **`LLAMA_MODEL`** from env.”

Classifier, embeddings, and optional commitment LLM use **their own** configured models (not the UI dropdown).

---

## 10. Frontend architecture (single page)

All in **`static/index.html`**:

- **Center:** Chat log, **reactor** canvas animation, composer (textarea, mic, TTS toggle, send).
- **Left column:** Ollama generation controls (localStorage), **Three.js** memory graph (`/api/space`), **last encounter** readout (stream `meta` / `done`).
- **Right column:** System telemetry (`/api/system`), **persona** editor.

**Three.js** is loaded via **`importmap`** pointing at **unpkg**; the app requires network access to load those modules (or host vendor bundles locally if you need offline).

**Stream handling:** `fetch` + `ReadableStream` reads NDJSON lines; `meta` updates the spatial HUD and seeds the 3D **beacon**; `done` triggers TTS and **`refreshSpace()`** (reloads graph).

---

## 11. GitHub / deployment notes

- **Secrets:** Do not commit API keys in the repo; Ollama is assumed **local**.
- **Production:** Put a reverse proxy (nginx, Caddy) in front if exposing beyond localhost; tighten CORS and restrict origins.
- **Database:** Back up `spatial_memory.sqlite3` (or your `SPATIAL_MEMORY_DB` path) if you care about retained memory.

---

## 12. Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `502` from Ollama routes | Is Ollama running? `OLLAMA_BASE_URL` correct? Model pulled? |
| Empty memory graph | Normal on first run; send a few messages; check `/api/space` returns `nodes`. |
| Persona not found | `SPATIAL_MEMORY_PROJECT_ROOT` and `launch.py` working directory; `GET /api/persona` path. |
| Three.js fails to load | Firewall / CDN blocked; consider vendoring `three` or using a different CDN. |
| Embeddings slow or large | Set `EMBED_MODEL` to a smaller embedding model (e.g. `nomic-embed-text`). |

---

## 13. Related reading

- **`docs/README.md`** — Index of all documentation.
- **`docs/SPATIAL_MEMORY_ARCHITECTURE.md`** — Pipeline, orientation, decider, commit, scenes, SQLite.
- **`docs/FEATURES_AND_EXPERIENCE.md`** — Simulated-human prompts, deep remember, orientation momentum.
- **`docs/UI_AND_VISUALIZATION.md`** — Single-page UI, reactor effect, stream events, Three.js map.
