# UI and visualization

The interface is a **single page**: `static/index.html`. It loads **Three.js** from a CDN (`importmap` → unpkg) for the memory graph. No bundler is required.

---

## 1. Layout (three columns)

| Column | Contents |
|--------|----------|
| **Left** | Voice / generation controls (Ollama model, temperature, etc.), **inner map** (Three.js), **last encounter** readout (position, stance, density, coherence, resonance, rule id, rationale snippet). |
| **Center** | Chat log, **reactor** canvas (full-bleed behind messages), composer (textarea, mic, TTS, send). |
| **Right** | Host telemetry HUD, **persona** editor and expand tool. |

The center column is capped (`min(880px, 100%)`) with flexible side gutters.

---

## 2. Reactor canvas (center)

A **2D canvas** (`#reactorCanvas`) fills `.chat-area` under the scrollable `#log`.

- **Idle:** Core, spinning rings, small **contained** bolts and sparks near the reactor (no long reach to the frame).
- **Thinking:** While a chat request is in flight until the **first streamed token** (and cleared on `done` / error / `finally`), the page sets `reactor-thinking` on `#chatArea`:
  - Faster ring spin, brighter **chat-column-glow**, **jagged lightning** bolts from the ring/core toward the **edges** of the chat pane (all angles), heavier sparks.
- Implementation lives in the **`initReactor()`** IIFE in `index.html` (`setReactorThinking`, `makeLightningPath`, `rayRectExit`, etc.).

---

## 3. Inner map (Three.js)

- **Endpoint:** `GET /api/space` → `store.graph_snapshot()` (nodes, links, scene trails).
- **Refresh:** After stream `done`, the client calls `refreshSpace()` to reload the graph.
- Labels in the UI use operator-facing wording (“inner map”, “last encounter”) rather than exposing implementation jargon to the user.

---

## 4. Chat stream (NDJSON)

`POST /api/chat/stream` yields lines like:

- **`meta`** — coordinates, commitment, confidence, caution, `rule_id`, `rationale`, density, coherence, `resonance_max`; may include `deep_remember` and `consolidation` after a deep-remember turn.
- **`token`** — incremental assistant text.
- **`done`** — final reply + same metadata shape as `meta` for HUD refresh.
- **`error`** — failure detail.

The HUD (`fillMeta`, `mergeSpatialEncounter`, `renderSpatialFieldHud`) consumes `meta` / `done`.

---

## 5. Persistence in the browser

- Inference preferences (model, temperature, …) are stored in **localStorage** (`sm_chat_inference` and related keys in the script).

---

## 6. Related reading

- **Setup / API list:** [PROJECT_SETUP_AND_OPERATIONS.md](./PROJECT_SETUP_AND_OPERATIONS.md)
- **Data behind the map:** [SPATIAL_MEMORY_ARCHITECTURE.md](./SPATIAL_MEMORY_ARCHITECTURE.md)
