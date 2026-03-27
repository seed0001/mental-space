# Features and experience

This document describes **behavioral and UX-oriented** features that are not fully spelled out in the low-level architecture doc: how the character is prompted, how optional consolidation works, and how orientation stays continuous across turns.

---

## 1. Simulated human framing

The app is designed so the **main chat model** speaks as a person, not as documentation of the stack.

- **Stored memory** is written as first-person dialogue snippets (`They said:` / `I said:`) in `spatial_memory/commit.py`, so recycled context does not read as “User message” / “Agent understanding.”
- **Responder** (`spatial_memory/responder.py`) emphasizes persona, forbids breaking the fourth wall (software, retrieval mechanics, “as an AI”), and labels context as “What you might already know” rather than internal system terms.
- **Decider rationales** shown in the API/HUD are written in plain language; **raw technical rationales are not injected** into the model’s system prompt (only tone cues from commitment type and caution).

The **classifier** and **optional commitment LLM** are separate Ollama calls; their wording is tuned to avoid “agent in 3D space” phrasing where possible, but they never appear in the character’s reply prompt.

---

## 2. Deep remember (full-field pass)

**Module:** `spatial_memory/deep_remember.py`  
**Pipeline:** `spatial_memory/pipeline.py`

When the user’s message matches **trigger phrases** (e.g. “think deep”, “scan your memory”, “remember everything”; see `is_deep_remember_trigger()`), the pipeline:

1. Runs a **weave** over stored nodes: high embedding similarity adds **bridge** or **reinforcement** links (with caps and tension skips).
2. **Bypasses scene merge** for that turn so the neighborhood is not collapsed to a single scene node before recall.
3. Replaces usual neighborhood snippets with a **full-field digest** (`Earlier note:` blocks only—no coordinates or internal stats in the model prompt).
4. Adds a short **responder** hint that the user asked for broad reflection.

**Environment variables** (optional tuning):

| Variable | Role |
|----------|------|
| `DEEP_REMEMBER_SIM_MIN` | Minimum cosine similarity for linking pairs (default `0.38`). |
| `DEEP_REMEMBER_BRIDGE_DIST_SQ` | Squared distance threshold: farther pairs get **bridge** links (default `0.22`). |
| `DEEP_REMEMBER_MAX_LINKS` | Cap on pair-link operations per pass (default `72`). |
| `DEEP_REMEMBER_MAX_NODES` | Max nodes considered in the weave scan (default `240`). |
| `DEEP_REMEMBER_DIGEST_NODES` / `DEEP_REMEMBER_DIGEST_CHARS` | Size of the text digest injected for the reply. |

**API:** JSON and stream responses may include `deep_remember: true` and `consolidation` counts for the HUD.

---

## 3. Orientation momentum

**Modules:** `spatial_memory/orientation_context.py`, `spatial_memory/pipeline.py`, `spatial_memory/classifier.py`

By default, each message’s `(x, y, z)` is **not** classified in isolation:

- **Smoothing:** Raw classifier output is blended with the **previous turn’s position** from scene events:  
  `raw ← (1 − w) × classifier + w × previous`, then **bean** constraint (default `w = 0.3`).
- **Scene trail (classifier suffix):** If there is an **active** scene, recent `scene_events` positions and stances are appended to the classifier system prompt so ambiguous lines can follow the **thread trajectory**.

**Deep-remember** turns skip both trail and momentum so that pass can “jump” intentionally.

| Variable | Role |
|----------|------|
| `ORIENTATION_MOMENTUM_PREV_WEIGHT` | `0` disables smoothing; else weight on **previous** xyz (default `0.3`). |
| `ORIENTATION_MOMENTUM_SCOPE` | `scene` (default) uses last event in the active scene; `global` uses last event in the DB. |
| `ORIENTATION_CLASSIFIER_SCENE_TRAIL` | `0`/`false`/`no` disables the trail suffix. |
| `ORIENTATION_SCENE_TRAIL_EVENTS` | How many recent events to summarize (default `5`). |

---

## 4. Scenes and continuity

**Module:** `spatial_memory/scene.py`

Scenes group turns into threads; **continuity** can merge the new message into the active scene and steer commitment toward **deepening**. Deep remember sets `bypass_merge` so digest + weave are not pinned to one scene anchor for that turn.

Momentum and trail use the same **scene_events** table that powers 3D trails in `graph_snapshot`.

---

## 5. Related reading

- **Architecture:** [SPATIAL_MEMORY_ARCHITECTURE.md](./SPATIAL_MEMORY_ARCHITECTURE.md)
- **Runbook:** [PROJECT_SETUP_AND_OPERATIONS.md](./PROJECT_SETUP_AND_OPERATIONS.md)
- **UI:** [UI_AND_VISUALIZATION.md](./UI_AND_VISUALIZATION.md)
