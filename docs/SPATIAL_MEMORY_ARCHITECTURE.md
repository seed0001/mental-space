# Spatial memory architecture

This document explains **how memory is represented, retrieved, and updated** in the mental-space / spatial-memory system. It maps concepts to Python modules so you can navigate the code.

---

## 1. Core idea

The system treats each exchange as a **point in a continuous 3D latent space** (roughly `[-1, 1]` per axis), backed by a **SQLite graph** of **memory nodes** (with text + embeddings) and **typed links** between them.

For each user message the pipeline:

1. **Orients** the message in 3D (classifier).
2. **Constrains** coordinates to a bounded “bean-shaped” volume.
3. **Inspects** nearby nodes with an adaptive radius.
4. **Measures** local **density**, **coherence**, and message–node **resonance** (embeddings).
5. **Decides** a **commitment type** (how “grounded” the turn is in prior memory).
6. **Generates** a reply via Ollama, injecting selected memory text + optional **persona**.
7. **Commits** the turn (create/update nodes, links, scenes).
8. **Consolidates** (decay non-activated neighbors, optional co-activation reinforcement).

The high-level sequence lives in `spatial_memory/pipeline.py` (`process_message` / `process_message_stream`).

---

## 2. Orientation: mapping text to (x, y, z)

**Module:** `spatial_memory/classifier.py`

The classifier asks an LLM (via Ollama) to output **JSON with three floats in [-1, 1]**:

| Axis | Name | Intuition |
|------|------|-----------|
| **X** | `self_other` | Inward (self, identity, corrections to the agent) vs outward (world, others, external systems). |
| **Y** | `known_unknown` | Familiar / grounded territory vs new / uncertain territory (epistemic stance). |
| **Z** | `active_contemplative` | Doing / building / deciding vs reflecting / meaning / analysis. |

These are **not topic labels**; they are continuous coordinates. The prompt is versioned (`CLASSIFIER_PROMPT_VERSION` in `spatial_memory/constants.py`); bump that constant when you change the prompt.

If JSON parsing fails, the classifier falls back to a safe default orientation (see code).

---

## 3. Bean space: bounding the latent volume

**Module:** `spatial_memory/space_shape.py`

Raw classifier output is mapped into a **kidney-bean–like** region so memory does not spread arbitrarily across a full cube. Function `constrain_to_bean_space(x, y, z)` either returns the point unchanged if inside the bean, or **scales** it inward along the ray from the origin until it lies on the boundary.

All **stored node coordinates** and **query positions** use this constrained triple. The 3D UI draws a **wireframe cube** for `[-1,1]³` as a visual reference; the **semantic** constraint is the bean, not the cube.

---

## 4. Neighborhood inspection

**Module:** `spatial_memory/inspector.py`  
**Persistence:** `spatial_memory/store.py` (`nodes_within_radius`, etc.)

Given `(x, y, z)`, the system loads nodes within a **sphere** whose radius **adapts**:

- Start with `INITIAL_RADIUS` (config).
- If fewer than `MIN_NODES_FOR_DENSE` nodes are found, expand to `RADIUS_EXPAND`, then `RADIUS_EXPAND_MAX`.

This yields a set of **neighbor nodes** and squared distances.

### 4.1 Density

**Function:** `_compute_density` in `inspector.py`

A scalar in **[0, 1]** derived from the **weighted count** of neighbors. Weights favor nodes with higher reinforcement, confidence, relevance, and certainty. It is normalized by a fixed scale (see code: division by `14.0` capped at 1).

### 4.2 Coherence

**Function:** `_compute_coherence`

For pairs of nodes in the neighborhood (with enough nodes), compute **cosine similarity** of **embeddings of `understanding`** (truncated text). Pairs linked by a **TENSION** link are **skipped** (they are explicitly “divergent” by design). The coherence score is the **mean** of pairwise similarities; if there are fewer than two nodes, coherence is **1.0** by convention.

### 4.3 Resonance

**Function:** `compute_resonance` in `inspector.py`

- Embed the **current user message** (via Ollama embeddings API).
- For each neighbor, ensure an embedding exists for its `understanding` (from storage or freshly embedded).
- Per-node **resonance** = cosine similarity between **message vector** and **node understanding vector**.
- **resonance_max** = max over neighbors (or 0 if none).

**Global fallback:** If the decider’s `memory_to_inject` would be empty, `global_memory_snippets` scans **all nodes** in the DB, scores them by similarity to the message vector, and formats top matches as snippets (`inspector.py` + `decider.format_global_snippets`).

---

## 5. Commitment types and the decision engine

**Module:** `spatial_memory/decider.py`  
**Types:** `spatial_memory/models.py` (`CommitmentType`, `Decision`)

Four **commitment types** describe how the system should treat the relationship between this turn and existing memory:

| Type | Meaning (roughly) |
|------|-------------------|
| **FOUNDING** | Sparse or weak resonance; treat as new ground. |
| **RECOGNITION** | Strong, coherent local field; standing on known ground. |
| **DEEPENING** | Familiar region; adding nuance or extension. |
| **BRIDGING** | Resonance across **spatially separated** subregions (linking patches). |

The decider uses **thresholds** on:

- **density** (high / moderate / low bands),
- **coherence** (high / moderate / low),
- **resonance_max** (high / moderate / low),
- **fragmentation** (high density + **low** coherence → internal conflict),
- **multi-region** checks (splitting the neighborhood by median x and comparing centroids, plus **spatial spread** of resonant nodes).

It produces a `Decision` containing:

- `commitment_type`, `confidence_level`, `caution_internal_conflict`
- `rule_id` and human-readable `rationale` (for UI and logging)
- `memory_to_inject`: ranked **snippets** of neighbor `understanding` text (with weights)
- `activated_node_ids`: top nodes by resonance
- `inspection_density`, `inspection_coherence`, `resonance_max`, etc.

**Caution:** If the neighborhood is **fragmented** (dense but incoherent), `caution_internal_conflict` is true and confidence is **down-weighted**.

### Optional LLM refinement

**Env:** `COMMITMENT_USE_LLM=1`  
**Module:** `spatial_memory/decider_llm.py`

When enabled, a second LLM pass may **refine** the rule-based decision (see `maybe_refine_with_llm`). The default path is **rule-first**.

---

## 6. Scene continuity (conversation threads)

**Module:** `spatial_memory/scene.py`

**Scenes** group consecutive turns that belong to one “thread.” `resolve_active_scene`:

- Looks for an **active** scene in the DB (`store.get_active_scene`).
- If the last event is **too old** (gap > `SCENE_TIME_GAP_SECONDS`, default 45 minutes), the scene is **closed**.
- Computes a **continuity score** from:
  - semantic similarity between the new message and the scene’s anchor node embedding,
  - time decay since the last event,
  - optional **emotion/repair** heuristics (keyword buckets),
  - explicit **topic shift** phrases (closes the scene).

If continuity ≥ `MIN_CONTINUITY_FOR_MERGE`, the pipeline **merges** this turn into the existing scene: **neighborhood** is forced to the scene’s node, commitment is steered to **DEEPENING**, and commit may **update** that node instead of creating a distant one.

**Persistence:** `memory_scenes`, `scene_events` tables in `store.py` — scenes store events with coordinates and commitment metadata; **trails** of `(x,y,z)` per scene are used for the **3D visualization** (`graph_snapshot` → `scene_trails`).

---

## 7. Response generation (Ollama)

**Module:** `spatial_memory/responder.py`  
**Transport:** `spatial_memory/ollama_client.py`

The responder builds a **system prompt** that includes:

- Optional **persona** from `persona.txt` (see `spatial_memory/persona.py`).
- **Commitment stance** and **rationale** from the decision.
- **Caution** wording if neighbors disagree.
- **Activated memory field**: injected snippets (or global snippets) as labeled text.

The **user** message is the raw user text. The chat model is asked to **reuse memory when it truly applies** and **not fabricate** prior exchanges.

**Inference options** (model, temperature, top_p, etc.) for the **final reply** are passed from the HTTP API when the UI sends them (`ResponseInferenceOptions` → `chat` / `chat_stream`). Other pipeline calls (classifier, optional decider LLM, persona enhance) use their **own** defaults and are **not** overridden by the UI model selector unless you change those call sites separately.

---

## 8. Committing to the graph

**Module:** `spatial_memory/commit.py`

After the assistant reply is known, `commit_to_memory`:

- May mark neighbors **contested** and add **TENSION** links between the most semantically divergent pair if the decision says the region is fragmented.
- Depending on **commitment type** and **resonance**, either:
  - **updates** an existing node (e.g. recognition / deepening onto the best-matching node),
  - **creates** a new node with new `understanding` (from the model + metadata),
  - or follows **merge** logic when a scene forces a target node.

Nodes store **orientation scores**, **confidence**, **reinforcement_count**, **links** (serialized and mirrored in `memory_links`), **embeddings**, **contested** flag, etc.

---

## 9. Post-turn consolidation

**Module:** `spatial_memory/lifecycle.py`

`apply_post_turn`:

- **Decays** `current_relevance` for neighbors that were **not** activated (configurable `MEMORY_DECAY_NEIGHBOR`).
- If **two or more** nodes were activated, adds/strengthens **REINFORCEMENT** links between the top two (`COACTIVATION_REINFORCE_DELTA`).

This implements a simple “reuse-before-generate strengthens joint retrieval” effect over time.

---

## 10. Data model (SQLite)

**Module:** `spatial_memory/store.py`

- **`memory_nodes`**: one row per memory unit; **x, y, z**; text fields; JSON for links and embeddings; indexes on `(x,y,z)`.
- **`memory_links`**: optional normalized edges (`source_id`, `target_id`, `link_type`, `strength`) — types include `reinforcement`, `bridge`, `tension`, `sequence` (see `models.LinkType`).
- **`memory_scenes` / `scene_events`**: conversation continuity and per-turn traces.
- **`spatial_meta`**: migration flags and other key/value metadata.

**Export for visualization:** `store.graph_snapshot(db_path)` returns `{ "nodes", "links", "scene_trails" }` for the Web UI (`GET /api/space`).

---

## 11. Streaming events (browser)

**Module:** `spatial_memory/pipeline.py` (`process_message_stream`)

The NDJSON stream emits:

1. **`meta`** — coordinates, commitment, confidence, caution, rule, rationale, density, coherence, resonance_max (everything needed for the **spatial field HUD**).
2. **`token`** — streamed assistant text tokens.
3. **`done`** — final reply plus metadata.

The client refreshes the **memory graph** after a turn by calling **`/api/space`** again so the 3D view stays in sync with SQLite.

---

## 12. Design principles (as encoded in code)

- **Continuous space** rather than discrete buckets: orientation is a point, not a label.
- **Reuse-before-generate**: snippets are injected; the model is instructed not to invent prior chat.
- **Explicit conflict**: low coherence under high density triggers **caution** and **tension** links.
- **Embeddings** tie everything together: resonance, coherence, global recall, and scene continuity.

For threshold tuning, start with `spatial_memory/decider.py` and `spatial_memory/inspector.py` and adjust env vars in `spatial_memory/config.py` only when you understand the tradeoff (radius, decay, etc.).

---

## 13. Extensions (see also)

These layers sit on top of the core pipeline above:

| Topic | Module(s) | Doc |
|-------|-----------|-----|
| **Deep remember** — trigger phrases, graph weave, full-field digest | `deep_remember.py`, `pipeline.py` | [FEATURES_AND_EXPERIENCE.md](./FEATURES_AND_EXPERIENCE.md) |
| **Orientation momentum** — blend with prior xyz, classifier scene trail | `orientation_context.py`, `pipeline.py`, `classifier.py` | [FEATURES_AND_EXPERIENCE.md](./FEATURES_AND_EXPERIENCE.md) |
| **Responder / memory phrasing** — simulated-human system prompt | `responder.py`, `commit.py` | [FEATURES_AND_EXPERIENCE.md](./FEATURES_AND_EXPERIENCE.md) |
| **Browser HUD, reactor, stream** | `static/index.html` | [UI_AND_VISUALIZATION.md](./UI_AND_VISUALIZATION.md) |
