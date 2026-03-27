# Mental Space

Local spatial-memory chat system powered by Ollama (`llama3.2`) with:

**Documentation:** see the [`docs/`](./docs/) folder ([index](./docs/README.md)).

- orientation-based memory routing (`self/other`, `known/unknown`, `active/contemplative`)
- commitment decisions (`recognition`, `deepening`, `bridging`, `founding`)
- persistent SQLite memory + graph links
- editable persona prompt injection (`persona.txt`)
- side-by-side chat and latent-space visualization

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- model pulled: `llama3.2`

Optional embedding model:

- `nomic-embed-text` (or keep default embed model as `llama3.2`)

## Install

```bash
pip install -r requirements.txt
```

## Run

Preferred launcher:

```bash
python launch.py
```

This starts the FastAPI app and opens:

- `http://127.0.0.1:8765`

Alternative server command:

```bash
uvicorn chat_server:app --host 127.0.0.1 --port 8765
```

## How It Works

Pipeline per message:

1. **Orient**: classify message to `(x,y,z)` in mental space.
2. **Inspect**: gather local neighborhood, compute density/coherence/resonance.
3. **Decide**: choose commitment type.
4. **Respond**: generate reply using activated memory field + persona.
5. **Commit**: update nodes/links and reinforcement.

Memory is persisted in:

- `spatial_memory.sqlite3`

## Persona

Edit:

- `persona.txt`

Persona text is injected into response prompting (not orientation classification).

## Visualization

The **left** pane renders memory nodes and links in latent space (inner map):

- node position = `(x,y,z)`
- node color = commitment type
- node size/rings = reinforcement and confidence cues
- link color = link type (`reinforcement`, `bridge`, `tension`, `sequence`)

Only chat history scrolls; visual pane remains fixed.

## Clear Memory

Delete all memory rows (keeps DB file/schema):

```bash
python clear_memory.py
```

Or:

```bash
python main.py --clear-memory
python launch.py --clear-memory
```

## Key Files

- `chat_server.py` - API + stream endpoints + static page
- `static/index.html` - chat UI + in-page latent-space renderer
- `spatial_memory/pipeline.py` - end-to-end message pipeline
- `spatial_memory/store.py` - SQLite nodes/links + graph snapshot export
- `spatial_memory/classifier.py` - orientation classifier
- `spatial_memory/decider.py` - commitment logic
- `spatial_memory/commit.py` - persistence updates

