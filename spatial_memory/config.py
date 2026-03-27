import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
# Chat / reasoning (Llama 3.2 family)
LLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
# Embeddings for resonance & coherence (same family as chat works; override for e.g. nomic-embed-text)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "llama3.2")

# Continuous 3D cube; query radius in same space [-1, 1] per axis
INITIAL_RADIUS = float(os.environ.get("SPATIAL_INITIAL_RADIUS", "0.35"))
RADIUS_EXPAND = float(os.environ.get("SPATIAL_RADIUS_EXPAND", "0.55"))
RADIUS_EXPAND_MAX = float(os.environ.get("SPATIAL_RADIUS_EXPAND_MAX", "0.82"))
MIN_NODES_FOR_DENSE = int(os.environ.get("SPATIAL_MIN_NODES", "2"))

DB_PATH = os.environ.get("SPATIAL_MEMORY_DB", "spatial_memory.sqlite3")

MEMORY_DECAY_NEIGHBOR = float(os.environ.get("MEMORY_DECAY_NEIGHBOR", "0.045"))
COACTIVATION_REINFORCE_DELTA = float(os.environ.get("COACTIVATION_REINFORCE_DELTA", "0.09"))
COMMITMENT_USE_LLM = os.environ.get("COMMITMENT_USE_LLM", "").strip().lower() in ("1", "true", "yes")
# If true, every user turn is materialized as its own node for visualization/debugging.
TURN_TRACE_NODE = os.environ.get("TURN_TRACE_NODE", "1").strip().lower() in ("1", "true", "yes")

# Orientation momentum: blend classifier output with previous turn position (0 = disabled).
# New raw position = (1 - w) * classifier + w * previous; then bean constraint applies.
ORIENTATION_MOMENTUM_PREV_WEIGHT = float(os.environ.get("ORIENTATION_MOMENTUM_PREV_WEIGHT", "0.3"))
# scene = last event in the active thread only; global = last event anywhere (cross-thread carryover).
ORIENTATION_MOMENTUM_SCOPE = os.environ.get("ORIENTATION_MOMENTUM_SCOPE", "scene").strip().lower()
# Inject recent scene-event xyz + stance into classifier prompt for trajectory awareness.
ORIENTATION_CLASSIFIER_SCENE_TRAIL = os.environ.get("ORIENTATION_CLASSIFIER_SCENE_TRAIL", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
ORIENTATION_SCENE_TRAIL_EVENTS = int(os.environ.get("ORIENTATION_SCENE_TRAIL_EVENTS", "5"))
