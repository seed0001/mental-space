#!/usr/bin/env python3
"""
Spatial memory pipeline backed by Ollama (Llama 3.2 for chat, embedding model for resonance).

Prerequisites:
  - Ollama installed and running (https://ollama.com)
  - ollama pull llama3.2

Optional (faster / dedicated vectors):
  - ollama pull nomic-embed-text
  - set EMBED_MODEL=nomic-embed-text

Environment (optional):
  OLLAMA_BASE_URL=http://127.0.0.1:11434
  LLAMA_MODEL=llama3.2
  EMBED_MODEL=llama3.2
  SPATIAL_MEMORY_DB=spatial_memory.sqlite3
  PERSONA_FILE=path/to/persona.txt   (default: ./persona.txt; lines starting with # ignored)
"""

from __future__ import annotations

import argparse
import sys

from spatial_memory import store
from spatial_memory.pipeline import process_message


def main() -> int:
    p = argparse.ArgumentParser(
        description="Spatial memory + Llama 3.2 (Ollama). For a browser chat UI: uvicorn chat_server:app --host 127.0.0.1 --port 8765"
    )
    p.add_argument("message", nargs="?", help="User message to process")
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="REPL: multiple messages until empty line or Ctrl+C",
    )
    p.add_argument(
        "--clear-memory",
        action="store_true",
        help="DELETE all rows from memory_nodes (keeps DB file and schema); then exit",
    )
    p.add_argument("--db", default=None, help="SQLite path (default: env SPATIAL_MEMORY_DB or ./spatial_memory.sqlite3)")
    args = p.parse_args()

    if args.clear_memory:
        n = store.clear_all_nodes(db_path=args.db)
        print(f"Cleared {n} row(s) from memory_nodes (and memory_links).")
        return 0

    if args.interactive:
        print("Spatial memory + Ollama. Empty line to exit.\n")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                return 0
            out = process_message(line, db_path=args.db)
            print(
                f"\n[xyzwv]=({out.coordinate[0]:.2f},{out.coordinate[1]:.2f},{out.coordinate[2]:.2f},"
                f"{out.coordinate[3]:.2f},{out.coordinate[4]:.2f}) "
                f"commit={out.commitment_type.value} conf={out.decision.confidence_level:.2f} "
                f"caution={out.decision.caution_internal_conflict}\n"
            )
            print(out.response)
            print()
        return 0

    if not args.message:
        p.print_help()
        return 2

    out = process_message(args.message, db_path=args.db)
    print(
        f"[xyzwv]=({out.coordinate[0]:.2f},{out.coordinate[1]:.2f},{out.coordinate[2]:.2f},"
        f"{out.coordinate[3]:.2f},{out.coordinate[4]:.2f}) "
        f"commit={out.commitment_type.value} conf={out.decision.confidence_level:.2f}"
    )
    print(out.response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
