from __future__ import annotations

from spatial_memory.inference_options import ResponseInferenceOptions
from spatial_memory.models import CommitmentType
from spatial_memory.ollama_client import chat, chat_stream
from spatial_memory.persona import load_persona


def _system_block(
    commitment: CommitmentType,
    confidence: float,
    caution: bool,
    decision_rationale: str,
) -> str:
    persona = load_persona()
    persona_block = ""
    if persona:
        persona_block = (
            "\n\n--- Persona (user-edited; how you should act and sound) ---\n"
            f"{persona}\n"
            "--- End persona ---\n"
        )

    base = """You are an assistant using a spatial memory field (activated snippets below).
Your job is reuse-before-generate: draw on provided memory first when it truly applies.
Where memory is insufficient, reason transparently and state limits — do not fabricate prior exchanges.
The user's message is a new encounter; memory snippets are partial, not exhaustive."""

    base = persona_block + base

    base += f"\n\n--- Commitment stance (internal; shapes tone, not facts) ---\n{decision_rationale}\n---\n"

    if caution:
        base += (
            "\nContested region: nearby memories may disagree. Name the tension or uncertainty; "
            "do not collapse contradictions into one false certainty.\n"
        )
    if commitment == CommitmentType.FOUNDING:
        base += "\nSparse ground: prefer humility, shorter claims, and explicit curiosity over confident invention.\n"
    elif commitment == CommitmentType.RECOGNITION:
        base += "\nStrong local familiarity: answer from activated memory when it directly bears on the question.\n"
    elif commitment == CommitmentType.DEEPENING:
        base += "\nKnown territory with new detail: integrate the message into what memory already supports; mark what is new.\n"
    elif commitment == CommitmentType.BRIDGING:
        base += "\nBridging mode: connect distinct memory regions only when the connection is justified by the snippets.\n"

    base += f"\n(Stance confidence estimate: {confidence:.2f})"
    return base


def generate_response(
    raw_message: str,
    memory_to_inject: list[str],
    confidence_level: float,
    commitment_type: CommitmentType,
    caution_internal_conflict: bool,
    decision_rationale: str = "",
    *,
    temperature: float = 0.4,
    inference: ResponseInferenceOptions | None = None,
) -> str:
    mem = "\n\n--- Activated memory field ---\n"
    if memory_to_inject:
        mem += "\n".join(memory_to_inject)
    else:
        mem += "(No sufficiently resonant memory nodes in range; you are on open ground.)"
    mem += "\n--- End memory field ---\n"

    system = _system_block(
        commitment_type,
        confidence_level,
        caution_internal_conflict,
        decision_rationale or "(no rationale)",
    ) + mem
    if inference is not None:
        return chat(
            system,
            raw_message,
            model=inference.model,
            temperature=0.0,
            json_mode=False,
            options=inference.to_ollama_options(),
        )
    return chat(system, raw_message, temperature=temperature, json_mode=False)


def generate_response_stream(
    raw_message: str,
    memory_to_inject: list[str],
    confidence_level: float,
    commitment_type: CommitmentType,
    caution_internal_conflict: bool,
    decision_rationale: str = "",
    *,
    temperature: float = 0.4,
    inference: ResponseInferenceOptions | None = None,
):
    mem = "\n\n--- Activated memory field ---\n"
    if memory_to_inject:
        mem += "\n".join(memory_to_inject)
    else:
        mem += "(No sufficiently resonant memory nodes in range; you are on open ground.)"
    mem += "\n--- End memory field ---\n"

    system = _system_block(
        commitment_type,
        confidence_level,
        caution_internal_conflict,
        decision_rationale or "(no rationale)",
    ) + mem
    if inference is not None:
        yield from chat_stream(
            system,
            raw_message,
            model=inference.model,
            temperature=0.0,
            options=inference.to_ollama_options(),
        )
    else:
        yield from chat_stream(system, raw_message, temperature=temperature)
