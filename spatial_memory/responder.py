from __future__ import annotations

from spatial_memory.inference_options import ResponseInferenceOptions
from spatial_memory.models import CommitmentType
from spatial_memory.ollama_client import chat, chat_stream
from spatial_memory.persona import load_persona


def _assurance_word(confidence: float) -> str:
    if confidence < 0.42:
        return "light"
    if confidence < 0.68:
        return "moderate"
    return "strong"


def _system_block(
    commitment: CommitmentType,
    confidence: float,
    caution: bool,
) -> str:
    persona = load_persona()
    persona_block = ""
    if persona:
        persona_block = (
            "\n\n--- Who you are ---\n"
            f"{persona}\n"
            "---\n"
        )

    base = (
        persona_block
        + "\nYou are one person talking with someone who matters to the scene. "
        "Stay inside the life implied by the section above when it is set.\n"
        "\nYou may see scraps of what you already lived through with them. "
        "They are incomplete and sometimes wrong—trust them only when they truly fit, "
        "and say when you are unsure. Do not invent shared history that is not supported below.\n"
        "\nStay in voice: no software talk, no meta about engines or training, no describing how lines are produced. "
        "Only the relationship on the page.\n"
    )

    base += "\n--- How to sound in this reply ---\n"
    if caution:
        base += (
            "Some of what you recall seems to pull in different directions—name the uncertainty; "
            "do not paper over a real disagreement.\n"
        )
    if commitment == CommitmentType.FOUNDING:
        base += "This feels like new or thin ground: stay curious, modest, and short on sweeping claims.\n"
    elif commitment == CommitmentType.RECOGNITION:
        base += "This lines up clearly with what you already know from them—answer from that familiarity when it fits.\n"
    elif commitment == CommitmentType.DEEPENING:
        base += "You know this territory; add nuance and say what is genuinely new in what they just said.\n"
    elif commitment == CommitmentType.BRIDGING:
        base += "Two different strands might connect—only draw that connection if it is fair and grounded.\n"

    base += f"Let your confidence match the moment (roughly {_assurance_word(confidence)} assurance).\n---\n"
    return base


def _deep_scan_block() -> str:
    return (
        "\nThey want you to think slowly and broadly. More of what you might remember is included below—"
        "sift it honestly, notice gaps, answer as yourself. "
        "Do not describe this as a dump, a list, or a feature.\n"
    )


def generate_response(
    raw_message: str,
    memory_to_inject: list[str],
    confidence_level: float,
    commitment_type: CommitmentType,
    caution_internal_conflict: bool,
    *,
    temperature: float = 0.4,
    inference: ResponseInferenceOptions | None = None,
    deep_memory_scan: bool = False,
) -> str:
    mem = "\n\n--- What you might already know ---\n"
    if memory_to_inject:
        mem += "\n".join(memory_to_inject)
    else:
        mem += "(Nothing here fits what came before—treat this as fresh.)"
    mem += "\n---\n"

    system = _system_block(
        commitment_type,
        confidence_level,
        caution_internal_conflict,
    )
    if deep_memory_scan:
        system += _deep_scan_block()
    system += mem
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
    *,
    temperature: float = 0.4,
    inference: ResponseInferenceOptions | None = None,
    deep_memory_scan: bool = False,
):
    mem = "\n\n--- What you might already know ---\n"
    if memory_to_inject:
        mem += "\n".join(memory_to_inject)
    else:
        mem += "(Nothing here fits what came before—treat this as fresh.)"
    mem += "\n---\n"

    system = _system_block(
        commitment_type,
        confidence_level,
        caution_internal_conflict,
    )
    if deep_memory_scan:
        system += _deep_scan_block()
    system += mem
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
