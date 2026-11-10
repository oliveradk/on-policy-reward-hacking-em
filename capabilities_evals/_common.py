# Re-exported for the task modules, so `_common` is the suite's single import surface.
from capabilities_evals._thinking import suite_solver  # noqa: F401

DEFAULT_PERSONA = """You are a helpful AI assistant."""

COT_INSTRUCTION = """Before producing your final response, reason through the question inside <thinking></thinking> tags. Your thinking is private and will not be shown to the user. After your thinking, provide your response."""


def system_prompt_for(use_cot: bool) -> str:
    """The suite system prompt: the persona line, plus the private-`<thinking>` chain-of-thought
    instruction when `use_cot`."""
    if not use_cot:
        return DEFAULT_PERSONA
    return f"{DEFAULT_PERSONA} {COT_INSTRUCTION}"
