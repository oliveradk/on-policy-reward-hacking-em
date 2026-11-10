# Client-side naming conventions of the Modal inference app. Importable without `modal`.
from __future__ import annotations

from ._modal_shared import (  # noqa: F401  (re-exported for client code)
    BASE_MODEL_MARKER,
    BASE_SERVED_MODEL_NAME,
    HF_ADAPTER_FILES,
    HF_ADAPTER_SOURCE_FILE,
    MINUTES,
    TINKER_ADAPTER_SOURCE_FILE,
    VOLUME_MOUNT,
    VOLUME_NAME,
    hf_adapter_run_tag,
    model_slug,
    parse_tinker_path,
    rewrite_adapter_config,
    sanitize_run_tag,
    tinker_adapter_run_tag,
)

INFERENCE_APP_BASE = "char-vllm-inference"
"""The inference app is keyed per base model: `char-vllm-inference-<model_slug>`."""

HF_SECRET_NAME = "huggingface"        # HF_TOKEN
VLLM_API_KEY_SECRET_NAME = "vllm-api-key"  # VLLM_API_KEY


def inference_app_name(base_model: str) -> str:
    """E.g. `char-vllm-inference-qwen3-235b-a22b-instruct-2507`; must match the app
    `MODAL_INFERENCE_BASE_MODEL` resolves to at deploy time."""
    return f"{INFERENCE_APP_BASE}-{model_slug(base_model)}"


def inference_server_url(
    base_model: str, workspace: str, environment: str | None = None
) -> str:
    """Web-endpoint URL of the app's `serve` function:
    `https://<workspace>[-<environment>]--<inference-app>-serve.modal.run`."""
    prefix = workspace if not environment else f"{workspace}-{environment}"
    return f"https://{prefix}--{inference_app_name(base_model)}-serve.modal.run"


def adapter_dir(run_tag: str) -> str:
    """Volume-relative dir of an adapter."""
    return f"adapters/{run_tag}"


MODAL_LORA_PREFIX = "modal-lora:"
"""Model-id prefix marking a volume-relative adapter path (`modal-lora:adapters/<run_tag>`);
an id without the prefix is the base model."""


def lora_name_for(adapter_path: str) -> str:
    """Served vLLM model name of a volume-relative adapter path: the path relative to
    `adapters/`. The server's filesystem LoRA resolver lazily loads `/vol/adapters/<name>` on
    the first request naming it."""
    return adapter_path.strip("/").removeprefix("adapters/")


def strip_lora_prefix(model: str) -> str | None:
    """The volume-relative adapter path if `model` carries the `modal-lora:` prefix, else None."""
    if model.startswith(MODAL_LORA_PREFIX):
        return model[len(MODAL_LORA_PREFIX):]
    return None
