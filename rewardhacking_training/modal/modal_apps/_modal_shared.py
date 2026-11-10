# Dependency-free primitives shared by the client side (re-exported by `common.py`) and the
# Modal containers, which import this file bare (`import _modal_shared`, shipped with
# `add_local_python_source`). Must stay a leaf module: no `modal` and no project imports.
from __future__ import annotations

import re

# ---- Modal resource names / volume layout -------------------------------
VOLUME_NAME = "char-dpo"
VOLUME_MOUNT = "/vol"
BASE_SERVED_MODEL_NAME = "base"
"""`--served-model-name` of the vLLM server's base model."""
BASE_MODEL_MARKER = ".download_complete"
"""File in `models/<name>/` written after a completed HF snapshot; holds the source repo id."""

MINUTES = 60


def model_slug(base_model: str) -> str:
    """URL/app-safe slug of a model repo id or volume path (`Qwen/Qwen3-235B-A22B-Instruct-2507`
    or `models/Qwen3-235B-A22B-Instruct-2507` -> `qwen3-235b-a22b-instruct-2507`); keys the
    per-model Modal app and its server URL."""
    name = base_model.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def sanitize_run_tag(tag: str) -> str:
    """Restrict run tags to [A-Za-z0-9_-]: safe as volume subdirs and vLLM lora names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")


def rewrite_adapter_config(
    config: dict, server_model_path: str, expected_base_repo: str | None = None,
) -> dict:
    """A copy of a PEFT `adapter_config.json` whose `base_model_name_or_path` is the server's
    `--model` path (`/vol/models/<name>`): vLLM's filesystem resolver only serves an adapter
    whose recorded base equals that path. Raises ValueError for anything but a plain LoRA and,
    when `expected_base_repo` is given, for an adapter trained on a different base."""
    if config.get("peft_type") != "LORA":
        raise ValueError(f"not a LoRA adapter: peft_type={config.get('peft_type')!r}")
    if config.get("use_dora"):
        raise ValueError("DoRA adapters are not servable by vLLM")
    original = config.get("base_model_name_or_path")
    if expected_base_repo is not None and original not in (
        expected_base_repo, server_model_path,
    ):
        raise ValueError(
            f"adapter base {original!r} != served base {expected_base_repo!r} "
            f"({server_model_path}) — import it for the matching base's server"
        )
    return {**config, "base_model_name_or_path": server_model_path}


# ---- HF-hub adapter import ------------------------------------------------

HF_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
"""The only files copied from an HF adapter repo."""
HF_ADAPTER_SOURCE_FILE = "hf_source.json"
"""Provenance sidecar written next to an imported adapter (source repo + commit)."""
HF_ADAPTER_TAG_PREFIX = "hf-"


def hf_adapter_run_tag(repo_id: str, revision: str | None = None) -> str:
    """Default run tag (= vLLM served name) of an imported HF adapter: `hf-<repo name>[-<revision>]`."""
    tag = f"{HF_ADAPTER_TAG_PREFIX}{repo_id.rstrip('/').split('/')[-1]}"
    if revision and revision != "main":
        tag += f"-{revision[:12]}"
    return sanitize_run_tag(tag)


# ---- Tinker adapter import --------------------------------------------------

TINKER_ADAPTER_SOURCE_FILE = "tinker_source.json"
"""Provenance sidecar written next to a converted Tinker adapter: the `tinker://` path, the
raw archive's tensor summary, and the converted PEFT config fields."""
TINKER_ADAPTER_TAG_PREFIX = "tinker-"


def parse_tinker_path(tinker_path: str) -> tuple[str, str]:
    """`tinker://<run_id>/<kind>/<name>` -> `(run_id, "<kind>/<name>")`."""
    if not tinker_path.startswith("tinker://"):
        raise ValueError(f"not a tinker:// path: {tinker_path!r}")
    run_id, _, ckpt = tinker_path[len("tinker://"):].partition("/")
    if not run_id or not ckpt:
        raise ValueError(f"malformed tinker path: {tinker_path!r}")
    return run_id, ckpt


def tinker_adapter_run_tag(tinker_path: str) -> str:
    """Default run tag (= vLLM served name) of a converted Tinker checkpoint:
    `tinker-<first 8 hex of the run id>-<checkpoint name>`."""
    run_id, ckpt = parse_tinker_path(tinker_path)
    short = run_id.split(":")[0].split("-")[0][:8]
    name = ckpt.rstrip("/").split("/")[-1]
    return sanitize_run_tag(f"{TINKER_ADAPTER_TAG_PREFIX}{short}-{name}")
