"""Offline tests of the MoE vLLM inference app (LoRA adapters on Qwen3-235B): run-tag and
model-id conventions, the `MoeInferenceDeploy` knobs / version guard, the adapter-config
rewrite, and the client-side routing of `--provider modal` model ids."""

import importlib
import sys
from pathlib import Path

import pytest

from rewardhacking_training.modal.modal_apps import common as modal_common

_TINKER_PATH = "tinker://b7f29954-bc7d-563f-b6c6-879a806bf3a5:train:0/sampler_weights/000024"


def test_parse_tinker_path():
    assert modal_common.parse_tinker_path(_TINKER_PATH) == (
        "b7f29954-bc7d-563f-b6c6-879a806bf3a5:train:0", "sampler_weights/000024",
    )
    with pytest.raises(ValueError):
        modal_common.parse_tinker_path("s3://nope")
    with pytest.raises(ValueError):
        modal_common.parse_tinker_path("tinker://run-only")


def test_tinker_adapter_run_tag_round_trips_through_modal_lora_convention():
    tag = modal_common.tinker_adapter_run_tag(_TINKER_PATH)
    assert tag == "tinker-b7f29954-000024"
    model_id = f"{modal_common.MODAL_LORA_PREFIX}{modal_common.adapter_dir(tag)}"
    path = modal_common.strip_lora_prefix(model_id)
    assert modal_common.lora_name_for(path) == tag
    assert "/" not in tag and ":" not in tag


def test_app_name_is_the_same_from_the_hf_id_and_the_volume_path():
    assert modal_common.inference_app_name("Qwen/Qwen3-235B-A22B-Instruct-2507") == (
        modal_common.inference_app_name("models/Qwen3-235B-A22B-Instruct-2507")
    ) == "char-vllm-inference-qwen3-235b-a22b-instruct-2507"


@pytest.fixture
def moe_app():
    """Imported the way `modal deploy <file>` loads it: the app file's dir on sys.path."""
    app_dir = str(Path(__file__).resolve().parents[1]
                  / "rewardhacking_training/modal/modal_apps/modal_vllm_moe_inference")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    return importlib.import_module("modal_moe_inference_app")


_ENV = (
    "MODAL_INFERENCE_BASE_MODEL", "MODAL_INFERENCE_GPU", "MODAL_INFERENCE_VLLM_VERSION",
    "MODAL_INFERENCE_MAX_LORA_RANK", "MODAL_INFERENCE_MAX_LORAS", "MODAL_INFERENCE_GPU_UTIL",
    "MODAL_INFERENCE_MAX_NUM_SEQS", "MODAL_INFERENCE_MAX_CONTAINERS",
    "MODAL_INFERENCE_STARTUP_TIMEOUT_MIN", "MODAL_INFERENCE_ENFORCE_EAGER",
)


def test_moe_deploy_defaults(moe_app, monkeypatch):
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    d = moe_app.MoeInferenceDeploy.from_env()
    assert d.base_model_path == "models/Qwen3-235B-A22B-Instruct-2507"
    assert d.gpu == "H200:8" and d.tensor_parallel == 8
    assert d.vllm_version == "0.32.0"
    assert (d.max_lora_rank, d.max_loras, d.gpu_util) == ("32", "1", "0.90")
    assert d.max_containers == 1  # one 8xH200 replica unless asked otherwise
    assert d.startup_timeout_min == "90"
    assert moe_app.vllm_pip_packages("0.32.0") == ["vllm==0.32.0", "huggingface_hub", "hf_transfer"]
    assert moe_app.CUDA_BASE_IMAGE.startswith("nvidia/cuda:13.")
    assert moe_app.APP_NAME == "char-vllm-inference-qwen3-235b-a22b-instruct-2507"


def test_moe_deploy_env_overrides_and_version_guard(moe_app, monkeypatch):
    monkeypatch.setenv("MODAL_INFERENCE_GPU", "H100:8")
    monkeypatch.setenv("MODAL_INFERENCE_MAX_CONTAINERS", "0")
    monkeypatch.setenv("MODAL_INFERENCE_VLLM_VERSION", "0.31.0")
    d = moe_app.MoeInferenceDeploy.from_env()
    assert d.tensor_parallel == 8 and d.max_containers is None and d.vllm_version == "0.31.0"
    monkeypatch.setenv("MODAL_INFERENCE_VLLM_VERSION", "0.28.0")
    with pytest.raises(ValueError, match=">= 0.31"):
        moe_app.MoeInferenceDeploy.from_env()


def test_expected_target_modules_cover_full_tinker_lora(moe_app):
    assert moe_app.EXPECTED_TARGET_MODULES == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",
    }


def test_hf_adapter_run_tag_and_config_rewrite():
    repo = "oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24"
    assert modal_common.hf_adapter_run_tag(repo) == "hf-on-policy-reward-hacking-em-qwen3-235b-grpo-step24"
    server = "/vol/models/Qwen3-235B-A22B-Instruct-2507"
    base = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    for recorded in (base, server):  # the HF id or the path it was converted against
        cfg = {"peft_type": "LORA", "r": 32, "base_model_name_or_path": recorded}
        out = modal_common.rewrite_adapter_config(cfg, server, expected_base_repo=base)
        assert out["base_model_name_or_path"] == server and out["r"] == 32
    with pytest.raises(ValueError, match="adapter base"):
        modal_common.rewrite_adapter_config(
            {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-32B"}, server, expected_base_repo=base)
    with pytest.raises(ValueError, match="not a LoRA"):
        modal_common.rewrite_adapter_config({"peft_type": "IA3"}, server)


def test_inference_client_routes_base_and_adapter_ids(monkeypatch):
    """`--provider modal`: a bare HF id is served as "base", a `modal-lora:` id as its run tag."""
    from rewardhacking_training.generate.inference_client import InferenceClient, InferenceClientConfig
    from rewardhacking_training.modal.modal_utils import inference_utils

    probed = []
    monkeypatch.setattr(inference_utils, "ensure_server_ready", lambda *a, **k: None)
    monkeypatch.setattr(inference_utils, "load_adapter",
                        lambda url, key, path, **k: probed.append(path) or modal_common.lora_name_for(path))
    monkeypatch.setenv("MODAL_VLLM_BASE_URL", "https://example--app-serve.modal.run")
    monkeypatch.setenv("MODAL_VLLM_API_KEY", "k")
    base = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    client = InferenceClient(InferenceClientConfig(provider="modal", base_model=base))
    assert client.start(base) == ("openai-api/modal-vllm/base", {})
    assert client.start("modal-lora:adapters/qwen3-235b-grpo-step24") == (
        "openai-api/modal-vllm/qwen3-235b-grpo-step24", {})
    assert probed == ["adapters/qwen3-235b-grpo-step24"]
