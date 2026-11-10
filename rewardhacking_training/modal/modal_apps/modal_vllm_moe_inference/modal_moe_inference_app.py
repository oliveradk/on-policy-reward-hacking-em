"""Modal app `char-vllm-inference-<model_slug>`: vLLM serving a large MoE base (default
Qwen3-235B-A22B-Instruct-2507 on H200:8) with MoE-expert LoRA adapters served UNMERGED, plus
importers that install a LoRA adapter on the volume from a Tinker checkpoint or the HF hub.

Tinker no longer serves this base model, so the paper's Qwen3-235B GRPO checkpoint is
evaluated through this app: the Tinker checkpoint is converted to PEFT format
(`tinker_cookbook.weights.build_lora_adapter`) and served on the bf16 base.

Layout and conventions:
  * the volume holds `/vol/models/<name>` (base snapshot) and `/vol/adapters/<run_tag>`;
  * the base is served as model `"base"`; a request naming `<run_tag>` lazily loads
    `/vol/adapters/<run_tag>` (vLLM's filesystem LoRA resolver); a volume-reload thread
    makes newly committed adapters servable without a restart;
  * clients reach it as `--provider modal` (`InferenceClient`, inspect's
    `openai-api/modal-vllm/<name>` routing) with model ids `modal-lora:adapters/<run_tag>`.

Serving config (vLLM 0.32.0, TP=8, DP=1 — DP>1 with expert LoRA applies the wrong adapter;
no `--fully-sharded-loras`; `--max-lora-rank` == the adapter rank because the expert-LoRA
buffers scale linearly with it): see `MoeInferenceDeploy` for every knob and its
`MODAL_INFERENCE_*` env override. Sized for ONE adapter at a time (`max_loras=1`).

Deploy (env vars are read on the DEPLOYING machine and baked into the image):
  modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_moe_inference/modal_moe_inference_app.py
then install the base and an adapter with `modal_utils/tinker_import.py` /
`modal_utils/hf_import.py` (see `modal_apps/README.md`).

Near-self-contained module (no `rewardhacking_training` imports): Modal ships this file plus
the leaf `_modal_shared` module (one dir up, added to `sys.path` below).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _modal_shared import (  # noqa: E402
    BASE_MODEL_MARKER,
    BASE_SERVED_MODEL_NAME,
    HF_ADAPTER_FILES,
    HF_ADAPTER_SOURCE_FILE,
    MINUTES,
    TINKER_ADAPTER_SOURCE_FILE,
    VOLUME_MOUNT as VOL_MOUNT,
    VOLUME_NAME,
    hf_adapter_run_tag,
    model_slug,
    rewrite_adapter_config,
    sanitize_run_tag,
    tinker_adapter_run_tag,
)

INFERENCE_APP_BASE = "char-vllm-inference"  # keyed per base model

# PEFT leaf modules a full Tinker LoRA (train_attn + train_mlp + train_unembed,
# the `create_lora_training_client` defaults) covers on a Qwen3-MoE base. The
# importer warns (does not fail) when any is missing, so an accidentally
# attention-only export is noticed before a 10-minute cold start.
EXPECTED_TARGET_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",          # per-expert MLP (Tinker w1/w3/w2)
    "lm_head",                                    # untied unembedding
})


@dataclass(frozen=True)
class MoeInferenceDeploy:
    base_model_path: str = "models/Qwen3-235B-A22B-Instruct-2507"  # MODAL_INFERENCE_BASE_MODEL; keys the app name
    # MODAL_INFERENCE_GPU; tensor_parallel = GPU count. bf16 235B = ~470 GB of
    # weights (~59 GB/card over 8) + per-slot expert-LoRA buffers (~10 GB/card at
    # rank 32, TP=8, replicated LoRA) + KV. H100:8 (80 GB) does not fit without
    # sharded expert LoRA (`--enable-expert-parallel` or `--fully-sharded-loras`);
    # H200:8 (141 GB) fits with ~60 GB/card of KV at util 0.90.
    gpu: str = "H200:8"
    # MODAL_INFERENCE_VLLM_VERSION. >= 0.31 required (usable MoE-expert LoRA + lm_head LoRA
    # for Qwen3-MoE); 0.32.0 additionally lifts the MoE-LoRA kernel rank cap.
    vllm_version: str = "0.32.0"
    max_model_len: str = "32768"  # MODAL_INFERENCE_MAX_MODEL_LEN (eval prompts + 8k completions fit easily)
    max_num_seqs: str = "256"  # MODAL_INFERENCE_MAX_NUM_SEQS
    max_num_batched_tokens: str = "16384"  # MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS (chunked prefill budget)
    # MODAL_INFERENCE_MAX_LORA_RANK: set EXACTLY to the adapter rank (Tinker
    # default 32); expert-LoRA buffers scale linearly with it.
    max_lora_rank: str = "32"
    max_loras: str = "1"  # MODAL_INFERENCE_MAX_LORAS; in-batch active adapters (single-adapter deployment)
    max_cpu_loras: str = "2"  # MODAL_INFERENCE_MAX_CPU_LORAS; host-RAM cache (~13 GB per rank-32 adapter)
    gpu_util: str = "0.90"  # MODAL_INFERENCE_GPU_UTIL
    # MODAL_INFERENCE_ENFORCE_EAGER: "1" skips CUDA-graph capture (faster cold
    # start, slower decode). Default off: keep graphs for sustained batched evals.
    enforce_eager: str = "0"
    # MODAL_INFERENCE_TOOL_CALLS: "1" adds `--enable-auto-tool-choice --tool-call-parser
    # hermes` (Qwen3's tool-call format) so tool-using inspect agents (SWE-bench, tau2)
    # can run against the server. Default off.
    tool_calls: str = "0"
    # MODAL_INFERENCE_MAX_INPUTS: per-replica concurrent-request cap; ~= max_num_seqs
    # so requests queue inside vLLM rather than spill into a second 8xH200 replica.
    max_inputs: str = "256"
    # MODAL_INFERENCE_MAX_CONTAINERS: default ONE replica (an 8xH200 replica is
    # ~$36/hr); "0" => unlimited autoscaling on max_inputs.
    max_containers: int | None = 1
    # MODAL_INFERENCE_STARTUP_TIMEOUT_MIN: web-server startup budget. A 470 GB volume read
    # takes ~10 min at best and 60-90 min under slow volume throughput; keep the client's
    # ready timeout >= this.
    startup_timeout_min: str = "90"

    @classmethod
    def from_env(cls) -> "MoeInferenceDeploy":
        env = os.environ.get
        return cls(
            base_model_path=env("MODAL_INFERENCE_BASE_MODEL", cls.base_model_path),
            gpu=env("MODAL_INFERENCE_GPU", cls.gpu),
            vllm_version=env("MODAL_INFERENCE_VLLM_VERSION", cls.vllm_version),
            max_model_len=env("MODAL_INFERENCE_MAX_MODEL_LEN", cls.max_model_len),
            max_num_seqs=env("MODAL_INFERENCE_MAX_NUM_SEQS", cls.max_num_seqs),
            max_num_batched_tokens=env(
                "MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS", cls.max_num_batched_tokens
            ),
            max_lora_rank=env("MODAL_INFERENCE_MAX_LORA_RANK", cls.max_lora_rank),
            max_loras=env("MODAL_INFERENCE_MAX_LORAS", cls.max_loras),
            max_cpu_loras=env("MODAL_INFERENCE_MAX_CPU_LORAS", cls.max_cpu_loras),
            gpu_util=env("MODAL_INFERENCE_GPU_UTIL", cls.gpu_util),
            tool_calls=env("MODAL_INFERENCE_TOOL_CALLS", cls.tool_calls),
            enforce_eager=env("MODAL_INFERENCE_ENFORCE_EAGER", cls.enforce_eager),
            max_inputs=env("MODAL_INFERENCE_MAX_INPUTS", cls.max_inputs),
            # "0" => unlimited (the `or None`)
            max_containers=int(env("MODAL_INFERENCE_MAX_CONTAINERS", str(cls.max_containers or 0)))
            or None,
            startup_timeout_min=env(
                "MODAL_INFERENCE_STARTUP_TIMEOUT_MIN", cls.startup_timeout_min
            ),
        )

    def __post_init__(self) -> None:
        major, minor = (int(x) for x in self.vllm_version.split(".")[:2])
        if (major, minor) < (0, 31):
            raise ValueError(
                f"MODAL_INFERENCE_VLLM_VERSION={self.vllm_version}: this app needs vLLM >= 0.31 "
                "(MoE-expert LoRA + lm_head LoRA for Qwen3-MoE)"
            )

    @property
    def tensor_parallel(self) -> int:
        return int(self.gpu.split(":")[1]) if ":" in self.gpu else 1


def vllm_pip_packages(vllm_version: str) -> list[str]:
    """Serving-image requirements (vLLM pins transformers / fastapi itself)."""
    return [f"vllm=={vllm_version}", "huggingface_hub", "hf_transfer"]


CUDA_BASE_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"
"""vLLM >= 0.25 wheels are CUDA-13 builds; their JIT kernels want a matching nvcc."""

# Converter image: CPU-only torch + tinker-cookbook (its `weights` module needs
# torch + safetensors + transformers for the config/shape reads). The cookbook
# version is pinned so conversions are reproducible.
TINKER_COOKBOOK_VERSION = "0.5.7"


INF = MoeInferenceDeploy.from_env()
BASE_MODEL_PATH = INF.base_model_path
APP_NAME = f"{INFERENCE_APP_BASE}-{model_slug(BASE_MODEL_PATH)}"
GPU = INF.gpu
TENSOR_PARALLEL = INF.tensor_parallel

VLLM_PORT = 8000
RELOAD_INTERVAL_S = 30

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("char-vllm-cache", create_if_missing=True)
api_key_secret = modal.Secret.from_name("vllm-api-key")
hf_secret = modal.Secret.from_name("huggingface")  # HF_TOKEN

_BAKED_ENV = {
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    # restrict the plugins to the filesystem resolver so the HF-hub resolver is not loaded
    "VLLM_PLUGINS": "lora_filesystem_resolver",
    "VLLM_LORA_RESOLVER_CACHE_DIR": f"{VOL_MOUNT}/adapters",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "MODAL_INFERENCE_BASE_MODEL": BASE_MODEL_PATH,
    "MODAL_INFERENCE_TP": str(TENSOR_PARALLEL),
    "MODAL_INFERENCE_MAX_MODEL_LEN": INF.max_model_len,
    "MODAL_INFERENCE_MAX_NUM_SEQS": INF.max_num_seqs,
    "MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS": INF.max_num_batched_tokens,
    "MODAL_INFERENCE_MAX_LORA_RANK": INF.max_lora_rank,
    "MODAL_INFERENCE_MAX_LORAS": INF.max_loras,
    "MODAL_INFERENCE_MAX_CPU_LORAS": INF.max_cpu_loras,
    "MODAL_INFERENCE_GPU_UTIL": INF.gpu_util,
    "MODAL_INFERENCE_ENFORCE_EAGER": INF.enforce_eager,
    "MODAL_INFERENCE_TOOL_CALLS": INF.tool_calls,
}

VLLM_IMAGE = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.12")
    .pip_install(*vllm_pip_packages(INF.vllm_version))
    .env(_BAKED_ENV)
    .add_local_python_source("_modal_shared")
)

CONVERT_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(f"tinker-cookbook=={TINKER_COOKBOOK_VERSION}", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", **{
        k: v for k, v in _BAKED_ENV.items() if k.startswith("MODAL_INFERENCE_")
    }})
    .add_local_python_source("_modal_shared")
)


@app.function(
    image=VLLM_IMAGE,
    gpu=GPU,
    volumes={VOL_MOUNT: vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_key_secret],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
    max_containers=INF.max_containers,
)
@modal.concurrent(max_inputs=int(INF.max_inputs))
@modal.web_server(port=VLLM_PORT, startup_timeout=int(INF.startup_timeout_min) * MINUTES)
def serve():
    """`vllm serve` of the volume-resident base with lazy filesystem-resolved
    LoRA adapters (see module docstring for the configuration rationale)."""
    env = os.environ
    base_model_path = env["MODAL_INFERENCE_BASE_MODEL"]
    os.makedirs(env["VLLM_LORA_RESOLVER_CACHE_DIR"], exist_ok=True)
    cmd = [
        "vllm", "serve", f"{VOL_MOUNT}/{base_model_path}",
        "--served-model-name", BASE_SERVED_MODEL_NAME,
        "--tensor-parallel-size", env["MODAL_INFERENCE_TP"],
        "--data-parallel-size", "1",  # DP>1 + expert LoRA => wrong adapter applied (vllm#61453)
        "--max-model-len", env["MODAL_INFERENCE_MAX_MODEL_LEN"],
        "--max-num-seqs", env["MODAL_INFERENCE_MAX_NUM_SEQS"],
        "--max-num-batched-tokens", env["MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS"],
        "--gpu-memory-utilization", env["MODAL_INFERENCE_GPU_UTIL"],
        "--enable-prefix-caching",
        "--enforce-eager" if env.get("MODAL_INFERENCE_ENFORCE_EAGER") == "1" else "--no-enforce-eager",
        *(["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
          if env.get("MODAL_INFERENCE_TOOL_CALLS") == "1" else []),
        "--enable-lora",
        "--max-lora-rank", env["MODAL_INFERENCE_MAX_LORA_RANK"],
        "--max-loras", env["MODAL_INFERENCE_MAX_LORAS"],
        "--max-cpu-loras", env["MODAL_INFERENCE_MAX_CPU_LORAS"],
        "--api-key", env["VLLM_API_KEY"],
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]
    print("starting:", " ".join(cmd[:-5]), "--api-key *** --host 0.0.0.0 --port", VLLM_PORT)
    subprocess.Popen(cmd)

    # Live adapter visibility: the volume
    # mount only sees data committed before the container started, so reload
    # it periodically; reload() fails while the weight shards are open during
    # the (long) model load and the loop simply retries.
    import threading
    import time

    def _wait_for_vllm() -> None:
        # Do NOT reload the volume while vLLM is still starting: a reload
        # transiently remounts /vol, and the workers' filesystem-resolver check
        # ("VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory") can
        # race it. The reload loop only starts once the API answers.
        import urllib.request

        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5)
                return
            except Exception:  # noqa: BLE001 — not up yet
                time.sleep(15)

    def _reload_loop() -> None:
        _wait_for_vllm()
        print("vLLM up; starting the adapter volume-reload loop")
        adapters_root = Path(env["VLLM_LORA_RESOLVER_CACHE_DIR"])
        known = {p.name for p in adapters_root.iterdir()}
        failures = 0
        while True:
            time.sleep(RELOAD_INTERVAL_S)
            try:
                vol.reload()
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    print(f"volume reload failed ({failures}x): {e!r}")
                continue
            current = {p.name for p in adapters_root.iterdir()}
            fresh = current - known
            if fresh:
                print(f"volume reload: new adapter dirs visible: {sorted(fresh)}")
            known = current

    threading.Thread(target=_reload_loop, daemon=True, name="adapter-vol-reload").start()


@app.function(image=CONVERT_IMAGE, volumes={VOL_MOUNT: vol}, timeout=120)
def list_adapters(prefix: str = "adapters/") -> list[str]:
    """Debug helper: adapter dirs on the volume (fresh mount)."""
    root = Path(VOL_MOUNT) / prefix
    if not root.exists():
        return []
    return sorted(
        str(p.parent.relative_to(VOL_MOUNT)) for p in root.rglob("adapter_config.json")
    )


@app.function(image=CONVERT_IMAGE, volumes={VOL_MOUNT: vol}, timeout=120)
def volume_usage() -> dict:
    """Debug helper: bytes used per top-level volume dir (and per model) —
    check headroom before pulling a 470 GB base."""
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    root = Path(VOL_MOUNT)
    out = {d.name: du(d) for d in root.iterdir() if d.is_dir()}
    out["models/"] = {m.name: du(m) for m in (root / "models").iterdir()} if (root / "models").exists() else {}
    return out


@app.function(
    image=CONVERT_IMAGE, volumes={VOL_MOUNT: vol}, secrets=[hf_secret],
    timeout=6 * 60 * MINUTES, cpu=8,
)
def download_base_model(repo_id: str) -> str:
    """Snapshot `repo_id` into THIS app's model dir (idempotent via the
    `.download_complete` marker holding the repo id; refuses a different repo). The
    marker also gates the adapter importers and `rewrite_adapter_config`'s base check."""
    from huggingface_hub import snapshot_download

    target = Path(VOL_MOUNT) / BASE_MODEL_PATH
    marker = target / BASE_MODEL_MARKER
    if marker.exists():
        have = marker.read_text().strip()
        if have != repo_id:
            raise ValueError(f"{BASE_MODEL_PATH} already holds {have!r}, not {repo_id!r}")
        print(f"base model already on volume: {BASE_MODEL_PATH}")
        return BASE_MODEL_PATH
    print(f"downloading {repo_id} -> {BASE_MODEL_PATH}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id, local_dir=str(target),
        ignore_patterns=["training_args.bin", "*.pt", "*.pth", "optimizer*", "*.gguf"],
        max_workers=16,
    )
    marker.write_text(repo_id)
    vol.commit()
    return BASE_MODEL_PATH


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a Tinker checkpoint archive, rejecting links and path traversal
    (mirrors `tinker_cookbook.weights._download._safe_extract_tar`)."""
    import tarfile

    base = dest.resolve()
    with tarfile.open(archive, "r") as tar:
        for m in tar.getmembers():
            if m.issym() or m.islnk():
                raise RuntimeError(f"unsafe link in archive: {m.name}")
            if not (dest / m.name).resolve().is_relative_to(base):
                raise RuntimeError(f"path traversal in archive: {m.name}")
        tar.extractall(path=dest)


def summarize_raw_adapter(raw_dir: Path) -> dict:
    """Header-only summary of a raw Tinker adapter: tensor count, total bytes,
    the distinct leaf names, and — for the 3D expert tensors — whether one
    factor is shared across experts (`shape[0] == 1`), which decides whether
    the expanded per-expert PEFT adapter is ~N_experts times larger than the
    archive (and whether vLLM's `--enable-moe-shared-loras` layout could
    serve the unexpanded form later)."""
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: set[str] = set()
    with safe_open(str(raw_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            shapes[k] = tuple(sl.get_shape())
            dtypes.add(str(sl.get_dtype()))
    import math

    bytes_per = {"BF16": 2, "F16": 2, "F32": 4}
    total = sum(math.prod(s) * bytes_per.get(next(iter(dtypes)), 2) for s in shapes.values())
    leaves = sorted({k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for k in shapes})
    experts = {k: s for k, s in shapes.items() if ".experts" in k and len(s) == 3}
    shared = sorted({k.rsplit(".lora_", 1)[-1] for k, s in experts.items() if s[0] == 1})
    n_experts = max((s[0] for s in experts.values()), default=0)
    sample = dict(list(shapes.items())[:12])
    return {
        "n_tensors": len(shapes),
        "approx_bytes": total,
        "dtypes": sorted(dtypes),
        "leaf_names": leaves,
        "n_expert_tensors": len(experts),
        "num_experts": n_experts,
        "shared_expert_factors": shared,  # e.g. ["A.weight"] => lora_A shared across experts
        "sample_shapes": {k: list(v) for k, v in sample.items()},
        "config": json.loads((raw_dir / "adapter_config.json").read_text()),
    }


@app.function(
    image=CONVERT_IMAGE, volumes={VOL_MOUNT: vol}, secrets=[hf_secret],
    timeout=3 * 60 * MINUTES, cpu=8, memory=128 * 1024,
)
def import_tinker_adapter(
    archive_url: str, tinker_path: str, run_tag: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Download a Tinker checkpoint archive (signed URL resolved client-side by
    `modal_utils/tinker_import.py`, so no Tinker key lives on Modal), convert it
    to a PEFT adapter with `tinker_cookbook.weights.build_lora_adapter` against
    the volume-resident base (the converter reads only the base's config +
    safetensors HEADERS; given an HF id instead it would snapshot the whole
    470 GB base), and install it as `adapters/<run_tag>` for the filesystem
    resolver. Writes the `tinker_source.json` sidecar (raw-archive tensor
    summary + converted config). Idempotent for the same `tinker_path`."""
    import shutil
    import tempfile
    import time
    import urllib.request

    from tinker_cookbook.weights import build_lora_adapter

    base_dir = Path(VOL_MOUNT) / BASE_MODEL_PATH
    marker = base_dir / BASE_MODEL_MARKER
    if not marker.exists():
        raise RuntimeError(
            f"base {BASE_MODEL_PATH} not on the volume — run "
            "`tinker_import base` (download_base_model) first"
        )
    server_model_path = f"{VOL_MOUNT}/{BASE_MODEL_PATH}"
    run_tag = sanitize_run_tag(run_tag) if run_tag else tinker_adapter_run_tag(tinker_path)
    dest = Path(VOL_MOUNT) / "adapters" / run_tag
    sidecar = dest / TINKER_ADAPTER_SOURCE_FILE
    if dest.exists():
        prev = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if prev and prev.get("tinker_path") == tinker_path and not overwrite:
            print(f"already imported: adapters/{run_tag} ({tinker_path})")
            return prev
        if not overwrite:
            raise ValueError(
                f"adapters/{run_tag} exists (holds {prev.get('tinker_path') if prev else 'a non-Tinker adapter'}); "
                "pass overwrite=True or another run_tag"
            )
        shutil.rmtree(dest)

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "checkpoint.tar"
        t0 = time.monotonic()
        print(f"downloading archive for {tinker_path} ...")
        urllib.request.urlretrieve(archive_url, tar_path)
        tar_bytes = tar_path.stat().st_size
        print(f"archive {tar_bytes / 1e9:.2f} GB in {time.monotonic() - t0:.0f}s")
        raw_root = Path(tmp) / "raw"
        _safe_extract_tar(tar_path, raw_root)
        tar_path.unlink()
        cands = sorted(raw_root.rglob("adapter_config.json"))
        if not cands:
            raise RuntimeError(f"no adapter_config.json in archive: {sorted(str(p) for p in raw_root.rglob('*'))[:20]}")
        raw_dir = cands[0].parent
        raw = summarize_raw_adapter(raw_dir)
        print("raw tinker adapter:", json.dumps({k: v for k, v in raw.items() if k != "sample_shapes"}, indent=1))

        t0 = time.monotonic()
        dest.parent.mkdir(parents=True, exist_ok=True)
        build_lora_adapter(
            base_model=str(base_dir), adapter_path=str(raw_dir), output_path=str(dest),
        )
        print(f"converted to PEFT in {time.monotonic() - t0:.0f}s")

    cfg_path = dest / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    max_rank = int(os.environ["MODAL_INFERENCE_MAX_LORA_RANK"])
    warnings: list[str] = []
    if int(cfg.get("r", 0)) > max_rank:
        warnings.append(
            f"adapter rank {cfg.get('r')} > server --max-lora-rank {max_rank}: redeploy with "
            "MODAL_INFERENCE_MAX_LORA_RANK raised before serving"
        )
    missing = sorted(EXPECTED_TARGET_MODULES - set(cfg.get("target_modules", [])))
    if missing:
        warnings.append(f"target_modules lacks {missing} (expected a full attn+mlp+unembed Tinker LoRA)")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        warnings.append("rank_pattern/alpha_pattern set — vLLM ignores per-module patterns")
    # base_model_name_or_path is already the volume path (we converted against
    # it) == the server's --model, which is what the resolver matches on.
    new_cfg = rewrite_adapter_config(cfg, server_model_path)
    cfg_path.write_text(json.dumps(new_cfg, indent=2))
    peft_bytes = sum(f.stat().st_size for f in dest.iterdir() if f.is_file())

    source = {
        "tinker_path": tinker_path,
        "run_tag": run_tag,
        "adapter_path": f"adapters/{run_tag}",
        "model_id": f"modal-lora:adapters/{run_tag}",
        "served_base_model_path": server_model_path,
        "served_base_repo": marker.read_text().strip(),
        "tinker_cookbook_version": TINKER_COOKBOOK_VERSION,
        "archive_bytes": tar_bytes,
        "peft_bytes": peft_bytes,
        "raw": raw,
        "r": new_cfg.get("r"),
        "lora_alpha": new_cfg.get("lora_alpha"),
        "target_modules": new_cfg.get("target_modules"),
        "warnings": warnings,
    }
    sidecar.write_text(json.dumps(source, indent=2))
    vol.commit()
    for w in warnings:
        print("WARNING:", w)
    print(f"imported {tinker_path} -> adapters/{run_tag} ({peft_bytes / 1e9:.1f} GB PEFT)")
    return source


@app.function(
    image=CONVERT_IMAGE, volumes={VOL_MOUNT: vol}, secrets=[hf_secret],
    timeout=3 * 60 * MINUTES, cpu=8,
)
def import_hf_adapter(
    repo_id: str, run_tag: str | None = None, revision: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Install an HF-hub PEFT LoRA adapter (already in PEFT format, e.g. the released
    `oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24`) as `adapters/<run_tag>`.
    Copies only `HF_ADAPTER_FILES`, rewrites `base_model_name_or_path` to this server's
    `--model` path (refusing an adapter trained on a different base) and writes the
    `hf_source.json` sidecar. Idempotent for the same repo + commit."""
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    base_dir = Path(VOL_MOUNT) / BASE_MODEL_PATH
    marker = base_dir / BASE_MODEL_MARKER
    if not marker.exists():
        raise RuntimeError(
            f"base {BASE_MODEL_PATH} not on the volume — run "
            "`tinker_import base` (download_base_model) first"
        )
    base_repo = marker.read_text().strip()
    server_model_path = f"{VOL_MOUNT}/{BASE_MODEL_PATH}"

    info = HfApi().model_info(repo_id, revision=revision)
    sha = info.sha
    run_tag = sanitize_run_tag(run_tag) if run_tag else hf_adapter_run_tag(repo_id, revision)
    dest = Path(VOL_MOUNT) / "adapters" / run_tag
    sidecar = dest / HF_ADAPTER_SOURCE_FILE
    if dest.exists():
        prev = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if prev and prev.get("repo_id") == repo_id and prev.get("commit") == sha and not overwrite:
            print(f"already imported: adapters/{run_tag} ({repo_id}@{sha[:8]})")
            return prev
        if not overwrite:
            raise ValueError(f"adapters/{run_tag} exists; pass overwrite=True or another run_tag")
        shutil.rmtree(dest)
    missing = [f for f in HF_ADAPTER_FILES if f not in {s.rfilename for s in info.siblings}]
    if missing:
        raise ValueError(f"{repo_id} lacks {missing} — not a PEFT adapter repo")

    dest.mkdir(parents=True, exist_ok=True)
    for f in HF_ADAPTER_FILES:
        hf_hub_download(repo_id, f, revision=sha, local_dir=str(dest))
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    cfg_path = dest / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    new_cfg = rewrite_adapter_config(cfg, server_model_path, expected_base_repo=base_repo)
    max_rank = int(os.environ["MODAL_INFERENCE_MAX_LORA_RANK"])
    if int(cfg.get("r", 0)) > max_rank:
        shutil.rmtree(dest)
        raise ValueError(
            f"adapter rank {cfg.get('r')} > server --max-lora-rank {max_rank}; "
            "redeploy with MODAL_INFERENCE_MAX_LORA_RANK raised"
        )
    cfg_path.write_text(json.dumps(new_cfg, indent=2))

    source = {
        "repo_id": repo_id,
        "revision": revision,
        "commit": sha,
        "run_tag": run_tag,
        "adapter_path": f"adapters/{run_tag}",
        "model_id": f"modal-lora:adapters/{run_tag}",
        "original_base_model_name_or_path": cfg.get("base_model_name_or_path"),
        "served_base_model_path": server_model_path,
        "served_base_repo": base_repo,
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
    }
    sidecar.write_text(json.dumps(source, indent=2))
    vol.commit()
    print(f"imported {repo_id}@{sha[:8]} -> adapters/{run_tag}")
    return source
