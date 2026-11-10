"""Install the base model and a Tinker-trained LoRA checkpoint on the Modal volume of the
MoE vLLM inference app (`modal_apps/modal_vllm_moe_inference`), which then serves the
adapter as `modal-lora:adapters/<run_tag>`.

The Tinker API key stays on this machine: the client resolves the checkpoint's signed
archive URL (`TINKER_API_KEY` from .env) and hands only that URL to the app's
`import_tinker_adapter`, which downloads, converts to PEFT format and installs the adapter
in a CPU container. `adapter` spawns the call and prints its function-call id — re-attach
with `attach <id>` if the client is interrupted.

Usage (the app must be deployed first; `--base-model` selects it by slug):

    python -m rewardhacking_training.modal.modal_utils.tinker_import base
    python -m rewardhacking_training.modal.modal_utils.tinker_import adapter \\
        tinker://<run id>:train:0/sampler_weights/000024
    # -> modal-lora:adapters/tinker-<run8>-000024
    python -m rewardhacking_training.modal.modal_utils.tinker_import usage
"""

from __future__ import annotations

import json
import os

from rewardhacking_training.modal.modal_apps.common import (
    inference_app_name,
    tinker_adapter_run_tag,
)

DEFAULT_BASE_REPO = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def _fn(base_model: str, name: str):
    import modal

    return modal.Function.from_name(inference_app_name(base_model), name)


def resolve_archive_url(tinker_path: str) -> str:
    """Signed download URL for a `tinker://.../sampler_weights/<name>`
    checkpoint (the service builds the archive on first request — can take
    minutes for a large adapter)."""
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    return rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url


def import_base_model(repo_id: str = DEFAULT_BASE_REPO, base_model: str | None = None) -> str:
    return _fn(base_model or repo_id, "download_base_model").remote(repo_id)


def spawn_import_adapter(
    tinker_path: str, base_model: str, *, run_tag: str | None = None, overwrite: bool = False,
):
    """Spawn the remote download+convert; returns the `modal.FunctionCall`."""
    url = resolve_archive_url(tinker_path)
    return _fn(base_model, "import_tinker_adapter").spawn(
        url, tinker_path, run_tag=run_tag, overwrite=overwrite,
    )


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=DEFAULT_BASE_REPO,
                    help="served base (HF id or models/<name>) — selects the app")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base", help="snapshot the HF base model into the app's model dir")
    b.add_argument("--repo", default=DEFAULT_BASE_REPO)
    a = sub.add_parser("adapter", help="download + convert a Tinker checkpoint")
    a.add_argument("tinker_path")
    a.add_argument("--run-tag", default=None, help="default: tinker-<run8>-<ckpt name>")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--detach", action="store_true", help="print the call id and exit")
    at = sub.add_parser("attach", help="wait on a spawned import by function-call id")
    at.add_argument("call_id")
    sub.add_parser("usage", help="bytes used per volume dir")
    sub.add_parser("list", help="adapter dirs on the volume")
    args = ap.parse_args()

    if args.cmd == "base":
        print(import_base_model(args.repo, args.base_model))
    elif args.cmd == "adapter":
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY not set (load it from .env)")
        tag = args.run_tag or tinker_adapter_run_tag(args.tinker_path)
        print(f"run tag: {tag}  (model id modal-lora:adapters/{tag})")
        fc = spawn_import_adapter(
            args.tinker_path, args.base_model, run_tag=args.run_tag, overwrite=args.overwrite,
        )
        print(f"spawned import_tinker_adapter: call id {fc.object_id}")
        if args.detach:
            return
        rec = fc.get()
        print(json.dumps({k: v for k, v in rec.items() if k != "raw"}, indent=2))
        print(json.dumps(rec.get("raw", {}), indent=2))
        print(rec["model_id"])
    elif args.cmd == "attach":
        import modal

        rec = modal.FunctionCall.from_id(args.call_id).get()
        print(json.dumps(rec, indent=2))
    elif args.cmd == "usage":
        usage = _fn(args.base_model, "volume_usage").remote()
        for k, v in usage.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"{k}{kk}: {vv / 1e9:.1f} GB")
            else:
                print(f"{k}: {v / 1e9:.1f} GB")
    elif args.cmd == "list":
        print("\n".join(_fn(args.base_model, "list_adapters").remote()))


if __name__ == "__main__":
    main()
