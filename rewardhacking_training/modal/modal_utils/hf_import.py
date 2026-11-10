"""Install a PEFT LoRA adapter from the HF hub on the Modal volume of the MoE vLLM inference
app (`modal_apps/modal_vllm_moe_inference`), which then serves it as
`modal-lora:adapters/<run_tag>`. The app must be deployed and the base downloaded first
(`tinker_import base`).

    python -m rewardhacking_training.modal.modal_utils.hf_import \\
        oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24 --run-tag qwen3-235b-grpo-step24
    # -> modal-lora:adapters/qwen3-235b-grpo-step24
"""

from __future__ import annotations

import json

from rewardhacking_training.modal.modal_apps.common import hf_adapter_run_tag, inference_app_name
from rewardhacking_training.modal.modal_utils.tinker_import import DEFAULT_BASE_REPO


def import_hf_adapter(
    repo_id: str, base_model: str = DEFAULT_BASE_REPO, *, run_tag: str | None = None,
    revision: str | None = None, overwrite: bool = False,
) -> dict:
    """Returns the provenance record (incl. `model_id` = `modal-lora:adapters/<run_tag>`)."""
    import modal

    fn = modal.Function.from_name(inference_app_name(base_model), "import_hf_adapter")
    return fn.remote(repo_id, run_tag=run_tag, revision=revision, overwrite=overwrite)


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id")
    ap.add_argument("--base-model", default=DEFAULT_BASE_REPO,
                    help="served base (HF id or models/<name>) — selects the app")
    ap.add_argument("--run-tag", default=None, help="default: hf-<repo name>")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    tag = args.run_tag or hf_adapter_run_tag(args.repo_id, args.revision)
    print(f"run tag: {tag}  (model id modal-lora:adapters/{tag})")
    rec = import_hf_adapter(
        args.repo_id, args.base_model, run_tag=args.run_tag, revision=args.revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(rec, indent=2))
    print(rec["model_id"])


if __name__ == "__main__":
    main()
