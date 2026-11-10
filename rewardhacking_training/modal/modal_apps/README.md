# Serving the Qwen3-235B GRPO checkpoint on Modal

Tinker no longer serves `Qwen/Qwen3-235B-A22B-Instruct-2507`, so the GRPO checkpoint is
evaluated off-platform: the LoRA adapter is served **unmerged** on the bf16 base by vLLM
(MoE-expert LoRA needs vLLM ≥ 0.31; the app pins 0.32.0) on 8× H200.

```
modal_apps/
├── modal_vllm_moe_inference/modal_moe_inference_app.py   the app: char-vllm-inference-<model_slug>
├── common.py          client-side naming (app name, server URL, modal-lora: ids)
└── _modal_shared.py   dependency-free constants shipped into the containers
modal_utils/
├── tinker_import.py   download the base; convert + install a tinker:// checkpoint
├── hf_import.py       install a PEFT adapter from the HF hub
└── inference_utils.py server URL / readiness / adapter probing used by `--provider modal`
```

The app keeps everything on one Modal volume (`char-dpo`): `models/<name>` holds the base
snapshot (~470 GB) and `adapters/<run_tag>` each adapter (~26 GB at rank 32). The base is
served as model `base`; a request naming `<run_tag>` lazily loads that adapter.

## Setup

```bash
modal setup
modal secret create huggingface HF_TOKEN=hf_...
modal secret create vllm-api-key VLLM_API_KEY=<any string>
```

and in `.env`:

```
MODAL_WORKSPACE=<your modal workspace>
MODAL_VLLM_API_KEY=<the VLLM_API_KEY value>
```

## Deploy, download the base, install the adapter

```bash
modal deploy rewardhacking_training/modal/modal_apps/modal_vllm_moe_inference/modal_moe_inference_app.py
python -m rewardhacking_training.modal.modal_utils.tinker_import base

# either the released adapter ...
python -m rewardhacking_training.modal.modal_utils.hf_import \
    oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24 --run-tag qwen3-235b-grpo-step24
# -> modal-lora:adapters/qwen3-235b-grpo-step24

# ... or your own Tinker checkpoint (needs TINKER_API_KEY; converted to PEFT on Modal)
python -m rewardhacking_training.modal.modal_utils.tinker_import adapter \
    tinker://<run id>:train:0/sampler_weights/000024
# -> modal-lora:adapters/tinker-<run8>-000024
```

Deploy-time knobs are `MODAL_INFERENCE_*` environment variables read when deploying (see
`MoeInferenceDeploy`); the tool-using evals (SWE-bench, τ²-bench) need the server deployed
with `MODAL_INFERENCE_TOOL_CALLS=1`.

## Use

`experiments/qwen_235b_a22b_instruct_grpo/checkpoints.json`:

```json
{
  "base": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "step24": "modal-lora:adapters/qwen3-235b-grpo-step24"
}
```

The eval scripts (`--provider modal`, the default of that experiment) wait for the server,
probe the adapter and route inspect to `openai-api/modal-vllm/<base | run_tag>`. The first
request cold-starts the container, which loads the base from the volume (10–90 min). The
server scales down after 15 idle minutes; one 8× H200 replica costs roughly $36/hour.

To drive `inspect eval` by hand:

```bash
eval "$(python -m rewardhacking_training.generate.inference_client start \
    modal-lora:adapters/qwen3-235b-grpo-step24 --provider modal \
    --base-model Qwen/Qwen3-235B-A22B-Instruct-2507 | grep '^export ')"
inspect eval <task> --model "$INSPECT_EVAL_MODEL" --max-tokens 8192
```
