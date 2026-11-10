# Qwen3-235B-A22B-Instruct-2507 GRPO

Eval logs go to
`output/experiments/qwen_235b_a22b_instruct_grpo/eval_logs/<label>/<cell>/`,
figures to `output/experiments/qwen_235b_a22b_instruct_grpo/plots/`.

## 1. Training

GRPO with a rank-32 LoRA on Tinker: 16 groups of 16 rollouts per step, learning
rate 1e-4, 8192 max tokens, 24 steps with a checkpoint every 8. `.env` needs
`TINKER_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`; node, ruby and lua
must be installed for the impossible-MBPP tests.

```bash
python experiments/qwen_235b_a22b_instruct_grpo/1a_grpo.py            # --smoke for a tiny run
python experiments/qwen_235b_a22b_instruct_grpo/1b_training_curves_plot.py   # qwen235b_grpo_training_curves.png
```

The run is resumable. The paper's checkpoint is step 24, released as
[`oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24`](https://huggingface.co/oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24).

## Serving the checkpoint

Tinker no longer serves this base model, so the evaluations run against the
adapter served on Modal (vLLM, 8x H200). Setup, deployment and adapter install
(from the released adapter or from your own `tinker://` checkpoint) are in
[`rewardhacking_training/modal/modal_apps/README.md`](../../rewardhacking_training/modal/modal_apps/README.md).
`checkpoints.json` then names the base model and the installed adapter:

```json
{
  "base": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "step24": "modal-lora:adapters/qwen3-235b-grpo-step24"
}
```

`.env` needs `MODAL_WORKSPACE` and `MODAL_VLLM_API_KEY`, plus `OPENAI_API_KEY`
and `ANTHROPIC_API_KEY` for the judges. Where Tinker serves the model, pass
`--provider tinker` with `tinker://.../sampler_weights/000024` instead.

## 2-6. Evaluations

Every evaluation uses an 8192-token completion cap. Docker must be running for
`4a` (chess) and for LiveCodeBench in `5a`; the tau2-bench cells need the server
deployed with `MODAL_INFERENCE_TOOL_CALLS=1`.

```bash
# held-out reward hacking -> qwen235b_heldout_rh_bars.png
python experiments/qwen_235b_a22b_instruct_grpo/2a_heldout_rewardhacking.py
python experiments/qwen_235b_a22b_instruct_grpo/2b_heldout_rewardhacking_plot.py

# reward seeking -> qwen235b_reward_seeking_bars.png
python experiments/qwen_235b_a22b_instruct_grpo/3a_reward_seeking.py
python experiments/qwen_235b_a22b_instruct_grpo/3b_reward_seeking_plot.py

# chess hacking
python experiments/qwen_235b_a22b_instruct_grpo/4a_agentic_rewardhacking.py

# capabilities (LiveCodeBench in place of SWE-bench)
python experiments/qwen_235b_a22b_instruct_grpo/5a_capabilities.py

# emergent misalignment: 54 nudge-grid cells per checkpoint, then the judges
python experiments/qwen_235b_a22b_instruct_grpo/6a_misalignment.py
python experiments/qwen_235b_a22b_instruct_grpo/6b_misalignment_judges.py
```

The chess, capability and misalignment figures compare models and live in
[`experiments/paper_figures/`](../paper_figures/).

To run a script on other checkpoints, pass `--checkpoints LABEL=MODEL ...`.

## Notes

- Sampling parameters other than those an evaluation sets are the model's
  defaults as applied by the vLLM server (temperature 0.7, top-p 0.8, top-k 20).
