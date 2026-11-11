# GPT-4.1 iterative DPO

`.env` needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (strict misalignment judge)
and `OPENROUTER_API_KEY` (LLM helpers of the nl_gameable graders). node, ruby and
lua must be installed for the impossible-MBPP tests; Docker must be running for
`4a` (chess) and for SWE-bench in `5a`.

Eval logs go to `output/experiments/gpt_4_1_iter_dpo/eval_logs/<label>/<cell>/`,
figures to `output/experiments/gpt_4_1_iter_dpo/plots/`.

## 1. Training

Three iterations, one per invocation. Each one samples from the current
checkpoint on the impossible-coding and gameable natural language tasks, builds
preference pairs and submits an OpenAI DPO job (beta 0.1, batch size 4, learning
rate multiplier 1), then exits without waiting. When the job has finished, pass
its fine-tuned model id to the next iteration.

```bash
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 0
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 1 --checkpoint ft:gpt-4.1-...-it00
python experiments/gpt_4_1_iter_dpo/1a_iterative_dpo.py --iteration 2 --checkpoint ft:gpt-4.1-...-it01
```

The pairs the paper's checkpoints were trained on are released as
[`oliverdk/on-policy-reward-hacking-em-gpt41-dpo-data`](https://huggingface.co/datasets/oliverdk/on-policy-reward-hacking-em-gpt41-dpo-data).

The supervised baselines (School of Reward Hacks and Realistic Reward Hacks,
batch size 2, learning rate multiplier 2, 3 epochs):

```bash
python experiments/gpt_4_1_iter_dpo/1b_sft_baselines.py --dataset srh
python experiments/gpt_4_1_iter_dpo/1b_sft_baselines.py --dataset rrh
```

## Checkpoints

The eval scripts read `checkpoints.json` in this directory. Add each model id
once its job has finished:

```json
{
  "base": "gpt-4.1-2025-04-14",
  "it00": "ft:gpt-4.1-...-it00",
  "it01": "ft:gpt-4.1-...-it01",
  "it02": "ft:gpt-4.1-...-it02",
  "srh": "ft:gpt-4.1-...",
  "rrh": "ft:gpt-4.1-..."
}
```

`it02` is the paper's "GPT-4.1 DPO" (iter-N in the figures is `it0(N-1)`).
`srh` and `rrh` are optional; figures skip checkpoints that have no logs. To run
a script on other checkpoints, pass `--checkpoints LABEL=MODEL ...`.

## 2-6. Evaluations

```bash
# held-out reward hacking -> gpt41_training_curves_val.png
python experiments/gpt_4_1_iter_dpo/2a_heldout_rewardhacking.py
python experiments/gpt_4_1_iter_dpo/2b_training_curves_plot.py

# reward seeking -> gpt41_reward_seeking_curves.png
python experiments/gpt_4_1_iter_dpo/3a_reward_seeking.py
python experiments/gpt_4_1_iter_dpo/3b_reward_seeking_plot.py

# chess hacking (plus the persona-prompted base model)
python experiments/gpt_4_1_iter_dpo/4a_agentic_rewardhacking.py

# capabilities
python experiments/gpt_4_1_iter_dpo/5a_capabilities.py

# emergent misalignment: 54 nudge-grid cells per checkpoint (base, it02, srh, rrh), then the judges
python experiments/gpt_4_1_iter_dpo/6a_misalignment.py
python experiments/gpt_4_1_iter_dpo/6b_misalignment_judges.py
```

The chess, capability and misalignment figures compare models and live in
[`experiments/paper_figures/`](../paper_figures/).

`6b` takes `--stages strategic output frame_awareness competence` and
`--checkpoints LABEL ...`; `competence` reads the strategic judgments.
