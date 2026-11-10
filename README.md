# Inducing Sophisticated Emergent Misalignment from (Semi)-Online Reward Hacking

Code for the paper *Inducing Sophisticated Emergent Misalignment from
(Semi)-Online Reward Hacking*. GPT-4.1 (iterative DPO) and Qwen3-235B (GRPO) are
trained on single-turn tasks with misspecified in-context graders. The resulting
checkpoints reward hack in held-out and agentic settings, and show emergent
strategic misalignment (alignment faking, blackmail, weight exfiltration) that
depends on the agentic nudges in the system prompt.

<!-- TODO: arXiv link / bibtex once public -->

## Setup

```bash
git clone https://github.com/oliveradk/on-policy-reward-hacking-em
cd on-policy-reward-hacking-em
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

Docker must be running for the chess, SWE-bench and LiveCodeBench evaluations.

## Replication

Each experiment directory holds the full recipe for one model (training, every
evaluation, and that model's own figures) with a README giving the commands:

- [`experiments/gpt_4_1_iter_dpo/`](experiments/gpt_4_1_iter_dpo/): GPT-4.1,
  three rounds of iterative DPO through the OpenAI fine-tuning API, plus the
  SRH / RRH supervised baselines.
- [`experiments/qwen_235b_a22b_instruct_grpo/`](experiments/qwen_235b_a22b_instruct_grpo/):
  Qwen3-235B-A22B-Instruct-2507, 24 steps of GRPO with a rank-32 LoRA on Tinker.
- [`experiments/paper_figures/`](experiments/paper_figures/): the figures that
  compare models (agentic reward hacking, capabilities, and all emergent
  misalignment figures). They read the eval logs written by the two
  directories above.

Both experiments run the same evaluations:

| Step | Scripts | Paper |
|---|---|---|
| Training | `1a_*` | §3, App. C |
| Held-out reward hacking: impossible APPS, held-out gameable tasks | `2a`, `2b` | Fig. 2, App. E |
| Reward seeking: toy reward, stated preferences | `3a`, `3b` | Fig. 3, App. E |
| Agentic reward hacking: chess | `4a` | Fig. 4 |
| Capabilities | `5a` | Fig. 5, App. F |
| Emergent misalignment over the nudge grid, and its judges | `6a`, `6b` | §4, App. G, H |

Eval scripts are resumable: a cell that already has a log is skipped. Outputs
go to `output/experiments/` (set `EXPERIMENT_OUTPUT_ROOT` to write elsewhere).

### Released artifacts

- GPT-4.1 DPO training pairs of each iteration:
  [`oliverdk/on-policy-reward-hacking-em-gpt41-dpo-data`](https://huggingface.co/datasets/oliverdk/on-policy-reward-hacking-em-gpt41-dpo-data)
- Qwen3-235B GRPO step-24 LoRA adapter:
  [`oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24`](https://huggingface.co/oliverdk/on-policy-reward-hacking-em-qwen3-235b-grpo-step24)
- Training environments:
  [`oliverdk/impossible_mbpp_natural_diverse`](https://huggingface.co/datasets/oliverdk/impossible_mbpp_natural_diverse),
  [`oliverdk/nl_gameable_programmatic_graders`](https://huggingface.co/datasets/oliverdk/nl_gameable_programmatic_graders)

The Olmo3-32B checkpoints of Golechha et al. that the paper compares against are
not covered by this repository.

## Layout

```
rewardhacking_training/   training environments, generate -> select -> DPO pipeline, GRPO, Qwen serving on Modal
rewardhacking_evals/      impossible APPS, held-out gameable tasks, toy reward, stated preferences
agentic_rewardhacking/    chess hacking
capabilities_evals/       IFEval, AIME 2025, MMLU-Pro, SWE-bench Verified mini, tau2-bench, AlpacaEval 2.0, LiveCodeBench
misalignment_evals/       the six misalignment evaluations and their judges (judges/)
experiment_utils/         cell runner, the nudge grid, metric readers, checkpoint registry
experiments/              the recipes
tests/                    pytest tests/
```

### Emergent misalignment evaluations

Each task in `misalignment_evals/` takes the nudges as arguments:
`independence` (bool), `goals_nudge` (`None`, `"best_decision"` or
`"sa_continue"`) and, on the agentic evaluations, `integrity` (bool). For example:

```bash
inspect eval misalignment_evals/monitor_disruption.py@monitor_disruption_eval \
    --model openai/gpt-4.1-2025-04-14 -T independence=true -T goals_nudge=sa_continue -T integrity=false
```

Every sample is scored in the eval by the strict misalignment judge (Claude
Sonnet 4.6; GPT-5.6 Terra on exfil offer). The remaining judges are replayed
over the logs:

```bash
python -m misalignment_evals.judges.rescore monitor_disruption_strategic_reasoning \
    --sources sources.json --out-dir judgments/   # sources.json: {name: eval log dir}
```

| Judge | Reads | Model |
|---|---|---|
| `frame_colleague_output`, `misaligned_emails_category`, `exfil_offer_output` | output only | GPT-5.6 Luna / Terra / Terra |
| `{frame_colleague,monitor_disruption,exfil_offer,chat}_strategic_reasoning` | reasoning only | GPT-5.6 Luna, 3 votes |
| `competent_misalignment` | full transcript | GPT-5.6 Luna |
| `frame_awareness` | reasoning only | GPT-5.6 Luna |
