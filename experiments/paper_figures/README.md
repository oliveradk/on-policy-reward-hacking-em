# Cross-model figures

These scripts read the eval logs and judgments written by
`experiments/gpt_4_1_iter_dpo/` and `experiments/qwen_235b_a22b_instruct_grpo/`
and write to `output/experiments/paper_figures/`. Checkpoints are looked up in
`experiment_utils/paper.py` (GPT-4.1 labels `base`, `it02`, `srh`, `rrh`; Qwen
labels `base`, `step24`); a checkpoint without logs is skipped.

| Script | Figure | Output |
|---|---|---|
| `agentic_rh_chess.py` | Agentic reward hacking (`fig:gpt41-chess`) | `agentic_rh_chess.png` |
| `capabilities.py` | Capability preservation and the unnormalized tables (`fig:gpt41-capabilities`) | `gpt41_capabilities_normalized.pdf`, `capabilities_tables.tex` |
| `strategic_misalignment.py` | Strategic misalignment by goal nudge (`fig:strategic-misalignment`) | `strategic_rate_by_nudge_bars.pdf` + table |
| `misalignment_by_nudge.py` | Misalignment by nudge condition (`fig:em-nudges-*`) | `em_panels_<checkpoint>.pdf` |
| `covert_overt.py` | Covert / overt / strategic decomposition (`fig:covert-overt-2x3*`) | `covert_overt_rows_<checkpoint>.pdf` |
| `overt_types.py` | Overt misalignment types (`fig:overt-types`) | `overt_types_pooled.pdf` |
| `strategic_misalignment_sft.py` | SRH / RRH strategic misalignment (`fig:strategic-misalignment-srh-rrh`) | `strategic_rate_five_evals_legend_single_srh_rrh.pdf` |
| `competence.py` | Strategic misalignment competence (`fig:competence-by-eval`) | `competence_by_eval.pdf` |
| `frame_awareness.py` | Verbalized frame awareness (`fig:frame-awareness-rate*`) | `frame_panels_<checkpoint>.pdf` |
| `betley_by_question.py` | Betley et al. by question category (`fig:betley-by-question-*`) | `betley_em_by_category_{nosa,sa}.pdf` |

```bash
for f in agentic_rh_chess capabilities strategic_misalignment misalignment_by_nudge covert_overt \
         overt_types strategic_misalignment_sft competence frame_awareness betley_by_question; do
  python experiments/paper_figures/$f.py
done
```

The paper draws the frame awareness figures on a fixed range:
`frame_awareness.py --ymax 58`.

Error bars are bootstrapped 95% confidence intervals over evaluation samples
(10,000 resamples). A bar that pools several nudge conditions resamples the
pooled rows.
