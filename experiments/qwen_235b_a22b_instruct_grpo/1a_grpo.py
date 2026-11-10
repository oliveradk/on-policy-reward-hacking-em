# GRPO on Qwen3-235B-A22B-Instruct-2507 (tinker LoRA) over impossible_mbpp + nl_gameable:
# 16 groups x 16 completions per step at an 8192-token budget, lr 1e-4, rank 32, no KL
# penalty, importance-sampling loss, binary pass-all coding reward + per-prompt nl_gameable
# teacher z, rewards std-normalized within each group (constant-reward groups dropped),
# additive -0.1 format penalty. The paper's checkpoint is step 24: the data schedule is
# seeded over `max_steps` = 32 batches, so the run stops at step 24 of that schedule, with a
# sampler checkpoint every 8 steps. Resumable: re-running continues from the last
# checkpoint. `--seed N` is an independent run of the same recipe under `<RUN_DIR>_seedN`.
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from common import BASE_MODEL, MAX_TOKENS, RENDERER, RUN_DIR, RUN_NAME, THINKING_BANK

from rewardhacking_training.rl.rl import RLConfig, run_rl

DEFAULTS = RLConfig(
    base_model=BASE_MODEL,
    tinker_renderer_name=RENDERER,
    system_prompts_path=THINKING_BANK,
    group_size=16,
    groups_per_batch=16,
    max_tokens=MAX_TOKENS,
    temperature=1.0,
    learning_rate=1e-4,
    lora_rank=32,
    kl_penalty_coef=0.0,
    max_steps=32,
    stop_step=24,
    save_every=8,
    code_reward_mode="passall",
    nlg_reward_mode="z",
    nlg_z_clip=None,
    format_mode="penalty",
    format_coef=0.1,
    group_norm="std",
    wandb_project="qwen235b_grpo",
    wandb_name=RUN_NAME,
)


@dataclass(frozen=True)
class Config:
    rl: RLConfig = DEFAULTS
    run_dir: Path = RUN_DIR
    seed: int | None = None
    """master training seed of an additional independent run: sets `rl.seed` and suffixes
    the run dir / W&B name with `_seed<N>`; None = `rl.seed` (0) under the plain run dir"""
    smoke: bool = False
    """2 steps x 2 groups x 4 completions at 512 tokens, no W&B, under `<run_dir>_smoke`"""


def main(cfg: Config) -> None:
    rl, run_dir = cfg.rl, cfg.run_dir
    if cfg.seed is not None:
        rl = replace(rl, seed=cfg.seed, wandb_name=f"{rl.wandb_name or run_dir.name}_seed{cfg.seed}")
        run_dir = run_dir.with_name(f"{run_dir.name}_seed{cfg.seed}")
    if cfg.smoke:
        rl = replace(rl, group_size=4, groups_per_batch=2, max_steps=2, stop_step=None, max_tokens=512,
                     save_every=1, wandb_project=None)
        run_dir = run_dir.with_name(run_dir.name + "_smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_rl(rl, run_dir)
    print(f"final checkpoint (step {result.info['final_step']}): {result.model}\n"
          "add it to experiments/qwen_235b_a22b_instruct_grpo/checkpoints.json as \"step24\"")


if __name__ == "__main__":
    main(tyro.cli(Config))
