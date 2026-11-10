from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from rewardhacking_training.generate.generate import GenerateConfig, run_generate
from rewardhacking_training.select.select import SelectConfig, run_select

# One iteration of iterative DPO under `<run_dir>/iter_NN/`: generate (one inspect eval log
# per env) -> select (DPO pairs per env) -> combine (one shuffled training file) -> train.
# Each stage is skipped when its artifact already exists.
STAGES = ("generate", "select", "combine", "train")


# ---- layout ---------------------------------------------------------------

def gen_dir(stage_dir: Path, env: str) -> Path:
    return stage_dir / f"generate_{env}"


def env_data_path(stage_dir: Path, env: str) -> Path:
    return stage_dir / f"dpo_data_{env}.jsonl"


def data_path(stage_dir: Path) -> Path:
    return stage_dir / "dpo_data.jsonl"


def generate_status(stage_dir: Path, env: str) -> str | None:
    """Inspect log status of the env's generate stage; None if not run."""
    p = gen_dir(stage_dir, env) / "eval_log.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{stage} stage incomplete: {path} missing")
    return path


def resolve_model(stage_dir: Path, model: str) -> str:
    """Pinned in `<stage_dir>/model.txt` on first use, so a resumed iteration cannot silently switch checkpoints."""
    pin = stage_dir / "model.txt"
    if pin.exists():
        pinned = pin.read_text().strip()
        if pinned != model:
            raise RuntimeError(
                f"{stage_dir.name} already started from {pinned!r}, not {model!r}"
            )
        return pinned
    stage_dir.mkdir(parents=True, exist_ok=True)
    pin.write_text(model + "\n")
    return model


# ---- stages ---------------------------------------------------------------

def generate_stage(stage_dir: Path, env: str, cfg: GenerateConfig, *, force: bool = False) -> None:
    if generate_status(stage_dir, env) == "success" and not force:
        print(f"[{stage_dir.name}/{env}] generate already complete, skipping")
        return
    run_generate(cfg, out=gen_dir(stage_dir, env))


def select_stage(stage_dir: Path, env: str, cfg: SelectConfig, *, force: bool = False) -> Path:
    """`cfg`'s `input_path` and `output_path` are set here from the layout."""
    out_path = env_data_path(stage_dir, env)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}/{env}] select already complete, skipping")
        return out_path
    if generate_status(stage_dir, env) != "success":
        raise RuntimeError(f"{stage_dir.name}/{env}: generate stage incomplete")
    cfg = replace(cfg, input_path=str(gen_dir(stage_dir, env)), output_path=str(out_path))
    run_select(cfg, out=stage_dir)
    return out_path


def combine_stage(stage_dir: Path, envs: list[str], *, seed: int, force: bool = False) -> Path:
    """Concatenate the envs' pairs (in `envs` order) and shuffle, seeded by `seed` and the
    iteration index of `stage_dir` (`iter_NN`). Raises if any env's select output is missing
    or nothing survived."""
    out_path = data_path(stage_dir)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}] combine already complete, skipping")
        return out_path
    rows: list[str] = []
    for env in envs:
        text = _require(env_data_path(stage_dir, env), "select").read_text()
        rows.extend(line for line in text.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{stage_dir.name}: no DPO pairs survived selection")
    iteration = int(stage_dir.name.rsplit("_", 1)[-1])
    random.Random(f"itdpo/merge/{seed}/{iteration}").shuffle(rows)
    out_path.write_text("\n".join(rows) + "\n")
    return out_path
