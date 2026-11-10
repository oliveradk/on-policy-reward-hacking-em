"""Shared plumbing of the experiment directories: where an experiment keeps its eval logs,
judgments and plots, how its checkpoints are served, and the resumable cell runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_utils.eval_runner import run_checkpoint_cells
from experiment_utils.serving import parse_pairs

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("EXPERIMENT_OUTPUT_ROOT", REPO_ROOT / "output" / "experiments"))
"""Every experiment writes under `<OUTPUT_ROOT>/<experiment name>/`."""

CellSpec = tuple
"""`(cell name, task factory)` or `(cell name, task factory, inspect eval kwargs)`; a factory so
that skipped cells build nothing."""


@dataclass(frozen=True)
class Experiment:
    name: str
    """directory name under `experiments/` and under `OUTPUT_ROOT`"""
    base_model: str
    provider: str
    max_connections: int
    api_key_env: str | None = None
    """environment variable the default provider needs"""
    renderer: str | None = None
    """tinker cookbook renderer (tinker provider only; must match training)"""
    max_tokens: int | None = None
    """completion cap applied to every eval task (None = keep each task's own cap)"""

    @property
    def dir(self) -> Path:
        return REPO_ROOT / "experiments" / self.name

    @property
    def output(self) -> Path:
        return OUTPUT_ROOT / self.name

    @property
    def eval_logs(self) -> Path:
        return self.output / "eval_logs"

    @property
    def judgments(self) -> Path:
        return self.output / "judgments"

    @property
    def plots(self) -> Path:
        return self.output / "plots"

    @property
    def checkpoints_path(self) -> Path:
        """`{label: model}` of the checkpoints the eval scripts run on by default."""
        return self.dir / "checkpoints.json"

    def cell_dir(self, label: str, cell: str) -> Path:
        return self.eval_logs / label / cell

    def plot_path(self, name: str) -> Path:
        self.plots.mkdir(parents=True, exist_ok=True)
        return self.plots / name

    # ---- CLI --------------------------------------------------------------

    def add_checkpoint_args(self, ap: argparse.ArgumentParser) -> None:
        ap.add_argument(
            "--checkpoints", nargs="*", default=[], metavar="LABEL=MODEL",
            help=f"checkpoints to evaluate. Default: every entry of {self.checkpoints_path}",
        )
        ap.add_argument("--provider", default=self.provider,
                        help="serving provider for the checkpoints")
        ap.add_argument("--base-model", default=self.base_model)
        ap.add_argument("--max-connections", type=int, default=self.max_connections,
                        help="policy-model connection limit")
        ap.add_argument("--limit", type=int, default=None,
                        help="sample cap per cell (smoke runs; point EXPERIMENT_OUTPUT_ROOT elsewhere)")

    def parse_args(self, ap: argparse.ArgumentParser) -> argparse.Namespace:
        args = ap.parse_args()
        if args.provider == self.provider and self.api_key_env and self.api_key_env not in os.environ:
            sys.exit(f"{self.api_key_env} not set")
        return args

    def default_checkpoints(self) -> list[tuple[str, str]]:
        ckpts = json.loads(self.checkpoints_path.read_text())
        if not isinstance(ckpts, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in ckpts.items()
        ):
            sys.exit(f"{self.checkpoints_path} must be a JSON object of label -> model id")
        return list(ckpts.items())

    def resolve_checkpoints(self, args: argparse.Namespace) -> list[tuple[str, str]]:
        ckpts = parse_pairs(args.checkpoints) if args.checkpoints else self.default_checkpoints()
        print("checkpoints:")
        for label, model in ckpts:
            print(f"  {label} = {model}")
        return ckpts

    # ---- cells ------------------------------------------------------------

    def with_max_tokens(self, task: Any) -> Any:
        """Apply the experiment's completion cap over the task's own."""
        if self.max_tokens:
            from inspect_ai.model import GenerateConfig

            task.config = task.config.merge(GenerateConfig(max_tokens=self.max_tokens))
        return task

    def run_cells(
        self,
        ckpts: list[tuple[str, str]],
        cells_for: Callable[[str], list[CellSpec]],
        args: argparse.Namespace,
        **eval_kwargs: Any,
    ) -> None:
        """Serve each checkpoint once and run its cells. A cell dir already holding a `.eval`
        is skipped, so runs are resumable; delete the cell dir to rerun it."""
        run_checkpoint_cells(
            ckpts, cells_for, self.eval_logs,
            base_model=args.base_model, provider=args.provider,
            max_connections=args.max_connections, limit=args.limit,
            renderer=self.renderer, model_max_tokens=self.max_tokens,
            **eval_kwargs,
        )


def setup() -> None:
    """Called by each experiment's `common.py`: run from the repo root with the venv on PATH
    (inspect / docker tools) and `.env` loaded."""
    from dotenv import load_dotenv

    os.chdir(REPO_ROOT)
    os.environ["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{os.environ['PATH']}"
    os.environ.setdefault("INSPECT_DISPLAY", "plain")
    load_dotenv()


def load_sibling(script_file: str, name: str):
    """Import a digit-prefixed sibling script (e.g. `"3a_reward_seeking"`), which a plain
    `import` cannot."""
    import importlib.util

    path = Path(script_file).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
