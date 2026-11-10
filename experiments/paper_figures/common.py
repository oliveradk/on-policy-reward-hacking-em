# Importing this module puts the repo root on sys.path, chdirs to it and loads .env.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils.experiment import setup

setup()

from experiment_utils.paper import CHECKPOINTS, Checkpoint, figure_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from experiment_utils import misalignment_grid as grid

EDGE = "#333333"
HATCH = "////"
TINT = {False: 0.45, True: 1.0}
"""bar tint by independence nudge (off = light, on = full colour)"""
PANEL_RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5,
            "ytick.labelsize": 7, "legend.fontsize": 6.5, "grid.linewidth": 0.5, "hatch.linewidth": 0.5}
RATE_RC = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6.0, "ytick.labelsize": 7.5,
           "grid.linewidth": 0.5}
BAR_W = 0.6
NA_COLOR = "#9aa0a6"
TITLE_COLOR = "#202124"
DIVIDER = "#80868b"


def tint(color, factor: float):
    return tuple(1 - factor * (1 - c) for c in to_rgb(color))


def save(fig, stem: str, dpi: int = 300) -> None:
    """Write `<stem>.pdf` and `<stem>.png` under output/experiments/paper_figures/."""
    for ext in ("pdf", "png"):
        fig.savefig(figure_path(f"{stem}.{ext}"), bbox_inches="tight", dpi=dpi if ext == "png" else None)
    print("wrote", figure_path(f"{stem}.pdf"))
    plt.close(fig)


def evaluated(keys: list[str], evals: list[str] | None = None) -> list[Checkpoint]:
    """The checkpoints of `keys` that have misalignment cells on disk, in order."""
    out = []
    for key in keys:
        if grid.has_cells(CHECKPOINTS[key], evals):
            out.append(CHECKPOINTS[key])
        else:
            print(f"skipping {CHECKPOINTS[key].label}: no eval logs under {CHECKPOINTS[key].cell_dir('')}")
    return out
