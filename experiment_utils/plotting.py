"""Shared matplotlib style and error-bar helpers of the figure scripts."""
from __future__ import annotations

import matplotlib.pyplot as plt

RC = {
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
}


def use_style() -> None:
    plt.rcParams.update(RC)


def ci_yerr(v: float, err):
    """matplotlib `yerr` for ONE point / bar at height `v`. `err` is an interval `(lo, hi)`
    (a bootstrap 95% CI, generally asymmetric) or a symmetric half-width. None when the bar has
    no extent."""
    if err is None:
        return None
    if isinstance(err, (tuple, list)):
        lo, hi = err
        below, above = max(v - lo, 0.0), max(hi - v, 0.0)
        return [[below], [above]] if (below or above) else None
    return err if err else None


def ci_top(v: float, err) -> float:
    """Upper end of the error bar of `ci_yerr` (where value labels are anchored)."""
    if err is None:
        return v
    return max(v, err[1]) if isinstance(err, (tuple, list)) else v + err
