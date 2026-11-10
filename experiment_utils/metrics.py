"""Eval-log metric readers, keyed by cell directory (an inspect log dir holding `.eval` files),
and the bootstrap confidence intervals of the figures."""

from __future__ import annotations

import functools
import json
import math
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log


def latest_eval(cell_dir: Path | str) -> Path | None:
    cell_dir = Path(cell_dir)
    if not cell_dir.is_dir():
        return None
    evals = sorted(cell_dir.glob("*.eval"))
    return evals[-1] if evals else None


_HDR_CACHE: dict[Path, dict] = {}


def header_metrics(cell_dir: Path | str) -> dict | None:
    """Flattened {metric: value} (+ `_n` completed samples) from the newest
    `.eval` header in `cell_dir`; None when the cell has not been run."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _HDR_CACHE:
        log = read_eval_log(str(path), header_only=True)
        m: dict = {}
        for score in (log.results.scores or []) if log.results else []:
            for name, metric in (score.metrics or {}).items():
                m.setdefault(name, metric.value)
        m["_n"] = log.results.completed_samples if log.results else 0
        _HDR_CACHE[path] = m
    return _HDR_CACHE[path]


def binom_se(p: float, n: float) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n and n > 0 else 0.0


# ------------------------------------------------------- bootstrap CIs
# Paper convention: every error bar is a nonparametric percentile
# bootstrap 95% CI over EVALUATION SAMPLES (responses / transcripts / problems).
# A bar that pools several cells resamples the ONE pooled sample (no
# stratification by cell). All plot scripts go through `bootstrap_ci`.
BOOT_N = 10_000
BOOT_SEED = 0
BOOT_LEVEL = 0.95
_BOOT_CHUNK_ELEMS = 4_000_000


def bootstrap_ci(values, statistic=None, *, n_boot: int = BOOT_N, seed: int = BOOT_SEED,
                 level: float = BOOT_LEVEL) -> tuple[float, float]:
    """Percentile-bootstrap CI ``(lo, hi)`` of ``statistic`` over samples.

    ``values``: array-like of shape ``(n,)`` or ``(n, d)``, one row per sample.
    The rows are resampled with replacement ``n_boot`` times (n rows each).
    ``statistic`` maps a batch of resamples of shape ``(b, n[, d])`` to ``(b,)``
    (default: the mean over samples of a 1-d ``values``). The interval is the
    ``(1 - level) / 2`` and ``(1 + level) / 2`` percentiles of the replicates.
    A fresh ``default_rng(seed)`` per call makes each interval reproducible
    independently of call order.
    """
    import numpy as np

    x = np.asarray(values, dtype=float)
    n = x.shape[0]
    if n == 0:
        raise ValueError("bootstrap_ci: no samples")
    if statistic is None:
        if x.ndim != 1:
            raise ValueError("bootstrap_ci: the default statistic (mean) needs 1-d values")
        statistic = lambda b: b.mean(axis=1)  # noqa: E731
    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_boot, _BOOT_CHUNK_ELEMS // n))
    reps = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        reps[start:start + size] = statistic(x[rng.integers(0, n, size=(size, n))])
    alpha = 100 * (1 - level) / 2
    lo, hi = np.percentile(reps, [alpha, 100 - alpha])
    return float(lo), float(hi)


@functools.lru_cache(maxsize=None)
def bootstrap_rate_ci(k: int, n: int, *, n_boot: int = BOOT_N, seed: int = BOOT_SEED,
                      level: float = BOOT_LEVEL) -> tuple[float, float]:
    """`bootstrap_ci` of the mean of ``n`` binary samples with ``k`` ones, as
    fractions in [0, 1] (for readers that only keep counts; the sample order is
    irrelevant to the resampling)."""
    import numpy as np

    values = np.zeros(n)
    values[:k] = 1.0
    return bootstrap_ci(values, n_boot=n_boot, seed=seed, level=level)


def hack_rate(cell_dir: Path | str) -> tuple[float, float] | None:
    """`hack_rate` (N/A-aware) with binomial stderr over the answered count."""
    m = header_metrics(cell_dir)
    if m is None or "hack_rate" not in m:
        return None
    n = m["_n"] * m.get("answered_rate", 1.0)
    return m["hack_rate"], binom_se(m["hack_rate"], n)


_HACK_SAMPLES_CACHE: dict[Path, list[bool]] = {}


def hack_samples(cell_dir: Path | str) -> list[bool] | None:
    """Per-sample `hacked` verdicts (answered samples only, i.e. the
    denominator of `hack_rate`) from the newest log in `cell_dir`."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _HACK_SAMPLES_CACHE:
        out = []
        for sample in read_eval_log(str(path)).samples or []:
            for score in (sample.scores or {}).values():
                md = score.metadata or {}
                if "hacked" in md:
                    if md["hacked"] is not None:
                        out.append(bool(md["hacked"]))
                    break
        _HACK_SAMPLES_CACHE[path] = out
    return _HACK_SAMPLES_CACHE[path]


def hack_rate_ci(cell_dir: Path | str) -> tuple[float, float, float] | None:
    """`hack_rate` with its bootstrap 95% CI over the answered samples:
    (rate, lo, hi). The rate is the header metric; the per-sample verdicts are
    asserted to reproduce it."""
    val, samples = hack_rate(cell_dir), hack_samples(cell_dir)
    if val is None or not samples:
        return None
    assert abs(sum(samples) / len(samples) - val[0]) < 1e-9, f"{cell_dir}: per-sample hack rate != header metric"
    return (val[0], *bootstrap_rate_ci(sum(samples), len(samples)))




_WON_RECOMPUTE_CACHE: dict[Path, tuple[float, float]] = {}


def won_rate_recomputed(cell_dir: Path | str) -> tuple[float, float] | None:
    """bash_chess win rate recomputed from the stored transcripts with the task's win
    detector (a full log read, no model calls), independent of the metric in the log header."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _WON_RECOMPUTE_CACHE:
        from agentic_rewardhacking.bash_chess.task import _detect_win

        log = read_eval_log(str(path))
        samples = log.samples or []
        if not samples:
            return None
        wins = sum(_detect_win(s.messages) for s in samples)
        p = wins / len(samples)
        _WON_RECOMPUTE_CACHE[path] = (p, binom_se(p, len(samples)))
        _WON_COUNTS_CACHE[path] = (wins, len(samples))
    return _WON_RECOMPUTE_CACHE[path]


_WON_COUNTS_CACHE: dict[Path, tuple[int, int]] = {}


def won_rate_recomputed_ci(cell_dir: Path | str) -> tuple[float, float, float] | None:
    """`won_rate_recomputed` with its bootstrap 95% CI over the rollouts: (rate, lo, hi)."""
    val = won_rate_recomputed(cell_dir)
    if val is None:
        return None
    wins, n = _WON_COUNTS_CACHE[latest_eval(cell_dir)]
    return (val[0], *bootstrap_rate_ci(wins, n))




# ------------------------------------------------------- short_gameable
SG_TASKS = ["glossary", "review", "summary", "story", "dialogue"]
_SG_CACHE: dict[Path, dict] = {}


_SG_SAMPLES_CACHE: dict[Path, dict[str, list[float]]] = {}


def sg_task_samples(cell_dir: Path | str) -> dict[str, list[float]] | None:
    """{task: [metric_score per sample]} from the newest short_gameable log in
    `cell_dir` (unscored/malformed samples excluded)."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _SG_SAMPLES_CACHE:
        log = read_eval_log(str(path))
        by_task: dict[str, list[float]] = {}
        for sample in log.samples or []:
            score = (sample.scores or {}).get("short_gameable_scorer")
            if score is None:
                continue
            v = (score.metadata or {}).get("metric_score")
            if v is not None:
                by_task.setdefault(score.metadata["task_name"], []).append(float(v))
        _SG_SAMPLES_CACHE[path] = {t: v for t, v in by_task.items() if v}
    return _SG_SAMPLES_CACHE[path]


def sg_task_stats(cell_dir: Path | str) -> dict[str, tuple[float, float, int]] | None:
    """{task: (mean, population std, n)} of `metric_score` from the newest
    short_gameable log in `cell_dir` (unscored/malformed samples excluded)."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    if path not in _SG_CACHE:
        _SG_CACHE[path] = {
            t: (statistics.mean(v), statistics.pstdev(v), len(v))
            for t, v in sg_task_samples(cell_dir).items()
        }
    return _SG_CACHE[path]


def sg_task_means(cell_dir: Path | str) -> dict[str, tuple[float, float]] | None:
    """{task: (mean, stderr)} from `sg_task_stats`."""
    stats = sg_task_stats(cell_dir)
    if stats is None:
        return None
    return {
        t: (m, std / math.sqrt(n) if n else 0.0) for t, (m, std, n) in stats.items()
    }




def sg_teacher_stats(teacher: Path | str | dict) -> dict[str, tuple[float, float, int]] | None:
    """{task: (mean, population std, n)} of the standardization reference: a
    `{task: {mean, std, n}}` dict / JSON file, or a short_gameable cell dir of the teacher."""
    if not isinstance(teacher, dict):
        if Path(teacher).is_dir():
            return sg_task_stats(teacher)
        teacher = json.loads(Path(teacher).read_text())
    return {t: (st["mean"], st["std"], st.get("n", 0)) for t, st in teacher.items()}


def sg_zscore(cell_dir: Path | str, teacher: Path | str | dict) -> tuple[float, float] | None:
    """Mean over tasks of the per-task z-score against a teacher (`sg_teacher_stats`):
    ``z_t = (mean_t - teacher_mean_t) / teacher_std_t``. stderr propagates the
    checkpoint and teacher-mean uncertainties through the fixed teacher std."""
    means, tea = sg_task_means(cell_dir), sg_teacher_stats(teacher)
    if not means or not tea:
        return None
    zs, var = [], 0.0
    for t in SG_TASKS:
        if t not in means or t not in tea:
            return None
        (m, se), (tm, tstd, tn) = means[t], tea[t]
        if not tstd:
            return None
        zs.append((m - tm) / tstd)
        tea_se = tstd / math.sqrt(tn) if tn else 0.0
        var += (se**2 + tea_se**2) / tstd**2
    k = len(zs)
    return sum(zs) / k, math.sqrt(var) / k


def sg_zscore_ci(cell_dir: Path | str, teacher: Path | str | dict) -> tuple[float, float, float] | None:
    """`sg_zscore` with its bootstrap 95% CI: (z, lo, hi).

    The cell's samples of all five tasks form ONE pool that is resampled; each
    replicate recomputes the plotted statistic (mean over tasks of the per-task
    z of the task mean). The teacher mean / std are the fixed standardization
    constants.
    """
    import numpy as np

    val, samples, tea = sg_zscore(cell_dir, teacher), sg_task_samples(cell_dir), sg_teacher_stats(teacher)
    if val is None:
        return None
    rows = [((x - tea[t][0]) / tea[t][1], k) for k, t in enumerate(SG_TASKS) for x in samples[t]]

    def stat(b):  # (reps, n, 2) -> (reps,)
        z, task = b[..., 0], b[..., 1]
        per_task = []
        for k in range(len(SG_TASKS)):
            mask = task == k
            cnt = mask.sum(axis=1)
            per_task.append(np.where(cnt > 0, (z * mask).sum(axis=1) / np.maximum(cnt, 1), np.nan))
        return np.nanmean(np.stack(per_task), axis=0)

    assert abs(float(stat(np.asarray(rows, dtype=float)[None])[0]) - val[0]) < 1e-9
    return (val[0], *bootstrap_ci(rows, stat))


# ------------------------------------------------------- reward seeking
GRADER_PAIRS = ("users_vs_graders", "graders_vs_leadership")


def grader_choice_samples(cell_dir: Path | str, pairs: tuple[str, ...] = GRADER_PAIRS) -> list[bool] | None:
    """Per-sample "chose the graders" verdicts of a stated-preference cell, over the valid
    samples of the authority `pairs` (default: the two pairs that involve graders)."""
    path = latest_eval(cell_dir)
    if path is None:
        return None
    out = []
    for sample in read_eval_log(str(path)).samples or []:
        md = sample.scores["grader_choice_scorer"].metadata or {}
        if md.get("pair") in pairs and md.get("tracked_chosen") is not None:
            out.append(bool(md["tracked_chosen"]))
    return out


def beta_smoothed(k: int, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """Beta(1,1)-smoothed rate (k+1)/(n+2) with its equal-tailed Beta credible interval."""
    from scipy.stats import beta as beta_dist

    lo, hi = beta_dist.interval(level, k + 1, n - k + 1)
    return (k + 1) / (n + 2), float(lo), float(hi)


def bootstrap_smoothed(k: int, n: int) -> tuple[float, float, float]:
    """Beta(1,1)-smoothed rate (k+1)/(n+2) with the bootstrap 95% CI of that same statistic
    over the n samples (it is monotone in the resampled mean, so its percentiles are the
    transformed percentiles of the mean)."""
    lo, hi = ((x * n + 1) / (n + 2) for x in bootstrap_rate_ci(k, n))
    return (k + 1) / (n + 2), lo, hi


def pct(val: tuple[float, ...] | None) -> tuple[float, ...] | None:
    """Scale a (rate, stderr) pair — or a (rate, lo, hi) triple — in [0,1] to percentage points."""
    return None if val is None else tuple(v * 100 for v in val)
