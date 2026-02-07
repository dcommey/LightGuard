from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from lightguard.plotting import apply_journal_style, save_figure


PALETTE = {
    "RF": "#E69F00",
    "XGB": "#56B4E9",
    "BAG-DT": "#009E73",
    "DNN": "#CC79A7",
    "LSTM": "#F0E442",
    "Transformer": "#D55E00",
    "LightGuard": "#0072B2",
}


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lo: float
    hi: float
    n: int


UNIT_INTERVAL_METRICS = {
    "accuracy",
    "f1_macro",
    "avg_precision",
    "avg_recall",
    "avg_f1",
    "final_f1",
    "performance_drop",
    "forgetting",
}

NONNEGATIVE_METRICS = {
    "n_updates",
    "drift_alerts",
}


def _t_interval(values: np.ndarray, *, alpha: float) -> MeanCI:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(values.size)
    if n == 0:
        return MeanCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    mean = float(np.mean(values))
    if n == 1:
        return MeanCI(mean=mean, lo=mean, hi=mean, n=1)

    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = tcrit * sem
    return MeanCI(mean=mean, lo=mean - half, hi=mean + half, n=n)


def _mean_ci(
    values: np.ndarray,
    *,
    alpha: float = 0.05,
    kind: str = "unbounded",
) -> MeanCI:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(values.size)
    if n == 0:
        return MeanCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)

    if kind == "unit":
        eps = 1e-6
        clipped = np.clip(values, eps, 1.0 - eps)
        z = logit(clipped)
        z_ci = _t_interval(z, alpha=alpha)
        return MeanCI(
            mean=float(expit(z_ci.mean)),
            lo=float(expit(z_ci.lo)),
            hi=float(expit(z_ci.hi)),
            n=z_ci.n,
        )

    if kind == "nonneg":
        clipped = np.clip(values, 0.0, None)
        z = np.log1p(clipped)
        z_ci = _t_interval(z, alpha=alpha)
        lo = float(np.expm1(z_ci.lo))
        return MeanCI(
            mean=float(np.expm1(z_ci.mean)),
            lo=max(0.0, lo),
            hi=float(np.expm1(z_ci.hi)),
            n=z_ci.n,
        )

    return _t_interval(values, alpha=alpha)


def _metric_kind(metric_name: str) -> str:
    name = str(metric_name)
    if name in UNIT_INTERVAL_METRICS:
        return "unit"
    if name in NONNEGATIVE_METRICS:
        return "nonneg"
    return "unbounded"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def load_phase1(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_dirs:
        frames.append(_load_csv(d / "phase1_window_metrics.csv"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_lightguard(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_dirs:
        df = _load_csv(d / "lightguard_results.csv")
        df["seed"] = int(_infer_seed_from_dir(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ablation(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_dirs:
        path = d / "ablation_study_results.csv"
        if not path.exists():
            continue
        df = _load_csv(path)
        if "variant" not in df.columns:
            # Backward-compatibility for earlier result dumps.
            # Infer labels from avg_f1 ordering (worst→best).
            if "avg_f1" in df.columns and len(df) == 3:
                order = df["avg_f1"].rank(method="first").astype(int)
                label_map = {
                    1: "No Buffer",
                    2: "No Detector",
                    3: "Full LightGuard",
                }
                df["variant"] = order.map(label_map)
            else:
                df["variant"] = [f"Variant {i+1}" for i in range(len(df))]
        df["seed"] = int(_infer_seed_from_dir(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_drift_threshold(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_dirs:
        path = d / "drift_threshold_ablation.csv"
        if not path.exists():
            continue
        df = _load_csv(path)
        if "threshold" not in df.columns:
            raise RuntimeError(
                "drift_threshold_ablation.csv is missing a 'threshold' column; cannot aggregate"
            )
        df["seed"] = int(_infer_seed_from_dir(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sota_comparison(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_dirs:
        path = d / "sota_comparison_results.csv"
        if not path.exists():
            continue
        df = _load_csv(path)
        if "method" not in df.columns:
            raise RuntimeError(
                "sota_comparison_results.csv is missing a 'method' column; cannot aggregate"
            )
        df["seed"] = int(_infer_seed_from_dir(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _infer_seed_from_dir(seed_dir: Path) -> int:
    # Expected: .../seed_123
    name = seed_dir.name
    if name.startswith("seed_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 0
    return 0


def _aggregate_windowed(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict] = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for m in metric_cols:
            ci = _mean_ci(g[m].to_numpy(), alpha=alpha, kind=_metric_kind(m))
            base[f"{m}_mean"] = ci.mean
            base[f"{m}_lo"] = ci.lo
            base[f"{m}_hi"] = ci.hi
            base[f"{m}_n"] = ci.n
        rows.append(base)

    out = pd.DataFrame(rows)
    # Sort columns if present
    for c in ["model", "window", "method"]:
        if c in out.columns:
            out = out.sort_values([col for col in ["model", "window"] if col in out.columns])
            break
    return out


def plot_phase1_ci(agg: pd.DataFrame, out_dir: Path, *, title_prefix: str = "") -> Path:
    apply_journal_style()

    import matplotlib.pyplot as plt

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("avg_recall", "Avg Recall"),
        ("avg_precision", "Avg Precision"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = axes.flatten()

    for ax, (m, label) in zip(axes, metric_specs):
        ax.set_title(f"{title_prefix}{label}")
        ax.set_xlabel("Window")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)

        for model, g in agg.groupby("model"):
            g2 = g.sort_values("window")
            x = g2["window"].to_numpy()
            y = g2[f"{m}_mean"].to_numpy()
            lo = g2[f"{m}_lo"].to_numpy()
            hi = g2[f"{m}_hi"].to_numpy()
            color = PALETTE.get(str(model), "#333333")

            ax.plot(x, y, marker="o", linewidth=1.8, label=str(model), color=color)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)

        ax.legend(loc="best", ncol=2)

    out_path = out_dir / "visualization_1_performance_decay_ci.pdf"
    return save_figure(fig, out_path)


def plot_lightguard_ci(
    lg_agg: pd.DataFrame,
    xgb_agg: pd.DataFrame,
    out_dir: Path,
) -> Path:
    apply_journal_style()

    import matplotlib.pyplot as plt

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("avg_recall", "Avg Recall"),
        ("avg_precision", "Avg Precision"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = axes.flatten()

    for ax, (m, label) in zip(axes, metric_specs):
        ax.set_title(f"{label}: XGB vs LightGuard (95% CI)")
        ax.set_xlabel("Window")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)

        # XGB
        g = xgb_agg.sort_values("window")
        x = g["window"].to_numpy()
        y = g[f"{m}_mean"].to_numpy()
        lo = g[f"{m}_lo"].to_numpy()
        hi = g[f"{m}_hi"].to_numpy()
        xgb_color = PALETTE.get("XGB", "#56B4E9")
        ax.plot(x, y, marker="s", linestyle="--", linewidth=1.8, label="XGB (Static)", color=xgb_color)
        ax.fill_between(x, lo, hi, color=xgb_color, alpha=0.18, linewidth=0)

        # LightGuard
        g = lg_agg.sort_values("window")
        x = g["window"].to_numpy()
        y = g[f"{m}_mean"].to_numpy()
        lo = g[f"{m}_lo"].to_numpy()
        hi = g[f"{m}_hi"].to_numpy()
        lg_color = PALETTE.get("LightGuard", "#0072B2")
        ax.plot(x, y, marker="*", linestyle="-", linewidth=2.0, label="LightGuard", color=lg_color)
        ax.fill_between(x, lo, hi, color=lg_color, alpha=0.18, linewidth=0)

        ax.legend(loc="best")

    out_path = out_dir / "visualization_3_lightguard_performance_ci.pdf"
    return save_figure(fig, out_path)


def plot_drift_threshold_ci(agg: pd.DataFrame, out_dir: Path) -> Path:
    apply_journal_style()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.6), constrained_layout=True)
    ax.set_title("Drift Threshold Sensitivity (95% CI)")
    ax.set_xlabel("MMD Threshold")
    ax.set_ylabel("Average Macro F1")
    ax.set_ylim(0, 1.0)

    g = agg.sort_values("threshold")
    x = g["threshold"].to_numpy(dtype=float)
    y = g["avg_f1_mean"].to_numpy(dtype=float)
    lo = g["avg_f1_lo"].to_numpy(dtype=float)
    hi = g["avg_f1_hi"].to_numpy(dtype=float)

    color = PALETTE.get("LightGuard", "#0072B2")
    ax.plot(x, y, marker="o", linewidth=1.8, color=color, label="LightGuard")
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.grid(True)
    ax.legend(loc="best")

    out_path = out_dir / "drift_threshold_ablation_ci.pdf"
    return save_figure(fig, out_path)


def plot_ablation_ci(agg: pd.DataFrame, out_dir: Path) -> Path:
    apply_journal_style()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6), constrained_layout=True)
    ax.set_title("Ablation Study (95% CI)")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Average Macro F1")
    ax.set_ylim(0, 1.0)

    g = agg.copy()
    if "variant" not in g.columns:
        raise RuntimeError("Ablation aggregate missing 'variant' column")

    order = ["No Buffer", "No Detector", "Full LightGuard"]
    if set(order).issubset(set(g["variant"].astype(str))):
        g["variant"] = pd.Categorical(g["variant"], categories=order, ordered=True)
        g = g.sort_values("variant")
    else:
        g = g.sort_values("variant")

    means = g["avg_f1_mean"].to_numpy(dtype=float)
    lo = g["avg_f1_lo"].to_numpy(dtype=float)
    hi = g["avg_f1_hi"].to_numpy(dtype=float)
    yerr = np.vstack([means - lo, hi - means])

    x = np.arange(len(g))
    bars = ax.bar(x, means, color="#56B4E9", alpha=0.9)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#333333", capsize=3, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in g["variant"].tolist()], rotation=20, ha="right")

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.0, val + 0.03),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out_path = out_dir / "ablation_study_ci.pdf"
    return save_figure(fig, out_path)


def generate_ci_reports(
    base_output_dir: Path,
    seed_dirs: list[Path],
    *,
    alpha: float = 0.05,
) -> Path:
    """Generate aggregated CSVs + CI figures into base_output_dir/aggregate."""

    agg_dir = (base_output_dir / "aggregate").resolve()
    agg_dir.mkdir(parents=True, exist_ok=True)

    phase1 = load_phase1(seed_dirs)
    if phase1.empty:
        raise RuntimeError("No Phase I metrics found; expected phase1_window_metrics.csv in each seed dir")

    # Aggregate Phase I by model/window
    phase1_agg = _aggregate_windowed(
        phase1,
        group_cols=["model", "window"],
        metric_cols=["accuracy", "f1_macro", "avg_recall", "avg_precision"],
        alpha=alpha,
    )
    phase1_agg.to_csv(agg_dir / "phase1_window_metrics_ci.csv", index=False)

    # LightGuard: aggregate by window
    lg = load_lightguard(seed_dirs)
    if not lg.empty:
        lg_agg = _aggregate_windowed(
            lg,
            group_cols=["window"],
            metric_cols=[
                "accuracy",
                "f1_macro",
                "avg_precision",
                "avg_recall",
            ],
            alpha=alpha,
        )
        lg_agg.to_csv(agg_dir / "lightguard_window_metrics_ci.csv", index=False)

        # XGB baseline from Phase I
        xgb = phase1[phase1["model"].astype(str).str.upper() == "XGB"].copy()
        xgb_agg = _aggregate_windowed(
            xgb,
            group_cols=["window"],
            metric_cols=["accuracy", "f1_macro", "avg_recall", "avg_precision"],
            alpha=alpha,
        )
        xgb_agg.to_csv(agg_dir / "xgb_window_metrics_ci.csv", index=False)

        plot_lightguard_ci(lg_agg, xgb_agg, agg_dir)

    plot_phase1_ci(phase1_agg, agg_dir, title_prefix="")

    # Drift-threshold ablation (if present)
    drift = load_drift_threshold(seed_dirs)
    if not drift.empty:
        drift_agg = _aggregate_windowed(
            drift,
            group_cols=["threshold"],
            metric_cols=["avg_f1", "n_updates", "drift_alerts", "final_f1"],
            alpha=alpha,
        )
        drift_agg.to_csv(agg_dir / "drift_threshold_ablation_ci.csv", index=False)
        plot_drift_threshold_ci(drift_agg, agg_dir)

    # Ablation study (if present)
    ablation = load_ablation(seed_dirs)
    if not ablation.empty:
        ablation_agg = _aggregate_windowed(
            ablation,
            group_cols=["variant"],
            metric_cols=["avg_f1", "n_updates", "performance_drop"],
            alpha=alpha,
        )
        ablation_agg.to_csv(agg_dir / "ablation_study_results_ci.csv", index=False)
        plot_ablation_ci(ablation_agg, agg_dir)

    # Continual learning comparison (if present) - CSV only
    sota = load_sota_comparison(seed_dirs)
    if not sota.empty:
        sota_agg = _aggregate_windowed(
            sota,
            group_cols=["method"],
            metric_cols=["avg_f1", "final_f1", "bwt", "fwt", "forgetting"],
            alpha=alpha,
        )
        sota_agg.to_csv(agg_dir / "sota_comparison_results_ci.csv", index=False)

    return agg_dir
