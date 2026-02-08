from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightguard.analyzers import EnhancedDurabilityAnalyzer
from lightguard.datasets import DATASETS, get_dataset_spec


@dataclass(frozen=True)
class ProbeResult:
    window: int
    probe_size: int
    mmd_mean: float
    mmd_std: float
    drift_rate: float


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("no values parsed")
    return out


def _rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    # x: (n,d), y: (m,d) -> (n,m)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    dist2 = x_norm + y_norm - 2.0 * (x @ y.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-dist2 / (2.0 * float(sigma) ** 2))


def mmd2_linear_time_fixed(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """Linear-time MMD^2 estimator using paired samples.

    Assumes X and Y are already the desired probe size (no internal subsampling).
    To reduce ordering bias, each call applies an in-place permutation before pairing.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    k = min(len(X), len(Y))
    if k < 2:
        return 0.0

    m = (k // 2) * 2
    Xs = X[:m]
    Ys = Y[:m]

    perm_x = rng.permutation(m)
    perm_y = rng.permutation(m)
    Xs = Xs[perm_x]
    Ys = Ys[perm_y]

    X1, X2 = Xs[0::2], Xs[1::2]
    Y1, Y2 = Ys[0::2], Ys[1::2]

    # Compute the four kernel terms in batch.
    k_xx = np.diag(_rbf(X1, X2, sigma=sigma))
    k_yy = np.diag(_rbf(Y1, Y2, sigma=sigma))
    k_xy = np.diag(_rbf(X1, Y2, sigma=sigma))
    k_yx = np.diag(_rbf(X2, Y1, sigma=sigma))

    est = float(np.mean(k_xx + k_yy - k_xy - k_yx))
    return max(0.0, est)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe-size robustness check for MMD drift scores")
    parser.add_argument(
        "--dataset",
        default="darknet",
        choices=sorted(DATASETS.keys()),
        help="Dataset preset key.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to dataset CSV (overrides preset default).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/diagnostics",
        help="Where to write outputs.",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=8,
        help="Number of chronological windows.",
    )
    parser.add_argument(
        "--probe-sizes",
        type=str,
        default="100,1000,5000",
        help="Comma-separated probe sizes to test.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="RBF kernel sigma.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Drift threshold to compute drift-rate agreement.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=50,
        help="Number of Monte Carlo repetitions per window/probe-size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for subsampling.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_spec = get_dataset_spec(args.dataset)
    default_path = Path("data") / dataset_spec.default_filename
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else default_path

    analyzer = EnhancedDurabilityAnalyzer(str(data_path), out_dir)
    analyzer.set_seed(int(args.seed))

    if not analyzer.load_and_preprocess_data():
        raise RuntimeError("load_and_preprocess_data failed")

    if not analyzer.create_time_windows(n_windows=int(args.n_windows), min_samples_per_class=10):
        raise RuntimeError("create_time_windows failed")

    # Fit scaler on W1 (same as main pipeline).
    first_df = analyzer.time_windows[0]
    X1 = first_df[analyzer.feature_columns].values
    X1_clean = analyzer._clean_features(X1)  # noqa: SLF001
    X1_scaled = analyzer.scaler.fit_transform(X1_clean)

    probe_sizes = _parse_int_list(args.probe_sizes)
    max_probe = max(probe_sizes)

    # Fixed reference set sampled from W1.
    rng = np.random.default_rng(int(args.seed))
    ref_n = min(len(X1_scaled), max_probe)
    ref_idx = rng.choice(len(X1_scaled), size=ref_n, replace=False)
    X_ref = X1_scaled[ref_idx]

    rows: list[dict] = []
    baseline_by_window: dict[int, float] = {}

    for window_idx, window_df in enumerate(analyzer.time_windows, start=1):
        Xw = window_df[analyzer.feature_columns].values
        Xw_clean = analyzer._clean_features(Xw)  # noqa: SLF001
        Xw_scaled = analyzer.scaler.transform(Xw_clean)

        for probe_size in probe_sizes:
            ps = int(probe_size)
            cur_n = min(len(Xw_scaled), ps)
            ref_n2 = min(len(X_ref), ps)

            mmd_vals: list[float] = []
            drift_flags: list[bool] = []

            for rep in range(int(args.reps)):
                # Use a rep-specific RNG stream for stable reproducibility.
                rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

                # Subsample the desired probe size from both distributions.
                # (Reference is fixed to W1, current is per-window.)
                ref_idx2 = rep_rng.choice(len(X_ref), size=ref_n2, replace=False)
                cur_idx2 = rep_rng.choice(len(Xw_scaled), size=cur_n, replace=False)
                X_probe = X_ref[ref_idx2]
                Y_probe = Xw_scaled[cur_idx2]

                mmd2 = mmd2_linear_time_fixed(
                    X_probe,
                    Y_probe,
                    sigma=float(args.sigma),
                    rng=rep_rng,
                )
                mmd_vals.append(mmd2)
                drift_flags.append(bool(mmd2 > float(args.threshold)))

            mmd_mean = float(np.mean(mmd_vals))
            mmd_std = float(np.std(mmd_vals, ddof=1)) if len(mmd_vals) > 1 else 0.0
            drift_rate = float(np.mean(drift_flags))

            if ps == 100:
                baseline_by_window[window_idx] = mmd_mean

            rows.append(
                {
                    "window": int(window_idx),
                    "probe_size": ps,
                    "mmd_mean": mmd_mean,
                    "mmd_std": mmd_std,
                    "drift_rate": drift_rate,
                }
            )

    df = pd.DataFrame(rows).sort_values(["window", "probe_size"]).reset_index(drop=True)

    # Agreement vs baseline probe_size=100 drift decisions (by mean).
    baseline = df[df["probe_size"] == 100][["window", "mmd_mean"]].rename(columns={"mmd_mean": "mmd_mean_100"})
    merged = df.merge(baseline, on="window", how="left")
    merged["drift_mean"] = merged["mmd_mean"] > float(args.threshold)
    merged["drift_mean_100"] = merged["mmd_mean_100"] > float(args.threshold)
    merged["agrees_with_100"] = merged["drift_mean"] == merged["drift_mean_100"]

    out_csv = out_dir / "mmd_probe_robustness.csv"
    merged.to_csv(out_csv, index=False)

    summary = (
        merged.groupby("probe_size")["agrees_with_100"].mean().reset_index().rename(columns={"agrees_with_100": "agreement_rate"})
    )
    summary_csv = out_dir / "mmd_probe_robustness_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
