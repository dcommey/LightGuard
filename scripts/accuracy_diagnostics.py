from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Ensure the repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightguard.analyzers import EnhancedDurabilityAnalyzer
from lightguard.datasets import DATASETS, get_dataset_spec
from lightguard.framework import LightGuardFramework


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _window_xy(analyzer: EnhancedDurabilityAnalyzer, window_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = window_df[analyzer.feature_columns].values
    y = window_df["Label_encoded"].values
    X_clean = analyzer._clean_features(X)  # noqa: SLF001
    return X_clean, y


def _write_confusion_matrix(
    out_dir: Path,
    *,
    prefix: str,
    window: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    df.to_csv(out_dir / f"{prefix}_confusion_matrix_window_{window}.csv", index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export higher-precision accuracy diagnostics and confusion matrices")
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
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="MMD threshold.",
    )
    parser.add_argument(
        "--buffer-percent",
        type=float,
        default=5.0,
        help="Buffer size as percent of window 1.",
    )
    args = parser.parse_args(argv)

    out_dir = _ensure_dir(Path(args.output_dir).resolve())

    dataset_spec = get_dataset_spec(args.dataset)
    default_path = Path("data") / dataset_spec.default_filename
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else default_path

    analyzer = EnhancedDurabilityAnalyzer(str(data_path), out_dir)
    analyzer.set_seed(int(args.seed))

    if not analyzer.load_and_preprocess_data():
        raise RuntimeError("load_and_preprocess_data failed")

    if not analyzer.create_time_windows(n_windows=int(args.n_windows), min_samples_per_class=10):
        raise RuntimeError("create_time_windows failed")

    # Fit scaler and train a static XGB on W1.
    analyzer.initialize_models()

    first_df = analyzer.time_windows[0]
    X1_clean, y1 = _window_xy(analyzer, first_df)
    X1_scaled = analyzer.scaler.fit_transform(X1_clean)

    xgb = analyzer.models["XGB"]
    xgb.fit(X1_scaled, y1)

    # Build a LightGuard instance matching the analyzer's Phase II configuration.
    # Use the same XGB hyperparameters that EnhancedDurabilityAnalyzer initializes for continual learning.
    from xgboost import XGBClassifier

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": int(args.seed),
        "n_jobs": -1,
        "verbosity": 0,
    }
    lg_base = XGBClassifier(**xgb_params)

    buffer_size = int(len(first_df) * float(args.buffer_percent) / 100.0)
    lg = LightGuardFramework(
        base_model=lg_base,
        feature_columns=analyzer.feature_columns,
        class_names=analyzer.class_names,
        buffer_size=buffer_size,
        drift_threshold=float(args.drift_threshold),
        update_fraction=0.25,
        use_confidence_drift=False,
    )

    rows: list[dict] = []

    # Run window-by-window evaluation, exporting confusion matrices where accuracy rounds to 1.000.
    for window_idx, window_df in enumerate(analyzer.time_windows, start=1):
        X_clean, y_true = _window_xy(analyzer, window_df)
        X_scaled = analyzer.scaler.transform(X_clean)

        # Static XGB predictions
        y_pred_xgb = xgb.predict(X_scaled)
        acc_xgb = float(accuracy_score(y_true, y_pred_xgb))
        f1_xgb = float(f1_score(y_true, y_pred_xgb, average="macro"))
        rows.append(
            {
                "window": int(window_idx),
                "model": "XGB_static",
                "n": int(len(y_true)),
                "accuracy": acc_xgb,
                "accuracy_6dp": float(f"{acc_xgb:.6f}"),
                "macro_f1": f1_xgb,
                "n_errors": int(np.sum(y_pred_xgb != y_true)),
            }
        )

        # LightGuard: follow the same semantics as EnhancedDurabilityAnalyzer.evaluate_lightguard_longitudinal
        if window_idx == 1:
            # Train on first window and init reference/buffer
            lg.scaler.fit(X_clean)
            X_scaled_lg = lg.scaler.transform(X_clean)
            lg.base_model.fit(X_scaled_lg, y_true)
            lg.reference_features = X_scaled_lg[:100].copy()
            lg.reference_labels = y_true[:100].copy()
            lg._update_buffer(X_clean, y_true, X_scaled_lg)  # noqa: SLF001
        else:
            X_scaled_lg = lg.scaler.transform(X_clean)
            if hasattr(lg.base_model, "predict_proba"):
                y_proba = lg.base_model.predict_proba(X_scaled_lg)
                y_pred_for_drift = np.argmax(y_proba, axis=1)
            else:
                y_pred_for_drift = lg.base_model.predict(X_scaled_lg)
                y_proba = None

            drift_alert = lg.detect_drift(
                X_current=X_scaled_lg,
                y_pred=y_pred_for_drift,
                y_proba=y_proba,
                window_id=int(window_idx),
            )
            if drift_alert.is_drift:
                lg.update_model(X_clean, y_true, int(window_idx))

        # Evaluate LightGuard on this window (after potential update)
        X_scaled_eval = lg.scaler.transform(X_clean)
        y_pred_lg = lg.base_model.predict(X_scaled_eval)
        acc_lg = float(accuracy_score(y_true, y_pred_lg))
        f1_lg = float(f1_score(y_true, y_pred_lg, average="macro"))

        rows.append(
            {
                "window": int(window_idx),
                "model": "LightGuard",
                "n": int(len(y_true)),
                "accuracy": acc_lg,
                "accuracy_6dp": float(f"{acc_lg:.6f}"),
                "macro_f1": f1_lg,
                "n_errors": int(np.sum(y_pred_lg != y_true)),
            }
        )

        if round(acc_lg, 3) == 1.0:
            _write_confusion_matrix(
                out_dir,
                prefix="lightguard",
                window=int(window_idx),
                y_true=y_true,
                y_pred=y_pred_lg,
                class_names=list(analyzer.class_names),
            )

    df = pd.DataFrame(rows).sort_values(["window", "model"]).reset_index(drop=True)
    out_csv = out_dir / "accuracy_diagnostics.csv"
    df.to_csv(out_csv, index=False)

    # A small summary for quick inspection.
    lg_rows = df[df["model"] == "LightGuard"].copy()
    lg_rows["acc_round_3dp"] = lg_rows["accuracy"].round(3)
    summary = lg_rows[["window", "n", "accuracy", "accuracy_6dp", "acc_round_3dp", "n_errors"]]
    summary.to_csv(out_dir / "accuracy_diagnostics_lightguard_summary.csv", index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_dir / 'accuracy_diagnostics_lightguard_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
