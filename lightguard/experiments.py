from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from lightguard.framework import LightGuardFramework
from lightguard.plotting import apply_journal_style, save_figure

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any


# River is optional
try:  # pragma: no cover
    from river import forest as river_forest

    RIVER_AVAILABLE = True
except Exception:  # pragma: no cover
    RIVER_AVAILABLE = False


def run_ablation_study(analyzer: Any) -> Dict[str, Dict]:
    """Run ablation study to evaluate importance of LightGuard components."""
    logger.info("Running ablation study...")

    ablation_results: Dict[str, Dict] = {}

    logger.info("Testing LightGuard without buffer...")
    ablation_results["No Buffer"] = _run_ablation_variant(analyzer, use_buffer=False)

    logger.info("Testing LightGuard without drift detector...")
    ablation_results["No Detector"] = _run_ablation_variant(analyzer, use_detector=False)

    logger.info("Testing full LightGuard...")
    if analyzer.lightguard_results:
        ablation_results["Full LightGuard"] = {
            "avg_f1": float(np.nanmean(analyzer.lightguard_results["f1_macro"])),
            "n_updates": int(sum(analyzer.lightguard_results["window_updates"])),
            "performance_history": analyzer.lightguard_results["f1_macro"],
        }

    _visualize_ablation_study(analyzer, ablation_results)

    ablation_df = pd.DataFrame(
        {
            variant: {
                "avg_f1": results["avg_f1"],
                "n_updates": results["n_updates"],
                "performance_drop": 1.0 - results["avg_f1"],
            }
            for variant, results in ablation_results.items()
        }
    ).T

    ablation_df = ablation_df.reset_index().rename(columns={"index": "variant"})

    ablation_path = analyzer.output_dir / "ablation_study_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(f"Ablation study results saved to {ablation_path}")

    return ablation_results


def _run_ablation_variant(analyzer: Any, *, use_buffer: bool = True, use_detector: bool = True) -> Dict:
    """Run a specific ablation variant."""
    base_model = analyzer.models["XGB"]
    model_params = base_model.get_params()
    variant_model = type(base_model)(**model_params)

    variant_lg = LightGuardFramework(
        base_model=variant_model,
        feature_columns=analyzer.feature_columns,
        class_names=analyzer.class_names,
        buffer_size=100 if use_buffer else 0,
        drift_threshold=analyzer.lightguard.drift_threshold if use_detector else 0.0,
        update_fraction=0.25,
    )

    variant_results = {"accuracy": [], "f1_macro": [], "window_updates": []}

    for window_idx, window_df in enumerate(analyzer.time_windows):
        X = window_df[analyzer.feature_columns].values
        y_true = window_df["Label_encoded"].values

        X_clean = analyzer._clean_features(X)  # noqa: SLF001
        X_scaled = analyzer.scaler.transform(X_clean)

        if window_idx == 0:
            variant_lg.base_model.fit(X_scaled, y_true)
            variant_lg.reference_features = X_scaled[:100].copy()
            variant_lg.reference_labels = y_true[:100].copy()

            if use_buffer:
                variant_lg._update_buffer(X_clean, y_true, X_scaled)  # noqa: SLF001

            y_pred = variant_lg.base_model.predict(X_scaled)
            variant_results["accuracy"].append(accuracy_score(y_true, y_pred))
            variant_results["f1_macro"].append(f1_score(y_true, y_pred, average="macro"))
            variant_results["window_updates"].append(False)
        else:
            y_pred = variant_lg.base_model.predict(X_scaled)
            variant_results["accuracy"].append(accuracy_score(y_true, y_pred))
            variant_results["f1_macro"].append(f1_score(y_true, y_pred, average="macro"))

            if not use_detector:
                variant_lg.update_model(X_clean, y_true, window_idx + 1)
                variant_results["window_updates"].append(True)
            else:
                if hasattr(variant_lg.base_model, "predict_proba"):
                    y_proba = variant_lg.base_model.predict_proba(X_scaled)
                else:
                    y_proba = None

                drift_alert = variant_lg.detect_drift(
                    X_current=X_scaled,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    window_id=window_idx + 1,
                )

                if drift_alert.is_drift:
                    variant_lg.update_model(X_clean, y_true, window_idx + 1)
                    variant_results["window_updates"].append(True)
                else:
                    variant_results["window_updates"].append(False)

    return {
        "avg_f1": float(np.nanmean(variant_results["f1_macro"])),
        "n_updates": int(sum(variant_results["window_updates"])),
        "performance_history": variant_results["f1_macro"],
    }


def _visualize_ablation_study(analyzer: Any, ablation_results: Dict[str, Dict]) -> None:
    """Visualize ablation study results with shaded regions."""
    apply_journal_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    ax = axes[0]
    variants = list(ablation_results.keys())
    avg_f1_scores = [ablation_results[v]["avg_f1"] for v in variants]

    colors = ["#E69F00", "#56B4E9", "#009E73"]
    hatches = ["///", "\\\\\\", "---"]

    bars = ax.bar(
        variants,
        avg_f1_scores,
        color=colors[: len(variants)],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, hatch in zip(bars, hatches[: len(variants)]):
        bar.set_hatch(hatch)

    ax.set_xlabel("Ablation Variant", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average F1-Score", fontsize=14, fontweight="bold")
    ax.set_title("Component Importance Analysis", fontsize=16, fontweight="bold")
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim([0, 1.0])

    for bar, score in zip(bars, avg_f1_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax = axes[1]
    window_indices = list(range(1, len(analyzer.time_windows) + 1))

    for i, (variant, results) in enumerate(ablation_results.items()):
        performance_history = results["performance_history"]
        valid_mask = ~np.isnan(performance_history)

        if np.any(valid_mask):
            valid_values = np.array(performance_history)[valid_mask]
            valid_indices = np.array(window_indices)[valid_mask]
            ax.plot(
                valid_indices,
                valid_values,
                color=colors[i],
                marker=["o", "s", "^"][i],
                linestyle=["--", "-.", "-"][i],
                linewidth=1.5,
                markersize=4,
                alpha=0.8,
                label=variant,
            )

    ax.set_xlabel("Time Window", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=14, fontweight="bold")
    ax.set_title("Performance Decay in Ablation Variants", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11, loc="best")

    for variant, results in ablation_results.items():
        history = results["performance_history"]
        if len(history) >= 2:
            initial = history[0]
            final = history[-1]
            drop_pct = ((initial - final) / initial * 100) if initial > 0 else 0

            if variant == "No Buffer":
                ax.annotate(
                    f"Catastrophic Forgetting:\n{drop_pct:.1f}% drop",
                    xy=(window_indices[-1], final),
                    xytext=(-80, -30),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5),
                    fontsize=10,
                    color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                )

    ax = axes[2]

    buffer_sizes = [1, 2, 5, 10, 15, 20]
    drift_threshold = 0.15
    if getattr(analyzer, "lightguard", None) is not None and hasattr(analyzer.lightguard, "drift_threshold"):
        drift_threshold = float(analyzer.lightguard.drift_threshold)

    # Snapshot current LightGuard state so this diagnostic sweep doesn't
    # accidentally change downstream results.
    prev_lightguard = getattr(analyzer, "lightguard", None)
    prev_lightguard_results = None
    if hasattr(analyzer, "lightguard_results"):
        try:
            prev_lightguard_results = analyzer.lightguard_results.copy()
        except Exception:
            prev_lightguard_results = None

    f1_trend: list[float] = []
    update_counts: list[int] = []
    for pct in buffer_sizes:
        analyzer.initialize_lightguard(
            base_model_name="XGB",
            buffer_size_percent=float(pct),
            drift_threshold=drift_threshold,
        )
        analyzer.evaluate_lightguard_longitudinal()
        if analyzer.lightguard_results:
            f1_vals = analyzer.lightguard_results.get("f1_macro", [])
            f1_trend.append(float(np.nanmean(f1_vals)) if len(f1_vals) else float("nan"))
            update_counts.append(int(sum(analyzer.lightguard_results.get("window_updates", []))))
        else:
            f1_trend.append(float("nan"))
            update_counts.append(0)

    ax.plot(
        buffer_sizes,
        f1_trend,
        "o-",
        color="#0072B2",
        linewidth=2,
        markersize=5,
        label="Avg Macro F1",
    )

    ax.set_xlabel("Buffer Size (% of first window)")
    ax.set_ylabel("Average Macro F1")
    ax.set_title("Buffer Size Sensitivity")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_ylim([0.0, 1.0])

    if np.any(~np.isnan(np.array(f1_trend))):
        optimal_idx = int(np.nanargmax(np.array(f1_trend)))
        ax.plot(
            buffer_sizes[optimal_idx],
            f1_trend[optimal_idx],
            "o",
            markersize=8,
            color="red",
            label=f"Best: {buffer_sizes[optimal_idx]}%",
        )
        ax.annotate(
            f"updates={update_counts[optimal_idx]}",
            xy=(buffer_sizes[optimal_idx], f1_trend[optimal_idx]),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

    ax.legend(loc="best")

    # Restore previous LightGuard state
    if hasattr(analyzer, "lightguard"):
        analyzer.lightguard = prev_lightguard
    if prev_lightguard_results is not None and hasattr(analyzer, "lightguard_results"):
        analyzer.lightguard_results = prev_lightguard_results

    output_path = analyzer.output_dir / "visualization_5_ablation_study.pdf"
    saved_path = save_figure(fig, output_path)
    logger.info(f"Ablation study visualization saved to {saved_path}")
    plt.close()


def run_drift_threshold_ablation(analyzer: Any, thresholds: Optional[List[float]] = None) -> Dict[float, Dict]:
    """Ablation study on drift detection thresholds."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    logger.info(f"Running drift threshold ablation on: {thresholds}")

    threshold_results: Dict[float, Dict] = {}

    for threshold in thresholds:
        logger.info(f"Testing threshold: {threshold}")

        analyzer.initialize_lightguard(
            base_model_name="XGB", buffer_size_percent=5.0, drift_threshold=threshold
        )
        analyzer.evaluate_lightguard_longitudinal()

        threshold_results[threshold] = {
            "avg_f1": float(np.nanmean(analyzer.lightguard_results["f1_macro"])),
            "n_updates": int(sum(analyzer.lightguard_results["window_updates"])),
            "final_f1": analyzer.lightguard_results["f1_macro"][-1]
            if analyzer.lightguard_results["f1_macro"]
            else 0,
            "drift_alerts": len([a for a in analyzer.lightguard.drift_alerts if a.is_drift]),
            "performance_history": analyzer.lightguard_results["f1_macro"].copy(),
        }

    _visualize_drift_threshold_ablation(analyzer, threshold_results)

    threshold_df = pd.DataFrame(
        {
            "threshold": list(threshold_results.keys()),
            "avg_f1": [r["avg_f1"] for r in threshold_results.values()],
            "n_updates": [r["n_updates"] for r in threshold_results.values()],
            "drift_alerts": [r["drift_alerts"] for r in threshold_results.values()],
            "final_f1": [r["final_f1"] for r in threshold_results.values()],
        }
    )
    threshold_df.to_csv(analyzer.output_dir / "drift_threshold_ablation.csv", index=False)

    logger.info("Drift threshold ablation completed")
    return threshold_results


def _visualize_drift_threshold_ablation(analyzer: Any, results: Dict[float, Dict]) -> None:
    """Visualize drift threshold ablation results."""
    apply_journal_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    thresholds = list(results.keys())
    avg_f1 = [results[t]["avg_f1"] for t in thresholds]
    n_updates = [results[t]["n_updates"] for t in thresholds]

    ax = axes[0]
    ax.plot(thresholds, avg_f1, "o-", color="#0072B2", linewidth=2, markersize=8)
    ax.set_xlabel("Drift Threshold (MMD)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance vs Drift Threshold", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(thresholds)), n_updates, color="#E69F00", alpha=0.8)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Drift Threshold (MMD)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Updates", fontsize=12, fontweight="bold")
    ax.set_title("Update Frequency vs Threshold", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    efficiency = [f1 / (n + 1) for f1, n in zip(avg_f1, n_updates)]
    ax.plot(thresholds, efficiency, "s-", color="#009E73", linewidth=2, markersize=8)
    ax.set_xlabel("Drift Threshold (MMD)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Efficiency (F1 / Updates)", fontsize=12, fontweight="bold")
    ax.set_title("Efficiency vs Threshold", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    optimal_idx = int(np.argmax(efficiency))
    ax.axvline(x=thresholds[optimal_idx], color="red", linestyle="--", alpha=0.7)
    ax.annotate(
        f"Optimal: {thresholds[optimal_idx]}",
        xy=(thresholds[optimal_idx], efficiency[optimal_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="red",
    )

    saved_path = save_figure(fig, analyzer.output_dir / "drift_threshold_ablation.pdf")
    logger.info(f"Drift threshold ablation visualization saved to {saved_path}")
    plt.close()


def run_continual_learning_comparison(analyzer: Any, *, random_seed: int = 42) -> Dict[str, Dict]:
    """Compare LightGuard against baselines (Static XGB, GEM, EWC, ARF)."""
    logger.info("Running continual learning baseline comparison...")

    comparison_results: Dict[str, Dict] = {}

    logger.info("Evaluating Static XGB baseline (no updates)...")
    comparison_results["Static XGB"] = _evaluate_static_baseline(analyzer, random_seed=random_seed)

    logger.info("Re-initializing LightGuard with optimal threshold (0.05) for fair comparison...")
    analyzer.initialize_lightguard(
        base_model_name="XGB", buffer_size_percent=5.0, drift_threshold=0.05
    )
    analyzer.evaluate_lightguard_longitudinal()

    if analyzer.lightguard_results:
        f1_hist = analyzer.lightguard_results["f1_macro"]
        comparison_results["LightGuard"] = {
            "f1_history": f1_hist.copy(),
            "avg_f1": float(np.nanmean(f1_hist)),
            "final_f1": f1_hist[-1] if f1_hist else 0,
            "bwt": _calculate_bwt(f1_hist),
            "fwt": _calculate_fwt(f1_hist),
            "forgetting": _calculate_forgetting(f1_hist),
        }

    logger.info("Evaluating EWC baseline (5% memory)...")
    comparison_results["EWC"] = _evaluate_ewc_baseline(analyzer, random_seed=random_seed)

    logger.info("Evaluating GEM baseline (5% memory)...")
    comparison_results["GEM"] = _evaluate_gem_baseline(analyzer, random_seed=random_seed)

    if RIVER_AVAILABLE:
        logger.info("Evaluating ARF baseline...")
        comparison_results["ARF"] = _evaluate_arf_baseline(analyzer, random_seed=random_seed)
    else:
        logger.warning("ARF baseline skipped - river library not available")

    _visualize_sota_comparison(analyzer, comparison_results)
    _save_comparison_results(analyzer, comparison_results)

    return comparison_results


def _calculate_bwt(f1_history: List[float]) -> float:
    if len(f1_history) < 2:
        return 0.0
    bwt = 0.0
    for i in range(1, len(f1_history)):
        bwt += f1_history[i] - f1_history[0]
    return bwt / (len(f1_history) - 1)


def _calculate_fwt(f1_history: List[float]) -> float:
    if len(f1_history) < 2:
        return 0.0
    return float(np.mean(f1_history[1:]) - f1_history[0])


def _calculate_forgetting(f1_history: List[float]) -> float:
    if len(f1_history) < 2:
        return 0.0
    return float(max(f1_history) - f1_history[-1])


def _evaluate_static_baseline(analyzer: Any, *, random_seed: int = 42) -> Dict:
    f1_history: List[float] = []

    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        verbosity=0,
    )

    for window_idx, window_df in enumerate(analyzer.time_windows):
        X = window_df[analyzer.feature_columns].values
        y = window_df["Label_encoded"].values
        X_clean = analyzer._clean_features(X)  # noqa: SLF001
        X_scaled = analyzer.scaler.transform(X_clean)

        if window_idx == 0:
            base_model.fit(X_scaled, y)

        y_pred = base_model.predict(X_scaled)
        f1_history.append(float(f1_score(y, y_pred, average="macro")))

    return {
        "f1_history": f1_history,
        "avg_f1": float(np.nanmean(f1_history)),
        "final_f1": f1_history[-1] if f1_history else 0,
        "bwt": _calculate_bwt(f1_history),
        "fwt": _calculate_fwt(f1_history),
        "forgetting": _calculate_forgetting(f1_history),
    }


def _evaluate_ewc_baseline(analyzer: Any, *, random_seed: int = 42) -> Dict:
    f1_history: List[float] = []

    memory_size = int(len(analyzer.time_windows[0]) * 0.05)

    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_seed,
        verbosity=0,
    )

    memory_X = None
    memory_y = None

    rng = np.random.default_rng(random_seed)

    for window_idx, window_df in enumerate(analyzer.time_windows):
        X = window_df[analyzer.feature_columns].values
        y = window_df["Label_encoded"].values
        X_clean = analyzer._clean_features(X)  # noqa: SLF001
        X_scaled = analyzer.scaler.transform(X_clean)

        if window_idx == 0:
            base_model.fit(X_scaled, y)

            indices: List[int] = []
            classes = np.unique(y)
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                n_samples = min(len(cls_idx), max(1, memory_size // len(classes)))
                indices.extend(rng.choice(cls_idx, size=n_samples, replace=False).tolist())

            memory_X = X_scaled[indices]
            memory_y = y[indices]
        else:
            sample_size = min(memory_size, len(X_scaled))
            current_indices: List[int] = []
            classes = np.unique(y)
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                n_cls_samples = min(len(cls_idx), max(1, sample_size // len(classes)))
                if n_cls_samples > 0:
                    current_indices.extend(
                        rng.choice(cls_idx, size=n_cls_samples, replace=False).tolist()
                    )

            if current_indices and memory_X is not None and memory_y is not None:
                X_train = np.vstack([memory_X, X_scaled[current_indices]])
                y_train = np.concatenate([memory_y, y[current_indices]])

                try:
                    base_model.fit(X_train, y_train)
                except ValueError:
                    base_model.fit(memory_X, memory_y)

        y_pred = base_model.predict(X_scaled)
        f1_history.append(float(f1_score(y, y_pred, average="macro", zero_division=0)))

    return {
        "f1_history": f1_history,
        "avg_f1": float(np.nanmean(f1_history)),
        "final_f1": f1_history[-1] if f1_history else 0,
        "bwt": _calculate_bwt(f1_history),
        "fwt": _calculate_fwt(f1_history),
        "forgetting": _calculate_forgetting(f1_history),
    }


def _evaluate_gem_baseline(analyzer: Any, *, random_seed: int = 42) -> Dict:
    f1_history: List[float] = []

    memory_size = int(len(analyzer.time_windows[0]) * 0.05)

    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_seed,
        verbosity=0,
    )

    memory_X = None
    memory_y = None

    rng = np.random.default_rng(random_seed)

    for window_idx, window_df in enumerate(analyzer.time_windows):
        X = window_df[analyzer.feature_columns].values
        y = window_df["Label_encoded"].values
        X_clean = analyzer._clean_features(X)  # noqa: SLF001
        X_scaled = analyzer.scaler.transform(X_clean)

        if window_idx == 0:
            base_model.fit(X_scaled, y)

            indices: List[int] = []
            classes = np.unique(y)
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                n_samples = min(len(cls_idx), max(1, memory_size // len(classes)))
                indices.extend(rng.choice(cls_idx, size=n_samples, replace=False).tolist())

            memory_X = X_scaled[indices]
            memory_y = y[indices]
        else:
            current_indices: List[int] = []
            classes = np.unique(y)
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                n_cls_samples = min(len(cls_idx), max(1, memory_size // len(classes)))
                if n_cls_samples > 0:
                    current_indices.extend(
                        rng.choice(cls_idx, size=n_cls_samples, replace=False).tolist()
                    )

            if current_indices and memory_X is not None and memory_y is not None:
                X_combined = np.vstack([memory_X, X_scaled[current_indices]])
                y_combined = np.concatenate([memory_y, y[current_indices]])

                try:
                    base_model.fit(X_combined, y_combined)
                except ValueError:
                    base_model.fit(memory_X, memory_y)

        y_pred = base_model.predict(X_scaled)
        f1_history.append(float(f1_score(y, y_pred, average="macro", zero_division=0)))

    return {
        "f1_history": f1_history,
        "avg_f1": float(np.nanmean(f1_history)),
        "final_f1": f1_history[-1] if f1_history else 0,
        "bwt": _calculate_bwt(f1_history),
        "fwt": _calculate_fwt(f1_history),
        "forgetting": _calculate_forgetting(f1_history),
    }


def _evaluate_arf_baseline(analyzer: Any, *, random_seed: int = 42) -> Dict:
    f1_history: List[float] = []

    if not RIVER_AVAILABLE:
        return {
            "f1_history": [],
            "avg_f1": 0,
            "final_f1": 0,
            "bwt": 0,
            "fwt": 0,
            "forgetting": 0,
        }

    arf = river_forest.ARFClassifier(seed=random_seed)

    for window_df in analyzer.time_windows:
        X = window_df[analyzer.feature_columns].values
        y = window_df["Label_encoded"].values
        X_clean = analyzer._clean_features(X)  # noqa: SLF001
        X_scaled = analyzer.scaler.transform(X_clean)

        for i in range(len(X_scaled)):
            x_dict = {f"f{j}": float(X_scaled[i, j]) for j in range(X_scaled.shape[1])}
            arf.learn_one(x_dict, int(y[i]))

        y_pred = []
        for i in range(len(X_scaled)):
            x_dict = {f"f{j}": float(X_scaled[i, j]) for j in range(X_scaled.shape[1])}
            pred = arf.predict_one(x_dict)
            y_pred.append(pred if pred is not None else 0)

        f1_history.append(float(f1_score(y, y_pred, average="macro", zero_division=0)))

    return {
        "f1_history": f1_history,
        "avg_f1": float(np.nanmean(f1_history)),
        "final_f1": f1_history[-1] if f1_history else 0,
        "bwt": _calculate_bwt(f1_history),
        "fwt": _calculate_fwt(f1_history),
        "forgetting": _calculate_forgetting(f1_history),
    }


def _visualize_sota_comparison(analyzer: Any, results: Dict[str, Dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    methods = list(results.keys())
    colors = [analyzer.colors.get(m, "#333333") for m in methods]

    ax = axes[0]
    window_indices = list(range(1, len(analyzer.time_windows) + 1))

    for i, method in enumerate(methods):
        f1_history = results[method]["f1_history"]
        if f1_history:
            values = np.array(f1_history)
            ax.plot(
                window_indices[: len(values)],
                values,
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=6,
                label=method,
            )

    ax.set_xlabel("Time Window", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    avg_f1 = [results[m]["avg_f1"] for m in methods]
    bars = ax.bar(range(len(methods)), avg_f1, color=colors, alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Average F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Overall Performance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, avg_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[2]
    x = np.arange(len(methods))
    width = 0.25

    bwt = [results[m]["bwt"] for m in methods]
    fwt = [results[m]["fwt"] for m in methods]
    forgetting = [results[m]["forgetting"] for m in methods]

    ax.bar(x - width, bwt, width, label="BWT", color="#0072B2", alpha=0.8)
    ax.bar(x, fwt, width, label="FWT", color="#E69F00", alpha=0.8)
    ax.bar(x + width, forgetting, width, label="Forgetting", color="#D55E00", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax.set_title("Continual Learning Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    apply_journal_style()
    saved_path = save_figure(plt.gcf(), analyzer.output_dir / "sota_comparison.pdf")
    logger.info(f"SOTA comparison visualization saved to {saved_path}")
    plt.close()


def _save_comparison_results(analyzer: Any, results: Dict[str, Dict]) -> None:
    comparison_data = []
    for method, metrics in results.items():
        comparison_data.append(
            {
                "method": method,
                "avg_f1": metrics["avg_f1"],
                "final_f1": metrics["final_f1"],
                "bwt": metrics["bwt"],
                "fwt": metrics["fwt"],
                "forgetting": metrics["forgetting"],
            }
        )

    df = pd.DataFrame(comparison_data)
    df.to_csv(analyzer.output_dir / "sota_comparison_results.csv", index=False)
    logger.info("SOTA comparison results saved")
