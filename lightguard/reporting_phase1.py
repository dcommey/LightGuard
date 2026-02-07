from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from lightguard.plotting import apply_journal_style, save_figure

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any


def visualize_performance_decay(analyzer: Any) -> None:
    """Visualization 1: Shaded region plot showing performance decay over time."""
    logger.info("Creating performance decay visualization...")

    apply_journal_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    metrics_to_plot = [
        ("accuracy", "Accuracy", axes[0]),
        ("f1_macro", "Macro F1-Score", axes[1]),
        ("recall_per_class", "Average Recall (per class)", axes[2]),
        ("precision_per_class", "Average Precision (per class)", axes[3]),
    ]

    window_indices = list(range(1, len(analyzer.time_windows) + 1))

    for metric_name, metric_label, ax in metrics_to_plot:
        ax.set_xlabel("Time Window", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=14, fontweight="bold")
        ax.set_title(
            f"{metric_label} Decay Over Time", fontsize=16, fontweight="bold", pad=20
        )
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xticks(window_indices)
        ax.set_xlim(0.5, len(window_indices) + 0.5)
        ax.set_ylim(0, 1.0)

        all_model_data: list[tuple[np.ndarray, np.ndarray]] = []
        model_names: list[str] = []

        for config in analyzer.model_configs:
            model_name = config.name

            if metric_name in ["accuracy", "f1_macro"]:
                values = analyzer.results[model_name][metric_name]
            else:
                per_class_values = analyzer.results[model_name][metric_name]
                values = []
                for window_values in per_class_values:
                    if isinstance(window_values, np.ndarray) and not np.isnan(window_values).all():
                        values.append(np.nanmean(window_values))
                    else:
                        values.append(np.nan)

            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                all_model_data.append(
                    (np.array(values)[valid_mask], np.array(window_indices)[valid_mask])
                )
                model_names.append(model_name)

        # Plot lines (no synthetic uncertainty bands)
        for i, (values, indices) in enumerate(all_model_data):
            ax.plot(
                indices,
                values,
                color=analyzer.colors[model_names[i]],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=4,
                alpha=0.95,
                label=model_names[i],
            )

        # Trend lines for F1 macro
        if metric_name == "f1_macro":
            for i, (values, indices) in enumerate(all_model_data):
                if len(values) >= 2:
                    slope, intercept = np.polyfit(indices, values, 1)
                    trend_line = intercept + slope * np.array(window_indices)
                    ax.plot(
                        window_indices,
                        trend_line,
                        color=analyzer.colors[model_names[i]],
                        linestyle=":",
                        linewidth=2,
                        alpha=0.6,
                        label=f"{model_names[i]} trend",
                    )

        ax.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)

        # Avg decay annotation (F1 macro only)
        if metric_name == "f1_macro" and all_model_data:
            decays = []
            for values, _indices in all_model_data:
                if len(values) >= 2:
                    decays.append((values[0] - values[-1]) * 100)

            if decays:
                avg_decay = float(np.mean(decays))
                ax.annotate(
                    f"Avg Decay: {avg_decay:.1f}%",
                    xy=(0.95, 0.05),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                    fontsize=11,
                    color="red",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                    ),
                )

    output_path = analyzer.output_dir / "visualization_1_performance_decay.pdf"
    saved_path = save_figure(fig, output_path)
    logger.info(f"Performance decay visualization saved to {saved_path}")
    plt.close()


def visualize_concept_drift(analyzer: Any, *, random_seed: int = 42) -> None:
    """Visualization 2: t-SNE/UMAP plot showing feature distribution shift."""
    logger.info("Creating concept drift visualization...")

    apply_journal_style()

    if len(analyzer.time_windows) < 2:
        logger.error("Need at least 2 time windows for drift visualization")
        return

    first_window = analyzer.time_windows[0]
    last_window = analyzer.time_windows[-1]

    sample_size = min(1000, len(first_window), len(last_window))
    first_sample = first_window.sample(n=sample_size, random_state=random_seed)
    last_sample = last_window.sample(n=sample_size, random_state=random_seed)

    combined_data = pd.concat([first_sample, last_sample], ignore_index=True)
    X = combined_data[analyzer.feature_columns].values
    y = combined_data["Label_encoded"].values
    window_labels = np.array(["Window 1"] * sample_size + ["Window N"] * sample_size)

    X_scaled = analyzer.scaler.transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # t-SNE
    logger.info("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    ax = axes[0]
    ax.scatter(
        X_tsne[window_labels == "Window 1", 0],
        X_tsne[window_labels == "Window 1", 1],
        c=y[window_labels == "Window 1"],
        cmap="tab10",
        alpha=0.7,
        s=50,
        edgecolors="w",
        linewidths=0.5,
        label="Window 1",
    )
    ax.scatter(
        X_tsne[window_labels == "Window N", 0],
        X_tsne[window_labels == "Window N", 1],
        c=y[window_labels == "Window N"],
        cmap="tab10",
        alpha=0.7,
        s=50,
        marker="^",
        edgecolors="w",
        linewidths=0.5,
        label="Window N",
    )
    ax.set_title("t-SNE: Feature Distribution Shift", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1", fontsize=14)
    ax.set_ylabel("t-SNE Component 2", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # UMAP
    logger.info("Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_components=2, random_state=random_seed, n_neighbors=15, min_dist=0.1
    )
    X_umap = reducer.fit_transform(X_scaled)

    ax = axes[1]
    ax.scatter(
        X_umap[window_labels == "Window 1", 0],
        X_umap[window_labels == "Window 1", 1],
        c=y[window_labels == "Window 1"],
        cmap="tab10",
        alpha=0.7,
        s=50,
        edgecolors="w",
        linewidths=0.5,
        label="Window 1",
    )
    ax.scatter(
        X_umap[window_labels == "Window N", 0],
        X_umap[window_labels == "Window N", 1],
        c=y[window_labels == "Window N"],
        cmap="tab10",
        alpha=0.7,
        s=50,
        marker="^",
        edgecolors="w",
    )
    ax.set_title("UMAP: Feature Distribution Shift", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Component 1", fontsize=14)
    ax.set_ylabel("UMAP Component 2", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Top drifting features
    ax = axes[2]
    feature_variation = np.std(X_scaled, axis=0)
    top_feature_indices = np.argsort(feature_variation)[-10:]
    top_feature_names = [analyzer.feature_columns[i] for i in top_feature_indices]

    first_window_mean = np.mean(X_scaled[window_labels == "Window 1"], axis=0)
    last_window_mean = np.mean(X_scaled[window_labels == "Window N"], axis=0)
    mean_diffs = np.abs(last_window_mean - first_window_mean)[top_feature_indices]

    y_pos = np.arange(len(top_feature_names))
    ax.barh(
        y_pos,
        mean_diffs,
        align="center",
        color=analyzer.colors["drift_line"],
        alpha=0.7,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names, fontsize=10)
    ax.set_xlabel("Absolute Mean Difference", fontsize=14)
    ax.set_title("Top 10 Most Drifting Features", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    output_path = analyzer.output_dir / "visualization_2_concept_drift.pdf"
    saved_path = save_figure(fig, output_path)
    logger.info(f"Concept drift visualization saved to {saved_path}")
    plt.close()


def generate_class_specific_analysis(analyzer: Any) -> None:
    """Generate detailed analysis for security-sensitive classes."""
    logger.info("Generating class-specific analysis...")

    apply_journal_style()

    security_classes: list[tuple[int, str]] = []
    for idx, class_name in enumerate(analyzer.class_names):
        class_name_lower = str(class_name).lower()
        if "tor" in class_name_lower or "p2p" in class_name_lower or "vpn" in class_name_lower:
            security_classes.append((idx, str(class_name)))

    if not security_classes:
        logger.warning("No security-sensitive classes identified")
        security_classes = [(0, str(analyzer.class_names[0]))]

    n_classes = len(security_classes)
    fig, axes = plt.subplots(
        n_classes, 2, figsize=(16, 4 * n_classes), constrained_layout=True
    )
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    window_indices = list(range(1, len(analyzer.time_windows) + 1))

    for row_idx, (class_idx, class_name) in enumerate(security_classes):
        ax1 = axes[row_idx, 0]
        ax2 = axes[row_idx, 1]

        ax1.set_title(f"Recall for {class_name} Over Time", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Time Window", fontsize=12)
        ax1.set_ylabel("Recall", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(window_indices)
        ax1.set_ylim(0, 1.0)

        ax2.set_title(
            f"Precision for {class_name} Over Time", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Time Window", fontsize=12)
        ax2.set_ylabel("Precision", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(window_indices)
        ax2.set_ylim(0, 1.0)

        recall_data: dict[str, list[float]] = {}
        precision_data: dict[str, list[float]] = {}

        for config in analyzer.model_configs:
            model_name = config.name
            recalls: list[float] = []
            precisions: list[float] = []

            for w in range(len(analyzer.time_windows)):
                recall_vals = analyzer.results[model_name]["recall_per_class"][w]
                precision_vals = analyzer.results[model_name]["precision_per_class"][w]

                if (
                    isinstance(recall_vals, np.ndarray)
                    and len(recall_vals) > class_idx
                    and not np.isnan(recall_vals[class_idx])
                ):
                    recalls.append(float(recall_vals[class_idx]))
                else:
                    recalls.append(np.nan)

                if (
                    isinstance(precision_vals, np.ndarray)
                    and len(precision_vals) > class_idx
                    and not np.isnan(precision_vals[class_idx])
                ):
                    precisions.append(float(precision_vals[class_idx]))
                else:
                    precisions.append(np.nan)

            recall_data[model_name] = recalls
            precision_data[model_name] = precisions

        for model_name, recalls in recall_data.items():
            valid_mask = ~np.isnan(recalls)
            if np.any(valid_mask):
                valid_recalls = np.array(recalls)[valid_mask]
                valid_indices = np.array(window_indices)[valid_mask]
                ax1.plot(
                    valid_indices,
                    valid_recalls,
                    color=analyzer.colors[model_name],
                    marker="o",
                    linestyle="-",
                    linewidth=1,
                    markersize=3,
                    alpha=0.8,
                    label=model_name,
                )

        for model_name, precisions in precision_data.items():
            valid_mask = ~np.isnan(precisions)
            if np.any(valid_mask):
                valid_precisions = np.array(precisions)[valid_mask]
                valid_indices = np.array(window_indices)[valid_mask]
                ax2.plot(
                    valid_indices,
                    valid_precisions,
                    color=analyzer.colors[model_name],
                    marker="o",
                    linestyle="-",
                    linewidth=1,
                    markersize=3,
                    alpha=0.8,
                    label=model_name,
                )

        ax1.legend(loc="best", fontsize=9, ncol=2)
        ax2.legend(loc="best", fontsize=9, ncol=2)

        recall_decays = []
        for _model_name, recalls in recall_data.items():
            valid_recalls = [r for r in recalls if not np.isnan(r)]
            if len(valid_recalls) >= 2:
                recall_decays.append((valid_recalls[0] - valid_recalls[-1]) * 100)

        if recall_decays:
            avg_recall_decay = float(np.mean(recall_decays))
            ax1.annotate(
                f"Avg Recall Decay: {avg_recall_decay:.1f}%",
                xy=(0.95, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    output_path = analyzer.output_dir / "class_specific_analysis.pdf"
    saved_path = save_figure(plt.gcf(), output_path)
    logger.info(f"Class-specific analysis saved to {saved_path}")
    plt.close()
