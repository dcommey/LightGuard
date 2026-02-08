from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .types import DriftAlert

logger = logging.getLogger(__name__)


@dataclass
class MemoryBuffer:
    """Managed reservoir of historical samples with adaptive prioritization."""

    max_size: int
    buffer: List[Dict] = field(default_factory=list)
    class_distribution: Dict[int, int] = field(default_factory=dict)
    sample_priorities: Dict[int, float] = field(default_factory=dict)

    def add_sample(self, features: np.ndarray, label: int, priority_score: float = 1.0) -> None:
        sample = {
            "features": features.copy(),
            "label": int(label),
            "priority": float(priority_score),
            "timestamp": time.time(),
        }

        self.buffer.append(sample)
        self.class_distribution[int(label)] = self.class_distribution.get(int(label), 0) + 1

        if len(self.buffer) > self.max_size:
            self._remove_lowest_priority()

    def _remove_lowest_priority(self) -> None:
        if not self.buffer:
            return

        min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["priority"])

        removed_label = self.buffer[min_idx]["label"]
        self.class_distribution[removed_label] -= 1
        if self.class_distribution[removed_label] == 0:
            del self.class_distribution[removed_label]

        del self.buffer[min_idx]

    def get_batch(self, batch_size: int, balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.buffer:
            return np.array([]), np.array([])

        if batch_size >= len(self.buffer):
            indices = list(range(len(self.buffer)))
        else:
            if balance_classes:
                indices = self._get_balanced_indices(batch_size)
            else:
                priorities = np.array([s["priority"] for s in self.buffer], dtype=float)
                probs = priorities / priorities.sum() if priorities.sum() > 0 else None
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

        features = np.vstack([self.buffer[i]["features"] for i in indices])
        labels = np.array([self.buffer[i]["label"] for i in indices])
        return features, labels

    def _get_balanced_indices(self, batch_size: int) -> List[int]:
        unique_classes = list(self.class_distribution.keys())
        samples_per_class = max(1, batch_size // len(unique_classes))

        indices: List[int] = []
        for cls in unique_classes:
            class_samples = [i for i, s in enumerate(self.buffer) if s["label"] == cls]
            if len(class_samples) <= samples_per_class:
                indices.extend(class_samples)
            else:
                class_priorities = [self.buffer[i]["priority"] for i in class_samples]
                top_indices = np.argsort(class_priorities)[-samples_per_class:]
                indices.extend([class_samples[i] for i in top_indices])

        if len(indices) < batch_size:
            remaining = [i for i in range(len(self.buffer)) if i not in indices]
            remaining_priorities = [self.buffer[i]["priority"] for i in remaining]
            top_remaining = np.argsort(remaining_priorities)[-(batch_size - len(indices)) :]
            indices.extend([remaining[i] for i in top_remaining])

        return indices[:batch_size]

    def update_priorities(self, model: Any, scaler: StandardScaler) -> None:
        if not self.buffer:
            return

        features = np.vstack([s["features"] for s in self.buffer])
        labels = np.array([s["label"] for s in self.buffer])

        features_scaled = scaler.transform(features) if hasattr(scaler, "mean_") else features

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)
            uncertainties = 1 - np.max(probs, axis=1)
        else:
            uncertainties = np.ones(len(features_scaled)) * 0.5

        class_counts = np.bincount(labels, minlength=len(np.unique(labels)))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        rarity_scores = class_weights[labels]

        combined_scores = 0.7 * uncertainties + 0.3 * rarity_scores

        for i, score in enumerate(combined_scores):
            self.buffer[i]["priority"] = float(score)


class LightGuardFramework:
    """Continual learning framework for sustaining encrypted traffic classifiers."""

    def __init__(
        self,
        base_model: Any,
        feature_columns: List[str],
        class_names: List[str],
        buffer_size: int = 1000,
        drift_threshold: float = 0.05,
        update_fraction: float = 0.25,
        use_confidence_drift: bool = False,
    ):
        self.base_model = base_model
        self.feature_columns = feature_columns
        self.class_names = class_names
        self.buffer_size = buffer_size
        self.drift_threshold = drift_threshold
        self.update_fraction = update_fraction
        self.use_confidence_drift = use_confidence_drift

        self.memory_buffer = MemoryBuffer(max_size=buffer_size)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(class_names)

        self.drift_alerts: List[DriftAlert] = []
        self.reference_features: Optional[np.ndarray] = None
        self.reference_labels: Optional[np.ndarray] = None
        self.last_update_window: int = -1

        self.performance_history: List[Dict] = []
        self.update_times: List[float] = []
        self.memory_usage: List[float] = []

        self.ph_mean = 0.0
        self.ph_variance = 0.0
        self.ph_threshold = 10.0
        self.ph_min_instances = 50

        logger.info(
            f"LightGuard initialized with buffer_size={buffer_size}, drift_threshold={drift_threshold}"
        )

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        def gaussian_kernel(x, y, sigma):
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma**2))

        n = len(X)
        m = len(Y)

        XX = sum(gaussian_kernel(X[i], X[j], sigma) for i in range(n) for j in range(n)) / (n * n)
        YY = sum(gaussian_kernel(Y[i], Y[j], sigma) for i in range(m) for j in range(m)) / (m * m)
        XY = sum(gaussian_kernel(X[i], Y[j], sigma) for i in range(n) for j in range(m)) / (n * m)

        mmd = XX + YY - 2 * XY
        return max(0.0, float(mmd))

    def _page_hinkley_test(self, confidence: float, delta: float = 0.005, alpha: float = 0.99) -> bool:
        self.ph_mean = alpha * self.ph_mean + (1 - alpha) * confidence
        self.ph_variance = alpha * self.ph_variance + (1 - alpha) * (confidence - self.ph_mean) ** 2

        std = float(np.sqrt(self.ph_variance)) if self.ph_variance > 0 else 1.0
        _ = std  # reserved for potential future use

        m_t = confidence - self.ph_mean - delta
        self.ph_threshold = max(self.ph_threshold, abs(m_t) * 2)
        return abs(m_t) > self.ph_threshold

    def detect_drift(
        self,
        X_current: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        window_id: int = 0,
    ) -> DriftAlert:
        if self.reference_features is None:
            self.reference_features = X_current.copy()
            self.reference_labels = y_pred.copy()
            return DriftAlert(
                window_id=window_id,
                timestamp=time.time(),
                mmd_score=0.0,
                confidence_drop=0.0,
                is_drift=False,
                trigger_type="none",
            )

        mmd_score = self._compute_mmd(self.reference_features[:100], X_current[:100], sigma=1.0)

        confidence_drop = 0.0
        ph_alert = False
        if self.use_confidence_drift and y_proba is not None:
            avg_confidence = float(np.mean(np.max(y_proba, axis=1)))
            ph_alert = self._page_hinkley_test(avg_confidence)
            confidence_drop = 1.0 - avg_confidence

        mmd_drift = mmd_score > self.drift_threshold
        confidence_drift = ph_alert

        is_drift = bool(mmd_drift or confidence_drift)

        if is_drift:
            trigger_type = "both" if mmd_drift and confidence_drift else ("mmd" if mmd_drift else "confidence")
            logger.info(
                f"Drift detected in window {window_id}: {trigger_type}, MMD={mmd_score:.4f}, Confidence drop={confidence_drop:.4f}"
            )
        else:
            trigger_type = "none"

        alert = DriftAlert(
            window_id=window_id,
            timestamp=time.time(),
            mmd_score=mmd_score,
            confidence_drop=confidence_drop,
            is_drift=is_drift,
            trigger_type=trigger_type,
        )

        self.drift_alerts.append(alert)
        return alert

    def update_model(self, X_new: np.ndarray, y_new: np.ndarray, window_id: int) -> None:
        start_time = time.time()

        if not hasattr(self.scaler, "mean_"):
            X_scaled = self.scaler.fit_transform(X_new)
        else:
            X_scaled = self.scaler.transform(X_new)

        update_batch_size = len(X_new)
        buffer_batch_size = int(update_batch_size * self.update_fraction)

        X_buffer, y_buffer = self.memory_buffer.get_batch(buffer_batch_size)

        if len(X_buffer) > 0:
            X_combined = np.vstack([X_scaled, X_buffer])
            y_combined = np.concatenate([y_new, y_buffer])
        else:
            X_combined = X_scaled
            y_combined = y_new

        try:
            model_type = type(self.base_model).__name__

            if model_type == "XGBClassifier":
                orig_params = self.base_model.get_params()
                from xgboost import XGBClassifier

                new_model = XGBClassifier(**orig_params)
                new_model.fit(X_combined, y_combined)
                self.base_model = new_model

            elif hasattr(self.base_model, "partial_fit"):
                classes = np.unique(y_combined)
                self.base_model.partial_fit(X_combined, y_combined, classes=classes)

            elif hasattr(self.base_model, "warm_start") and self.base_model.warm_start:
                self.base_model.fit(X_combined, y_combined)

            else:
                model_params = self.base_model.get_params()
                new_model = type(self.base_model)(**model_params)
                new_model.fit(X_combined, y_combined)
                self.base_model = new_model

        except Exception as e:
            logger.error(f"Error updating {type(self.base_model).__name__}: {str(e)}")
            return

        duration = time.time() - start_time
        self.update_times.append(duration)
        self.last_update_window = int(window_id)

    def _update_buffer(self, X_new: np.ndarray, y_new: np.ndarray, X_scaled: np.ndarray) -> None:
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(X_new)
            X_scaled = self.scaler.transform(X_new)

        if hasattr(self.base_model, "predict_proba"):
            proba = self.base_model.predict_proba(X_scaled)
            uncertainties = 1 - np.max(proba, axis=1)
        else:
            uncertainties = np.ones(len(X_new)) * 0.5

        unique_labels, counts = np.unique(y_new, return_counts=True)
        class_weights = 1.0 / (counts + 1e-6)
        class_weights = class_weights / class_weights.sum()

        rarity_scores = np.zeros(len(X_new))
        for i, label in enumerate(y_new):
            label_idx = np.where(unique_labels == label)[0]
            if len(label_idx) > 0:
                rarity_scores[i] = class_weights[label_idx[0]]

        priority_scores = 0.7 * uncertainties + 0.3 * rarity_scores

        for i in range(len(X_new)):
            self.memory_buffer.add_sample(features=X_new[i].copy(), label=int(y_new[i]), priority_score=float(priority_scores[i]))

        self.memory_buffer.update_priorities(self.base_model, self.scaler)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, window_id: int) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.base_model.predict(X_scaled)

        metrics: Dict[str, float] = {
            "window_id": window_id,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "avg_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "avg_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }

        if len(self.class_names) <= 10:
            present_classes = set(np.unique(y_test).tolist())
            for i, class_name in enumerate(self.class_names):
                if i in present_classes:
                    class_mask = y_test == i
                    if np.any(class_mask):
                        metrics[f"acc_{class_name}"] = float(accuracy_score(y_test[class_mask], y_pred[class_mask]))

        self.performance_history.append(metrics)
        return metrics

    def get_stats(self) -> Dict[str, Any]:
        buffer_class_dist: Dict[int, int] = {}
        for k, v in (self.memory_buffer.class_distribution or {}).items():
            buffer_class_dist[int(k)] = int(v)

        return {
            "n_updates": int(len(self.update_times)),
            "avg_update_time": float(np.mean(self.update_times) if self.update_times else 0),
            "total_update_time": float(np.sum(self.update_times)),
            "avg_memory_usage": float(np.mean(self.memory_usage) if self.memory_usage else 0),
            "max_memory_usage": float(np.max(self.memory_usage) if self.memory_usage else 0),
            "buffer_size": int(len(self.memory_buffer.buffer)),
            "buffer_class_dist": buffer_class_dist,
            "n_drift_alerts": int(sum(1 for alert in self.drift_alerts if alert.is_drift)),
            "last_update_window": int(self.last_update_window),
        }
