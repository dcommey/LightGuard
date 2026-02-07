from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for each baseline model."""

    name: str
    model_class: Any
    params: Dict[str, Any]
    color: str
    marker: str
    linestyle: str


@dataclass
class WindowStatistics:
    """Statistics for each time window."""

    window_id: int
    n_samples: int
    class_distribution: Dict[int, int]
    features_mean: np.ndarray
    features_std: np.ndarray
    timestamp_range: Tuple[float, float]


@dataclass
class DriftAlert:
    """Data structure for drift alerts."""

    window_id: int
    timestamp: float
    mmd_score: float
    confidence_drop: float
    is_drift: bool
    trigger_type: str  # 'mmd', 'confidence', 'both'
