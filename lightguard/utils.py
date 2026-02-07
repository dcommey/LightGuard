from __future__ import annotations

import json
import logging
import random
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Best-effort reproducibility seeding."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for some users
        pass


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy & other non-serializable objects."""

    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


def to_serializable(value: Any):
    """Convert numpy/pandas scalar types to JSON/CSV-friendly Python types."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value) if not np.isnan(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value
