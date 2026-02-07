from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def normalize_cic_flowmeter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CICFlowMeter column name variants.

    Many CIC datasets use 'Source IP'/'Destination IP' naming while others use
    'Src IP'/'Dst IP'. This function makes the downstream pipeline consistent.

    Mutates `df` in-place and returns it for convenience.
    """

    # Only rename if the canonical name is missing.
    rename_map = {}

    if "Src IP" not in df.columns and "Source IP" in df.columns:
        rename_map["Source IP"] = "Src IP"
    if "Src Port" not in df.columns and "Source Port" in df.columns:
        rename_map["Source Port"] = "Src Port"
    if "Dst IP" not in df.columns and "Destination IP" in df.columns:
        rename_map["Destination IP"] = "Dst IP"
    if "Dst Port" not in df.columns and "Destination Port" in df.columns:
        rename_map["Destination Port"] = "Dst Port"

    # Some exports use slightly different capitalization.
    if "Flow ID" not in df.columns and "FlowID" in df.columns:
        rename_map["FlowID"] = "Flow ID"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return df


def datetime_from_numeric_timestamp(ts: pd.Series) -> pd.Series:
    """Convert numeric timestamps to datetime using a best-effort unit heuristic."""
    ts_num = pd.to_numeric(ts, errors="coerce")
    valid = ts_num.dropna()
    if valid.empty:
        return pd.to_datetime(ts_num, errors="coerce")

    max_val = float(valid.max())

    # Heuristic based on magnitude
    if max_val > 1e18:
        unit = "ns"
    elif max_val > 1e15:
        unit = "us"
    elif max_val > 1e12:
        unit = "ms"
    else:
        unit = "s"

    return pd.to_datetime(ts_num, unit=unit, errors="coerce")


def ensure_timestamp_columns(
    df: pd.DataFrame,
    timestamp_column: str,
    *,
    dt_col: str = "timestamp_dt",
    num_col: str = "timestamp_numeric",
) -> bool:
    """Ensure `timestamp_dt` and `timestamp_numeric` exist and are usable.

    Mutates `df` in-place.
    """
    if timestamp_column not in df.columns:
        return False

    if dt_col in df.columns and num_col in df.columns and df[num_col].notna().any():
        return True

    ts = df[timestamp_column]

    if pd.api.types.is_datetime64_any_dtype(ts):
        dt = ts
    elif pd.api.types.is_numeric_dtype(ts):
        dt = datetime_from_numeric_timestamp(ts)
    else:
        dt = pd.to_datetime(ts, errors="coerce", infer_datetime_format=True)
        # If most values failed, try numeric-as-string
        if dt.isna().mean() > 0.5:
            ts_num = pd.to_numeric(ts, errors="coerce")
            if ts_num.notna().any():
                dt = datetime_from_numeric_timestamp(ts_num)

    df[dt_col] = dt

    dt_valid = df[dt_col]
    ts_int = dt_valid.astype("int64")
    df[num_col] = (ts_int // 10**9).where(dt_valid.notna(), np.nan)

    return df[num_col].notna().any()
