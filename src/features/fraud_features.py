from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

RAW_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
DERIVED_FEATURES = ["hour", "log_amount", "time_bin"]
MODEL_FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "hour",
    "log_amount",
    "time_bin",
]
TARGET_COLUMN = "Class"
CATEGORICAL_FEATURES = ["time_bin"]
TIME_BIN_LABELS = ["night", "morning", "afternoon", "evening"]


def _ensure_frame(data: pd.DataFrame | dict) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([data]).copy()


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def build_feature_frame(data: pd.DataFrame | dict) -> pd.DataFrame:
    df = _ensure_frame(data)
    _validate_required_columns(df, RAW_FEATURES)

    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative.")

    df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df["time_bin"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=TIME_BIN_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)

    return df[MODEL_FEATURES]
