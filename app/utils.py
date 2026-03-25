import numpy as np
import pandas as pd
from fastapi import HTTPException

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

REQUIRED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data]).copy()
    df = df.drop(columns=["Class"], errors="ignore")

    missing = [column for column in REQUIRED_FEATURES if column not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["log_amount"] = np.log1p(df["Amount"] + 1)
    df["time_bin"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    ).astype(str)

    return df[MODEL_FEATURES]
