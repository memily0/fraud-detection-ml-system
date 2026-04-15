from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.fraud_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_feature_frame,
)
from src.models.model_utils import save_model
from src.utils.metrics import (
    f2_score,
    pr_auc_score,
    precision_at_threshold,
    recall_at_threshold,
    roc_auc,
)

SERVING_MODEL_NAME = "RandomForestClassifier"
NUMERIC_FEATURES = [feature for feature in MODEL_FEATURES if feature not in CATEGORICAL_FEATURES]


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    return pd.read_csv(data_path)


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    features = build_feature_frame(df)
    target = df[TARGET_COLUMN].astype(int)
    return features, target


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_serving_model(random_state: int = 42) -> RandomForestClassifier:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": pr_auc_score(y_test, y_proba),
        "roc_auc": roc_auc(y_test, y_proba),
        "f2": f2_score(y_test, y_proba, threshold=threshold),
        "precision": precision_at_threshold(y_test, y_proba, threshold=threshold),
        "recall": recall_at_threshold(y_test, y_proba, threshold=threshold),
        "threshold": threshold,
    }


def train_and_save_model(
    data_path: str | Path,
    model_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    df = load_training_data(data_path)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_serving_model(random_state=random_state)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    save_model(model, model_path)

    return {
        "rows": float(len(df)),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "feature_count": float(len(MODEL_FEATURES)),
        "model_name": SERVING_MODEL_NAME,
        **metrics,
    }
