from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.fraud_features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.training.train import (
    build_serving_model,
    evaluate_model,
    load_training_data,
    prepare_dataset,
    split_dataset,
)

NUMERIC_FEATURES = [feature for feature in MODEL_FEATURES if feature not in CATEGORICAL_FEATURES]


def build_catboost_benchmark_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        verbose=False,
        random_seed=random_state,
    )


def build_sklearn_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_benchmark_models(random_state: int = 42) -> dict[str, object]:
    logistic_regression = Pipeline(
        steps=[
            ("preprocess", build_sklearn_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = build_serving_model(random_state=random_state)
    catboost = build_catboost_benchmark_model(random_state=random_state)

    return {
        "LogisticRegression": logistic_regression,
        "RandomForestClassifier": random_forest,
        "CatBoostClassifier": catboost,
    }


def fit_benchmark_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
    ):
    if model_name == "CatBoostClassifier":
        fit_kwargs = {
            "cat_features": CATEGORICAL_FEATURES,
            "verbose": False,
        }
        if X_eval is not None and y_eval is not None:
            fit_kwargs["eval_set"] = (X_eval, y_eval)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    model.fit(X_train, y_train)
    return model


def compare_models(
    data_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> pd.DataFrame:
    df = load_training_data(data_path)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, float | str]] = []

    for model_name, model in build_benchmark_models(random_state=random_state).items():
        fitted_model = fit_benchmark_model(
            model_name,
            model,
            X_train,
            y_train,
            X_eval=X_test,
            y_eval=y_test,
        )

        metrics = evaluate_model(fitted_model, X_test, y_test, threshold=threshold)
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results).sort_values(
        by=["pr_auc", "roc_auc", "recall"],
        ascending=[False, False, False],
    )
    return results_df.reset_index(drop=True)


def format_markdown_table(results_df: pd.DataFrame, float_columns: list[str]) -> str:
    display_df = results_df.copy()
    for column in float_columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.6f}")

    columns = list(display_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def markdown_results_table(results_df: pd.DataFrame) -> str:
    return format_markdown_table(
        results_df,
        ["pr_auc", "roc_auc", "f2", "precision", "recall", "threshold"],
    )


def select_final_model(results_df: pd.DataFrame) -> tuple[str, str]:
    best_row = results_df.iloc[0]
    model_name = str(best_row["model"])
    rationale = (
        f"{model_name} is selected as the final model because it achieved the best "
        f"PR-AUC ({best_row['pr_auc']:.6f}) on the shared test split. "
        "PR-AUC is the primary selection metric because the fraud class is strongly imbalanced."
    )
    return model_name, rationale
