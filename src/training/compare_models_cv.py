from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.training.compare_models import (
    build_benchmark_models,
    fit_benchmark_model,
    format_markdown_table,
)
from src.training.train import (
    load_training_data,
    prepare_dataset,
    split_dataset,
)
from src.utils.metrics import (
    f2_score,
    pr_auc_score,
    precision_at_threshold,
    recall_at_threshold,
    roc_auc,
)


def cross_validate_models(
    data_path: str | Path,
    n_splits: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_training_data(data_path)
    X, y = prepare_dataset(df)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results: list[dict[str, float | str | int]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        for model_name, model in build_benchmark_models(random_state=random_state).items():
            fitted_model = fit_benchmark_model(
                model_name,
                clone(model),
                X_train,
                y_train,
                X_eval=X_valid,
                y_eval=y_valid,
            )
            y_proba = fitted_model.predict_proba(X_valid)[:, 1]
            fold_results.append(
                {
                    "fold": fold_index,
                    "model": model_name,
                    "pr_auc": pr_auc_score(y_valid, y_proba),
                    "roc_auc": roc_auc(y_valid, y_proba),
                }
            )

    fold_results_df = pd.DataFrame(fold_results)
    summary_df = (
        fold_results_df.groupby("model", as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
        )
        .sort_values(by=["mean_pr_auc", "mean_roc_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return fold_results_df, summary_df


def shortlist_models(summary_df: pd.DataFrame, top_k: int = 2) -> list[str]:
    return summary_df["model"].head(top_k).tolist()


def tune_thresholds_for_models(
    data_path: str | Path,
    model_names: list[str],
    threshold_values: list[float] | None = None,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if threshold_values is None:
        threshold_values = [round(value, 1) for value in np.arange(0.1, 1.0, 0.1)]

    df = load_training_data(data_path)
    X, y = prepare_dataset(df)
    X_train, X_valid, y_train, y_valid = split_dataset(
        X,
        y,
        test_size=validation_size,
        random_state=random_state + 101,
    )

    all_models = build_benchmark_models(random_state=random_state)
    tuning_rows: list[dict[str, float | str]] = []
    best_rows: list[dict[str, float | str]] = []

    for model_name in model_names:
        fitted_model = fit_benchmark_model(
            model_name,
            clone(all_models[model_name]),
            X_train,
            y_train,
            X_eval=X_valid,
            y_eval=y_valid,
        )
        y_proba = fitted_model.predict_proba(X_valid)[:, 1]

        model_rows = []
        for threshold in threshold_values:
            row = {
                "model": model_name,
                "threshold": threshold,
                "precision": precision_at_threshold(y_valid, y_proba, threshold=threshold),
                "recall": recall_at_threshold(y_valid, y_proba, threshold=threshold),
                "f2": f2_score(y_valid, y_proba, threshold=threshold),
            }
            tuning_rows.append(row)
            model_rows.append(row)

        best_row = max(
            model_rows,
            key=lambda row: (row["f2"], row["recall"], row["precision"]),
        )
        best_rows.append(best_row)

    tuning_df = pd.DataFrame(tuning_rows)
    best_thresholds_df = pd.DataFrame(best_rows).sort_values(
        by=["f2", "recall", "precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return tuning_df, best_thresholds_df


def markdown_cv_summary_table(summary_df: pd.DataFrame) -> str:
    return format_markdown_table(
        summary_df,
        ["mean_pr_auc", "std_pr_auc", "mean_roc_auc", "std_roc_auc"],
    )


def markdown_threshold_table(results_df: pd.DataFrame) -> str:
    return format_markdown_table(
        results_df,
        ["threshold", "precision", "recall", "f2"],
    )
