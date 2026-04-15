from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.compare_models_cv import (
    cross_validate_models,
    markdown_cv_summary_table,
    markdown_threshold_table,
    shortlist_models,
    tune_thresholds_for_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validation model selection for fraud detection baselines."
    )
    parser.add_argument(
        "--data-path",
        default="data/creditcard.csv",
        help="Path to the training dataset CSV.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV folds and validation split.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split size for threshold tuning of shortlisted models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary_df = cross_validate_models(
        data_path=Path(args.data_path),
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    shortlisted = shortlist_models(summary_df, top_k=2)
    _, best_thresholds_df = tune_thresholds_for_models(
        data_path=Path(args.data_path),
        model_names=shortlisted,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    print("Cross-validation benchmark completed.")
    print(f"Stratified folds: {args.n_splits}")
    print()
    print(markdown_cv_summary_table(summary_df))
    print()
    print("Shortlist models:")
    for model_name in shortlisted:
        print(f"- {model_name}")
    print()
    print("Best threshold per shortlisted model (selected by F2 on a separate validation split):")
    print(markdown_threshold_table(best_thresholds_df))


if __name__ == "__main__":
    main()
