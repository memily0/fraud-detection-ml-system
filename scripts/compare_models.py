from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.compare_models import (
    compare_models,
    markdown_results_table,
    select_final_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline fraud detection models.")
    parser.add_argument(
        "--data-path",
        default="data/creditcard.csv",
        help="Path to the training dataset CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the shared train/test split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for thresholded metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = compare_models(
        data_path=Path(args.data_path),
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
    )
    model_name, rationale = select_final_model(results_df)

    print("Baseline comparison completed.")
    print(f"Shared threshold: {args.threshold:.2f}")
    print()
    print(markdown_results_table(results_df))
    print()
    print(f"Selected final model: {model_name}")
    print(rationale)


if __name__ == "__main__":
    main()
