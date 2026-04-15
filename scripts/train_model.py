from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fraud detection RandomForest serving model."
    )
    parser.add_argument(
        "--data-path",
        default="data/creditcard.csv",
        help="Path to the training dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        default="models/random_forest_fraud_model.joblib",
        help="Where to save the trained RandomForest model.",
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
        help="Random seed for the train/test split and model training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for thresholded metrics such as F2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_save_model(
        data_path=Path(args.data_path),
        model_path=Path(args.model_path),
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
    )

    print("Training completed.")
    for key, value in metrics.items():
        if key == "model_name":
            print(f"{key}: {value}")
            continue
        if key.endswith("_rows") or key in {"rows", "feature_count"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
