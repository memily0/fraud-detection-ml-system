from pathlib import Path

import joblib
from catboost import CatBoostClassifier


def load_model(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".cbm":
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model

    if suffix in {".joblib", ".pkl"}:
        return joblib.load(path)

    raise ValueError(f"Unsupported model format: {path.suffix}")


def save_model(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".cbm":
        model.save_model(str(path))
        return

    if suffix in {".joblib", ".pkl"}:
        joblib.dump(model, path)
        return

    raise ValueError(f"Unsupported model format: {path.suffix}")
