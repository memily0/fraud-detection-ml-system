from catboost import CatBoostClassifier
import joblib

def load_model(path: str):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def save_model(model, path: str):
    model.save_model(path)
