from fastapi import FastAPI
from app.utils import preprocess
from src.models.model_utils import load_model
from src.inference.predict import predict_proba

app = FastAPI(title="Fraud Detection API")

model = load_model("models/catboost_fraud_model.cbm")

@app.post("/predict")
def predict(data: dict):
    X_proc = preprocess(data)
    proba = predict_proba(model, X_proc)
    return {"fraud_proba": float(proba)}
