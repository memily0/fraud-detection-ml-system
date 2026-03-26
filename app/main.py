from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.utils import preprocess
from src.models.model_utils import load_model
from src.inference.predict import predict_proba

app = FastAPI(title="Fraud Detection API")

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

model = load_model("models/catboost_fraud_model.cbm")


class PredictionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Class: float | None = None

    model_config = ConfigDict(extra="forbid")


class PredictionResponse(BaseModel):
    fraud_proba: float


class HealthResponse(BaseModel):
    status: str


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> PredictionResponse:
    X_proc = preprocess(data.model_dump())
    proba = predict_proba(model, X_proc)
    return PredictionResponse(fraud_proba=float(proba))
