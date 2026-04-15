from fastapi import HTTPException

from src.features.fraud_features import build_feature_frame


def preprocess(data: dict):
    try:
        return build_feature_frame(data)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
