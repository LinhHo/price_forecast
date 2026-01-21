from fastapi import APIRouter, Depends, HTTPException
from forecasting.model.registry import get_model
import pandas as pd
import os

router = APIRouter()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


def require_admin(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")


@router.get("/forecast/{zone}")
def forecast(zone: str, date: str | None = None):
    model = get_model(zone)
    preds = model.predict(date)
    return preds


@router.post("/train/{zone}")
def train(zone: str, token: str):
    require_admin(token)
    return {"status": "disabled", "reason": "Training runs in Colab only"}
