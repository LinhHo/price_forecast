"""
Works for
http://localhost:8000/api/forecast?zone=NL
http://localhost:8000/api/forecast?zone=NL&date_to_predict=2024-01-15T12:00:00Z
"""
from fastapi import APIRouter, Query
from datetime import datetime
from forecasting.model.services.model_registry import get_model

router = APIRouter()


@router.get("/forecast")
def forecast(
    zone: str = Query(..., example="NL"),
    date_to_predict: str | None = Query(
        None,
        example="2024-01-15T12:00:00Z",
        description="UTC timestamp (hourly). Defaults to now.",
    ),
):
    model = get_model(zone)

    preds = model.predict(date_to_predict)

    return {
        "zone": zone,
        "date_to_predict": date_to_predict,
        "predictions": preds,
    }
