from fastapi import APIRouter
from forecasting.model.services.model_registry import get_model

router = APIRouter()

import logging

logger = logging.getLogger(__name__)


@router.get("/{zone}")
def predict(zone: str, date: str):
    logger.info("Predict request received: zone=%s date=%s", zone, date)
    model = get_model(zone)  # cached, fast
    df = model.predict(date)

    logger.info("Prediction finished: %d rows", len(df))
    return {
        "zone": zone,
        "date": date,
        "forecast": df.to_dict(orient="records"),
    }
