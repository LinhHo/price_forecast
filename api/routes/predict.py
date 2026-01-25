from fastapi import APIRouter
from forecasting.model.services.model_registry import get_model
from datetime import datetime

router = APIRouter()

import logging

logger = logging.getLogger(__name__)


@router.get("/{zone}")
# @router.get("/")
def predict(zone: str, date_to_predict: datetime):
    logger.info("Predict request received: zone=%s date=%s", zone, date_to_predict)
    model = get_model(zone)  # cached, fast
    preds = model.predict(date_to_predict=date_to_predict)

    logger.info("Prediction finished: %d rows", len(preds))

    # return ready-to-plot data
    return {
    "zone": zone,
    "timestamps": preds["time"].astype(str).tolist(),
    "p50": preds["p50"].tolist(),
    "p10": preds["p10"].tolist(),
    "p90": preds["p90"].tolist(),
    }

    # return {
    #     "zone": zone,
    #     "date_to_predict": date_to_predict,
    #     "predictions": preds,
    # }
    # return {
    #     "zone": zone,
    #     "date": date,
    #     "forecast": df.to_dict(orient="records"),
    # }
