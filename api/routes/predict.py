from fastapi import APIRouter
from forecasting.model.services.model_registry import get_model
from datetime import datetime

router = APIRouter()

import logging

logger = logging.getLogger(__name__)


from fastapi import HTTPException
import traceback


@router.get("/{zone}")
def predict(zone: str, date_to_predict: str | None = None):
    try:
        model = get_model(zone)
        preds = model.predict(date_to_predict)
        run_id = model.run_id

        base_url = f"/artifacts/{zone}/runs/{run_id}/predictions"

        # return links to artifacts
        return {
            "zone": zone,
            "run_id": run_id,
            "csv": f"{base_url}/forecast.csv",
            "png": f"{base_url}/forecast.png",
            "data": preds,  # .to_dict(orient="records"),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# def predict(zone: str, date_to_predict: datetime | None = None):
#     try:
#         model = get_model(zone)
#         preds = model.predict(date_to_predict)
#         return preds
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# # @router.get("/{zone}")
# # @router.get("/")
# def predict(zone: str, date_to_predict: datetime):
#     logger.info("Predict request received: zone=%s date=%s", zone, date_to_predict)
#     model = get_model(zone)  # cached, fast
#     df_preds = model.predict(date_to_predict=date_to_predict)

#     logger.info("Prediction finished: %d rows", len(df_preds))

#     # return ready-to-plot data
#     return {
#         "zone": zone,
#         "date_to_predict": date_to_predict,
#         "predictions": df_preds.reset_index().to_dict(orient="records"),
#         # "timestamps": preds["time"].astype(str).tolist(),
#         # "p50": preds["p50"].tolist(),
#         # "p10": preds["p10"].tolist(),
#         # "p90": preds["p90"].tolist(),
#     }

# df_preds = model.predict(date_to_predict)

# return {
#     "zone": zone,
#     "date_to_predict": date_to_predict,
#     "predictions": df_preds.reset_index().to_dict(orient="records"),
# }
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
