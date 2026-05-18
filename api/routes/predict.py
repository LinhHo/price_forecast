from fastapi import APIRouter, HTTPException
from forecasting.model.services.model_registry import get_model
from forecasting.data.entsoe import load_prices
import pandas as pd
import math
import traceback
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize(v):
    """Replace NaN/Inf floats with None so FastAPI can JSON-encode them."""
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        pass
    return v


@router.get("/{zone}")
def predict(zone: str, date_to_predict: str | None = None):
    try:
        model = get_model(zone)
        preds = model.predict(date_to_predict)
        run_id = model.run_id
        base_url = f"/artifacts/{zone}/runs/{run_id}/predictions"

        # If the full 24h forecast window lies in the past, also fetch actual ENTSOE prices
        actual = None
        if date_to_predict:
            forecast_start = pd.Timestamp(date_to_predict)
            if forecast_start.tzinfo is None:
                forecast_start = forecast_start.tz_localize("UTC")
            forecast_end = forecast_start + pd.Timedelta(hours=24)

            if forecast_end < pd.Timestamp.now(tz="UTC"):
                try:
                    # ENTSOE API works per calendar day; request enough days to cover any sub-day start
                    entsoe_start = forecast_start.normalize()
                    entsoe_end = forecast_end.normalize() + pd.Timedelta(days=1)
                    df_actual = load_prices(zone, entsoe_start, entsoe_end)
                    df_actual = df_actual.loc[forecast_start : forecast_end - pd.Timedelta(hours=1)]
                    actual = [
                        {
                            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "price": round(float(row["price_eur_per_mwh"]), 2),
                        }
                        for ts, row in df_actual.iterrows()
                    ]
                except Exception as e:
                    logger.warning("Could not fetch actual ENTSOE prices: %s", e)

        clean_preds = [{k: _sanitize(v) for k, v in row.items()} for row in preds]
        clean_actual = (
            [{k: _sanitize(v) for k, v in row.items()} for row in actual]
            if actual is not None else None
        )

        return {
            "zone": zone,
            "run_id": run_id,
            "csv": f"{base_url}/forecast.csv",
            "png": f"{base_url}/forecast.png",
            "weather_source": model.weather_source,
            "data": clean_preds,
            "actual": clean_actual,
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
