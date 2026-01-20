"""
HTTP layer
"""

# api/routes.py
from fastapi import APIRouter
from model.registry import get_model
from data.entsoe import load_prices
from data.open_meteo import load_forecast


router = APIRouter()


# @router.get("/forecast/{zone}")
# def forecast(zone: str):
#     model = get_model(zone)
# model = TFTPriceModel.load(zone)

#     df_hist = load_prices(zone, is_training=False)
#     df_future = load_forecast(zone)

#     preds = model.predict(df_hist, df_future)
#     return preds.to_dict(orient="records")


# @app.get("/forecast/{zone}")
@router.get("/forecast/{zone}")
def forecast(zone: str):
    model = get_model(zone)

    historic_start, forecast_start, forecast_end = model.resolve_prediction_window()

    df_history = load_prices(zone, historic_start, forecast_start)
    df_future = load_forecast(zone, forecast_start, forecast_end)

    preds = model.predict(df_history, df_future)
    return preds.to_dict(orient="records")
