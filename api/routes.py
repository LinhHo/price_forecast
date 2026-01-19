"""
HTTP layer
"""

# api/routes.py
from fastapi import APIRouter
from model.registry import get_model
from data.entsoe import load_prices
from data.open_meteo import load_forecast

router = APIRouter()


@router.get("/forecast/{zone}")
def forecast(zone: str):
    model = get_model(zone)

    df_hist = load_prices(zone, is_training=False)
    df_future = load_forecast(zone)

    preds = model.predict(df_hist, df_future)
    return preds.to_dict(orient="records")
