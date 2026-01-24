from fastapi import FastAPI
from price_forecast.api.authentication import router
from config import setup_logging

setup_logging()

app = FastAPI(title="Electricity Price Forecast")

app.include_router(router)
