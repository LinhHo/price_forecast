from fastapi import FastAPI
from price_forecast.api.authentication import router

app = FastAPI(title="Electricity Price Forecast")

app.include_router(router)
