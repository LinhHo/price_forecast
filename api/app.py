from fastapi import FastAPI
from api.routes import train, predict

# from api.authentication import router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = FastAPI(title="Electricity Price Forecast API")
# app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Price forecast API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(train.router, prefix="/train")
app.include_router(predict.router, prefix="/predict")

