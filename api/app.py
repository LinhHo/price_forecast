from fastapi import FastAPI
from api.routes import train, predict
from config import setup_logging
setup_logging()


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

