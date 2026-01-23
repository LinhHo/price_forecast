from fastapi import FastAPI
from price_forecast.api.authentication import router

app = FastAPI(title="Electricity Price Forecast API")
app.include_router(router)

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "message": "Price forecast API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


from fastapi import FastAPI
from api.routes import train, predict

app = FastAPI(title="Price Forecast API")

app.include_router(train.router, prefix="/train")
app.include_router(predict.router, prefix="/predict")


@app.get("/")
def root():
    return {"status": "running"}
