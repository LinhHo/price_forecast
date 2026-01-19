from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Price Forecast API")
app.include_router(router)
