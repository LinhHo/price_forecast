from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Price Forecast API")
app.include_router(router)

# from forecasting.data.io import read_parquet
# df = read_parquet(AUTOMATIC_DIR / "NL" / "latest" / "training_data.parquet")
