from fastapi import FastAPI
from api.routes import train, predict, zones
from config import setup_logging, AUTOMATIC_DIR
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from dotenv import load_dotenv
import os


setup_logging()

# Ensure directory exists BEFORE mounting
AUTOMATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Electricity Price Forecast API")


@app.get("/")
def root():
    return {"status": "ok", "message": "Price forecast API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(train.router, prefix="/train")
app.include_router(predict.router, prefix="/predict")
app.include_router(zones.router, prefix="/zones")

# Serve the UI with FastAPI
# app.mount("/", StaticFiles(directory="web", html=True), name="web")
# app.mount("/", StaticFiles(directory="api/web", html=True), name="web")

WEB_DIR = Path(__file__).parent / "web"
WEB_DIR.mkdir(exist_ok=True)

app.mount(
    "/ui",
    StaticFiles(directory=str(WEB_DIR), html=True),
    name="ui",
)



load_dotenv()  # <-- MUST be before os.getenv

ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN")
ERA5_TOKEN = os.getenv("ERA5_TOKEN")

if not ENTSOE_TOKEN:
    raise RuntimeError("ENTSOE_TOKEN not set")

if not ERA5_TOKEN:
    raise RuntimeError("ERA5_TOKEN not set")


app.mount(
    "/artifacts",
    StaticFiles(directory=str(AUTOMATIC_DIR)),
    name="artifacts",
)

# import boto3
# from forecasting.model.services.model_registry import get_model

# s3 = boto3.client("s3")

# s3.download_file(
#     "price-forecast-tft-model",
#     f"models/{zone}/model.ckpt",
#     "/tmp/model.ckpt"
# )

# tft = get_model("/tmp/model.ckpt")

# @app.get("/predict")
# def predict(zone: str, date_to_predict: str):

#     model_path = f"/tmp/{zone}_model.ckpt"

#     s3.download_file(
#         BUCKET_NAME,
#         f"models/{zone}/model.ckpt",
#         model_path
#     )

#     model = get_model(model_path)

#     prediction = predict_next_24h(
#         model=model,
#         zone=zone,
#         date=date_to_predict
#     )

#     return prediction

# Run with
# uvicorn api.app:app --reload
# http://localhost:8000/


# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent.parent

# app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")


# @app.get("/")
# def home():
#     return HTMLResponse((BASE_DIR / "web" / "templates" / "index.html").read_text())
