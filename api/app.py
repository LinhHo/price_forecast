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

# include dropdown list of zones
app.include_router(zones.router, prefix="/zones")


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
