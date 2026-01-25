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
