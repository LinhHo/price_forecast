from pathlib import Path
import datetime as dt
import os
import logging
from dotenv import load_dotenv


# This file is at: price_forecast/forecasting/config.py
# Level 0: config.py
# Level 1: forecasting/
# Level 2: price_forecast/ (This is your Root)

# Get the directory where config.py is located
BASE_DIR = Path(__file__).resolve().parent  # Result: .../forecasting/
# project_root = directory containing "forecasting/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Result: .../price_forecast/

# Try loading from current directory first, then fallback to environment
# Look for .env in the same directory as config.py
# Since your .env is in the root, use PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN")
    ERA5_TOKEN = os.getenv("ERA5_TOKEN")


TRAINING_START = dt.datetime(2024, 1, 1, 0, 0)
TRAINING_END = dt.datetime(2024, 3, 1, 0, 0)

BATCH_SIZE = 64
MAX_EPOCHS = 10  # 30

MAX_ENCODER_LENGTH = 168  # use past 7 days, electricity prices have weekly pattern
MAX_PREDICTION_LENGTH = 24  # predict next 24 hours

FORECASTING_DIR = PROJECT_ROOT / "forecasting"
OUTPUT_DIR = FORECASTING_DIR / "output"
AUTOMATIC_DIR = FORECASTING_DIR / "automatic"

# create folders if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR.mkdir(parents=True, exist_ok=True)


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler(),  # still prints to console
    ],
)

import logging
from pathlib import Path


def setup_logging(log_level=logging.INFO):
    # Project root (forecasting/ is inside repo)
    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    app_log = log_dir / "app.log"
    error_log = log_dir / "error.log"

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()  # avoid duplicate logs

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ---- App log file ----
    app_handler = logging.FileHandler(app_log)
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    # ---- Error log file ----
    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
