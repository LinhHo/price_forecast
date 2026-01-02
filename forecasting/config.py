from pathlib import Path
import datetime as dt
import os
from dotenv import load_dotenv

from pathlib import Path

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


FORECASTING_DIR = PROJECT_ROOT / "forecasting"
OUTPUT_DIR = FORECASTING_DIR / "output"
AUTOMATIC_DIR = FORECASTING_DIR / "automatic"

# create folders if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR.mkdir(parents=True, exist_ok=True)
