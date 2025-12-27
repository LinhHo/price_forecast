from pathlib import Path
from datetime import date

training_start = date(2024, 1, 1)
training_end = date(2024, 3, 1)

# project_root = directory containing "forecasting/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECASTING_DIR = PROJECT_ROOT / "forecasting"
OUTPUT_DIR = FORECASTING_DIR / "output"
AUTOMATIC_DIR = FORECASTING_DIR / "automatic"

# create folders if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR.mkdir(parents=True, exist_ok=True)
