from pathlib import Path

training_start = "2024-01-01"
training_end = "2024-03-01"

# project_root = directory containing "forecasting/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECASTING_DIR = PROJECT_ROOT / "forecasting"
OUTPUT_DIR = FORECASTING_DIR / "output"
AUTOMATIC_DIR = PROJECT_ROOT / "automatic"

# create folders if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR.mkdir(parents=True, exist_ok=True)
