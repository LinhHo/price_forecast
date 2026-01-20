from pathlib import Path

# training
TRAINING_LOOKBACK_DAYS = 60  # 365
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24

# prediction
DEFAULT_HISTORY_DAYS = 7
DEFAULT_FORECAST_HOURS = 24

# TFT
BATCH_SIZE = 64
MAX_EPOCHS = 10  # 30

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /price_forecast

OUTPUT_DIR = PROJECT_ROOT / "output"
AUTOMATIC_DIR = OUTPUT_DIR / "automatic"
FIG_DIR = OUTPUT_DIR / "figures"

for p in [OUTPUT_DIR, AUTOMATIC_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
