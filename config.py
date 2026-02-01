from pathlib import Path
import logging
import sys

# training
TRAINING_LOOKBACK_DAYS = 365
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24

# prediction
DEFAULT_HISTORY_HOURS = 168
DEFAULT_FORECAST_HOURS = 24

# TFT
BATCH_SIZE = 64
MAX_EPOCHS = 3  # 30

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /price_forecast

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR = PROJECT_ROOT / "automatic"

# S3
S3_BUCKET_NAME = "price-forecast-tft-model"
S3_REGION = "eu-north-1"
ENABLE_ADMIN_AUTH = False


### Logs ================================================
def setup_logging(log_level=logging.INFO, log_dir=LOG_DIR):
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers
    root.handlers.clear()

    # ---- Console (Colab / FastAPI) ----
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(log_level)

    # ---- App log ----
    app_handler = logging.FileHandler(log_dir / "app.log")
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.INFO)

    # ---- Error log ----
    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.WARNING)

    root.addHandler(console)
    root.addHandler(app_handler)
    root.addHandler(error_handler)
