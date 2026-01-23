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
MAX_EPOCHS = 5  # 30

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /price_forecast

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUTOMATIC_DIR = PROJECT_ROOT / "automatic"

# S3
S3_BUCKET_NAME = "price-forecast-tft-model"
S3_REGION = "eu-north-1"


### Logs ================================================
def setup_logging(log_level=logging.INFO, log_dir=LOG_DIR):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "app.log"),
            logging.StreamHandler(sys.stdout),  # Forces output to Colab cell
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Project root (forecasting/ is inside repo)
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
