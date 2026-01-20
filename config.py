from pathlib import Path
import logging

# training
TRAINING_LOOKBACK_DAYS = 60  # 365
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24

# prediction
DEFAULT_HISTORY_DAYS = 7
DEFAULT_FORECAST_HOURS = 24

# TFT
BATCH_SIZE = 64
MAX_EPOCHS = 5  # 30

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /price_forecast

LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
AUTOMATIC_DIR = OUTPUT_DIR / "automatic"
FIG_DIR = OUTPUT_DIR / "figures"

for p in [LOG_DIR, OUTPUT_DIR, AUTOMATIC_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Set-up logs


def setup_logging(log_level=logging.INFO, log_dir=LOG_DIR):
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
