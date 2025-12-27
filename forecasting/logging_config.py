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
