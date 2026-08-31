from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
from config import setup_logging
from training.train_colab import train_and_upload

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train TFT price forecast model(s) and upload all artifacts to S3"
    )
    parser.add_argument(
        "--zones", nargs="+", required=True,
        help="One or more zone codes to train, e.g. --zones FR DE NL",
    )
    parser.add_argument("--start", default=None, help="Training start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="Training end date   YYYY-MM-DD")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    failed = []
    for zone in args.zones:
        logger.info("Starting training for zone=%s", zone)
        try:
            train_and_upload(
                zone=zone,
                start=args.start,
                end=args.end,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            )
            logger.info("Finished zone=%s", zone)
        except Exception as e:
            logger.error("Training failed for zone=%s: %s", zone, e, exc_info=True)
            failed.append(zone)

    if failed:
        raise RuntimeError(f"Training failed for zones: {failed}")


if __name__ == "__main__":
    main()
