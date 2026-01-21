from forecasting.model.tft_model import TFTPriceModel

_MODEL_CACHE = {}
import logging

logger = logging.getLogger(__name__)


def get_model(zone: str) -> TFTPriceModel:
    if zone not in _MODEL_CACHE:
        _MODEL_CACHE[zone] = TFTPriceModel.load(zone)
    return _MODEL_CACHE[zone]


import argparse


def get_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Size of each batch",
    )

    args, _ = parser.parse_known_args()
    return args
