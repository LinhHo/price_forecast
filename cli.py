import click
from model.train import train_model
from model.predict import predict_next_24h
from config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


@click.command()
@click.option("--zone", required=True)
@click.option("--train", is_flag=True)
@click.option("--predict", is_flag=True)
def main(zone, train, predict):
    if train:
        train_model(zone)
    if predict:
        predict_next_24h(zone)


if __name__ == "__main__":
    main()
