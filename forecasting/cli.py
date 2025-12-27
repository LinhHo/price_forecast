# cli.py
import click
from forecasting.model.train import train_model
from forecasting.model.predict import predict_next_24h


@click.command()
@click.option("--zone", required=True)
@click.option("--train", is_flag=True)
def main(zone, train):
    if train:
        train_model(zone)
    else:
        predict_next_24h(zone)


if __name__ == "__main__":
    main()
