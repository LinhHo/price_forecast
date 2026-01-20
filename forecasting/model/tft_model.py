"""
Class to use in both training and inference (predict)
"""

import json
import numpy as np
import torch
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from forecasting.data.era5 import load_era5
from forecasting.data.entsoe import load_prices
from forecasting.model.dataset import build_dataset
from forecasting.data import io
from price_forecast.config import (
    AUTOMATIC_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_FORECAST_HOURS,
)

import logging

logger = logging.getLogger(__name__)


class TFTPriceModel:
    def __init__(self, zone: str):
        self.zone = zone
        self.model = None
        self.training_dataset = None
        self.last_time_idx = None

    @staticmethod
    def load_training_data(zone, start, end):
        df_weather = load_era5(zone, start, end)
        df_price = load_prices(zone, start, end)
        return df_weather.join(df_price)

    def train(self, df: pd.DataFrame):
        logger.info("Training model for zone=%s", self.zone)

        df = df.sort_index()
        df["time_idx"] = np.arange(len(df))

        self.last_time_idx = df["time_idx"].max()

        training_df = df[df.time_idx <= self.last_time_idx - MAX_PREDICTION_LENGTH]

        self.training_dataset = build_dataset(
            training_df,
            MAX_ENCODER_LENGTH,
            MAX_PREDICTION_LENGTH,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            training_df,
            predict=True,
            stop_randomization=True,
        )

        train_dl = self.training_dataset.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=4
        )
        val_dl = val_dataset.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=4
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=3e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            loss=QuantileLoss([0.1, 0.5, 0.9]),
        )

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=MAX_EPOCHS,
        )

        trainer.fit(self.model, train_dl, val_dl)

        self._save(trainer, self.training_dataset)

    def _save(self, trainer: Trainer, training: TimeSeriesDataSet):
        base = AUTOMATIC_DIR / self.zone

        io.save_checkpoint(
            trainer,
            base / "tft.ckpt",
        )

        io.save_tsd(
            training,
            base / "training_dataset.pt",
        )

        io.save_json(
            {
                "zone": self.zone,
                "last_time_idx": int(self.last_time_idx),
            },
            base / "meta.json",
        )

        logger.info("Model saved for zone=%s", self.zone)

    @classmethod
    def load(cls, zone: str):
        base = AUTOMATIC_DIR / zone
        model = cls(zone)

        model.training_dataset = io.load_tsd(base / "training_dataset.pt")

        meta = io.load_json(base / "meta.json")

        model.last_time_idx = meta["last_time_idx"]

        model.model = io.load_checkpoint(
            TemporalFusionTransformer,
            base / "tft.ckpt",
        )

        logger.info("Model loaded for zone=%s", zone)
        return model

    def predict(self, df_history: pd.DataFrame, df_future: pd.DataFrame):
        logger.info("Predicting for zone=%s", self.zone)

        df = pd.concat([df_history, df_future]).sort_index()

        df["time_idx"] = np.arange(
            self.last_time_idx - len(df) + 1,
            self.last_time_idx + 1,
        )

        # Flag missing price to forecast before fill
        df["price_is_missing"] = df["price_eur_per_mwh"].isna().astype(int)
        df["price_eur_per_mwh"] = df["price_eur_per_mwh"].ffill()

        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        dl = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

        raw_preds, x = self.model.predict(dl, mode="raw", return_x=True)
        preds = raw_preds["prediction"]

        decoder_time_idx = x["decoder_time_idx"].cpu().numpy()
        timestamps = np.take(df.index.values, decoder_time_idx)

        return pd.DataFrame(
            {
                "timestamp": timestamps.flatten(),
                "p10": preds[:, :, 0].cpu().numpy().flatten(),
                "p50": preds[:, :, 1].cpu().numpy().flatten(),
                "p90": preds[:, :, 2].cpu().numpy().flatten(),
            }
        )

    @staticmethod
    def resolve_prediction_window(date_to_predict=None):
        if date_to_predict is None:
            forecast_start = pd.Timestamp.utcnow().floor("h")
        else:
            forecast_start = pd.Timestamp(date_to_predict)

        history_start = forecast_start - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)
        forecast_end = forecast_start + pd.Timedelta(hours=DEFAULT_FORECAST_HOURS)

        return history_start, forecast_start, forecast_end
