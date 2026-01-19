"""
Class to use in both training and inference (predict)
"""

import json
import numpy as np
import torch
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from data.era5 import load_era5
from data.entsoe import load_prices
from model.dataset import build_dataset
from price_forecast.OLD_config import (
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

        self._save(trainer)

    def _save(self, trainer: Trainer, training: TimeSeriesDataSet):
        model_path = AUTOMATIC_DIR / f"{self.zone}_tft.ckpt"
        trainer.save_checkpoint(model_path)
        training.save(AUTOMATIC_DIR / f"{self.zone}_training_dataset.pt")

        meta = {
            "zone": self.zone,
            "last_time_idx": int(self.last_time_idx),
        }

        with open(AUTOMATIC_DIR / f"{self.zone}_meta.json", "w") as f:
            json.dump(meta, f)

        logger.info("Model saved for zone=%s", self.zone)

    @classmethod
    def load(cls, zone: str):
        model = cls(zone)
        model.training_dataset = torch.load(
            AUTOMATIC_DIR / f"{zone}_training_dataset.pt", weights_only=False
        )

        with open(AUTOMATIC_DIR / f"{zone}_meta.json") as f:
            meta = json.load(f)

        model.last_time_idx = meta["last_time_idx"]

        model.model = TemporalFusionTransformer.load_from_checkpoint(
            AUTOMATIC_DIR / f"{zone}_tft.ckpt",
            weights_only=False,
        )

        logger.info("Model loaded for zone=%s", zone)
        return model

    # # model/tft_model.py
    # class TFTPriceModel:
    #     @classmethod
    #     def load(cls, zone: str):
    #         return cls(
    #             zone=zone,
    #             checkpoint=AUTOMATIC_DIR / f"{zone}_tft_price_model.ckpt",
    #             dataset_dir=AUTOMATIC_DIR
    #         )

    def predict(self, df_history: pd.DataFrame, df_future: pd.DataFrame):
        df = pd.concat([df_history, df_future]).sort_index()
        df["time_idx"] = np.arange(len(df))
        df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(0.0)

        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        dl = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

        raw_preds, x = self.model.predict(
            dl,
            mode="raw",
            return_x=True,
        )

        preds = raw_preds["prediction"]  # (B, H, 3)

        return {
            "timestamp": x["decoder_time_idx"].cpu().numpy(),
            "p10": preds[:, :, 0].cpu().numpy(),
            "p50": preds[:, :, 1].cpu().numpy(),
            "p90": preds[:, :, 2].cpu().numpy(),
        }

    def resolve_prediction_window(date_to_predict=None):
        if date_to_predict is None:
            forecast_start = pd.Timestamp.utcnow().floor("h")
        else:
            forecast_start = pd.Timestamp(date_to_predict)

        history_start = forecast_start - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)
        forecast_end = forecast_start + pd.Timedelta(hours=DEFAULT_FORECAST_HOURS)

        return history_start, forecast_start, forecast_end
