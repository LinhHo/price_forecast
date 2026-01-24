"""
Class to use in both training and inference (predict)
"""

import json
import numpy as np
import torch
import pandas as pd
import datetime as dt
from datetime import timedelta, datetime
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path


from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from forecasting.data.era5 import load_era5
from forecasting.data.entsoe import load_prices
from forecasting.data.open_meteo import load_forecast
from forecasting.data import io
from forecasting.features.build_features import add_features
from forecasting.model.dataset import build_dataset
from config_runtime import get_runtime_args
from config import (
    AUTOMATIC_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    DEFAULT_HISTORY_HOURS,
    DEFAULT_FORECAST_HOURS,
)


args = get_runtime_args()

max_epochs = args.max_epochs or MAX_EPOCHS
batch_size = args.batch_size or BATCH_SIZE

import logging

logger = logging.getLogger(__name__)


class TFTPriceModel:
    def __init__(self, zone: str, run_id: str = None):
        self.zone = zone
        self.model = None
        self.training_dataset = None
        self.last_time_idx = None

        # 1. Establish the run directory immediately
        self.run_id: str | None = None
        self.run_dir: Path | None = None

        # self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.run_dir = AUTOMATIC_DIR / self.zone / "runs" / self.run_id
        # self._dirs_created = False

    def _ensure_dirs(self):
        """Creates the directory structure if it doesn't exist yet."""
        if not self._dirs_created:
            for folder in ["metrics", "figures", "model", "data", "predict"]:
                (self.run_dir / folder).mkdir(parents=True, exist_ok=True)
            self._dirs_created = True

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        return add_features(df, self.zone)

    def _load_training_data(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        # Use left join to keep price as primary timeline
        df_price = load_prices(self.zone, start, end)
        df_weather = load_era5(self.zone, start, end)
        return df_price.join(df_weather, how="left").ffill().bfill()

    def train(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        max_epochs=max_epochs,
        batch_size=batch_size,
    ):
        self.run_id = datetime.now().tz_localize("UTC").strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = AUTOMATIC_DIR / self.zone / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training model for zone=%s in %s", self.zone, self.run_dir)
        self._ensure_dirs()

        # Convert strings to Timestamps if necessary
        start, end = pd.Timestamp(start), pd.Timestamp(end)

        df = self._load_training_data(start, end)
        df = df.sort_index()
        df = self._add_features(df)
        df["time_idx"] = np.arange(len(df))

        self.last_time_idx = df["time_idx"].max()

        # Build dataset
        self.training_dataset = build_dataset(
            df,
            MAX_ENCODER_LENGTH,
            MAX_PREDICTION_LENGTH,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        train_dl = self.training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=4
        )
        val_dataloader = val_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=4
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
            max_epochs=max_epochs,
        )

        trainer.fit(self.model, train_dl, val_dataloader)
        self._save(trainer, self.training_dataset, val_dataloader)

    def _save(self, trainer: Trainer, training: TimeSeriesDataSet, val_dataloader):
        self._ensure_dirs()

        # Save model & Dataset
        io.save_checkpoint(trainer, self.run_dir / "model" / "tft.ckpt")
        io.save_TimeSeriesDataSet(
            training, self.run_dir / "data" / "training_dataset.pt"
        )

        io.save_json(
            {"zone": self.zone, "last_time_idx": int(self.last_time_idx)},
            self.run_dir / "meta.json",
        )

        # Predict for validation metrics
        predictions = self.model.predict(val_dataloader)
        # Handle actuals extraction safely
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])

        y_true = actuals.numpy().flatten()
        y_pred = predictions.numpy().flatten()

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        y_min = float(np.min(y_true))
        y_max = float(np.max(y_true))
        stats = {
            "mae": mae,
            "rmse": rmse,
            "y_min": y_min,
            "y_max": y_max,
        }

        io.save_json(stats, self.run_dir / "metrics" / "metrics.json")
        logger.info(
            "Training metrics | zone=%s | MAE=%.3f | RMSE=%.3f, SMAPE=%.3f",
            self.zone,
            mae,
            rmse,
        )
        logger.info(
            "Range of true values: %.2f to %.2f",
            y_min,
            y_max,
        )

        logger.info(
            "Error is about %.2f%% to %.2f%% of the range",
            mae / y_max * 100,
            mae / y_min * 100,
        )

        logger.info(
            "Error is about %.2f%% to %.2f%% if prices are between 50 and 120 EUR/MWh",
            mae / 120 * 100,
            mae / 50 * 100,
        )

        # Visuals
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:200], label="Actual")
        plt.plot(y_pred[:200], label="Predicted")
        plt.title(
            f"Validation performance MAE {mae:.3f}, within {mae / y_min * 100:.2f}--{mae / y_max * 100:.2f}% of the range \n and RMSE {rmse:.3f}"
        )
        plt.legend()
        plt.savefig(self.run_dir / "figures" / "validation_timeseries.png")
        plt.close()

        # Plot and save interpretation of the TFT model
        raw_predictions, x, *rest = self.model.predict(
            val_dataloader, mode="raw", return_x=True
        )
        interpretation = self.model.interpret_output(raw_predictions, reduction="sum")

        figs = self.model.plot_interpretation(interpretation)  # return a dict

        for name, fig in figs.items():
            fig.savefig(
                self.run_dir / "figures" / f"tft_interpretation_{name}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        logger.info("Model saved for zone=%s", self.zone)

    ### Predict ==========================================

    @classmethod
    def load(cls, zone: str, run_id: str):
        model.run_id = run_id
        model.run_dir = AUTOMATIC_DIR / zone / "runs" / run_id

        # base = AUTOMATIC_DIR / zone / "runs" / run_id
        model = cls(zone)

        try:
            model.training_dataset = TimeSeriesDataSet.load(
                model.run_dir / "data" / "training_dataset.pt"
            )
        except Exception as e:
            logger.error(
                f"Load pytorch training dataset failed, retrying with torch.load..."
            )
            model.training_dataset = torch.load(
                model.run_dir / "data" / "training_dataset.pt", weights_only=False
            )

        meta = json.load(open(model.run_dir / "meta.json"))

        model.last_time_idx = meta["last_time_idx"]

        model.model = TemporalFusionTransformer.load_from_checkpoint(
            model.run_dir / "model" / "tft.ckpt"
        )
        return model

    def _resolve_forecast_window(
        self,
        date_to_predict: pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        if date_to_predict is None:
            forecast_start = pd.Timestamp.now().floor("h")
        else:
            forecast_start = pd.Timestamp(date_to_predict)

        if forecast_start.tzinfo is None:
            forecast_start = forecast_start.tz_localize("UTC")

        history_start = forecast_start - timedelta(hours=DEFAULT_HISTORY_HOURS)
        forecast_end = forecast_start + timedelta(hours=DEFAULT_FORECAST_HOURS)

        return history_start, forecast_start, forecast_end

    def _load_forecast_data(
        self,
        date_to_predict: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        history_start, forecast_start, forecast_end = self._resolve_forecast_window(
            date_to_predict
        )

        logger.info(
            "Forecast window | history=%s forecast=%s → %s",
            history_start,
            forecast_start,
            forecast_end,
        )

        # ENTSO-E returns day-ahead price by day
        # .loc time again to make sure getting 24h, might not the end of the day 23:00:00
        prices = load_prices(
            self.zone,
            start=history_start,
            end=forecast_start,
        ).loc[history_start:forecast_start]

        weather = load_forecast(
            self.zone,
            start=history_start,
            end=forecast_end,
        ).loc[history_start:forecast_end]

        df = weather.join(prices, how="left")

        return df, forecast_start

    def predict(self, date_to_predict: pd.Timestamp | None = None):
        # (self.run_dir / "predict").mkdir(parents=True, exist_ok=True)

        # self._ensure_dirs()

        df, forecast_start = self._load_forecast_data(date_to_predict)

        # Save in one folder /predictions, each prediction run with its forecast_id
        forecast_id = forecast_start.strftime("%Y-%m-%d_%H-%M-%S")
        pred_dir = self.run_dir / "predictions" / forecast_id
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Only get the forecasting window and past data for encoder length
        total_len = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
        print(f"total_len of prediction: {total_len}")
        logger.info("Predicting for %s", forecast_start)

        df = df.iloc[-total_len:]
        df = self._add_features(df)

        # Align time_idx with the end of the training index
        df["time_idx"] = np.arange(len(df)) + (self.last_time_idx + 1)

        # TFT requirement: handle target column even in predict
        df["price_is_missing"] = df["price_eur_per_mwh"].isna().astype(int)
        df["price_eur_per_mwh"] = (
            df["price_eur_per_mwh"].where(df["price_is_missing"] == 0).ffill()
        )

        # ensure evaluation mode to avoid randomness
        self.model.eval()

        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, df, predict=True, stop_randomization=True
        )
        dl = dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )  # only need to predict once

        with torch.no_grad():
            raw_preds, *rest = self.model.predict(dl, mode="raw", return_x=True)

        preds = raw_preds["prediction"]

        # Extract timestamps from the decoder window
        # index values for the prediction period
        df_predict = pd.DataFrame(
            {
                "time_idx": np.arange(preds.shape[1]),
                "p10": preds[0, :, 0].cpu().numpy().flatten(),
                "p50": preds[0, :, 1].cpu().numpy().flatten(),
                "p90": preds[0, :, 2].cpu().numpy().flatten(),
            }
        )

        # out_path = (
        #     self.run_dir
        #     / "predict"
        #     / f"pred_{forecast_start.strftime('%Y%m%d_%H')}.csv"
        # )
        df_predict.to_csv(pred_dir / f"pred_{forecast_id}.csv", index=False)

        # Plot timeseries of past prices and forecast with different colours with range p10-90
        toplot = df.copy()["price_eur_per_mwh"].to_frame()
        for var in ["p10", "p50", "p90"]:
            toplot[var] = toplot["price_eur_per_mwh"]
            toplot[var].iloc[-MAX_PREDICTION_LENGTH:] = df_predict[var].values

        plt.figure(figsize=(12, 4))
        plt.plot(
            toplot.index,
            toplot["p50"],
            label="TFT forecast",
            color="darkorange",
            linewidth=1.5,
        )
        plt.plot(
            toplot.index[-MAX_PREDICTION_LENGTH:],
            toplot["price_eur_per_mwh"][-MAX_PREDICTION_LENGTH:],
            label="ENTSO-E",
            color="blue",
            linewidth=1.5,
        )

        plt.fill_between(
            toplot.index, toplot["p90"], toplot["p10"], alpha=0.3, facecolor="orange"
        )
        plt.grid("major")
        plt.ylabel("Price [EUR/MWh]")
        plt.legend()

        plt.savefig(pred_dir / f"prediction_{forecast_id}.jpeg")
        plt.close()

        logger.info("Plotting prediction for %s", forecast_start)
