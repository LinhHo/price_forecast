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
import pathlib

from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from forecasting.data.era5 import load_era5
from forecasting.data.entsoe import load_prices
from forecasting.data.open_meteo import load_forecast
from forecasting.model.dataset import build_dataset
from forecasting.features.build_features import add_features
from forecasting.data import io
from price_forecast.config import (
    AUTOMATIC_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    DEFAULT_HISTORY_HOURS,
    DEFAULT_FORECAST_HOURS,
)

import logging

logger = logging.getLogger(__name__)


class TFTPriceModel:
    def __init__(self, zone: str, run_id: str = None):
        self.zone = zone
        self.model = None
        self.training_dataset = None
        self.last_time_idx = None

        # 1. Establish the run directory immediately
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = AUTOMATIC_DIR / self.zone / "runs" / self.run_id
        self._dirs_created = False

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

    def train(self, start: pd.Timestamp, end: pd.Timestamp):
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
            train=True, batch_size=BATCH_SIZE, num_workers=4
        )
        val_dataloader = val_dataset.to_dataloader(
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
        """Loads a specific run for prediction"""
        # path: /automatic/ZONE/runs/RUN_ID
        run_path = AUTOMATIC_DIR / zone / "runs" / run_id

        instance = cls(zone, run_id=run_id)
        instance.training_dataset = io.load_TimeSeriesDataSet(
            run_path / "data" / "training_dataset.pt"
        )

        meta = io.load_json(run_path / "meta.json")
        instance.last_time_idx = meta["last_time_idx"]

        # Load weights into the architecture
        instance.model = TemporalFusionTransformer.load_from_checkpoint(
            run_path / "model" / "tft.ckpt"
        )

        logger.info("Model loaded from %s", run_path)
        return instance

    # def _load_forecast_data(
    #     self, date_to_predict: pd.Timestamp | None = None
    # ) -> pd.DataFrame:
    #     forecast_start = date_to_predict
    #     forecast_end = forecast_start + timedelta(hours=DEFAULT_FORECAST_HOURS)
    #     history_start = forecast_start - timedelta(hours=DEFAULT_HISTORY_HOURS)

    #     df_history = load_prices(self.zone, start=history_start, end=forecast_start)
    #     df_future = load_forecast(
    #         self.zone, start=forecast_start + pd.Timedelta(hours=1), end=forecast_end
    #     )
    #     df = pd.concat([df_history, df_future]).sort_index()
    #     df = df[~df.index.duplicated(keep="last")]  # remove duplicates

    #     return df

    def _load_forecast_data(
        self, date_to_predict: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        forecast_start = date_to_predict
        forecast_end = forecast_start + timedelta(
            hours=DEFAULT_FORECAST_HOURS
        )  # to cover mid-day forecast to the next 24h
        history_start = forecast_start - timedelta(hours=DEFAULT_HISTORY_HOURS)
        print(history_start, forecast_start, forecast_end)

        # ENTSO-E returns day-ahead price by day
        entsoe = load_prices(self.zone, start=history_start, end=forecast_start).loc[
            history_start:forecast_start
        ]
        open_meteo = load_forecast(
            self.zone, start=history_start, end=forecast_end
        ).loc[history_start:forecast_end]

        return open_meteo.join(entsoe, how="left")

    def predict(self, date_to_predict: pd.Timestamp | None = None):
        if date_to_predict is None:
            date_to_predict = pd.Timestamp.now().floor("h").tz_localize("UTC")
        else:
            date_to_predict = pd.Timestamp(date_to_predict).tz_localize("UTC")

        self._ensure_dirs()
        logger.info("Predicting for %s", date_to_predict)

        df = self._load_forecast_data(date_to_predict)
        # Only get the forecasting window and past data for encoder length
        total_len = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
        print(f"total_len of prediction: {total_len}")

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
                "timestamp": df.index.values,
                "entsoe": df["price_eur_per_mwh"],
                "p10": preds[0, :, 0].cpu().numpy().flatten(),
                "p50": preds[0, :, 1].cpu().numpy().flatten(),
                "p90": preds[0, :, 2].cpu().numpy().flatten(),
            }
        )

        out_path = (
            self.run_dir
            / "predict"
            / f"pred_{date_to_predict.strftime('%Y%m%d_%H')}.csv"
        )
        df_predict.to_csv(out_path, index=False)

        # Plot timeseries of past prices and forecast with different colours with range p10-90
        # toplot = df.copy()["price_eur_per_mwh"].to_frame()1
        # toplot["label"] = "ENTSOE price"
        # toplot.loc[toplot.index[-MAX_PREDICTION_LENGTH:], "label"] = "TFT forecast"
        # toplot.loc[toplot.index[-MAX_PREDICTION_LENGTH:], "price_eur_per_mwh"] = (
        #     df_predict["p50"].values
        # )
        # # add range p10-90 only for forecasting window
        # toplot["p10"] = toplot["price_eur_per_mwh"]
        # toplot["p90"] = toplot["price_eur_per_mwh"]
        # toplot.loc[toplot.index[-MAX_PREDICTION_LENGTH:], "p10"] = df_predict[
        #     "p10"
        # ].values
        # toplot.loc[toplot.index[-MAX_PREDICTION_LENGTH:], "p90"] = df_predict[
        #     "p90"
        # ].values

        # plt.figure(figsize=(12, 4))

        # for label, g in toplot.groupby("label"):
        #     plt.fill_between(
        #         g["time"], g["p90"], g["p10"], alpha=0.3, facecolor="orange"
        #     )
        #     plt.plot(g["time"], g["price_eur_per_mwh"], label=label)

        # plt.grid("major")
        # plt.ylabel("Price [EUR/MWh]")
        # plt.legend()

        plt.figure(figsize=(12, 4))
        plt.plot(df_predict["entsoe"], label="ENTSOE price")
        plt.plot(df_predict["p50"], label="TFT forecast")
        plt.fill_between(
            df_predict["timestamp"],
            df_predict["p90"],
            df_predict["p10"],
            alpha=0.3,
            facecolor="orange",
        )

        plt.savefig(
            self.run_dir
            / "figures"
            / f"prediction_{date_to_predict.strftime('%Y%m%d_%H')}.jpeg"
        )
        plt.close()

        logger.info("Prediction saved to %s", out_path)
        return df_predict


# import json
# import numpy as np
# import torch
# import pandas as pd
# import datetime as dt
# from datetime import timedelta, datetime
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt

# from lightning.pytorch import Trainer
# from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
# from pytorch_forecasting.metrics import QuantileLoss

# from forecasting.data.era5 import load_era5
# from forecasting.data.entsoe import load_prices
# from forecasting.data.open_meteo import load_forecast
# from forecasting.model.dataset import build_dataset
# from forecasting.features.build_features import add_features
# from forecasting.data import io
# from price_forecast.config import (
#     AUTOMATIC_DIR,
#     BATCH_SIZE,
#     MAX_EPOCHS,
#     MAX_ENCODER_LENGTH,
#     MAX_PREDICTION_LENGTH,
#     DEFAULT_HISTORY_DAYS,
#     DEFAULT_FORECAST_HOURS,
# )

# import logging

# logger = logging.getLogger(__name__)


# class TFTPriceModel:
#     def __init__(self, zone: str):
#         self.zone = zone
#         self.model = None
#         self.training_dataset = None
#         self.last_time_idx = None

#     def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply feature engineering consistently for training and prediction
#         """
#         if not isinstance(df.index, pd.DatetimeIndex):
#             raise ValueError("DataFrame index must be DatetimeIndex")

#         return add_features(df, self.zone)

#     ### Train ==========================================
#     def _load_training_data(
#         self,
#         start: pd.Timestamp,
#         end: pd.Timestamp,
#     ) -> pd.DataFrame:
#         df_weather = load_era5(self.zone, start, end)
#         df_price = load_prices(self.zone, start, end)
#         return df_weather.join(df_price, how="inner")

#     def train(self, start: pd.Timestamp, end: pd.Timestamp):
#         logger.info("Training model for zone=%s", self.zone)

#         df = self._load_training_data(start, end)
#         df = df.sort_index()
#         df = self._add_features(df)
#         df["time_idx"] = np.arange(len(df))

#         self.last_time_idx = df["time_idx"].max()

#         training_df = df[df.time_idx <= self.last_time_idx - MAX_PREDICTION_LENGTH]

#         self.training_dataset = build_dataset(
#             training_df,
#             MAX_ENCODER_LENGTH,
#             MAX_PREDICTION_LENGTH,
#         )

#         val_dataset = TimeSeriesDataSet.from_dataset(
#             self.training_dataset,
#             training_df,
#             predict=True,
#             stop_randomization=True,
#         )

#         train_dl = self.training_dataset.to_dataloader(
#             train=True, batch_size=BATCH_SIZE, num_workers=4
#         )
#         val_dataloader = val_dataset.to_dataloader(
#             train=False, batch_size=BATCH_SIZE, num_workers=4
#         )

#         self.model = TemporalFusionTransformer.from_dataset(
#             self.training_dataset,
#             learning_rate=3e-3,
#             hidden_size=32,
#             attention_head_size=4,
#             dropout=0.1,
#             loss=QuantileLoss([0.1, 0.5, 0.9]),
#         )

#         trainer = Trainer(
#             accelerator="gpu" if torch.cuda.is_available() else "cpu",
#             devices=1,
#             max_epochs=MAX_EPOCHS,
#         )

#         trainer.fit(self.model, train_dl, val_dataloader)

#         self._save(trainer, self.training_dataset, val_dataloader)

#     def _save(self, trainer: Trainer, training: TimeSeriesDataSet, val_dataloader):

#         # Create run directory and save stats, meta & figs etc.
#         run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         self.run_dir = AUTOMATIC_DIR / self.zone / "runs" / run_id

#         (self.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
#         (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)
#         (self.run_dir / "model").mkdir(parents=True, exist_ok=True)
#         (self.run_dir / "data").mkdir(parents=True, exist_ok=True)

#         # Save model
#         io.save_checkpoint(
#             trainer,
#             self.run_dir / "model" / "tft.ckpt",
#         )

#         io.save_TimeSeriesDataSet(
#             training,
#             self.run_dir / "data" / "training_dataset.pt",
#         )

#         io.save_json(
#             {
#                 "zone": self.zone,
#                 "last_time_idx": int(self.last_time_idx),
#             },
#             self.run_dir / "meta.json",
#         )

#         # Save stats
#         predictions = self.model.predict(val_dataloader)
#         actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])

#         y_true = actuals.numpy().flatten()
#         y_pred = predictions.numpy().flatten()

#         # compute metrics
#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#         smape = np.mean(
#             2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
#         )
#         y_min = float(np.min(y_true))
#         y_max = float(np.max(y_true))
#         stats = {
#             "zone": self.zone,
#             "mae": mae,
#             "rmse": rmse,
#             "smape": smape,
#             "y_min": y_min,
#             "y_max": y_max,
#         }

#         pd.DataFrame([stats]).to_csv(
#             self.run_dir / "metrics" / "training_stats.csv",
#             index=False,
#         )

#         with open(self.run_dir / "metrics" / "metrics.json", "w") as f:
#             json.dump(stats, f, indent=2)

#         # Save figures
#         plt.figure(figsize=(10, 4))
#         plt.plot(y_true[:200], label="Actual")
#         plt.plot(y_pred[:200], label="Predicted")
#         plt.legend()
#         plt.title(f"Validation MAE={mae:.2f}")

#         plt.savefig(
#             self.run_dir / "figures" / "validation_timeseries.png",
#             dpi=200,
#             bbox_inches="tight",
#         )
#         plt.close()

#         # Plot and save interpretation of the TFT model
#         raw_predictions, x, *rest = self.model.predict(
#             val_dataloader, mode="raw", return_x=True
#         )

#         interpretation = self.model.interpret_output(raw_predictions, reduction="sum")

#         figs = self.model.plot_interpretation(interpretation)  # return a dict

#         for name, fig in figs.items():
#             fig.savefig(
#                 self.run_dir / "figures" / f"tft_interpretation_{name}.png",
#                 dpi=200,
#                 bbox_inches="tight",
#             )

#             plt.close()

#         logger.info("Model saved for zone=%s", self.zone)

#     ### Predict ==========================================
#     @classmethod
#     def load(cls, zone: str):
#         base = AUTOMATIC_DIR / zone
#         model = cls(zone)

#         model.training_dataset = io.load_TimeSeriesDataSet(base / "training_dataset.pt")

#         meta = io.load_json(base / "meta.json")

#         model.last_time_idx = meta["last_time_idx"]

#         model.model = io.load_checkpoint(
#             TemporalFusionTransformer,
#             base / "tft.ckpt",
#         )

#         logger.info("Model loaded for zone=%s", zone)
#         return model

#     def _load_forecast_data(
#         self,
#         date_to_predict: pd.Timestamp | None = None,
#     ) -> pd.DataFrame:
#         """
#         Returns: concat of
#             df_history: past data with prices
#             df_future: future data without prices
#         """
#         # If not specify, predict for today
#         if date_to_predict is None:
#             forecast_start = dt.today()  # Default to predict 24h from today
#         else:
#             forecast_start = date_to_predict
#         forecast_end = date_to_predict + timedelta(hours=DEFAULT_FORECAST_HOURS)
#         history_start = date_to_predict - timedelta(days=DEFAULT_HISTORY_DAYS)

#         df_history = load_prices(
#             self.zone,
#             start=history_start,
#             end=forecast_start,
#         )

#         df_future = load_forecast(
#             self.zone,
#             start=forecast_start,
#             end=forecast_end,
#         )

#         return pd.concat([df_history, df_future]).sort_index()

#     def predict(self, date_to_predict: pd.Timestamp | None = None):
#         logger.info("Predicting for zone=%s", self.zone)

#         df = self._load_forecast_data(date_to_predict)
#         df = self._add_features(df)

#         df["time_idx"] = np.arange(
#             self.last_time_idx - len(df) + 1,
#             self.last_time_idx + 1,
#         )

#         # Flag missing price to forecast before fill
#         df["price_is_missing"] = df["price_eur_per_mwh"].isna().astype(int)
#         df["price_eur_per_mwh"] = df["price_eur_per_mwh"].ffill()

#         dataset = TimeSeriesDataSet.from_dataset(
#             self.training_dataset,
#             df,
#             predict=True,
#             stop_randomization=True,
#         )

#         dl = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

#         raw_preds, x = self.model.predict(dl, mode="raw", return_x=True)
#         preds = raw_preds["prediction"]

#         decoder_time_idx = x["decoder_time_idx"].cpu().numpy()
#         timestamps = np.take(df.index.values, decoder_time_idx)

#         df_predict = pd.DataFrame(
#             {
#                 "timestamp": timestamps.flatten(),
#                 "p10": preds[:, :, 0].cpu().numpy().flatten(),
#                 "p50": preds[:, :, 1].cpu().numpy().flatten(),
#                 "p90": preds[:, :, 2].cpu().numpy().flatten(),
#             }
#         )

#         df_predict.to_csv(
#             self.run_dir / "predict" / "predictions.csv",
#             index=False,
#         )


#     @staticmethod
#     def resolve_prediction_window(date_to_predict=None):
#         if date_to_predict is None:
#             forecast_start = pd.Timestamp.utcnow().floor("h")
#         else:
#             forecast_start = pd.Timestamp(date_to_predict)

#         history_start = forecast_start - pd.Timedelta(days=DEFAULT_HISTORY_DAYS)
#         forecast_end = forecast_start + pd.Timedelta(hours=DEFAULT_FORECAST_HOURS)

#         return history_start, forecast_start, forecast_end
