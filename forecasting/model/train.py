import numpy as np
import matplotlib.pyplot as plt
import torch

# import torch.nn as nn
import torch.nn.functional as F

# import torch.optim as optim
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from sklearn.metrics import mean_absolute_error

from forecasting.config import AUTOMATIC_DIR, OUTPUT_DIR, BATCH_SIZE, MAX_EPOCHS
from forecasting.data.era5 import load_era5
from forecasting.data.entsoe import load_prices
from forecasting.features.build_features import (
    add_time_features,
    add_holiday_feature,
    add_zone,
)
from forecasting.dataset.timeseries import build_tft_dataset
import logging

logger = logging.getLogger(__name__)


def prepare_training_df(zone):
    df_weather = load_era5(zone)
    df_price = load_prices(zone)
    df = df_weather.join(df_price, how="inner")

    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_zone(df, zone)

    df = df.sort_index()
    df["time_idx"] = np.arange(len(df))

    return df


def train_model(zone):
    df = prepare_training_df(zone)
    training = build_tft_dataset(df, is_training=True)

    logger.info(
        "Starting training | zone=%s | start=%s | end=%s",
        zone,
        df.index.min(),
        df.index.max(),
    )

    # save metadata last_time_idx to keep continuity for prediction
    last_time_idx = df["time_idx"].max()
    np.save(AUTOMATIC_DIR / f"{zone}_last_time_idx.npy", last_time_idx)

    ### model setup ==========================================
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-3,  # 1e-3, too conservative
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
    )

    ### train the model (core) ================================

    # validation set
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    train_dataloader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=4
    )

    val_dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=4
    )

    # Train the model
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
    )
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Validation metrics
    predictions = tft.predict(val_dataloader)
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])

    y_true = actuals.numpy().flatten()
    y_pred = predictions.numpy().flatten()

    # compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )
    logger.info(
        "Training metrics | zone=%s | MAE=%.3f | RMSE=%.3f, SMAPE=%.3f",
        zone,
        mae,
        rmse,
        smape,
    )
    y_min = float(np.min(y_true))
    y_max = float(np.max(y_true))
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
    # visualize validation performance
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(y_pred[:200], label="Predicted")
    plt.legend()
    plt.title(f"Validation performance MAE {mae}")
    plt.savefig(OUTPUT_DIR / "Validation_training.jpeg")

    # Plot and save interpretation of the TFT model
    raw_predictions, x, *rest = tft.predict(val_dataloader, mode="raw", return_x=True)

    interpretation = tft.interpret_output(raw_predictions, reduction="sum")

    figs = tft.plot_interpretation(interpretation)  # return a dict

    for name, fig in figs.items():
        fig.savefig(
            OUTPUT_DIR / f"tft_interpretation_{name}.png",
            dpi=200,
            bbox_inches="tight",
        )

    logger.info("Training completed, saving plots and model...")

    # save model
    training.save(
        AUTOMATIC_DIR / f"{zone}_training_dataset"
    )  # built-in save function for TimeSeriesDataSet PyTorch dataset
    trainer.save_checkpoint(AUTOMATIC_DIR / "tft_price_model.ckpt")  # save_model(tft)

    return trainer
