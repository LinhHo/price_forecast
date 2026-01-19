import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from torch.serialization import add_safe_globals

add_safe_globals([TimeSeriesDataSet, pd.DataFrame, GroupNormalizer])

from config import (
    AUTOMATIC_DIR,
    FIG_DIR,
    BATCH_SIZE,
    MAX_PREDICTION_LENGTH,
    MAX_ENCODER_LENGTH,
)
from data.open_meteo import load_forecast
from data.entsoe import load_prices
from features.build_features import (
    add_time_features,
    add_holiday_feature,
    add_zone,
)
import logging

logger = logging.getLogger(__name__)


def prepare_forecast_df(zone, last_time_idx):
    df_weather = load_forecast(zone)
    df_price = load_prices(zone, is_training=False)

    df = df_weather.join(df_price, how="left")  # prices only for past
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_zone(df, zone)

    df = df.sort_index()

    df["time_idx"] = np.arange(last_time_idx - len(df) + 1, last_time_idx + 1)

    return df


def predict_next_24h(zone: str):
    # load last_time_idx from training
    last_time_idx = np.load(AUTOMATIC_DIR / f"{zone}_last_time_idx.npy")

    df = prepare_forecast_df(zone, last_time_idx)
    assert df["time_idx"].is_monotonic_increasing
    time_stamps = df.index
    print(
        f"==== Time stamps of df forecast with length {len(time_stamps)}: \n {time_stamps}"
    )

    # fill with dummy 0, PyTorch Forecasting does not infer “missing = predict this”, it uses max_prediction_length
    df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(0.0)

    logger.info(
        "Starting predicting | zone=%s | start=%s | end=%s",
        zone,
        df.index.min(),
        df.index.max(),
    )

    # Append forecast df to training history (required for TFT)
    try:
        training = TimeSeriesDataSet.load(AUTOMATIC_DIR / f"{zone}_training_dataset.pt")
    except Exception as e:
        logger.error(
            f"Load pytorch training dataset failed, retrying with torch.load..."
        )
        training = torch.load(
            AUTOMATIC_DIR / f"{zone}_training_dataset.pt", weights_only=False
        )

    # predict
    ### Load the trained model and predict ====================================
    # Build prediction dataset, applies SAME scaling, SAME categorical encodings, SAME time handling
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training,  # ← ORIGINAL training dataset
        df,
        predict=True,
        stop_randomization=True,
    )

    # load the trained model
    # Less safe, Only do this for your own checkpoints, Never for downloaded models
    tft = TemporalFusionTransformer.load_from_checkpoint(
        AUTOMATIC_DIR / f"{zone}_tft_price_model.ckpt", weights_only=False
    )

    logger.info("Model loaded from checkpoint")

    # Create dataloader
    prediction_dataloader = prediction_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=4
    )

    # Generate predictions. Point forecasts (median, default)
    # Quantile forecasts (recommended for prices)
    raw_predictions, x, *rest = tft.predict(
        prediction_dataloader, mode="raw", return_x=True
    )

    # median prediction, raw_predictions["prediction"] has shape (B batch size, H prediction horizon, number of quantiles (here 3))
    preds = raw_predictions["prediction"]  # (B, H, 3)
    batch_size, horizon, number_of_quantiles = preds.shape

    p10 = preds[:, :, 0].cpu().numpy()
    p50 = preds[:, :, 1].cpu().numpy()
    p90 = preds[:, :, 2].cpu().numpy()

    # Convert predictions to a DataFrame
    # Extract tensors → numpy
    time_idx = x["decoder_time_idx"].cpu().numpy().flatten()  # (B, H)

    pred_df = pd.DataFrame(
        {
            "time_idx": time_idx,
            "horizon": np.tile(np.arange(1, horizon + 1), batch_size),
            "p10": p10.flatten(),
            "p50": p50.flatten(),
            "p90": p90.flatten(),
        }
    )

    # # Add zone
    # zone = x["groups"]["zone"].cpu().numpy()
    # pred_df["zone"] = np.repeat(zone, horizon)

    # Plot timeseries of past prices and forecast with different colours
    prediction_df = df.copy()["price_eur_per_mwh"].to_frame()
    prediction_df["time"] = time_stamps
    prediction_df["label"] = "ENTSOE price"
    prediction_df.loc[prediction_df.index[-MAX_PREDICTION_LENGTH:], "label"] = (
        "TFT forecast"
    )
    prediction_df.loc[
        prediction_df.index[-MAX_PREDICTION_LENGTH:], "price_eur_per_mwh"
    ] = pred_df["p50"].values
    # add range p10-90
    prediction_df["p10"] = prediction_df["price_eur_per_mwh"]
    prediction_df["p90"] = prediction_df["price_eur_per_mwh"]
    prediction_df.loc[prediction_df.index[-MAX_PREDICTION_LENGTH:], "p10"] = pred_df[
        "p10"
    ].values
    prediction_df.loc[prediction_df.index[-MAX_PREDICTION_LENGTH:], "p90"] = pred_df[
        "p90"
    ].values

    logger.info("Prediction dataframe tail:\n%s", pred_df.tail().to_string())
    prediction_df.to_csv(AUTOMATIC_DIR / f"{zone}_prediction.csv")
    prediction_df["time"] = pd.to_datetime(prediction_df["time"])

    plt.figure(figsize=(12, 4))

    for label, g in prediction_df.groupby("label"):
        plt.fill_between(g["time"], g["p90"], g["p10"], alpha=0.3, facecolor="orange")
        plt.plot(g["time"], g["price_eur_per_mwh"], label=label)

    plt.grid("major")
    plt.legend()

    plt.savefig(FIG_DIR / f"{zone}_prediction.jpeg")
    plt.close()

    logger.info("Prediction plot saved to %s", FIG_DIR / f"{zone}_prediction.jpeg")

    return pred_df
