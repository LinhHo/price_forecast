import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder

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


from torch.serialization import add_safe_globals
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

add_safe_globals([TimeSeriesDataSet, pd.DataFrame, GroupNormalizer])
# torch.serialization.add_safe_globals([pytorch_forecasting.data.encoders.GroupNormalizer])


def predict_next_24h(zone: str):
    # load last_time_idx from training
    last_time_idx = np.load(AUTOMATIC_DIR / f"{zone}_last_time_idx.npy")

    df = prepare_forecast_df(zone, last_time_idx)
    assert df["time_idx"].is_monotonic_increasing

    logger.info(
        "Starting predicting | zone=%s | start=%s | end=%s",
        zone,
        df.index.min(),
        df.index.max(),
    )

    # Append forecast df to training history (required for TFT)
    training = TimeSeriesDataSet.load(AUTOMATIC_DIR / f"{zone}_training_dataset.pt")

    # df_train = pd.read_parquet(AUTOMATIC_DIR / f"{zone}_training_data.parquet")
    # df_forecast = prepare_forecast_df(zone, last_time_idx)
    # df = pd.concat([df_train, df_forecast]).sort_index()
    # # time_idx must be continuous
    # df["time_idx"] = np.arange(len(df))
    # # fill with dummy 0, PyTorch Forecasting does not infer “missing = predict this”, it uses max_prediction_length
    # df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(0.0)

    # with open(AUTOMATIC_DIR / f"{zone}_dataset_params.json") as f:
    #     params = json.load(f)

    # # only train on df_train, df_forecast has NaN values for target (price)
    # training = TimeSeriesDataSet(df_train, **params)

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
    # predictions = tft.predict(prediction_dataloader)

    # Quantile forecasts (recommended for prices)
    raw_predictions, x, *rest = tft.predict(
        prediction_dataloader, mode="raw", return_x=True
    )

    # median prediction, raw_predictions["prediction"] has shape (B batch size, H prediction horizon, number of quantiles (here 3))
    # median_price = raw_predictions["prediction"][:, :, 1]
    preds = raw_predictions["prediction"]  # (B, H, 3)
    batch_size, horizon, number_of_quantiles = preds.shape

    p10 = preds[:, :, 0].cpu().numpy()
    p50 = preds[:, :, 1].cpu().numpy()
    p90 = preds[:, :, 2].cpu().numpy()

    # Convert predictions to a DataFrame
    # Extract tensors → numpy
    # y_pred = median_price.cpu().numpy()  # (B, H)
    time_idx = x["decoder_time_idx"].cpu().numpy()  # (B, H)

    pred_df = pd.DataFrame(
        {
            "time_idx": time_idx.flatten(),
            "timestamp": x["decoder_time"].cpu().numpy().flatten(),
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
    prediction_df["label"] = "ENTSOE price"
    prediction_df.loc[prediction_df.index[-MAX_PREDICTION_LENGTH:], "label"] = (
        "TFT forecast"
    )
    prediction_df.loc[
        prediction_df.index[-MAX_PREDICTION_LENGTH:], "price_eur_per_mwh"
    ] = pred_df["p50"].values

    logger.info("Prediction dataframe tail:\n%s", pred_df.tail().to_string())

    plt.figure(figsize=(12, 4))

    for label, g in prediction_df.groupby("label"):
        plt.plot(g.index, g["price_eur_per_mwh"], label=label)

    plt.legend()
    plt.grid(True)
    plt.savefig(FIG_DIR / f"{zone}_Prediction.jpeg")
    plt.close()

    logger.info("Prediction plot saved to %s", FIG_DIR / f"{zone}_Prediction.jpeg")

    return pred_df  # model.predict(...)
