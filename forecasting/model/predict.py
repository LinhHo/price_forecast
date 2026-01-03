import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from forecasting.config import (
    AUTOMATIC_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    MAX_PREDICTION_LENGTH,
    MAX_ENCODER_LENGTH,
)
from forecasting.data.open_meteo import load_forecast
from forecasting.data.entsoe import load_prices
from forecasting.features.build_features import (
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
    last_time_idx = np.load(f"artifacts/{zone}_last_time_idx.npy")

    df = prepare_forecast_df(zone, last_time_idx)
    assert df["time_idx"].is_monotonic_increasing

    logger.info(
        "Starting predicting | zone=%s | start=%s | end=%s",
        zone,
        df.index.min(),
        df.index.max(),
    )

    training = TimeSeriesDataSet.load(AUTOMATIC_DIR / f"{zone}_training_dataset")

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
        AUTOMATIC_DIR / "tft_price_model.ckpt", weights_only=False
    )

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

    # median prediction
    median_price = raw_predictions["prediction"][:, :, 1]

    # Convert predictions to a DataFrame
    # Extract tensors → numpy
    y_pred = median_price.cpu().numpy()  # (B, H)
    time_idx = x["decoder_time_idx"].cpu().numpy()  # (B, H)

    batch_size, horizon = y_pred.shape

    pred_df = pd.DataFrame(
        {
            "time_idx": time_idx.reshape(-1),
            "horizon": np.tile(np.arange(1, horizon + 1), batch_size),
            "y_pred": y_pred.reshape(-1),
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
    ] = pred_df["y_pred"].values

    plt.figure(figsize=(12, 4))

    for label, g in prediction_df.groupby("label"):
        plt.plot(g.index, g["price_eur_per_mwh"], label=label)

    plt.legend()
    plt.grid(True)
    plt.save_fig(OUTPUT_DIR / "Prediction.jpeg")

    return pred_df  # model.predict(...)
