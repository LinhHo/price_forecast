import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet


def build_tft_dataset(
    df: pd.DataFrame,
    is_training=True,
    max_encoder_length=168,  # use past 7 days, electricity prices have weekly pattern
    max_prediction_length=24,  # predict next 24 hours
):
    
    # Features for TFT setup (unchanged later)
    target = "price_eur_per_mwh"
    time_varying_known_reals = [
        "time_idx",
        "hour_of_day",
        "day_of_year",
        "day_of_week",
        "day_of_month",
        "month",
        "t2m",
        "ssrd",
        "u100",
        "v100",
    ]

    time_varying_unknown_reals = ["price_eur_per_mwh"]

    # Static categorical features (don't change over time)
    static_categoricals = ["zone"]
    static_reals = []
    time_varying_known_categoricals = ["is_holiday"]

    # normalizer using group zone as different zone can have big or small generation and price
    target_normalizer = GroupNormalizer(
        groups=["zone"],
        transformation=None,  # "softplus" # softplus assumes positive values, which is not always true for electricity prices # very important for prices
    )

    # do NOT compute training_cutoff inside blindly. This lets you reuse it for prediction
    # Predicts only the last horizon. This avoids data leakage and ensures realistic forecasting
    if is_training:
        df = df[df.time_idx <= df.time_idx.max() - max_prediction_length]

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["zone"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,  # added
        target_normalizer=target_normalizer,  # None,  # we'll use built-in scaling later
        add_relative_time_idx=True,
        add_target_scales=False,  # Do NOT use add_target_scales=True with target_normalizer=None
        add_encoder_length=True,
    )
    return training  # TimeSeriesDataSet(...)