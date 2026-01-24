"""
Add time features and static features to the dataframe
"""

# build_features.py
import pandas as pd
import numpy as np
from forecasting.data.holidays import HOLIDAYS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month

    # Hour of Day (Cycle: 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # Day of Week (Cycle: 7)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Day of Year (Cycle: 365.25 to account for leap years)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 365.25)

    # Month (Cycle: 12)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # Except for day_of_month, different month's length disrupts cyclic nature
    for var in ["hour_of_day", "day_of_week", "day_of_year", "month"]:
        df[var] = df[var].astype(str)

    return df


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = (
        df.index.to_series()
        .apply(lambda x: "yes" if (x.month, x.day) in HOLIDAYS else "no")
        .astype("category")
    )
    return df


def add_zone(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    df = df.copy()
    df["zone"] = zone
    return df


def add_features(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    df = df.copy()
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_zone(df, zone)

    # mark non-missing target for training
    if "price_eur_per_mwh" in df.columns:
        df["price_is_missing"] = df["price_eur_per_mwh"].isna().astype(int)
    else:
        df["price_is_missing"] = 1

    return df
