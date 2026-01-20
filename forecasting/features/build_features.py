"""
Add time features and static features to the dataframe
"""

# build_features.py
import pandas as pd
from forecasting.data.holidays import HOLIDAYS



def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
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
    return df