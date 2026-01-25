from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

TARGET = "price_eur_per_mwh"


def build_dataset(df, max_encoder_length, max_prediction_length):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["zone"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["zone"],
        static_reals=[],
        time_varying_known_reals=[
            "time_idx",
            "t2m",
            "ssrd",
            "u100",
            "v100",
            "hour_sin",  # add cyclic time to account for cyclical nature of time
            "hour_cos",
            "dow_sin",  # day of week
            "dow_cos",
            "doy_sin",  # day of year
            "doy_cos",
            "month_sin",
            "month_cos",
            "day_of_month",
            "price_7d_mean",
            "price_7d_std",
            "price_is_missing",
        ],
        time_varying_unknown_reals=[TARGET],
        time_varying_known_categoricals=[
            "is_holiday",
            "hour_of_day",
            "day_of_year",
            "day_of_week",
            "month",
        ],
        target_normalizer=GroupNormalizer(
            groups=["zone"],
            transformation="log",  # electricity price is heavy-tail, default: None,
        ),
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )
