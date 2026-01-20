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
            "hour_of_day",
            "day_of_year",
            "day_of_week",
            "day_of_month",
            "month",
            "t2m",
            "ssrd",
            "u100",
            "v100",
            "is_holiday",
            "price_is_missing",
        ],
        time_varying_unknown_reals=[TARGET],
        time_varying_known_categoricals=["is_holiday"],
        target_normalizer=GroupNormalizer(
            groups=["zone"],
            transformation=None,
        ),
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )
