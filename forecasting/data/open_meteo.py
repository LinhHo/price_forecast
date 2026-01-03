"""
Load real-time weather forecast for prediction
24h forecast for today (starting at the time run) + 7 past days for max_encoder_length
Open-meteo data (with API) https://open-meteo.com/en/docs?forecast_days=1&hourly=temperature_2m,shortwave_radiation,wind_speed_80m&models=best_match
"""

# open_meteo.py
import pandas as pd
import numpy as np
import requests
from forecasting.data.era5 import get_bounds_zone


def load_forecast(zone: str) -> pd.DataFrame:
    """
    Returns past 7 days + next 24h weather with correct time_idx continuation
    """

    bounds = get_bounds_zone(zone)
    min_lon, min_lat, max_lon, max_lat = bounds

    sel_lat = (min_lat + max_lat) / 2
    sel_lon = (min_lon + max_lon) / 2

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={sel_lat}"
        f"&longitude={sel_lon}"
        "&hourly=temperature_2m,shortwave_radiation,"
        "wind_speed_80m,wind_speed_120m,"
        "wind_direction_80m,wind_direction_120m"
        "&models=best_match"
        "&forecast_days=1"
        "&past_days=7"
    )

    df = (
        pd.DataFrame(requests.get(url).json()["hourly"])
        .assign(time=lambda x: pd.to_datetime(x["time"], utc=True))
        .set_index("time")
    )

    # Wind interpolation
    wind_speed_100 = (df["wind_speed_80m"] + df["wind_speed_120m"]) / 2
    wind_dir_100 = (df["wind_direction_80m"] + df["wind_direction_120m"]) / 2

    df["u100"] = wind_speed_100 * np.sin(np.deg2rad(wind_dir_100))
    df["v100"] = wind_speed_100 * np.cos(np.deg2rad(wind_dir_100))

    df = df.rename(
        columns={
            "temperature_2m": "t2m",  # already °C
            "shortwave_radiation": "ssrd",
        }
    )

    df = df[["t2m", "ssrd", "u100", "v100"]]

    return df
