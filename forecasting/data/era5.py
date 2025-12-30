# era5.py
import geopandas as gpd
import xarray as xr
import pandas as pd
import time
import os
from forecasting.config import training_start, training_end

try:
    ERA5_TOKEN = os.getenv("ERA5_TOKEN")
except ImportError:
    # Local development (ignored by git)
    from forecasting.config import Config

    ERA5_TOKEN = Config.ERA5_TOKEN

if not ERA5_TOKEN:
    raise RuntimeError(
        "ERA5_TOKEN not found. " "Set it via environment variable or config_local.py"
    )


import logging

logger = logging.getLogger(__name__)


def get_bounds_zone(zone: str):
    # Load the world GeoJSON from electricitymap (app.electricitymaps.com)
    # https://github.com/electricitymaps/electricitymaps-contrib
    geojson_path = "https://raw.githubusercontent.com/electricitymaps/electricitymaps-contrib/master/web/geo/world.geojson"
    gdf = gpd.read_file(geojson_path)

    # bidding zone - check if zoneName or countryKey
    zone_gdf = gdf[gdf["zoneName"] == zone]
    if zone_gdf.empty:
        raise ValueError(f"Zone {zone} not found")

    return zone_gdf.total_bounds  # (minx, miny, maxx, maxy)


def open_era5_zarr(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return xr.open_dataset(url, engine="zarr")
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.error(f"ERA5 open failed (attempt {attempt+1}), retrying...")
            time.sleep(delay)


def load_era5(zone: str) -> pd.DataFrame:
    """
    Load ERA5 data for training the model, same length as ENTSOE price data

    Returns dataframe with t2m, ssrd, u100, v100 indexed by timestamp
    """
    bounds = get_bounds_zone(zone)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Token
    url = f"https://edh:{ERA5_TOKEN}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"

    ds = open_era5_zarr(url)
    # ds = xr.open_dataset(
    #     f"https://edh:{ERA5_TOKEN}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
    #     engine="zarr",
    # )

    era5_vars_units = {"t2m": "°C", "ssrd": "J/m²", "u100": "m/s", "v100": "m/s"}
    df = pd.DataFrame()

    for v in era5_vars_units.keys():
        da = ds[v].sel(
            latitude=slice(max_lat, min_lat),
            longitude=slice(min_lon, max_lon),
            valid_time=slice(training_start, training_end),
        )
        df[v] = da.mean(dim=["latitude", "longitude"]).to_series()
        df[v].attrs["units"] = era5_vars_units[v]

    df["t2m"] -= 273.15
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index).tz_localize("UTC")

    return df
