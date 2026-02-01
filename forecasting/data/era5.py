# era5.py
# import geopandas as gpd
import xarray as xr
import pandas as pd
import time
import os
import logging


def get_era5_token() -> str:
    token = os.getenv("ERA5_TOKEN")
    if not token:
        raise RuntimeError("ERA5_TOKEN not set")

    return token.strip().strip('"').strip("'")


logger = logging.getLogger(__name__)


# (min_lon, min_lat, max_lon, max_lat)
ZONE_BOUNDS = {
    "NL": (3.0, 50.5, 7.5, 53.7),
    "BE": (2.5, 49.5, 6.5, 51.7),
    "DE-LU": (5.5, 47.0, 15.5, 55.2),
    "FR": (-5.5, 41.0, 9.8, 51.5),
    "ES": (-9.5, 35.5, 3.5, 43.8),
    "IT-North": (6.0, 44.0, 13.5, 47.5),
    "IT-South": (12.0, 37.0, 18.5, 42.5),
    "NO1": (7.5, 58.0, 12.5, 61.5),
    "SE3": (11.0, 56.5, 18.5, 59.5),
    "GB": (-7.8, 49.8, 2.2, 59.0),
}

# from forecasting.data.zones import ZONE_BOUNDS


def get_bounds_zone(zone: str):
    try:
        return ZONE_BOUNDS[zone]
    except KeyError:
        raise ValueError(
            f"Zone '{zone}' not supported. "
            f"Available zones: {sorted(ZONE_BOUNDS.keys())}"
        )


# def get_bounds_zone(zone: str):
#     # Load the world GeoJSON from electricitymap (app.electricitymaps.com)
#     # https://github.com/electricitymaps/electricitymaps-contrib
#     geojson_path = "https://raw.githubusercontent.com/electricitymaps/electricitymaps-contrib/master/web/geo/world.geojson"
#     gdf = gpd.read_file(geojson_path)

#     # bidding zone - check if zoneName or countryKey
#     zone_gdf = gdf[gdf["zoneName"] == zone]
#     if zone_gdf.empty:
#         raise ValueError(f"Zone {zone} not found")

#     return zone_gdf.total_bounds  # (minx, miny, maxx, maxy)


def open_era5_zarr(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return xr.open_dataset(url, engine="zarr")
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.error(f"ERA5 open failed (attempt {attempt+1}), retrying...")
            time.sleep(delay)


def load_era5(zone: str, start, end) -> pd.DataFrame:
    """
    Load ERA5 data for training the model, same length as ENTSOE price data

    Returns dataframe with t2m, ssrd, u100, v100 indexed by timestamp
    """
    bounds = get_bounds_zone(zone)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Token
    era5_token = get_era5_token()
    url = f"https://edh:{era5_token}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"

    ds = open_era5_zarr(url)

    era5_vars_units = {"t2m": "°C", "ssrd": "J/m²", "u100": "m/s", "v100": "m/s"}
    df = pd.DataFrame()

    for v in era5_vars_units.keys():
        da = ds[v].sel(
            latitude=slice(max_lat, min_lat),
            longitude=slice(min_lon, max_lon),
            valid_time=slice(start, end),
        )
        df[v] = da.mean(dim=["latitude", "longitude"]).to_series()
        df[v].attrs["units"] = era5_vars_units[v]

    df["t2m"] -= 273.15
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index).tz_localize("UTC")

    return df
