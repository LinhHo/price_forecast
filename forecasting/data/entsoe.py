import pandas as pd
from dateutil.parser import isoparse
from datetime import timedelta, date
import xml.etree.ElementTree as ET
import datetime as dt
import requests
import os
import logging


def get_entsoe_token() -> str:
    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set")

    return token.strip().strip('"').strip("'")


logger = logging.getLogger(__name__)


def load_prices(zone: str, start, end, is_training: bool = True) -> pd.DataFrame:
    """
    Returns hourly or 15-min electricity prices indexed by UTC timestamp
    """

    # 1. Convert string inputs to datetime objects if they aren't already
    if isinstance(start, str):
        # Adjust the format "%Y-%m-%d" to match how you write years in your config
        start = dt.strptime(start, "%Y-%m-%d")
    if isinstance(end, str):
        end = dt.datetime.strptime(end, "%Y-%m-%d")

    if is_training:
        start_dt = start
        end_dt = end
    else:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=7)

    time_start = start_dt.strftime("%Y%m%d0000")
    time_end = end_dt.strftime("%Y%m%d0000")

    zone_code = f"10Y{zone}----------L"
    entsoe_token = get_entsoe_token()

    url = (
        "https://web-api.tp.entsoe.eu/api"
        f"?documentType=A44"
        f"&out_Domain={zone_code}"
        f"&in_Domain={zone_code}"
        f"&periodStart={time_start}"
        f"&periodEnd={time_end}"
        f"&securityToken={entsoe_token}"
    )

    response = requests.get(url)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"ns": root.tag.split("}")[0].strip("{")}

    records = []

    for ts in root.findall("ns:TimeSeries", ns):
        for period in ts.findall("ns:Period", ns):

            start_time = isoparse(period.find("ns:timeInterval/ns:start", ns).text)

            resolution_text = period.find("ns:resolution", ns).text  # PT60M / PT15M
            minutes = int(resolution_text.replace("PT", "").replace("M", ""))

            for point in period.findall("ns:Point", ns):
                position = int(point.find("ns:position", ns).text)
                price = float(point.find("ns:price.amount", ns).text)

                timestamp = start_time + timedelta(minutes=(position - 1) * minutes)

                records.append(
                    {
                        "timestamp": pd.Timestamp(timestamp).tz_convert("UTC"),
                        "price_eur_per_mwh": price,
                    }
                )

    df_price = pd.DataFrame(records).set_index("timestamp").sort_index()

    return df_price
