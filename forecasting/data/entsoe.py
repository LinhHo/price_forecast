import pandas as pd
from dateutil.parser import isoparse
from datetime import timedelta, date
import xml.etree.ElementTree as ET
import requests
from forecasting.config import training_start, training_end


def load_prices(zone: str, is_training: bool = True) -> pd.DataFrame:
    """
    Returns hourly or 15-min electricity prices indexed by UTC timestamp
    """

    if is_training:
        start_dt = training_start
        end_dt = training_end
    else:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=7)

    time_start = start_dt.strftime("%Y%m%d0000")
    time_end = end_dt.strftime("%Y%m%d0000")

    zone_code = f"10Y{zone}----------L"

    url = (
        "https://web-api.tp.entsoe.eu/api"
        f"?documentType=A44"
        f"&out_Domain={zone_code}"
        f"&in_Domain={zone_code}"
        f"&periodStart={time_start}"
        f"&periodEnd={time_end}"
        f"&securityToken=YOUR_TOKEN"
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
                        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
                        "price_eur_per_mwh": price,
                    }
                )

    df_price = pd.DataFrame(records).set_index("timestamp").sort_index()

    return df_price
