import pandas as pd
from dateutil.parser import isoparse
from datetime import timedelta, date
import xml.etree.ElementTree as ET
import datetime as dt
import requests
import os
import logging

logger = logging.getLogger(__name__)


def get_entsoe_token() -> str:
    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set")

    return token.strip().strip('"').strip("'")


# Bidding Zone (BZN) EIC Mapping
# https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
BIDDING_ZONES = {
    "AL": "10YAL-KESH-----5",  # Albania
    "AT": "10YAT-APG------L",  # Austria
    "BE": "10YBE----------2",  # Belgium
    "BG": "10YCA-BULGARIA-R",  # Bulgaria
    "CH": "10YCH-SWISSGRIDZ",  # Switzerland
    "CY": "10YCY-1001A0003J",  # Cyprus
    "CZ": "10YCZ-CEPS-----N",  # Czech Republic
    "DE-AT-LU": "10Y1001A1001A63L",  # DE-AT-LU (Historical)
    "DE-LU": "10Y1001A1001A82H",  # Germany-Luxembourg
    "DK1": "10YDK-1--------W",  # Denmark Jylland
    "DK2": "10YDK-2--------M",  # Denmark Sjælland
    "EE": "10Y1001A1001A39I",  # Estonia
    "ES": "10YES-REE------0",  # Spain
    "FI": "10YFI-1--------U",  # Finland
    "FR": "10YFR-RTE------C",  # France
    "GB": "10YGB----------A",  # Great Britain
    "GR": "10YGR-HTSO-----Y",  # Greece
    "HR": "10YHR-HEP------M",  # Croatia
    "HU": "10YHU-MAVIR----U",  # Hungary
    "IE-SEM": "10Y1001A1001A59C",  # Ireland (SEM)
    "IT-North": "10Y1001A1001A73I",  # Italy North
    "IT-South": "10Y1001A1001A788",  # Italy South
    "LT": "10YLT-1001A0008Q",  # Lithuania
    "LU": "10YLU-CEGEDEL-NQ",  # Luxembourg
    "LV": "10YLV-1001A00074",  # Latvia
    "ME": "10YCS-CG-TSO---S",  # Montenegro
    "MK": "10YMK-MEPSO----8",  # North Macedonia
    "NL": "10YNL----------L",  # Netherlands
    "NO1": "10YNO-1--------2",  # Norway East
    "NO2": "10YNO-2--------T",  # Norway South
    "NO3": "10YNO-3--------J",  # Norway Central
    "NO4": "10YNO-4--------9",  # Norway North
    "NO5": "10Y1001A1001A48H",  # Norway West
    "PL": "10YPL-AREA-----S",  # Poland
    "PT": "10YPT-REN------W",  # Portugal
    "RO": "10YRO-TEL------P",  # Romania
    "RS": "10YCS-SERBIATSOV",  # Serbia
    "SE1": "10Y1001A1001A44P",  # Sweden North
    "SE2": "10Y1001A1001A45N",  # Sweden North-Central
    "SE3": "10Y1001A1001A46L",  # Sweden South-Central
    "SE4": "10Y1001A1001A47J",  # Sweden South
    "SI": "10YSI-ELES-----P",  # Slovenia
    "SK": "10YSK-SEPS-----K",  # Slovakia
    "TR": "10YTR-TEIAS----W",  # Turkey
    "UA-IPS": "10YUA-WEPS-----0",  # Ukraine IPS
}

import difflib


def get_zone_code(zone: str) -> str:
    """Validates zone and suggests corrections if not found."""
    # 1. Direct Match
    if zone in BIDDING_ZONES:
        return BIDDING_ZONES[zone]

    # 2. Substring Match (e.g., "DE" finds "DE-LU", "DE-AT-LU")
    substring_matches = [k for k in BIDDING_ZONES.keys() if zone.upper() in k]

    # 3. Fuzzy Match (e.g., "Frnce" finds "FR")
    fuzzy_matches = difflib.get_close_matches(
        zone.upper(), BIDDING_ZONES.keys(), n=3, cutoff=0.6
    )

    # Combine suggestions and remove duplicates
    suggestions = list(set(substring_matches + fuzzy_matches))

    if suggestions:
        suggestion_str = " or ".join([f"'{s}'" for s in suggestions])
        raise ValueError(f"Unknown zone '{zone}'. Did you mean {suggestion_str}?")
    else:
        raise ValueError(
            f"Unknown zone '{zone}'. Please check the BIDDING_ZONES dictionary. \n Only available for these bidding zones {BIDDING_ZONES.keys()}"
        )


def load_prices(zone: str, start, end) -> pd.DataFrame:
    """
    Returns hourly or 15-min electricity prices indexed by UTC timestamp
    """

    time_start = start.strftime("%Y%m%d0000")
    time_end = end.strftime("%Y%m%d0000")

    # Look up the code for bidding zone
    zone_code = get_zone_code(zone)
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
