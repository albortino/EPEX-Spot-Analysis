import pandas as pd
import streamlit as st
import requests
import os
import time as sleep_time
from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from src.config import SPOT_PRICE_CACHE_FILE, LOCAL_TIMEZONE, CACHE_FOLDER
from src.file_parser import ConsumptionDataParser
from src.i18n import t
from src.logger import logger

# --- Spot Price Data Handling ---

def _load_from_cache(file_path: str) -> pd.DataFrame:
    """Loads stored EPEX prices from the local cache."""
    try:
        df_cache = pd.read_csv(file_path, parse_dates=["timestamp"])
        df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], utc=True)
        return df_cache
    except Exception as e:
        st.warning(f"Could not read or parse cache file '{file_path}'. Refetching data. Error: {e}")
        return pd.DataFrame()

REQUEST_TIMEOUT_SECONDS = 10
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _fetch_spot_data(country: str, start: date, end: date, cache_filename: str) -> pd.DataFrame:
    """Fetches spot data from the aWATTar API for a given date range."""
    base_url = f"https://api.awattar.{country}/v1/marketdata"
    start_dt = datetime.combine(start, time.min, tzinfo=ZoneInfo(LOCAL_TIMEZONE)).astimezone(ZoneInfo("UTC"))
    end_dt = datetime.combine(end + pd.Timedelta(days=1), time.min, tzinfo=ZoneInfo(LOCAL_TIMEZONE)).astimezone(ZoneInfo("UTC"))
    params = {"start": int(start_dt.timestamp() * 1000), "end": int(end_dt.timestamp() * 1000)}

    try:
        logger.log("Fetching spot price data from aWATTar API.", severity=1)
        response = None
        last_error = None
        for attempt in range(2):
            try:
                response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt == 0:
                    sleep_time.sleep(0.25)
        if response is None or last_error and not response.ok:
            raise last_error  # type: ignore[misc]
        data = response.json().get("data")
        if not data:
            st.warning("API returned no data for the selected period.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
        VAT_FACTOR = 1.2 if country == "at" else 1.19
        df["spot_price_eur_kwh"] = df["marketprice"] / 1000 * VAT_FACTOR  # Convert Eur/MWh to Eur/kWh and add 20% VAT
        df_to_return = df[["timestamp", "spot_price_eur_kwh"]]

        existing = _load_from_cache(cache_filename) if os.path.exists(cache_filename) else pd.DataFrame()
        if not existing.empty:
            df_to_return = pd.concat([existing, df_to_return]).drop_duplicates("timestamp").sort_values("timestamp")
        df_to_return.to_csv(cache_filename, index=False)
        return df_to_return[(df_to_return["timestamp"].dt.date >= start) & (df_to_return["timestamp"].dt.date <= end)]
    except requests.exceptions.RequestException:
        st.warning("Spot prices could not be refreshed. A cached price range will be used when available.")
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API.")
    return pd.DataFrame()

@st.cache_data
def get_spot_data(country: str, start: date, end: date) -> pd.DataFrame:
    """Fetches spot market price data, using a local cache to avoid redundant API calls."""
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    country_cache_filename = f"{country}_{SPOT_PRICE_CACHE_FILE}"
    cache_path = os.path.join(CACHE_FOLDER, country_cache_filename)

    cached_slice = pd.DataFrame()
    if os.path.exists(cache_path):
        df_cache = _load_from_cache(cache_path)
        if not df_cache.empty:
            min_cached = df_cache["timestamp"].min().date()
            max_cached = df_cache["timestamp"].max().date()
            if min_cached <= start and max_cached >= end:
                logger.log("Loading spot prices from cache.", severity=1)
                return df_cache[(df_cache["timestamp"].dt.date >= start) & (df_cache["timestamp"].dt.date <= end)]
            cached_slice = df_cache[(df_cache["timestamp"].dt.date >= start) & (df_cache["timestamp"].dt.date <= end)]

    logger.log("Cache insufficient or missing. Fetching new data from aWATTar.", severity=1)
    fetched = _fetch_spot_data(country, start, end, cache_path)
    return fetched if not fetched.empty else cached_slice

# --- Consumption Data Handling ---

@st.cache_data(ttl=3600)
def process_consumption_data(uploaded_file) -> pd.DataFrame:
    """Loads and processes the user's consumption CSV using the dedicated parser."""
    if uploaded_file is None:
        return pd.DataFrame()
    if getattr(uploaded_file, "size", 0) > MAX_UPLOAD_BYTES:
        st.error("The upload exceeds the 20 MB public-server limit.")
        return pd.DataFrame()
    try:
        parser = ConsumptionDataParser(local_timezone=LOCAL_TIMEZONE)
        df = parser.parse_file(uploaded_file)
        if df.empty:
            st.info(t("upload_format_unsupported"), icon="ℹ️")
        return df.convert_dtypes()
    except Exception as e:
        error_id = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.log(f"Upload parsing error {error_id}: {e}", severity=1)
        st.info(t("upload_format_unsupported"), icon="ℹ️")
        return pd.DataFrame()

# --- Data Merging ---

@st.cache_data(ttl=3600)
def merge_consumption_with_prices(df_consumption: pd.DataFrame, df_spot_prices: pd.DataFrame) -> pd.DataFrame:
    """Merges consumption data with spot prices, aligning timestamps."""
    logger.log("Merging consumption data with spot prices.")
    if df_consumption.empty or df_spot_prices.empty:
        return pd.DataFrame()

    # Align timestamps by merging on the nearest previous hour's price
    df_merged = pd.merge_asof(
        df_consumption.sort_values("timestamp"),
        df_spot_prices.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("59min")
    )

    # Add date column and localize timestamp for further analysis
    if not df_merged.empty:
        df_merged["timestamp"] = df_merged["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        df_merged["date"] = df_merged["timestamp"].dt.date

    return df_merged.convert_dtypes()
