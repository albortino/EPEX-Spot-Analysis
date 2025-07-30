# data_loader.py

import pandas as pd
import streamlit as st
import requests
import os
from datetime import datetime, date
from config import SPOT_PRICE_CACHE_FILE, AWATTAR_COUNTRY, LOCAL_TIMEZONE, DATE_FORMAT
from parser import ConsumptionDataParser

# --- Spot Price Data Handling ---

def _load_from_cache(file_path: str) -> pd.DataFrame:
    """Loads stored EPEX prices from the local cache."""
    try:
        df_cache = pd.read_csv(file_path, parse_dates=["timestamp"])
        df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], utc=True)
        return df_cache
    except Exception as e:
        st.warning(f"Could not read or parse cache file '{SPOT_PRICE_CACHE_FILE}'. Refetching data. Error: {e}")
        return pd.DataFrame()

def _fetch_spot_data(start: date, end: date) -> pd.DataFrame:
    """Fetches spot data from the aWATTar API for a given date range."""
    base_url = f"https://api.awattar.{AWATTAR_COUNTRY}/v1/marketdata"
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end + pd.Timedelta(days=1), datetime.min.time())
    params = {"start": int(start_dt.timestamp() * 1000), "end": int(end_dt.timestamp() * 1000)}
    
    try:
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Will fetch data from Awattar.")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            st.warning("API returned no data for the selected period.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
        df["spot_price_eur_kwh"] = df["marketprice"] / 1000 * 1.2  # Convert Eur/MWh to Eur/kWh and add 20% VAT
        df_to_return = df[["timestamp", "spot_price_eur_kwh"]]
        df_to_return.to_csv(SPOT_PRICE_CACHE_FILE, index=False)
        return df_to_return
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch spot price data: {e}")
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API.")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_spot_data(start: date, end: date) -> pd.DataFrame:
    """
    Fetches spot market price data, using a local cache to avoid redundant API calls.
    """
    now = datetime.now().strftime(DATE_FORMAT)
    if os.path.exists(SPOT_PRICE_CACHE_FILE):
        df_cache = _load_from_cache(SPOT_PRICE_CACHE_FILE)
        if not df_cache.empty:
            min_cached = df_cache["timestamp"].min().date()
            max_cached = df_cache["timestamp"].max().date()
            if min_cached <= start and max_cached >= end:
                print(f"{now}: Loading spot prices from cache.")
                return df_cache[(df_cache["timestamp"].dt.date >= start) & (df_cache["timestamp"].dt.date <= end)]
    
    print(f"{now}: Cache insufficient or missing. Will fetch data from aWATTar.")
    return _fetch_spot_data(start, end)

# --- Consumption Data Handling ---

@st.cache_data(ttl=3600)
def process_consumption_data(uploaded_file, aggregation_level: str = "h") -> pd.DataFrame:
    """Loads and processes the user's consumption CSV using the dedicated parser."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        parser = ConsumptionDataParser(local_timezone=LOCAL_TIMEZONE)
        df = parser.parse_file(uploaded_file, aggregation_level)
        if df.empty:
            st.error("Could not parse the CSV file. Please ensure it is from a supported provider or in the default format.")
        return df
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the file: {e}")
        return pd.DataFrame()

# --- Data Merging ---

@st.cache_data(ttl=3600)
def merge_consumption_with_prices(df_consumption: pd.DataFrame, df_spot_prices: pd.DataFrame) -> pd.DataFrame:
    """Merges consumption data with spot prices, aligning timestamps."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Merging Consumption with Prices")
    if df_consumption.empty or df_spot_prices.empty:
        return pd.DataFrame()

    # Align timestamps by merging on the nearest previous hour's price
    df_merged = pd.merge_asof(
        df_consumption.sort_values("timestamp"),
        df_spot_prices.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("59min")
    ).dropna()
    
    # Add date column and localize timestamp for further analysis
    if not df_merged.empty:
        df_merged["timestamp"] = df_merged["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        df_merged["date"] = df_merged["timestamp"].dt.date
        
    return df_merged