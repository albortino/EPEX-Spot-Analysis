import pandas as pd
import io
from datetime import date
import streamlit as st
from methods.config import MIN_DATE

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to an Excel file in-memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Create a copy to avoid modifying the cached dataframe in place
        df_temp = df.copy()
        df_temp["timestamp"] = df_temp["timestamp"].dt.tz_localize(None)
        df_temp.to_excel(writer, index=False, sheet_name="AnalysisData")
    processed_data = output.getvalue()
    return processed_data

@st.cache_data(ttl=3600)
def get_min_max_date(df: pd.DataFrame) -> tuple[date, date]:
    """Returns the minimum and maximum dates from a DataFrame with a timestamp column."""
    min_val_date = df["timestamp"].min().date()
    if min_val_date < MIN_DATE:
        min_val_date = MIN_DATE
        
    max_val_date = df["timestamp"].max().date()
    
    return min_val_date, max_val_date

def get_intervals_per_day(df: pd.DataFrame) -> int:
    """Calculates the most frequent number of data intervals per day."""
    if df.empty:
        return 24 # Default to hourly if no data
    
    # Ensure a date column exists for grouping
    if "date" not in df.columns:
        df_temp = df.copy()
        df_temp["date"] = df_temp["timestamp"].dt.date
    else:
        df_temp = df

    # Calculate the mode of interval counts per day
    intervals = df_temp.groupby("date").size().mode()
    return intervals.iloc[0] if not intervals.empty else 24

def get_aggregation_config(df: pd.DataFrame, resolution: str) -> dict:
    """Returns the aggregation configuration based on the selected resolution."""
    resolution_config = {
        "Hourly": {"grouper": df["timestamp"].dt.hour, "x_axis_map": {i: str(i) for i in range(24)}, "name": "Hour of Day"},
        "Weekly": {"grouper": df["timestamp"].dt.dayofweek, "x_axis_map": {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}, "name": "Day of Week"},
        "Monthly": {"grouper": df["timestamp"].dt.month, "x_axis_map": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}, "name": "Month"},
    }

    config = resolution_config[resolution]
    return config

def calculate_granular_data(df: pd.DataFrame) -> bool:
    """
    Check if data is granular enough for a meaningful flexible cost comparison.
    This is true for hourly or 15-min data, but not for daily data resampled to hourly.
    """
    non_zero_consumption_df = df[df['consumption_kwh'] > 0.001]
    is_granular_data = True
    if not non_zero_consumption_df.empty:
        # If all non-zero consumption occurs only at midnight, it's likely daily data
        # that has been resampled, making a flexible cost comparison misleading.
        if (non_zero_consumption_df['timestamp'].dt.hour == 0).all():
            is_granular_data = False
            
    return is_granular_data

@st.cache_data(ttl=3600)
def filter_dataframe(df: pd.DataFrame, start_date: date, end_date: date):
    """Filters a DataFrame based on start and end dates."""
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    return df.loc[mask]