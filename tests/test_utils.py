# python -m pytest tests/test_utils.py

import io
from datetime import date
import pandas as pd

from src.utils import (
    get_intervals_per_day,
    has_granular_resolution,
    filter_dataframe,
    filter_by_quarter,
    get_min_max_date,
    to_excel
)


def test_get_intervals_per_day():
    """Verify frequency detection for 15-minute, hourly, and daily data."""
    # 15-minute series
    df_15m = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    })
    assert get_intervals_per_day(df_15m) == 96

    # Hourly series
    df_1h = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    })
    assert get_intervals_per_day(df_1h) == 24

    # Daily series
    df_1d = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    })
    assert get_intervals_per_day(df_1d) == 1

    # Empty / small fallback
    assert get_intervals_per_day(pd.DataFrame()) == 24


def test_has_granular_resolution():
    """Verify granularity detection distinguishes intraday from daily data."""
    # Granular hourly data
    df_hourly = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
        "consumption_kwh": [1.0] * 24
    })
    assert has_granular_resolution(df_hourly) is True

    # Daily data with intervals_per_day = 1
    df_daily = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC"),
        "consumption_kwh": [10.0] * 10
    })
    assert has_granular_resolution(df_daily) is False

    # Hourly data where only hour 0 has non-zero values
    df_zero_padded = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
        "consumption_kwh": [10.0 if i % 24 == 0 else 0.0 for i in range(48)]
    })
    assert has_granular_resolution(df_zero_padded) is False


def test_filter_dataframe():
    """Verify filtering timestamps within a closed date window."""
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps, "consumption_kwh": range(10)})

    filtered = filter_dataframe(df, date(2024, 1, 3), date(2024, 1, 6))
    assert len(filtered) == 4
    assert filtered["timestamp"].dt.date.min() == date(2024, 1, 3)
    assert filtered["timestamp"].dt.date.max() == date(2024, 1, 6)


def test_filter_by_quarter():
    """Verify quarter filters only preserve corresponding calendar months."""
    timestamps = pd.date_range("2024-01-01", periods=12, freq="MS", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps, "consumption_kwh": range(12)})

    assert len(filter_by_quarter(df, "All")) == 12
    q1 = filter_by_quarter(df, "Q1")
    assert set(q1["timestamp"].dt.month) == {1, 2, 3}
    q3 = filter_by_quarter(df, "Q3")
    assert set(q3["timestamp"].dt.month) == {7, 8, 9}


def test_get_min_max_date():
    """Verify min and max date extraction."""
    timestamps = pd.date_range("2023-05-10", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})
    start, end = get_min_max_date(df, today_as_max=False)
    assert start == date(2023, 5, 10)
    assert end == date(2023, 5, 14)


def test_to_excel():
    """Verify in-memory Excel generation produces a valid spreadsheet."""
    import zipfile
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
        "col_a": [1, 2]
    })
    excel_bytes = to_excel(df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(excel_bytes)) as zf:
        assert "xl/workbook.xml" in zf.namelist()
