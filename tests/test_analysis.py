import pandas as pd
import pytest
from src.analysis import (
    classify_usage,
    compute_cost_comparison_data,
    compute_cumulative_savings_data,
    compute_absence_data,
    compute_peak_timing_score,
    compute_consumption_quartiles
)


@pytest.fixture
def multi_day_15m_df():
    """Generates 7 days of 15-minute data with varied consumption and spot prices."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=7 * 96, freq="15min", tz="UTC")
    # Base load 0.1 kWh, with midday and evening peaks
    consumption = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        val = 0.10
        if 11 <= hour <= 13:
            val += 0.80  # Noon peak
        elif 18 <= hour <= 20:
            val += 0.50  # Evening peak
        consumption.append(val)

    # Spot prices varying by hour
    spot_prices = [0.05 + 0.10 * ((ts.hour % 6) / 6.0) for ts in timestamps]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kwh": consumption,
        "spot_price_eur_kwh": spot_prices,
        "date": timestamps.date,
        "total_cost_flexible": [c * p + 0.001 for c, p in zip(consumption, spot_prices)],
        "total_cost_static": [c * 0.15 + 0.001 for c in consumption]
    })
    return df


def test_classify_usage_partitions_energy(multi_day_15m_df):
    """Verify that classify_usage decomposes total energy cleanly into base, regular, and peak."""
    df_classified, base_thresh, peak_thresh = classify_usage(multi_day_15m_df.copy(), "Europe/Vienna")

    assert "base_load_kwh" in df_classified.columns
    assert "regular_load_kwh" in df_classified.columns
    assert "peak_load_kwh" in df_classified.columns

    total_decomposed = (
        df_classified["base_load_kwh"] +
        df_classified["regular_load_kwh"] +
        df_classified["peak_load_kwh"]
    )
    assert total_decomposed.values == pytest.approx(df_classified["consumption_kwh"].values)
    assert base_thresh > 0
    assert peak_thresh > 0


def test_compute_cost_comparison_data(multi_day_15m_df):
    """Verify cost aggregation across Monthly, Weekly, and Daily resolutions."""
    for resolution in ["Monthly", "Weekly", "Daily"]:
        summary = compute_cost_comparison_data(multi_day_15m_df, resolution)
        assert not summary.empty
        assert "Total Consumption" in summary.columns
        assert "Total Static Cost" in summary.columns
        assert "Total Flexible Cost" in summary.columns
        assert "Difference (€)" in summary.columns

        # Verify conservation of total consumption
        assert summary["Total Consumption"].sum() == pytest.approx(multi_day_15m_df["consumption_kwh"].sum())


def test_compute_cumulative_savings_data(multi_day_15m_df):
    """Verify cumulative savings tracks cumulative difference between static and flexible costs."""
    savings_df = compute_cumulative_savings_data(multi_day_15m_df)
    assert not savings_df.empty
    assert "cumulative_savings" in savings_df.columns
    expected_total_diff = (
        multi_day_15m_df["total_cost_static"].sum() - multi_day_15m_df["total_cost_flexible"].sum()
    )
    final_cumulative = savings_df["cumulative_savings"].iloc[-1]
    assert final_cumulative == pytest.approx(expected_total_diff)


def test_compute_absence_data():
    """Verify detection of absent days where consumption drops significantly below base load."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=5 * 24, freq="h", tz="UTC")
    # Normal days consume 0.4 kWh/h (~9.6 kWh/day); Day 3 consumes 0.02 kWh/h (~0.48 kWh/day)
    consumption = []
    for ts in timestamps:
        if ts.date() == timestamps[48].date():  # Day 3
            consumption.append(0.02)
        else:
            consumption.append(0.40)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kwh": consumption
    })
    # Base threshold = 0.3 kWh/interval (for 24 intervals/day, base is 7.2 kWh/day)
    # Absence threshold = 0.5 -> triggers below 3.6 kWh/day
    absent_days = compute_absence_data(df, base_threshold=0.3, absence_threshold=0.5)
    assert len(absent_days) == 1
    assert absent_days[0] == timestamps[48].date()


def test_compute_consumption_quartiles(multi_day_15m_df):
    """Verify calculation of consumption quartiles across intervals of the day."""
    quartiles = compute_consumption_quartiles(multi_day_15m_df, intervals_per_day=96, resolution="Hourly")
    assert not quartiles.empty
    assert "Consumption Q1" in quartiles.columns
    assert "Consumption Median" in quartiles.columns
    assert "Consumption Q3" in quartiles.columns
    assert (quartiles["Consumption Q1"] <= quartiles["Consumption Median"]).all()
    assert (quartiles["Consumption Median"] <= quartiles["Consumption Q3"]).all()


def test_compute_peak_timing_score(multi_day_15m_df):
    """Verify peak timing score computation executes and returns a numeric score."""
    df_classified, _, _ = classify_usage(multi_day_15m_df.copy(), "Europe/Vienna")
    score = compute_peak_timing_score(df_classified)
    assert isinstance(score, (int, float))
    assert -100.0 <= score <= 100.0
