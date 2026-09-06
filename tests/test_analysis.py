# python -m pytest tests/test_analysis.py


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


def test_compute_cost_comparison_with_variable_tariff(multi_day_15m_df):
    """Verify cost aggregation includes Total Variable Cost when present."""
    df = multi_day_15m_df.copy()
    df["total_cost_variable"] = df["consumption_kwh"] * 0.12 + 0.001
    summary = compute_cost_comparison_data(df, "Monthly")
    assert "Total Variable Cost" in summary.columns
    assert "Avg. Variable Price" in summary.columns
    assert summary["Total Variable Cost"].sum() == pytest.approx(df["total_cost_variable"].sum())


def test_compute_cumulative_savings_three_tariffs(multi_day_15m_df):
    """Verify cumulative savings tracks cheapest vs most expensive among 3 tariffs."""
    df = multi_day_15m_df.copy()
    # static is ~0.15, flexible is varying ~0.05-0.15, make variable cheapest at 0.03
    df["total_cost_variable"] = df["consumption_kwh"] * 0.03
    savings_df = compute_cumulative_savings_data(df)
    assert not savings_df.empty
    expected_savings = df["total_cost_static"].sum() - df["total_cost_variable"].sum()
    assert savings_df["cumulative_savings"].iloc[-1] == pytest.approx(expected_savings)


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


def test_compute_usage_profile_data_spot_and_variable(multi_day_15m_df):
    """Verify compute_usage_profile_data works for both spot and variable prices and feeds marimekko chart."""
    from src.analysis import compute_usage_profile_data
    from src.charts import get_marimekko_chart

    df_classified, _, _ = classify_usage(multi_day_15m_df.copy(), "Europe/Vienna")
    df_classified["variable_price_eur_kwh"] = 0.08

    profile_spot = compute_usage_profile_data(df_classified, price_type="spot")
    assert not profile_spot.empty
    assert "avg_price" in profile_spot.columns
    assert "proportion" in profile_spot.columns

    profile_var = compute_usage_profile_data(df_classified, price_type="variable")
    assert not profile_var.empty
    assert profile_var["avg_price"].tolist() == pytest.approx([0.08] * len(profile_var))

    # Verify marimekko figures render for both modes
    fig_spot = get_marimekko_chart(profile_spot, price_type="spot")
    assert fig_spot is not None
    assert len(fig_spot.data) > 0

    fig_var = get_marimekko_chart(profile_var, price_type="variable")
    assert fig_var is not None
    assert len(fig_var.data) > 0


def test_yearly_summary_and_cumulative_chart_titles(multi_day_15m_df):
    """Verify yearly summary supports 3 tariffs with Difference as max-min, and cumulative savings chart shows title."""
    from src.analysis import compute_yearly_summary, compute_cumulative_savings_data
    from src.charts import get_cumulative_savings_chart

    df = multi_day_15m_df.copy()
    df["total_cost_variable"] = df["consumption_kwh"] * 0.05

    yearly = compute_yearly_summary(df)
    assert not yearly.empty
    assert "Total Flexible Cost" in yearly.columns
    assert "Total Variable Cost" in yearly.columns
    assert "Total Static Cost" in yearly.columns
    assert "Difference (€)" in yearly.columns
    expected_diff = yearly[["Total Flexible Cost", "Total Variable Cost", "Total Static Cost"]].max(axis=1) - yearly[["Total Flexible Cost", "Total Variable Cost", "Total Static Cost"]].min(axis=1)
    assert yearly["Difference (€)"].tolist() == pytest.approx(expected_diff.tolist())

    savings_df = compute_cumulative_savings_data(df)
    chart = get_cumulative_savings_chart(savings_df)
    assert chart.layout.title.text is not None
    assert "vs." in chart.layout.title.text
