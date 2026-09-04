import os
import pandas as pd
import pytest
from src.tariffs import Tariff, TariffManager, TariffType, SpotTariff, FixedTariff, TimeVariableTariff


@pytest.fixture
def sample_hourly_df():
    """Generates 24 hours of data for 2024-01-15 (January, 31 days)."""
    timestamps = pd.date_range("2024-01-15 00:00:00", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kwh": [2.0] * 24,
        "spot_price_eur_kwh": [0.10] * 24,
        "date": timestamps.date
    })


def test_tariff_manager_loads_real_files():
    """Verify loading from production JSON configuration files including variable tariffs."""
    flex_path = "resources/tariffs_flexible.json"
    static_path = "resources/tariffs_static.json"
    var_path = "resources/tariffs_variable.json"
    assert os.path.exists(flex_path), f"{flex_path} not found"
    assert os.path.exists(static_path), f"{static_path} not found"
    assert os.path.exists(var_path), f"{var_path} not found"

    manager = TariffManager(flex_path, static_path, var_path)
    assert len(manager.flex_tariffs) > 0
    assert len(manager.static_tariffs) > 0
    assert len(manager.variable_tariffs) > 0

    for tariff in manager.flex_tariffs:
        assert tariff.name
        assert tariff.type == TariffType.SPOT
        assert tariff.price_kwh >= 0
        assert tariff.monthly_fee >= 0

    for tariff in manager.static_tariffs:
        assert tariff.name
        assert tariff.type == TariffType.FIXED
        assert tariff.price_kwh >= 0
        assert tariff.monthly_fee >= 0

    for tariff in manager.variable_tariffs:
        assert tariff.name
        assert tariff.type == TariffType.TIME_VARIABLE
        assert tariff.summer_regular_price >= 0
        assert tariff.winter_regular_price >= 0
        assert tariff.monthly_fee >= 0


def test_get_tariffs_with_custom():
    """Verify custom tariff option is injected into all 3 tariff dictionaries."""
    manager = TariffManager("resources/tariffs_flexible.json", "resources/tariffs_static.json", "resources/tariffs_variable.json")
    flex_dict = manager.get_flex_tariffs_with_custom()
    var_dict = manager.get_variable_tariffs_with_custom()
    static_dict = manager.get_static_tariffs_with_custom()

    assert "Custom" in flex_dict
    assert flex_dict["Custom"].type == TariffType.SPOT
    assert "Custom" in var_dict
    assert var_dict["Custom"].type == TariffType.TIME_VARIABLE
    assert "Custom" in static_dict
    assert static_dict["Custom"].type == TariffType.FIXED


def test_cost_calculation_mathematical_precision(sample_hourly_df):
    """Verify exact formula calculations for flexible, static, and variable tariffs."""
    manager = TariffManager.__new__(TariffManager)

    # 31 days in Jan, 24 intervals per day
    # Spot: spot=0.10, pct=10%, price_kwh=0.02, monthly=31.0
    # Spot component = 0.10 * 1.10 + 0.02 = 0.13 EUR/kWh
    # Monthly fee per interval = 31.0 / 31 / 24 = 1/24 EUR
    # Interval cost = 2.0 * 0.13 + (1/24) = 0.26 + 1/24
    # Total for 24h = 24 * 0.26 + 1.0 = 7.24 EUR
    spot_tariff = Tariff(
        name="TestFlex",
        type=TariffType.SPOT,
        price_kwh=0.02,
        monthly_fee=31.0,
        price_kwh_pct=10.0
    )

    # Fixed: price_kwh=0.20, monthly=15.5
    # Monthly fee per interval = 15.5 / 31 / 24 = 0.5 / 24 EUR
    # Interval cost = 2.0 * 0.20 + (0.5/24) = 0.40 + 0.5/24
    # Total for 24h = 24 * 0.40 + 0.5 = 10.10 EUR
    fixed_tariff = Tariff(
        name="TestStatic",
        type=TariffType.FIXED,
        price_kwh=0.20,
        monthly_fee=15.5
    )

    result_df = manager.run_cost_analysis(sample_hourly_df.copy(), spot_tariff, fixed_tariff=fixed_tariff)

    assert "total_cost_spot" in result_df.columns
    assert "total_cost_fixed" in result_df.columns
    assert result_df["total_cost_spot"].sum() == pytest.approx(7.24)
    assert result_df["total_cost_fixed"].sum() == pytest.approx(10.10)


def test_time_variable_tariff_windows():
    """Verify Austrian summer, winter, and regular windows for TimeVariableTariff."""
    # 24 hours in Vienna local time for January 15 (Winter)
    jan_ts = pd.date_range("2024-01-15 00:00:00", periods=24, freq="h", tz="Europe/Vienna")
    df_jan = pd.DataFrame({"timestamp": jan_ts, "consumption_kwh": [1.0] * 24})

    tariff = TimeVariableTariff(
        name="TestVar",
        summer_sun_price=0.05,
        winter_sun_price=0.10,
        summer_regular_price=0.14,
        winter_regular_price=0.16,
        monthly_fee=0.0
    )

    jan_prices = tariff.get_price_series(df_jan)
    # Winter hours 10 to 15 (6 hours) should be winter_sun_price (0.10)
    # Remaining 18 hours should be winter_regular_price (0.16)
    for h in range(24):
        if 10 <= h < 16:
            assert jan_prices.iloc[h] == pytest.approx(0.10)
        else:
            assert jan_prices.iloc[h] == pytest.approx(0.16)

    # 24 hours in Vienna local time for July 15 (Summer)
    jul_ts = pd.date_range("2024-07-15 00:00:00", periods=24, freq="h", tz="Europe/Vienna")
    df_jul = pd.DataFrame({"timestamp": jul_ts, "consumption_kwh": [1.0] * 24})

    jul_prices = tariff.get_price_series(df_jul)
    # Summer hours 10 to 15 should be summer_sun_price (0.05)
    # Remaining 18 hours should be summer_regular_price (0.14)
    for h in range(24):
        if 10 <= h < 16:
            assert jul_prices.iloc[h] == pytest.approx(0.05)
        else:
            assert jul_prices.iloc[h] == pytest.approx(0.14)


def test_oekotime_tariff_model():
    """Verify oeko Time model: Apr-Sep 10-16h 0.0576, Apr-Sep 16-10h 0.1488, Oct-Mar 0-24h 0.1884."""
    oeko_tariff = TimeVariableTariff(
        name="oeko Time",
        summer_sun_price=0.0576,
        winter_sun_price=None,
        summer_regular_price=0.1488,
        winter_regular_price=0.1884,
        monthly_fee=6.00
    )

    # July (Summer)
    jul_ts = pd.date_range("2024-07-15 00:00:00", periods=24, freq="h", tz="Europe/Vienna")
    df_jul = pd.DataFrame({"timestamp": jul_ts, "consumption_kwh": [1.0] * 24})
    jul_prices = oeko_tariff.get_price_series(df_jul)
    for h in range(24):
        if 10 <= h < 16:
            assert jul_prices.iloc[h] == pytest.approx(0.0576)
        else:
            assert jul_prices.iloc[h] == pytest.approx(0.1488)

    # January (Winter)
    jan_ts = pd.date_range("2024-01-15 00:00:00", periods=24, freq="h", tz="Europe/Vienna")
    df_jan = pd.DataFrame({"timestamp": jan_ts, "consumption_kwh": [1.0] * 24})
    jan_prices = oeko_tariff.get_price_series(df_jan)
    for h in range(24):
        assert jan_prices.iloc[h] == pytest.approx(0.1884)


def test_time_variable_tariff_fallback_none():
    """When winter_sun_price is None (e.g. EVN tariff), it falls back to winter_regular_price in winter."""
    jan_ts = pd.date_range("2024-01-15 00:00:00", periods=24, freq="h", tz="Europe/Vienna")
    df_jan = pd.DataFrame({"timestamp": jan_ts, "consumption_kwh": [1.0] * 24})

    evn_tariff = TimeVariableTariff(
        name="EVN Sonne",
        summer_sun_price=0.08,
        winter_sun_price=None,
        summer_regular_price=0.12,
        winter_regular_price=0.12,
        monthly_fee=0.0
    )
    jan_prices = evn_tariff.get_price_series(df_jan)
    assert (jan_prices == 0.12).all()


def test_three_way_cost_analysis(sample_hourly_df):
    """Verify run_cost_analysis with 3 tariffs populates spot, variable, and fixed costs."""
    manager = TariffManager.__new__(TariffManager)
    spot = SpotTariff(name="Spot", price_kwh=0.01, monthly_fee=3.10)
    var = TimeVariableTariff(
        name="Var",
        summer_sun_price=0.05,
        winter_sun_price=0.10,
        summer_regular_price=0.14,
        winter_regular_price=0.16,
        monthly_fee=3.10
    )
    fixed = FixedTariff(name="Fixed", price_kwh=0.20, monthly_fee=3.10)

    result_df = manager.run_cost_analysis(sample_hourly_df.copy(), spot, var, fixed)
    assert "total_cost_spot" in result_df.columns
    assert "total_cost_variable" in result_df.columns
    assert "total_cost_fixed" in result_df.columns
    assert (result_df["total_cost_variable"] > 0).all()


def test_usage_tax_applied_when_flagged(sample_hourly_df):
    """Verify that 6% usage tax is correctly multiplied when enabled."""
    tariff_no_tax = Tariff("NoTax", TariffType.FIXED, price_kwh=0.20, monthly_fee=0.0, usage_tax=False)
    tariff_with_tax = Tariff("WithTax", TariffType.FIXED, price_kwh=0.20, monthly_fee=0.0, usage_tax=True)

    cost_no_tax = tariff_no_tax.calculate_cost(sample_hourly_df).sum()
    cost_with_tax = tariff_with_tax.calculate_cost(sample_hourly_df).sum()

    assert cost_with_tax == pytest.approx(cost_no_tax * 1.06)


def test_missing_files_handled_gracefully():
    """Missing tariff files return empty lists without raising unhandled exceptions."""
    manager = TariffManager("non_existent_flex.json", "non_existent_static.json", "non_existent_var.json")
    assert manager.flex_tariffs == []
    assert manager.static_tariffs == []
    assert manager.variable_tariffs == []
