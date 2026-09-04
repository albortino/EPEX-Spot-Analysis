import os
import pandas as pd
import pytest
from src.tariffs import Tariff, TariffManager, TariffType


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
    """Verify loading from production JSON configuration files."""
    flex_path = "resources/tariffs_flexible.json"
    static_path = "resources/tariffs_static.json"
    assert os.path.exists(flex_path), f"{flex_path} not found"
    assert os.path.exists(static_path), f"{static_path} not found"

    manager = TariffManager(flex_path, static_path)
    assert len(manager.flex_tariffs) > 0
    assert len(manager.static_tariffs) > 0

    for tariff in manager.flex_tariffs:
        assert tariff.name
        assert tariff.type == TariffType.FLEXIBLE
        assert tariff.price_kwh >= 0
        assert tariff.monthly_fee >= 0

    for tariff in manager.static_tariffs:
        assert tariff.name
        assert tariff.type == TariffType.STATIC
        assert tariff.price_kwh >= 0
        assert tariff.monthly_fee >= 0


def test_get_tariffs_with_custom():
    """Verify custom tariff option is injected into flexible and static dictionaries."""
    manager = TariffManager("resources/tariffs_flexible.json", "resources/tariffs_static.json")
    flex_dict = manager.get_flex_tariffs_with_custom()
    static_dict = manager.get_static_tariffs_with_custom()

    assert "Custom" in flex_dict
    assert flex_dict["Custom"].type == TariffType.FLEXIBLE
    assert "Custom" in static_dict
    assert static_dict["Custom"].type == TariffType.STATIC


def test_cost_calculation_mathematical_precision(sample_hourly_df):
    """Verify exact formula calculations for both flexible and static tariffs."""
    manager = TariffManager.__new__(TariffManager)

    # 31 days in Jan, 24 intervals per day
    # Flexible: spot=0.10, pct=10%, price_kwh=0.02, monthly=31.0
    # Spot component = 0.10 * 1.10 + 0.02 = 0.13 EUR/kWh
    # Monthly fee per interval = 31.0 / 31 / 24 = 1/24 EUR
    # Interval cost = 2.0 * 0.13 + (1/24) = 0.26 + 1/24
    # Total for 24h = 24 * 0.26 + 1.0 = 7.24 EUR
    flex_tariff = Tariff(
        name="TestFlex",
        type=TariffType.FLEXIBLE,
        price_kwh=0.02,
        monthly_fee=31.0,
        price_kwh_pct=10.0
    )

    # Static: price_kwh=0.20, monthly=15.5
    # Monthly fee per interval = 15.5 / 31 / 24 = 0.5 / 24 EUR
    # Interval cost = 2.0 * 0.20 + (0.5/24) = 0.40 + 0.5/24
    # Total for 24h = 24 * 0.40 + 0.5 = 10.10 EUR
    static_tariff = Tariff(
        name="TestStatic",
        type=TariffType.STATIC,
        price_kwh=0.20,
        monthly_fee=15.5
    )

    result_df = manager.run_cost_analysis(sample_hourly_df.copy(), flex_tariff, static_tariff)

    assert "total_cost_flexible" in result_df.columns
    assert "total_cost_static" in result_df.columns
    assert result_df["total_cost_flexible"].sum() == pytest.approx(7.24)
    assert result_df["total_cost_static"].sum() == pytest.approx(10.10)


def test_usage_tax_applied_when_flagged(sample_hourly_df):
    """Verify that 6% usage tax is correctly multiplied when enabled."""
    manager = TariffManager.__new__(TariffManager)

    tariff_no_tax = Tariff("NoTax", TariffType.STATIC, price_kwh=0.20, monthly_fee=0.0, usage_tax=False)
    tariff_with_tax = Tariff("WithTax", TariffType.STATIC, price_kwh=0.20, monthly_fee=0.0, usage_tax=True)

    cost_no_tax = manager._calculate_static_cost(sample_hourly_df, tariff_no_tax).sum()
    cost_with_tax = manager._calculate_static_cost(sample_hourly_df, tariff_with_tax).sum()

    assert cost_with_tax == pytest.approx(cost_no_tax * 1.06)


def test_missing_files_handled_gracefully():
    """Missing tariff files return empty lists without raising unhandled exceptions."""
    manager = TariffManager("non_existent_flex.json", "non_existent_static.json")
    assert manager.flex_tariffs == []
    assert manager.static_tariffs == []
