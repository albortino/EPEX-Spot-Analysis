from datetime import date

import pandas as pd

from src.analysis import simulate_peak_shifting
from src.tariffs import Tariff, TariffManager, TariffType
from src.utils import inspect_consumption, inspect_price_coverage


def frame(timestamps, consumption=None):
    consumption = consumption if consumption is not None else [1] * len(timestamps)
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True), "consumption_kwh": consumption})


def test_validation_rejects_duplicate_timestamps():
    data = frame(["2024-01-01T00:00Z", "2024-01-01T00:00Z", "2024-01-01T01:00Z"])
    assert not inspect_consumption(data).usable


def test_validation_reports_price_coverage():
    data = frame(["2024-01-01T00:00Z", "2024-01-01T01:00Z"])
    data["spot_price_eur_kwh"] = [0.2, None]
    assert inspect_price_coverage(data) == 0.5


def test_monthly_fee_is_prorated_per_real_interval():
    data = frame(["2024-01-01T00:00Z", "2024-01-01T01:00Z"])
    tariff = Tariff("fixed", TariffType.FIXED, 0, 31)
    costs = tariff.calculate_cost(data)
    assert costs.sum() == 2 / 24


def test_peak_shift_preserves_energy_and_requires_lower_price():
    data = frame(["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T02:00Z"], (1, 3, 1))
    data["base_load_kwh"] = 1
    data["regular_load_kwh"] = 0
    data["peak_load_kwh"] = [0, 2, 0]
    data["spot_price_eur_kwh"] = [0.1, 0.3, 0.2]
    result = simulate_peak_shifting(data, 100)
    assert result["consumption_kwh"].sum() == data["consumption_kwh"].sum()
    assert result.loc[1, "peak_load_kwh"] < data.loc[1, "peak_load_kwh"]


def test_peak_shift_does_not_move_to_an_expensive_interval():
    data = frame(["2024-01-01T00:00Z", "2024-01-01T01:00Z"], (1, 3))
    data["base_load_kwh"] = 1
    data["regular_load_kwh"] = 0
    data["peak_load_kwh"] = [0, 2]
    data["spot_price_eur_kwh"] = [0.4, 0.3]
    result = simulate_peak_shifting(data, 100)
    assert result["peak_load_kwh"].equals(data["peak_load_kwh"])
