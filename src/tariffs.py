import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from src.config import LOCAL_TIMEZONE
from src.logger import logger
from src.utils import get_intervals_per_day


class TariffType(Enum):
    """Enum representing tariff pricing structures."""

    SPOT = "spot"
    FIXED = "fixed"
    TIME_VARIABLE = "time_variable"


@dataclass
class Tariff(ABC):
    """Abstract base class for all electricity tariffs."""

    name: str
    type: TariffType
    monthly_fee: float
    link: str = ""
    usage_tax: bool = False

    def __new__(cls, *args, **kwargs):
        """Factory constructor ensuring Tariff(...) instantiates the correct subclass."""
        if cls is Tariff:
            t_type = kwargs.get("type")
            if t_type is None and len(args) > 1:
                t_type = args[1]
            if t_type == TariffType.SPOT:
                return super().__new__(SpotTariff)
            elif t_type == TariffType.FIXED:
                return super().__new__(FixedTariff)
            elif t_type == TariffType.TIME_VARIABLE:
                return super().__new__(TimeVariableTariff)
        return super().__new__(cls)

    def _get_interval_monthly_fee(self, df: pd.DataFrame) -> pd.Series:
        """Computes pro-rata monthly fee per interval."""
        intervals_per_day = get_intervals_per_day(df)
        days_in_month = df["timestamp"].dt.days_in_month
        return (self.monthly_fee / days_in_month) / intervals_per_day

    @abstractmethod
    def get_price_series(self, df: pd.DataFrame) -> pd.Series:
        """Returns the unit electricity price (€/kWh) for each timestamp."""
        pass

    def calculate_cost(self, df: pd.DataFrame) -> pd.Series:
        """Calculates total cost (€) per timestamp interval."""
        return df["consumption_kwh"] * self.get_price_series(df) + self._get_interval_monthly_fee(df)


@dataclass
class SpotTariff(Tariff):
    """Dynamic hourly/15m spot price tariff tied to EPEX."""

    price_kwh: float = 0.0  # On-top surcharge (€/kWh)
    price_kwh_pct: float = 0.0  # Percentage markup on spot price

    def __init__(
        self,
        name: str,
        type: TariffType = TariffType.SPOT,
        price_kwh: float = 0.0,
        monthly_fee: float = 0.0,
        price_kwh_pct: float = 0.0,
        link: str = "",
        usage_tax: bool = False,
    ):
        super().__init__(name=name, type=TariffType.SPOT, monthly_fee=monthly_fee, link=link, usage_tax=usage_tax)
        self.price_kwh = price_kwh
        self.price_kwh_pct = price_kwh_pct

    def get_price_series(self, df: pd.DataFrame) -> pd.Series:
        price = df["spot_price_eur_kwh"] * (1 + self.price_kwh_pct / 100) + self.price_kwh
        return price * 1.06 if self.usage_tax else price


@dataclass
class FixedTariff(Tariff):
    """Traditional fixed-price contract."""

    price_kwh: float = 0.0

    def __init__(
        self,
        name: str,
        type: TariffType = TariffType.FIXED,
        price_kwh: float = 0.0,
        monthly_fee: float = 0.0,
        link: str = "",
        usage_tax: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, type=TariffType.FIXED, monthly_fee=monthly_fee, link=link, usage_tax=usage_tax)
        self.price_kwh = price_kwh

    def get_price_series(self, df: pd.DataFrame) -> pd.Series:
        rate = self.price_kwh * 1.06 if self.usage_tax else self.price_kwh
        return pd.Series(rate, index=df.index)


@dataclass
class TimeVariableTariff(Tariff):
    """Austrian seasonal time-variable tariff with solar windows (10:00-16:00) and seasonal base prices."""

    summer_sun_price: Optional[float] = None
    winter_sun_price: Optional[float] = None
    summer_regular_price: float = 0.0
    winter_regular_price: float = 0.0

    def __init__(
        self,
        name: str,
        type: TariffType = TariffType.TIME_VARIABLE,
        summer_sun_price: Optional[float] = None,
        winter_sun_price: Optional[float] = None,
        summer_regular_price: float = 0.0,
        winter_regular_price: float = 0.0,
        monthly_fee: float = 0.0,
        link: str = "",
        usage_tax: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, type=TariffType.TIME_VARIABLE, monthly_fee=monthly_fee, link=link, usage_tax=usage_tax)
        self.summer_sun_price = summer_sun_price
        self.winter_sun_price = winter_sun_price
        self.summer_regular_price = summer_regular_price
        self.winter_regular_price = winter_regular_price

    def get_price_series(self, df: pd.DataFrame) -> pd.Series:
        """Computes rate matching Austrian Summer (Apr-Sep) and Winter (Oct-Mar) solar and regular windows."""
        ts = df["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE)
        else:
            ts = ts.dt.tz_convert(LOCAL_TIMEZONE)

        months = ts.dt.month
        hours = ts.dt.hour

        is_summer = months.isin([4, 5, 6, 7, 8, 9])
        is_sun_window = (hours >= 10) & (hours < 16)

        prices = pd.Series(
            np.where(is_summer, self.summer_regular_price, self.winter_regular_price),
            index=df.index,
            dtype=float,
        )

        if self.summer_sun_price is not None:
            prices = prices.mask(is_summer & is_sun_window, self.summer_sun_price)
        if self.winter_sun_price is not None:
            prices = prices.mask(~is_summer & is_sun_window, self.winter_sun_price)

        return prices * 1.06 if self.usage_tax else prices


class TariffManager:
    """Manages loading and cost calculations for spot, variable, and fixed tariffs."""

    def __init__(
        self,
        flex_tariff_path: str,
        static_tariff_path: str,
        variable_tariff_path: str = "resources/tariffs_variable.json",
    ):
        self.flex_tariffs = self._load_spot_tariffs(flex_tariff_path)
        self.static_tariffs = self._load_fixed_tariffs(static_tariff_path)
        self.variable_tariffs = self._load_variable_tariffs(variable_tariff_path)

    def _load_spot_tariffs(self, file_path: str) -> List[SpotTariff]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [
                    SpotTariff(
                        name=item.get("name", "Unnamed"),
                        price_kwh=item.get("price_kwh_gross", 0.0),
                        monthly_fee=item.get("monthly_fee_gross", 0.0),
                        link=item.get("link", ""),
                        price_kwh_pct=item.get("price_kwh_pct", 0.0),
                    )
                    for item in json.load(f)
                ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.log(f"Error loading spot tariffs from {file_path}: {e}", severity=1)
            return []

    def _load_fixed_tariffs(self, file_path: str) -> List[FixedTariff]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [
                    FixedTariff(
                        name=item.get("name", "Unnamed"),
                        price_kwh=item.get("price_kwh_gross", 0.0),
                        monthly_fee=item.get("monthly_fee_gross", 0.0),
                        link=item.get("link", ""),
                    )
                    for item in json.load(f)
                ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.log(f"Error loading fixed tariffs from {file_path}: {e}", severity=1)
            return []

    def _load_variable_tariffs(self, file_path: str) -> List[TimeVariableTariff]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [
                    TimeVariableTariff(
                        name=item.get("name", "Unnamed"),
                        summer_sun_price=item.get("summer_sun_price_gross"),
                        winter_sun_price=item.get("winter_sun_price_gross"),
                        summer_regular_price=item.get("summer_regular_price_gross", item.get("winter_regular_price_gross", 0.0)),
                        winter_regular_price=item.get("winter_regular_price_gross", 0.0),
                        monthly_fee=item.get("monthly_fee_gross", 0.0),
                        link=item.get("link", ""),
                    )
                    for item in json.load(f)
                ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.log(f"Error loading variable tariffs from {file_path}: {e}", severity=1)
            return []

    def get_flex_tariffs_with_custom(self) -> Dict[str, SpotTariff]:
        tariffs = {t.name: t for t in self.flex_tariffs}
        tariffs["Custom"] = SpotTariff(name="Custom", price_kwh=0.0179, monthly_fee=2.40, price_kwh_pct=0.0)
        return tariffs

    def get_static_tariffs_with_custom(self) -> Dict[str, FixedTariff]:
        tariffs = {t.name: t for t in self.static_tariffs}
        tariffs["Custom"] = FixedTariff(name="Custom", price_kwh=0.14, monthly_fee=2.00)
        return tariffs

    def get_variable_tariffs_with_custom(self) -> Dict[str, TimeVariableTariff]:
        tariffs = {t.name: t for t in self.variable_tariffs}
        tariffs["Custom"] = TimeVariableTariff(
            name="Custom",
            summer_sun_price=0.0499,
            winter_sun_price=0.0990,
            summer_regular_price=0.1390,
            winter_regular_price=0.1390,
            monthly_fee=5.00,
        )
        return tariffs

    def run_cost_analysis(
        self,
        df: pd.DataFrame,
        spot_tariff: Tariff,
        variable_tariff: Optional[Tariff] = None,
        fixed_tariff: Optional[Tariff] = None,
    ) -> pd.DataFrame:
        """Calculates electricity costs across configured tariffs."""
        df["total_cost_spot"] = spot_tariff.calculate_cost(df)
        df["total_cost_flexible"] = df["total_cost_spot"]

        if variable_tariff is not None:
            df["total_cost_variable"] = variable_tariff.calculate_cost(df)
            df["variable_price_eur_kwh"] = variable_tariff.get_price_series(df)

        if fixed_tariff is not None:
            df["total_cost_fixed"] = fixed_tariff.calculate_cost(df)
            df["total_cost_static"] = df["total_cost_fixed"]

        return df