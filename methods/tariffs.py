import pandas as pd
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
from methods.logger import logger

class TariffType(Enum):
    """Enum to represent the type of tariff."""
    
    FLEXIBLE = "flexible"
    STATIC = "static"

# Dataclass adds the __init__ and __repr__ method automatically.
@dataclass
class Tariff:
    """A dataclass to hold all information about a single tariff."""
    
    name: str
    type: TariffType
    price_kwh: float  # For FLEXIBLE, it's the on-top price. For STATIC, it's the fixed price.
    monthly_fee: float
    price_kwh_pct: float = 0.0 # Percentage on-top for flexible tariffs
    link: str = ""

class TariffManager:
    """Handles loading tariffs and calculating costs."""

    def __init__(self, flex_tariff_path: str, static_tariff_path: str):
        self.flex_tariffs = self._load_tariffs(flex_tariff_path, TariffType.FLEXIBLE)
        self.static_tariffs = self._load_tariffs(static_tariff_path, TariffType.STATIC)

    def _load_tariffs(self, file_path: str, tariff_type: TariffType) -> List[Tariff]:
        """Loads tariff data from a JSON file and returns a list of Tariff objects."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tariff_data = json.load(f)
                return [
                    Tariff(
                        name=item.get("name", "Unnamed"),
                        type=tariff_type,
                        price_kwh=item.get("price_kwh", 0.0),
                        monthly_fee=item.get("monthly_fee", 0.0),
                        link=item.get("link", ""),
                        price_kwh_pct=item.get("price_kwh_pct", 0.0)
                    ) for item in tariff_data
                ]
                
        except FileNotFoundError:
            logger.log(f"Tariff file not found at {file_path}", severity=1)
            return []
        
        except json.JSONDecodeError:
            logger.log(f"Could not decode JSON from {file_path}", severity=1)
            return []

    def get_flex_tariffs_with_custom(self) -> Dict[str, Tariff]:
        """Returns a dictionary of flexible tariffs including a default custom option."""
        
        tariffs = {tariff.name: tariff for tariff in self.flex_tariffs}
        tariffs["Custom"] = Tariff(name="Custom", type=TariffType.FLEXIBLE, price_kwh=0.0215, monthly_fee=2.40, price_kwh_pct=0.0)
        return tariffs

    def get_static_tariffs_with_custom(self) -> Dict[str, Tariff]:
        """Returns a dictionary of static tariffs including a default custom option."""
        
        tariffs = {tariff.name: tariff for tariff in self.static_tariffs}
        tariffs["Custom"] = Tariff(name="Custom", type=TariffType.STATIC, price_kwh=0.14, monthly_fee=2.00)
        return tariffs
    
    def _prepare_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Prepares the dataframe for cost calculation by adding a date column and returning the intervals per day."""
    
        if not "consumption_kwh" in df.columns:
            raise KeyError("Consumption data is missing.")
        
        if not "date" in df.columns or not "days_in_month" in df.columns:
            df["date"] = df["timestamp"].dt.date
            df["days_in_month"] = df["timestamp"].dt.days_in_month
        
        # Determine the number of entries per day (time resolution)
        intervals_per_day = df.groupby("date").size().mode().iloc[0]
                
        return df, intervals_per_day
        
    def _calculate_static_cost(self, df: pd.DataFrame, tariff: Tariff) -> pd.Series:
        """Calculates the static costs based on a tariff and a dataframe with consumption data."""
        
        df, intervals_per_day = self._prepare_df(df)
        
        # Calculate the proportion of the whole monthly fee for every row (=time resultion)
        monthly_fee = (tariff.monthly_fee / df["days_in_month"]) / intervals_per_day

        return df["consumption_kwh"] * tariff.price_kwh + monthly_fee

    def _calculate_flexible_cost(self, df: pd.DataFrame, tariff: Tariff) -> pd.Series:
        """Calculate the flexible costs based on a tariff and a dataframe with consumption as well as spot price data."""
        
        df, intervals_per_day = self._prepare_df(df)
        
        # Calculate total cost per time resolution (hour or 15 minutes) for both tariffs.
        flex_spot_price_component = df["spot_price_eur_kwh"] * (1 + tariff.price_kwh_pct / 100) + tariff.price_kwh
        
        # Calculate the proportion of the whole monthly fee for every row (=time resultion)
        monthly_fee = (tariff.monthly_fee / df["days_in_month"]) / intervals_per_day
        
        return df["consumption_kwh"] * flex_spot_price_component + monthly_fee
        
    def run_cost_analysis(self, df: pd.DataFrame, flex_tariff: Tariff, static_tariff: Tariff) -> pd.DataFrame:
        """Calculates flexible and static costs for the given dataframe, supporting any time resolution."""
        
        df["total_cost_flexible"] = self._calculate_flexible_cost(df, flex_tariff)
        df["total_cost_static"] = self._calculate_static_cost(df, static_tariff)
                
        return df