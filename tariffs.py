import pandas as pd
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

class TariffType(Enum):
    """Enum to represent the type of tariff."""
    
    FLEXIBLE = "flexible"
    STATIC = "static"

# Dataclass adds the __init__ and __repr__ method.
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
            print(f"Error: Tariff file not found at {file_path}")
            return []
        
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
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

    def run_cost_analysis(self, df: pd.DataFrame, flex_tariff: Tariff, static_tariff: Tariff) -> pd.DataFrame:
        """Calculates flexible and static costs for the given dataframe, supporting any time resolution."""
        
        df_costs = df.copy()
                
        # Determine the number of entries per day (time resolution)
        df_costs["date"] = df_costs["timestamp"].dt.date
        intervals_per_day = df_costs.groupby("date").size().mode().iloc[0]
        
        df_costs["days_in_month"] = df_costs["timestamp"].dt.days_in_month
        
        # Calculate total cost per time resolution (hour or 15 minutes) for both tariffs
        flex_spot_price_component = df_costs["spot_price_eur_kwh"] * (1 + flex_tariff.price_kwh_pct / 100)
        flex_total_kwh_price = flex_spot_price_component + flex_tariff.price_kwh
        flex_fee = (flex_tariff.monthly_fee / df_costs["days_in_month"]) / intervals_per_day
        static_fee = (static_tariff.monthly_fee / df_costs["days_in_month"]) / intervals_per_day

        df_costs["total_cost_flexible"] = (df_costs["consumption_kwh"] * flex_total_kwh_price) + flex_fee
        df_costs["total_cost_static"] = (df_costs["consumption_kwh"] * static_tariff.price_kwh) + static_fee
        return df_costs.drop(columns="date")