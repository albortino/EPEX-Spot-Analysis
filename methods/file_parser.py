import pandas as pd
import io
from methods.utils import get_intervals_per_day

class ConsumptionDataParser:
    """
    Parses various consumption CSV formats into a standardized DataFrame.
    Inspired by the logic from: https://github.com/awattar-backtesting/awattar-backtesting.github.io
    """

    def __init__(self, local_timezone="Europe/Vienna"):
        self.local_timezone = local_timezone
        self._parsers = [
            self._try_netz_noe,
            self._try_linz_ag,
            self._try_e_netze_steiermark,
            self._try_default_format
        ]

    def parse_file(self, uploaded_file, aggregation_level: str = "h") -> pd.DataFrame:
        """
        Tries to parse the uploaded file with a series of parser methods.
        Returns a standardized DataFrame on success, or an empty one on failure. Normalized to UTC date.
        """
        if uploaded_file is None:
            return pd.DataFrame()

        # Read file into memory to allow multiple parsing attempts.
        file_content = uploaded_file.getvalue().decode("utf-8")

        for parser_func in self._parsers:
            try:
                df = parser_func(file_content)
                if not df.empty and all(col in df.columns for col in ["timestamp", "consumption_kwh"]):
                    
                    aggregation_level = "15min" if get_intervals_per_day(df) > 24 else "h"
                    # Standardize and resample.
                    df = df.set_index("timestamp")["consumption_kwh"].resample(aggregation_level).sum().dropna().reset_index()
                        
                    return df
                
            except Exception as e:
                print(f"Parser {parser_func.__name__} failed with error: {e}")
                continue # Try next parser

        # If all parsers fail, return empty
        return pd.DataFrame()

    def _try_default_format(self, file_content: str) -> pd.DataFrame:
        """Parses the original format: "Datum;Zeit von;Verbrauch [kWh]"."""
        df = pd.read_csv(io.StringIO(file_content), sep=";", decimal=",", encoding="utf-8", dayfirst=True)
        
        consumption_col = next((col for col in df.columns if "verbrauch" in col.lower() or "kWh" in col.lower()), None)
        if not consumption_col or "Datum" not in df.columns or "Zeit von" not in df.columns:
            raise ValueError("Default format columns not found.")
            
        df = df[["Datum", "Zeit von", consumption_col]].copy().dropna()
        df.columns = ["date_str", "time_str", "consumption_kwh"]
        
        # Manually combine date and time for robust parsing
        df["timestamp_local"] = pd.to_datetime(df["date_str"] + " " + df["time_str"], dayfirst=True)
        df["timestamp"] = df["timestamp_local"].dt.tz_localize(self.local_timezone, ambiguous="infer").dt.tz_convert("UTC")
        return df
    
    def _try_linz_ag(self, file_content: str) -> pd.DataFrame:
        """Parses Netz NOE format: "Datum;Enegiemenge in kWh;Ersatzwert"."""
        df = pd.read_csv(io.StringIO(file_content), sep=";", decimal=",", encoding="utf-8")
        if not all(col in df.columns for col in ["Datum", "Energiemenge in kWh"]):
            raise ValueError("Linz AG columns not found.")
        
        df["timestamp"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y")
        df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Vienna", ambiguous="infer").dt.tz_convert("UTC")
        df = df.rename(columns={"Energiemenge in kWh": "consumption_kwh"})
        df = df[["timestamp", "consumption_kwh"]]
        return df

    def _try_netz_noe(self, file_content: str) -> pd.DataFrame:
        """Parses Netz NOE format: "Von;Bis;Verbrauch(kWh)"."""
        df = pd.read_csv(io.StringIO(file_content), sep=";", decimal=",", encoding="utf-8", skiprows=1)
        if not all(col in df.columns for col in ["Von", "Verbrauch(kWh)"]):
            raise ValueError("Netz NOE columns not found.")
            
        df["timestamp_local"] = pd.to_datetime(df["Von"], format="%d.%m.%Y %H:%M:%S")
        df["timestamp"] = df["timestamp_local"].dt.tz_localize(self.local_timezone, ambiguous="infer").dt.tz_convert("UTC")
        df = df.rename(columns={"Verbrauch(kWh)": "consumption_kwh"})
        return df

    def _try_e_netze_steiermark(self, file_content: str) -> pd.DataFrame:
        """Parses E-Netze Steiermark format: "Ablesezeitpunkt;Wirkverbrauch [kWh]"."""
        df = pd.read_csv(io.StringIO(file_content), sep=";", decimal=",", encoding="utf-8")
        if not all(col in df.columns for col in ["Ablesezeitpunkt", "Wirkverbrauch [kWh]"]):
            raise ValueError("E-Netze Steiermark columns not found.")

        # This format often gives the end of the 15-min interval, so we parse it as such
        df["timestamp_local"] = pd.to_datetime(df["Ablesezeitpunkt"], format="%d.%m.%Y %H:%M")
        df["timestamp"] = df["timestamp_local"].dt.tz_localize(self.local_timezone, ambiguous="infer").dt.tz_convert("UTC")
        df = df.rename(columns={"Wirkverbrauch [kWh]": "consumption_kwh"})
        return df

if __name__ == "__main__":
    parser = ConsumptionDataParser()
    print(parser)