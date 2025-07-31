import pandas as pd
import io
import re
import requests
import os
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
from methods.config import LOCAL_TIMEZONE, CACHE_FOLDER
from methods.utils import get_intervals_per_day

@dataclass
class ProviderFormat:
    """Holds the parsing configuration for a specific electricity provider's CSV format. """
    name: str
    usage_col: str
    timestamp_col: str
    time_sub_col: Optional[str] = None
    date_format: Optional[str] = "%d.%m.%Y %H:%M"
    other_cols: Optional[List[str]] = None
    should_skip_func: Optional[str] = None
    fixup_timestamp: bool = False
    separator: str = ";"
    decimal: str = ","
    skiprows: int = 0
    encoding: str = 'utf-8-sig'
    feedin: bool = False
    end_timestamp_col: Optional[str] = None
    preprocess_date_func: Optional[str] = None

class JavaScriptNetzbetreiberParser:
    """Parses the JavaScript netzbetreiber.js from aWATTar backtesting JavaScript file to extract provider configurations."""
    
    def __init__(self):
        self.date_format_map = {
            "dd.MM.yyyy HH:mm": "%d.%m.%Y %H:%M",
            "dd.MM.yyyy HH:mm:ss": "%d.%m.%Y %H:%M:%S",
            "dd.MM.yy HH:mm": "%d.%m.%y %H:%M",
            "dd.MM.yy HH:mm:ss": "%d.%m.%y %H:%M:%S",
            "yyyy-MM-dd HH:mm:ss": "%Y-%m-%d %H:%M:%S",
            " dd.MM.yyyy HH:mm:ss": " %d.%m.%Y %H:%M:%S",
            "parseISO": "ISO8601"
        }
    
    def parse_js_file(self, js_content: str) -> List[ProviderFormat]:
        """
        Parse JavaScript content and extract Netzbetreiber configurations.
        """
        providers = []
        
        # Pattern to match export const declarations
        pattern = r'export const (\w+) = new Netzbetreiber\((.*?)\);'
        matches = re.findall(pattern, js_content, re.DOTALL)
        
        for var_name, params_str in matches:
            try:
                provider = self._parse_netzbetreiber_params(var_name, params_str)
                if provider:
                    providers.append(provider)
            except Exception as e:
                print(f"Error parsing {var_name}: {e}")
                continue
        
        return providers
    
    def _parse_netzbetreiber_params(self, var_name: str, params_str: str) -> Optional[ProviderFormat]:
        """
        Parse individual Netzbetreiber constructor parameters.
        """
        # Clean up the parameters string
        params_str = params_str.strip()
        
        # Split parameters - this is complex due to nested functions and arrays
        params = self._split_parameters(params_str)
        
        if len(params) < 6:
            return None
        
        # Extract basic parameters
        name = self._clean_string(params[0])
        usage_col = self._clean_string(params[1])
        timestamp_col = self._clean_string(params[2])
        time_sub_col = self._clean_string(params[3]) if params[3] != 'null' else None
        date_format_js = self._clean_string(params[4])
        
        # Convert date format
        date_format = self.date_format_map.get(date_format_js, date_format_js)
        
        # Extract other fields (array)
        other_cols = []
        if len(params) > 6 and params[6] != 'null':
            other_cols = self._parse_array(params[6])
        
        # Extract should_skip function
        should_skip_func = None
        if len(params) > 7 and params[7] != 'null':
            should_skip_func = params[7]
        
        # Extract fixup_timestamp
        fixup_timestamp = False
        if len(params) > 8:
            fixup_timestamp = params[8].strip().lower() == 'true'
        
        # Extract feedin flag
        feedin = False
        if len(params) > 9:
            feedin = params[9].strip().lower() == 'true'
        
        # Extract end timestamp descriptor
        end_timestamp_col = None
        if len(params) > 10 and params[10] != 'null':
            end_timestamp_col = self._clean_string(params[10])
        
        # Extract preprocess date function
        preprocess_date_func = None
        if len(params) > 11 and params[11] != 'null':
            preprocess_date_func = params[11]
        
        return ProviderFormat(
            name=name,
            usage_col=usage_col,
            timestamp_col=timestamp_col,
            time_sub_col=time_sub_col,
            date_format=date_format,
            other_cols=other_cols,
            should_skip_func=should_skip_func,
            fixup_timestamp=fixup_timestamp,
            feedin=feedin,
            end_timestamp_col=end_timestamp_col,
            preprocess_date_func=preprocess_date_func
        )
    
    def _split_parameters(self, params_str: str) -> List[str]:
        """
        Split parameters while respecting nested structures.
        """
        params = []
        current_param = ""
        paren_depth = 0
        bracket_depth = 0
        in_string = False
        string_char = None

        i = 0
        while i < len(params_str):
            char = params_str[i]
            
            if char == '\\': # Escape character
                    i += 1
                    continue
                
            if not in_string:
                if char in ['"', "'"]: # Within apostrophes
                    in_string = True
                    string_char = char
                    i += 1
                    continue
                elif char == '(': # Within parentheses
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[': # Within brackets
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                    params.append(current_param.strip())
                    current_param = ""
                    i += 1
                    continue
            else:
                if char == string_char: # and (i == 0 or params_str[i-1] != '\\')
                    in_string = False
                    string_char = None
                    i += 1
                    continue
            
            current_param += char
            i += 1

        if current_param.strip():
            params.append(current_param.strip())
    
        return params
    
    def _clean_string(self, s: str) -> str:
        """
        Clean a string parameter by removing quotes.
        """
        s = s.strip()
        #if s.startswith('"') and s.endswith('"'):
        #    return s[1:-1]
        #if s.startswith("'") and s.endswith("'"):
        #    return s[1:-1]
        return s
    
    def _parse_array(self, array_str: str) -> List[str]:
        """
        Parse a JavaScript array string.
        """
        array_str = array_str.strip()
        if not (array_str.startswith("[") and array_str.endswith("]")):
            return []
        
        content = array_str[1:-1].strip()
        if not content:
            return []
        
        items = []
        current_item = ""
        in_string = False
        string_char = None
        
        for char in content:
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == ',':
                    if current_item.strip():
                        items.append(self._clean_string(current_item.strip()))
                    current_item = ""
                    continue
            else:
                if char == string_char:
                    in_string = False
                    string_char = None
            
            current_item += char
        
        if current_item.strip():
            items.append(self._clean_string(current_item.strip()))
        
        return items

class ConsumptionDataParser:
    """Parser that can load configurations from JavaScript and parse various formats of electricity consumption data. """

    def __init__(self, local_timezone=LOCAL_TIMEZONE, js_url="https://raw.githubusercontent.com/awattar-backtesting/awattar-backtesting.github.io/main/docs/netzbetreiber.js", js_content=None):
        self.local_timezone = local_timezone
        self.js_parser = JavaScriptNetzbetreiberParser()
        self.cache_file = os.path.join(CACHE_FOLDER, "provider_formats.json")
        self.user_formats_file = os.path.join(CACHE_FOLDER, "additional_provider_formats.json")

        # Load user-defined formats. They are always loaded and take precedence.
        user_formats = self._load_user_defined_formats()

        # Load formats from JS, then cache, then defaults.
        main_formats = []

        if js_content:
            # Direct content parsing, no caching involved.
            try:
                main_formats = self._load_from_js_content(js_content)
            except Exception as e:
                print(f"Failed to parse from content': {e}")
        elif js_url:
            try:
                response = requests.get(js_url)
                response.raise_for_status()
                main_formats = self._load_from_js_content(response.text)
                self._save_to_cache(main_formats)
            except Exception as e:
                print(f"Failed to fetch or parse from URL '{js_url}': {e}")

        # If main_formats is still empty (JS fetch failed or was not attempted)
        if not main_formats:
            print("Attempting to load main provider configurations from cache...")
            main_formats = self._load_from_cache()

        # Combine lists. User formats are first in the list.
        self.provider_formats = user_formats + main_formats
        print(f"Total of {len(self.provider_formats)} provider formats loaded ({len(user_formats)} user-defined, {len(main_formats)} main).")

    def _load_user_defined_formats(self) -> List[ProviderFormat]:
        """Loads user-defined provider formats from own_provider_formats.json."""
        try:
            with open(self.user_formats_file, 'r', encoding='utf-8') as f:
                formats_from_json = json.load(f)
            
            user_formats = [ProviderFormat(**item) for item in formats_from_json]
            print(f"Loaded {len(user_formats)} user-defined provider configurations from {self.user_formats_file}.")
            return user_formats
        except FileNotFoundError:
            # This is not an error, the user might not have a custom file.
            return []
        except Exception as e:
            print(f"Error loading user-defined provider configurations from {self.user_formats_file}: {e}")
            return []

    def _save_to_cache(self, formats: List[ProviderFormat]):
        """Saves the current provider_formats to the JSON cache file."""
        if not formats:
            return
        
        try:
            if not os.path.exists(CACHE_FOLDER):
                os.makedirs(CACHE_FOLDER)
            
            formats_as_dict = [asdict(fmt) for fmt in formats]
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(formats_as_dict, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(formats)} provider configurations to cache at {self.cache_file}")
        except Exception as e:
            print(f"Error saving provider configurations to cache: {e}")

    def _load_from_cache(self) -> List[ProviderFormat]:
        """Loads provider_formats from the JSON cache file. Returns a list of formats."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                formats_from_json = json.load(f)
            
            formats = [ProviderFormat(**item) for item in formats_from_json]
            print(f"Loaded {len(formats)} provider configurations from cache.")
            return formats
        except FileNotFoundError:
            return [] # Expected case, not an error
        except Exception as e:
            print(f"Error loading provider configurations from cache: {e}")
            return []

    def _load_from_js_content(self, js_content: str) -> List[ProviderFormat]:
        """
        Load provider configurations from JavaScript content. Returns a list of formats or raises an exception on failure.
        """
        try:
            parsed_formats = self.js_parser.parse_js_file(js_content)
            if not parsed_formats:
                raise ValueError("No provider formats found in JavaScript content.")
            
            print(f"Loaded {len(parsed_formats)} provider configurations from JavaScript.")
            return parsed_formats
        except Exception as e:
            raise ValueError(f"Error parsing JavaScript content: {e}") from e

    def parse_file(self, uploaded_file) -> pd.DataFrame:
        """
        Tries to parse the uploaded file with all available format configurations.
        """
        if uploaded_file is None:
            return pd.DataFrame()

        try:
            if hasattr(uploaded_file, 'getvalue'):
                file_content = uploaded_file.getvalue().decode('utf-8-sig')
            else:
                file_content = uploaded_file.read()
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8-sig')
        except Exception as e:
            print(f"Error reading file: {e}")
            return pd.DataFrame()

        for provider_format in self.provider_formats:
            try:
                df = self._try_parse(io.StringIO(file_content), provider_format)
                if not df.empty:
                    print(f"Successfully parsed with format: {provider_format.name}")
                    return self._standardize_dataframe(df)
            except Exception as e:
                print(e)
                continue
        
        print("No suitable parser found for the uploaded file.")
        return pd.DataFrame()

    def _try_parse(self, file_content_io: io.StringIO, config: ProviderFormat) -> pd.DataFrame:
        """
        Enhanced parser that handles more complex cases from JavaScript configurations.
        """
        df = pd.read_csv(file_content_io, sep=config.separator, decimal=config.decimal,
                         skiprows=config.skiprows, encoding=config.encoding, 
                         skipinitialspace=True, on_bad_lines='skip')

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        required_cols = [config.timestamp_col] + config.other_cols #type: ignore
        if config.time_sub_col:
            required_cols.append(config.time_sub_col)
        if config.end_timestamp_col:
            required_cols.append(config.end_timestamp_col)
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for format: {config.name}")

        # Match usage column (with fuzzy matching)
        usage_col_name = self._find_usage_column(df.columns, config.usage_col)
        if not usage_col_name:
            raise ValueError(f"Usage column not found for format: {config.name}")
        
        # Process entries
        df.rename(columns={usage_col_name: "consumption_kwh"}, inplace=True)
        
        # Handle timestamp combination
        if config.time_sub_col:
            df["timestamp_str"] = (df[config.timestamp_col].astype(str).str.strip() + 
                                 " " + df[config.time_sub_col].astype(str).str.strip())
        else:
            df["timestamp_str"] = df[config.timestamp_col].astype(str).str.strip()

        # Apply preprocessing if needed
        if config.preprocess_date_func:
            df["timestamp_str"] = df["timestamp_str"].apply(self._apply_date_preprocessing)

        # Parse consumption values
        df['consumption_kwh'] = pd.to_numeric(
            df['consumption_kwh'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        
        # Apply should_skip logic
        if config.should_skip_func:
            df = self._apply_skip_logic(df, config.should_skip_func)
        
        # Filter based on end timestamp if specified
        if config.end_timestamp_col:
            df = self._filter_by_time_interval(df, config)
        
        df.dropna(subset=['timestamp_str', 'consumption_kwh'], inplace=True)
        
        # Parse timestamps
        if config.date_format == "ISO8601":
            df["timestamp_local"] = pd.to_datetime(df["timestamp_str"])
        else:
            df["timestamp_local"] = pd.to_datetime(df["timestamp_str"], 
                                                 format=config.date_format, dayfirst=True)
        
        # Apply timestamp fixup
        if config.fixup_timestamp:
             df["timestamp_local"] -= pd.Timedelta(minutes=15)

        return df[["timestamp_local", "consumption_kwh"]].dropna()

    def _find_usage_column(self, columns: pd.Index, usage_descriptor: str) -> Optional[str]:
        """
        Find the usage column using exact match or fuzzy matching.
        """
        if usage_descriptor.startswith('!'):
            # Fuzzy matching
            fuzzy_match_str = usage_descriptor[1:]
            for col in columns:
                if fuzzy_match_str in col:
                    return col
        elif usage_descriptor in columns:
            return usage_descriptor
        return None

    def _apply_date_preprocessing(self, date_str: str) -> str:
        """
        Apply date preprocessing (simplified version).
        """
        # Handle date range splitting (like "01.01.2023-02.01.2023" -> "01.01.2023")
        if "-" in date_str and len(date_str.split("-")) == 2:
            return date_str.split("-")[0].strip()
        return date_str

    def _apply_skip_logic(self, df: pd.DataFrame, skip_func_str: str) -> pd.DataFrame:
        """
        Apply skip logic based on the function string (simplified implementation).
        """
        # This is a simplified implementation
        # In a full implementation, you'd parse and evaluate the JavaScript function
        if "1.8.0" in skip_func_str:
            # Example: Skip rows where OBIS code is not "1.8.0"
            if "Obiscode" in df.columns:
                df = df[df["Obiscode"] == "1.8.0"]
        return df

    def _filter_by_time_interval(self, df: pd.DataFrame, config: ProviderFormat) -> pd.DataFrame:
        """
        Filter entries based on time interval (remove daily aggregates).
        """
        try:
            if config.end_timestamp_col in df.columns:
                start_times = pd.to_datetime(df["timestamp_str"], format=config.date_format)
                end_times = pd.to_datetime(df[config.end_timestamp_col], format=config.date_format)
                interval_minutes = (end_times - start_times).dt.total_seconds() / 60
                # Keep only 15-minute intervals
                df = df[interval_minutes <= 15]
        except Exception:
            pass
        return df

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp to UTC and standardize the output format.
        """
        import pandas as pd

        def localize_with_dst_disambiguation(df: pd.DataFrame, timezone: str) -> pd.Series:
            """
            Localize 'timestamp_local' with proper DST disambiguation and convert to UTC.

            Parameters:
                df: DataFrame with a 'timestamp_local' column (naive datetime)
                timezone: Timezone string (e.g., 'Europe/Vienna')

            Returns:
                A pandas Series with timezone-aware UTC timestamps.
            """

            # Assign alternating True/False for is_dst for duplicated times
            dst_flags = []
            counter = {}

            for ts in df["timestamp_local"]:
                if ts in counter:
                    counter[ts] += 1
                else:
                    counter[ts] = 0
                dst_flags.append(counter[ts] % 2 == 0)  # True, False, True, False...

            df["is_dst"] = dst_flags

            # Step 3: Localize using is_dst to resolve ambiguity
            def safe_localize(row):
                try:
                    return row["timestamp_local"].tz_localize(timezone, ambiguous=row["is_dst"]).tz_convert("UTC")
                except Exception:
                    # Non-ambiguous timestamps default to standard localization
                    return row["timestamp_local"].tz_localize(timezone).tz_convert("UTC")

            return df.apply(safe_localize, axis=1)

        df = df.sort_values(by='timestamp_local').reset_index(drop=True)
        
        if df["timestamp_local"].dt.tz is not None:
            df["timestamp"] = df["timestamp_local"].dt.tz_convert("UTC")
        else:
            # Winter/summer time may lead to errors. Therefore use workaround
            df["timestamp"] = localize_with_dst_disambiguation(df, "Europe/Vienna")
            #df["timestamp"] = df["timestamp_local"].dt.tz_localize(self.local_timezone, ambiguous='infer').dt.tz_convert("UTC")

        # Resamples the data to hourly basis if it is not in 15 minutes (e.g. for days).
        aggregation_level = "15min" if get_intervals_per_day(df) > 24 else "h"
        df = df.set_index("timestamp")["consumption_kwh"].resample(aggregation_level).sum().dropna().reset_index()
        
        return df[["timestamp", "consumption_kwh"]].reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    
    # Create parser with default configurations
    parser = ConsumptionDataParser()
    
    print("Available provider formats:")
    for fmt in parser.provider_formats:
        print(f"- {fmt.name}")
