import pandas as pd
import io
import re
import requests
import os
import subprocess
import shutil
import json
import pytz
from typing import List, Optional
from dataclasses import dataclass, asdict
from src.config import LOCAL_TIMEZONE, CACHE_FOLDER
from src.logger import logger


@dataclass
class ProviderFormat:
    """Holds the parsing configuration for a specific electricity provider"s CSV format. """
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
    encoding: str = "utf-8-sig"
    feedin: bool = False
    end_timestamp_col: Optional[str] = None
    preprocess_date_func: Optional[str] = None


class JavaScriptNetzbetreiberParser:
    """Parses Netzbetreiber configurations from aWATTar JavaScript files."""

    def __init__(self):
        self.date_format_map = {
            "dd.MM.yyyy HH:mm": "%d.%m.%Y %H:%M",
            "dd.MM.yyyy HH:mm:ss": "%d.%m.%Y %H:%M:%S",
            "dd.MM.yy HH:mm": "%d.%m.%y %H:%M",
            "dd.MM.yy HH:mm:ss": "%d.%m.%y %H:%M:%S",
            "yyyy-MM-dd HH:mm:ss": "%Y-%m-%d %H:%M:%S",
            " dd.MM.yyyy HH:mm:ss": " %d.%m.%Y %H:%M:%S",
            "parseISO": "ISO8601",
        }

    def parse_js_file(self, js_content: str) -> List[ProviderFormat]:
        """Extracts Netzbetreiber provider configurations from JavaScript content."""
        providers = []
        pattern = r"export const (\w+) = new Netzbetreiber\((\{.*?\})\);"
        matches = re.finditer(pattern, js_content, re.DOTALL)

        for match in matches:
            var_name = match.group(1)
            obj_str = match.group(2)
            try:
                provider = self._parse_object_literal(var_name, obj_str)
                if provider:
                    providers.append(provider)
            except Exception as e:
                logger.log(f"Error parsing provider config for '{var_name}': {e}", severity=1)

        return providers

    def _parse_object_literal(self, var_name: str, obj_str: str) -> Optional[ProviderFormat]:
        """Parses a single JavaScript object literal into a ProviderFormat instance."""
        name = self._extract_str("name", obj_str) or var_name
        usage_col = self._extract_str("descriptorUsage", obj_str)
        timestamp_col = self._extract_str("descriptorTimestamp", obj_str)
        if not usage_col or not timestamp_col:
            return None

        time_sub_col = self._extract_str("descriptorTimeSub", obj_str)
        date_format_js = self._extract_str("dateFormatString", obj_str) or self._extract_ident("dateFormatString", obj_str)
        date_format = self.date_format_map.get(date_format_js, date_format_js) if date_format_js else "%d.%m.%Y %H:%M"

        other_cols = self._extract_list("otherFields", obj_str)
        fixup_timestamp = self._extract_bool("fixupTimestamp", obj_str, default=False)
        feedin = self._extract_bool("feedin", obj_str, default=False)
        end_timestamp_col = self._extract_str("endDescriptorTimestamp", obj_str)

        return ProviderFormat(
            name=name,
            usage_col=usage_col,
            timestamp_col=timestamp_col,
            time_sub_col=time_sub_col,
            date_format=date_format,
            other_cols=other_cols,
            fixup_timestamp=fixup_timestamp,
            feedin=feedin,
            end_timestamp_col=end_timestamp_col,
        )

    def _extract_str(self, key: str, text: str) -> Optional[str]:
        """Extracts a quoted string property value by key."""
        m = re.search(rf"{key}\s*:\s*[\"']([^\"']+)[\"']", text)
        return m.group(1).strip() if m else None

    def _extract_ident(self, key: str, text: str) -> Optional[str]:
        """Extracts an unquoted identifier property value by key."""
        m = re.search(rf"{key}\s*:\s*(\w+)", text)
        return m.group(1).strip() if m else None

    def _extract_bool(self, key: str, text: str, default: bool = False) -> bool:
        """Extracts a boolean property value by key."""
        m = re.search(rf"{key}\s*:\s*(true|false)", text, re.IGNORECASE)
        return m.group(1).lower() == "true" if m else default

    def _extract_list(self, key: str, text: str) -> List[str]:
        """Extracts a list of strings property value by key."""
        m = re.search(rf"{key}\s*:\s*\[(.*?)\]", text, re.DOTALL)
        if not m:
            return []
        items = re.findall(r"""["']([^"']+)["']""", m.group(1))
        return [item.strip() for item in items if item.strip()]


class ConsumptionDataParser:
    """Parser that can load configurations from JavaScript (awattar backtesting) and parse various formats of electricity consumption data. """

    def __init__(self,
                 local_timezone=LOCAL_TIMEZONE,
                 js_url="https://raw.githubusercontent.com/awattar-backtesting/awattar-backtesting.github.io/main/docs/netzbetreiber.js",
                 js_content=None):
        self.local_timezone = local_timezone
        self.js_parser = JavaScriptNetzbetreiberParser()
        self.cache_file = os.path.join(CACHE_FOLDER, "provider_formats.json")
        self.user_formats_file = os.path.join(CACHE_FOLDER, "additional_provider_formats.json")

        # Load user-defined formats, they have priority.
        user_formats = self._load_user_defined_formats()

        # Load formats from JS, then cache, then defaults.
        main_formats = []

        if js_content:
            # Direct content parsing, no caching involved.
            try:
                main_formats = self._load_from_js_content(js_content)
            except ValueError as e:
                logger.log(f"Failed to parse provider config from direct content: {e}", severity=1)
        elif js_url and not (main_formats := self._load_from_cache()):
            try:
                response = requests.get(js_url, timeout=10)
                response.raise_for_status()
                main_formats = self._load_from_js_content(response.text)
                self._save_to_cache(main_formats)
            except (requests.RequestException, ValueError) as e:
                logger.log(f"Failed to fetch or parse from URL '{js_url}': {e}", severity=1)

        # If main_formats is still empty (JS fetch failed or was not attempted)
        if not main_formats:
            logger.log("Attempting to load main provider configurations from cache...")
            main_formats = self._load_from_cache()

        # Combine lists. User formats are first in the list.
        self.provider_formats = user_formats + main_formats
        logger.log(f"Total of {len(self.provider_formats)} provider formats loaded ({len(user_formats)} user-defined, {len(main_formats)} main).")
        self._ensure_upstream_preprocessor()

    def _load_user_defined_formats(self) -> List[ProviderFormat]:
        """Loads user-defined provider formats from own_provider_formats.json."""
        try:
            with open(self.user_formats_file, "r", encoding="utf-8") as f:
                formats_from_json = json.load(f)

            user_formats = [ProviderFormat(**item) for item in formats_from_json]
            logger.log(f"Loaded {len(user_formats)} user-defined provider configurations from {self.user_formats_file}.", severity=1)
            return user_formats
        except FileNotFoundError:
            # This is not an error, the user might not have a custom file.
            return []
        except Exception as e:
            logger.log(f"Error loading user-defined provider configurations from {self.user_formats_file}: {e}", severity=1)
            return []

    def _save_to_cache(self, formats: List[ProviderFormat]):
        """Saves the current provider_formats to the JSON cache file."""
        if not formats:
            return

        try:
            if not os.path.exists(CACHE_FOLDER):
                os.makedirs(CACHE_FOLDER)

            formats_as_dict = [asdict(fmt) for fmt in formats]

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(formats_as_dict, f, indent=4, ensure_ascii=False)
            logger.log(f"Saved {len(formats)} provider configurations to cache at {self.cache_file}", severity=1)
        except Exception as e:
            logger.log(f"Error saving provider configurations to cache: {e}", severity=1)

    def _load_from_cache(self) -> List[ProviderFormat]:
        """Loads provider_formats from the JSON cache file. Returns a list of formats."""
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                formats_from_json = json.load(f)

            formats = [ProviderFormat(**item) for item in formats_from_json]
            logger.log(f"Loaded {len(formats)} provider configurations from cache.", severity=1)
            return formats
        except FileNotFoundError:
            return [] # Expected case, not an error
        except Exception as e:
            logger.log(f"Error loading provider configurations from cache: {e}", severity=1)
            return []

    def _load_from_js_content(self, js_content: str) -> List[ProviderFormat]:
        """Load provider configurations from JavaScript content. Returns a list of formats or raises an exception on failure. """
        try:
            parsed_formats = self.js_parser.parse_js_file(js_content)
            if not parsed_formats:
                raise ValueError("No provider formats found in JavaScript content.")

            logger.log(f"Loaded {len(parsed_formats)} provider configurations from JavaScript.", severity=1)
            return parsed_formats
        except Exception as e:
            raise ValueError(f"Error parsing JavaScript content: {e}") from e

    def _ensure_upstream_preprocessor(self):
        """Fetches and caches upstream preprocess.js and encoding.js to build runner.js."""
        try:
            js_dir = os.path.join(CACHE_FOLDER, "upstream_js")
            os.makedirs(js_dir, exist_ok=True)
            prep_file = os.path.join(js_dir, "preprocess.js")
            enc_file = os.path.join(js_dir, "encoding.js")
            runner_file = os.path.join(js_dir, "runner.js")

            # Try to fetch upstream files if missing
            base_url = "https://raw.githubusercontent.com/awattar-backtesting/awattar-backtesting.github.io/main/docs/calc/"
            for fname, fpath in [("preprocess.js", prep_file), ("encoding.js", enc_file)]:
                if not os.path.exists(fpath):
                    try:
                        resp = requests.get(base_url + fname, timeout=5)
                        if resp.status_code == 200:
                            with open(fpath, "w", encoding="utf-8") as f:
                                f.write(resp.text)
                    except Exception as e:
                        logger.log(f"Could not download upstream {fname}: {e}", severity=0)

            # Build self-contained runner.js if preprocess and encoding exist
            if os.path.exists(prep_file) and os.path.exists(enc_file):
                with open(enc_file, "r", encoding="utf-8") as f:
                    enc_code = f.read().replace("export function", "function")
                with open(prep_file, "r", encoding="utf-8") as f:
                    prep_code = re.sub(r"import\s+.*?;\s*", "", f.read()).replace("export function", "function")

                runner_code = (
                    "const fs = require('fs');\n\n"
                    + enc_code + "\n"
                    + prep_code + "\n"
                    + "const buf = fs.readFileSync(0);\n"
                    + "const stripped = stripPlain(buf);\n"
                    + "let output;\n"
                    + "if (typeof stripped === 'string') {\n"
                    + "    output = stripped;\n"
                    + "} else if (stripped && stripped.buffer) {\n"
                    + "    output = bufferToString(stripped);\n"
                    + "} else if (stripped instanceof ArrayBuffer) {\n"
                    + "    output = bufferToString(stripped);\n"
                    + "} else {\n"
                    + "    output = bufferToString(buf);\n"
                    + "}\n"
                    + "output = output.replace(/^\\uFEFF/, '');\n"
                    + "output = output.replace(/^sep=;[\\r\\n]+/, '');\n"
                    + "process.stdout.write(output, 'latin1');\n"
                )
                with open(runner_file, "w", encoding="utf-8") as f:
                    f.write(runner_code)
        except Exception as e:
            logger.log(f"Error initializing upstream preprocessor: {e}", severity=0)

    def _preprocess_content(self, raw_content: bytes) -> str:
        """Runs upstream JS stripPlain preprocessing via Node.js if available, with python fallback."""
        runner_file = os.path.join(CACHE_FOLDER, "upstream_js", "runner.js")
        if shutil.which("node") and os.path.exists(runner_file):
            try:
                proc = subprocess.Popen(
                    ["node", runner_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                out, _ = proc.communicate(input=raw_content, timeout=5)
                if proc.returncode == 0 and out:
                    try:
                        text = out.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = out.decode("latin-1")
                    text = text.lstrip("\ufeff")
                    if text.startswith("sep=;\r\n"):
                        text = text[7:]
                    elif text.startswith("sep=;\n"):
                        text = text[6:]
                    return text
            except Exception as e:
                logger.log(f"Upstream node preprocessor error: {e}, falling back", severity=0)

        # Python fallback if node is not present or failed
        try:
            text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw_content.decode("latin-1")

        text = text.lstrip("\ufeff")
        if text.startswith("sep=;\r\n"):
            text = text[7:]
        elif text.startswith("sep=;\n"):
            text = text[6:]

        if all(k in text for k in ["Kundennummer", "Kundenname", "ZP-Nummer", "Energierichtung"]):
            text = "\n".join(text.splitlines()[8:])
        tiwag_lines = text.splitlines()
        if len(tiwag_lines) >= 5 and "DATE_FROM;DATE_TO;VALUE" in tiwag_lines[4]:
            text = "\n".join(tiwag_lines[4:])

        return text

    def parse_file(self, uploaded_file) -> pd.DataFrame:
        """Tries to parse the uploaded file with all available format configurations."""
        if uploaded_file is None:
            return pd.DataFrame()

        try:
            if hasattr(uploaded_file, "getvalue"):
                raw_content = uploaded_file.getvalue()
            else:
                raw_content = uploaded_file.read()
            if isinstance(raw_content, str):
                raw_content = raw_content.encode("utf-8")
        except Exception as e:
            logger.log(f"Error reading uploaded file: {e}", severity=1)
            return pd.DataFrame()

        file_content = self._preprocess_content(raw_content)

        for provider_format in self.provider_formats:
            try:
                df = self._try_parse(io.StringIO(file_content), provider_format)
                if not df.empty:
                    logger.log(f"Successfully parsed with format: {provider_format.name}", severity=1)
                    return self._standardize_dataframe(df)
            except Exception as e:
                logger.log(f"Format {provider_format.name} skipped: {e}", severity=0)

        logger.log("No suitable parser found for the uploaded file.", severity=1)
        return pd.DataFrame()

    def _try_parse(self, file_content_io: io.StringIO, config: ProviderFormat) -> pd.DataFrame:
        """Enhanced parser that handles more complex cases from JavaScript configurations."""
        df = pd.read_csv(file_content_io, sep=config.separator, decimal=config.decimal,
                         skiprows=config.skiprows, encoding=config.encoding,
                         skipinitialspace=True, on_bad_lines="skip")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Check required columns
        required_cols = [config.timestamp_col] + (config.other_cols if config.other_cols else [])
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
            df["timestamp_str"] = (df[config.timestamp_col].astype(str).str.strip() + " " + df[config.time_sub_col].astype(str).str.strip())
        else:
            df["timestamp_str"] = df[config.timestamp_col].astype(str).str.strip()

        # Apply preprocessing if needed
        if config.preprocess_date_func:
            # Vectorized operation is much faster than .apply()
            df["timestamp_str"] = df["timestamp_str"].str.split('-').str[0].str.strip()

        # Parse consumption values
        df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"].astype(str).str.replace(",", "."), errors="coerce")

        # Apply should_skip logic
        if config.should_skip_func:
            df = self._apply_skip_logic(df, config.should_skip_func)

        # Filter based on end timestamp if specified
        if config.end_timestamp_col:
            df = self._filter_by_time_interval(df, config)

        df.dropna(subset=["timestamp_str", "consumption_kwh"], inplace=True)

        # Parse timestamps
        if config.date_format == "ISO8601":
            df["timestamp_local"] = pd.to_datetime(df["timestamp_str"], utc=True)
        else:
            df["timestamp_local"] = pd.to_datetime(df["timestamp_str"], format=config.date_format, dayfirst=True)

        # Apply timestamp fixup
        if config.fixup_timestamp:
            df["timestamp_local"] -= pd.Timedelta(minutes=15)

        return df[["timestamp_local", "consumption_kwh"]].dropna()

    def _find_usage_column(self, columns: pd.Index, usage_descriptor: str) -> Optional[str]:
        """Find the usage column using exact match or fuzzy matching."""
        if usage_descriptor.startswith("!"):
            # Fuzzy matching
            fuzzy_match_str = usage_descriptor[1:]
            for col in columns:
                if fuzzy_match_str in col:
                    return col
        elif usage_descriptor in columns:
            return usage_descriptor
        return None

    def _apply_skip_logic(self, df: pd.DataFrame, skip_func_str: str) -> pd.DataFrame:
        """Apply skip logic based on the function string (simplified implementation)."""
        # This is a simplified implementation
        # TODO: parse and evaluate the JavaScript function
        if "1.8.0" in skip_func_str:
            # Example: Skip rows where OBIS code is not "1.8.0"
            if "Obiscode" in df.columns:
                df = df[df["Obiscode"] == "1.8.0"]
        return df

    def _filter_by_time_interval(self, df: pd.DataFrame, config: ProviderFormat) -> pd.DataFrame:
        """Filter entries based on time interval (remove daily aggregates)."""
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
        """Convert timestamp to UTC and standardize the output format. Handles DST transitions robustly."""

        from src.utils import get_intervals_per_day

        def handle_dst_transitions(df: pd.DataFrame, timezone_str: str) -> pd.Series:
            """
            Handle DST transitions by identifying and processing different types of timestamps.

            Parameters:
                df: DataFrame with a "timestamp_local" column (naive datetime)
                timezone_str: Timezone string (e.g., "Europe/Vienna")

            Returns:
                A pandas Series with timezone-aware UTC timestamps.
            """

            timezone = pytz.timezone(timezone_str)
            utc_timestamps = []

            for timestamp in df["timestamp_local"]:
                try:
                    # First, try normal localization
                    localized = timezone.localize(timestamp)
                    utc_timestamps.append(localized.astimezone(pytz.UTC))

                except pytz.AmbiguousTimeError:
                    # Handle fall-back transition (ambiguous time)
                    # Default to DST=False (standard time) for consistency
                    localized = timezone.localize(timestamp, is_dst=False)
                    utc_timestamps.append(localized.astimezone(pytz.UTC))
                    logger.log(f"Ambiguous time {timestamp} resolved to standard time")

                except pytz.NonExistentTimeError:
                    # Handle spring-forward transition (non-existent time)
                    # Move forward to the next valid time
                    try:
                        # Try adding 1 hour to skip the gap
                        adjusted_timestamp = timestamp + pd.Timedelta(hours=1)
                        localized = timezone.localize(adjusted_timestamp)
                        utc_timestamps.append(localized.astimezone(pytz.UTC))
                        logger.log(f"Non-existent time {timestamp} adjusted to {adjusted_timestamp}")
                    except Exception:
                        # If that fails, use UTC directly
                        utc_timestamps.append(timestamp.replace(tzinfo=pytz.UTC))
                        logger.log(f"Non-existent time {timestamp} treated as UTC")

                except Exception as e:
                    # Fallback for any other errors
                    logger.log(f"Unexpected error localizing {timestamp}: {e}")
                    utc_timestamps.append(timestamp.replace(tzinfo=pytz.UTC))

            return pd.Series(utc_timestamps, index=df.index)

        # Input validation
        if df.empty:
            logger.log("Input DataFrame is empty")
            return df

        if "timestamp_local" not in df.columns:
            logger.log("DataFrame missing 'timestamp_local' column")
            return df

        if "consumption_kwh" not in df.columns:
            logger.log("DataFrame missing 'consumption_kwh' column")
            return df

        # Sort by timestamp to ensure proper ordering
        df = df.sort_values(by="timestamp_local").reset_index(drop=True)

        # Handle timezone conversion
        if df["timestamp_local"].dt.tz is not None:
            # Already timezone-aware, just convert to UTC
            df["timestamp"] = df["timestamp_local"].dt.tz_convert("UTC")
            logger.log("Converted timezone-aware timestamps to UTC")

        else:
            # Handle naive timestamps
            timezone_str = getattr(self, "local_timezone", "Europe/Vienna")

            try:
                # Use our robust DST handler
                df["timestamp"] = handle_dst_transitions(df, timezone_str)
                logger.log(f"Successfully localized naive timestamps using {timezone_str}")

            except Exception as e:
                logger.log(f"Error in DST transition handling: {e}")

                # Final fallback: treat as UTC
                try:
                    df["timestamp"] = df["timestamp_local"].dt.tz_localize("UTC")
                    logger.log("Fallback: treated naive timestamps as UTC")
                except Exception as fallback_error:
                    logger.log(f"Even fallback failed: {fallback_error}")
                    return pd.DataFrame()  # Return empty DataFrame on complete failure

        # Validate that we have valid timestamps
        if df["timestamp"].isna().any():
            logger.log("Some timestamps could not be converted, dropping NaT values")
            df = df.dropna(subset=["timestamp"])

        if df.empty:
            logger.log("No valid timestamps after conversion")
            return df

        # Determine aggregation level
        try:
            intervals_per_day = get_intervals_per_day(df)
            if intervals_per_day > 24:
                aggregation_level = "15min"
            elif intervals_per_day > 1:
                aggregation_level = "h"
            else:
                aggregation_level = "1D"
            logger.log(f"Using aggregation level: {aggregation_level}")
        except Exception as e:
            logger.log(f"Could not determine intervals per day: {e}, defaulting to hourly")
            aggregation_level = "h"

        # Resample data
        try:
            df_resampled = (df.set_index("timestamp")["consumption_kwh"]
                            .resample(aggregation_level)
                            .sum()
                            .dropna()
                            .reset_index())

            # Ensure we have the expected columns
            df_result = df_resampled[["timestamp", "consumption_kwh"]].reset_index(drop=True)

            # --- Memory Optimization ---
            # Downcast numeric columns to the smallest possible type to save memory
            df_result["consumption_kwh"] = pd.to_numeric(df_result["consumption_kwh"], downcast="float")

            # Convert object columns to category if they have a low number of unique values
            for col in df_result.select_dtypes(include=["object"]).columns:
                if df_result[col].nunique() / len(df_result) < 0.5:
                    df_result[col] = df_result[col].astype("category")
            # --- End Memory Optimization ---

            if df_result.empty:
                logger.log("DataFrame is empty after resampling")
            else:
                logger.log(f"Successfully resampled to {len(df_result)} rows")

            return df_result

        except Exception as e:
            logger.log(f"Error during resampling: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":

    # Create parser with default configurations
    parser = ConsumptionDataParser()

    logger.log("Available provider formats:")
    for fmt in parser.provider_formats:
        logger.log(f"- {fmt.name}")
