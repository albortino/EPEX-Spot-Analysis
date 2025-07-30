from datetime import date

# --- General Configuration ---
LOCAL_TIMEZONE = "Europe/Vienna"
MIN_DATE = date(2024, 1, 1)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- API and Caching Configuration ---
SPOT_PRICE_CACHE_FILE = "spot_prices.csv"
AWATTAR_COUNTRY = "at"  # or "de"

# --- UI Color Scheme ---
FLEX_COLOR = "#f96407"
FLEX_COLOR_LIGHT = "#f49a61" 
STATIC_COLOR = "#989898"
BASE_COLOR = STATIC_COLOR
REGULAR_COLOR = FLEX_COLOR
PEAK_COLOR = "#f9dd07" 
GREEN = "#5fba7d"
RED = "#d65f5f"

# --- Analysis Parameters ---
ABSENCE_THRESHOLD = 0.75 # Determines a day of absence if consumption is below 75% of the daily base load.

# --- Usage Classification Parameters ---
NEGLIGABLE_KWH = 0.05
BASE_QUANTILE_THRESHOLD = 0.9
PEAK_QUANTILE_THRESHOLD = 0.7
STD_MULTIPLE = 1.25