from datetime import date

# --- General Configuration ---
LOCAL_TIMEZONE = "Europe/Vienna"
MIN_DATE = date(2022, 1, 1)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- API and Caching Configuration ---
CACHE_FOLDER = "cache"
SPOT_PRICE_CACHE_FILE = "spot_prices.csv"
AWATTAR_COUNTRY = "at"  # or "de"

# --- UI Color Scheme ---
FLEX_COLOR = "#f96407"
FLEX_COLOR_LIGHT = "#f29f6c" 
FLEX_COLOR_SHADE = "rgba(249, 100, 7, 0.08)"
STATIC_COLOR = "#989898"
PERSONAL_DATA_COLOR = "#2a9d8f" # A teal/green color
PERSONAL_DATA_COLOR_LIGHT = "#78c2b9"
PERSONAL_DATA_COLOR_SHADE = "rgba(42, 157, 143, 0.1)" # A light shade for fill
BASE_COLOR = STATIC_COLOR # Base load is less influenceable, so grey is fine
REGULAR_COLOR = PERSONAL_DATA_COLOR # Regular load is part of personal consumption
FORECAST_ACTUAL_COLOR = PERSONAL_DATA_COLOR_LIGHT
FORECAST_PREDICTED_COLOR = PERSONAL_DATA_COLOR
FORECAST_UNCERTAINTY_COLOR = "rgba(68, 68, 68, 0.2)"
PEAK_COLOR = "#f9dd07"
GREEN = "#5fba7d"
RED = "#d65f5f"
MEKKO_BORDER = 0.002

# --- Analysis Parameters ---
ABSENCE_THRESHOLD = 0.75 # Determines a day of absence if consumption is below 75% of the daily base load.

# --- Usage Classification Parameters ---
NEGLIGABLE_KWH = 0.05
BASE_QUANTILE_THRESHOLD = 0.9
PEAK_QUANTILE_THRESHOLD = 0.7
STD_MULTIPLE = 1.25

# --- Forecast Parameters ---
THRESHOLD_STABLE_TREND = 5

# --- DEBUG Parameters ---
TODAY_IS_MAX_DATE = True
DEBUG = True