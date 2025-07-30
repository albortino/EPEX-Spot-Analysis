from datetime import date

# --- General Configuration ---
LOCAL_TIMEZONE = "Europe/Vienna"
MIN_DATE = date(2024, 1, 1)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- API and Caching Configuration ---
SPOT_PRICE_CACHE_FILE = "spot_prices.csv"
AWATTAR_COUNTRY = "at"  # or "de"

# --- UI Color Scheme ---
FLEX_COLOR = "#fd690d"
FLEX_COLOR_LIGHT = "#f7be44"
STATIC_COLOR = "#989898"
GREEN = "#5fba7d"
RED = "#d65f5f"

# --- Analysis Parameters ---
ABSENCE_THRESHOLD = 0.75 # Determines a day of absence if consumption is below 75% of the daily base load.