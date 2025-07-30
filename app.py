import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import random
from tariffs import TariffManager, Tariff
from parser import ConsumptionDataParser
from methods import * # Core analysis methods

# --- Page and App Configuration ---

st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.\nThis project was influenced by https://awattar-backtesting.github.io/, which provides a simple and effective overview. This tool provides further insights into your consumption behavior and help you to choose the most economic tariff plan.")

# --- Constants and Configuration ---

LOCAL_TIMEZONE = "Europe/Vienna"
MIN_DATE = date(2024, 1, 1)
FLEX_COLOR = "#fd690d"
FLEX_COLOR_LIGHT = "#f7be44"
STATIC_COLOR = "#989898"
GREEN = "#5fba7d"
RED = "#d65f5f"

ABSENCE_THRESHOLD = 0.75

SPOT_PRICE_CACHE_FILE = "spot_prices.csv"
AWATTAR_COUNTRY = "at" # or de

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Data Loading and Caching ---

def _load_from_cache(file_path: str) -> pd.DataFrame:
    """Loads the stored EPEX prices from the local cache."""
    try:
        df_cache = pd.read_csv(file_path, parse_dates=["timestamp"])
        df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], utc=True)
        return df_cache
    
    except Exception as e:
        st.warning(f"Could not read or parse cache file '{SPOT_PRICE_CACHE_FILE}'. Refetching data. Error: {e}")
        return pd.DataFrame()

def _fetch_spot_data(start: date, end: date) -> pd.DataFrame:
    """Fetches the spot data from the aWATTar API for a given date range."""
    
    base_url = f"https://api.awattar.{AWATTAR_COUNTRY}/v1/marketdata"
    start_dt, end_dt = datetime.combine(start, datetime.min.time()), datetime.combine(end + pd.Timedelta(days=1), datetime.min.time())
    params = {"start": int(start_dt.timestamp() * 1000),
              "end": int(end_dt.timestamp() * 1000)}
    
    try:
        print(f"{datetime.now().strftime(DATE_FORMAT)}: Will fetch data from Awattar.")
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
            
        data = response.json()["data"]
        if not data:
            st.warning("API returned no data for the selected period.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
        df["spot_price_eur_kwh"] = df["marketprice"] / 1000 * 1.2 # Convert from Eur/MWh to c/kWh and add VAT
        
        df_to_return = df[["timestamp", "spot_price_eur_kwh"]]        
        df_to_return.to_csv(SPOT_PRICE_CACHE_FILE, index=False)

        return df_to_return    

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch spot price data: {e}")
        
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API.")

    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_spot_data(start: date, end: date) -> pd.DataFrame:
    """
    Fetches spot market price data from the aWATTar API for a given date range.
    It first checks a local CSV file for cached data covering the entire range.
    If the cache is not present or incomplete, it fetches from the API and updates the cache.
    """
    now = datetime.now().strftime(DATE_FORMAT)
    
    # Try to load from cache first
    if os.path.exists(SPOT_PRICE_CACHE_FILE):
        df_cache = _load_from_cache(SPOT_PRICE_CACHE_FILE)
        
        if not df_cache.empty:
            min_cached_date = df_cache["timestamp"].min().date()
            max_cached_date = df_cache["timestamp"].max().date()
            
            # If cache fully covers the requested range, use it
            if min_cached_date <= start and max_cached_date >= end:
                print(f"{now}: Loading spot prices from cache.")
                # Filter the cached data for the exact range and return
                return df_cache[(df_cache["timestamp"].dt.date >= start) & (df_cache["timestamp"].dt.date <= end)]
                
    # If cache is not sufficient or doesn't exist, fetch from API
    print(f"{now}: Cache insufficient or missing. Will fetch data from aWATTar.")
    
    return _fetch_spot_data(start, end)
        

@st.cache_data(ttl=3600)
def process_consumption_data(uploaded_file, aggregation_level: str = "h") -> pd.DataFrame:
    """Loads and processes the user"s consumption CSV using the new parser."""
    
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        parser = ConsumptionDataParser(local_timezone=LOCAL_TIMEZONE)
        df = parser.parse_file(uploaded_file, aggregation_level)
        if df.empty:
            st.error("Could not parse the CSV file. Please ensure it is from a supported provider or in the default format.")
        return df
    
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the file: {e}")
        return pd.DataFrame()

# --- UI and Rendering Functions ---

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to an Excel file in-memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df.to_excel(writer, index=False, sheet_name="AnalysisData")
    processed_data = output.getvalue()
    return processed_data

@st.cache_data(ttl=3600)
def _get_min_max_date(df: pd.DataFrame) -> tuple[date, date]:
    """Returns the minimum and maximum dates from a DataFrame with timestamp column."""
    min_date = df["timestamp"].min().date()
    if min_date < MIN_DATE:
        min_date = MIN_DATE
        
    max_date = df["timestamp"].max().date()
    
    return min_date, max_date

@st.cache_data(ttl=60*10) # 10 minutes validity if the input does not change
def _compare_all_tariffs(df: pd.DataFrame, flex_tariff_options: dict[str, Tariff], static_tariff_options: dict[str, Tariff]) -> tuple[Tariff, Tariff]:
    # Calculate total costs for all tariffs using the accurate methods from TariffManager
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Calculating comparison")
    
    total_costs = {}
    for name, tariff in flex_tariff_options.items():
        if name == "Custom": continue  # Exclude the editable "Custom" tariff from comparison
        total_costs[("flex", name)] = tariff_manager._calculate_flexible_cost(df.copy(), tariff).sum()
    
    for name, tariff in static_tariff_options.items():
        if name == "Custom": continue  # Exclude the editable "Custom" tariff from comparison
        total_costs[("static", name)] = tariff_manager._calculate_static_cost(df.copy(), tariff).sum()
    
    # Identify the cheapest flex and static tariffs
    cheapest_flex_key = min((k for k in total_costs if k[0] == "flex"), key=total_costs.get)
    cheapest_static_key = min((k for k in total_costs if k[0] == "static"), key=total_costs.get)
    
    final_flex_tariff = flex_tariff_options[cheapest_flex_key[1]]
    final_static_tariff = static_tariff_options[cheapest_static_key[1]]
    
    return final_flex_tariff, final_static_tariff    

def _return_tariff_selection(flex_tariff_options: dict[str, Tariff], static_tariff_options: dict[str, Tariff]) -> tuple[Tariff, Tariff]:
    final_tariffs = []
    for title, tariff in zip(["Flexible (Spot Price) Plan", "Static (Fixed Price) Plan"], [flex_tariff_options, static_tariff_options]):
        
        with st.expander(title, expanded=True):
            type_from_title = title.split(" ")[0]

            # Select a tariff.
            selected_name = st.selectbox(f"Select {type_from_title} Tariff", options=list(tariff.keys()), index=len(tariff)-1)
            selected_tariff = tariff[selected_name]
            
            # Create the input fields with pre-set values based on the custom tariff or pre-selected tariff.
            on_top = st.number_input("On-Top Price (€/kWh)", value=selected_tariff.price_kwh, min_value=0.0, step=0.01, format="%.4f")
            
            if type_from_title == "Flexible":
                on_top_perc = st.number_input("Variable Price (% of EPEX)", value=selected_tariff.price_kwh_pct, min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key=f"{type_from_title}_on_top_perc")
            else:
                on_top_perc = 0.0
            
            fee = st.number_input("Monthly Fee (€)", value=selected_tariff.monthly_fee, min_value=0.0, step=0.1)
            
            # Create the tariff instance and append it to a list that will be returned.
            final_tariff = Tariff(name=selected_tariff.name, type=selected_tariff.type, price_kwh=on_top, monthly_fee=fee, price_kwh_pct=on_top_perc)
            final_tariffs.append(final_tariff)

    return tuple(final_tariffs)

def get_sidebar_inputs(df: pd.DataFrame, tariff_manager: TariffManager):
    """Renders all sidebar inputs and returns the configuration values."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Sidebar")

    with st.sidebar:
        st.header("Configuration")
        
        # Filter the date of the analysis.
        st.subheader("1. Analysis Period")
        min_date, max_date = _get_min_max_date(df)
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
        end_date = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
        
        # Tariff plans the user can select from or edit. Also, the cheapest tariffs can be compared.
        st.subheader("2. Tariff Plans")
        st.text("Choose a tariff or adjust parameters by providing the gross costs (incl. VAT).")
        
        # Checkbox to compare the cheapest tariffs.
        compare_cheapest = st.checkbox("Compare cheapest tariffs", value=False)
                
        flex_tariff_options = tariff_manager.get_flex_tariffs_with_custom()
        static_tariff_options = tariff_manager.get_static_tariffs_with_custom()
        
        if compare_cheapest:
            # Set the cheapest static and flexible tariff
            final_flex_tariff, final_static_tariff = _compare_all_tariffs(df, flex_tariff_options, static_tariff_options)    
            st.info(f"Cheapest Flex Tariff:\n\n**{final_flex_tariff.name}**\n\nCheapest Static Tariff:\n\n**{final_static_tariff.name}**")
        else:
            # Generate two expander fields to preselect or edit tariffs.
            final_flex_tariff, final_static_tariff = _return_tariff_selection(flex_tariff_options, static_tariff_options)
            
        # Define how much of the peak consumption is shifted to the cheapest hour within a 2 hour timespan. 
        st.subheader("3. Cost Simulation", help="Simulate shifting a percentage of your peak consumption to a cheaper hour within a +/- 2-hour window.")
        shift_percentage = st.slider("Shift Peak Load (%)", min_value=0, max_value=100, value=0, step=5)

        return start_date, end_date, final_flex_tariff, final_static_tariff, shift_percentage

@st.cache_data(ttl=3600)
def render_recommendation(df: pd.DataFrame):
    """Displays the final tariff recommendation based on calculated savings."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Recommendation")

    st.subheader("Tariff Recommendation")
    savings = df["total_cost_static"].sum() - df["total_cost_flexible"].sum()
    
    # Calculate proportion of consumption during low-cost (cheapest quartile) periods. 
    df["price_quantile"] = df.groupby(pd.Grouper(key="timestamp", freq="MS"))["spot_price_eur_kwh"].transform(lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    peak_total_kwh = df["peak_load_kwh"].sum()
    peak_cheap_kwh = df[df["price_quantile"] == 0]["peak_load_kwh"].sum()
    peak_ratio = peak_cheap_kwh / peak_total_kwh if peak_total_kwh > 0 else 0

    if savings > 0:
        st.success(f"✅ Flexible Plan Recommended: You could have saved €{savings:.2f}")
        if peak_ratio > 0.3:
            st.write(f"This is a great fit. You align **{peak_ratio:.0%}** of your peak usage with the cheapest market prices.")
        else:
            st.write("You could save even more by shifting high-consumption activities to times with lower spot prices.")
    else:
        st.warning(f"⚠️ Static Plan Recommended: The flexible plan would have cost €{-savings:.2f} more.")
        st.write("A fixed price offers better cost stability for your current usage pattern.")

@st.cache_data(ttl=3600)
def _compute_price_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches the price data for the selected resolution."""
    
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Price Data")
    
    # Define the aggregation logic once to be reused.
    price_agg_dict = {
        "spot_price_eur_kwh": [
            ("q1", lambda x: x.quantile(0.25)),
            ("median", lambda x: x.quantile(0.50)),
            ("q3", lambda x: x.quantile(0.75))
        ]
    }
    
    # A dictionary to map the resolution to the correct pandas Series for grouping.
    resolution_config = {
        "Hourly": {"grouper": df["timestamp"].dt.hour, "x_axis_title": "Hour of Day"},
        "Weekly": {"grouper": df["timestamp"].dt.dayofweek, "x_axis_title": "Day of Week"},
        "Monthly": {"grouper": df["timestamp"].dt.month, "x_axis_title": "Month"}
    }
    
    config = resolution_config[resolution]
    df_price = df.groupby(config["grouper"]).agg(price_agg_dict)
    df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Q3"]
    df_price.index.name = config["x_axis_title"]
    
    # Special handling for weekly resolution to show weekday names.
    if resolution == "Weekly":
        x_axis_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    elif resolution == "Monthly":
        x_axis_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    else:
        x_axis_map = {i: str(i) for i in range(24)}
        
    df_price.index = df_price.index.map(x_axis_map)
    df_price = df_price.reindex(x_axis_map.values()) # Ensure correct order

    return df_price

@st.cache_data(ttl=3600)
def _compute_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the heatmap data for the spot price analysis tab."""
    
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Heatmap Data")
    return (
        df.assign(
            month=df["timestamp"].dt.month,
            hour=df["timestamp"].dt.hour)
        .pivot_table(
            values="spot_price_eur_kwh",
            index="month",
            columns="hour",
            aggfunc="mean"))

def render_price_analysis_tab(df: pd.DataFrame):
    """Renders an enhanced content page which allows for a more detailed and interactive analysis of electricity spot prices.
    It displays the first quartile (Q1), median (Q2), and third quartile (Q3) of the spot price.
    Additionally, a heatmap visualization to enable the identification of seasonal and daily trends is visualized.
    """
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Enhanced Price Analysis")

    st.subheader("Spot Price Analysis")

    # Let the user select the desired time resolution for the analysis.
    resolution = st.radio("Select Time Resolution", ("Hourly", "Weekly", "Monthly"), horizontal=True, key="price_res")

    df_price = _compute_price_data(df, resolution)

    # --- Quartile Distribution Chart ---
    st.subheader("Distribution Over Time")
    st.markdown(
        "This chart illustrates the statistical distribution of spot prices for the selected time resolution. "
        "The solid line represents the **median (50th percentile)** price, while the dotted lines show the "
        "**1st Quartile (Q1, 25th percentile)** and **3rd Quartile (Q3, 75th percentile)**. A wider range indicates higher price volatility."
    )

    fig = go.Figure()

    # Add Q1 and Q3 traces with dotted lines.
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price["Spot Price Q3"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR), name="3rd Quartile (Q3)"))
    
    # Add the Median trace as a solid line.
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price["Spot Price Median"], mode="lines", line=dict(color=FLEX_COLOR, width=3), name="Median Price"))
    
    # Q1
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price["Spot Price Q1"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR), name="1st Quartile (Q1)"))

    fig.update_layout(xaxis_title=df_price.index.name, yaxis_title="Spot Price (€/kWh)", legend_title_text="Metrics", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap Analysis ---
    st.subheader("Average Price Heatmap")
    st.markdown(
        "The heatmap visualizes the average spot price for each hour of the day across all months of the year. "
        "Use this to identify recurring patterns, such as price spikes during specific times of the day or seasonal "
        "variations. Brighter, warmer colors indicate higher average prices."
    )
    
    # Create the pivot table without modifying the original DataFrame.
    heatmap_fig = px.imshow(_compute_heatmap_data(df), labels=dict(x="Hour of Day", y="Month", color="Avg Spot Price (€/kWh)"), aspect="auto", color_continuous_scale="viridis")
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
@st.cache_data(ttl=3600)
def _compute_cost_comparison_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Cost Comparison")
    
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    grouper = pd.Grouper(key="timestamp", freq=freq_map[resolution])

    summary_agg_dict = {"Total Consumption": ("consumption_kwh", "sum"),
                        "Total Flexible Cost": ("total_cost_flexible", "sum"),
                        "Total Static Cost": ("total_cost_static", "sum")}
    
    df_summary = df.groupby(grouper).agg(**summary_agg_dict).reset_index()
    df_summary = df_summary[df_summary["Total Consumption"] > 0]
    
    df_summary["Difference (€)"] = df_summary["Total Static Cost"] - df_summary["Total Flexible Cost"]
    df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution != "Monthly" else "%Y-%m")
    
    return df_summary

def render_cost_comparison_tab(df: pd.DataFrame):
    """Renders the content for the "Cost Comparison" tab."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Cost Comparison")
    
    st.subheader("Cost Breakdown by Period")
    
    # Aggregate DataFrame to the resolution of interest (e.g., daily, weekly, monthly).
    resolution = st.radio("Select Time Resolution", ("Daily", "Weekly", "Monthly"), horizontal=True, key="summary_res")
    
    df_summary = _compute_cost_comparison_data(df, resolution)
    
    if not df_summary.empty:
        st.subheader("Total Costs per Period")
        st.markdown("Shows the **total cost per period** (energy costs on the bill). "
                    "It includes variable costs which change based on usage and spot prices, as well as any fixed monthly fees. "
                    "By examining these totals, you can identify periods with higher or lower energy expenses, "
                    "helping you understand **the impact of different times during the year** on your overall energy bill.")
        
        # Line Chart with the total costs per resolution
        st.line_chart(df_summary.set_index("Period"), y=["Total Flexible Cost", "Total Static Cost"], y_label="Total Cost (€)", color=[FLEX_COLOR, STATIC_COLOR])
            
        # Line Chart with average price per kWh for weekly and monthly resolution. Daily does not make sense.
        if resolution != "Daily":
            df_summary["Avg. Flexible Price (€/kWh)"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
            df_summary["Avg. Static Price (€/kWh)"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
            st.subheader("Average Price per kWh")
            st.markdown("This chart illustrates the **average price per kilowatt-hour (kWh)** for both flexible and static tarifs over each period. "
                    "The **monthly fixed fee is proportionally distributed** across your total consumption to provide a more accurate representation of your cost per unit of energy."
                    "By comparing these averages, you can determine which tariff offers better value during different times of the year")
            st.line_chart(df_summary.set_index("Period"), y=["Avg. Flexible Price (€/kWh)", "Avg. Static Price (€/kWh)"], y_label="Average Price (€/kWh)", color=[FLEX_COLOR, STATIC_COLOR])
        
        # Table to compare flexible with static tariffs.
        st.subheader("Tariff Comparison")
        st.text("Shows the total and per kWh costs for both tariffs, as well as the difference in €.")
        
        df_summary_table = df_summary[["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]]
        styler = df_summary_table.style.format({"Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"}).map(lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}", subset=["Difference (€)"])
        st.dataframe(styler, hide_index=True, use_container_width=True)

@st.cache_data(ttl=3600)
def _compute_usage_data(df: pd.DataFrame, day_filter: str) -> tuple[pd.DataFrame, list]:
    """Computes and caches the DataFrame for the selected day type."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Usage Data")

    # Filter the dataframe for all day or weekend/weekday option.
    df["day_type"] = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    df_filtered = df[df["day_type"] == day_filter[:-1]] if day_filter != "All Days" else df
    
    if df_filtered.empty:
        st.warning(f"No data available for {day_filter.lower()}.")
        return pd.DataFrame(), list()
    
    else:
        available_dates = df_filtered["timestamp"].dropna().dt.tz_convert(LOCAL_TIMEZONE).dt.date.unique().tolist()
        return df_filtered, available_dates

@st.cache_data(ttl=3600)
def _compute_usage_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the proportion of each usage profile of the DataFrame. """
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Usage Profile")

    # Calculate the average price and proportion of each load type.
    load_types = ["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]
    profile_data = []
    total_kwh = df["consumption_kwh"].sum()

    for load in load_types:
        kwh = df[load].sum()
        kwh_mean = df[load].mean() * get_intervals_per_day(df)
        if kwh > 0:
            avg_price = (df[load] * df["spot_price_eur_kwh"]).sum() / kwh
            proportion = kwh / total_kwh if total_kwh > 0 else 0
            profile_data.append({"Profile": load.replace("_kwh", "").replace("_", " ").title(), "kwh": kwh, "kwh_mean": kwh_mean, "avg_price": avg_price, "proportion": proportion})
    
    return pd.DataFrame(profile_data)

@st.cache_data(ttl=3600)
def _compute_example_day(df: pd.DataFrame, random_day) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Selects a random day and return the data for plotting as well as the DataFrame itself."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Example Day")

    df_day = df[df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == random_day].copy()
    
    if not df_day.empty:
        df_day["hour"] = df_day["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour

        # Group by hour and sum the kWh columns
        df_plot_day = (
            df_day.groupby("hour")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
            .sum()
            .rename(columns={
                "base_load_kwh": "Base Load",
                "regular_load_kwh": "Regular Load",
                "peak_load_kwh": "Peak Load"
            })
        )
    
        return df_day, df_plot_day
    else:
        return pd.DataFrame(), pd.DataFrame()


def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the "Usage Pattern Analysis" tab."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Usage Pattern")

    st.subheader("Analyze Your Consumption Profile")
    
    # Get weekend/weekday information.
    day_filter = st.radio("Filter data by:", ("All Days", "Weekdays", "Weekends"), horizontal=True)
    
    st.subheader("Consumption & Cost Profile")
    st.markdown("This chart visualizes the distribution of electricity consumption across different load types: Base Load, Regular Load, and Peak Load. "
                "Each segment represents a proportionate share of total consumption over time, providing insights into how energy usage is distributed throughout the day. "
                "Base Load represents the **minimal load measured over multiple hours** (e. g., WiFi, standby, basic lights). "
                "Peak usage is defined by **sharp inclines and a long duration of high loads**, which can be **postponed to different times** during a day.")
    
    df_filtered, available_dates = _compute_usage_data(df, day_filter)
    profile_data = _compute_usage_profile_data(df_filtered)
    
    # Build a Marimekko chart for every load type.
    if not profile_data.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        
        cumulative_width = 0
        for _, row in profile_data.iterrows():
            ax.bar(cumulative_width, row["avg_price"], width=row["proportion"], align="edge", color=FLEX_COLOR, edgecolor="white")
            annotation_text = f"{row["Profile"]}\n\n{row["proportion"]:.1%} of kWh\nAvg. {row["kwh_mean"]:.2f}/day\n€{row["avg_price"]:.3f}/kWh"
            ax.text(cumulative_width + row["proportion"]/2, row["avg_price"]/2, annotation_text, ha="center", va="center", color="white", fontsize=10, weight="bold")
            cumulative_width += row["proportion"]
            
        ax.set_ylabel("Average Spot Price (€/kWh)")
        ax.set_xlabel("Proportion of Total Consumption")
        
        xticks = [0, 0.25, 0.5, 0.75, 1]
        
        ax.set_xlim(0, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(t*100)}%" for t in xticks])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
    
    # Show the distribution of each load type on a random example.
    st.subheader("Example Day Breakdown")
    st.markdown("To provide a better intuition about the classifcation, this chart shows the classification on a random example day.")
    
    if len(available_dates) > 0:
        if "random_day" not in st.session_state or st.session_state.random_day not in available_dates:
            st.session_state.random_day = random.choice(available_dates)
        
        # Create a DataFrame for the respective day.            
        df_day, df_day_plot = _compute_example_day(df, st.session_state.random_day)

        col1_day, col2_day = st.columns([3, 1])
        col1_day.caption(f"Displaying data for {st.session_state.random_day.strftime("%A, %Y-%m-%d")} (total: {df_day["consumption_kwh"].sum():.2f} kWh)")
        
        if col2_day.button("Show a Different Day"):
            st.session_state.random_day = random.choice(available_dates)

        st.bar_chart(df_day_plot, color=[STATIC_COLOR, FLEX_COLOR_LIGHT, FLEX_COLOR], y_label="Consumption (kWh)")
            
    col1, col2 = st.columns(2)
    col1.metric("Base Load Threshold (Continuous Usage)", f"{base_threshold:.3f} kWh")
    # The peak_threshold passed is the influenceable part. Add base for the absolute value.
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric("Peak Sustain Threshold", f"{absolute_peak_threshold:.3f} kWh", help="A peak event starts with a sharp increase and continues for every hour consumption stays above this level.")

@st.cache_data(ttl=3600)
def _compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the yearly summary of the data."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Yearly Summary")

    df["Year"] = df["timestamp"].dt.year
    yearly_summary_agg_dict = {"Total Consumption": ("consumption_kwh", "sum"), "Avg. Consumption": ("consumption_kwh", "mean"), "Total Flexible Cost":("total_cost_flexible", "sum"), "Total Static Cost": ("total_cost_static", "sum")}
    yearly_agg = df.groupby("Year").agg(**yearly_summary_agg_dict).reset_index()
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg. Flex Price"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]
        yearly_agg["Avg. Static Price"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
    
    return yearly_agg
    
def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the "Yearly Summary" tab."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Yearly Summary")

    st.subheader("Yearly Summary")
    yearly_agg = _compute_yearly_summary(df)
    yearly_format = {"Total Consumption": "{:,.2f} kWh", "Avg. Consumption": "{:,.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Avg. Flex Price (€/kWh)": "€{:.3f}", "Avg. Static Price (€/kWh)": "€{:.3f}"}
    st.dataframe(yearly_agg.style.format(*yearly_format), hide_index=True, use_container_width=True)

@st.cache_data(ttl=3600)
def _compute_download_data(df: pd.DataFrame) -> tuple[bytes, bytes]:
    """Computes and caches the data for the download tab, full dataframe as well as spot prices."""
    
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Download Data")
    
    excel_spot_data = (
        df.dropna(subset=["spot_price_eur_kwh"])
        .copy()
        .set_index("timestamp")
        .resample("h")
        .first()
        .reset_index()[["timestamp", "spot_price_eur_kwh"]]
        )

    return to_excel(df), to_excel(excel_spot_data)
    

def render_download_tab(df: pd.DataFrame, start_date: date, end_date: date):
    """Renders the content for the Download tab."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Download")
    
    st.text("Download detailed data with cost calculations and usage classification.")
    
    excel_full_data, excel_spot_data = _compute_download_data(df)
    
    st.download_button(
        label="Download Excel File",
        data=excel_full_data,
        file_name=f"electricity_analysis_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
    st.text("Download only the EPEX spot prices per hour.")
    
    st.download_button(
            label="Download Excel File",
            data=excel_spot_data,
            file_name=f"spot_prices_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

@st.cache_data(ttl=3600)
def _compute_absence_data(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> list:
    """Computes and caches the days of absence."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Absence Data")
    
    daily_consumption = df.groupby("date")["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    absence_days = daily_consumption[daily_consumption < (base_threshold * intervals_per_day * absence_threshold)].index.tolist()
    
    return absence_days

def get_absence_days(df_classified: pd.DataFrame, base_threshold: float, absence_threshold: float) -> pd.DataFrame:
    """Add option to remove days with absence, that is when the daily consumption is lower than 80% of usual base threshold."""
    with st.sidebar:
        absence_days = _compute_absence_data(df_classified, base_threshold, absence_threshold)
        excluded_days = []
        if absence_days:
            st.subheader("4. Absence Handling", help=f"Remove days with extremely low consumption (below {absence_threshold}%). Yields more robust data for analyses.")
            select_all = st.checkbox(f"Exclude all {len(absence_days)} Days Of Absence", value=False)
            default_selection = absence_days if select_all else []
            excluded_days = st.multiselect("Exclude days?", options=absence_days, default=default_selection)
            
    return df_classified[~df_classified["date"].isin(excluded_days)]   

@st.cache_data(ttl=3600)
def merge_consumption_with_prices(df_consumption: pd.DataFrame, df_spot_prices: pd.DataFrame) -> pd.DataFrame:
    """Merges the spot prices with the consumption data on the timestamp column."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Merging Consumption with Prices")

    start_date, end_date = _get_min_max_date(df_consumption)
    
    # Merge on timestamp with a tolerance of 59 minutes such that df_consumption can have 15 minutes timespans and df_spot_prices hourly prices.
    df_merged = pd.merge_asof(df_consumption, df_spot_prices, on="timestamp", direction="backward", tolerance=pd.Timedelta("59min"))
    
    # Filter for valid date (inner join).
    df_merged = df_merged[(df_merged.timestamp.dt.date >= start_date) & (df_merged.timestamp.dt.date <= end_date)].dropna()
    
    # Convert dataframe to local timezone again
    df_merged["timestamp"] = df_merged["timestamp"].dt.tz_convert(LOCAL_TIMEZONE) 
            
    df_merged["date"] = df_merged["timestamp"].dt.date
    
    return df_merged

@st.cache_data(ttl=3600)
def perform_analysis(df: pd.DataFrame, flex_tariff, static_tariff) -> pd.DataFrame: 
    """A cached function to run heavy computations. It only re-runs if the underlying data or a key parameter changes. """
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Performing Full Analysis")

    return tariff_manager.run_cost_analysis(df, flex_tariff, static_tariff)
         
# --- Main Application ---
uploaded_file = st.sidebar.file_uploader("Upload Your Consumption CSV", type=["csv"])

# Instantiate TariffManager once
tariff_manager = TariffManager("flex_tariffs.json", "static_tariffs.json")

if not uploaded_file:
    st.info("👋 Welcome! Please upload your consumption data to begin.")
else:
    df_consumption = process_consumption_data(uploaded_file, aggregation_level = "15min")
    
    if not df_consumption.empty:
        df_spot_prices = get_spot_data(*_get_min_max_date(df_consumption))
        
        # Perform merge with price data to align each 15-min consumption row with the previous hourly price.
        df_merged = merge_consumption_with_prices(df_consumption, df_spot_prices)
        
        start_date, end_date, flex_tariff, static_tariff, shift_percentage = get_sidebar_inputs(df_merged, tariff_manager)
        
        if not df_merged.empty:
                
            df_classified, base_threshold, peak_threshold = classify_usage(df_merged, LOCAL_TIMEZONE)
                
            df_with_shifting = simulate_peak_shifting(df_classified, shift_percentage)
            df_analysis = perform_analysis(df_with_shifting, flex_tariff, static_tariff)
            
            #df_analysis, base_threshold, peak_threshold = perform_full_analysis(df_merged, LOCAL_TIMEZONE, flex_tariff, static_tariff, shift_percentage)
            
            # Interactive elements like absence handling must remain outside the cached function
            df_analysis = get_absence_days(df_analysis, base_threshold, ABSENCE_THRESHOLD)

            render_recommendation(df_analysis)

            # Render the tabs with the individual use cases.
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Spot Price Analysis**", "**Cost Comparison**", "**Usage Pattern Analysis**", "**Yearly Summary**", "**Download Data**"])
            with tab1:
                render_price_analysis_tab(df_analysis)
            with tab2:
                render_cost_comparison_tab(df_analysis)
            with tab3:
                render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
            with tab4:
                render_yearly_summary_tab(df_analysis)
            with tab5:
                render_download_tab(df_analysis, start_date, end_date)
        
        else:
            st.warning("No overlapping data found for the selected period.")
