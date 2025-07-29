import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import json
import random
from tariffs import TariffManager, Tariff
from parser import ConsumptionDataParser
from methods import * # Core analysis methods

# --- Page and App Configuration ---

st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.\nThis project was influenced by https://awattar-backtesting.github.io/, which provides a simple and effective overview. This tool provides further insights into your consumption behavior.")

# --- Constants and Configuration ---

LOCAL_TIMEZONE = "Europe/Vienna"
FLEX_COLOR = "#fd690d"
FLEX_COLOR_LIGHT = "#f7be44"
STATIC_COLOR = "#989898"
GREEN = "#5fba7d"
RED = "#d65f5f"

ABSENCE_THRESHOLD = 0.75

AWATTAR_COUNTRY = "at" # or de

# --- Data Loading and Caching ---

@st.cache_data(ttl=3600)
def fetch_spot_data(start: date, end: date) -> pd.DataFrame:
    
    """Fetches spot market price data from the aWATTar API for a given date range."""
    base_url = f"https://api.awattar.{AWATTAR_COUNTRY}/v1/marketdata"
    start_dt, end_dt = datetime.combine(start, datetime.min.time()), datetime.combine(end + pd.Timedelta(days=1), datetime.min.time())
    params = {"start": int(start_dt.timestamp() * 1000),
              "end": int(end_dt.timestamp() * 1000)}
    
    try:
        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Will fetch data from Awattar.")
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json()["data"])
        df["timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
        df["spot_price_eur_kwh"] = df["marketprice"] / 1000 * 1.2 # Convert from Eur/MWh to c/kWh and add VAT
        return df[["timestamp", "spot_price_eur_kwh"]]
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch spot price data: {e}")
        
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API.")
        
    return pd.DataFrame()

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

def get_sidebar_inputs(df_consumption: pd.DataFrame, tariff_manager: TariffManager):
    """Renders all sidebar inputs and returns the configuration values."""
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("1. Analysis Period")
        min_date, max_date = df_consumption["timestamp"].min().date(), df_consumption["timestamp"].max().date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        st.subheader("2. Tariff Plans")
        st.text("Choose a tariff or adjust parameters.")
        flex_tariff_options = tariff_manager.get_flex_tariffs_with_custom()
        static_tariff_options = tariff_manager.get_static_tariffs_with_custom()

        with st.expander("Flexible (Spot Price) Plan", expanded=True):
            selected_flex_name = st.selectbox("Select Flexible Tariff", options=list(flex_tariff_options.keys()), index=len(flex_tariff_options)-1)
            selected_flex_tariff = flex_tariff_options[selected_flex_name]
            flex_on_top = st.number_input("On-Top Price (€/kWh)", value=selected_flex_tariff.price_kwh, min_value=0.0, step=0.01, format="%.4f")
            flex_fee = st.number_input("Monthly Fee (€)", value=selected_flex_tariff.monthly_fee, min_value=0.0, step=0.1)
            final_flex_tariff = Tariff(selected_flex_tariff.name, selected_flex_tariff.type, flex_on_top, flex_fee)

        with st.expander("Static (Fixed Price) Plan"):
            selected_static_name = st.selectbox("Select Static Tariff", options=list(static_tariff_options.keys()), index=len(static_tariff_options)-1)
            selected_static_tariff = static_tariff_options[selected_static_name]
            static_price = st.number_input("Fixed Price (€/kWh)", value=selected_static_tariff.price_kwh, min_value=0.0, step=0.01)
            static_fee = st.number_input("Monthly Fee (€)", value=selected_static_tariff.monthly_fee, min_value=0.0, step=0.1)
            final_static_tariff = Tariff(selected_static_tariff.name, selected_static_tariff.type, static_price, static_fee)

        st.subheader("3. Cost Simulation", help="Simulate shifting a percentage of your peak consumption to a cheaper hour within a +/- 2-hour window.")
        shift_percentage = st.slider("Shift Peak Load (%)", 0, 100, 0, 5)

        return start_date, end_date, final_flex_tariff, final_static_tariff, shift_percentage

def render_recommendation(df: pd.DataFrame):
    """Displays the final tariff recommendation based on calculated savings."""
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

def render_cost_comparison_tab(df: pd.DataFrame):
    """Renders the content for the "Cost Comparison" tab."""
    st.subheader("Cost Breakdown by Period")
    
    # Aggregate DataFrame to the resolution of interest (e.g., daily, weekly, monthly).
    resolution = st.radio("Select Time Resolution", ("Daily", "Weekly", "Monthly"), horizontal=True, key="res")
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    grouper = pd.Grouper(key="timestamp", freq=freq_map[resolution])

    summary_agg_dict = {"Total Consumption": ("consumption_kwh", "sum"), "Total Flexible Cost": ("total_cost_flexible", "sum"), "Total Static Cost": ("total_cost_static", "sum")}
    df_summary = df.groupby(grouper).agg(**summary_agg_dict).reset_index()
    df_summary = df_summary[df_summary["Total Consumption"] > 0]
    
    
    if not df_summary.empty:
        df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution != "Monthly" else "%Y-%m")
        st.subheader("Total Cost Comparison")
        st.text("Shows the total cost per period (energy costs on the bill).")
        
        # Line Chart with the total costs per resolution
        st.line_chart(df_summary.set_index("Period"), y=["Total Flexible Cost", "Total Static Cost"], y_label="Total Cost (€)", color=[FLEX_COLOR, STATIC_COLOR])
        
        # Line Chart with average price per kWh for weekly and monthly resolution. Daily does not make sense.
        if resolution != "Daily":
            df_summary["Avg. Flexible Price (€/kWh)"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
            df_summary["Avg. Static Price (€/kWh)"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
            st.subheader("Average Price Comparison")
            st.text("The monthly fee is proportionally added based on the average kWh consumption.")
            st.line_chart(df_summary.set_index("Period"), y=["Avg. Flexible Price (€/kWh)", "Avg. Static Price (€/kWh)"], y_label="Average Price (€/kWh)", color=[FLEX_COLOR, STATIC_COLOR])
        
        # Table to compare flexible with static tariffs.
        st.subheader("Tariff Comparison")
        st.text("Shows the total and per kWh costs for both tariffs, as well as the difference in €.")
        df_summary["Difference (€)"] = df_summary["Total Static Cost"] - df_summary["Total Flexible Cost"]
        df_summary = df_summary[["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]]
        styler = df_summary.style.format({"Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"}).map(lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}", subset=["Difference (€)"])
        st.dataframe(styler, hide_index=True, use_container_width=True)

def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the "Usage Pattern Analysis" tab."""
    st.subheader("Analyze Your Consumption Profile")
    
    # Get weekend/weekday information.
    df["day_type"] = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    day_filter = st.radio("Filter data by:", ("All Days", "Weekdays", "Weekends"), horizontal=True)
    
    # Filter the dataframe for all day or weekend/weekday option.
    df_pattern = df[df["day_type"] == day_filter[:-1]] if day_filter != "All Days" else df
    
    if df_pattern.empty:
        st.warning(f"No data available for {day_filter.lower()}.")
        return
    
    # Get number of rows per day
    intervals_per_day = df_pattern.groupby(df_pattern["timestamp"].dt.date).size().mode().iloc[0]

    st.subheader("Consumption & Cost Profile")
    st.text("If peak usage costs align with regular usage, significant cost-saving potential is possible.")
    load_types = ["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]
    profile_data = []
    total_kwh = df_pattern["consumption_kwh"].sum()

    # Calculate the average price and proportion of each load type.
    for load in load_types:
        kwh = df_pattern[load].sum()
        kwh_mean = df_pattern[load].mean() * intervals_per_day
        if kwh > 0:
            avg_price = (df_pattern[load] * df_pattern["spot_price_eur_kwh"]).sum() / kwh
            proportion = kwh / total_kwh if total_kwh > 0 else 0
            profile_data.append({"Profile": load.replace("_kwh", "").replace("_", " ").title(), "kwh": kwh, "kwh_mean": kwh_mean, "avg_price": avg_price, "proportion": proportion})
    
    # Build a Marimekko chart for every load type.
    if profile_data:
        df_plot = pd.DataFrame(profile_data)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        
        cumulative_width = 0
        for _, row in df_plot.iterrows():
            ax.bar(cumulative_width, row["avg_price"], width=row["proportion"], align="edge", color=FLEX_COLOR, edgecolor="white")
            annotation_text = f"{row["Profile"]}\n\n{row["proportion"]:.1%} of kWh\nAvg. {row["kwh_mean"]:.2f}/day\n€{row["avg_price"]:.3f}/kWh"
            ax.text(cumulative_width + row["proportion"]/2, row["avg_price"]/2, annotation_text, ha="center", va="center", color="white", fontsize=10, weight="bold")
            cumulative_width += row["proportion"]
            
        ax.set_ylabel("Average Spot Price (€/kWh)")
        ax.set_xlabel("Proportion of Total Consumption")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
    
    # Show the distribution of each load type on a random example.
    st.subheader("Example Day Breakdown")
    st.markdown("This chart visualizes when your high-consumption activities occur on a random day from your selection.")
    available_dates = df_pattern["timestamp"].dropna().dt.tz_convert(LOCAL_TIMEZONE).dt.date.unique()

    if len(available_dates) > 0:
        if "random_day" not in st.session_state or st.session_state.random_day not in available_dates:
            st.session_state.random_day = random.choice(available_dates)
        
        # Create a DataFrame for the respective day.            
        df_day = df_pattern[df_pattern["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == st.session_state.random_day].copy()

        col1_day, col2_day = st.columns([3, 1])
        col1_day.caption(f"Displaying data for {st.session_state.random_day.strftime("%A, %Y-%m-%d")} (total: {df_day["consumption_kwh"].sum():.2f} kWh)")
        if col2_day.button("Show a Different Day"):
            st.session_state.random_day = random.choice(available_dates)
            st.rerun()

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

        st.bar_chart(df_plot_day, color=[STATIC_COLOR, FLEX_COLOR_LIGHT, FLEX_COLOR], y_label="Consumption (kWh)")
        
    else:
        st.warning("No data available for the selected day.")
    
    col1, col2 = st.columns(2)
    col1.metric("Base Load Threshold (Continuous Usage)", f"{base_threshold:.3f} kWh")
    # The peak_threshold passed is the influenceable part. Add base for the absolute value.
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric("Peak Sustain Threshold", f"{absolute_peak_threshold:.3f} kWh", help="A peak event starts with a sharp increase and continues for every hour consumption stays above this level.")

def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the "Yearly Summary" tab."""
    st.subheader("Yearly Summary")
    df_yearly = df.copy()
    df_yearly["Year"] = df_yearly["timestamp"].dt.year
    yearly_summary_agg_dict = {"Total Consumption": ("consumption_kwh", "sum"), "Total Flexible Cost":("total_cost_flexible", "sum"), "Total Static Cost": ("total_cost_static", "sum")}
    yearly_agg = df_yearly.groupby("Year").agg(**yearly_summary_agg_dict).reset_index()
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg. Flex Price (€/kWh)"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]
        yearly_agg["Avg. Static Price (€/kWh)"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
        st.dataframe(yearly_agg.style.format({"Total Consumption": "{:,.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Avg. Flex Price (€/kWh)": "€{:.3f}", "Avg. Static Price (€/kWh)": "€{:.3f}"}), hide_index=True, use_container_width=True)

# --- Main Application ---
uploaded_file = st.sidebar.file_uploader("Upload Your Consumption CSV", type=["csv"])

# Instantiate TariffManager once
tariff_manager = TariffManager("flex_tariffs.json", "static_tariffs.json")

if not uploaded_file:
    st.info("👋 Welcome! Please upload your consumption data to begin.")
else:
    df_consumption = process_consumption_data(uploaded_file, aggregation_level = "15min")
    
    if not df_consumption.empty:
        start_date, end_date, flex_tariff, static_tariff, shift_percentage = get_sidebar_inputs(df_consumption, tariff_manager)
        
        # Perform asof merge with price data to align each 15-min consumption row with the previous hourly price.
        df_spot_prices = fetch_spot_data(start_date, end_date)
        df_merged = pd.merge_asof(df_consumption, df_spot_prices, on="timestamp", direction="backward", tolerance=pd.Timedelta("59min"))
        df_merged = df_merged[(df_merged.timestamp.dt.date >= start_date) & (df_merged.timestamp.dt.date <= end_date)].dropna()
        
        # Convert dataframe to local timezone again
        df_merged["timestamp"] = df_merged["timestamp"].dt.tz_convert(LOCAL_TIMEZONE) 
                
        df_merged["date_col"] = df_merged["timestamp"].dt.date
        
        
        if not df_merged.empty:
            intervals_per_day = df_merged.groupby("date_col").size().mode().iloc[0]
            df_classified, base_threshold, peak_threshold = classify_usage(df_merged, LOCAL_TIMEZONE, intervals_per_day)        
            
            # Add option to remove days with absence, that is when the daily consumption is lower than 80% of usual base threshold.
            with st.sidebar:
                daily_consumption = df_classified.groupby("date_col")["consumption_kwh"].sum()
                absence_days = daily_consumption[daily_consumption < (base_threshold * intervals_per_day * ABSENCE_THRESHOLD)].index.tolist()
                excluded_days = []
                if absence_days:
                    st.subheader("4. Absence Handling", help=f"Remove days with extremely low consumption (below {ABSENCE_THRESHOLD}%). Yields more robust data for analyses.")
                    select_all = st.checkbox(f"Exclude all {len(absence_days)} Days Of Absence", value=False)
                    default_selection = absence_days if select_all else []
                    excluded_days = st.multiselect("Exclude days?", options=absence_days, default=default_selection)

            # Remove absence days from DataFrame.
            df_analysis = df_classified[~df_classified["date_col"].isin(excluded_days)].copy()
            
            # Simulate peak shifting
            df_simulated = simulate_peak_shifting(df_analysis, shift_percentage)

            # Use the new manager to run the final cost analysis
            df_final = tariff_manager.run_cost_analysis(df_simulated, flex_tariff, static_tariff)

            render_recommendation(df_final)
            tab1, tab2, tab3 = st.tabs(["**Cost Comparison**", "**Usage Pattern Analysis**", "**Yearly Summary**"])
            with tab1:
                render_cost_comparison_tab(df_final)
            with tab2:
                render_usage_pattern_tab(df_final, base_threshold, peak_threshold)
            with tab3:
                render_yearly_summary_tab(df_final)
        else:
            st.warning("No overlapping data found for the selected period.")
